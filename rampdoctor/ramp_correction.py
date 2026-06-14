import numpy as np
from pathlib import Path
from scipy.interpolate import griddata
from scipy.optimize import curve_fit


def build_correction_map(cube, mask=None):
    """
    Derive the per-pixel group correction map from a reference ramp cube.
    Use a quiescent observation (no flares/transients) to build this.

    Parameters
    ----------
    cube : ndarray, shape (n_int, n_groups, ny, nx)
        Stage-1 corrected ramp cube (raw group values).
    mask : ndarray, shape (ny, nx), bool, optional
        True = pixels to interpolate over (bad pixels, saturated core, etc.)

    Returns
    -------
    C_map : ndarray, shape (n_groups-1, ny, nx)
        Multiplicative correction factor per group per pixel.
        C_map[0] = 1 everywhere (group 0 is the reference).
    """
    grads = np.diff(cube, axis=1).astype(float)
    med_grad = np.median(grads, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        C_map = np.where(med_grad != 0, med_grad[-2:-1] / med_grad, np.nan)

    if mask is not None:
        ny, nx = mask.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        good = ~mask & np.isfinite(C_map[0])
        good_yx = np.column_stack([yy[good], xx[good]])

        for g in range(C_map.shape[0]):
            needs_fill = mask | ~np.isfinite(C_map[g])
            if not needs_fill.any():
                continue
            fill_yx = np.column_stack([yy[needs_fill], xx[needs_fill]])
            good_vals = C_map[g][good]
            filled = griddata(good_yx, good_vals, fill_yx, method='linear')
            still_nan = ~np.isfinite(filled)
            if still_nan.any():
                filled[still_nan] = griddata(
                    good_yx, good_vals, fill_yx[still_nan], method='nearest'
                )
            C_map[g][needs_fill] = filled

    return C_map


def correct_reset_decay(cube, method='median', mask=None, mask_dilation=0,
                        edge_margin=10, dq=None, sat_bit=2,
                        diagnostics=False, save_path=None):
    """
    Correct charge reset decay in MIRI ramp data.

    tau is fitted globally from the spatial mean gradient profile and is the
    same for all pixels. The last group-to-group gradient is always excluded
    (last-frame anomaly).

    Three methods:

    'median' (default)
        Fits C + A*exp(-g/tau) to the per-pixel median gradient profile.
        A and C are per-pixel via linear regression with tau fixed.
        A is constant across integrations.

    'per_int'
        Fits [C, A, delta] independently for each integration and each pixel
        using linear regression with tau fixed. Removes residual offsets caused
        by integration-to-integration variation in A (e.g. from charge-dependent
        decay amplitude). Noisier than 'median' for individual pixels but
        produces unbiased aperture-summed lightcurves.

    'stretched_exp'
        Fits per-pixel A from the median gradient profile (same first step),
        then fits A(Q) = scale * exp(beta * Q^c) across pixels. For each ramp,
        A is evaluated from the charge Q at the last good group, giving a
        per-integration per-pixel amplitude while tau remains global.

    Parameters
    ----------
    cube : ndarray (n_int, n_groups, ny, nx), float
        Raw SCI data from uncal.fits.
    method : {'median', 'per_int', 'stretched_exp'}
    mask : ndarray (ny, nx) bool, optional
        True = non-science pixel. Masked pixels are excluded from the tau
        spatial mean fit and the A(Q) fit. Does not affect per-pixel A fitting
        or the correction itself.
    mask_dilation : int
        Dilate the mask by this many pixels (circular) before applying to
        fitting statistics. Excludes pixels near masked regions.
    edge_margin : int
        Border pixels excluded from the A(Q) fit in 'stretched_exp'.
    dq : ndarray (n_int, n_groups, ny, nx) uint8, optional
        GROUPDQ array. Used in 'stretched_exp' to find the last unsaturated
        group per ramp for Q estimation.
    sat_bit : int
        GROUPDQ bit value for SATURATED (default 2).
    diagnostics : bool
        If True, produce diagnostic figures.
    save_path : str or Path, optional
        File path to save the diagnostic figure. Only used when diagnostics=True.

    Returns
    -------
    cube_cor : ndarray (n_int, n_groups, ny, nx)
        Corrected SCI cube. Groups 1 through n_groups-2 have the cumulative
        decay subtracted; group 0 is corrected for the first-frame offset.
    """
    cube = np.asarray(cube, dtype=float)
    n_int, n_groups, ny, nx = cube.shape
    n_grads = n_groups - 2  # drop last gradient (last-frame anomaly)

    grads = np.diff(cube, axis=1)[:, :n_grads]        # (n_int, n_grads, ny, nx)
    med_grad = np.median(grads, axis=0)                # (n_grads, ny, nx)
    g_arr = np.arange(n_grads, dtype=float)

    if mask is not None and mask_dilation > 0:
        from scipy.ndimage import binary_dilation
        r = mask_dilation
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        struct = (yy**2 + xx**2) <= r**2
        mask = binary_dilation(mask, structure=struct)
    sci = ~mask if mask is not None else np.ones((ny, nx), dtype=bool)

    # Global tau from spatial mean over science pixels, excluding gradient 0.
    # Gradient 0 is suppressed by the first-frame anomaly (group 0 has extra
    # reset charge), which breaks the monotonic-decay assumption at g=0.
    mean_profile = np.nanmean(med_grad[:, sci], axis=1)
    mean_profile_fit = mean_profile[1:]
    def _exp_model(g, C, A, t):
        return C + A * np.exp(-g / t)
    popt, _ = curve_fit(_exp_model, g_arr[1:], mean_profile_fit,
                        p0=[mean_profile_fit[-1],
                            mean_profile_fit[0] - mean_profile_fit[-1],
                            1.5])
    tau = float(popt[2])

    # Per-pixel fit: [C, A, delta] where delta is the first-frame offset.
    # The design matrix has a -1 in the delta column only for g=0, accounting
    # for the suppression of gradient 0 by the first-frame anomaly.
    exp_g = np.exp(-g_arr / tau)                       # (n_grads,)
    ff_col = np.zeros(n_grads); ff_col[0] = -1.0
    X = np.column_stack([np.ones(n_grads), exp_g, ff_col])  # (n_grads, 3)
    params, _, _, _ = np.linalg.lstsq(
        X, med_grad.reshape(n_grads, -1), rcond=None)
    A_map = params[1].reshape(ny, nx)                  # (ny, nx)
    delta_map = params[2].reshape(ny, nx)              # first-frame offset (ny, nx)

    if method == 'median':
        decay_cumsum = np.cumsum(A_map * exp_g[:, None, None], axis=0)  # (n_grads, ny, nx)
        cube_cor = cube.copy()
        cube_cor[:, 1:n_grads + 1] -= decay_cumsum[None]
        cube_cor[:, 0] -= delta_map[None]

    elif method == 'per_int':
        # Fit [C_i, A_i, delta_i] independently per integration per pixel.
        # tau is still global. This removes residual offsets from integration-
        # to-integration variation in A (charge-dependent decay amplitude).
        grads_flat = grads.reshape(n_int, n_grads, -1)    # (n_int, n_grads, ny*nx)
        A_int = np.empty((n_int, ny * nx))
        delta_int = np.empty((n_int, ny * nx))
        for i in range(n_int):
            p, _, _, _ = np.linalg.lstsq(X, grads_flat[i], rcond=None)
            A_int[i] = p[1]
            delta_int[i] = p[2]
        A_int = A_int.reshape(n_int, ny, nx)
        delta_int = delta_int.reshape(n_int, ny, nx)

        decay_cumsum = np.cumsum(
            A_int[:, None, :, :] * exp_g[None, :, None, None], axis=1)
        cube_cor = cube.copy()
        cube_cor[:, 1:n_grads + 1] -= decay_cumsum
        cube_cor[:, 0] -= delta_int

    else:
        # --- method == 'stretched_exp' ---
        edge_mask = np.zeros((ny, nx), dtype=bool)
        edge_mask[:edge_margin] = True
        edge_mask[-edge_margin:] = True
        edge_mask[:, :edge_margin] = True
        edge_mask[:, -edge_margin:] = True

        Q_med = np.median(cube[:, n_grads, :, :], axis=0)  # (ny, nx)
        fit_mask = ~edge_mask & sci & np.isfinite(A_map) & (Q_med > 0)

        def _stretched(Q, scale, beta, c):
            return scale * np.exp(beta * Q**c)
        Q_fit, A_fit = Q_med[fit_mask], A_map[fit_mask]
        popt_s, _ = curve_fit(_stretched, Q_fit, A_fit,
                              p0=[np.percentile(A_fit, 10), 1e-3, 0.6],
                              maxfev=50000)
        scale, beta, c = popt_s

        # Per-ramp Q from last unsaturated group
        if dq is not None:
            bad = (dq[:, :n_grads + 1] & sat_bit) > 0
            not_bad_rev = ~bad[:, ::-1]
            last_rev = np.argmax(not_bad_rev, axis=1)
            last_good = np.clip(n_grads - last_rev, 0, n_grads)
            ii = np.arange(n_int)[:, None, None]
            yy = np.arange(ny)[None, :, None]
            xx = np.arange(nx)[None, None, :]
            Q_int = cube[ii, last_good, yy, xx]        # (n_int, ny, nx)
        else:
            Q_int = cube[:, n_grads, :, :]

        Q_int = np.clip(Q_int, 1.0, None)
        A_int = scale * np.exp(beta * Q_int**c)        # (n_int, ny, nx)

        decay_cumsum = np.cumsum(
            A_int[:, None, :, :] * exp_g[None, :, None, None], axis=1)
        cube_cor = cube.copy()
        cube_cor[:, 1:n_grads + 1] -= decay_cumsum
        cube_cor[:, 0] -= delta_map[None]

    if diagnostics:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n_panels = 3 if method == 'stretched_exp' else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))

        ax = axes[0]
        g_fine = np.linspace(1, n_grads - 1, 200)
        C_fit, A_fit_mean = float(popt[0]), float(popt[1])
        ax.plot(g_arr[0], mean_profile[0], 'o', color='gray', ms=5, label='g=0 (excluded from fit)')
        ax.plot(g_arr[1:], mean_profile[1:], 'o', color='k', ms=5, label='Spatial mean')
        ax.plot(g_fine, C_fit + A_fit_mean * np.exp(-g_fine / tau),
                '--', color='C3', lw=1.5, label=f'Fit  τ={tau:.2f} grp')
        ax.set_xlabel('Gradient index')
        ax.set_ylabel('Mean gradient (DN/group)')
        ax.set_title('Global τ fit')
        ax.legend(fontsize=8)
        ax.set_xticks(g_arr.astype(int))

        ax = axes[1]
        vmax = np.nanpercentile(A_map, 99)
        im = ax.imshow(A_map, origin='lower', vmin=0, vmax=vmax, cmap='viridis')
        fig.colorbar(im, ax=ax, label='DN/group')
        ax.set_title('Decay amplitude A')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        if method == 'stretched_exp':
            ax = axes[2]
            ax.scatter(Q_med[fit_mask], A_map[fit_mask], s=1, alpha=0.1,
                       color='C0', rasterized=True)
            q_line = np.linspace(np.nanpercentile(Q_med[fit_mask], 1),
                                 np.nanpercentile(Q_med[fit_mask], 99), 300)
            ax.plot(q_line, scale * np.exp(beta * q_line**c), '-', color='C3',
                    lw=1.5, label=f'scale={scale:.2f}, β={beta:.3e}, c={c:.3f}')
            ax.set_xlabel('Q at last group (DN)')
            ax.set_ylabel('A (DN/group)')
            ax.set_title('A(Q) stretched exponential fit')
            ax.legend(fontsize=8)

        fig.suptitle(f'Reset decay correction diagnostics  (method={method})',
                     fontsize=11, fontweight='bold')
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(Path(save_path), dpi=150, bbox_inches='tight')
        plt.close(fig)

    return cube_cor


def fit_bfe_params(cube, alpha_bfe=3.43, b_bfe=None, c_bfe=None,
                   bg_mask=None, sci_mask=None,
                   bfe_early_groups=None, bfe_late_groups=None,
                   ap_radius=5, cut=20, fit_r=None, verbose=False,
                   diagnostics=False, save_path=None, return_model=False):
    """
    Find the brightest source in the image and fit the BFE kernel via the
    source-centric forward model, K = -(1 + b r + c r^2) / r^alpha.

    Uses SEP to locate the source, fits the reset-decay parameters (tau,
    rate_map, Adec_map) from the median gradient, then fits the BFE amplitude A
    (and a constant background, jointly by linear least squares) by minimising
    the residual between the modelled and observed late−early normalised PSF
    difference. The kernel shape (alpha, b, c) is fit nonlinearly.

    Three fitting modes, selected by the inputs:
      * alpha_bfe=None              -> fit alpha, b, c, A, background.
      * alpha_bfe set, b/c None     -> fix alpha, fit b, c, A, background.
      * alpha_bfe, b_bfe, c_bfe set -> fix the whole kernel shape, fit only A
        and background. Use this for faint sources, passing the consensus
        kernel from brighter stars.

    The forward model runs on a cropped region around the star to keep
    the fftconvolve tractable on large detectors.

    Parameters
    ----------
    cube : ndarray (n_int, n_groups, ny, nx), float
        Raw ramp cube.
    alpha_bfe : float or None
        BFE kernel power-law index (default 3.43, the bright-star consensus for
        the quadratic-numerator kernel). If None it is fitted; otherwise fixed.
    b_bfe, c_bfe : float or None
        Quadratic-numerator coefficients. If both are given (with alpha_bfe set)
        the kernel shape is fully fixed and only A and the background are fit;
        if None they are fitted.
    bg_mask : ndarray (ny, nx) bool, optional
        True = background pixels for tau fitting. If None an annulus around
        the detected source is used.
    sci_mask : ndarray (ny, nx) bool, optional
        True = good science pixels. Passed to SEP to exclude bad pixels.
    bfe_early_groups : list of int, optional
        Gradient indices defining early groups for the PSF difference.
        Default: groups 1 to min(3, n_grads//4).
    bfe_late_groups : list of int, optional
        Gradient indices defining late groups for the PSF difference.
        Default: last three valid gradients.
    ap_radius : float
        Aperture radius in pixels for PSF normalisation.
    cut : int
        Half-size of the PSF cutout in pixels.
    fit_r : float
        Radius in pixels within the cutout used for chi-squared fitting.
    verbose : bool
    diagnostics : bool
        If True, save a figure comparing the observed and best-fit model
        late-early PSF differences, the residual, and radial profiles.
    save_path : str or Path, optional
        File path for the diagnostic figure. Defaults to
        'bfe_fit_diagnostics.png' in the current directory.

    Returns
    -------
    A_bfe : float
        Fitted BFE amplitude (None if no source meets the threshold).
    alpha : float
        BFE kernel power-law index (fitted if alpha_bfe is None, else echoed).
    b, c : float
        Quadratic-numerator kernel coefficients K = -(1 + b r + c r^2)/r^alpha.
    sx, sy : int
        Detected star position (x, y).
    """
    import sep
    from scipy.signal import fftconvolve

    cube = np.asarray(cube, dtype=float)
    n_int, n_groups, ny, nx = cube.shape
    n_grads = n_groups - 2
    g_arr = np.arange(n_grads, dtype=float)

    grads = np.diff(cube, axis=1)[:, :n_grads]
    med_grad = np.median(grads, axis=0)

    # Detect brightest round source (excludes elongated edge artifacts)
    detect_img = np.median(grads[:, 1:n_grads], axis=(0, 1)).astype(np.float64)
    sep_mask = (~sci_mask.astype(bool)) if sci_mask is not None else None
    bkg = sep.Background(detect_img, mask=sep_mask)
    img_sub = (detect_img - bkg.back()).astype(np.float64)
    objects = sep.extract(img_sub, thresh=5.0, err=bkg.globalrms, mask=sep_mask)
    edge = 20
    interior = ((objects['x'] > edge) & (objects['x'] < nx - edge) &
                (objects['y'] > edge) & (objects['y'] < ny - edge))
    round_sources = objects[interior & (objects['a'] / objects['b'] < 3)]
    if len(round_sources) == 0:
        round_sources = objects[interior]
    if len(round_sources) == 0:
        round_sources = objects
    if len(round_sources) == 0 or round_sources[np.argsort(round_sources['flux'])[-1]]['flux'] < 50000:
        if verbose:
            print('  No source meets brightness threshold — skipping BFE fit')
        return None, (alpha_bfe if alpha_bfe else 3.43), 0.0, 0.0, nx // 2, ny // 2
    round_sources = round_sources[np.argsort(round_sources['flux'])[::-1]]
    star = round_sources[0]
    sy, sx = int(round(star['y'])), int(round(star['x']))
    if verbose:
        print(f'  Brightest source at x={sx}, y={sy}  flux={star["flux"]:.0f}')

    # Background mask for tau fitting
    yy_full, xx_full = np.mgrid[:ny, :nx]
    r_star = np.sqrt((yy_full - sy)**2 + (xx_full - sx)**2)
    if bg_mask is not None:
        _bg = bg_mask.astype(bool)
    else:
        _bg = (r_star > 15) & (r_star < min(ny, nx) // 3)
        if sci_mask is not None:
            _bg &= sci_mask.astype(bool)

    mean_bg = np.nanmean(med_grad[1:, _bg], axis=1)
    def _exp1(g, C, A, t): return C + A * np.exp(-g / t)
    popt, _ = curve_fit(_exp1, g_arr[1:], mean_bg,
                        p0=[mean_bg[-1], mean_bg[0] - mean_bg[-1], 1.5])
    tau = float(popt[2])
    if verbose:
        print(f'  tau = {tau:.4f} groups')

    exp_g = np.exp(-g_arr / tau)
    ff_col = np.zeros(n_grads); ff_col[0] = -1.0
    X = np.column_stack([np.ones(n_grads), exp_g, ff_col])
    params, _, _, _ = np.linalg.lstsq(X, med_grad.reshape(n_grads, -1), rcond=None)
    rate_map = params[0].reshape(ny, nx)
    Adec_map = params[1].reshape(ny, nx)
    delta_map = params[2].reshape(ny, nx)

    # Crop to a region around the star for the forward model
    kh = 20
    crop = cut + kh + 30
    y0, y1 = max(0, sy - crop), min(ny, sy + crop + 1)
    x0, x1 = max(0, sx - crop), min(nx, sx + crop + 1)
    rate_c = rate_map[y0:y1, x0:x1]
    Adec_c = Adec_map[y0:y1, x0:x1]
    delta_c = delta_map[y0:y1, x0:x1]
    grads_c = grads[:, :, y0:y1, x0:x1]
    cy, cx = sy - y0, sx - x0   # star position in cropped frame
    nyc, nxc = rate_c.shape

    # Group selections: skip group 1 (still affected by first-frame residuals)
    # use 2-3 groups from the early/late thirds of the valid ramp
    if bfe_early_groups is None:
        n_e = max(2, min(3, n_grads // 4))
        start = 1 if n_grads < 8 else 2
        bfe_early_groups = list(range(start, start + n_e))
    if bfe_late_groups is None:
        n_e = max(2, min(3, n_grads // 4))
        bfe_late_groups = list(range(n_grads - n_e, n_grads))

    yy_c, xx_c = np.mgrid[:2*cut+1, :2*cut+1]
    r_map_c = np.sqrt((yy_c - cut)**2 + (xx_c - cut)**2)

    ii, jj = np.mgrid[-kh:kh+1, -kh:kh+1].astype(float)
    rker = np.sqrt(ii**2 + jj**2)

    def _make_kernel(al, b, c):
        # Flux-conserving BFE kernel: K = -(1 + b r + c r^2) / r^al, centre set
        # to -(off-centre sum) so sum(K)=0. The quadratic numerator yields a
        # compact kernel and a physical, source-independent index al~3.
        with np.errstate(divide='ignore', invalid='ignore'):
            Kk = np.where(rker > 0, -(1.0 + b * rker + c * rker**2) / rker**al, 0.0)
        Kk[kh, kh] = -Kk.sum()
        return Kk

    def _build_true(rate_c, Adec_c, delta_c):
        tg_all = np.zeros((n_grads, nyc, nxc))
        Qg = np.zeros((n_grads, nyc, nxc))
        Qa = np.zeros((nyc, nxc))
        for g in range(n_grads):
            tg = rate_c + Adec_c * np.exp(-g / tau)
            if g == 0:
                tg = tg - delta_c
            tg_all[g] = tg
            Qg[g] = Qa
            Qa = Qa + tg
        return tg_all, Qg

    med_obs_c = np.median(grads_c, axis=0)

    # Normalisation: background-subtract each group and divide by its TOTAL
    # observed PSF flux, so each group is a unit-flux PSF and the late-early
    # difference is flux-conserving by construction (no pedestal, no arbitrary
    # core aperture, no model dependence). The divisor is observed data
    # (A-independent), so the model stays exactly linear in A.
    _norm_ap = r_map_c <= (cut - 6)                       # encloses PSF + ring
    _bg_ann = (r_map_c > (cut - 6)) & (r_map_c <= (cut - 1))

    def _cut2d(img):
        return img[cy-cut:cy+cut+1, cx-cut:cx+cut+1]

    def _starflux(img):
        c = _cut2d(img)
        return float((c - np.median(c[_bg_ann]))[_norm_ap].sum())
    _fobs = np.array([_starflux(med_obs_c[g]) for g in range(n_grads)])

    def _norm_diff(stack):
        def _acc(glist):
            s = 0.0
            for g in glist:
                c = _cut2d(stack[g])
                s = s + (c - np.median(c[_bg_ann])) / _fobs[g]
            return s / len(glist)
        return _acc(bfe_late_groups) - _acc(bfe_early_groups)

    from scipy.optimize import minimize as _minimize
    _opt = dict(method='Powell',
                options={'xtol': 1e-5, 'ftol': 1e-9, 'maxiter': 20000})

    # fit radius from the initial (raw) maps; held fixed across iterations
    true_grads, Q_grads = _build_true(rate_c, Adec_c, delta_c)
    obs_diff = _norm_diff(med_obs_c)
    noise_diff = np.std([_norm_diff(grads_c[i]) for i in range(n_int)], axis=0) / np.sqrt(n_int)
    noise_diff = np.clip(noise_diff, noise_diff[noise_diff > 0].min() * 0.1, None)
    if fit_r is None:
        snr_profile = np.array([
            np.mean(np.abs(obs_diff[np.round(r_map_c).astype(int) == ri])) /
            np.mean(noise_diff[np.round(r_map_c).astype(int) == ri])
            for ri in range(1, cut)])
        above = np.where(snr_profile > 2.0)[0]
        fit_r = max(5, int(above[-1]) + 1) if len(above) > 0 else 5
        if verbose:
            print(f'  Auto fit_r = {fit_r} px (SNR-based)')
    fit_mask = r_map_c <= fit_r

    def _fit_bfe_once(true_grads, Q_grads):
        # Fit kernel shape (alpha, b, c) nonlinearly; A and a constant
        # background are linear (solved by weighted least squares).
        obs = _norm_diff(med_obs_c)
        noise = np.std([_norm_diff(grads_c[i]) for i in range(n_int)], axis=0) / np.sqrt(n_int)
        noise = np.clip(noise, noise[noise > 0].min() * 0.1, None)
        D_true = _norm_diff(true_grads)
        wfit = (1.0 / noise)[fit_mask]
        obsfit = obs[fit_mask]; dtruefit = D_true[fit_mask]
        onesfit = np.ones(int(fit_mask.sum()))

        def deflection(al, b, c):
            Kk = _make_kernel(al, b, c)
            defl = np.zeros((n_grads, nyc, nxc))
            for g in range(n_grads):
                defl[g] = fftconvolve(Q_grads[g] * true_grads[g], Kk, mode='same')
            return _norm_diff(defl)

        def solve_AB(Ddefl):
            Dd = Ddefl[fit_mask]
            r0 = wfit * (dtruefit - obsfit)
            M = np.column_stack([-wfit * Dd, wfit * onesfit])
            x, *_ = np.linalg.lstsq(M, -r0, rcond=None)
            return x[0], x[1], float(np.sum((r0 + M @ x)**2))

        def chi2_shape(al, b, c):
            if not (0.5 <= al <= 6.0 and -0.6 <= b <= 0.6 and -0.08 <= c <= 0.08):
                return 1e30
            Dd = deflection(al, b, c)
            if not np.all(np.isfinite(Dd)):
                return 1e30
            return solve_AB(Dd)[2]

        if alpha_bfe is None:
            res = _minimize(lambda p: chi2_shape(p[0], p[1], p[2]),
                            x0=[3.43, -0.4, 0.04], **_opt)
            al, b, c = res.x
        elif b_bfe is None or c_bfe is None:
            res = _minimize(lambda p: chi2_shape(alpha_bfe, p[0], p[1]),
                            x0=[-0.4, 0.04], **_opt)
            al, b, c = alpha_bfe, res.x[0], res.x[1]
        else:
            al, b, c = alpha_bfe, b_bfe, c_bfe
        A_, bg_, chi2_ = solve_AB(deflection(al, b, c))
        sim = D_true - A_ * deflection(al, b, c) + bg_
        return A_, al, b, c, bg_, chi2_, obs, D_true, sim

    # Iterate RCD <-> BFE: the reset-decay maps (rate, Adec, delta) are refit
    # from the BFE-removed gradient each pass, so the decay can no longer absorb
    # the BFE ring. The two are coupled because BFE depends on the accumulated
    # charge Q built from the reset-decay model, so this converges to the
    # simultaneous solution without a full coupled nonlinear solve.
    MAX_FIT_ITER, A_TOL = 12, 0.01
    A_prev = None
    for _it in range(MAX_FIT_ITER):
        true_grads, Q_grads = _build_true(rate_c, Adec_c, delta_c)
        (A_bfe_fit, alpha_fit, b_fit, c_fit, bg_fit, _chi2_fit,
         obs_diff, D_true, sim_diff) = _fit_bfe_once(true_grads, Q_grads)
        if verbose:
            print(f'  [iter {_it}] A_bfe = {A_bfe_fit:.4e}  alpha = {alpha_fit:.4f}  '
                  f'b = {b_fit:.4f}  c = {c_fit:.4f}  '
                  f'chi2/n = {_chi2_fit/max(int(fit_mask.sum())-2, 1):.3f}')
        # stop once A_bfe has stabilised (RCD/BFE split has converged)
        if A_prev is not None and abs(A_bfe_fit - A_prev) <= A_TOL * abs(A_bfe_fit):
            break
        A_prev = A_bfe_fit
        if _it == MAX_FIT_ITER - 1:
            break
        # remove BFE from the observed median gradient, refit RCD maps
        Kk = _make_kernel(alpha_fit, b_fit, c_fit)
        med_corr = np.empty_like(med_obs_c)
        for g in range(n_grads):
            med_corr[g] = med_obs_c[g] + A_bfe_fit * fftconvolve(
                Q_grads[g] * true_grads[g], Kk, mode='same')
        pc, _, _, _ = np.linalg.lstsq(X, med_corr.reshape(n_grads, -1), rcond=None)
        rate_c = pc[0].reshape(nyc, nxc)
        Adec_c = pc[1].reshape(nyc, nxc)
        delta_c = pc[2].reshape(nyc, nxc)

    if diagnostics:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        res_diff = obs_diff - sim_diff

        rr = np.round(r_map_c).astype(int)
        r_int = np.arange(0, cut)
        rp_obs = np.array([np.nanmean(obs_diff[rr == ri]) for ri in r_int])
        rp_sim = np.array([np.nanmean(sim_diff[rr == ri]) for ri in r_int])

        vabs = np.nanpercentile(np.abs(obs_diff), 99.5)
        ext = [-cut - 0.5, cut + 0.5, -cut - 0.5, cut + 0.5]

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, img, title in [
            (axes[0, 0], obs_diff, 'Observed late$-$early'),
            (axes[0, 1], sim_diff,
             f'Model (A={A_bfe_fit:.2e}, α={alpha_fit:.2f}, b={b_fit:.2f}, c={c_fit:.3f})'),
            (axes[1, 0], res_diff, 'Residual (obs$-$model)'),
        ]:
            im = ax.imshow(img, origin='lower', cmap='RdBu_r',
                           vmin=-vabs, vmax=vabs, extent=ext)
            fig.colorbar(im, ax=ax, label=r'Norm. $\Delta$flux')
            ax.set_title(title)
            ax.set_xlabel(r'$\Delta x$ (px)')
            ax.set_ylabel(r'$\Delta y$ (px)')

        ax = axes[1, 1]
        ax.plot(r_int, rp_obs, 'k-', lw=1.5, label='Observed')
        ax.plot(r_int, rp_sim, color='C3', ls='--', lw=1.5, label='Model')
        ax.axhline(0, color='k', lw=0.5, ls=':')
        ax.axvline(fit_r, color='C0', lw=1.0, ls='--', label=f'fit_r = {fit_r} px')
        ax.set_xlabel('Radius (px)')
        ax.set_ylabel(r'Mean $\Delta$flux')
        ax.set_title('Radial profile')
        ax.legend(fontsize=8)

        fig.suptitle(f'BFE fit diagnostics  (star x={sx}, y={sy})',
                     fontsize=11, fontweight='bold')
        fig.tight_layout()
        if save_path is None:
            save_path = 'bfe_fit_diagnostics.png'
        fig.savefig(Path(save_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        if verbose:
            print(f'  Saved BFE fit diagnostics to {save_path}')

    if return_model:
        model = dict(obs=obs_diff, model=sim_diff, r_map=r_map_c,
                     A=A_bfe_fit, alpha=alpha_fit, b=b_fit, c=c_fit,
                     bg=bg_fit, chi2_n=_chi2_fit / max(int(fit_mask.sum()) - 2, 1),
                     cy=cy, cx=cx, cut=cut)
        return A_bfe_fit, alpha_fit, b_fit, c_fit, sx, sy, model
    return A_bfe_fit, alpha_fit, b_fit, c_fit, sx, sy


def correct_bfe_rcd(cube, A_bfe=1.035e-6, alpha_bfe=3.43, b_bfe=-0.50, c_bfe=0.056,
                    bg_mask=None, late_groups=None, verbose=False,
                    fit_bfe=False, sci_mask=None,
                    bfe_early_groups=None, bfe_late_groups=None,
                    ap_radius=5, cut=20, fit_r=10,
                    diagnostics=False, save_path=None):
    """
    Joint BFE + reset-decay correction for MIRI ramp data.

    Three sequential steps applied to gradients:
      1. Causal BFE inversion: each gradient is divided by (1 - A_bfe * K⊛Q)
         where Q is the accumulated charge from all previous groups.
      2. Parametric RCD subtraction: fit C + A*exp(-g/tau) with tau global
         (from background pixels) and [A, C, delta] per pixel via lstsq.
         Subtract the fitted decay from every integration.
      3. Non-parametric residual removal: subtract the per-pixel per-group
         median over integrations, then add back the flat rate estimated from
         late groups. Removes any residual group-correlated structure not
         captured by the exponential model.

    The last gradient (last-frame anomaly) is BFE-corrected but excluded from
    the RCD and median subtraction steps, matching the convention in
    correct_reset_decay.

    Parameters
    ----------
    cube : ndarray (n_int, n_groups, ny, nx), float
        Raw SCI data from uncal.fits.
    A_bfe : float
        BFE kernel amplitude (default 1.035e-6).
    alpha_bfe : float
        BFE kernel power-law index (default 3.43, bright-star consensus).
    b_bfe, c_bfe : float
        Quadratic-numerator coefficients of the kernel
        K = -(1 + b r + c r^2) / r^alpha (centre = -(off-centre sum), so
        sum(K)=0). Defaults (b=-0.50, c=0.056) are the bright-star consensus
        kernel; b=c=0 reduces K to the bare power law. When fit_bfe=True these
        are overwritten by the fitted values.
    bg_mask : ndarray (ny, nx) bool, optional
        True = background pixels used to fit the global RCD timescale tau.
        If None, all pixels are used.
    late_groups : list of int, optional
        Gradient indices used to estimate the flat rate for median subtraction.
        Defaults to the last three good gradients.
    verbose : bool
        Print BFE inversion progress.
    fit_bfe : bool
        If True, ignore A_bfe and fit it from the brightest source using
        fit_bfe_params before applying the correction.
    sci_mask : ndarray (ny, nx) bool, optional
        True = good science pixels. Passed to fit_bfe_params for SEP source
        detection. Only used when fit_bfe=True.
    bfe_early_groups, bfe_late_groups : list of int, optional
        Gradient indices for the early/late PSF groups used in the BFE fit.
        Only used when fit_bfe=True.
    ap_radius, cut, fit_r : float
        PSF normalisation aperture, cutout half-size, and fit radius in pixels.
        Only used when fit_bfe=True.
    diagnostics : bool
        If True, save a figure showing the global tau fit, the decay
        amplitude map, the BFE correction size per group, and the
        background gradient profile at each correction stage.
    save_path : str or Path, optional
        File path for the diagnostic figure. Defaults to
        'bfe_rcd_diagnostics.png' in the current directory.

    Returns
    -------
    cube_cor : ndarray (n_int, n_groups, ny, nx)
        Corrected SCI cube reconstructed from corrected gradients.
        Group 0 is unchanged (reset level reference).
    """
    from scipy.signal import fftconvolve

    cube = np.asarray(cube, dtype=float)
    n_int, n_groups, ny, nx = cube.shape
    n_grads_all = n_groups - 1        # all gradients
    n_grads = n_groups - 2            # gradients to correct (exclude last-frame anomaly)

    grads_raw = np.diff(cube, axis=1)   # (n_int, n_grads_all, ny, nx)
    g_arr = np.arange(n_grads, dtype=float)

    if late_groups is None:
        late_groups = list(range(n_grads - 3, n_grads))

    if fit_bfe:
        if verbose:
            print('Fitting A_bfe from brightest source...')
        fit_result = fit_bfe_params(
            cube, alpha_bfe=alpha_bfe,
            bg_mask=bg_mask, sci_mask=sci_mask,
            bfe_early_groups=bfe_early_groups, bfe_late_groups=bfe_late_groups,
            ap_radius=ap_radius, cut=cut, fit_r=fit_r, verbose=verbose)
        A_bfe_fit, alpha_fit, b_fit, c_fit, _sx, _sy = fit_result
        if A_bfe_fit is None:
            if verbose:
                print('No source meets brightness threshold — skipping BFE correction')
            A_bfe = 0.0
        else:
            A_bfe = A_bfe_fit
            alpha_bfe, b_bfe, c_bfe = alpha_fit, b_fit, c_fit
            if verbose:
                print(f'Using fitted A_bfe={A_bfe:.4e}  alpha={alpha_bfe:.3f}  '
                      f'b={b_bfe:.3f}  c={c_bfe:.4f} at x={_sx}, y={_sy}')

    # Step 1: causal iterative BFE correction — flux conserving
    # Forward model: grad_obs = true_grad - A * K ⊛ (Q * true_grad)
    # Iterative inversion: true_grad^(n+1) = grad_obs + A * K ⊛ (Q * true_grad^(n))
    # Since K sums to zero, K̂(0)=0 → total image flux is exactly conserved.
    N_ITER = 3
    kh = 20
    ii, jj = np.mgrid[-kh:kh+1, -kh:kh+1].astype(float)
    r = np.sqrt(ii**2 + jj**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        K = np.where(r > 0, -(1.0 + b_bfe * r + c_bfe * r**2) / r**alpha_bfe, 0.0)
    K[kh, kh] = -K.sum()

    grads_bfe = grads_raw.copy()
    Q_med = np.zeros((ny, nx))
    for g in range(n_grads_all):
        if g > 0:
            Q_med = Q_med + np.median(grads_bfe[:, g-1], axis=0)
        med_obs_g = np.median(grads_raw[:, g], axis=0)
        true_grad_est = med_obs_g.copy()
        for _ in range(N_ITER):
            true_grad_est = med_obs_g + A_bfe * fftconvolve(Q_med * true_grad_est, K, mode='same')
        KQg = fftconvolve(Q_med * true_grad_est, K, mode='same')
        grads_bfe[:, g] = grads_raw[:, g] + A_bfe * KQg[None]
        if verbose:
            print(f'  BFE g={g}', end='\r')
    if verbose:
        print()

    # Step 2: fit global tau from BFE-corrected background, excluding g=0
    med_bfe = np.median(grads_bfe[:, :n_grads], axis=0)   # (n_grads, ny, nx)
    g_fit = g_arr[1:]
    if bg_mask is not None:
        mean_bg = np.nanmean(med_bfe[1:, bg_mask], axis=1)
    else:
        mean_bg = np.nanmean(med_bfe[1:].reshape(n_grads-1, -1), axis=1)

    def _exp1(g, C, A, tau): return C + A * np.exp(-g / tau)
    popt, _ = curve_fit(_exp1, g_fit, mean_bg,
                        p0=[mean_bg[-1], mean_bg[0] - mean_bg[-1], 1.5])
    tau = float(popt[2])

    exp_g = np.exp(-g_arr / tau)
    ff_col = np.zeros(n_grads); ff_col[0] = -1.0
    X = np.column_stack([np.ones(n_grads), exp_g, ff_col])
    params, _, _, _ = np.linalg.lstsq(
        X, med_bfe.reshape(n_grads, -1), rcond=None)
    Adec_map = params[1].reshape(ny, nx)
    delta_map = params[2].reshape(ny, nx)

    grads_joint = grads_bfe.copy()
    for g in range(n_grads):
        decay_g = Adec_map * np.exp(-g / tau)
        if g == 0:
            grads_joint[:, 0] = grads_bfe[:, 0] - decay_g[None] + delta_map[None]
        else:
            grads_joint[:, g] = grads_bfe[:, g] - decay_g[None]

    # Step 3: non-parametric median subtraction
    med_joint = np.median(grads_joint[:, :n_grads], axis=0)   # (n_grads, ny, nx)
    C_hat = np.mean(med_joint[late_groups], axis=0)            # (ny, nx)

    grads_cor = grads_joint.copy()
    for g in range(n_grads):
        grads_cor[:, g] = grads_joint[:, g] - med_joint[g][None] + C_hat[None]

    # Reconstruct corrected cube: group 0 unchanged, integrate corrected gradients
    cube_cor = cube.copy()
    cube_cor[:, 1:] = cube[:, :1] + np.cumsum(grads_cor, axis=1)

    if diagnostics:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        ax = axes[0, 0]
        g_fine = np.linspace(1, n_grads - 1, 200)
        ax.plot(g_fit, mean_bg, 'o', color='k', ms=5, label='Background mean')
        ax.plot(g_fine, popt[0] + popt[1] * np.exp(-g_fine / tau), '--',
                color='C3', lw=1.5, label=f'Fit  τ={tau:.2f} grp')
        ax.set_xlabel('Gradient index')
        ax.set_ylabel('Mean gradient (DN/group)')
        ax.set_title('Global τ fit (after BFE step)')
        ax.legend(fontsize=8)
        ax.set_xticks(g_arr.astype(int))

        ax = axes[0, 1]
        vmax = np.nanpercentile(Adec_map, 99)
        im = ax.imshow(Adec_map, origin='lower', vmin=0, vmax=vmax,
                       cmap='viridis')
        fig.colorbar(im, ax=ax, label='DN/group')
        ax.set_title('Decay amplitude A')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax = axes[1, 0]
        bfe_size = np.array([
            np.nanpercentile(np.abs(np.median(grads_bfe[:, g] - grads_raw[:, g], axis=0)), 99.9)
            for g in range(n_grads_all)])
        ax.plot(range(n_grads_all), bfe_size, 'o-', color='k', ms=5)
        ax.set_yscale('log')
        ax.set_xlabel('Gradient index')
        ax.set_ylabel('99.9th pct |BFE correction| (DN/group)')
        ax.set_title(f'BFE step size  (A={A_bfe:.2e}, α={alpha_bfe:.3f})')
        ax.set_xticks(range(n_grads_all))

        ax = axes[1, 1]
        bg = bg_mask if bg_mask is not None else np.ones((ny, nx), dtype=bool)
        for grads_i, label, color in [
            (grads_raw, 'Raw', 'k'),
            (grads_joint, 'BFE + RCD', 'C0'),
            (grads_cor, '+ median sub', 'C3'),
        ]:
            prof = np.nanmean(np.median(grads_i[:, :n_grads], axis=0)[:, bg], axis=1)
            ax.plot(g_arr, prof, 'o-', ms=4, color=color, label=label)
        ax.set_xlabel('Gradient index')
        ax.set_ylabel('Mean background gradient (DN/group)')
        ax.set_title('Background profile by stage')
        ax.legend(fontsize=8)
        ax.set_xticks(g_arr.astype(int))

        fig.suptitle('BFE + RCD correction diagnostics',
                     fontsize=11, fontweight='bold')
        fig.tight_layout()
        if save_path is None:
            save_path = 'bfe_rcd_diagnostics.png'
        fig.savefig(Path(save_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        if verbose:
            print(f'Saved correction diagnostics to {save_path}')

    return cube_cor


def correct_ramp(cube, C_map):
    """
    Apply the per-pixel correction to a ramp cube, returning corrected
    group-gradients ready for difference imaging.

    Parameters
    ----------
    cube  : ndarray, shape (n_int, n_groups, ny, nx)
        Stage-1 corrected ramp cube (raw group values).
    C_map : ndarray, shape (n_groups-1, ny, nx)
        Correction map from build_correction_map.

    Returns
    -------
    grads_corrected : ndarray, shape (n_int, n_groups-1, ny, nx)
        All group-gradients are on a common photometric scale.
        Subtract any two frames directly for difference imaging.
    """
    grads = np.diff(cube, axis=1).astype(float)
    return grads * C_map[None]
