import warnings

import numpy as np
from pathlib import Path
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from scipy.stats import f as f_dist
from astropy.stats import sigma_clipped_stats

# fit_r floor-to-FWHM ratios, derived from the original MIRI calibration:
# the MIRI test datasets (wolf-359, evlac, dxcancri -- 3 of 5 -- all use
# F2100W; JDOX FWHM = 6.127 px) were fit with the hardcoded fit_r floors
# below (8 px for fit_migration_params, 5 px for fit_bfe_params). Both are
# tied to that specific filter's PSF size, not a universal constant, so
# applying them as fixed numbers to NIRCam (FWHM ~2-3x smaller depending
# on filter) reaches well past the real signal into unrelated background.
# Instead, carry the floor as a RATIO to FWHM (verified against the actual
# calibration point) and re-derive the floor per-exposure from that
# exposure's own measured PSF FWHM (in pixels; e.g. Jurassic._get_psf_fwhm_px
# for NIRCam, JDOX-tabulated for MIRI), passed in via psf_fwhm_px.
_MIRI_CAL_FWHM_PX = 6.127          # F2100W, JDOX
_FIT_R_FWHM_RATIO_MIGRATION = 8.0 / _MIRI_CAL_FWHM_PX
_FIT_R_FWHM_RATIO_KERNEL = 5.0 / _MIRI_CAL_FWHM_PX


def _mask_radial_outliers(data, r_map, mask, nsigma=8.0):
    """
    Rejects single-pixel outliers (cosmic rays, hot pixels) from ``mask`` in
    place, by comparing each pixel to the MEDIAN (not mean -- the outlier
    itself would drag a mean) of all pixels at the same integer radius in
    ``r_map``, scaled by the radial bin's median absolute deviation. Used to
    protect a per-pixel chi2 fit from a single bad pixel dominating the sum
    of squares regardless of the fit aperture -- verified directly: a single
    cosmic-ray pixel at r~13 (in an aperture of ~500 pixels) inflated
    chi2/n by ~6x on an otherwise clean NIRCam BFE fit.

    Returns the number of pixels rejected.
    """
    r_int = np.round(r_map[mask]).astype(int)
    vals = data[mask]
    med_map = np.zeros_like(vals)
    mad_map = np.zeros_like(vals)
    for ri in np.unique(r_int):
        sel = r_int == ri
        m = np.nanmedian(vals[sel])
        mad = np.nanmedian(np.abs(vals[sel] - m))
        med_map[sel] = m
        mad_map[sel] = mad
    outlier = np.abs(vals - med_map) > nsigma * 1.4826 * np.maximum(mad_map, 1e-12)
    idx = np.flatnonzero(mask)[outlier]
    mask.flat[idx] = False
    return int(outlier.sum())


def _inpaint_bad_pixels(img, bad_mask):
    """
    Replaces bad_mask==True pixels in img with a local interpolated
    estimate from surrounding good pixels, instead of leaving the raw
    (possibly detector-defect- or cosmic-ray-driven) value in place.

    This matters because rate_c/Adec_c/delta_c (the per-pixel maps that
    seed the forward model in fit_bfe_params/fit_migration_params) are
    built from a per-pixel median/lstsq fit over the WHOLE cropped region,
    with no DQ/sci_mask screening at all -- masking only the final chi2
    region (fit_mask/fitmask) isn't enough, since a bad pixel anywhere in
    the crop still biases the model's own inputs, and the forward model
    then just reproduces that contamination (verified directly: a
    permanently DO_NOT_USE-flagged pixel cluster, plus a genuine one-group
    cosmic ray next to it, showed up matching in BOTH the "observed" and
    "model" diagnostic panels -- not because the model predicted it, but
    because both were built from the same unmasked input).
    """
    if bad_mask is None or not np.any(bad_mask):
        return img
    from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
    work = np.asarray(img, dtype=float).copy()
    work[bad_mask] = np.nan
    kernel = Gaussian2DKernel(x_stddev=2)
    filled = interpolate_replace_nans(work, kernel)
    still_bad = ~np.isfinite(filled)
    if np.any(still_bad):
        filled[still_bad] = img[still_bad]
    return filled


def _grad_to_group_indices(grad_indices):
    """
    Maps gradient indices to the GROUPDQ group indices that must be
    checked for each: gradient g = cube[g+1] - cube[g], so a jump landing
    in that gradient can be flagged on group g OR group g+1 -- verified
    directly: a real jump showed DQ=0 on the earlier group and DQ=4
    (JUMP_DET) only on the later one. Checking only group index g (as if
    gradient and group indices were the same axis) misses jumps flagged
    on the later group, which is where JWST's jump step actually records
    them.
    """
    out = set()
    for g in grad_indices:
        out.add(g)
        out.add(g + 1)
    return sorted(out)


def _dq_exclusion_mask(dq, y0, y1, x0, x1, groups, protect_center=None, protect_r=2):
    """
    Builds a (y1-y0, x1-x0) boolean mask (True = bad/excluded) over
    dq[..., y0:y1, x0:x1], from GROUPDQ JUMP_DET (4) or SATURATED (2)
    flags in any of ``groups``, unioned over integrations. ``dq`` may be
    (n_int, n_groups, ny, nx) or (n_groups, ny, nx). Returns None if dq is
    None.

    protect_center : (row, col) in the returned mask's own coordinates,
    optional. If given, JUMP_DET flags within protect_r of this point are
    NOT excluded -- jump detection is well known to false-positive on the
    target source's own bright, real signal, especially on short ramps
    (few groups means little baseline to distinguish a genuine cosmic ray
    from ordinary source brightness/Poisson noise). Verified directly: the
    star's own peak pixel was flagged JUMP_DET, and excluding it removed
    the actual signal being measured. SATURATED is NOT protected -- a
    genuinely saturated core pixel really doesn't carry usable linear
    signal, that exclusion is legitimate.
    """
    if dq is None:
        return None
    dq = np.asarray(dq)
    if dq.ndim == 3:
        dq = dq[np.newaxis, ...]
    ny, nx = dq.shape[-2], dq.shape[-1]
    out = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    if y0 < 0 or x0 < 0 or y1 > ny or x1 > nx:
        return out
    core = None
    if protect_center is not None:
        yy, xx = np.mgrid[:y1 - y0, :x1 - x0]
        core = (yy - protect_center[0]) ** 2 + (xx - protect_center[1]) ** 2 <= protect_r ** 2
    for g in groups:
        if g < 0 or g >= dq.shape[1]:
            continue
        patch = dq[:, g, y0:y1, x0:x1]
        sat = ((patch & 2) > 0).any(axis=0)
        jump = ((patch & 4) > 0).any(axis=0)
        if core is not None:
            jump = jump & ~core
        out |= sat | jump
    return out


def _isolated_brightest(sources, all_objects, min_sep, verbose=False):
    """
    Returns the brightest object in ``sources`` that has no OTHER detected
    object (from ``all_objects``, the full unfiltered detection table)
    within ``min_sep`` pixels of it -- rejects crowded/blended candidates
    before they're ever selected as the fit target. Verified directly: a
    close visual binary (~9 px separation, PSF wings visibly overlapping)
    picked as "the brightest source" produced a smeared, elongated model
    that no amount of masking the companion's core alone could fully
    clean up, because the two stars' real flux genuinely blends in the
    space between them.

    Falls back to the single brightest object (with a warning) if no
    candidate is isolated, rather than returning nothing.
    """
    order = np.argsort(sources['flux'])[::-1]
    for i in order:
        cand = sources[i]
        d2 = (all_objects['x'] - cand['x']) ** 2 + (all_objects['y'] - cand['y']) ** 2
        n_near = int(np.sum((d2 > 1e-6) & (d2 <= min_sep ** 2)))
        if n_near == 0:
            return cand
    if verbose:
        print(f'  No isolated source found within {min_sep:.0f} px of any neighbour -- '
              f'falling back to the brightest (possibly blended) candidate')
    return sources[order[0]]


def _other_source_mask(detect_img, exclude_y, exclude_x, sci_mask=None, exclude_r=3.0):
    """
    Detects sources in detect_img via SEP and builds a mask of pixels
    belonging to any source OTHER than the one nearest (exclude_y,
    exclude_x) (the star being fit). A wide crop used to build
    rate_c/Adec_c/delta_c (or the flux-normalisation aperture) can contain
    other real, unflagged sources -- DQ flags don't catch this, it's real
    signal, just not the target's. Verified directly: a genuine second
    point source ~9 px from a target leaked into the forward model's own
    input maps unmasked, producing a second, unrelated blob in the model
    prediction that had nothing to do with the target's own migration.

    detect_img : ndarray, the same shape as the mask to return (e.g. a
    crop of the per-pixel rate/gradient map).
    """
    import sep
    sub = np.ascontiguousarray(detect_img, dtype=np.float64)
    sep_mask = np.ascontiguousarray(~sci_mask.astype(bool)) if sci_mask is not None else None
    bkg = sep.Background(sub, mask=sep_mask)
    obj = sep.extract((sub - bkg.back()).astype(np.float64), 8.0,
                      err=bkg.globalrms, mask=sep_mask, minarea=5)
    mask = np.zeros(sub.shape, dtype=bool)
    if len(obj) == 0:
        return mask
    d2 = (obj['x'] - exclude_x) ** 2 + (obj['y'] - exclude_y) ** 2
    target_idx = np.argmin(d2)
    yy, xx = np.mgrid[:sub.shape[0], :sub.shape[1]]
    for i, o in enumerate(obj):
        if i == target_idx:
            continue
        r = max(exclude_r, 2.5 * max(float(o['a']), float(o['b'])))
        mask |= (yy - o['y']) ** 2 + (xx - o['x']) ** 2 <= r ** 2
    return mask


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
                   ap_radius=5, cut=None, fit_r=None, verbose=False,
                   diagnostics=False, save_path=None, return_model=False,
                   instrument='miri', fit_rcd_decay=None, drop_last_gradient=None,
                   master_ref=None, flux_min=None, snr_min=10.0, dq=None,
                   psf_fwhm_px=None):
    """
    Find the brightest source in the image and fit the BFE kernel via the
    source-centric forward model, K = -(1 + b r + c r^2) / r^alpha.

    psf_fwhm_px : float, optional
        This exposure's PSF FWHM in pixels (e.g. from
        Jurassic._get_psf_fwhm_px). When given and fit_r is not set
        explicitly, the auto fit_r floor is re-derived from this exposure's
        own PSF size (see _FIT_R_FWHM_RATIO_KERNEL) instead of the fixed
        MIRI-F2100W-calibrated floor -- important for NIRCam, whose PSF
        FWHM varies ~2-3x smaller across filters than the MIRI filter the
        default was calibrated against, so the fixed floor reaches well
        past the real signal into unrelated background/field structure.
    dq : ndarray (n_int, n_groups, ny, nx) or (n_groups, ny, nx), optional
        GROUPDQ array from the exposure. When given, any pixel flagged
        JUMP_DET (cosmic ray) or SATURATED in any of the groups used to
        build the early/late gradient difference is excluded from the fit
        region -- this is the direct, physically-grounded complement to
        the statistical radial-outlier rejection (_mask_radial_outliers,
        always applied): DQ catches contaminants the pipeline's own jump
        step already identified (including ones that survive statistical
        screening, e.g. a 2-3 pixel cluster that doesn't stand out enough
        within its own radial bin), while the statistical screen catches
        anything DQ doesn't (verified necessary on real NIRCam data: a
        single-pixel cosmic ray corrupted an otherwise clean fit even
        though most of its ring was fine).
    master_ref : ndarray (ny, nx), optional
        A clean, deep, cosmic-ray-free detection image on the same pixel
        grid as ``cube`` (e.g. a jump-rejected rate image or a multi-
        exposure combined mosaic). When given, source detection (finding
        the brightest star to fit) runs on this image instead of on the
        cube's own median gradient. Strongly recommended: a single ramp's
        median gradient can be dominated by an undetected cosmic-ray hit
        (a "snowball" event spanning 2+ groups survives 3-group median
        rejection), which silently makes the BFE fit measure the cosmic
        ray's spatial profile instead of a real star's. The pipeline's own
        GROUPDQ jump flag would catch this per-group, but a plain median
        image has no such information — a master reference sidesteps the
        problem entirely by never looking at the single-ramp gradient data
        for detection at all. Only used for detection; the BFE fit itself
        still uses the star's position to extract ramp/gradient data from
        ``cube``.
    flux_min : float, optional
        Minimum SEP-measured flux (in the detection image's own units) for
        the brightest source to be used, when detection runs on the cube's
        own gradient image with instrument='miri' (the original,
        MIRI-calibrated criterion, tuned for gradient DN/group). Defaults
        to 50000. Ignored (an SNR cut is used instead, see snr_min) when
        master_ref is given, or when instrument='nircam': gradient DN/group
        and master_ref units differ per instrument/detector/exposure
        (different gain, full well, integration time), so a fixed absolute
        threshold isn't portable across MIRI and NIRCam the way it was
        implicitly assumed to be.
    snr_min : float, default 10.0
        Minimum source flux / (background rms * sqrt(npix)) required to
        accept the brightest detected source, used whenever flux_min isn't
        (see above).

    instrument : {'miri', 'nircam'}
        Case-insensitive. Selects instrument-appropriate defaults for
        fit_rcd_decay, drop_last_gradient, and (when bfe_early_groups/
        bfe_late_groups aren't given explicitly) the early/late group
        split:
          * 'miri' (default): fit_rcd_decay=True, drop_last_gradient=True
            (exclude the last-frame anomaly), multi-group early/late
            split -- the original, MIRI-calibrated behaviour.
          * 'nircam': fit_rcd_decay=False (HgCdTe detectors have no RCD
            -- fitting a decay timescale to background that has none is
            both meaningless and numerically fragile on short ramps),
            drop_last_gradient=False (the last-frame-anomaly exclusion
            is a MIRI-derived convention, not established for NIRCam),
            single-group early=[0]/late=[last] split -- gradient 0 is
            included (verified directly on real NIRCam data: the
            background gradient level is flat across all groups including
            0, with no decaying-transient shape, so the MIRI-derived
            "first-frame contamination" assumption that used to exclude it
            here does not hold for NIRCam and was dropped. NIRCam ramps
            are also usually too short for the multi-group averaging
            MIRI's default uses -- verified directly: it degenerates to
            an empty/identical early-late split on a 5-group NIRCam ramp.
            Explicit fit_rcd_decay/drop_last_gradient/
            bfe_early_groups/bfe_late_groups still override these.
    fit_rcd_decay, drop_last_gradient : bool or None
        Explicit override for the instrument default. None (default)
        resolves from `instrument`.

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

    instrument = instrument.lower() if instrument else instrument

    if fit_rcd_decay is None:
        fit_rcd_decay = (instrument == 'miri')
    if drop_last_gradient is None:
        drop_last_gradient = (instrument == 'miri')
    if cut is None:
        # cut sets norm_ap's radius (cut-6) and bg_ann's span (cut-6..cut-1)
        # -- MIRI's default (cut=20 -> radius 14 px) was tuned for MIRI's
        # more extended PSF/pixel scale. NIRCam's PSF is far more compact,
        # so that aperture reaches well past the real signal into unrelated
        # background/field structure -- more area for stray cosmic rays or
        # other sources to land in and bias the flux-normalisation sum
        # (verified directly: a >10 px away cosmic ray inside that aperture
        # biased the normalisation). cut=11 -> norm_ap radius=5 px.
        cut = 11 if instrument == 'nircam' else 20

    cube = np.asarray(cube, dtype=float)
    n_int, n_groups, ny, nx = cube.shape
    # dropping the last gradient (last-frame anomaly) is a MIRI-derived
    # convention; set drop_last_gradient=False to use all n_groups-1
    # gradients instead (e.g. for NIRCam, where this hasn't been
    # established as necessary)
    n_grads = n_groups - 2 if drop_last_gradient else n_groups - 1
    g_arr = np.arange(n_grads, dtype=float)

    if instrument == 'nircam':
        if bfe_early_groups is None:
            bfe_early_groups = [0]
        if bfe_late_groups is None:
            bfe_late_groups = [n_grads - 1]

    grads = np.diff(cube, axis=1)[:, :n_grads]
    med_grad = np.median(grads, axis=0)

    # Detect brightest round source (excludes elongated edge artifacts).
    # Prefer a clean external master_ref over the cube's own median gradient
    # (see master_ref docstring: a cosmic-ray/snowball hit can survive
    # median rejection and get mistaken for a star).
    if master_ref is not None:
        detect_img = np.asarray(master_ref, dtype=np.float64)
    else:
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
    if len(round_sources) > 0:
        brightest = round_sources[np.argsort(round_sources['flux'])[-1]]
    # A fixed absolute-DN threshold only makes sense for the original
    # MIRI-calibrated case (detection on the cube's own gradient image, in
    # gradient DN/group). master_ref units and NIRCam's gradient DN/group
    # scale both differ from that calibration, so use an SNR cut instead.
    use_snr_cut = (master_ref is not None) or (instrument == 'nircam')
    if use_snr_cut:
        meets_threshold = (len(round_sources) > 0 and brightest['npix'] > 0 and
                          brightest['flux'] / (bkg.globalrms * np.sqrt(brightest['npix'])) > snr_min)
    else:
        _flux_min = flux_min if flux_min is not None else 50000.0
        meets_threshold = len(round_sources) > 0 and brightest['flux'] >= _flux_min
    if not meets_threshold:
        if verbose:
            print('  No source meets brightness threshold — skipping BFE fit')
        return None, (alpha_bfe if alpha_bfe else 3.43), 0.0, 0.0, nx // 2, ny // 2
    # Reject crowded/blended candidates -- see _isolated_brightest docstring.
    star = _isolated_brightest(round_sources, objects, cut, verbose=verbose)
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

    if fit_rcd_decay:
        # RCD (reset charge decay) is a physical effect specific to Si:As
        # IBC detectors (e.g. MIRI); it has no counterpart in NIRCam's
        # HgCdTe detectors. Fitting a decay timescale to background
        # gradients that have no real decay in them is both meaningless
        # and numerically fragile (needs >=3 usable gradient points --
        # fails outright on short NIRCam ramps like NGROUPS=5). Set
        # fit_rcd_decay=False to skip this model entirely: Adec/delta are
        # fixed at 0 (no decay, no last-frame-anomaly term) and rate_map
        # is just the median gradient, so the BFE forward model
        # downstream (_build_true) reduces to a flat rate with no decay
        # component, which is the physically correct assumption for an
        # instrument with no RCD.
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
    else:
        tau = 1.0  # unused: Adec_map is all zero, so the exp(-g/tau) term vanishes
        if verbose:
            print('  fit_rcd_decay=False -- skipping RCD decay model (no decay assumed)')
        rate_map = np.median(med_grad, axis=0)
        Adec_map = np.zeros((ny, nx))
        delta_map = np.zeros((ny, nx))

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

    # rate_c/Adec_c/delta_c feed the forward model directly -- a bad pixel
    # anywhere in this crop (not just within fit_r) biases the model's own
    # inputs, not just the final chi2 region masked below. Inpaint over
    # sci_mask-flagged and DQ-flagged (any group used) pixels here.
    _crop_bad = np.zeros((nyc, nxc), dtype=bool)
    if sci_mask is not None:
        _crop_bad |= ~sci_mask[y0:y1, x0:x1].astype(bool)
    _crop_dq_bad = _dq_exclusion_mask(dq, y0, y1, x0, x1, list(range(n_grads + 1)),
                                      protect_center=(cy, cx))
    if _crop_dq_bad is not None:
        _crop_bad |= _crop_dq_bad
    # Other real, unflagged sources within this (wide) crop also don't
    # belong in the target's own rate/Adec/delta maps -- see
    # _other_source_mask docstring (verified directly: a genuine second
    # star leaked into the model, producing an unrelated second blob).
    _crop_bad |= _other_source_mask(rate_c, cy, cx,
                                    sci_mask=(sci_mask[y0:y1, x0:x1] if sci_mask is not None else None))
    if np.any(_crop_bad):
        if verbose:
            print(f'  Inpainting {int(_crop_bad.sum())} bad pixel(s) in the model input maps')
        rate_c = _inpaint_bad_pixels(rate_c, _crop_bad)
        Adec_c = _inpaint_bad_pixels(Adec_c, _crop_bad)
        delta_c = _inpaint_bad_pixels(delta_c, _crop_bad)

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

    # Per-group bad-pixel stamp: the background median and (critically) the
    # total-flux normalisation sum below are shared, single scalar values
    # that every pixel's normalised difference is divided by -- a cosmic
    # ray or bad pixel landing inside _norm_ap biases that ONE number for
    # every pixel in the image, not just its own. Verified directly: one
    # DQ-flagged cluster inside _norm_ap inflated a group's flux sum by
    # >10%, silently rescaling the entire normalised difference. Excluding
    # bad pixels from these sums (their own raw value elsewhere is still
    # left alone, and separately excluded from the fit via fit_mask).
    _sci_bad_stamp = (~_cut2d(sci_mask[y0:y1, x0:x1].astype(bool))
                      if sci_mask is not None else np.zeros((2*cut+1, 2*cut+1), dtype=bool))
    _bad_by_group = []
    for g in range(n_grads):
        _dqg = _dq_exclusion_mask(dq, sy - cut, sy + cut + 1, sx - cut, sx + cut + 1,
                                  _grad_to_group_indices([g]), protect_center=(cut, cut))
        _bad_by_group.append(_sci_bad_stamp if _dqg is None else (_sci_bad_stamp | _dqg))

    def _clean_bg_and_flux(c, bad):
        good_bg = _bg_ann & ~bad
        bgm = np.median(c[good_bg]) if np.any(good_bg) else np.median(c[_bg_ann])
        good_ap = _norm_ap & ~bad
        flux = float((c - bgm)[good_ap].sum()) if np.any(good_ap) else float((c - bgm)[_norm_ap].sum())
        return bgm, flux

    def _starflux(img, g):
        c = _cut2d(img)
        _, flux = _clean_bg_and_flux(c, _bad_by_group[g])
        return flux
    _fobs = np.array([_starflux(med_obs_c[g], g) for g in range(n_grads)])

    def _norm_diff(stack):
        def _acc(glist):
            s = 0.0
            for g in glist:
                c = _cut2d(stack[g])
                bgm, _ = _clean_bg_and_flux(c, _bad_by_group[g])
                s = s + (c - bgm) / _fobs[g]
            return s / len(glist)
        return _acc(bfe_late_groups) - _acc(bfe_early_groups)

    from scipy.optimize import minimize as _minimize
    _opt = dict(method='Powell',
                options={'xtol': 1e-5, 'ftol': 1e-9, 'maxiter': 20000})

    # fit radius from the initial (raw) maps; held fixed across iterations
    true_grads, Q_grads = _build_true(rate_c, Adec_c, delta_c)
    obs_diff = _norm_diff(med_obs_c)
    if n_int > 1:
        noise_diff = np.std([_norm_diff(grads_c[i]) for i in range(n_int)], axis=0) / np.sqrt(n_int)
        noise_diff = np.clip(noise_diff, noise_diff[noise_diff > 0].min() * 0.1, None)
    else:
        # single-integration exposure (e.g. NIRCam imaging, NINTS=1) --
        # can't estimate noise from inter-integration scatter (nothing to
        # compare against). Fall back to the SPATIAL std of the
        # background-annulus pixels in obs_diff itself, as a uniform
        # noise floor -- less precise than a per-pixel/inter-integration
        # estimate, but well-defined for n_int=1.
        bg_std = float(np.std(obs_diff[_bg_ann]))
        if verbose:
            print(f'  n_int=1 -- using spatial background std as noise floor: {bg_std:.4g}')
        noise_diff = np.full_like(obs_diff, max(bg_std, 1e-12))
    if fit_r is None:
        _fit_r_floor = (max(4, round(_FIT_R_FWHM_RATIO_KERNEL * psf_fwhm_px))
                       if psf_fwhm_px else 5)
        snr_profile = np.array([
            np.mean(np.abs(obs_diff[np.round(r_map_c).astype(int) == ri])) /
            np.mean(noise_diff[np.round(r_map_c).astype(int) == ri])
            for ri in range(1, cut)])
        above = np.where(snr_profile > 2.0)[0]
        fit_r = max(_fit_r_floor, int(above[-1]) + 1) if len(above) > 0 else _fit_r_floor
        if verbose:
            print(f'  Auto fit_r = {fit_r} px (SNR-based, floor={_fit_r_floor})')
    fit_mask = r_map_c <= fit_r
    # Exclude pixels the pipeline's own jump step already flagged (cosmic
    # rays) or that saturated, in any group feeding the early/late
    # difference -- catches contaminants (including small clusters) that
    # the purely-statistical screen below can miss.
    _dq_bad = _dq_exclusion_mask(dq, sy - cut, sy + cut + 1, sx - cut, sx + cut + 1,
                                 _grad_to_group_indices(list(bfe_early_groups) + list(bfe_late_groups)),
                                 protect_center=(cut, cut))
    if _dq_bad is not None:
        _n_dq = int((fit_mask & _dq_bad).sum())
        fit_mask = fit_mask & ~_dq_bad
        if _n_dq > 0 and verbose:
            print(f'  Excluded {_n_dq} DQ-flagged (jump/saturated) pixel(s) from the fit region')
    # Reject single-pixel outliers (cosmic rays, hot pixels) against the
    # median radial profile of the observed late-early difference itself --
    # a per-pixel chi2 over a small aperture is vulnerable to one bad pixel
    # dominating the sum of squares regardless of fit_r (seen directly on
    # real NIRCam data: a single cosmic-ray pixel corrupted an otherwise
    # clean fit). The model is azimuthally symmetric by construction, so
    # comparing each pixel to its own radial-bin median (not mean, which the
    # outlier itself would drag) is a robust, model-independent screen —
    # a backstop for contamination the DQ flags above don't cover.
    _n_out = _mask_radial_outliers(obs_diff, r_map_c, fit_mask)
    if _n_out > 0 and verbose:
        print(f'  Rejected {_n_out} radial-outlier pixel(s) from the fit region')

    def _fit_bfe_once(true_grads, Q_grads):
        # Fit kernel shape (alpha, b, c) nonlinearly; A and a constant
        # background are linear (solved by weighted least squares).
        obs = _norm_diff(med_obs_c)
        if n_int > 1:
            noise = np.std([_norm_diff(grads_c[i]) for i in range(n_int)], axis=0) / np.sqrt(n_int)
            noise = np.clip(noise, noise[noise > 0].min() * 0.1, None)
        else:
            # single-integration exposure -- see the matching fallback
            # above (noise_diff) for why inter-integration std can't be
            # used here
            bg_std = float(np.std(obs[_bg_ann]))
            noise = np.full_like(obs, max(bg_std, 1e-12))
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
        if not fit_rcd_decay:
            # no RCD model to refit (Adec_c/delta_c are pinned at 0) --
            # rate_c/Adec_c/delta_c stay fixed, so true_grads/Q_grads are
            # identical next iteration and A_bfe_fit converges trivially
            continue
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
        # Median, not mean -- a single cosmic-ray/hot pixel in a radial bin
        # otherwise dominated by clean background pixels can drag the mean
        # far from the true value (verified directly: one contaminated
        # pixel shifted a bin's mean by >100x its median on real data).
        rp_obs = np.array([np.nanmedian(obs_diff[rr == ri]) for ri in r_int])
        rp_sim = np.array([np.nanmedian(sim_diff[rr == ri]) for ri in r_int])

        # Colour scale: exclude DQ-flagged (cosmic ray/saturated) pixels
        # from ANY group, so a contaminant visible elsewhere in the raw
        # panels doesn't stretch the range and wash out the real core
        # signal -- but keep the core itself protected/included (see
        # _dq_exclusion_mask's protect_center: DQ flags on the source's
        # own peak are a known jump-detection false-positive mode, not
        # necessarily real contamination).
        _display_bad = _dq_exclusion_mask(dq, sy - cut, sy + cut + 1, sx - cut, sx + cut + 1,
                                          list(range(n_grads + 1)), protect_center=(cut, cut))
        _vabs_src = obs_diff if _display_bad is None else np.where(_display_bad, np.nan, obs_diff)
        vabs = np.nanpercentile(np.abs(_vabs_src), 99.5)
        ext = [-cut - 0.5, cut + 0.5, -cut - 0.5, cut + 0.5]

        import math
        _exp = int(math.floor(math.log10(vabs))) if vabs > 0 else 0
        _scale = 10 ** _exp
        _exp_str = f'$\\times10^{{{_exp}}}$' if _exp != 0 else ''

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, img, title in [
            (axes[0, 0], obs_diff, 'Observed late$-$early'),
            (axes[0, 1], sim_diff,
             f'Model (A={A_bfe_fit:.2e}, α={alpha_fit:.2f}, b={b_fit:.2f}, c={c_fit:.3f})'),
            (axes[1, 0], res_diff, 'Residual (obs$-$model)'),
        ]:
            im = ax.imshow(img / _scale, origin='lower', cmap='RdBu_r',
                           vmin=-vabs / _scale, vmax=vabs / _scale, extent=ext)
            fig.colorbar(im, ax=ax, label=rf'Norm. $\Delta$flux {_exp_str}')
            ax.set_title(title)
            ax.set_xlabel(r'$\Delta x$ (px)')
            ax.set_ylabel(r'$\Delta y$ (px)')

        ax = axes[1, 1]
        ax.plot(r_int, rp_obs / _scale, 'k-', lw=1.5, label='Observed')
        ax.plot(r_int, rp_sim / _scale, color='C3', ls='--', lw=1.5, label='Model')
        ax.axhline(0, color='k', lw=0.5, ls=':')
        ax.axvline(fit_r, color='C0', lw=1.0, ls='--', label=f'fit_r = {fit_r} px')
        ax.set_xlabel('Radius (px)')
        ax.set_ylabel(rf'Mean $\Delta$flux {_exp_str}')
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


# ===========================================================================
# Charge-migration BFE model (the default correction method).
#
# Physical picture: accumulated charge migrates out of high-charge pixels,
# flowing down its own gradient with a charge-weighted mobility (bright cores
# diffuse faster), as a conserved flux. A force threshold suppresses migration
# below a gradient barrier. The per-pixel update is a charge-weighted discrete
# Laplacian of the image; total charge is conserved exactly.
# ===========================================================================
def _mig_step(Q, kx, ky, thr):
    """One conserved migration tick along cardinal axes only (no diagonal).

    Charge flows down-gradient between edge-sharing neighbours with mobility
    proportional to the face-averaged charge. Both axes are computed from the
    same Q so the update is simultaneous and exactly cardinal (diagonal pixels
    receive zero flux by construction).

    The mobility term D is floored at zero per-pixel before averaging
    (D = 0.5*(max(Q_i,0) + max(Q_j,0))), not 0.5*(Q_i+Q_j). Mobility should
    track local charge *density*, which is never negative; on long ramps Q
    can go locally negative (a charge deficit built up by repeated
    migration), and without the floor that negative Q flips the flux term's
    sign, turning the diffusive update into anti-diffusion -- an
    unconditionally unstable amplification of the very gradient it's meant
    to smooth, regardless of step size (this is why taking smaller substeps
    made the old runaway happen sooner, not later: finer stepping just
    tracked the unstable continuous dynamics more faithfully). Flooring
    D removes the sign flip while leaving the conserved flux-difference
    structure (and therefore exact charge conservation) untouched, and is a
    near no-op in the short-ramp regime the model was originally calibrated
    on."""
    out = Q.copy()
    for axis, k in ((0, ky), (1, kx)):
        if k <= 0:
            continue
        Ql = np.swapaxes(Q, 0, axis)
        dQ = Ql[:-1] - Ql[1:]
        D = 0.5 * (np.maximum(Ql[:-1], 0.0) + np.maximum(Ql[1:], 0.0))
        if thr > 0:
            dQ = np.sign(dQ) * np.maximum(np.abs(dQ) - thr, 0.0)
        f = k * D * dQ
        o = np.swapaxes(out, 0, axis)
        o[:-1] -= f
        o[1:] += f
    return out


def _mig_group(Q, Mx, My, thr):
    """Apply anisotropic cardinal migration (Mx in x, My in y) over one group.

    Sub-stepped for CFL stability using the larger of Mx, My."""
    qmax = Q.max()
    if qmax <= 0 or (Mx <= 0 and My <= 0):
        return Q
    n_sub = max(1, int(np.ceil(max(Mx, My) * qmax / 0.2)))
    kx = Mx / n_sub
    ky = My / n_sub
    for _ in range(min(n_sub, 600)):
        Q = _mig_step(Q, kx, ky, thr)
    return Q


def _mig_forward(true_grads, Mx, My, thr, q_bound=None):
    """Forward model: accumulate true gradients group by group, migrating the
    accumulated charge between readouts; return the observed gradients (readout
    differences).

    On very long ramps (300+ groups) the per-group migration update can enter
    a runaway feedback loop: the flux term is quadratic-ish in the local
    charge (f ~ k * D * dQ with D the local charge average), so once a bright
    pixel's accumulated Q exceeds the regime the model was calibrated on, each
    group's migration overshoots further than the last, doubling every group
    or two until Q overflows to inf/NaN. This isn't a step-size (CFL) issue —
    forcing more substeps makes it worse, not better, since it just applies
    the same unstable map more times per group. Q is physically bounded (it's
    accumulated charge from true_grads, which can never exceed the sum of the
    single largest per-group gradient repeated every group), so clip to that
    bound each group as a safety valve against the numerical runaway without
    touching the well-behaved regime the physical model targets.

    q_bound : float, optional
        Fixed clip bound. _mig_invert's Born iteration calls this repeatedly
        with an evolving true_grads, so it passes a bound derived once from
        the original med_obs — deriving it fresh from true_grads each call
        would let a diverged previous iteration's magnitude set (and inflate)
        the next iteration's bound. Defaults to a bound derived from this
        call's own true_grads, for standalone use.
    """
    n_g = true_grads.shape[0]
    if q_bound is None:
        q_bound = float(np.nanmax(np.abs(true_grads))) * n_g
    Q = np.zeros(true_grads.shape[1:])
    Qprev = Q.copy()
    out = np.zeros_like(true_grads)
    clipped = False
    for g in range(n_g):
        Q = _mig_group(Q + true_grads[g], Mx, My, thr)
        if not np.isfinite(Q).all() or np.abs(Q).max() > q_bound:
            clipped = True
            Q = np.nan_to_num(Q, nan=0.0, posinf=q_bound, neginf=-q_bound)
            Q = np.clip(Q, -q_bound, q_bound)
        out[g] = Q - Qprev
        Qprev = Q.copy()
    if clipped:
        warnings.warn(
            'BFE migration model diverged during forward integration and was '
            'clipped to a physically-motivated bound — likely a very long ramp '
            'pushing local charge beyond the regime the migration model is '
            'calibrated for. Treat the BFE correction on affected pixels with '
            'caution.', RuntimeWarning)
    return out


def _mig_invert(med_obs, Mx, My, thr, n_born=4):
    """Recover the migration-free (true) gradients from observed median
    gradients by a Born-style fixed-point iteration on the full ramp."""
    # Fixed across all Born iterations and derived from the original
    # observed data, not the evolving true_grads estimate — otherwise a
    # diverged iteration's inflated magnitude would set (and re-inflate)
    # the clip bound for the next iteration in _mig_forward.
    q_bound = float(np.nanmax(np.abs(med_obs))) * med_obs.shape[0]
    true_grads = med_obs.copy()
    for _ in range(n_born):
        pred = _mig_forward(true_grads, Mx, My, thr, q_bound=q_bound)
        true_grads = true_grads + (med_obs - pred)
        true_grads = np.clip(np.nan_to_num(true_grads, nan=0.0, posinf=q_bound, neginf=-q_bound),
                              -q_bound, q_bound)
    return true_grads


def _poly_deriv(Q, a_lin):
    """Local derivative of the charge-to-DN response, d(DN)/d(true charge),
    for a simple quadratic 'vertical' (per-pixel, non-redistributive)
    responsivity-loss model: DN(Q) = Q + a_lin*Q^2, so poly'(Q) = 1 + 2*a_lin*Q.

    a_lin > 0 means responsivity falls as accumulated charge grows (fewer DN
    recorded per unit of true incoming charge), independent of and in
    addition to lateral BFE migration. Floored at 0.5 to avoid the forward
    model blowing up for pathological parameter values during optimisation.
    """
    return np.maximum(1.0 + 2.0 * a_lin * Q, 0.5)


def _cubic_deriv(Q, a2, a3):
    """Generalisation of _poly_deriv to a cubic vertical-loss model:
    DN(Q) = Q + a2*Q^2 + a3*Q^3, so d(DN)/dQ = 1 + 2*a2*Q + 3*a3*Q^2.

    A quadratic term alone (_poly_deriv) has constant curvature -- the same
    d(pd)/dQ everywhere. Cross-target testing (see bdplus60/CONTEXT.md,
    dxcancri/CONTEXT.md) found the true nonlinearity's curvature itself
    grows with Q (near-zero at ~15,000 DN, much steeper at ~30,000+ DN
    sustained), which a single quadratic coefficient cannot represent no
    matter how it's tuned. The a3 term lets the curvature increase with Q.
    a3=0 recovers _poly_deriv exactly.
    """
    return np.maximum(1.0 + 2.0 * a2 * Q + 3.0 * a3 * Q ** 2, 0.5)


def _persist_deriv(Q, S, a2, a3, b):
    """Generalisation of _cubic_deriv adding a slow, cumulative 'persistence'
    (trap-filling) term: pd(Q, S) = 1 + 2*a2*Q + 3*a3*Q^2 + b*S, where S is
    the running charge-time integral (sum of Q over all prior groups in the
    ramp), not just the instantaneous charge Q.

    Motivation: cross-target testing found BD+60 1753 (peak/sustained
    charge ~30,000 DN held for 100+ groups) needs several times more
    vertical-loss correction than Wolf-359/EV Lac (similar ~31,000 DN peak,
    but reached in only ~9-11 groups) -- i.e. targets with matched peak
    charge but very different TIME spent at that charge disagree, which a
    purely charge-dependent term (however high-order) cannot explain. This
    is the signature of a persistence/trap-filling effect (well documented
    in Si:As IBC detectors, the technology MIRI uses) operating on a slow
    timescale (much longer than RSCD's ~1-2 groups) in addition to, not
    instead of, the instantaneous Q-dependent nonlinearity. b=0 recovers
    _cubic_deriv exactly.
    """
    return np.maximum(1.0 + 2.0 * a2 * Q + 3.0 * a3 * Q ** 2 + b * S, 0.5)


def fit_migration_params(cube, M_init=4.2e-7, thr_init=37.2, bg_mask=None,
                         sci_mask=None, bfe_early_groups=None, bfe_late_groups=None,
                         ap_radius=5, cut=None, fit_r=None, verbose=False,
                         max_iter=6, M_tol=0.02, aniso=False,
                         fix_M=None, fix_thr=None,
                         fit_a_lin=False, a_lin_init=2.5e-6, a_lin_fixed=None,
                         a3_fixed=None, b_fixed=None,
                         flux_min=50000.0, snr_min=10.0,
                         return_chi2=False, return_model=False,
                         instrument='miri', fit_rcd_decay=None, drop_last_gradient=None,
                         master_ref=None, nframes=1, groupgap=0, dq=None,
                         psf_fwhm_px=None,
                         diagnostics=False, save_path=None):
    """Fit the charge-migration BFE parameters from the brightest source.

    psf_fwhm_px : float, optional
        This exposure's PSF FWHM in pixels. When given and fit_r is not
        set explicitly, the auto fit_r floor is re-derived from this
        exposure's own PSF size instead of the fixed MIRI-F2100W-
        calibrated floor. See fit_bfe_params' psf_fwhm_px docstring.
    dq : ndarray (n_int, n_groups, ny, nx) or (n_groups, ny, nx), optional
        GROUPDQ array from the exposure; pixels flagged JUMP_DET or
        SATURATED in any group used for the early/late difference are
        excluded from the fit region. See fit_bfe_params' dq docstring.

    nframes, groupgap : int, default 1, 0
        On-board readout structure of each stored group: `nframes`
        individual frame reads are averaged together, followed by
        `groupgap` dropped (unsaved, but still-accumulating) frames before
        the next group. The migration model sub-steps per individual frame
        and averages the kept ones, matching what the detector itself does
        -- a single step per stored group (the nframes=1, groupgap=0
        default) is only correct for readout patterns with no on-board
        averaging (e.g. MIRI's FASTR1). For NIRCam, read these off the
        exposure's own header: NFRAMES, GROUPGAP (e.g. MEDIUM8 is
        nframes=8, groupgap=2 -- verify per exposure, patterns vary:
        BRIGHT1/DEEP2/etc. all differ). There is no safe universal NIRCam
        default, so this must be passed explicitly for NIRCam data;
        leaving it at 1, 0 silently reproduces the (wrong, MIRI-shaped)
        atomic-step behaviour.
    master_ref : ndarray (ny, nx), optional
        A clean, deep, cosmic-ray-free detection image on the same pixel
        grid as ``cube``. When given, source detection runs on this image
        instead of the cube's own median gradient. See fit_bfe_params'
        master_ref docstring for why this matters.
    flux_min : float, default 50000.0
        Minimum SEP-measured flux for the brightest source, used only for
        the original MIRI-calibrated case (detection on the cube's own
        gradient image with instrument='miri'). Ignored — an SNR cut is
        used instead (snr_min) — when master_ref is given or
        instrument='nircam', since gradient DN/group and master_ref units
        both differ per instrument/detector/exposure and aren't portable
        against a fixed absolute threshold.
    snr_min : float, default 10.0
        Minimum source flux / (background rms * sqrt(npix)) required to
        accept the brightest detected source, used whenever flux_min isn't
        (see above).

    When aniso=False (default) fits a single isotropic migration strength M.
    When aniso=True fits separate Mx (x-axis) and My (y-axis) strengths.
    Uses an iterative RCD <-> migration fit so decay cannot absorb the BFE.

    fit_a_lin : bool, default False
        If True, jointly fit an additional 'vertical' (per-pixel, non-
        redistributive) charge-dependent responsivity-loss term alongside
        the lateral migration parameters (isotropic case only). See
        ``_poly_deriv``. Intended for use on raw (uncal) data where no
        standard-pipeline linearity correction has been applied, so that
        migration (lateral) and responsivity loss (vertical) are fit
        self-consistently from the same high-contrast source rather than
        the vertical component being taken from a flat-field-calibrated,
        spatially-uniform reference curve.
    a_lin_init : float, default 2.5e-6
        Initial value for the vertical-loss coefficient when fit_a_lin=True.
    a_lin_fixed : float, optional
        If given (and fit_a_lin=False), hold the vertical-loss coefficient
        fixed at this value (e.g. derived from the CRDS linearity reference
        file's quadratic term) while still fitting M/thr against it. Useful
        when a free 3-parameter fit is poorly constrained (faint/low-contrast
        sources) but an external, better-calibrated estimate of the vertical
        term exists.
    a3_fixed : float, optional
        If given, adds a cubic term to the vertical-loss model (see
        ``_cubic_deriv``): DN(Q) = Q + a_lin*Q^2 + a3_fixed*Q^3. Held fixed
        (not fit), analogous to a_lin_fixed. Use together with a_lin_fixed
        for a shared, cross-target-calibrated nonlinearity curve rather
        than a single quadratic coefficient.
    b_fixed : float, optional
        If given, adds a slow 'persistence' (charge-time history) term (see
        ``_persist_deriv``): pd(Q,S) = 1 + 2*a_lin*Q + 3*a3*Q^2 + b_fixed*S,
        where S is the running sum of Q over all prior groups. Held fixed,
        analogous to a_lin_fixed/a3_fixed.
    flux_min : float, default 50000.0
        Minimum sep-measured flux for the brightest detected source to be
        used for the fit; fields with no source above this are skipped.
    return_chi2 : bool, default False
        If True, also return the final chi2/n (reduced chi-square) of the
        fit as an extra value, for comparing/calibrating parameters across
        targets.
    instrument : {'miri', 'nircam'}, default 'miri'
        Case-insensitive. Selects instrument-appropriate defaults for
        fit_rcd_decay and drop_last_gradient (see below), and for
        bfe_early_groups/bfe_late_groups when neither is given explicitly.
        MIRI's Si:As detectors show reset charge decay (RCD) and a
        last-frame anomaly; NIRCam's HgCdTe detectors show neither.
    fit_rcd_decay : bool, optional
        If True, fit a parametric C + A*exp(-g/tau) reset-decay model on
        background pixels before the migration fit (MIRI only — NIRCam has
        no RCD). Defaults to True for instrument='miri', False for
        instrument='nircam'.
    drop_last_gradient : bool, optional
        If True, exclude the final group-to-group gradient (MIRI's
        last-frame anomaly) from the fit. Defaults to True for
        instrument='miri', False for instrument='nircam'.

    Returns (Mx, My, thr, sx, sy, a_lin) or, if return_chi2=True,
    (Mx, My, thr, sx, sy, a_lin, chi2n). Mx is None if no source meets
    threshold. For the isotropic case Mx == My. a_lin is 0.0 unless
    fit_a_lin=True.
    """
    import sep
    from scipy.optimize import minimize as _minimize

    instrument = instrument.lower() if instrument else instrument
    if fit_rcd_decay is None:
        fit_rcd_decay = (instrument == 'miri')
    if drop_last_gradient is None:
        drop_last_gradient = (instrument == 'miri')
    if cut is None:
        # See fit_bfe_params' cut docstring for why: MIRI's default
        # (cut=20 -> norm_ap radius 14 px) is too large for NIRCam's more
        # compact PSF.
        cut = 11 if instrument == 'nircam' else 20

    cube = np.asarray(cube, dtype=float)
    n_int, n_groups, ny, nx = cube.shape
    n_grads = n_groups - 2 if drop_last_gradient else n_groups - 1
    g_arr = np.arange(n_grads, dtype=float)
    grads = np.diff(cube, axis=1)[:, :n_grads]
    med = np.median(grads, axis=0)

    if master_ref is not None:
        detect = np.asarray(master_ref, dtype=np.float64)
    else:
        detect = np.median(grads[:, 1:n_grads], axis=(0, 1)).astype(np.float64)
    sep_mask = (~sci_mask.astype(bool)) if sci_mask is not None else None
    bkg = sep.Background(detect, mask=sep_mask)
    all_obj = sep.extract((detect - bkg.back()).astype(np.float64), 5.0,
                          err=bkg.globalrms, mask=sep_mask)
    edge = 25
    obj = all_obj[(all_obj['x'] > edge) & (all_obj['x'] < nx-edge) & (all_obj['y'] > edge) &
                  (all_obj['y'] < ny-edge) & (all_obj['a']/all_obj['b'] < 3)]
    if len(obj) > 0:
        brightest = obj[np.argmax(obj['flux'])]
    # flux_min (fixed absolute DN) is only meaningful for the original
    # MIRI-calibrated case; master_ref units and NIRCam's gradient DN/group
    # scale both differ from that calibration, so use an SNR cut instead.
    use_snr_cut = (master_ref is not None) or (instrument == 'nircam')
    if use_snr_cut:
        meets_threshold = (len(obj) > 0 and brightest['npix'] > 0 and
                          brightest['flux'] / (bkg.globalrms * np.sqrt(brightest['npix'])) > snr_min)
    else:
        meets_threshold = len(obj) > 0 and brightest['flux'] >= flux_min
    if not meets_threshold:
        if verbose:
            print('  No source meets brightness threshold — skipping migration fit')
        if return_chi2:
            return None, None, thr_init, nx//2, ny//2, 0.0, None
        return None, None, thr_init, nx//2, ny//2, 0.0
    # Reject crowded/blended candidates -- see _isolated_brightest docstring.
    star = _isolated_brightest(obj, all_obj, cut, verbose=verbose)
    sy, sx = int(round(star['y'])), int(round(star['x']))

    yy, xx = np.mgrid[:ny, :nx]
    rs = np.sqrt((yy-sy)**2 + (xx-sx)**2)
    _bg = bg_mask.astype(bool) if bg_mask is not None else (rs > 15) & (rs < min(ny, nx)//3)
    if sci_mask is not None and bg_mask is None:
        _bg &= sci_mask.astype(bool)
    if fit_rcd_decay:
        mb = np.nanmean(med[1:][:, _bg], axis=1)
        popt, _ = curve_fit(lambda g, C, A, t: C+A*np.exp(-g/t), g_arr[1:], mb,
                            p0=[mb[-1], mb[0]-mb[-1], 1.5])
        tau = float(popt[2])
        exp_g = np.exp(-g_arr/tau)
        ff = np.zeros(n_grads); ff[0] = -1.0
        Xb = np.column_stack([np.ones(n_grads), exp_g, ff])
        p, _, _, _ = np.linalg.lstsq(Xb, med.reshape(n_grads, -1), rcond=None)
        rate, Adec, delta = (p[i].reshape(ny, nx) for i in range(3))
    else:
        tau = 1.0
        exp_g = np.exp(-g_arr/tau)
        Xb = np.column_stack([np.ones(n_grads), exp_g, np.zeros(n_grads)])
        rate = np.median(med, axis=0)
        Adec = np.zeros((ny, nx))
        delta = np.zeros((ny, nx))

    crop = cut + 50
    y0, y1 = max(0, sy-crop), min(ny, sy+crop+1)
    x0, x1 = max(0, sx-crop), min(nx, sx+crop+1)
    cy, cx = sy-y0, sx-x0
    rate_c, Adec_c, delta_c = (m[y0:y1, x0:x1] for m in (rate, Adec, delta))
    gc = grads[:, :, y0:y1, x0:x1]
    med_obs_c = np.median(gc, axis=0)
    nyc, nxc = rate_c.shape

    # rate_c/Adec_c/delta_c feed the forward model directly -- see the
    # equivalent inpainting step in fit_bfe_params for why this is needed
    # (a bad pixel anywhere in the crop biases the model's own inputs, not
    # just the chi2 region masked by fitmask below).
    _crop_bad = np.zeros((nyc, nxc), dtype=bool)
    if sci_mask is not None:
        _crop_bad |= ~sci_mask[y0:y1, x0:x1].astype(bool)
    _crop_dq_bad = _dq_exclusion_mask(dq, y0, y1, x0, x1, list(range(n_grads + 1)),
                                      protect_center=(cy, cx))
    if _crop_dq_bad is not None:
        _crop_bad |= _crop_dq_bad
    # Other real, unflagged sources within this (wide) crop also don't
    # belong in the target's own rate/Adec/delta maps -- see
    # _other_source_mask docstring (verified directly: a genuine second
    # star leaked into the model, producing an unrelated second blob).
    _crop_bad |= _other_source_mask(rate_c, cy, cx,
                                    sci_mask=(sci_mask[y0:y1, x0:x1] if sci_mask is not None else None))
    if np.any(_crop_bad):
        if verbose:
            print(f'  Inpainting {int(_crop_bad.sum())} bad pixel(s) in the model input maps')
        rate_c = _inpaint_bad_pixels(rate_c, _crop_bad)
        Adec_c = _inpaint_bad_pixels(Adec_c, _crop_bad)
        delta_c = _inpaint_bad_pixels(delta_c, _crop_bad)

    if instrument == 'nircam':
        if bfe_early_groups is None:
            bfe_early_groups = [0]
        if bfe_late_groups is None:
            bfe_late_groups = [n_grads - 1]
    if bfe_early_groups is None:
        n_e = max(2, min(3, n_grads // 4)); start = 1 if n_grads < 8 else 2
        bfe_early_groups = list(range(start, start + n_e))
    if bfe_late_groups is None:
        n_e = max(2, min(3, n_grads // 4))
        bfe_late_groups = list(range(n_grads - n_e, n_grads))

    yy_c, xx_c = np.mgrid[:2*cut+1, :2*cut+1]
    rmap = np.sqrt((yy_c-cut)**2 + (xx_c-cut)**2)
    norm_ap = rmap <= (cut-6)
    bg_ann = (rmap > (cut-6)) & (rmap <= (cut-1))

    def _cut(img):
        return img[cy-cut:cy+cut+1, cx-cut:cx+cut+1]

    # See fit_bfe_params' equivalent block for why: a bad pixel inside
    # norm_ap biases the shared per-group flux-normalisation sum for every
    # pixel, not just its own (verified directly: >10% bias from one
    # DQ-flagged cluster in the aperture).
    _sci_bad_stamp = (~_cut(sci_mask[y0:y1, x0:x1].astype(bool))
                      if sci_mask is not None else np.zeros((2*cut+1, 2*cut+1), dtype=bool))
    _bad_by_group = []
    for _g in range(n_grads):
        _dqg = _dq_exclusion_mask(dq, sy - cut, sy + cut + 1, sx - cut, sx + cut + 1,
                                  _grad_to_group_indices([_g]), protect_center=(cut, cut))
        _bad_by_group.append(_sci_bad_stamp if _dqg is None else (_sci_bad_stamp | _dqg))

    def norm_diff(stack):
        def acc(gl):
            s = 0.0
            for g in gl:
                c = _cut(stack[g])
                bad = _bad_by_group[g]
                good_bg = bg_ann & ~bad
                bgm = np.median(c[good_bg]) if np.any(good_bg) else np.median(c[bg_ann])
                c = c - bgm
                good_ap = norm_ap & ~bad
                flux = c[good_ap].sum() if np.any(good_ap) else c[norm_ap].sum()
                s = s + c / flux
            return s / len(gl)
        return acc(bfe_late_groups) - acc(bfe_early_groups)

    obs = norm_diff(med_obs_c)
    if n_int > 1:
        noise = np.std([norm_diff(gc[i]) for i in range(n_int)], axis=0) / np.sqrt(n_int)
        noise = np.clip(noise, np.nanpercentile(noise[noise > 0], 5), None)
    else:
        bg_std = float(np.std(obs[bg_ann]))
        noise = np.full_like(obs, max(bg_std, 1e-12))
    if fit_r is None:
        _fit_r_floor = (max(4, round(_FIT_R_FWHM_RATIO_MIGRATION * psf_fwhm_px))
                       if psf_fwhm_px else 8)
        snr = np.array([np.mean(np.abs(obs[np.round(rmap).astype(int) == ri])) /
                        np.mean(noise[np.round(rmap).astype(int) == ri])
                        for ri in range(1, cut)])
        above = np.where(snr > 2.0)[0]
        fit_r = max(_fit_r_floor, int(above[-1]) + 1) if len(above) > 0 else _fit_r_floor
        if verbose:
            print(f'  Auto fit_r = {fit_r} px (SNR-based, floor={_fit_r_floor})')
    fitmask = rmap <= fit_r
    _dq_bad = _dq_exclusion_mask(dq, sy - cut, sy + cut + 1, sx - cut, sx + cut + 1,
                                 _grad_to_group_indices(list(bfe_early_groups) + list(bfe_late_groups)),
                                 protect_center=(cut, cut))
    if _dq_bad is not None:
        _n_dq = int((fitmask & _dq_bad).sum())
        fitmask = fitmask & ~_dq_bad
        if _n_dq > 0 and verbose:
            print(f'  Excluded {_n_dq} DQ-flagged (jump/saturated) pixel(s) from the fit region')
    _n_out = _mask_radial_outliers(obs, rmap, fitmask)
    if _n_out > 0 and verbose:
        print(f'  Rejected {_n_out} radial-outlier pixel(s) from the fit region')
    w = 1.0 / noise

    def model_diff(Mx, My, thr, rate_c, Adec_c, delta_c, a_val=0.0, a3_val=0.0, b_val=0.0):
        # Frame-resolved: each stored group is the on-board average of
        # `nframes` individual frame reads, followed by `groupgap` dropped
        # (unsaved, but still-accumulating) frames -- see get_readtimes() in
        # stcal/ramp_fitting/likely_fit.py and group_scale.py's own
        # docstring ("on-board frame averaging"), verified directly against
        # this exposure's own header (NFRAMES/GROUPGAP/TGROUP all
        # consistent). Migration is a nonlinear, thresholded process, so it
        # does not commute with that averaging -- applying one migration
        # step per stored group (as if nframes=1, true for MIRI's FASTR1)
        # systematically under-predicts the redistribution for any readout
        # pattern with nframes>1 (e.g. NIRCam's MEDIUM8/BRIGHT2/DEEP8/...).
        # Sub-stepping per frame and averaging the kept sub-steps reproduces
        # what the detector itself actually does.
        tot_frames = nframes + groupgap
        dQ_frame_base = rate_c / tot_frames
        Q = np.zeros((nyc, nxc)); S = np.zeros((nyc, nxc))
        group_avg_prev = np.zeros((nyc, nxc))
        out = np.zeros((n_grads, nyc, nxc))
        for g in range(n_grads):
            frame_sum = np.zeros((nyc, nxc))
            for f in range(tot_frames):
                pd = _persist_deriv(Q, S, a_val, a3_val, b_val)
                dQ = dQ_frame_base / pd
                Q = _mig_group(Q + dQ, Mx, My, thr)
                S = S + Q
                if f < nframes:
                    frame_sum = frame_sum + Q
            group_avg = frame_sum / nframes
            photo = group_avg - group_avg_prev
            group_avg_prev = group_avg
            tg = photo + Adec_c * exp_g[g]
            if g == 0:
                tg = tg - delta_c
            out[g] = tg
        return norm_diff(out)

    logM0, thr0, log_a0 = np.log10(M_init), thr_init, np.log10(a_lin_init)
    Mx_fit = My_fit = M_init
    thr_fit = thr_init
    a_lin_fit = float(a_lin_fixed) if a_lin_fixed is not None else 0.0
    a3_fit = float(a3_fixed) if a3_fixed is not None else 0.0
    b_fit = float(b_fixed) if b_fixed is not None else 0.0
    M_prev = None

    # If M and thr are fixed and a_lin isn't being fit, skip the iterative
    # optimisation entirely. If fit_a_lin is requested, keep iterating so
    # a_lin can still be solved for with M/thr pinned (chi2() below pins
    # them via fix_M/fix_thr while log_a stays free).
    if fix_M is not None and fix_thr is not None and not fit_a_lin:
        Mx_fit = My_fit = float(fix_M)
        thr_fit = float(fix_thr)
        if verbose:
            print(f'  M and thr fixed: M={Mx_fit:.4e}  thr={thr_fit:.1f} DN')
        max_iter = 0

    for _it in range(max_iter):
        if aniso:
            if fix_M is not None:
                logMx0 = logMy0 = np.log10(float(fix_M))
            else:
                logMx0 = logMy0 = logM0
            _thr0 = float(fix_thr) if fix_thr is not None else thr0

            def chi2(par):
                logMx, logMy, thr = par
                if fix_M is not None:
                    logMx = logMy = np.log10(float(fix_M))
                if fix_thr is not None:
                    thr = float(fix_thr)
                if not (-9 <= logMx <= -2 and -9 <= logMy <= -2 and 0 <= thr <= 5000):
                    return 1e30
                d = model_diff(10**logMx, 10**logMy, thr, rate_c, Adec_c, delta_c, a_lin_fit, a3_fit, b_fit)
                if not np.all(np.isfinite(d)):
                    return 1e30
                return float(np.sum((((d - obs) * w)[fitmask])**2))
            x0 = [logMx0, logMy0, _thr0]
            res = _minimize(chi2, x0, method='Powell',
                            options={'xtol': 1e-4, 'ftol': 1e-7, 'maxiter': 2000})
            Mx_fit = float(fix_M) if fix_M is not None else 10**res.x[0]
            My_fit = float(fix_M) if fix_M is not None else 10**res.x[1]
            thr_fit = float(fix_thr) if fix_thr is not None else max(0.0, res.x[2])
            logM0 = (np.log10(Mx_fit) + np.log10(My_fit)) / 2
            thr0 = thr_fit
            M_now = (Mx_fit + My_fit) / 2
            if verbose:
                print(f'  [iter {_it}] Mx={Mx_fit:.4e}  My={My_fit:.4e}  '
                      f'thr={thr_fit:.1f} DN  chi2/n={res.fun/max(int(fitmask.sum())-3, 1):.3f}')
        elif fit_a_lin:
            _logM0 = np.log10(float(fix_M)) if fix_M is not None else logM0
            _thr0 = float(fix_thr) if fix_thr is not None else thr0

            def chi2(par):
                logM, thr, log_a = par
                if fix_M is not None:
                    logM = np.log10(float(fix_M))
                if fix_thr is not None:
                    thr = float(fix_thr)
                if not (-9 <= logM <= -2 and 0 <= thr <= 5000 and -9 <= log_a <= -3):
                    return 1e30
                d = model_diff(10**logM, 10**logM, thr, rate_c, Adec_c, delta_c, 10**log_a, a3_fit, b_fit)
                if not np.all(np.isfinite(d)):
                    return 1e30
                return float(np.sum((((d - obs) * w)[fitmask])**2))
            res = _minimize(chi2, [_logM0, _thr0, log_a0], method='Powell',
                            options={'xtol': 1e-4, 'ftol': 1e-7, 'maxiter': 2000})
            Mx_fit = My_fit = float(fix_M) if fix_M is not None else 10**res.x[0]
            thr_fit = float(fix_thr) if fix_thr is not None else max(0.0, res.x[1])
            a_lin_fit = 10**res.x[2]
            logM0, thr0, log_a0 = np.log10(Mx_fit), thr_fit, res.x[2]
            M_now = Mx_fit
            if verbose:
                print(f'  [iter {_it}] M={Mx_fit:.4e}  thr={thr_fit:.1f} DN  '
                      f'a_lin={a_lin_fit:.3e}  '
                      f'chi2/n={res.fun/max(int(fitmask.sum())-3, 1):.3f}')
        else:
            _logM0 = np.log10(float(fix_M)) if fix_M is not None else logM0
            _thr0 = float(fix_thr) if fix_thr is not None else thr0

            def chi2(par):
                logM, thr = par
                if fix_M is not None:
                    logM = np.log10(float(fix_M))
                if fix_thr is not None:
                    thr = float(fix_thr)
                if not (-9 <= logM <= -2 and 0 <= thr <= 5000):
                    return 1e30
                d = model_diff(10**logM, 10**logM, thr, rate_c, Adec_c, delta_c, a_lin_fit, a3_fit, b_fit)
                if not np.all(np.isfinite(d)):
                    return 1e30
                return float(np.sum((((d - obs) * w)[fitmask])**2))
            res = _minimize(chi2, [_logM0, _thr0], method='Powell',
                            options={'xtol': 1e-4, 'ftol': 1e-7, 'maxiter': 1500})
            Mx_fit = My_fit = float(fix_M) if fix_M is not None else 10**res.x[0]
            thr_fit = float(fix_thr) if fix_thr is not None else max(0.0, res.x[1])
            logM0 = np.log10(Mx_fit)
            thr0 = thr_fit
            M_now = Mx_fit
            if verbose:
                print(f'  [iter {_it}] M={Mx_fit:.4e}  thr={thr_fit:.1f} DN  '
                      f'chi2/n={res.fun/max(int(fitmask.sum())-2, 1):.3f}')

        if M_prev is not None and abs(M_now - M_prev) <= M_tol*abs(M_now):
            break
        M_prev = M_now
        if _it == max_iter - 1:
            break
        if not fit_rcd_decay:
            continue
        # remove migration (and vertical loss, if fitted), refit reset-decay
        # on the de-migrated gradient
        Q = np.zeros((nyc, nxc)); Qprev = Q.copy(); S = np.zeros((nyc, nxc))
        photo = np.zeros((n_grads, nyc, nxc))
        for g in range(n_grads):
            pd = _persist_deriv(Q, S, a_lin_fit, a3_fit, b_fit)
            dQ = rate_c / pd
            Q = _mig_group(Q + dQ, Mx_fit, My_fit, thr_fit)
            S = S + Q
            photo[g] = Q - Qprev; Qprev = Q.copy()
        med_corr = med_obs_c - (photo - rate_c[None])
        pc, _, _, _ = np.linalg.lstsq(Xb, med_corr.reshape(n_grads, -1), rcond=None)
        rate_c = pc[0].reshape(nyc, nxc)
        Adec_c = pc[1].reshape(nyc, nxc)
        delta_c = pc[2].reshape(nyc, nxc)

    if verbose:
        if aniso:
            print(f'  Mx={Mx_fit:.4e}  My={My_fit:.4e}  threshold={thr_fit:.1f} DN  '
                  f'at x={sx}, y={sy}')
        elif fit_a_lin:
            print(f'  M={Mx_fit:.4e}  threshold={thr_fit:.1f} DN  a_lin={a_lin_fit:.3e}  '
                  f'at x={sx}, y={sy}')
        else:
            print(f'  M={Mx_fit:.4e}  threshold={thr_fit:.1f} DN  at x={sx}, y={sy}')

    if diagnostics:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        sim = model_diff(Mx_fit, My_fit, thr_fit, rate_c, Adec_c, delta_c, a_lin_fit, a3_fit, b_fit)
        res = obs - sim

        rr = np.round(rmap).astype(int)
        r_int = np.arange(0, cut)
        # Median, not mean -- see fit_bfe_params' equivalent diagnostics
        # block for why (single contaminated pixels dominate a bin's mean).
        rp_obs = np.array([np.nanmedian(obs[rr == ri]) for ri in r_int])
        rp_sim = np.array([np.nanmedian(sim[rr == ri]) for ri in r_int])

        # Colour scale: exclude DQ-flagged pixels from any group (core
        # protected) -- see fit_bfe_params' equivalent diagnostics block.
        _display_bad = _dq_exclusion_mask(dq, sy - cut, sy + cut + 1, sx - cut, sx + cut + 1,
                                          list(range(n_grads + 1)), protect_center=(cut, cut))
        _vabs_src = obs if _display_bad is None else np.where(_display_bad, np.nan, obs)
        vabs = np.nanpercentile(np.abs(_vabs_src), 99.5)
        ext = [-cut - 0.5, cut + 0.5, -cut - 0.5, cut + 0.5]

        # Auto-scale to nearest 10^n so values are readable
        import math
        _exp = int(math.floor(math.log10(vabs))) if vabs > 0 else 0
        _scale = 10 ** _exp
        _exp_str = f'$\\times10^{{{_exp}}}$' if _exp != 0 else ''

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        if aniso:
            title_model = f'Model (Mx={Mx_fit:.2e}, My={My_fit:.2e}, thr={thr_fit:.1f})'
        elif fit_a_lin:
            title_model = f'Model (M={Mx_fit:.2e}, thr={thr_fit:.1f}, a_lin={a_lin_fit:.2e})'
        else:
            title_model = f'Model (M={Mx_fit:.2e}, thr={thr_fit:.1f} DN)'
        for ax, img, title in [
            (axes[0, 0], obs,  'Observed late$-$early'),
            (axes[0, 1], sim,  title_model),
            (axes[1, 0], res,  'Residual (obs$-$model)'),
        ]:
            im = ax.imshow(img / _scale, origin='lower', cmap='RdBu_r',
                           vmin=-vabs / _scale, vmax=vabs / _scale, extent=ext)
            fig.colorbar(im, ax=ax, label=rf'Norm. $\Delta$flux {_exp_str}')
            ax.set_title(title)
            ax.set_xlabel(r'$\Delta x$ (px)')
            ax.set_ylabel(r'$\Delta y$ (px)')

        ax = axes[1, 1]
        ax.plot(r_int, rp_obs / _scale, 'k-', lw=1.5, label='Observed')
        ax.plot(r_int, rp_sim / _scale, color='C3', ls='--', lw=1.5, label='Model')
        ax.axhline(0, color='k', lw=0.5, ls=':')
        ax.axvline(fit_r, color='C0', lw=1.0, ls='--', label=f'fit_r = {fit_r} px')
        ax.set_xlabel('Radius (px)')
        ax.set_ylabel(rf'Mean $\Delta$flux {_exp_str}')
        ax.set_title('Radial profile')
        ax.legend(fontsize=8)

        fig.suptitle(f'Migration fit diagnostics  (star x={sx}, y={sy})',
                     fontsize=11, fontweight='bold')
        fig.tight_layout()
        if save_path is None:
            save_path = 'migration_fit_diagnostics.png'
        fig.savefig(Path(save_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        if verbose:
            print(f'  Saved migration fit diagnostics to {save_path}')

    d = model_diff(Mx_fit, My_fit, thr_fit, rate_c, Adec_c, delta_c, a_lin_fit, a3_fit, b_fit)
    npar = 3 if fit_a_lin else 2
    chi2n = float(np.sum((((d - obs) * w)[fitmask]) ** 2)) / max(int(fitmask.sum()) - npar, 1)

    if return_model:
        model = dict(obs=obs, model=d, r_map=rmap, fitmask=fitmask,
                    M=Mx_fit, thr=thr_fit, chi2_n=chi2n, cy=cy, cx=cx, cut=cut,
                    norm_ap=norm_ap, bg_ann=bg_ann)
        if return_chi2:
            return Mx_fit, My_fit, thr_fit, sx, sy, a_lin_fit, chi2n, model
        return Mx_fit, My_fit, thr_fit, sx, sy, a_lin_fit, model

    if return_chi2:
        return Mx_fit, My_fit, thr_fit, sx, sy, a_lin_fit, chi2n

    return Mx_fit, My_fit, thr_fit, sx, sy, a_lin_fit


def _fit_charge_adaptive_scale(grads_bfe, n_grads, Adec_map, tau, delta_map,
                               n_sigma=5.0, min_excess_dn=100.0,
                               min_ramp_median_dn=500.0, sci_mask=None,
                               fit_alpha=0.05, verbose=False):
    """
    Detect integrations where a pixel's whole ramp is far brighter than its
    own baseline for that pixel, and fit an extra per-pixel-per-integration
    reset-decay amplitude for just those integrations, on top of the
    existing median-derived Adec_map.

    The per-pixel Adec_map fit (in correct_bfe_rcd) uses the median over
    integrations, so it implicitly represents a "typical" charge level. A
    transient (moving object, unmasked cosmic ray) that makes a pixel's
    ramp much brighter for a whole integration increases the true
    reset-decay amplitude there, which the fixed per-pixel Adec_map can't
    capture on its own.

    Detection uses the median gradient over each integration's ramp at each
    pixel, which is robust to a single-group cosmic-ray spike landing at
    any group (a CR shows up as one outlier value among many groups, so it
    barely moves the median) while still responding to genuine whole-ramp
    brightening (which elevates most/all groups together).

    A pixel's own charge-vs-excess relation can't be pooled across pixels
    (different pixels respond very differently at the same brightness), so
    each flagged (pixel, integration) is fit independently. Doing that from
    a single point (e.g. just the g=0 deviation from a trend fit through
    later groups) is circular -- it is guaranteed to zero out that one
    point regardless of whether the deviation was real or just noise, and
    ignores that the "trend" groups themselves still carry a small
    (non-negligible) tail of the same decay. Instead, for each flagged
    ramp, an extra decay amplitude is fit jointly against *every* group in
    one linear least-squares solve:

        grads_fixed[g] = C + slope*g + extra_A * exp(-g/tau)

    where ``grads_fixed`` is this pixel's gradients after the existing
    (non-adaptive) Adec_map/delta_map correction, and ``tau`` is the same
    fixed, detector-wide decay timescale used everywhere else (never
    refit per-ramp or per-pixel). All three parameters come from the same
    fit, so the linear trend and the decay amplitude are disentangled
    using the whole ramp rather than an arbitrary early/late group split.

    Parameters
    ----------
    grads_bfe : ndarray (n_int, n_grads_all, ny, nx)
        BFE-corrected gradients (Step 1 output of correct_bfe_rcd).
    n_grads : int
        Number of gradients to correct (excludes the last-frame anomaly).
    Adec_map : ndarray (ny, nx)
        Per-pixel decay amplitude from the existing median-based fit.
    tau : float
        Fixed, detector-wide reset-decay timescale (groups). Shared with
        the main correct_bfe_rcd fit; never refit here.
    delta_map : ndarray (ny, nx)
        Per-pixel first-frame offset from the existing median-based fit.
    n_sigma : float
        Threshold, in robust (MAD-based) sigma above a pixel's own baseline
        median gradient, for flagging an integration as "whole ramp bright".
    min_excess_dn : float
        Floor, in DN, on the excess required to flag: the threshold is
        baseline + max(n_sigma*robust_sigma, min_excess_dn). Protects
        against pixels with an artificially small noise estimate making
        n_sigma*robust_sigma trivially small in absolute terms.
    min_ramp_median_dn : float
        Floor, in DN, on the flagged ramp_median value itself (not the
        excess above baseline). Excludes low-brightness pixels where even
        a "significant" excursion is small in absolute terms.
    sci_mask : ndarray (ny, nx) bool, optional
        True = good science pixels. Non-science pixels (reference columns,
        etc.) have very different statistical behavior and should never be
        flagged. Strongly recommended whenever available.
    fit_alpha : float, default 0.05
        Significance level for the F-test comparing the linear+decay model
        against a linear-only model. A ramp's fitted extra_A is only used
        if the decay term reduces the residual sum of squares by more than
        chance at this significance level; otherwise extra_A=0 for that
        ramp (falls back to the existing per-pixel Adec_map only).

    Returns
    -------
    scale : ndarray (n_int, ny, nx)
        Multiplicative factor (>=1) to apply to Adec_map per integration,
        equal to 1 + extra_A/Adec_map for flagged (pixel, integration)
        pairs and 1 elsewhere.
    median_extra_A : float or None
        Median fitted extra amplitude (DN) across flagged points, or None
        if there weren't enough flagged points to fit anything.
    """
    n_int, _, ny, nx = grads_bfe.shape

    # Per-integration "typical" ramp level: a plain median over the
    # n_grads (~11) points in that integration's ramp. Iterative
    # sigma-clipping is unreliable at this sample size -- astropy always
    # performs at least one clip pass regardless of maxiters, which can
    # asymmetrically discard legitimate low (or high) values on a small
    # sample and bias the result away from the true center. A plain
    # median is already fairly resistant to a single-group CR spike (one
    # outlier among ~11 points has limited leverage on the median).
    ramp_median = np.median(grads_bfe[:, :n_grads], axis=1)           # (n_int, ny, nx)

    # Per-pixel baseline and robust scatter across integrations, same
    # reasoning: sigma-clipping excludes the minority of integrations
    # where a real transient is present from biasing the "typical" value.
    # Clip sigma is intentionally wide (15, not e.g. 3-5): too aggressive a
    # clip here excludes genuine, modest pixel-to-pixel scatter (e.g. a
    # slowly-decaying persistence trend spread over many integrations)
    # from the sigma estimate, artificially shrinking it and making
    # ordinary variation look like a significant excursion. At sigma=15 it
    # still excludes a real large transient (which should dominate its own
    # detection) without swallowing normal scatter into "not baseline".
    _, baseline_median, robust_sigma = sigma_clipped_stats(
        ramp_median, axis=0, sigma=15.0, maxiters=5,
        cenfunc='median', stdfunc='mad_std')                          # (ny, nx) each

    # min_excess_dn is a floor on the excess required to flag, not an
    # independent condition: a pixel with an artificially tiny noise
    # estimate can make n_sigma*robust_sigma trivially small in absolute
    # terms, so the excess above baseline must be at least min_excess_dn
    # DN regardless of how many "sigma" that nominally represents.
    threshold = baseline_median + np.maximum(n_sigma * robust_sigma, min_excess_dn)
    flagged = (ramp_median > threshold[None]) & (ramp_median >= min_ramp_median_dn)

    valid_pixel = (Adec_map > 0.5) & (baseline_median > 0)
    if sci_mask is not None:
        valid_pixel = valid_pixel & sci_mask.astype(bool)
    flagged = flagged & valid_pixel[None]
    n_flagged = int(flagged.sum())
    if verbose:
        print(f'  charge_adaptive: {n_flagged} flagged (pixel,integration) pairs')
    if n_flagged < 10:
        if verbose:
            print('  charge_adaptive: too few flagged points — skipping scaling')
        return np.ones((n_int, ny, nx)), None

    # Baseline (non-adaptive) RCD-corrected gradients: isolates any extra
    # charge-dependent excess above what the existing per-pixel fit already
    # removes, which is what the per-ramp joint fit below targets.
    g_arr = np.arange(n_grads)
    exp_g = np.exp(-g_arr / tau)
    grads_fixed = grads_bfe[:, :n_grads].copy()
    grads_fixed -= Adec_map[None, None] * exp_g[None, :, None, None]
    grads_fixed[:, 0] += delta_map[None]

    # Shared design matrices (same g_arr/tau for every ramp) -> shared
    # pseudo-inverses, reused for every flagged (pixel, integration).
    # X_lin: linear-trend-only model (no decay term), the null hypothesis.
    # X_full: linear trend + decay term, the model actually applied.
    X_lin = np.column_stack([np.ones(n_grads), g_arr])         # (n_grads, 2)
    X_full = np.column_stack([np.ones(n_grads), g_arr, exp_g])  # (n_grads, 3)
    X_lin_pinv = np.linalg.pinv(X_lin)
    X_full_pinv = np.linalg.pinv(X_full)

    ints, ys, xs = np.where(flagged)
    ramps = grads_fixed[ints, :, ys, xs]                      # (n_flagged, n_grads)

    coefs_lin = ramps @ X_lin_pinv.T                          # (n_flagged, 2)
    resid_lin = ramps - coefs_lin @ X_lin.T
    rss_lin = np.sum(resid_lin**2, axis=1)

    coefs_full = ramps @ X_full_pinv.T                        # (n_flagged, 3)
    resid_full = ramps - coefs_full @ X_full.T
    rss_full = np.sum(resid_full**2, axis=1)

    # F-test: does adding the decay term (1 extra parameter) significantly
    # reduce the residual sum of squares versus the linear-only model? A
    # per-ramp least-squares amplitude will always come out nonzero even
    # when the ramp's shape is pure noise/trend -- this is what stops that
    # amplitude from being applied when it isn't actually explaining the
    # ramp's shape any better than a straight line would.
    dof_full = n_grads - 3
    safe_rss_full = np.where(rss_full > 0, rss_full, 1e-12)
    F_stat = np.where(rss_full > 0,
                       (rss_lin - rss_full) / (safe_rss_full / dof_full),
                       np.inf)
    F_crit = f_dist.ppf(1 - fit_alpha, 1, dof_full)
    good_fit = F_stat > F_crit

    extra_A = np.where(good_fit, np.clip(coefs_full[:, 2], 0.0, None), 0.0)

    if verbose:
        n_good = int(good_fit.sum())
        print(f'  charge_adaptive: full-ramp joint fit, {n_good}/{n_flagged} ramps '
              f'passed the decay-term significance test (alpha={fit_alpha}), '
              f'median extra amplitude {np.median(extra_A[good_fit]) if n_good else 0:.2f} DN')

    safe_A = np.where(Adec_map > 0.5, Adec_map, 1.0)
    extra_A_map = np.zeros((n_int, ny, nx))
    extra_A_map[ints, ys, xs] = extra_A
    scale_flagged = 1.0 + extra_A_map / safe_A[None]
    scale = np.where(flagged, scale_flagged, 1.0)
    return scale, float(np.median(extra_A))


def fit_migration_params_joint(cube, n_stars=5, star_positions=None,
                               M_init=4.2e-7, thr_init=37.2,
                               bg_mask=None, sci_mask=None,
                               bfe_early_groups=None, bfe_late_groups=None,
                               cut=None, fit_r=None, max_iter=6, M_tol=0.02,
                               flux_min=None, snr_min=10.0,
                               instrument='miri', fit_rcd_decay=None,
                               drop_last_gradient=None, nframes=1, groupgap=0,
                               master_ref=None, dq=None, psf_fwhm_px=None,
                               verbose=False, diagnostics=False, save_path=None):
    """
    Fits a single, shared isotropic migration amplitude M and force
    threshold from MULTIPLE bright sources simultaneously, instead of one
    star at a time. The migration parameters describe a detector-wide
    physical process (charge redistribution), not a per-star property, so
    a joint fit across several independent sources is better-constrained
    than any single star's fit -- especially useful for cross-checking
    whether independent single-star fits (which can each be biased by
    that star's own residual contamination) actually agree.

    star_positions : list of (x, y), optional
        Explicit source positions to use. If None, the brightest n_stars
        round, unsaturated sources are auto-detected from master_ref (or
        the cube's own median gradient) via SEP, ranked by flux.
    n_stars : int, default 5
        Number of sources to use when star_positions is None.

    All other parameters match fit_migration_params (isotropic case only --
    no aniso, fit_a_lin, fix_M/fix_thr; those need a per-star treatment
    that doesn't make sense shared across stars). See its docstring for
    fit_rcd_decay/drop_last_gradient/cut/nframes/groupgap/dq/psf_fwhm_px.

    Returns
    -------
    dict with keys: M, thr, chi2_n (total, over all stars), n_stars_used,
    stars (list of dicts, one per star: x, y, chi2_n, n_fit_pix).
    """
    import sep
    from scipy.optimize import minimize as _minimize

    instrument = instrument.lower() if instrument else instrument
    if fit_rcd_decay is None:
        fit_rcd_decay = (instrument == 'miri')
    if drop_last_gradient is None:
        drop_last_gradient = (instrument == 'miri')
    if cut is None:
        cut = 11 if instrument == 'nircam' else 20

    cube = np.asarray(cube, dtype=float)
    n_int, n_groups, ny, nx = cube.shape
    n_grads = n_groups - 2 if drop_last_gradient else n_groups - 1
    g_arr = np.arange(n_grads, dtype=float)

    if instrument == 'nircam':
        if bfe_early_groups is None:
            bfe_early_groups = [0]
        if bfe_late_groups is None:
            bfe_late_groups = [n_grads - 1]
    if bfe_early_groups is None:
        n_e = max(2, min(3, n_grads // 4)); start = 1 if n_grads < 8 else 2
        bfe_early_groups = list(range(start, start + n_e))
    if bfe_late_groups is None:
        n_e = max(2, min(3, n_grads // 4))
        bfe_late_groups = list(range(n_grads - n_e, n_grads))

    grads = np.diff(cube, axis=1)[:, :n_grads]
    med = np.median(grads, axis=0)

    # Global rate/RCD maps -- shared across all stars (same as the
    # single-star fitter; only the crop position differs per star).
    yy_full, xx_full = np.mgrid[:ny, :nx]
    if bg_mask is not None:
        _bg_full = bg_mask.astype(bool)
    else:
        _bg_full = np.ones((ny, nx), dtype=bool)
    if sci_mask is not None:
        _bg_full = _bg_full & sci_mask.astype(bool)

    if fit_rcd_decay:
        mb = np.nanmean(med[1:][:, _bg_full], axis=1)
        popt, _ = curve_fit(lambda g, C, A, t: C + A * np.exp(-g / t), g_arr[1:], mb,
                            p0=[mb[-1], mb[0] - mb[-1], 1.5])
        tau = float(popt[2])
        exp_g = np.exp(-g_arr / tau)
        ff = np.zeros(n_grads); ff[0] = -1.0
        Xb = np.column_stack([np.ones(n_grads), exp_g, ff])
        p, _, _, _ = np.linalg.lstsq(Xb, med.reshape(n_grads, -1), rcond=None)
        rate, Adec, delta = (p[i].reshape(ny, nx) for i in range(3))
    else:
        tau = 1.0
        rate = np.median(med, axis=0)
        Adec = np.zeros((ny, nx))
        delta = np.zeros((ny, nx))
    exp_g = np.exp(-g_arr / tau)

    # Detect sources
    if star_positions is None:
        if master_ref is not None:
            detect = np.asarray(master_ref, dtype=np.float64)
        else:
            detect = np.median(grads[:, 1:n_grads], axis=(0, 1)).astype(np.float64)
        sep_mask = (~sci_mask.astype(bool)) if sci_mask is not None else None
        bkg = sep.Background(detect, mask=sep_mask)
        all_obj = sep.extract((detect - bkg.back()).astype(np.float64), 5.0,
                              err=bkg.globalrms, mask=sep_mask)
        edge = cut + 5
        obj = all_obj[(all_obj['x'] > edge) & (all_obj['x'] < nx - edge) & (all_obj['y'] > edge) &
                      (all_obj['y'] < ny - edge) & (all_obj['a'] / all_obj['b'] < 3)]
        obj = obj[np.argsort(obj['flux'])[::-1]]
        star_positions = []
        for o in obj:
            sxx, syy = int(round(o['x'])), int(round(o['y']))
            if dq is not None:
                _sat = _dq_exclusion_mask(dq, syy - 3, syy + 4, sxx - 3, sxx + 4,
                                          list(range(n_groups)))
                if _sat is not None and np.any(_sat):
                    continue
            # Reject crowded/blended candidates -- see
            # _isolated_brightest docstring: no other detected source
            # within `cut` px (the normalisation-aperture scale).
            d2 = (all_obj['x'] - o['x']) ** 2 + (all_obj['y'] - o['y']) ** 2
            if int(np.sum((d2 > 1e-6) & (d2 <= cut ** 2))) > 0:
                if verbose:
                    print(f'  Skipping ({sxx},{syy}) -- another source within {cut} px (blended)')
                continue
            star_positions.append((sxx, syy))
            if len(star_positions) >= n_stars:
                break
    if verbose:
        print(f'  Using {len(star_positions)} stars: {star_positions}')

    yy_c, xx_c = np.mgrid[:2 * cut + 1, :2 * cut + 1]
    rmap = np.sqrt((yy_c - cut) ** 2 + (xx_c - cut) ** 2)
    norm_ap = rmap <= (cut - 6)
    bg_ann = (rmap > (cut - 6)) & (rmap <= (cut - 1))

    stars = []
    for (sx, sy) in star_positions:
        crop = cut + 50
        y0, y1 = max(0, sy - crop), min(ny, sy + crop + 1)
        x0, x1 = max(0, sx - crop), min(nx, sx + crop + 1)
        cy, cx = sy - y0, sx - x0
        rate_c, Adec_c, delta_c = (m[y0:y1, x0:x1] for m in (rate, Adec, delta))
        gc = grads[:, :, y0:y1, x0:x1]
        med_obs_c = np.median(gc, axis=0)
        nyc, nxc = rate_c.shape

        _crop_bad = np.zeros((nyc, nxc), dtype=bool)
        if sci_mask is not None:
            _crop_bad |= ~sci_mask[y0:y1, x0:x1].astype(bool)
        _crop_dq_bad = _dq_exclusion_mask(dq, y0, y1, x0, x1, list(range(n_grads + 1)),
                                          protect_center=(cy, cx))
        if _crop_dq_bad is not None:
            _crop_bad |= _crop_dq_bad
        _crop_bad |= _other_source_mask(rate_c, cy, cx,
                                        sci_mask=(sci_mask[y0:y1, x0:x1] if sci_mask is not None else None))
        if np.any(_crop_bad):
            if verbose:
                print(f"    star ({sx},{sy}): inpainting {int(_crop_bad.sum())} bad pixel(s) in the model input maps")
            rate_c = _inpaint_bad_pixels(rate_c, _crop_bad)
            Adec_c = _inpaint_bad_pixels(Adec_c, _crop_bad)
            delta_c = _inpaint_bad_pixels(delta_c, _crop_bad)

        def _cut(img, cy=cy, cx=cx):
            return img[cy - cut:cy + cut + 1, cx - cut:cx + cut + 1]

        _sci_bad_stamp = (~_cut(sci_mask[y0:y1, x0:x1].astype(bool))
                          if sci_mask is not None else np.zeros((2 * cut + 1, 2 * cut + 1), dtype=bool))
        _bad_by_group = []
        for _g in range(n_grads):
            _dqg = _dq_exclusion_mask(dq, sy - cut, sy + cut + 1, sx - cut, sx + cut + 1,
                                      _grad_to_group_indices([_g]), protect_center=(cut, cut))
            _bad_by_group.append(_sci_bad_stamp if _dqg is None else (_sci_bad_stamp | _dqg))

        def _norm_diff(stack, _bad_by_group=_bad_by_group, _cut=_cut):
            def acc(gl):
                s = 0.0
                for g in gl:
                    c = _cut(stack[g])
                    bad = _bad_by_group[g]
                    good_bg = bg_ann & ~bad
                    bgm = np.median(c[good_bg]) if np.any(good_bg) else np.median(c[bg_ann])
                    c = c - bgm
                    good_ap = norm_ap & ~bad
                    flux = c[good_ap].sum() if np.any(good_ap) else c[norm_ap].sum()
                    s = s + c / flux
                return s / len(gl)
            return acc(bfe_late_groups) - acc(bfe_early_groups)

        obs = _norm_diff(med_obs_c)
        bg_std = float(np.std(obs[bg_ann])) if n_int == 1 else np.std(
            [_norm_diff(gc[i]) for i in range(n_int)], axis=0)
        noise = np.full_like(obs, max(np.mean(bg_std) if hasattr(bg_std, '__len__') else bg_std, 1e-12))

        _fit_r_floor = (max(4, round(_FIT_R_FWHM_RATIO_MIGRATION * psf_fwhm_px))
                       if psf_fwhm_px else 8)
        _fit_r = fit_r
        if _fit_r is None:
            snr = np.array([np.mean(np.abs(obs[np.round(rmap).astype(int) == ri])) /
                            np.mean(noise[np.round(rmap).astype(int) == ri])
                            for ri in range(1, cut)])
            above = np.where(snr > 2.0)[0]
            _fit_r = max(_fit_r_floor, int(above[-1]) + 1) if len(above) > 0 else _fit_r_floor
        fitmask = rmap <= _fit_r
        _dq_bad = _dq_exclusion_mask(dq, sy - cut, sy + cut + 1, sx - cut, sx + cut + 1,
                                     _grad_to_group_indices(list(bfe_early_groups) + list(bfe_late_groups)),
                                     protect_center=(cut, cut))
        if _dq_bad is not None:
            fitmask = fitmask & ~_dq_bad
        _mask_radial_outliers(obs, rmap, fitmask)
        w = 1.0 / noise

        stars.append(dict(sx=sx, sy=sy, rate_c=rate_c, Adec_c=Adec_c, delta_c=delta_c,
                          nyc=nyc, nxc=nxc, obs=obs, w=w, fitmask=fitmask,
                          norm_diff=_norm_diff, fit_r=_fit_r))

    tot_frames = nframes + groupgap

    def _model_diff_star(Mx, thr, st):
        rate_c, Adec_c, delta_c = st['rate_c'], st['Adec_c'], st['delta_c']
        nyc, nxc = st['nyc'], st['nxc']
        dQ_frame = rate_c / tot_frames
        Q = np.zeros((nyc, nxc)); S = np.zeros((nyc, nxc))
        group_avg_prev = np.zeros((nyc, nxc))
        out = np.zeros((n_grads, nyc, nxc))
        for g in range(n_grads):
            frame_sum = np.zeros((nyc, nxc))
            for f in range(tot_frames):
                dQ = dQ_frame
                Q = _mig_group(Q + dQ, Mx, Mx, thr)
                S = S + Q
                if f < nframes:
                    frame_sum = frame_sum + Q
            group_avg = frame_sum / nframes
            photo = group_avg - group_avg_prev
            group_avg_prev = group_avg
            tg = photo + Adec_c * exp_g[g]
            if g == 0:
                tg = tg - delta_c
            out[g] = tg
        return st['norm_diff'](out)

    def chi2(par):
        logM, thr = par
        if not (-9 <= logM <= -2 and 0 <= thr <= 5000):
            return 1e30
        M = 10 ** logM
        total = 0.0
        for st in stars:
            d = _model_diff_star(M, thr, st)
            if not np.all(np.isfinite(d)):
                return 1e30
            total += float(np.sum((((d - st['obs']) * st['w'])[st['fitmask']]) ** 2))
        return total

    logM0, thr0 = np.log10(M_init), thr_init
    M_fit, thr_fit = M_init, thr_init
    for _it in range(max_iter):
        res = _minimize(chi2, [logM0, thr0], method='Powell',
                        options={'xtol': 1e-5, 'ftol': 1e-9, 'maxiter': 20000})
        M_now = 10 ** res.x[0]
        thr_fit = max(0.0, res.x[1])
        if verbose:
            print(f'  [iter {_it}] M={M_now:.4e}  thr={thr_fit:.1f} DN  chi2={res.fun:.3f}')
        if abs(M_now - M_fit) <= M_tol * abs(M_now) and _it > 0:
            M_fit = M_now
            break
        M_fit = M_now
        logM0, thr0 = res.x[0], thr_fit

    total_pix = sum(int(st['fitmask'].sum()) for st in stars)
    chi2n = chi2([np.log10(M_fit), thr_fit]) / max(total_pix - 2, 1)
    per_star = []
    for st in stars:
        d = _model_diff_star(M_fit, thr_fit, st)
        npix = int(st['fitmask'].sum())
        c2 = float(np.sum((((d - st['obs']) * st['w'])[st['fitmask']]) ** 2)) / max(npix - 2, 1)
        per_star.append(dict(x=st['sx'], y=st['sy'], chi2_n=c2, n_fit_pix=npix, fit_r=st['fit_r']))
        if verbose:
            print(f"    star ({st['sx']},{st['sy']}): chi2_n={c2:.3f}  n_fit_pix={npix}  fit_r={st['fit_r']}")

    if diagnostics:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import math

        if save_path is None:
            save_path = 'migration_fit_joint_diagnostics.png'
        save_path = Path(save_path)
        rr = np.round(rmap).astype(int)
        r_int = np.arange(0, cut)
        ext = [-cut - 0.5, cut + 0.5, -cut - 0.5, cut + 0.5]

        for i, st in enumerate(stars):
            d = _model_diff_star(M_fit, thr_fit, st)
            res = st['obs'] - d

            _display_bad = _dq_exclusion_mask(dq, st['sy'] - cut, st['sy'] + cut + 1,
                                              st['sx'] - cut, st['sx'] + cut + 1,
                                              list(range(n_grads + 1)), protect_center=(cut, cut))
            _vabs_src = st['obs'] if _display_bad is None else np.where(_display_bad, np.nan, st['obs'])
            vabs = np.nanpercentile(np.abs(_vabs_src), 99.5)
            _exp = int(math.floor(math.log10(vabs))) if vabs > 0 else 0
            _scale = 10 ** _exp
            _exp_str = f'$\\times10^{{{_exp}}}$' if _exp != 0 else ''

            rp_obs = np.array([np.nanmedian(st['obs'][rr == ri]) for ri in r_int])
            rp_sim = np.array([np.nanmedian(d[rr == ri]) for ri in r_int])

            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            for ax, img, title in [
                (axes[0, 0], st['obs'], 'Observed late$-$early'),
                (axes[0, 1], d, f"Model (M={M_fit:.2e}, thr={thr_fit:.1f})"),
                (axes[1, 0], res, 'Residual (obs$-$model)'),
            ]:
                im = ax.imshow(img / _scale, origin='lower', cmap='RdBu_r',
                               vmin=-vabs / _scale, vmax=vabs / _scale, extent=ext)
                fig.colorbar(im, ax=ax, label=rf'Norm. $\Delta$flux {_exp_str}')
                ax.set_title(title)
                ax.set_xlabel(r'$\Delta x$ (px)')
                ax.set_ylabel(r'$\Delta y$ (px)')

            ax = axes[1, 1]
            ax.plot(r_int, rp_obs / _scale, 'k-', lw=1.5, label='Observed')
            ax.plot(r_int, rp_sim / _scale, color='C3', ls='--', lw=1.5, label='Model')
            ax.axhline(0, color='k', lw=0.5, ls=':')
            ax.axvline(st['fit_r'], color='C0', lw=1.0, ls='--', label=f"fit_r = {st['fit_r']} px")
            ax.set_xlabel('Radius (px)')
            ax.set_ylabel(rf'Median $\Delta$flux {_exp_str}')
            ax.set_title('Radial profile')
            ax.legend(fontsize=8)

            fig.suptitle(f"Joint migration fit -- star ({st['sx']}, {st['sy']})  "
                        f"chi2_n={per_star[i]['chi2_n']:.3f}", fontsize=11, fontweight='bold')
            fig.tight_layout()
            star_path = save_path.with_name(f"{save_path.stem}_star{i}_x{st['sx']}_y{st['sy']}{save_path.suffix}")
            fig.savefig(star_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            if verbose:
                print(f'  Saved per-star diagnostics to {star_path}')

        # Compact summary grid across all stars
        n = len(stars)
        fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.4), squeeze=False)
        for i, st in enumerate(stars):
            d = _model_diff_star(M_fit, thr_fit, st)
            rp_obs = np.array([np.nanmedian(st['obs'][rr == ri]) for ri in r_int])
            rp_sim = np.array([np.nanmedian(d[rr == ri]) for ri in r_int])
            ax = axes[0, i]
            ax.plot(r_int, rp_obs, 'k-', lw=1.5, label='Observed')
            ax.plot(r_int, rp_sim, 'C3--', lw=1.5, label='Model')
            ax.axhline(0, color='k', lw=0.5, ls=':')
            ax.axvline(st['fit_r'], color='C0', lw=1.0, ls='--')
            ax.set_title(f"({st['sx']},{st['sy']})  chi2_n={per_star[i]['chi2_n']:.2f}", fontsize=9)
            ax.set_xlabel('r (px)')
            if i == 0:
                ax.legend(fontsize=7)
            ax2 = axes[1, i]
            res_img = st['obs'] - d
            vabs2 = np.nanpercentile(np.abs(st['obs'][st['fitmask']]), 95)
            ax2.imshow(res_img, origin='lower', cmap='RdBu_r', vmin=-vabs2, vmax=vabs2)
            ax2.set_title('Residual', fontsize=9)
        fig.suptitle(f'Joint migration fit: M={M_fit:.3e}  thr={thr_fit:.1f} DN  '
                     f'chi2_n={chi2n:.3f}  ({n} stars)', fontweight='bold')
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        if verbose:
            print(f'  Saved joint fit summary to {save_path}')

    return dict(M=M_fit, thr=thr_fit, chi2_n=chi2n, n_stars_used=len(stars), stars=per_star)


def correct_bfe_rcd(cube, method='migration',
                    M_mig=4.2e-7, M_mig_y=1, thr_mig=37.2,
                    A_bfe=1.035e-6, alpha_bfe=3.43, b_bfe=-0.50, c_bfe=0.056,
                    bg_mask=None, late_groups=None, verbose=False,
                    fit_bfe=False, sci_mask=None,
                    bfe_early_groups=None, bfe_late_groups=None,
                    ap_radius=5, cut=None, fit_r=10,
                    star_x=None, star_y=None,
                    fix_M=None, fix_thr=None,
                    nonparametric=True,
                    charge_adaptive=False, charge_adaptive_nsigma=5.0,
                    charge_adaptive_min_excess_dn=100.0,
                    charge_adaptive_min_ramp_median_dn=500.0,
                    return_stages=False,
                    instrument='miri', fit_rcd_decay=None, drop_last_gradient=None,
                    master_ref=None, psf_fwhm_px=None,
                    diagnostics=False, save_path=None):
    """
    Joint BFE + reset-decay correction for JWST ramp data.

    Three sequential steps applied to gradients:
      1. Causal BFE inversion (method-dependent, see ``method``).
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
    instrument : {'miri', 'nircam'}, default 'miri'
        Case-insensitive. Selects instrument-appropriate defaults for
        fit_rcd_decay and drop_last_gradient (see below), and for
        bfe_early_groups/bfe_late_groups (passed through to the BFE fitters
        when fit_bfe=True). MIRI's Si:As detectors show reset charge decay
        (RCD) and a last-frame anomaly; NIRCam's HgCdTe detectors show
        neither.
    fit_rcd_decay : bool, optional
        If True, run the parametric RCD subtraction (step 2: fit
        C + A*exp(-g/tau) and subtract). Defaults to True for
        instrument='miri', False for instrument='nircam' (no decay
        assumed — Adec_map and delta_map are fixed at zero instead).
    drop_last_gradient : bool, optional
        If True, exclude the final group-to-group gradient (MIRI's
        last-frame anomaly) from correction. Defaults to True for
        instrument='miri', False for instrument='nircam'.
    master_ref : ndarray (ny, nx), optional
        A clean, deep, cosmic-ray-free detection image on the same pixel
        grid as ``cube``, used for source detection when fit_bfe=True
        (passed through to fit_migration_params/fit_bfe_params instead of
        detecting on the cube's own median gradient). See fit_bfe_params'
        master_ref docstring.
    method : {'migration', 'kernel'}
        BFE model for step 1. 'migration' (default) uses the charge-migration
        model: accumulated charge diffuses out of high-charge pixels with a
        charge-weighted mobility and a force threshold (parameters M_mig,
        thr_mig), inverted by a Born-style fixed point. 'kernel' uses the
        source-centric flux-conserving kernel K = -(1 + b r + c r^2)/r^alpha
        (parameters A_bfe, alpha_bfe, b_bfe, c_bfe). Both conserve charge.
    M_mig : float
        Migration strength per group (used when method='migration').
    thr_mig : float
        Migration force threshold in DN (used when method='migration').
    A_bfe : float
        BFE kernel amplitude (used when method='kernel', default 1.035e-6).
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
    nonparametric : bool, default True
        If True, apply step 3 (per-pixel per-group median subtraction with
        flat-rate restoration). Set to False to skip this step and return
        only the parametric BFE + RCD correction.
    charge_adaptive : bool, default False
        If True, automatically detect integrations where a pixel's whole
        ramp is much brighter than its own baseline (e.g. a moving object
        transiting that pixel) and scale up the per-pixel decay amplitude
        for just those integrations, rather than applying the same
        median-derived amplitude everywhere. Detection is robust to
        cosmic-ray hits at any group (uses the median gradient over each
        integration's ramp, so a single-group spike doesn't trigger it).
        See ``_fit_charge_adaptive_scale`` for the full method.
    charge_adaptive_nsigma : float, default 5.0
        Threshold, in robust sigma above a pixel's own baseline, for
        flagging an integration as "whole ramp bright". Only used when
        charge_adaptive=True.
    charge_adaptive_min_excess_dn : float, default 100.0
        Floor, in DN, on the excess required to flag: the effective
        threshold is baseline + max(charge_adaptive_nsigma*robust_sigma,
        charge_adaptive_min_excess_dn). Protects against pixels with an
        artificially small noise estimate making the sigma-based threshold
        trivially small in absolute terms. Only used when
        charge_adaptive=True.
    charge_adaptive_min_ramp_median_dn : float, default 500.0
        Floor, in DN, on the flagged ramp_median value itself (not the
        excess above baseline). Excludes low-brightness pixels where even
        a "significant" excursion is small in absolute terms. Only used
        when charge_adaptive=True.
    return_stages : bool, default False
        If True, return a tuple ``(cube_cor, stages)`` where ``stages`` is a
        dict with keys ``'grads_raw'``, ``'grads_bfe'``, ``'grads_joint'``.
        The RCD-only corrected gradients are
        ``grads_raw - (grads_bfe - grads_joint)`` (BFE kept, RCD removed).
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
        If ``return_stages=True``, returns ``(cube_cor, stages)`` where
        ``stages`` is a dict with keys ``'grads_raw'``, ``'grads_bfe'``,
        ``'grads_joint'``.
    """
    from scipy.signal import fftconvolve

    instrument = instrument.lower() if instrument else instrument
    if fit_rcd_decay is None:
        fit_rcd_decay = (instrument == 'miri')
    if drop_last_gradient is None:
        drop_last_gradient = (instrument == 'miri')

    cube = np.asarray(cube, dtype=float)
    n_int, n_groups, ny, nx = cube.shape
    n_grads_all = n_groups - 1        # all gradients
    n_grads = n_groups - 2 if drop_last_gradient else n_groups - 1   # gradients to correct

    grads_raw = np.diff(cube, axis=1)   # (n_int, n_grads_all, ny, nx)
    g_arr = np.arange(n_grads, dtype=float)

    if late_groups is None:
        late_groups = list(range(n_grads - 3, n_grads))

    _My_aniso = None  # set below if aniso fit runs

    if fit_bfe:
        if method == 'migration':
            if verbose:
                print('Fitting migration parameters from brightest source...')
            Mx_fit, My_fit, thr_fit, _sx, _sy, _a_lin = fit_migration_params(
                cube, M_init=M_mig, thr_init=thr_mig, bg_mask=bg_mask,
                sci_mask=sci_mask, bfe_early_groups=bfe_early_groups,
                bfe_late_groups=bfe_late_groups, ap_radius=ap_radius, cut=cut,
                fit_r=fit_r, verbose=verbose, aniso=(M_mig_y is None),
                fix_M=fix_M, fix_thr=fix_thr, instrument=instrument,
                fit_rcd_decay=fit_rcd_decay, drop_last_gradient=drop_last_gradient,
                master_ref=master_ref, psf_fwhm_px=psf_fwhm_px)
            if Mx_fit is None:
                if verbose:
                    print('No source meets brightness threshold — skipping BFE correction')
                M_mig = 0.0
            else:
                M_mig, thr_mig = Mx_fit, thr_fit
                if M_mig_y is None:
                    _My_aniso = My_fit
                if verbose:
                    print(f'Using fitted M={M_mig:.4e}  threshold={thr_mig:.1f} DN '
                          f'at x={_sx}, y={_sy}')
        else:
            if verbose:
                print('Fitting A_bfe from brightest source...')
            fit_result = fit_bfe_params(
                cube, alpha_bfe=alpha_bfe,
                bg_mask=bg_mask, sci_mask=sci_mask,
                bfe_early_groups=bfe_early_groups, bfe_late_groups=bfe_late_groups,
                ap_radius=ap_radius, cut=cut, fit_r=fit_r, verbose=verbose,
                instrument=instrument, fit_rcd_decay=fit_rcd_decay,
                drop_last_gradient=drop_last_gradient, master_ref=master_ref,
                psf_fwhm_px=psf_fwhm_px)
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

    # Step 1: BFE inversion (method-dependent), flux conserving.
    if method == 'migration':
        # Charge-migration inversion: recover the migration-free gradients with
        # a Born-style fixed point on the median ramp, then apply the per-group
        # correction to every integration. Charge is conserved by construction.
        # Migration is cardinal-only (x and y axes; diagonal = 0 by construction).
        # M_mig_y is a ratio: My = M_mig * M_mig_y. None means aniso fit was run.
        _M_y = _My_aniso if _My_aniso is not None else M_mig * (M_mig_y if M_mig_y is not None else 1)
        med_obs = np.median(grads_raw[:, :n_grads_all], axis=0)
        grads_bfe = grads_raw.copy()
        if M_mig and M_mig > 0:
            if verbose:
                print('  migration inversion...')
            true_grads = _mig_invert(med_obs, M_mig, _M_y, thr_mig)
            for g in range(n_grads_all):
                grads_bfe[:, g] = grads_raw[:, g] + (true_grads[g] - med_obs[g])[None]
    else:
        # Kernel Born inversion: grad_obs = true - A * K ⊛ (Q * true). K sums to
        # zero, so total image flux is exactly conserved.
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
    popt = None
    if not fit_rcd_decay:
        tau = 1.0
        if verbose:
            print('  fit_rcd_decay=False — skipping RCD decay model (no decay assumed)')
    elif len(g_fit) >= 3:
        popt, _ = curve_fit(_exp1, g_fit, mean_bg,
                            p0=[mean_bg[-1], mean_bg[0] - mean_bg[-1], 1.5])
        tau = float(popt[2])
    else:
        # Too few points to fit tau; use fixed default and fit only A and C.
        tau = 1.5
        if verbose:
            print(f'  too few groups to fit tau — using tau={tau:.2f} (fixed)')

    if fit_rcd_decay:
        exp_g = np.exp(-g_arr / tau)
        ff_col = np.zeros(n_grads); ff_col[0] = -1.0
        X = np.column_stack([np.ones(n_grads), exp_g, ff_col])
        params, _, _, _ = np.linalg.lstsq(
            X, med_bfe.reshape(n_grads, -1), rcond=None)
        Adec_map = params[1].reshape(ny, nx)
        delta_map = params[2].reshape(ny, nx)
    else:
        Adec_map = np.zeros((ny, nx))
        delta_map = np.zeros((ny, nx))

    charge_scale = np.ones((n_int, ny, nx))
    charge_scale_p = None
    if charge_adaptive:
        charge_scale, charge_scale_p = _fit_charge_adaptive_scale(
            grads_bfe, n_grads, Adec_map, tau, delta_map,
            n_sigma=charge_adaptive_nsigma,
            min_excess_dn=charge_adaptive_min_excess_dn,
            min_ramp_median_dn=charge_adaptive_min_ramp_median_dn,
            sci_mask=sci_mask, verbose=verbose)

    grads_joint = grads_bfe.copy()
    for g in range(n_grads):
        decay_g = Adec_map[None] * charge_scale * np.exp(-g / tau)   # (n_int, ny, nx)
        if g == 0:
            grads_joint[:, 0] = grads_bfe[:, 0] - decay_g + delta_map[None]
        else:
            grads_joint[:, g] = grads_bfe[:, g] - decay_g

    # Step 3: non-parametric median subtraction
    if nonparametric:
        med_joint = np.median(grads_joint[:, :n_grads], axis=0)   # (n_grads, ny, nx)
        C_hat = np.mean(med_joint[late_groups], axis=0)            # (ny, nx)
        grads_cor = grads_joint.copy()
        for g in range(n_grads):
            grads_cor[:, g] = grads_joint[:, g] - med_joint[g][None] + C_hat[None]
    else:
        grads_cor = grads_joint

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
        if popt is not None:
            ax.plot(g_fine, popt[0] + popt[1] * np.exp(-g_fine / tau), '--',
                    color='C3', lw=1.5, label=f'Fit  tau={tau:.2f} grp')
        else:
            ax.axhline(np.nanmean(mean_bg), color='C3', ls='--', lw=1.5,
                       label=f'Fixed tau={tau:.2f} grp')
        ax.set_xlabel('Gradient index')
        ax.set_ylabel('Mean gradient (DN/group)')
        ax.set_title('Global tau fit (after BFE step)')
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
        ax.set_title(f'BFE step size  (M={M_mig:.2e})')
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

        if star_x is not None and star_y is not None:
            r_between, r_ring = _diag_pixel_locations(
                grads_raw, star_x, star_y, n_grads, ny, nx,
                save_path=str(save_path).replace('.png', '_pixel_locations.png'),
                verbose=verbose)
            _diag_pixel_ramps(
                cube, grads_raw, grads_bfe, grads_joint, grads_cor,
                star_x, star_y, n_grads_all, r_between, r_ring,
                save_path=str(save_path).replace('.png', '_pixel_ramps.png'),
                verbose=verbose)

    if return_stages:
        return cube_cor, {'grads_raw': grads_raw, 'grads_bfe': grads_bfe,
                          'grads_joint': grads_joint}
    return cube_cor


def _diag_pixel_ramps(cube, grads_raw, grads_bfe, grads_joint, grads_cor,
                      star_x, star_y, n_grads_all, r_between, r_ring,
                      save_path='bfe_rcd_pixel_ramps.png', verbose=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_int, n_groups = cube.shape[0], cube.shape[1]
    ny, nx = cube.shape[2], cube.shape[3]
    groups = np.arange(n_groups)

    sky_y = max(5, min(ny - 5, star_y - 15))
    sky_x = max(5, min(nx - 5, star_x + 10))

    diag_pixels = [
        (star_y, star_x, 'Star centre ($r=0$)'),
        (star_y, star_x + r_between, rf'Between core \& ring ($r={r_between}$)'),
        (star_y, star_x + r_ring,    rf'On ring ($r={r_ring}$)'),
        (sky_y, sky_x, 'Sky'),
    ]

    fig_width = 240.0 / 72.27
    plt.rcParams.update({'font.family': 'serif', 'text.usetex': True, 'font.size': 9})

    fig, axes = plt.subplots(4, 4, figsize=(4 * fig_width, 3 * fig_width),
                             gridspec_kw={'height_ratios': [2, 1, 1, 1]})

    def _mean_ramp(grads_stage, py, px):
        px0 = cube[:, 0, py, px]
        csum = np.cumsum(grads_stage[:, :n_grads_all, py, px], axis=1)
        return np.column_stack([px0[:, None], px0[:, None] + csum]).mean(axis=0)

    def _all_ramps(grads_stage, py, px):
        px0 = cube[:, 0, py, px]
        csum = np.cumsum(grads_stage[:, :n_grads_all, py, px], axis=1)
        return np.column_stack([px0[:, None], px0[:, None] + csum])

    for col, (sy, sx, label) in enumerate(diag_pixels):
        ramp_raw = _mean_ramp(grads_raw, sy, sx)
        ramp_bfe_px = _mean_ramp(grads_bfe, sy, sx)
        ramp_joint_px = _mean_ramp(grads_joint, sy, sx)
        ramp_cor = _mean_ramp(grads_cor, sy, sx)

        bfe_contrib = ramp_bfe_px - ramp_raw
        rcd_contrib = ramp_joint_px - ramp_bfe_px
        nonpar_contrib = ramp_cor - ramp_joint_px

        ax_top = axes[0, col]
        ax_mid = axes[1, col]
        ax_bot = axes[2, col]
        ax_cmp = axes[3, col]

        for ramp, all_ramps, color, ls, stage, ax_res in [
            (ramp_raw, _all_ramps(grads_raw, sy, sx), 'k', '-', 'Before', ax_mid),
            (ramp_cor, _all_ramps(grads_cor, sy, sx), 'C3', '--', 'After', ax_bot),
        ]:
            coeffs, cov = np.polyfit(groups[1:-1], ramp[1:-1], 1, cov=True)
            slope, intercept = coeffs
            slope_err = np.sqrt(cov[0, 0])
            line = slope * groups + intercept
            resid = ramp - line
            all_resid = all_ramps - line[None, :]

            ax_top.plot(groups[1:-1], ramp[1:-1], 'o', color=color, ms=2, zorder=3)
            ax_top.plot(groups[[0, -1]], ramp[[0, -1]], 'o', color=color, ms=2, mfc='none', zorder=3)
            ax_top.plot(groups, line, ls=ls, color=color, lw=1.0,
                        label=rf'{stage}: $\hat{{m}}={slope:.1f}\pm{slope_err:.2g}$')

            ax_res.axhline(0, color='gray', lw=0.5, ls=':')
            ax_res.scatter(np.tile(groups, n_int), all_resid.ravel(),
                           s=0.5, color=color, alpha=0.05, zorder=1, linewidths=0)
            ax_res.plot(groups, resid, '-', color=color, lw=0.8)
            ax_res.plot(groups[1:-1], resid[1:-1], 'o', color=color, ms=2)
            ax_res.plot(groups[[0, -1]], resid[[0, -1]], 'o', color=color, ms=2, mfc='none')
            ax_res.set_xticks(groups[::2])
            if col == 0:
                ax_res.set_ylabel(f'{stage} resid.\\ (DN)')

            inner = resid[1:-1]
            pad = max(np.ptp(inner) * 0.4, 5.0)
            ax_res.set_ylim(inner.min() - pad, inner.max() + pad)

        ax_top.scatter(np.tile(groups, n_int), cube[:, :, sy, sx].ravel(),
                       s=0.5, color='k', alpha=0.05, zorder=1, linewidths=0)
        ax_top.set_title(label, fontsize=8)
        ax_top.set_xticks(groups[::2])
        ax_top.tick_params(labelbottom=False)
        if col == 0:
            ax_top.set_ylabel('DN')
        ax_top.legend(fontsize=9, frameon=False)

        ax_mid.tick_params(labelbottom=False)
        ax_bot.tick_params(labelbottom=False)

        ax_cmp.axhline(0, color='gray', lw=0.5, ls=':')
        ax_cmp.plot(groups, bfe_contrib, '-', color='C0', lw=0.8)
        ax_cmp.plot(groups[1:-1], bfe_contrib[1:-1], 'o', color='C0', ms=2, label='BFE')
        ax_cmp.plot(groups[[0, -1]], bfe_contrib[[0, -1]], 'o', color='C0', ms=2, mfc='none')
        ax_cmp.plot(groups, rcd_contrib, '-', color='C1', lw=0.8)
        ax_cmp.plot(groups[1:-1], rcd_contrib[1:-1], 's', color='C1', ms=2, label='RCD')
        ax_cmp.plot(groups[[0, -1]], rcd_contrib[[0, -1]], 's', color='C1', ms=2, mfc='none')
        ax_cmp.plot(groups, nonpar_contrib, '-', color='C2', lw=0.8)
        ax_cmp.plot(groups[1:-1], nonpar_contrib[1:-1], '^', color='C2', ms=2, label='Non-par.')
        ax_cmp.plot(groups[[0, -1]], nonpar_contrib[[0, -1]], '^', color='C2', ms=2, mfc='none')
        ax_cmp.set_xlabel('Group index')
        ax_cmp.set_xticks(groups[::2])
        if col == 0:
            ax_cmp.set_ylabel('Correction (DN)')
            ax_cmp.legend(fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.rcParams.update({'font.family': 'sans-serif', 'text.usetex': False, 'font.size': 10})
    if verbose:
        print(f'Saved pixel ramp diagnostics to {save_path}')



def _diag_pixel_locations(grads_raw, star_x, star_y, n_grads, ny, nx,
                           save_path='bfe_rcd_pixel_locations.png', verbose=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import math

    med_raw = np.median(grads_raw[:, :n_grads], axis=0)   # (n_grads, ny, nx)

    cut = 18
    early_groups_loc = [g for g in [2, 3] if g < n_grads]
    late_groups_loc = [g for g in [n_grads - 3, n_grads - 2, n_grads - 1] if 0 <= g < n_grads]

    cy, cx = star_y, star_x
    y0 = max(0, cy - cut)
    y1 = min(ny, cy + cut + 1)
    x0 = max(0, cx - cut)
    x1 = min(nx, cx + cut + 1)

    yy, xx = np.mgrid[:2*cut+1, :2*cut+1]
    r_map = np.sqrt((yy - cut)**2 + (xx - cut)**2)
    norm_ap = r_map <= (cut - 4)
    bg_ann = (r_map > (cut - 4)) & (r_map <= (cut - 1))

    def _norm_group(g):
        c = med_raw[g, y0:y1, x0:x1]
        if c.shape != (2*cut+1, 2*cut+1):
            return np.zeros((2*cut+1, 2*cut+1))
        c = c - np.median(c[bg_ann])
        s = c[norm_ap].sum()
        return c / s if s != 0 else c

    early_img = np.mean([_norm_group(g) for g in early_groups_loc], axis=0)
    late_img = np.mean([_norm_group(g) for g in late_groups_loc], axis=0)
    bfe_img = late_img - early_img
    bfe_img_scaled = bfe_img / 1e-4

    # Derive r_between (first zero crossing) and r_ring (peak after crossing)
    r_int = np.arange(0, cut)
    rr    = np.round(r_map).astype(int)
    rp    = np.array([np.nanmean(bfe_img[rr == ri]) for ri in r_int])

    r_zero = None
    sign0  = np.sign(rp[0]) if rp[0] != 0 else 1
    for i in range(1, len(rp)):
        if np.sign(rp[i]) != 0 and np.sign(rp[i]) != sign0:
            frac   = abs(rp[i-1]) / (abs(rp[i-1]) + abs(rp[i]) + 1e-30)
            r_zero = r_int[i-1] + frac
            break
    if r_zero is None:
        r_zero = cut // 3

    r_between = max(1, int(round(r_zero)))
    search_start = r_between + 1
    if search_start < len(rp):
        r_ring = search_start + int(np.argmax(rp[search_start:]))
    else:
        r_ring = r_between + 1

    sky_y = max(5, min(ny - 5, star_y - 15))
    sky_x = max(5, min(nx - 5, star_x + 10))

    loc_pixels = [
        (star_y, star_x, 'Star centre', 'C3', '*', 12),
        (star_y, star_x + r_between, rf'Between core \& ring ($r={r_between}$)', 'C0', 'o', 8),
        (star_y, star_x + r_ring,    rf'On ring ($r={r_ring}$)',                  'C1', 'X', 8),
        (sky_y, sky_x, 'Sky', 'C4', 'D', 6),
    ]

    plt.rcParams.update({'font.family': 'serif', 'text.usetex': True, 'font.size': 9})

    fig, ax = plt.subplots(figsize=(4.5, 4))
    vabs = np.nanpercentile(np.abs(bfe_img_scaled), 99.5)
    ext = [cx - cut - 0.5, cx + cut + 0.5, cy - cut - 0.5, cy + cut + 0.5]
    im = ax.imshow(bfe_img_scaled, origin='lower', cmap='RdBu_r',
                   vmin=-vabs, vmax=vabs, extent=ext)
    fig.colorbar(im, ax=ax, label=r'Norm.\ $\Delta$flux ($\times10^{-4}$)',
                 fraction=0.046, pad=0.04)

    for py, px, label, color, marker, ms in loc_pixels:
        ax.plot(px, py, marker=marker, color=color, ms=ms, mew=1.5,
                markeredgecolor='w', ls='none', zorder=5, label=label)

    ax.set_xlabel(r'$x$ (px)')
    ax.set_ylabel(r'$y$ (px)')
    ax.legend(fontsize=7, frameon=True, facecolor='0.15', labelcolor='white',
              loc='upper left')

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.rcParams.update({'font.family': 'sans-serif', 'text.usetex': False, 'font.size': 10})
    if verbose:
        print(f'Saved pixel location diagnostics to {save_path}')
    return r_between, r_ring


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
