import numpy as np
from pathlib import Path

from .ramp_correction import (build_correction_map, correct_reset_decay,
                              fit_bfe_params, fit_migration_params,
                              correct_bfe_rcd, correct_ramp)


class RampDoctor:
    """
    Joint BFE + reset-decay correction for JWST up-the-ramp data.

    Wraps the correction pipeline in a stateful interface: load a ramp cube
    (or uncal FITS file), optionally fit the BFE amplitude from the brightest
    source, then apply the correction.

    Parameters
    ----------
    cube : ndarray (n_int, n_groups, ny, nx), optional
        Raw SCI data. Either cube or file must be given.
    file : str or Path, optional
        Path to an uncal FITS file. The SCI extension is loaded, and
        GROUPDQ if present.
    dq : ndarray (n_int, n_groups, ny, nx), optional
        GROUPDQ array. Overrides any DQ loaded from file.
    A_bfe : float
        BFE kernel amplitude (default 1.035e-6).
    alpha_bfe : float
        BFE kernel power-law index (default 3.43, bright-star consensus).
    b_bfe, c_bfe : float
        Quadratic-numerator coefficients of K = -(1 + b r + c r^2)/r^alpha
        (defaults -0.50, 0.056). Overwritten by fit_bfe().
    bg_mask : ndarray (ny, nx) bool, optional
        True = background pixels for the global tau fit.
    sci_mask : ndarray (ny, nx) bool, optional
        True = good science pixels, used during source detection.
    verbose : bool

    Examples
    --------
    >>> from rampdoctor import RampDoctor
    >>> rd = RampDoctor(file='jw_mirimage_uncal.fits', verbose=True)
    >>> rd.fit_bfe()
    >>> cube_cor = rd.correct()
    """

    def __init__(self, cube=None, file=None, dq=None, method='migration',
                 M_mig=4.2e-7, M_mig_y=1, thr_mig=37.2, A_bfe=1.035e-6,
                 alpha_bfe=3.43, b_bfe=-0.50, c_bfe=0.056,
                 bg_mask=None, sci_mask=None, verbose=False):
        self.file = Path(file) if file is not None else None
        if file is not None:
            from astropy.io import fits
            with fits.open(self.file) as hdul:
                cube = hdul['SCI'].data.astype(float)
                if dq is None and 'GROUPDQ' in hdul:
                    dq = hdul['GROUPDQ'].data
        if cube is None:
            raise ValueError('Provide either cube or file.')

        self.cube = np.asarray(cube, dtype=float)
        if self.cube.ndim != 4:
            raise ValueError('cube must have shape (n_int, n_groups, ny, nx).')
        self.dq = dq
        self.method = method
        self.M_mig = M_mig
        self.M_mig_y = M_mig_y
        self.thr_mig = thr_mig
        self.A_bfe = A_bfe
        self.alpha_bfe = alpha_bfe
        self.b_bfe = b_bfe
        self.c_bfe = c_bfe
        self.bg_mask = bg_mask
        self.sci_mask = sci_mask
        self.verbose = verbose

        self.cube_cor = None
        self.star_x = None
        self.star_y = None

    @property
    def shape(self):
        return self.cube.shape

    @property
    def grads(self):
        """Raw group-to-group gradients (n_int, n_groups-1, ny, nx)."""
        return np.diff(self.cube, axis=1)

    @property
    def grads_cor(self):
        """Corrected gradients; requires correct() to have been run."""
        if self.cube_cor is None:
            raise RuntimeError('Run correct() first.')
        return np.diff(self.cube_cor, axis=1)

    def fit_bfe(self, fit_alpha=False, bfe_early_groups=None,
                bfe_late_groups=None, ap_radius=5, cut=20, fit_r=None,
                diagnostics=False, save_path=None):
        """
        Fit the BFE parameters from the brightest source, using the configured
        ``method``. For method='migration' this fits the migration strength M
        and force threshold; for method='kernel' the kernel amplitude (and
        optionally alpha). Results are stored on self and used by correct().

        Returns
        -------
        float or None
            The fitted amplitude (kernel) or migration strength M, or None if
            no source met the brightness threshold.
        """
        if self.method == 'migration':
            Mx, My, thr, sx, sy = fit_migration_params(
                self.cube, M_init=self.M_mig, thr_init=self.thr_mig,
                bg_mask=self.bg_mask, sci_mask=self.sci_mask,
                bfe_early_groups=bfe_early_groups, bfe_late_groups=bfe_late_groups,
                ap_radius=ap_radius, cut=cut, fit_r=fit_r, verbose=self.verbose,
                aniso=(self.M_mig_y is None),
                diagnostics=diagnostics, save_path=save_path)
            if Mx is None:
                return None
            self.M_mig, self.thr_mig = Mx, thr
            if self.M_mig_y is None:
                self.M_mig_y = My / Mx
            self.star_x, self.star_y = sx, sy
            return Mx

        alpha = None if fit_alpha else self.alpha_bfe
        result = fit_bfe_params(
            self.cube, alpha_bfe=alpha, bg_mask=self.bg_mask,
            sci_mask=self.sci_mask, bfe_early_groups=bfe_early_groups,
            bfe_late_groups=bfe_late_groups, ap_radius=ap_radius,
            cut=cut, fit_r=fit_r, verbose=self.verbose,
            diagnostics=diagnostics, save_path=save_path)

        A_fit, alpha_fit, b_fit, c_fit, sx, sy = result

        if A_fit is None:
            return None
        self.A_bfe = A_fit
        self.alpha_bfe = alpha_fit
        self.b_bfe = b_fit
        self.c_bfe = c_fit
        self.star_x = sx
        self.star_y = sy
        return A_fit

    def correct(self, fit_bfe=False, late_groups=None, diagnostics=False,
                save_path=None, **kwargs):
        """
        Apply the joint BFE + reset-decay correction.

        Parameters
        ----------
        fit_bfe : bool
            If True, fit A_bfe from the brightest source before correcting
            (equivalent to calling fit_bfe() first).
        late_groups : list of int, optional
            Gradient indices for the flat-rate estimate in the median
            subtraction step. Defaults to the last three good gradients.
        diagnostics : bool
            If True, save a correction diagnostics figure (tau fit, decay
            amplitude map, BFE step size, background profile by stage;
            default 'bfe_rcd_diagnostics.png'). When fit_bfe=True a BFE
            fit diagnostics figure is also saved.
        save_path : str or Path, optional
            File path for the correction diagnostics figure.
        **kwargs
            Passed through to correct_bfe_rcd.

        Returns
        -------
        cube_cor : ndarray (n_int, n_groups, ny, nx)
            Corrected SCI cube; also stored as self.cube_cor.
        """
        if fit_bfe:
            self.fit_bfe(diagnostics=diagnostics)
        self.cube_cor = correct_bfe_rcd(
            self.cube, method=self.method,
            M_mig=self.M_mig, M_mig_y=self.M_mig_y, thr_mig=self.thr_mig,
            A_bfe=self.A_bfe, alpha_bfe=self.alpha_bfe,
            b_bfe=self.b_bfe, c_bfe=self.c_bfe,
            bg_mask=self.bg_mask, late_groups=late_groups,
            sci_mask=self.sci_mask, verbose=self.verbose,
            star_x=self.star_x, star_y=self.star_y,
            diagnostics=diagnostics, save_path=save_path, **kwargs)
        return self.cube_cor

    def correct_reset_decay(self, method='median', **kwargs):
        """
        Apply the reset-decay-only correction (no BFE step).
        See ramp_correction.correct_reset_decay for methods and options.
        """
        kwargs.setdefault('dq', self.dq)
        if self.sci_mask is not None:
            kwargs.setdefault('mask', ~self.sci_mask)
        self.cube_cor = correct_reset_decay(self.cube, method=method, **kwargs)
        return self.cube_cor

    def build_correction_map(self, mask=None):
        """Per-pixel multiplicative group correction map from this cube."""
        return build_correction_map(self.cube, mask=mask)

    def correct_ramp(self, C_map):
        """Apply a correction map to this cube; returns corrected gradients."""
        return correct_ramp(self.cube, C_map)

    def save(self, path, overwrite=False):
        """
        Write the corrected cube to a FITS file. If the object was created
        from a file, the original HDU structure is preserved with SCI
        replaced; otherwise a minimal FITS file is written.
        """
        if self.cube_cor is None:
            raise RuntimeError('Run correct() first.')
        from astropy.io import fits
        path = Path(path)
        if self.file is not None:
            with fits.open(self.file) as hdul:
                hdul['SCI'].data = self.cube_cor.astype(np.float32)
                hdul.writeto(path, overwrite=overwrite)
        else:
            fits.PrimaryHDU(self.cube_cor.astype(np.float32)).writeto(
                path, overwrite=overwrite)
        if self.verbose:
            print(f'Saved corrected cube to {path}')
