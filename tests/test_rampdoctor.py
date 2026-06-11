import numpy as np
import pytest

from rampdoctor import (RampDoctor, build_correction_map, correct_ramp,
                        fit_bfe_params)


def make_cube(n_int=6, n_groups=10, ny=50, nx=50, star=True, seed=0):
    """Synthetic ramp cube: flat rate + exponential reset decay + noise."""
    rng = np.random.default_rng(seed)
    rate = np.full((ny, nx), 10.0)
    if star:
        rate[ny // 2, nx // 2] = 5000.0
    g = np.arange(n_groups - 1)
    decay = 3.0 * np.exp(-g / 1.5)
    grads = (rate[None, None] + decay[None, :, None, None]
             + rng.normal(0, 0.5, (n_int, n_groups - 1, ny, nx)))
    cube = np.concatenate(
        [np.zeros((n_int, 1, ny, nx)), np.cumsum(grads, axis=1)], axis=1)
    return cube


def make_bfe_cube(A_bfe=1e-6, alpha=2.797, n_int=4, n_groups=8,
                  ny=120, nx=120, seed=1):
    """Synthetic cube with a bright Gaussian star and a BFE imprint from
    the pixel-area forward model grad_obs = tg * (1 - A * K convolved Q)."""
    from scipy.signal import fftconvolve
    rng = np.random.default_rng(seed)

    yy, xx = np.mgrid[:ny, :nx]
    r2 = (yy - ny // 2)**2 + (xx - nx // 2)**2
    rate = 10.0 + 30000.0 * np.exp(-r2 / (2 * 2.0**2))
    g = np.arange(n_groups - 1)
    decay = 3.0 * np.exp(-g / 1.5)

    kh = 20
    ii, jj = np.mgrid[-kh:kh + 1, -kh:kh + 1].astype(float)
    rk = np.sqrt(ii**2 + jj**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        K = np.where(rk > 0, -1.0 / rk**alpha, 0.0)
    K[kh, kh] = -K.sum()

    Q = np.zeros((ny, nx))
    grads_obs = np.zeros((n_groups - 1, ny, nx))
    for gi in range(n_groups - 1):
        tg = rate + decay[gi]
        KQ = fftconvolve(Q, K, mode='same')
        grads_obs[gi] = tg * (1.0 - A_bfe * KQ)
        Q += tg

    grads = (grads_obs[None]
             + rng.normal(0, 0.3, (n_int, n_groups - 1, ny, nx)))
    cube = np.concatenate(
        [np.zeros((n_int, 1, ny, nx)), np.cumsum(grads, axis=1)], axis=1)
    return cube


class TestInit:
    def test_requires_data(self):
        with pytest.raises(ValueError):
            RampDoctor()

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError):
            RampDoctor(cube=np.zeros((5, 10, 10)))

    def test_shape_property(self):
        cube = make_cube()
        rd = RampDoctor(cube=cube)
        assert rd.shape == cube.shape

    def test_grads_property(self):
        cube = make_cube()
        rd = RampDoctor(cube=cube)
        assert rd.grads.shape == (6, 9, 50, 50)

    def test_grads_cor_requires_correct(self):
        rd = RampDoctor(cube=make_cube())
        with pytest.raises(RuntimeError):
            rd.grads_cor


class TestCorrect:
    def test_output_shape(self):
        cube = make_cube()
        rd = RampDoctor(cube=cube)
        cor = rd.correct()
        assert cor.shape == cube.shape
        assert rd.cube_cor is cor

    def test_group0_unchanged(self):
        cube = make_cube()
        rd = RampDoctor(cube=cube)
        cor = rd.correct()
        assert np.array_equal(cor[:, 0], cube[:, 0])

    def test_flattens_background_profile(self):
        cube = make_cube(star=False)
        rd = RampDoctor(cube=cube)
        rd.correct()
        n_grads = cube.shape[1] - 2
        prof_raw = np.median(rd.grads[:, :n_grads], axis=0).mean(axis=(1, 2))
        prof_cor = np.median(rd.grads_cor[:, :n_grads], axis=0).mean(axis=(1, 2))
        assert np.ptp(prof_cor) < 0.1 * np.ptp(prof_raw)

    def test_diagnostics_file(self, tmp_path):
        pytest.importorskip('matplotlib')
        out = tmp_path / 'diag.png'
        rd = RampDoctor(cube=make_cube())
        rd.correct(diagnostics=True, save_path=out)
        assert out.exists()


class TestResetDecay:
    @pytest.mark.parametrize('method', ['median', 'per_int'])
    def test_methods(self, method):
        cube = make_cube(star=False)
        rd = RampDoctor(cube=cube)
        cor = rd.correct_reset_decay(method=method)
        assert cor.shape == cube.shape
        n_grads = cube.shape[1] - 2
        prof_cor = np.median(np.diff(cor, axis=1)[:, :n_grads],
                             axis=0).mean(axis=(1, 2))
        assert np.ptp(prof_cor) < 0.5


class TestCorrectionMap:
    def test_roundtrip(self):
        cube = make_cube(star=False)
        rd = RampDoctor(cube=cube)
        C_map = rd.build_correction_map()
        assert C_map.shape == (cube.shape[1] - 1, 50, 50)
        grads_cor = rd.correct_ramp(C_map)
        assert grads_cor.shape == (6, 9, 50, 50)
        # corrected gradients should be on a common scale across groups
        prof = np.median(grads_cor, axis=0).mean(axis=(1, 2))
        assert np.ptp(prof[:-1]) < 0.5


class TestSave:
    def test_save_from_cube(self, tmp_path):
        fits = pytest.importorskip('astropy.io.fits')
        rd = RampDoctor(cube=make_cube())
        rd.correct()
        out = tmp_path / 'cor.fits'
        rd.save(out)
        with fits.open(out) as hdul:
            assert hdul[0].data.shape == rd.shape

    def test_save_requires_correct(self, tmp_path):
        rd = RampDoctor(cube=make_cube())
        with pytest.raises(RuntimeError):
            rd.save(tmp_path / 'cor.fits')


class TestFitBFE:
    def test_recovers_amplitude(self):
        pytest.importorskip('sep')
        A_true = 1e-6
        cube = make_bfe_cube(A_bfe=A_true)
        rd = RampDoctor(cube=cube)
        A_fit = rd.fit_bfe()
        assert A_fit is not None
        assert 0.3 < A_fit / A_true < 3.0
        assert abs(rd.star_x - 60) <= 1
        assert abs(rd.star_y - 60) <= 1

    def test_faint_field_returns_none(self):
        pytest.importorskip('sep')
        cube = make_cube(star=False)
        rd = RampDoctor(cube=cube, A_bfe=1.5e-6)
        result = fit_bfe_params(cube)
        assert result[0] is None
        # class keeps prior parameters when the fit is skipped
        assert rd.fit_bfe() is None
        assert rd.A_bfe == 1.5e-6

    def test_diagnostics_file(self, tmp_path):
        pytest.importorskip('sep')
        pytest.importorskip('matplotlib')
        out = tmp_path / 'bfe_diag.png'
        rd = RampDoctor(cube=make_bfe_cube())
        rd.fit_bfe(diagnostics=True, save_path=out)
        assert out.exists()
