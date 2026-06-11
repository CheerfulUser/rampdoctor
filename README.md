# rampdoctor

[![tests](https://github.com/CheerfulUser/rampdoctor/actions/workflows/tests.yml/badge.svg)](https://github.com/CheerfulUser/rampdoctor/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)

`rampdoctor` corrects two detector systematics in JWST up-the-ramp data at
the group level, before ramp fitting. The **brighter-fatter effect (BFE)**
redistributes charge from bright pixels into their neighbours as the wells
fill, broadening the PSF over the course of an integration and biasing
aperture photometry built from group differences. **Reset charge decay
(RCD)** adds an exponentially decaying signal to the first few groups after
each detector reset, imprinting a common ramp-shaped systematic on every
integration. Both effects are fit directly from the science data — no
reference files are required: the BFE amplitude is fit from the
brighter-fatter signature of the brightest source in the field, and the
decay timescale from background pixels. Applying the joint correction to a
JWST MIRI imaging time series of EV Lac reduces the group-level lightcurve
RMS from 2.48% to 0.15%. The method operates on `uncal` SCI cubes
(`n_int, n_groups, ny, nx`) and is developed and validated on MIRI, but the
approach applies to any instrument with non-destructive up-the-ramp
sampling.

## Installation

```bash
pip install -e .
```

## Usage

```python
from rampdoctor import RampDoctor

rd = RampDoctor(file='jw_mirimage_uncal.fits', verbose=True)
rd.fit_bfe()              # fit A_bfe from the brightest source (optional)
cube_cor = rd.correct()   # joint BFE + RCD correction
rd.save('corrected_uncal.fits')
```

Pass `diagnostics=True` to `fit_bfe()` or `correct()` to save diagnostic
figures (PSF-difference fit quality, global tau fit, decay amplitude map,
BFE step size, and background profiles by correction stage). Requires
matplotlib (`pip install rampdoctor[plot]`).

Or with an array already in memory:

```python
rd = RampDoctor(cube=cube, bg_mask=bg_mask)
cube_cor = rd.correct()
```

The underlying functions (`correct_bfe_rcd`, `fit_bfe_params`,
`correct_reset_decay`, `build_correction_map`, `correct_ramp`) are also
importable directly.

## Method

The joint correction applies three sequential steps to the group-to-group
gradients:

1. **Causal BFE inversion** — iterative, flux-conserving inversion of the
   pixel-area forward model `grad_obs = true_grad (1 - A K⊛Q)`, where `Q` is
   the accumulated charge and `K` a power-law kernel.
2. **Parametric RCD subtraction** — global decay timescale fitted from
   background pixels; per-pixel amplitude, rate, and first-frame offset via
   linear least squares.
3. **Non-parametric residual removal** — per-pixel, per-group median
   subtraction with the flat rate restored from late groups.

## Example diagnostics

BFE fit on a JWST MIRI observation of EV Lac (`rd.fit_bfe(diagnostics=True)`):
the observed late−early PSF difference, the best-fit forward model, the
residual, and the radial profiles.

![BFE fit diagnostics](figs/bfe_fit_diagnostics.png)

Correction diagnostics (`rd.correct(diagnostics=True)`): the global decay
timescale fit, the per-pixel decay amplitude map, the size of the BFE
correction per group, and the background gradient profile at each stage of
the correction.

![Correction diagnostics](figs/bfe_rcd_diagnostics.png)

## Tests

```bash
pip install -e .[test]
pytest tests/
```
