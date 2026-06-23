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
integration.

Both effects are fit directly from the science data — no reference files are
required. The BFE amplitude and threshold are fit from the brighter-fatter
signature of the brightest source in the field; the decay timescale is fit
from background pixels. On JWST MIRI time-series observations the joint
correction reduces group-level lightcurve RMS by 3–10×.

The package operates on raw (`uncal`) or pipeline-processed (`ramp`) SCI
cubes of shape `(n_int, n_groups, ny, nx)`. It is developed and validated on
MIRI but the approach applies to any instrument with non-destructive
up-the-ramp sampling.

## Installation

```bash
pip install -e .
```

## Quick start

```python
from rampdoctor.ramp_correction import correct_bfe_rcd

# Pipeline-processed (ramp.fits) data — linearity already applied
cube_cor = correct_bfe_rcd(cube, fit_bfe=True, verbose=True)

# Raw uncal data — jointly fit detector linearity with BFE parameters
cube_cor = correct_bfe_rcd(cube, fit_bfe=True, fit_a_lin=True, verbose=True)
```

Pass `diagnostics=True` and a `save_path` to produce diagnostic figures
(PSF-difference fit quality, migration model residual, pixel ramp profiles,
and background gradient profiles at each correction stage).

Or use the object-oriented interface:

```python
from rampdoctor import RampDoctor

rd = RampDoctor(file='jw_mirimage_uncal.fits', verbose=True)
rd.fit_bfe()
cube_cor = rd.correct()
rd.save('corrected.fits')
```

## Method

The correction applies two sequential steps to the group-to-group gradients:

**Step 1 — BFE inversion (charge-migration model)**

The forward model per group is:

```
dQ = F_true / poly'(Q) + A_crd * exp(-g / tau)
Q  = _mig_group(Q + dQ, M, threshold)
```

where `F_true` is the true photon flux rate, `poly'(Q) = 1 + 2aQ` is the
detector linearity derivative (fitted jointly on uncal data), `A_crd` is the
per-pixel RCD amplitude, `M` is the charge-migration strength, and
`threshold` is an activation level below which migration is negligible.
Migration-free gradients are recovered by Born iteration on `F_true`.
Calibrated defaults (`M = 4.2×10⁻⁷`, `threshold = 37.2 DN`) are derived
from fits to linearised MIRI ramp data across two targets.

**Step 2 — RCD subtraction**

The global decay timescale `tau` is fitted from background pixels (groups
≥ 1). Per-pixel amplitude `A_crd`, flat rate, and first-frame offset are
then fitted via linear least squares. The decay `A_crd · exp(−g/τ)` is
subtracted from every gradient of every integration.

**Optional: BFE parameter fitting**

When `fit_bfe=True`, `fit_migration_params` detects the brightest source via
SEP and fits `M` and `threshold` (optionally also the linearity coefficient
`a`) by minimising the residual between the observed and modelled
late−early PSF difference image. An iterative RCD re-fit prevents the decay
and migration parameters from absorbing each other.

## Validation

| Target | Data | M (fit) | Threshold | RMS before | RMS after |
|---|---|---|---|---|---|
| Wolf 359 | ramp.fits | 4.32×10⁻⁷ | 37.8 DN | — | — |
| TRAPPIST-1 | ramp.fits | 4.09×10⁻⁷ | 36.6 DN | 0.80% | 0.22% |

The consistent `M` and threshold across two independent MIRI targets
(observed in different modes) support their use as instrument-level
defaults. On raw (`uncal`) data, jointly fitting the linearity coefficient
`a` alongside `M` reduces chi²/N from 11.6 to 5.5 for Wolf 359.

## Example diagnostics

BFE fit on a JWST MIRI observation (`fit_bfe=True, diagnostics=True`):
observed late−early PSF difference, best-fit migration model, residual,
and radial profiles.

![BFE fit diagnostics](figs/bfe_fit_diagnostics.png)

Correction diagnostics (`diagnostics=True`): the global decay timescale
fit, per-pixel decay amplitude map, BFE correction per group, and background
gradient profiles at each stage.

![Correction diagnostics](figs/bfe_rcd_diagnostics.png)

## Tests

```bash
pip install -e .[test]
pytest tests/
```
