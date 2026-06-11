# rampdoctor

Brighter-fatter effect (BFE) and reset charge decay (RCD) corrections for
JWST up-the-ramp data, developed and validated on MIRI imaging ramps.

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
