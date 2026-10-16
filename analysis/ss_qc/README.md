# Secondary-standard (SS) calibration QC

Analysis scripts for checking how well the secondary-standard flux
calibration works, and when an SS should be skipped. All scripts only read
reduced data; none modifies it.

Interim findings (night 260904) are summarised below. The full write-up, with
figures, is in the shared interim report.

## Findings so far (260904)

- **No SS failures** on 260904: all 28 exposure/spectrograph pairs have
  `SECCOR=True`, S/N 24–37, and every star is 2–5″ inside its bundle.
- **The free Moffat β is the main source of SS flux error.** In the
  same-exposure AAOmega/Spector flux ratio, where transparency cancels, the
  pipeline extraction scatters by 6.6% (blue) and 8.7% (red) RMS between
  dithers. The error tracks the fitted β (Spearman ρ = +0.89), which swings
  between 1.5 and 6 on the 37-core SS bundles.
- **Fixing β at the right value helps; a wrong value hurts.** A Moffat refit
  with β = 2.5 (the night's median from free fits) gives 2.4% / 2.8%
  (normalisation) and 1.3% / 1.7% (spectral shape). β = 3.5 or 5 is worse
  than leaving β free, so a fixed β must be measured from the data.
- **Provisional edge threshold (simulations):** skip the SS when
  `SSEDGE` < 1.0″ for a 5% tolerance (< 1.25–1.75″ for 2%). `SSEDGE` in arcsec
  is more stable than `SSEDGEF` (in FWHM units). No 260904 frame would be
  skipped at 5%.

Caveats: one night; a simplified single-band refit without the chromatic
(DAR/CvD) model; simulated rather than real edge stars. The pipeline's own SS
step has not been re-run (it needs the tiles' robot files).

## Quality metrics

Computed by `dr/ss_quality.py` and written to the `FLUX_CALIBRATION` header by
`telluric.extract_secondary_standard`. None is used to skip frames yet; after
testing on real data, one (or two) will be kept.

| Key | Meaning |
| --- | --- |
| `SSEDGE` | Distance (arcsec) from the fitted centre to the bundle footprint edge (hull of good fibres grown by the core radius); > 0 inside, < 0 outside |
| `SSEDGEF` | `SSEDGE` / fitted FWHM |
| `SSPHI` | Direction of the star: 0° towards a bundle corner, 30° towards the middle of a side |
| `SSOUTER` | Fraction of the observed bundle flux in the outermost fibres |
| `SSASYM`, `SSASYMR` | Fitted-minus-light-weighted centre offset, and its radial part |
| `SSCHI2`, `SSNFIB` | PSF-fit reduced pseudo-χ²; number of good fibres |
| `SSEXSUM` | Extracted / summed bundle flux |
| `SSBETAFX` | True if β was held fixed (`ss_beta`) |

## Scripts

Run from this directory with the pipeline on the path, for example:

```bash
export PYTHONPATH=~/hector-pipeline
QC=~/hector/260904_ss_qc
python ss_census.py ~/hector/260904 --out-dir $QC
```

| Script | Input | Output |
| --- | --- | --- |
| `ss_census.py` | reduced date dir(s) or manager root(s) | `ss_census_{frames,exposures,fields}.csv`: SS outcome per frame (`ok`, `fail_low_snr`, `fail_no_tf`, `ss_fit_crash`, `short_exp`, …), and fields with a mix of good and failed dithers |
| `ss_metrics_existing.py` | `ss_census_frames.csv` | `ss_metrics_existing.csv`: the quality metrics for frames reduced before they existed, from the stored PSF fit |
| `ss_dither_consistency.py` | QC dir | `fig5`: per-dither `RESCALE` residuals, same-exposure spectrograph ratio, correlations |
| `ss_refit.py` | QC dir | `ss_refit*.csv`, `fig6`: free vs fixed-β refits ranked by dither-to-dither consistency |
| `ss_refit_chromatic.py` | QC dir | `fig8`: the same in 100 Å chunks (spectral shape) |
| `ss_edge_sim.py` | QC dir | `ss_edge_sim.csv`: simulated SS stars moved past the bundle edge (real H/U layouts, 260904 flux and noise); ~15 min |
| `ss_edge_thresholds.py` | QC dir | `ss_edge_*.csv`, `fig7`: provisional skip thresholds |
| `plot_ss_quality_tests.py` | — | `fig1`–`fig4`: the metrics on synthetic test stars |

Figures go to `figures/` (not tracked in git).

## Edge thresholds (fig 7)

For each simulated frame, the *extra error from position* is the flux error
minus the same method's mean error for a centred star with the same true
seeing and β. That removes the bias from assuming the wrong profile, which a
position threshold cannot fix.

The onset cut: bin the frames by the metric and, starting from the safe end
(star well inside the bundle), step towards the edge. The cut is placed at the
first bin in which fewer than 68% of frames have an extra error within the
tolerance. Frames on the unsafe side of the cut would be skipped. The dashed
line in fig 7 is this cut for the β = 2.5 fit at 2% tolerance.

## Tests

```bash
python -m pytest tests/test_ss_quality.py
```

Synthetic Moffat stars on the real 37-core H and U layouts from
`utils/Fibre_slitInfo_*.csv`.
