"""
Refit the secondary standard (SS) stars with several extraction methods and
compare how consistent the resulting fluxes are between dithers.

Motivation: on 260904 the same-exposure AAOmega/Spector RESCALE ratio, in
which transparency cancels, scatters by 4-7% between dithers and tracks the
fitted Moffat beta.  The bundle (~5" radius) does not constrain the PSF
wings, but the wings set how much flux the model extrapolates.  This script
tests whether constraining the profile shape (fixing beta) reduces that
scatter.

For each exposure and spectrograph, per-fibre fluxes are summed over a band
common to both spectrographs (from the primary-calibrated fcal.fits) and
fitted with each method:

  free         Moffat, alpha and beta free (as in the pipeline)
  beta2.5/3.5/5  Moffat, beta fixed
  gauss        Gaussian (beta -> infinity)

The figure of merit is the robust scatter, across the dithers of a field,
of F(AAOmega)/F(Spector) from the same exposure.  Transparency cancels in
that ratio, so its scatter measures extraction error.

The fit is to a single band image: no chromatic (DAR/CvD) model, so in
absolute terms it is not the pipeline fit, but every method sees the same
data.  The 'free' method's scatter is compared with the pipeline RESCALE as
a check.

Read-only.  Usage:
    python ss_refit.py <qc_dir>      (needs ss_census_frames.csv there)
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import astropy.io.fits as pf
from scipy.optimize import least_squares

from hector.config import fibre_diameter_arcsec

warnings.simplefilter('ignore')

R_CORE = fibre_diameter_arcsec / 2.0
BANDS = {
    'blue': {'ccd': {'AAOmega': '1', 'Spector': '3'},
             'range': (4400.0, 5400.0), 'mask': []},
    'red': {'ccd': {'AAOmega': '2', 'Spector': '4'},
            'range': (6300.0, 7300.0), 'mask': [(6860.0, 6930.0)]},
}
METHODS = ['free', 'beta2.5', 'beta3.5', 'beta5', 'gauss']

# sub-sampling of a fibre core for the model integral
_g = np.linspace(-R_CORE, R_CORE, 11)
_gx, _gy = np.meshgrid(_g, _g)
_in = _gx**2 + _gy**2 <= R_CORE**2
SUB_DX, SUB_DY = _gx[_in], _gy[_in]
SUB_AREA = (_g[1] - _g[0])**2


def profile(r2, fwhm, beta):
    """Normalised (unit total flux) circular Moffat or Gaussian surface
    brightness at squared radius r2."""
    if beta is None:  # Gaussian
        sigma = fwhm / 2.3548
        return np.exp(-r2 / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    alpha = fwhm / (2.0 * np.sqrt(2.0**(1.0 / beta) - 1.0))
    return (beta - 1.0) / (np.pi * alpha**2) * (1 + r2 / alpha**2)**(-beta)


def fibre_fractions(x0, y0, fwhm, beta, xf, yf):
    """Fraction of the star's total flux landing on each fibre core."""
    dx = xf[:, None] + SUB_DX[None, :] - x0
    dy = yf[:, None] + SUB_DY[None, :] - y0
    return profile(dx**2 + dy**2, fwhm, beta).sum(axis=1) * SUB_AREA


def read_band(path, probenum, wrange, mask):
    """Per-fibre band flux and variance for the SS bundle in one arm."""
    with pf.open(path) as hdul:
        hdr = hdul[0].header
        n = hdr['NAXIS1']
        lam = hdr['CRVAL1'] + (np.arange(n) + 1 - hdr['CRPIX1']) * hdr['CDELT1']
        use = (lam >= wrange[0]) & (lam <= wrange[1])
        for lo, hi in mask:
            use &= ~((lam >= lo) & (lam <= hi))
        table = hdul['FIBRES_IFU'].data
        good = (table['PROBENUM'] == probenum) & (table['TYPE'] == 'P')
        dec = np.mean(table['FIB_MDEC'][good])
        x = table['XPOS'][good] * np.cos(np.deg2rad(dec))
        y = table['YPOS'][good].astype(float)
        idx = table['SPEC_ID'][good] - 1
        data = hdul[0].data[idx][:, use]
        var = hdul['VARIANCE'].data[idx][:, use]
        ok = np.isfinite(data) & np.isfinite(var)
        # mean flux density over the band, per fibre
        npix = ok.sum(axis=1)
        flux = np.where(ok, data, 0).sum(axis=1) / npix
        fvar = np.where(ok, var, 0).sum(axis=1) / npix**2
        # pipeline extraction over the same pixels, for the check
        pipe = hdul['FLUX_CALIBRATION'].data[0][use]
        pipe = np.nanmean(pipe)
    x = x.astype(float)
    return x - x.mean(), y - y.mean(), flux, fvar, pipe


def fit(x, y, flux, var, beta, beta_free=False):
    """Least-squares fit of total flux, centre, FWHM (and beta), plus a
    constant background per fibre.  Returns dict of parameters."""
    sigma = np.sqrt(var + (0.02 * np.clip(flux, 0, None))**2)
    tot0 = max(flux.sum(), 1e-3)
    w = np.clip(flux, 0, None)
    x0 = np.sum(w * x) / w.sum()
    y0 = np.sum(w * y) / w.sum()
    p0 = [tot0, x0, y0, 1.5, 0.0]
    lo = [0, x0 - 6, y0 - 6, 0.3, -np.inf]
    hi = [np.inf, x0 + 6, y0 + 6, 6.0, np.inf]
    if beta_free:
        p0.append(3.0)
        lo.append(1.05)
        hi.append(20.0)

    def resid(p):
        b = p[5] if beta_free else beta
        model = p[0] * fibre_fractions(p[1], p[2], p[3], b, x, y) + p[4]
        return (flux - model) / sigma

    sol = least_squares(resid, p0, bounds=(lo, hi), x_scale='jac')
    p = sol.x
    out = {'F': p[0], 'x0': p[1], 'y0': p[2], 'fwhm': p[3], 'bkg': p[4],
           'beta': p[5] if beta_free else beta,
           'chi2': np.sum(sol.fun**2) / max(len(flux) - len(p), 1)}
    frac = fibre_fractions(p[1], p[2], p[3], out['beta'], x, y).sum()
    out['frac_on_fibres'] = frac
    return out


def extract(x, y, flux, var):
    """Total flux from each method."""
    res = {}
    f = fit(x, y, flux, var, None, beta_free=True)
    res['free'] = f
    for b in (2.5, 3.5, 5.0):
        res['beta%g' % b] = fit(x, y, flux, var, b)
    res['gauss'] = fit(x, y, flux, var, None)
    return res


def robust_sigma(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def run(qc_dir):
    frames = pd.read_csv(os.path.join(qc_dir, 'ss_census_frames.csv'))
    frames = frames[(~frames['MNGRSPMS']) & frames['has_fluxcal_ext']]
    rows = []
    for band, info in BANDS.items():
        ccds = {'ccd_' + c: spec for spec, c in info['ccd'].items()}
        sel = frames[frames['ccd'].isin(ccds)]
        for r in sel.itertuples():
            fcal = r.path_red.replace('red.fits', 'fcal.fits')
            x, y, flux, var, pipe = read_band(fcal, r.PROBENUM, info['range'],
                                              info['mask'])
            res = extract(x, y, flux, var)
            for method, f in res.items():
                rows.append(dict(
                    band=band, method=method, fileroot=r.fileroot,
                    exposure_id=r.exposure_id, field_id=r.field_id,
                    spectrograph=r.spectrograph, F=f['F'],
                    fwhm=f.get('fwhm'), beta=f.get('beta'),
                    x0=f.get('x0'), y0=f.get('y0'), chi2=f.get('chi2'),
                    frac_on_fibres=f.get('frac_on_fibres'),
                    F_pipeline=pipe, pipe_beta=r.BETA, pipe_fwhm=r.FWHM,
                    RESCALE=r.RESCALE))
            print('%s %s done' % (band, r.fileroot), end='\r')
    return pd.DataFrame(rows)


def summarise(df):
    """Per method and band: robust scatter of the same-exposure ratio and of
    the per-spectrograph residuals."""
    out = []
    for (band, method), g in df.groupby(['band', 'method']):
        p = g.pivot_table(index=['field_id', 'exposure_id'],
                          columns='spectrograph', values='F').dropna()
        lr = 100 * np.log(p['AAOmega'] / p['Spector'])
        lr = lr - lr.groupby(level=0).transform('median')
        per = 100 * np.log(g['F'] / g.groupby(['field_id', 'spectrograph'])
                           ['F'].transform('median'))
        rec = dict(band=band, method=method,
                   ratio_scatter=robust_sigma(lr),
                   ratio_rms=np.sqrt(np.mean(lr**2)),
                   per_frame_scatter=robust_sigma(per))
        for field, v in lr.groupby(level=0):
            rec['ratio_scatter_' + field.split('_')[0]] = robust_sigma(v)
        out.append(rec)
    s = pd.DataFrame(out)
    order = {m: i for i, m in enumerate(METHODS)}
    return s.sort_values(['band', 'method'], key=lambda c: c.map(order)
                         if c.name == 'method' else c)


def ratio_resid(g, col='F'):
    """Same-exposure log ratio AAOmega/Spector, % from the field median."""
    p = g.pivot_table(index=['field_id', 'exposure_id'],
                      columns='spectrograph', values=col).dropna()
    lr = 100 * np.log(p['AAOmega'] / p['Spector'])
    return lr - lr.groupby(level=0).transform('median')


def bootstrap_rms(lr, n=2000, seed=0):
    """RMS of lr and its 68% bootstrap interval, resampling within fields."""
    rng = np.random.default_rng(seed)
    v = lr.values
    fields = lr.index.get_level_values(0)
    groups = [np.where(fields == f)[0] for f in np.unique(fields)]
    s = [np.sqrt(np.mean(v[np.concatenate(
        [rng.choice(g, len(g)) for g in groups])]**2)) for _ in range(n)]
    return np.sqrt(np.mean(v**2)), np.percentile(s, 16), np.percentile(s, 84)


def make_figure(df, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plot_ss_quality_tests import SERIES, INK, INK2, MUTED, GRID

    labels = {'PIPELINE': 'pipeline extraction (as reduced)',
              'free': 'Moffat fit, beta free',
              'beta2.5': 'Moffat fit, beta = 2.5',
              'beta3.5': 'Moffat fit, beta = 3.5',
              'beta5': 'Moffat fit, beta = 5',
              'gauss': 'Gaussian fit'}
    order = ['PIPELINE', 'free', 'beta2.5', 'beta3.5', 'beta5', 'gauss']
    band_colour = {'blue': SERIES[0], 'red': SERIES[1]}

    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1])
    ax = fig.add_subplot(gs[:, 0])
    ypos = np.arange(len(order))[::-1]
    for k, band in enumerate(('blue', 'red')):
        g = df[df['band'] == band]
        for y, m in zip(ypos, order):
            lr = (ratio_resid(g[g['method'] == 'free'], 'F_pipeline')
                  if m == 'PIPELINE' else ratio_resid(g[g['method'] == m]))
            rms, lo, hi = bootstrap_rms(lr)
            yy = y + (0.15 if k == 0 else -0.15)
            ax.errorbar(rms, yy, xerr=[[rms - lo], [hi - rms]], fmt='o',
                        color=band_colour[band], ms=7, capsize=0, lw=2,
                        label='%s band %g-%g A' % (
                            band, *BANDS[band]['range']) if y == ypos[0]
                        else None)
    ax.set_yticks(ypos)
    ax.set_yticklabels([labels[m] for m in order])
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 3, 5, 10, 20, 30])
    ax.set_xticklabels(['1', '2', '3', '5', '10', '20', '30'])
    ax.set_xlabel('RMS of same-exposure AAOmega/Spector flux ratio (%)\n'
                  '(transparency cancels; 68% bootstrap interval)')
    ax.set_title('(a) Extraction methods ranked by dither-to-dither '
                 'consistency')
    ax.legend(loc='upper right')
    ax.grid(axis='x', color=GRID, lw=0.8)

    for k, band in enumerate(('blue', 'red')):
        ax = fig.add_subplot(gs[k, 1])
        g = df[df['band'] == band]
        traces = [('pipeline extraction',
                   ratio_resid(g[g['method'] == 'free'], 'F_pipeline'), INK2,
                   '--'),
                  ('Moffat fit, beta = 2.5',
                   ratio_resid(g[g['method'] == 'beta2.5']), SERIES[2],
                   '-')]
        for label, lr, colour, ls in traces:
            for (field, sub), marker in zip(lr.groupby(level=0), ('o', '^')):
                ax.plot(np.arange(len(sub)), sub.values, ls=ls,
                        marker=marker, color=colour, lw=1.8, ms=7,
                        label='%s, %s' % (label, field.split('_')[0]))
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_ylabel('ratio - field median (%)')
        ax.set_title('(%s) %s band: per-dither ratio' % ('bc'[k], band))
        ax.grid(color=GRID, lw=0.8)
        if k == 1:
            ax.set_xlabel('dither (time order)')
            ax.legend(ncol=2, fontsize=8, loc='upper center',
                      bbox_to_anchor=(0.5, -0.22))
    for a in fig.axes:
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
    fig.suptitle('260904: SS flux extraction methods compared '
                 '(%d exposures, 2 fields)'
                 % df['exposure_id'].nunique(), fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('qc_dir')
    parser.add_argument('--fig-dir', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'figures'))
    args = parser.parse_args(argv)
    df = run(args.qc_dir)
    os.makedirs(args.fig_dir, exist_ok=True)
    make_figure(df, os.path.join(args.fig_dir, 'fig6_refit_methods.png'))
    df.to_csv(os.path.join(args.qc_dir, 'ss_refit.csv'), index=False)
    s = summarise(df)
    s.to_csv(os.path.join(args.qc_dir, 'ss_refit_summary.csv'), index=False)
    pd.set_option('display.width', 200)
    print('\nScatter (%) of same-exposure AAOmega/Spector flux ratio, and of '
          'per-frame flux, by method:')
    print(s.to_string(index=False, float_format=lambda v: '%.2f' % v))


if __name__ == '__main__':
    main()
