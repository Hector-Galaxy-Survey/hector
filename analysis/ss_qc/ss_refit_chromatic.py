"""
Wavelength-dependent (spectral shape) consistency of SS extraction methods.

ss_refit.py compares one band flux per arm, which tests the normalisation of
the secondary transfer function.  This script tests its shape: each frame
is refitted in wavelength chunks across both arms (centre and FWHM free per
chunk, so differential refraction is followed), and for each exposure the
AAOmega/Spector ratio is formed chunk by chunk over the wavelengths both
spectrographs cover.  Each exposure's mean ratio is removed, leaving its
shape; the scatter of that shape between dithers (in each chunk, then
averaged) is the colour error a method would put into the secondary TF.

Methods: pipeline (FLUX_CALIBRATION extraction, as reduced), free-beta
Moffat fit, and the aperture method with beta = 2.5 and 3.5 corrections.

Read-only.  Usage:
    python ss_refit_chromatic.py <qc_dir> [--chunk 200]
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ss_refit import fit, read_band
from plot_ss_quality_tests import SERIES, INK, INK2, MUTED, GRID

warnings.simplefilter('ignore')

ARMS = {  # arm: {spectrograph: ccd}, usable range
    'blue': ({'AAOmega': '1', 'Spector': '3'}, (3800.0, 5700.0)),
    'red': ({'AAOmega': '2', 'Spector': '4'}, (6300.0, 7400.0)),
}
TELLURIC = [(6860.0, 6930.0), (7150.0, 7350.0)]
METHODS = ['pipeline', 'free', 'aper_b2.5', 'aper_b3.5']
LABEL = {'pipeline': 'pipeline extraction',
         'free': 'Moffat fit, beta free',
         'aper_b2.5': 'aperture, beta = 2.5 corr.',
         'aper_b3.5': 'aperture, beta = 3.5 corr.'}


def chunks(lo, hi, width):
    edges = np.arange(lo, hi + 1e-6, width)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if any(a < t1 and b > t0 for t0, t1 in TELLURIC):
            continue
        out.append((a, b))
    return out


def chunk_fluxes(fcal, probenum, wrange):
    x, y, flux, var, pipe = read_band(fcal, probenum, wrange, [])
    res = {'pipeline': pipe}
    free = fit(x, y, flux, var, None, beta_free=True)
    res['free'] = free['F']
    for b in (2.5, 3.5):
        f = fit(x, y, flux, var, b)
        res['aper_b%g' % b] = ((flux.sum() - f['bkg'] * len(flux))
                               / f['frac_on_fibres'])
    return res


def run(qc_dir, width):
    frames = pd.read_csv(os.path.join(qc_dir, 'ss_census_frames.csv'))
    frames = frames[(~frames['MNGRSPMS']) & frames['has_fluxcal_ext']]
    rows = []
    for arm, (ccds, wrange) in ARMS.items():
        ccd_spec = {'ccd_' + c: s for s, c in ccds.items()}
        sel = frames[frames['ccd'].isin(ccd_spec)]
        for r in sel.itertuples():
            fcal = r.path_red.replace('red.fits', 'fcal.fits')
            for a, b in chunks(*wrange, width):
                res = chunk_fluxes(fcal, r.PROBENUM, (a, b))
                for m, v in res.items():
                    rows.append(dict(arm=arm, lam=(a + b) / 2,
                                     exposure_id=r.exposure_id,
                                     field_id=r.field_id,
                                     spectrograph=r.spectrograph,
                                     method=m, F=v))
            print('%s %s' % (arm, r.fileroot), end='\r')
    return pd.DataFrame(rows)


def shape_residuals(df):
    """log ratio AAOmega/Spector per exposure and chunk, with the exposure
    mean (normalisation) and the field's median shape removed."""
    p = df.pivot_table(index=['method', 'field_id', 'exposure_id', 'lam'],
                       columns='spectrograph', values='F').dropna()
    lr = (100 * np.log(p['AAOmega'] / p['Spector'])).rename('lr').reset_index()
    lr['lr'] -= lr.groupby(['method', 'exposure_id'])['lr'].transform('mean')
    lr['lr'] -= lr.groupby(['method', 'field_id', 'lam'])['lr'].transform(
        'median')
    return lr


def make_figure(lr, path):
    colour = dict(zip(METHODS, [INK2, MUTED, SERIES[2], SERIES[1]]))
    fields = sorted(lr['field_id'].unique())
    fig, axes = plt.subplots(1, len(fields) + 1, figsize=(17, 4.8))
    for ax, field in zip(axes, fields):
        for m in ('pipeline', 'aper_b3.5'):
            g = lr[(lr['method'] == m) & (lr['field_id'] == field)]
            for k, (exp, e) in enumerate(g.groupby('exposure_id')):
                for blue_arm in (True, False):  # no line across the gap
                    s = e[(e['lam'] < 6000) == blue_arm]
                    ax.plot(s['lam'], s['lr'], '-', color=colour[m], lw=1.3,
                            alpha=0.8,
                            label=LABEL[m] if k == 0 and blue_arm else None)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xlabel('wavelength (A)')
        ax.set_ylabel('AAOmega/Spector shape - field median (%)')
        ax.set_title('%s: one line per dither' % field.split('_')[0])
    ax = axes[-1]
    for m in METHODS:
        s = (lr[lr['method'] == m].groupby('lam')['lr']
             .apply(lambda v: np.sqrt(np.mean(v**2))))
        for arm_range in ((3700, 5800), (6200, 7500)):
            seg = s[(s.index > arm_range[0]) & (s.index < arm_range[1])]
            ax.plot(seg.index, seg.values, '-o', color=colour[m], lw=2, ms=4,
                    label=('%s (mean %.1f%%)' % (LABEL[m], s.mean()))
                    if arm_range[0] == 3700 else None)
    ax.set_xlabel('wavelength (A)')
    ax.set_ylabel('RMS shape scatter between dithers (%)')
    ax.set_title('Shape scatter by method')
    ax.legend(fontsize=8)
    for a in axes:
        a.grid(color=GRID, lw=0.8)
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle('260904: spectral shape of the same-exposure AAOmega/Spector '
                 'SS flux ratio (normalisation removed)', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('qc_dir')
    parser.add_argument('--chunk', type=float, default=200.0)
    parser.add_argument('--fig-dir', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'figures'))
    args = parser.parse_args(argv)
    df = run(args.qc_dir, args.chunk)
    df.to_csv(os.path.join(args.qc_dir, 'ss_refit_chromatic.csv'),
              index=False)
    lr = shape_residuals(df)
    os.makedirs(args.fig_dir, exist_ok=True)
    make_figure(lr, os.path.join(args.fig_dir, 'fig8_shape_consistency.png'))
    print('\nRMS shape scatter between dithers (%), by method and arm:')
    lr['arm'] = np.where(lr['lam'] < 6000, 'blue', 'red')
    print(lr.groupby(['method', 'arm'])['lr']
          .apply(lambda v: np.sqrt(np.mean(v**2))).unstack().round(2)
          .to_string())


if __name__ == '__main__':
    main()
