"""
Dither-to-dither consistency of the SS calibration, and how it relates to
the SS quality metrics.

RESCALE = 10**((MAG_measured - CATMAG)/2.5) in g (cubing.scale_cube_pair_to_mag),
i.e. catalogue flux / extracted SS flux after primary calibration.  It moves
with transparency AND with SS extraction error.  Two ways to separate them:

  * Within a field and spectrograph, the residual of each dither from the
    median over dithers removes the field's mean throughput and star
    colour/magnitude errors; what is left is transparency + extraction error.
  * In the same exposure, AAOmega and Spector see the same sky, so
    RESCALE(AAOmega)/RESCALE(Spector) cancels transparency.  Its scatter
    across dithers is extraction error (of either star).

Inputs: ss_census_frames.csv and ss_metrics_existing.csv.
Outputs: ss_dither_consistency.csv and fig5_dither_consistency.png.

Usage:
    python ss_dither_consistency.py <qc_dir> [--fig-dir DIR]
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from plot_ss_quality_tests import SERIES, INK, INK2, GRID  # shared style

SPEC_COLOUR = {'AAOmega': SERIES[0], 'Spector': SERIES[1]}
MARKERS = ['o', '^', 's', 'D', 'v', 'P']


def load(qc_dir):
    frames = pd.read_csv(os.path.join(qc_dir, 'ss_census_frames.csv'))
    frames = frames[(frames['arm'] == 'blue') & (~frames['MNGRSPMS'])]
    metrics = pd.read_csv(os.path.join(qc_dir, 'ss_metrics_existing.csv'))
    keep = ['fileroot', 'exposure_id', 'plate_id', 'field_id', 'spectrograph',
            'SNR', 'FWHM', 'BETA', 'RESCALE', 'STDNAME', 'PROBENAM']
    d = frames[keep].merge(
        metrics.drop(columns=['exposure_id', 'plate_id', 'field_id',
                              'spectrograph']), on='fileroot')
    # residual from the field/spectrograph median, in per cent
    grp = d.groupby(['field_id', 'spectrograph'])['RESCALE']
    d['rescale_resid'] = 100 * (d['RESCALE'] / grp.transform('median') - 1)
    return d


def spectrograph_ratio(d):
    """RESCALE(AAOmega)/RESCALE(Spector) per exposure, as % from field median."""
    p = d.pivot_table(index=['field_id', 'exposure_id'],
                      columns='spectrograph', values='RESCALE').dropna()
    p['ratio'] = p['AAOmega'] / p['Spector']
    p['ratio_resid'] = 100 * (p['ratio'] /
                              p.groupby('field_id')['ratio'].transform('median')
                              - 1)
    return p.reset_index()


def robust_sigma(x):
    x = np.asarray(x, float)
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def make_figure(d, ratio, path):
    fields = sorted(d['field_id'].unique())
    marker = dict(zip(fields, MARKERS))
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])

    # (a) RESCALE residual per dither
    ax = fig.add_subplot(gs[0, :2])
    for (field, spec), g in d.groupby(['field_id', 'spectrograph']):
        g = g.sort_values('exposure_id')
        ax.plot(np.arange(len(g)), g['rescale_resid'], '-',
                marker=marker[field], color=SPEC_COLOUR[spec], lw=1.5, ms=7,
                label='%s %s' % (field, spec))
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xlabel('dither (time order)')
    ax.set_ylabel('RESCALE - field median (%)')
    ax.set_title('(a) Per-dither SS flux scaling, each field and spectrograph')
    ax.legend(ncol=2, loc='lower left')

    # (b) spectrograph ratio per dither
    ax = fig.add_subplot(gs[0, 2:])
    for field, g in ratio.groupby('field_id'):
        g = g.sort_values('exposure_id')
        ax.plot(np.arange(len(g)), g['ratio_resid'], '-',
                marker=marker[field], color=INK, lw=1.5, ms=7,
                label='%s  (scatter %.1f%%)' % (field,
                                                robust_sigma(g['ratio_resid'])))
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xlabel('dither (time order)')
    ax.set_ylabel('AAOmega / Spector RESCALE - median (%)')
    ax.set_title('(b) Same-exposure ratio: transparency cancels, '
                 'extraction error remains')
    ax.legend(loc='lower left')

    # (c-f) residual against each metric
    panels = [('BETA', 'Moffat beta (fitted)'),
              ('FWHM', 'FWHM (arcsec, fitted)'),
              ('SSEDGEF', 'SSEDGEF (edge distance / FWHM)'),
              ('SSOUTER', 'SSOUTER (outer-ring flux fraction)')]
    for k, (col, label) in enumerate(panels):
        ax = fig.add_subplot(gs[1, k])
        for (field, spec), g in d.groupby(['field_id', 'spectrograph']):
            ax.plot(g[col], g['rescale_resid'], ls='', marker=marker[field],
                    color=SPEC_COLOUR[spec], ms=7, mec='white', mew=0.8)
        rho, p = spearmanr(d[col], d['rescale_resid'])
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xlabel(label)
        if k == 0:
            ax.set_ylabel('RESCALE - field median (%)')
        ax.set_title('(%s) Spearman rho = %.2f (p = %.2f)'
                     % ('cdef'[k], rho, p), fontsize=10)
    for ax in fig.axes:
        ax.grid(color=GRID, lw=0.8)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    fig.suptitle('SS calibration consistency between dithers  '
                 '(%d exposures x 2 spectrographs)' % (len(d) // 2),
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('qc_dir')
    parser.add_argument('--fig-dir', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'figures'))
    args = parser.parse_args(argv)
    d = load(args.qc_dir)
    ratio = spectrograph_ratio(d)
    d.to_csv(os.path.join(args.qc_dir, 'ss_dither_consistency.csv'),
             index=False)
    os.makedirs(args.fig_dir, exist_ok=True)
    fig_path = os.path.join(args.fig_dir, 'fig5_dither_consistency.png')
    make_figure(d, ratio, fig_path)

    print('Robust scatter of RESCALE residual (%), per field & spectrograph:')
    print(d.groupby(['field_id', 'spectrograph'])['rescale_resid']
          .agg(robust_sigma).round(1).to_string())
    print('\nRobust scatter of AAOmega/Spector ratio (%), per field:')
    print(ratio.groupby('field_id')['ratio_resid'].agg(robust_sigma)
          .round(1).to_string())
    print('\nSpearman correlation of RESCALE residual with each quantity:')
    for col in ('BETA', 'FWHM', 'SSEDGE', 'SSEDGEF', 'SSOUTER', 'SSASYM',
                'SSASYMR', 'SNR'):
        rho, p = spearmanr(d[col], d['rescale_resid'])
        print('  %-8s rho = %+.2f  p = %.3f' % (col, rho, p))
    print('\nwrote', fig_path)


if __name__ == '__main__':
    main()
