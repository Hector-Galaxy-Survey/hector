"""
Turn the edge simulations (ss_edge_sim.py) into provisional SS skip
thresholds.

A threshold cannot fix the error from assuming the wrong profile, which is
the same wherever the star sits.  So the quantity judged here is the extra
error caused by position:

    delta = err - mean(err of the same method for a centred star with the
                       same true FWHM, beta and flux)

A frame is "bad" if |delta| > tol.  For each method, quality metric
(SSEDGE, SSEDGEF, SSOUTER) and tolerance, the cut on the metric is chosen to
reject at least 95% of bad frames (or all of them, if possible), and the
fraction of good frames it also rejects is reported.

Usage:
    python ss_edge_thresholds.py <qc_dir> [--tol 2 5]
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_ss_quality_tests import SERIES, INK, INK2, MUTED, GRID

METHOD_LABEL = {'free': 'Moffat fit, beta free',
                'beta2.5': 'Moffat fit, beta = 2.5',
                'aper_b2.5': 'aperture, beta = 2.5 corr.',
                'aper_b3.5': 'aperture, beta = 3.5 corr.'}
METRICS = {  # name: (column, True if larger = safer)
    'SSEDGE': ('SSEDGE', True),
    'SSEDGEF': ('SSEDGEF', True),
    'SSOUTER': ('SSOUTER', False),
}
TRUTH = ['spectrograph', 'band', 'flux_scale', 'fwhm_true', 'beta_true',
         'method']


def add_delta(df):
    centre = (df[df['r'] == 0].groupby(TRUTH)['err'].mean()
              .rename('err_centre'))
    df = df.join(centre, on=TRUTH)
    df['delta'] = df['err'] - df['err_centre']
    return df


def best_cut(metric, safe_high, bad, catch=0.95):
    """Cut on metric that rejects >= catch of bad frames.

    Returns (cut, fraction of bad rejected, fraction of good rejected)."""
    m = metric.values
    b = bad.values
    if b.sum() == 0:
        return np.nan, np.nan, 0.0
    vals = np.sort(m[b])
    # keep = metric > cut (safe_high) or metric < cut (otherwise)
    if safe_high:
        cut = np.quantile(vals, catch)
        rejected = m <= cut
    else:
        cut = np.quantile(vals, 1 - catch)
        rejected = m >= cut
    return cut, rejected[b].mean(), rejected[~b].mean()


def threshold_table(df, tols):
    rows = []
    for (scale, method), g in df.groupby(['flux_scale', 'method']):
        for tol in tols:
            bad = g['delta'].abs() > tol
            for name, (col, safe_high) in METRICS.items():
                cut, caught, lost = best_cut(g[col], safe_high, bad)
                rows.append(dict(flux_scale=scale, method=method, tol=tol,
                                 metric=name, cut=cut, bad_frac=bad.mean(),
                                 bad_rejected=caught, good_rejected=lost))
    return pd.DataFrame(rows)


def binned(g, col, bins):
    idx = np.digitize(g[col], bins)
    out = []
    for i in range(1, len(bins)):
        s = g['delta'][idx == i]
        if len(s) >= 5:
            out.append(((bins[i - 1] + bins[i]) / 2, np.median(s),
                        np.percentile(s, 16), np.percentile(s, 84),
                        np.percentile(s.abs(), 95)))
    return np.array(out)


def make_figure(df, table, path, tol_line=2.0):
    methods = list(METHOD_LABEL)
    colour = dict(zip(methods, [MUTED, SERIES[0], SERIES[2], SERIES[1]]))
    scales = sorted(df['flux_scale'].unique(), reverse=True)
    fig, axes = plt.subplots(len(scales), 3, figsize=(16, 4.6 * len(scales)),
                             squeeze=False)
    bins = {'SSEDGE': np.arange(-1.5, 5.6, 0.35),
            'SSEDGEF': np.arange(-1.0, 4.6, 0.25),
            'SSOUTER': np.arange(0.0, 1.0001, 0.05)}
    for i, scale in enumerate(scales):
        g0 = df[df['flux_scale'] == scale]
        for j, name in enumerate(METRICS):
            ax = axes[i, j]
            for m in methods:
                b = binned(g0[g0['method'] == m], METRICS[name][0], bins[name])
                if not len(b):
                    continue
                ax.fill_between(b[:, 0], b[:, 2], b[:, 3], color=colour[m],
                                alpha=0.18, lw=0)
                ax.plot(b[:, 0], b[:, 1], color=colour[m], lw=2,
                        label=METHOD_LABEL[m])
            for t in (-tol_line, tol_line):
                ax.axhline(t, color=INK2, lw=1, ls=':')
            ax.axhline(0, color=MUTED, lw=1)
            ax.set_ylim(-25, 25)
            if METRICS[name][1]:
                ax.invert_xaxis()
            row = table[(table['flux_scale'] == scale) &
                        (table['method'] == 'aper_b3.5') &
                        (table['tol'] == tol_line) &
                        (table['metric'] == name)].iloc[0]
            if np.isfinite(row['cut']):
                ax.axvline(row['cut'], color=SERIES[1], lw=1.5, ls='--')
                ax.text(row['cut'], 22, '  aper cut %.2f\n  (loses %.0f%% '
                        'good)' % (row['cut'], 100 * row['good_rejected']),
                        color=INK, fontsize=8, va='top')
            ax.set_xlabel(name + ('  (<- towards/outside edge)'
                                  if METRICS[name][1] else
                                  '  (more light in outer ring ->)'))
            if j == 0:
                ax.set_ylabel('extra flux error from position (%%)\n'
                              'flux x%g of 260904 SS' % scale)
            ax.set_title('%s: median and 16-84%% range' % name, fontsize=10)
            ax.grid(color=GRID, lw=0.8)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
    axes[0, 0].legend(loc='lower left', fontsize=8)
    fig.suptitle('Simulated SS stars moved out past the bundle edge '
                 '(real H/U layouts and 260904 noise; dotted = +/-%g%%)'
                 % tol_line, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('qc_dir')
    parser.add_argument('--tol', type=float, nargs='+', default=[2.0, 5.0])
    parser.add_argument('--fig-dir', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'figures'))
    args = parser.parse_args(argv)
    df = pd.read_csv(os.path.join(args.qc_dir, 'ss_edge_sim.csv'))
    df = add_delta(df[df['method'].isin(METHOD_LABEL)])
    table = threshold_table(df, args.tol)
    table.to_csv(os.path.join(args.qc_dir, 'ss_edge_thresholds.csv'),
                 index=False)
    os.makedirs(args.fig_dir, exist_ok=True)
    make_figure(df, table, os.path.join(args.fig_dir,
                                        'fig7_edge_thresholds.png'),
                tol_line=args.tol[0])
    pd.set_option('display.width', 200)
    print(table.to_string(index=False, float_format=lambda v: '%.2f' % v))

    print('\nCentred-star absolute error (mean over truths, %), showing the '
          'profile-mismatch bias a threshold cannot remove:')
    c = df[df['r'] == 0].groupby(['flux_scale', 'method', 'beta_true'])['err']
    print(c.mean().unstack().round(1).to_string())


if __name__ == '__main__':
    main()
