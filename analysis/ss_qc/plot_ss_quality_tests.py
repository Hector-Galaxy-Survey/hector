"""
Diagnostic plots for the SS quality metrics (dr/ss_quality.py) on the
synthetic configurations used in tests/test_ss_quality.py.

Figures (written to analysis/ss_qc/figures/ by default):

  fig1_layouts.png      real H and U fibre layouts, footprint, outer ring,
                        corner directions, broken fibre, old GOODPSF limit
  fig2_metric_maps.png  SSEDGE, SSOUTER and SSASYMR as a function of star
                        position over bundle U (FWHM 2")
  fig3_test_stars.png   the centre / side / corner test stars: fibre fluxes,
                        fitted vs observed centroid, metric values
  fig4_vs_edge.png      SSOUTER and SSASYMR against SSEDGE, towards a corner
                        and towards a side, for several seeing FWHMs

Usage:
    python plot_ss_quality_tests.py [--out-dir DIR]
"""

import argparse
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Circle, Polygon

warnings.simplefilter('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'tests'))
from test_ss_quality import bundle_layout, moffat_fibre_flux  # noqa: E402

from hector.config import fibre_diameter_arcsec  # noqa: E402
from hector.dr import ss_quality as ssq  # noqa: E402

R_CORE = fibre_diameter_arcsec / 2.0
GOODPSF_LIMIT = 6.0  # arcsec, fluxcal2.check_psf_parameters

# Palette (reference instance from the dataviz method): categorical slots
# in fixed order, a single-hue sequential ramp and a two-hue diverging ramp
# with a neutral grey midpoint.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']
INK, INK2, MUTED, SURFACE = '#0b0b0b', '#52514e', '#a3a29c', '#fcfcfb'
GRID = '#e4e3df'
SEQ = LinearSegmentedColormap.from_list(
    'seq_blue', ['#eef4fc', '#9cc3ef', '#2a78d6', '#123e75'])
DIV = LinearSegmentedColormap.from_list(
    'div_blue_orange', ['#123e75', '#2a78d6', '#d9d8d4', '#eb6834', '#8a3210'])

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'text.color': INK,
    'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.edgecolor': MUTED, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'legend.frameon': False, 'font.size': 10,
})


def footprint_outline(xf, yf, n=24):
    """Polygon of the hull grown by the core radius (Minkowski sum)."""
    vertices, _ = ssq.footprint_hull(xf, yf)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = (vertices[:, None, :] +
           R_CORE * np.stack([np.cos(t), np.sin(t)], -1)[None]).reshape(-1, 2)
    from scipy.spatial import ConvexHull
    return pts[ConvexHull(pts).vertices]


def style_bundle_axes(ax, lim=7.0):
    ax.set_aspect('equal')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('x (arcsec)')
    ax.set_ylabel('y (arcsec)')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def draw_fibres(ax, xf, yf, outer=None, face='none'):
    for i, (x, y) in enumerate(zip(xf, yf)):
        is_outer = outer is not None and outer[i]
        ax.add_patch(Circle((x, y), R_CORE, facecolor=face,
                            edgecolor=SERIES[1] if is_outer else MUTED,
                            lw=1.6 if is_outer else 0.8))


def metrics(x0, y0, xf, yf, fwhm=2.0):
    flux = moffat_fibre_flux(x0, y0, xf, yf, fwhm=fwhm)
    chunked = {'xfibre': xf, 'yfibre': yf, 'data': flux[:, None]}
    out = dict((k, v) for k, v, _ in ssq.assess_secondary_quality(
        chunked, {'xcen_ref': x0, 'ycen_ref': y0}, fwhm=fwhm))
    return out, flux


def outermost_fibre_towards(phi, xf, yf):
    outer = np.where(ssq.outer_fibres(xf, yf))[0]
    dphi = np.angle(np.exp(1j * (np.arctan2(yf[outer], xf[outer]) - phi)))
    i = outer[np.argmin(np.abs(dphi))]
    return xf[i], yf[i]


def broken_fibres(name):
    """Positions of BROKEN fibres, in the frame of bundle_layout(name)."""
    import pandas as pd
    from hector.config import plate_scale
    from test_ss_quality import FIBRE_FILE
    rows = pd.read_csv(FIBRE_FILE)
    rows = rows[rows['Bundle/plate'] == name]
    kind = rows['Hexabundle or sky fibre']
    x = rows['Bundle_Xc'].to_numpy(float) * plate_scale / 1000.0
    y = rows['Bundle_Yc'].to_numpy(float) * plate_scale / 1000.0
    good = (kind == 'H').to_numpy()
    broken = (kind == 'BROKEN').to_numpy()
    # bundle_layout centres on the mean of the good fibres
    return x[broken] - x[good].mean(), y[broken] - y[good].mean()


def fig_layouts(path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, name in zip(axes, 'UH'):
        xf, yf = bundle_layout(name)
        outer = ssq.outer_fibres(xf, yf)
        draw_fibres(ax, xf, yf, outer)
        for x, y in zip(*broken_fibres(name)):
            ax.add_patch(Circle((x, y), R_CORE, facecolor='none',
                                edgecolor=INK2, lw=0.8, ls=':'))
            ax.plot(x, y, 'x', color=INK, ms=9, mew=2)
            ax.annotate('broken fibre', (x, y), xytext=(10, -14),
                        textcoords='offset points', color=INK2, fontsize=9)
        ax.add_patch(Polygon(footprint_outline(xf, yf), closed=True,
                             fill=False, edgecolor=SERIES[0], lw=2))
        phi0 = ssq.hexagon_orientation(xf, yf)
        for k in range(6):
            p = phi0 + k * np.pi / 3
            ax.plot([0, 6.3 * np.cos(p)], [0, 6.3 * np.sin(p)], '--',
                    color=SERIES[0], lw=1, alpha=0.6)
        ax.add_patch(Circle((0, 0), GOODPSF_LIMIT, fill=False,
                            edgecolor=INK2, lw=1, ls='-.'))
        ax.annotate('current GOODPSF limit (6")', (0, -GOODPSF_LIMIT),
                    xytext=(0, -12), textcoords='offset points',
                    ha='center', color=INK2, fontsize=9)
        r = np.hypot(xf, yf)
        ax.set_title('Bundle %s: %d good fibres, outer ring %d '
                     '(r = %.2f"-%.2f")' % (name, len(xf), outer.sum(),
                                           r[outer].min(), r[outer].max()))
        style_bundle_axes(ax, 7.2)
    handles = [
        plt.Line2D([], [], color=SERIES[0], lw=2,
                   label='footprint (hull + core radius)'),
        plt.Line2D([], [], marker='o', ls='', mfc='none', mec=SERIES[1],
                   mew=1.6, ms=10, label='outer-ring fibre (SSOUTER)'),
        plt.Line2D([], [], color=SERIES[0], ls='--', lw=1,
                   label='corner directions (SSPHI = 0)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_metric_maps(path, fwhm=2.0, n=61, lim=6.5):
    xf, yf = bundle_layout('U')
    g = np.linspace(-lim, lim, n)
    edge = np.full((n, n), np.nan)
    outer = np.full((n, n), np.nan)
    asym = np.full((n, n), np.nan)
    for j, y0 in enumerate(g):
        for i, x0 in enumerate(g):
            m, _ = metrics(x0, y0, xf, yf, fwhm)
            edge[j, i], outer[j, i], asym[j, i] = (
                m['SSEDGE'], m['SSOUTER'], m['SSASYMR'])
    outline = footprint_outline(xf, yf)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        (edge, DIV, TwoSlopeNorm(0, edge.min(), edge.max()),
         'SSEDGE (arcsec)', 'distance to edge; < 0 outside'),
        (outer, SEQ, None, 'SSOUTER',
         'fraction of observed flux in outer ring'),
        (asym, DIV, TwoSlopeNorm(0, min(asym.min(), -0.1), asym.max()),
         'SSASYMR (arcsec)', 'fit further out than observed light'),
    ]
    for ax, (img, cmap, norm, label, sub) in zip(axes, panels):
        im = ax.imshow(img, origin='lower', extent=(-lim, lim, -lim, lim),
                       cmap=cmap, norm=norm, interpolation='nearest')
        ax.add_patch(Polygon(outline, closed=True, fill=False,
                             edgecolor=INK, lw=1.2))
        ax.plot(xf, yf, '.', color=INK, ms=2, alpha=0.5)
        if label.startswith('SSEDGE'):
            ax.contour(g, g, img, levels=[0.5, 1.0, 2.0, 3.0],
                       colors=INK2, linewidths=0.6)
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label(label)
        ax.set_title('%s\n%s' % (label.split(' ')[0], sub))
        style_bundle_axes(ax, lim)
    fig.suptitle('Metrics vs true star position, bundle U, Moffat '
                 'FWHM %.1f", beta 4 (noise-free)' % fwhm, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_test_stars(path, fwhm=2.0):
    xf, yf = bundle_layout('U')
    phi_c = ssq.hexagon_orientation(xf, yf)
    cases = [('centre', (0.0, 0.0)),
             ('middle of a side', outermost_fibre_towards(phi_c + np.pi / 6,
                                                          xf, yf)),
             ('corner', outermost_fibre_towards(phi_c, xf, yf))]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
    vmax = None
    for ax, (label, (x0, y0)) in zip(axes, cases):
        m, flux = metrics(x0, y0, xf, yf, fwhm)
        vmax = vmax or flux.max()
        frac = flux / flux.sum()
        for x, y, f in zip(xf, yf, frac):
            ax.add_patch(Circle((x, y), R_CORE * 0.95,
                                facecolor=SEQ(min(f / 0.35, 1.0)),
                                edgecolor=MUTED, lw=0.6))
        ax.add_patch(Polygon(footprint_outline(xf, yf), closed=True,
                             fill=False, edgecolor=SERIES[0], lw=1.5))
        xo, yo = ssq.observed_centroid(flux, xf, yf)
        ax.plot(x0, y0, '+', color=SERIES[1], ms=16, mew=2.5,
                label='fitted (= true) centre')
        ax.plot(xo, yo, 'o', mfc='none', mec=INK, ms=10, mew=1.8,
                label='observed light centroid')
        txt = ('SSEDGE  = %5.2f"\nSSEDGEF = %5.2f\nSSPHI   = %5.1f deg\n'
               'SSOUTER = %5.2f\nSSASYM  = %5.2f"\nSSASYMR = %5.2f"'
               % (m['SSEDGE'], m['SSEDGEF'], m['SSPHI'], m['SSOUTER'],
                  m['SSASYM'], m['SSASYMR']))
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top',
                family='monospace', fontsize=9, color=INK)
        ax.set_title('Star at %s' % label)
        style_bundle_axes(ax, 7.0)
    axes[0].legend(loc='lower left')
    sm = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, 0.35))
    cb = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.01)
    cb.set_label('fraction of observed flux per fibre')
    fig.suptitle('Test-star configurations (tests/test_ss_quality.py), '
                 'bundle U, FWHM %.1f"' % fwhm, fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_vs_edge(path):
    xf, yf = bundle_layout('U')
    phi_c = ssq.hexagon_orientation(xf, yf)
    r = np.linspace(0, 6.5, 66)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for colour, fwhm in zip(SERIES, (1.5, 2.0, 3.0)):
        for phi, ls, where in ((phi_c, '-', 'corner'),
                               (phi_c + np.pi / 6, '--', 'side')):
            edge, outer, asym = [], [], []
            for rr in r:
                m, _ = metrics(rr * np.cos(phi), rr * np.sin(phi), xf, yf,
                               fwhm)
                edge.append(m['SSEDGE'])
                outer.append(m['SSOUTER'])
                asym.append(m['SSASYMR'])
            kw = dict(color=colour, ls=ls, lw=2)
            axes[0].plot(edge, outer, **kw)
            axes[1].plot(edge, asym, **kw)
    for ax, ylabel in zip(axes, ('SSOUTER', 'SSASYMR (arcsec)')):
        ax.axvline(0, color=INK2, lw=1)
        ax.set_xlabel('SSEDGE (arcsec)   <- outside | inside ->')
        ax.set_ylabel(ylabel)
        ax.grid(color=GRID, lw=0.8)
        ax.invert_xaxis()
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0].set_title('Observed flux in outer ring')
    axes[1].set_title('Fitted centre beyond observed centroid')
    handles = ([plt.Line2D([], [], color=c, lw=2, label='FWHM %.1f"' % f)
                for c, f in zip(SERIES, (1.5, 2.0, 3.0))] +
               [plt.Line2D([], [], color=INK2, lw=2, ls='-',
                           label='towards corner'),
                plt.Line2D([], [], color=INK2, lw=2, ls='--',
                           label='towards side')])
    fig.legend(handles=handles, loc='lower center', ncol=5)
    fig.suptitle('Star moved outwards from the centre of bundle U '
                 '(noise-free): does SSEDGE alone predict truncation?',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out-dir', default=os.path.join(HERE, 'figures'))
    args = parser.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, func in (('fig1_layouts.png', fig_layouts),
                       ('fig2_metric_maps.png', fig_metric_maps),
                       ('fig3_test_stars.png', fig_test_stars),
                       ('fig4_vs_edge.png', fig_vs_edge)):
        path = os.path.join(args.out_dir, name)
        func(path)
        print('wrote', path)


if __name__ == '__main__':
    main()
