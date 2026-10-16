"""
Simulate secondary standards moving from the centre of the bundle out past
its edge, to find where each flux-extraction method stops being reliable and
so set provisional SS skip thresholds on SSEDGE / SSEDGEF / SSOUTER.

The night 260904 has no SS stars near a bundle edge, so instead:

  * fibre layouts: the real H (AAOmega) and U (Spector) bundles from a
    260904 frame (FIBRES_IFU, fit coordinates);
  * star brightness: the median SS band flux on 260904 for that
    spectrograph and band (from ss_refit.csv, aperture method);
  * noise: per-fibre variance = v0 + g * flux, fitted to the real variance
    of the same frame and band;
  * truth: circular Moffat with FWHM 1.2/1.8/2.4" and beta 2/2.5/3.5,
    stepped outwards towards a bundle corner and towards a side;
  * each configuration gets several noise realisations, then every method
    in ss_refit.extract is applied.

Because the true flux is known, the error here is absolute.  The quality
metrics are computed from the beta = 3.5 fit's centre and FWHM, as the
pipeline would compute them from its own fit.

Read-only.  Usage:
    python ss_edge_sim.py <qc_dir> [--n-noise 10]
"""

import argparse
import itertools
import os
import time
import warnings

import numpy as np
import pandas as pd

from hector.dr import ss_quality as ssq
from ss_refit import BANDS, extract, fibre_fractions, read_band

warnings.simplefilter('ignore')

CASES = [  # (spectrograph, band) pairs: both layouts, both noise regimes
    ('AAOmega', 'blue'),
    ('Spector', 'red'),
]
FLUX_SCALES = (1.0, 0.1)  # 260904-like stars, and ~2.5 mag fainter
FWHMS = (1.2, 1.8, 2.4)
BETAS = (2.0, 2.5, 3.5)
RADII = np.round(np.arange(0.0, 7.01, 0.35), 2)
METHODS = ('free', 'beta2.5', 'aper_b2.5', 'aper_b3.5', 'aper_b5',
           'bundlesum')


def noise_model(qc_dir, spectrograph, band):
    """Layout, star flux and variance model from a real 260904 frame."""
    frames = pd.read_csv(os.path.join(qc_dir, 'ss_census_frames.csv'))
    refit = pd.read_csv(os.path.join(qc_dir, 'ss_refit.csv'))
    ccd = 'ccd_' + BANDS[band]['ccd'][spectrograph]
    f = frames[(frames['ccd'] == ccd) & (~frames['MNGRSPMS'])
               & frames['has_fluxcal_ext']].iloc[0]
    fcal = f['path_red'].replace('red.fits', 'fcal.fits')
    x, y, flux, var, _ = read_band(fcal, f['PROBENUM'],
                                   BANDS[band]['range'], BANDS[band]['mask'])
    # var = v0 + g * flux, robustly over fibres
    g, v0 = np.polyfit(np.clip(flux, 0, None), var, 1)
    v0 = max(v0, np.percentile(var, 10))
    g = max(g, 0.0)
    sel = refit[(refit['band'] == band) & (refit['method'] == 'aper_b3.5')
                & (refit['spectrograph'] == spectrograph)]
    f_star = np.median(sel['F'])
    return dict(x=x, y=y, v0=v0, g=g, F=f_star, name=f['PROBENAM'],
                frame=f['fileroot'])


def simulate(model, n_noise, rng):
    x, y = model['x'], model['y']
    phi_corner = ssq.hexagon_orientation(x, y)
    _, centre = ssq.footprint_hull(x, y)
    rows = []
    for scale, fwhm in itertools.product(FLUX_SCALES, FWHMS):
        for beta in BETAS:
            for direction, phi in (('corner', phi_corner),
                                   ('side', phi_corner + np.pi / 6)):
                for r in RADII:
                    x0 = centre[0] + r * np.cos(phi)
                    y0 = centre[1] + r * np.sin(phi)
                    true_edge = ssq.edge_distance(x0, y0, x, y)
                    f_true = scale * model['F']
                    truth = f_true * fibre_fractions(x0, y0, fwhm, beta,
                                                     x, y)
                    var = model['v0'] + model['g'] * truth
                    for k in range(n_noise):
                        flux = truth + rng.normal(0.0, np.sqrt(var))
                        res = extract(x, y, flux, var)
                        ref = res['beta3.5']
                        chunked = {'xfibre': x, 'yfibre': y,
                                   'data': flux[:, None]}
                        q = {kk: v for kk, v, _ in
                             ssq.assess_secondary_quality(
                                 chunked, {'xcen_ref': ref['x0'],
                                           'ycen_ref': ref['y0']},
                                 fwhm=ref['fwhm'])}
                        snr = flux.sum() / np.sqrt(var.sum())
                        for m in METHODS:
                            rows.append(dict(
                                flux_scale=scale,
                                fwhm_true=fwhm, beta_true=beta,
                                direction=direction, r=r,
                                true_edge=true_edge, realisation=k,
                                method=m,
                                err=100 * (res[m]['F'] / f_true - 1),
                                snr=snr, **q))
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('qc_dir')
    parser.add_argument('--n-noise', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    out = []
    for spectrograph, band in CASES:
        model = noise_model(args.qc_dir, spectrograph, band)
        t0 = time.time()
        d = simulate(model, args.n_noise, rng)
        d['spectrograph'] = spectrograph
        d['band'] = band
        d['bundle'] = model['name']
        out.append(d)
        print('%s %s (bundle %s, from %s): F=%.3g v0=%.3g g=%.3g, '
              '%d rows in %.0fs' % (spectrograph, band, model['name'],
                                    model['frame'], model['F'], model['v0'],
                                    model['g'], len(d), time.time() - t0))
    df = pd.concat(out, ignore_index=True)
    path = os.path.join(args.qc_dir, 'ss_edge_sim.csv')
    df.to_csv(path, index=False)
    print('wrote', path)


if __name__ == '__main__':
    main()
