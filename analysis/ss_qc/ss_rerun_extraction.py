"""
Re-run the pipeline's own SS extraction (telluric.extract_secondary_standard,
the full chromatic CvD PSF fit) on copies of reduced frames, with the Moffat
beta free or fixed, and compare dither-to-dither consistency.

The original files are never modified: each configuration works on its own
copy of the field's fcal.fits frames under <out_dir>/<config>/, keeping the
<date>/<plate>/<field>/<name>/<ccd>/ layout.  Needs the tile's robot file in
hector/Tiles/Robot_files/ (utils.ifu.IFU).

Configurations, run in order:
  free      beta fitted (current pipeline behaviour); also checks that the
            re-run reproduces the original FLUX_CALIBRATION
  betamed   beta fixed at the median of the 'free' fits for the night,
            excluding failed fits (exception, or GOODPSF False)
  beta2.5   beta fixed at 2.5 (the value preferred by ss_refit.py)

Usage:
    python ss_rerun_extraction.py <date_dir> <field_id> <out_dir> [--n-cpu 7]
"""

import argparse
import glob
import multiprocessing
import os
import shutil
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import astropy.io.fits as pf

warnings.simplefilter('ignore')

PAIRS = {'AAOmega': ('1', '2'), 'Spector': ('3', '4')}
BANDS = {'blue': (4400.0, 5400.0), 'red': (6300.0, 7300.0)}


def field_pairs(date_dir, field_id):
    """(spectrograph, exposure, blue fcal path, red fcal path) for each frame
    pair of the field's main science frames."""
    out = []
    base = glob.glob(os.path.join(date_dir, '*', field_id, 'main'))
    if len(base) != 1:
        raise ValueError('expected one main/ dir for %s, found %s'
                         % (field_id, base))
    base = base[0]
    for spec, (cb, cr) in PAIRS.items():
        for blue in sorted(glob.glob(os.path.join(base, 'ccd_' + cb,
                                                  '*fcal.fits'))):
            fn = os.path.basename(blue)
            red = os.path.join(base, 'ccd_' + cr, fn[:5] + cr + fn[6:])
            if os.path.exists(red):
                out.append((spec, fn[:5] + fn[6:10], blue, red))
    return out


def copy_pair(blue, red, date_dir, dest_root):
    paths = []
    for p in (blue, red):
        rel = os.path.relpath(p, os.path.dirname(date_dir.rstrip('/')))
        dest = os.path.join(dest_root, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(p, dest)
        paths.append(dest)
    return paths


def _extract(args):
    """Worker: run the pipeline's SS extraction on one copied pair."""
    spec, exposure, blue, red, ss_beta = args
    from hector.dr import telluric
    t0 = time.time()
    try:
        telluric.extract_secondary_standard(
            [blue, red], model_name='ref_centre_alpha_circ_hdr_cvd',
            debug=False, verbose=False, ss_beta=ss_beta)
        err = None
    except Exception:
        err = traceback.format_exc().strip().splitlines()[-1]
    return dict(spectrograph=spec, exposure=exposure, blue=blue, red=red,
                error=err, seconds=time.time() - t0)


def band_flux(path, wrange):
    with pf.open(path) as hdul:
        hdr = hdul[0].header
        n = hdr['NAXIS1']
        lam = hdr['CRVAL1'] + (np.arange(n) + 1 - hdr['CRPIX1']) * hdr['CDELT1']
        use = (lam >= wrange[0]) & (lam <= wrange[1])
        return np.nanmean(hdul['FLUX_CALIBRATION'].data[0][use])


def summarise_pair(blue, red):
    fc = pf.getheader(blue, 'FLUX_CALIBRATION')
    keys = ['BETA', 'FWHM', 'ALPHAREF', 'XCENREF', 'YCENREF', 'GOODPSF',
            'SNR', 'SSEDGE', 'SSCHI2', 'SSBETAFX']
    rec = {k: fc.get(k) for k in keys}
    rec['F_blue'] = band_flux(blue, BANDS['blue'])
    rec['F_red'] = band_flux(red, BANDS['red'])
    return rec


def run_config(name, ss_beta, pairs, date_dir, out_dir, n_cpu):
    dest_root = os.path.join(out_dir, name)
    jobs = []
    for spec, exposure, blue, red in pairs:
        b, r = copy_pair(blue, red, date_dir, dest_root)
        jobs.append((spec, exposure, b, r, ss_beta))
    t0 = time.time()
    with multiprocessing.get_context('spawn').Pool(n_cpu) as pool:
        results = pool.map(_extract, jobs)
    rows = []
    for res in results:
        rec = dict(config=name, ss_beta=ss_beta, **res)
        if res['error'] is None:
            rec.update(summarise_pair(res['blue'], res['red']))
        rows.append(rec)
    print('%s (beta=%s): %d pairs in %.0f s, %d failed'
          % (name, ss_beta, len(rows), time.time() - t0,
             sum(r['error'] is not None for r in rows)))
    return pd.DataFrame(rows)


def original(pairs):
    rows = []
    for spec, exposure, blue, red in pairs:
        rec = dict(config='original', spectrograph=spec, exposure=exposure)
        rec.update(summarise_pair(blue, red))
        rows.append(rec)
    return pd.DataFrame(rows)


def ratio_rms(df):
    """RMS (%) of the same-exposure AAOmega/Spector band-flux ratio about
    its median over dithers, per band."""
    out = {}
    for band in ('blue', 'red'):
        p = df.pivot_table(index='exposure', columns='spectrograph',
                           values='F_' + band).dropna()
        lr = 100 * np.log(p['AAOmega'] / p['Spector'])
        lr = lr - lr.median()
        out[band] = np.sqrt(np.mean(lr**2))
        out['n_' + band] = len(lr)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('date_dir')
    parser.add_argument('field_id')
    parser.add_argument('out_dir')
    parser.add_argument('--n-cpu', type=int, default=7)
    args = parser.parse_args(argv)

    pairs = field_pairs(args.date_dir, args.field_id)
    print('%d frame pairs in %s' % (len(pairs), args.field_id))
    tables = [original(pairs)]

    free = run_config('free', None, pairs, args.date_dir, args.out_dir,
                      args.n_cpu)
    tables.append(free)
    ok = free[free['error'].isna() & (free['GOODPSF'] == True)]
    beta_med = float(np.median(ok['BETA']))
    print('median free beta over %d good fits: %.2f (range %.2f-%.2f)'
          % (len(ok), beta_med, ok['BETA'].min(), ok['BETA'].max()))

    for name, beta in (('betamed', round(beta_med, 2)), ('beta2.5', 2.5)):
        tables.append(run_config(name, beta, pairs, args.date_dir,
                                 args.out_dir, args.n_cpu))

    df = pd.concat(tables, ignore_index=True)
    csv = os.path.join(args.out_dir, 'ss_rerun_%s.csv' % args.field_id)
    df.to_csv(csv, index=False)

    # check: does the free re-run reproduce the original extraction?
    m = original(pairs).merge(free, on=['spectrograph', 'exposure'],
                              suffixes=('_orig', '_rerun'))
    for band in ('blue', 'red'):
        d = 100 * (m['F_%s_rerun' % band] / m['F_%s_orig' % band] - 1)
        print('free re-run vs original, %s band flux: median %+.2f%%, '
              'max |diff| %.2f%%' % (band, np.median(d), np.max(np.abs(d))))
    print('beta: original %s' % np.round(m['BETA_orig'].values, 2))
    print('      re-run   %s' % np.round(m['BETA_rerun'].values, 2))

    print('\nRMS of same-exposure AAOmega/Spector flux ratio (%):')
    for name, g in df.groupby('config', sort=False):
        r = ratio_rms(g[g['error'].isna()])  # NaN for 'original' rows too
        print('  %-9s blue %5.2f  red %5.2f   (%d exposures)'
              % (name, r['blue'], r['red'], r['n_blue']))
    print('wrote', csv)


if __name__ == '__main__':
    main()
