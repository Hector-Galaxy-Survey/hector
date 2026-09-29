"""
Census of secondary-standard (SS) flux calibration outcomes in reduced Hector data.

Walks one or more manager roots (the directory containing ``reduced/``) and
records, for every MFOBJECT frame, how far it got through the SS calibration
and what the SS fit looked like.  Writes a per-frame CSV, a per-exposure CSV
(one row per spectrograph per exposure) and prints a summary.

Frames can fail the SS step in ways that never set ``SECCOR``:

* the PSF fit in ``telluric.extract_secondary_standard`` raises (e.g. leastsq
  failure in ``fluxcal2.fit_model_flux``, or a bundle that is not a star), so
  ``manager.telluric_correct_pair`` never writes the ``sci.fits`` file and
  ``fluxcal_secondary`` silently skips the frame;
* the frame is shorter than ``minexp`` so the secondary TF is never applied.

Those are counted here alongside the explicit ``SECCOR=False`` cases.

Reduced directory layout (``manager.set_reduced_path``)::

    <root>/reduced/<date>/<plate_id>/<field_id>/<name>/<ccd>/<fileroot>{red,fcal,sci}.fits

Usage::

    python ss_census.py <root> [<root> ...] [--out-dir DIR] [--minexp 600]
"""

import argparse
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import astropy.io.fits as pf

warnings.filterwarnings('ignore', category=pf.verify.VerifyWarning)

# Spectrograph and arm for each CCD (see manager.other_arm / other_inst)
CCD_INFO = {
    'ccd_1': ('AAOmega', 'blue'),
    'ccd_2': ('AAOmega', 'red'),
    'ccd_3': ('Spector', 'blue'),
    'ccd_4': ('Spector', 'red'),
}

# Keywords written to FLUX_CALIBRATION by utils/fluxcal2_io.save_extracted_flux,
# manager.scale_frame_pair and manager.fluxcal_secondary
FCAL_KEYS = ['PROBENUM', 'PROBENAM', 'STDNAME', 'MODEL', 'GOODPSF', 'SNR',
             'FWHM', 'XCENREF', 'YCENREF', 'ALPHAREF', 'BETA', 'RESCALE',
             'CATMAGG', 'CATMAGR',
             # SS quality metrics from dr/ss_quality.py (branch ss-fluxcal-qc)
             'SSEDGE', 'SSEDGEF', 'SSPHI', 'SSOUTER', 'SSASYM', 'SSASYMR',
             'SSCHI2', 'SSNFIB', 'SSEXSUM']
PRIMARY_KEYS = ['NDFCLASS', 'EXPOSED', 'UTDATE', 'UTSTART', 'EPOCH',
                'INSTRUME', 'ZDSTART', 'MNGRSPMS', 'MNGRNAME']

FILEROOT_RE = re.compile(r'^(\d{2}[a-z]{3})(\d)(\d{4})red\.fits$')


def find_red_frames(root):
    """Yield paths of all *red.fits files under root/reduced/.../ccd_N/."""
    reduced = os.path.join(root, 'reduced')
    if not os.path.isdir(reduced):
        reduced = root
    for dirpath, dirnames, filenames in os.walk(reduced, followlinks=True):
        if os.path.basename(dirpath) not in CCD_INFO:
            continue
        for fn in filenames:
            if FILEROOT_RE.match(fn):
                yield os.path.join(dirpath, fn)


def path_components(path, root):
    """Return date, plate_id, field_id, name, ccd from the reduced path."""
    parts = os.path.normpath(path).split(os.sep)
    try:
        i = parts.index('reduced')
    except ValueError:
        # root was the reduced dir itself
        i = len(os.path.normpath(root).split(os.sep)) - 1
    rel = parts[i + 1:-1]
    if len(rel) != 5:
        return dict(date=None, plate_id=None, field_id=None, name=None,
                    ccd=parts[-2])
    return dict(zip(['date', 'plate_id', 'field_id', 'name', 'ccd'], rel))


def read_header_safe(path, ext=0):
    try:
        return pf.getheader(path, ext)
    except (OSError, KeyError, IndexError):
        return None


def has_extension(path, extname):
    try:
        with pf.open(path) as hdul:
            return extname in hdul
    except OSError:
        return False


def census_frame(red_path, root, minexp):
    """Collect SS information for one reduced frame; None if not MFOBJECT."""
    comp = path_components(red_path, root)
    fn = os.path.basename(red_path)
    m = FILEROOT_RE.match(fn)
    day, ccd_digit, number = m.groups()
    fileroot = fn[:-len('red.fits')]
    red_hdr = read_header_safe(red_path)
    if red_hdr is None:
        return None

    # Manager keywords are written to the raw file; fall back to the raw
    # symlink in the reduced dir if the reduced header lacks them.
    raw_link = os.path.join(os.path.dirname(red_path), fileroot + '.fits')
    raw_hdr = read_header_safe(raw_link) if os.path.exists(raw_link) else None

    def pkey(key):
        if key in red_hdr:
            return red_hdr[key]
        if raw_hdr is not None and key in raw_hdr:
            return raw_hdr[key]
        return None

    if pkey('NDFCLASS') != 'MFOBJECT':
        return None

    ccd = comp['ccd']
    spectrograph, arm = CCD_INFO.get(ccd, (None, None))
    row = dict(run=os.path.basename(os.path.normpath(root)), **comp,
               fileroot=fileroot, day=day, ccd_digit=int(ccd_digit),
               number=int(number), spectrograph=spectrograph, arm=arm,
               exposure_id=day + number)
    for key in PRIMARY_KEYS:
        row[key] = pkey(key)

    fcal_path = os.path.join(os.path.dirname(red_path), fileroot + 'fcal.fits')
    sci_path = os.path.join(os.path.dirname(red_path), fileroot + 'sci.fits')
    row['has_fcal'] = os.path.exists(fcal_path)
    row['has_sci'] = os.path.exists(sci_path)

    # SS extraction results: prefer sci (post-telluric), fall back to fcal
    src = sci_path if row['has_sci'] else (fcal_path if row['has_fcal'] else None)
    fc_hdr = read_header_safe(src, 'FLUX_CALIBRATION') if src else None
    row['has_fluxcal_ext'] = fc_hdr is not None
    for key in FCAL_KEYS:
        row[key] = fc_hdr.get(key) if fc_hdr is not None else None

    if row['has_sci']:
        sci_hdr = read_header_safe(sci_path)
        row['SECCOR'] = sci_hdr.get('SECCOR') if sci_hdr is not None else None
        row['has_fluxcal2_ext'] = has_extension(sci_path, 'FLUX_CALIBRATION2')
    else:
        row['SECCOR'] = None
        row['has_fluxcal2_ext'] = False

    row['status'] = classify(row, minexp)
    return row


def classify(row, minexp):
    """Assign a single SS outcome to a frame."""
    if row['MNGRSPMS']:
        return 'primary_std'
    if not row['has_fcal']:
        return 'no_fcal'            # failed upstream of the SS step
    if not row['has_sci']:
        return 'ss_fit_crash'       # telluric/SS extraction raised; no SECCOR
    exposed = row['EXPOSED']
    if exposed is not None and exposed < minexp:
        return 'short_exp'          # secondary TF never applied
    if row['SECCOR'] is None:
        return 'not_processed'      # fluxcal_secondary not run on this frame
    if row['SECCOR']:
        return 'ok'
    # apply_secondary_tf skips on either condition; low SNR is the more
    # specific cause (has_fluxcal2_ext keeps the other)
    snr = row['SNR']
    if snr is not None and snr < 2:
        return 'fail_low_snr'
    if not row['has_fluxcal2_ext']:
        return 'fail_no_tf'
    return 'fail_other'


FAIL_STATES = ('ss_fit_crash', 'fail_no_tf', 'fail_low_snr', 'fail_other')


def exposure_table(frames):
    """One row per (exposure, spectrograph), using the blue-arm SS outcome.

    SECCOR is written to both arms together, so the blue arm is
    representative; red-arm disagreements are flagged.
    """
    rows = []
    group_keys = ['run', 'date', 'plate_id', 'field_id', 'exposure_id',
                  'spectrograph']
    for keys, g in frames.groupby(group_keys, dropna=False):
        blue = g[g['arm'] == 'blue']
        red = g[g['arm'] == 'red']
        ref = blue.iloc[0] if len(blue) else g.iloc[0]
        rec = dict(zip(group_keys, keys))
        for col in ['name', 'EXPOSED', 'UTSTART', 'STDNAME', 'PROBENAM',
                    'SNR', 'GOODPSF', 'FWHM', 'RESCALE', 'SECCOR']:
            rec[col] = ref[col]
        rec['status'] = ref['status']
        rec['red_status'] = red.iloc[0]['status'] if len(red) else None
        rec['arm_mismatch'] = (rec['red_status'] is not None and
                               rec['red_status'] != rec['status'])
        rows.append(rec)
    exp = pd.DataFrame(rows)

    # Does the other spectrograph fail in the same exposure?
    other = {'AAOmega': 'Spector', 'Spector': 'AAOmega'}
    lookup = exp.set_index(['run', 'exposure_id', 'spectrograph'])['status']
    exp['other_spec_status'] = [
        lookup.get((r.run, r.exposure_id, other.get(r.spectrograph)), None)
        for r in exp.itertuples()]
    return exp


def field_table(exp):
    """Per field and spectrograph: number of dithers and failures.

    Fields with a mix of ok and failed dithers are the Phase 2 sample.
    """
    sci = exp[~exp['status'].isin(['primary_std', 'no_fcal'])]
    rows = []
    for keys, g in sci.groupby(['run', 'date', 'plate_id', 'field_id',
                                'spectrograph'], dropna=False):
        n_ok = int((g['status'] == 'ok').sum())
        n_fail = int(g['status'].isin(FAIL_STATES).sum())
        rows.append(dict(zip(['run', 'date', 'plate_id', 'field_id',
                              'spectrograph'], keys),
                         n_exp=len(g), n_ok=n_ok, n_fail=n_fail,
                         n_short=int((g['status'] == 'short_exp').sum()),
                         mixed=(n_ok > 0 and n_fail > 0),
                         stdname=g['STDNAME'].dropna().iloc[0]
                         if g['STDNAME'].notna().any() else None))
    return pd.DataFrame(rows)


def print_summary(exp, fields):
    sci = exp[exp['status'] != 'primary_std']
    print('\n=== SS outcome per exposure & spectrograph (blue-arm status) ===')
    print(sci['status'].value_counts().to_string())

    print('\n=== by spectrograph ===')
    print(pd.crosstab(sci['spectrograph'], sci['status'], margins=True)
          .to_string())

    print('\n=== by run ===')
    print(pd.crosstab(sci['run'], sci['status'], margins=True).to_string())

    fails = sci[sci['status'].isin(FAIL_STATES)]
    if len(fails):
        print('\n=== failures: other spectrograph in same exposure ===')
        print(pd.crosstab(fails['status'],
                          fails['other_spec_status'].fillna('missing'))
              .to_string())
        print('\n=== failures by standard star (top 20) ===')
        print(fails['STDNAME'].fillna('unknown').value_counts().head(20)
              .to_string())

    mism = sci[sci['arm_mismatch']]
    if len(mism):
        print('\nWARNING: %d exposure(s) where blue/red arm status differ'
              % len(mism))

    print('\n=== fields ===')
    print('fields x spectrograph: %d, with >=1 failure: %d, '
          'mixed ok/failed (Phase 2 sample): %d'
          % (len(fields), int((fields['n_fail'] > 0).sum()),
             int(fields['mixed'].sum())))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('roots', nargs='+',
                        help='manager root(s) containing reduced/')
    parser.add_argument('--out-dir', default='.',
                        help='where to write the CSV files')
    parser.add_argument('--minexp', type=float, default=600.0,
                        help='minimum exposure for secondary TF '
                             '(manager.fluxcal_secondary default)')
    args = parser.parse_args(argv)

    rows = []
    for root in args.roots:
        n = 0
        for red_path in find_red_frames(root):
            row = census_frame(red_path, root, args.minexp)
            if row is not None:
                rows.append(row)
                n += 1
        print('%s: %d MFOBJECT frames' % (root, n), file=sys.stderr)
    if not rows:
        print('No MFOBJECT frames found.', file=sys.stderr)
        return 1

    frames = pd.DataFrame(rows)
    exp = exposure_table(frames)
    fields = field_table(exp)

    os.makedirs(args.out_dir, exist_ok=True)
    frames.to_csv(os.path.join(args.out_dir, 'ss_census_frames.csv'),
                  index=False)
    exp.to_csv(os.path.join(args.out_dir, 'ss_census_exposures.csv'),
               index=False)
    fields.to_csv(os.path.join(args.out_dir, 'ss_census_fields.csv'),
                  index=False)
    print_summary(exp, fields)
    print('\nWrote CSVs to %s' % os.path.abspath(args.out_dir))
    return 0


if __name__ == '__main__':
    sys.exit(main())
