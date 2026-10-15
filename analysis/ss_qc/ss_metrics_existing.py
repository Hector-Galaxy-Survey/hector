"""
Compute the SS quality metrics (dr/ss_quality.py) for frames that were
reduced before those metrics existed, using the PSF fit already stored in
each frame's FLUX_CALIBRATION header.  Read-only: no data file is modified.

The fibre positions and fluxes come straight from the FIBRES_IFU table and
the data array, so this does not need the tile/robot files that
utils.ifu.IFU requires.  Coordinates match the fit: x = XPOS * cos(dec),
y = YPOS (see fluxcal2.read_chunked_data).  Fibre fluxes are summed over
both arms; the per-fibre fractions are unaffected by the (fibre-independent)
flux calibration applied to sci.fits.

Usage:
    python ss_metrics_existing.py ss_census_frames.csv --out ss_metrics.csv
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import astropy.io.fits as pf

from hector.dr import ss_quality as ssq

warnings.simplefilter('ignore')

BLUE_TO_RED = {'1': '2', '3': '4'}


def bundle_fibres(path, probenum):
    """Good-fibre x, y (fit coordinates) and summed flux for one arm."""
    with pf.open(path) as hdul:
        table = hdul['FIBRES_IFU'].data
        good = (table['PROBENUM'] == probenum) & (table['TYPE'] == 'P')
        dec = np.mean(table['FIB_MDEC'][good])
        x = table['XPOS'][good] * np.cos(np.deg2rad(dec))
        y = table['YPOS'][good].astype(float)
        spec_id = table['SPEC_ID'][good] - 1
        flux = np.nansum(hdul[0].data[spec_id, :], axis=1)
    return x.astype(float), y, flux


def frame_metrics(blue_path):
    """Metrics for one exposure/spectrograph from its blue- and red-arm files."""
    fc = pf.getheader(blue_path, 'FLUX_CALIBRATION')
    probenum = fc['PROBENUM']
    fn = os.path.basename(blue_path)
    red_path = os.path.join(
        os.path.dirname(os.path.dirname(blue_path)),
        'ccd_' + BLUE_TO_RED[fn[5]], fn[:5] + BLUE_TO_RED[fn[5]] + fn[6:])
    x, y, flux = bundle_fibres(blue_path, probenum)
    if os.path.exists(red_path):
        xr, yr, flux_r = bundle_fibres(red_path, probenum)
        if len(xr) == len(x):
            flux = flux + flux_r
    psf = {'xcen_ref': fc['XCENREF'], 'ycen_ref': fc['YCENREF']}
    chunked = {'xfibre': x, 'yfibre': y, 'data': flux[:, None]}
    rows = ssq.assess_secondary_quality(chunked, psf, fwhm=fc.get('FWHM'))
    out = {k: v for k, v, _ in rows}
    out['SSNFIB'] = len(x)
    # bundle-relative fitted centre, for plotting
    _, centre = ssq.footprint_hull(x, y)
    out['xcen_rel'] = fc['XCENREF'] - centre[0]
    out['ycen_rel'] = fc['YCENREF'] - centre[1]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('census_frames', help='ss_census_frames.csv')
    parser.add_argument('--out', default='ss_metrics_existing.csv')
    args = parser.parse_args(argv)

    frames = pd.read_csv(args.census_frames)
    sel = frames[(frames['arm'] == 'blue') & (~frames['MNGRSPMS'].astype(bool))
                 & frames['has_fluxcal_ext'] & frames['path_sci'].notna()]
    out = []
    for r in sel.itertuples():
        m = frame_metrics(r.path_sci)
        m.update(fileroot=r.fileroot, exposure_id=r.exposure_id,
                 spectrograph=r.spectrograph, plate_id=r.plate_id,
                 field_id=r.field_id)
        out.append(m)
    pd.DataFrame(out).to_csv(args.out, index=False)
    print('wrote %d rows to %s' % (len(out), args.out))


if __name__ == '__main__':
    main()
