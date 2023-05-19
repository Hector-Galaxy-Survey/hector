from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import os
import warnings

import numpy as np
from scipy.optimize import leastsq, curve_fit
from scipy.interpolate import LSQUnivariateSpline, CubicSpline, griddata

from scipy.ndimage.filters import median_filter, gaussian_filter1d
from scipy.ndimage import zoom

import functools

from astropy import coordinates as coord
from astropy import units
from astropy import table
from astropy.io import fits as pf
from astropy.io import ascii
from astropy import __version__ as ASTROPY_VERSION
# extra astropy bits to calculate airmass
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from scipy.special import erfc
from scipy.stats import binned_statistic
from shutil import copyfile

# required for test plotting:
import pylab as py
import matplotlib as mpl
from matplotlib.patches import Circle, Arrow, Wedge
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Arc
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec


# The Hector package Sam developed
from hop.hexabundle_allocation.hector import constants

from ..utils.ifu import IFU
from ..utils.mc_adr import parallactic_angle, adr_r
from ..utils.other import saturated_partial_pressure_water
from ..config import millibar_to_mmHg
from ..utils.fluxcal2_io import read_model_parameters, save_extracted_flux
from .telluric2 import TelluricCorrectPrimary as telluric_correct_primary
from . import dust


# from ..manager import read_stellar_mags
# from ..qc.fluxcal import measure_band

try:
    from bottleneck import nansum, nanmean
except ImportError:
    from numpy import nansum, nanmean

    warnings.warn("Not Using bottleneck: Speed will be improved if you install bottleneck")

# import of ppxf for fitting of secondary stds:
from ppxf.ppxf import ppxf
from ppxf.ppxf_util import log_rebin
# from ppxf import ppxf
# from ppxf_util import log_rebin

import hector

# Get the astropy version as a tuple of integers
ASTROPY_VERSION = tuple(int(x) for x in ASTROPY_VERSION.split('.'))
hector_path = str(hector.__path__[0]) + '/'
STANDARD_CATALOGUES = (hector_path + 'standards/ESO/ESOstandards.dat',
                       hector_path + 'standards/Bessell/Bessellstandards.dat')
SSO_EXTINCTION_TABLE = hector_path + 'standards/extinsso.tab'
OPTICAL_MODEL_FILES = hector_path + 'optical_model/optical_model_data_files.yaml'

REFERENCE_WAVELENGTH = 5000.0

FIBRE_RADIUS = 0.798

TELLURIC_BANDS = np.array([[6850, 6960],
                           [7130, 7360],
                           [7560, 7770],
                           [8100, 8360]])

ARCSEC_IN_MICRONS = 105.0/1.6  # 105 microns = 1.6 arcseconds


def generate_subgrid(fibre_radius, n_inner=6, n_rings=10, wt_profile=False):
    """Generate a subgrid of points within a fibre."""
    radii = np.arange(0., n_rings) + 0.5
    rot_angle = 0.0
    radius = []
    theta = []
    for i_ring, radius_ring in enumerate(radii):
        n_points = int(np.round(n_inner * radius_ring))
        theta_ring = (np.linspace(0.0, 2.0*np.pi, n_points, endpoint=False) +
                      rot_angle)
        radius = np.hstack((radius, np.ones(n_points) * radius_ring))
        theta = np.hstack((theta, theta_ring))
        rot_angle += theta_ring[1] / 2.0
    radius *= fibre_radius / n_rings
    xsub = radius * np.cos(theta)
    ysub = radius * np.sin(theta)
    # generate a weight for the points based on the radial profile.  In this case
    # we use an error function that goes to 0.5 at 0.8 of the radius of the fibre.
    # this is just experimental, no evidence it makes much improvement:
    if (wt_profile):
        wsub = 0.5*erfc((radius-fibre_radius*0.8)*4.0)
        wnorm = float(np.size(radius))/np.sum(wsub)
        wsub = wsub * wnorm
    else:
        # or unit weighting:
        wsub = np.ones(np.size(xsub))
    return xsub, ysub, wsub

XSUB, YSUB, WSUB= generate_subgrid(FIBRE_RADIUS)
N_SUB = len(XSUB)


def get_cvd_parameters(path_list, star_match, max_sep_arcsec=60.0,
                             catalogues=STANDARD_CATALOGUES,
                             model_name='ref_centre_alpha_dist_circ_hdratm',   # NEED to change the name here
                             n_trim=0, smooth='spline', molecfit_available=False,
                             molecfit_dir='', speed='', tell_corr_primary=False):


    data_chunked = read_chunked_data(path_list, star_match['probenum'])
    trim_chunked_data(data_chunked, n_trim)

    psf_params_xcen, psf_params_ycen, psf_params_lambda = [], [], []
    for islice in range(np.shape(data_chunked['data'])[1]):
        chunked_vals, chunked_var, chunked_lambda, chunked_data = chunk_again(data_chunked, islice)

        # Fit the PSF
        fixed_parameters = set_fixed_parameters(
            path_list, model_name, probenum=star_match['probenum'])
        psf_parameters = fit_model_flux(
            chunked_data['data'],
            chunked_data['variance'],
            chunked_data['xfibre'],
            chunked_data['yfibre'],
            chunked_data['wavelength'],
            model_name,
            fixed_parameters=fixed_parameters)

        psf_params_ycen.append(psf_parameters['ycen_ref'])
        psf_params_xcen.append(psf_parameters['xcen_ref'])
        psf_params_lambda.append(np.nanmean(chunked_lambda))


    av_xref_cen = np.mean(psf_params_xcen)
    av_yref_cen = np.mean(psf_params_ycen)
    av_xref_cen_adjust = psf_params_xcen
    av_yref_cen_adjust = psf_params_ycen
    wavelengths = np.array(psf_params_lambda)
    # print(wavelengths)

    central_ilambda = np.argmin(np.abs(wavelengths - 5406.65608207))
    blue_ilambda = np.arange(0, central_ilambda)
    red_ilambda = np.arange(central_ilambda, len(wavelengths))

    CUT_OFF = ARCSEC_IN_MICRONS * 0.2

    av_xref_cen_adjust_check = av_xref_cen_adjust[central_ilambda]
    av_yref_cen_adjust_check = av_yref_cen_adjust[central_ilambda]
    av_xref_cen_adjust_goodb, av_yref_cen_adjust_goodb, lambda_goodb = [], [], []

    for ilambda in blue_ilambda[::-1]:
        if (np.abs(av_xref_cen_adjust_check - av_xref_cen_adjust[ilambda]) <= CUT_OFF) & \
                (np.abs(av_yref_cen_adjust_check - av_yref_cen_adjust[ilambda]) <= CUT_OFF):
            av_xref_cen_adjust_goodb.append(av_xref_cen_adjust[ilambda])
            av_yref_cen_adjust_goodb.append(av_yref_cen_adjust[ilambda])
            lambda_goodb.append(wavelengths[ilambda])

            av_xref_cen_adjust_check = av_xref_cen_adjust[ilambda]
            av_yref_cen_adjust_check = av_yref_cen_adjust[ilambda]
        # else:
        #     av_xref_cen_adjust_goodb.append(np.NaN)
        #     av_yref_cen_adjust_goodb.append(np.NaN)

    av_xref_cen_adjust_check = av_xref_cen_adjust[central_ilambda]
    av_yref_cen_adjust_check = av_yref_cen_adjust[central_ilambda]
    av_xref_cen_adjust_goodr, av_yref_cen_adjust_goodr, lambda_goodr = [], [], []
    for ilambda in red_ilambda:
        if (np.abs(av_xref_cen_adjust_check - av_xref_cen_adjust[ilambda]) <= CUT_OFF) & \
                (np.abs(av_yref_cen_adjust_check - av_yref_cen_adjust[ilambda]) <= CUT_OFF):
            av_xref_cen_adjust_goodr.append(av_xref_cen_adjust[ilambda])
            av_yref_cen_adjust_goodr.append(av_yref_cen_adjust[ilambda])
            lambda_goodr.append(wavelengths[ilambda])

            av_xref_cen_adjust_check = av_xref_cen_adjust[ilambda]
            av_yref_cen_adjust_check = av_yref_cen_adjust[ilambda]
        # else:
        #     av_xref_cen_adjust_goodr.append(np.NaN)
        #     av_yref_cen_adjust_goodr.append(np.NaN)

    av_xref_cen_adjust_good = np.concatenate(
        (np.array(av_xref_cen_adjust_goodb[::-1][3::]), np.array(av_xref_cen_adjust_goodr[0:-3])))
    av_yref_cen_adjust_good = np.concatenate(
        (np.array(av_yref_cen_adjust_goodb[::-1][3::]), np.array(av_yref_cen_adjust_goodr[0:-3])))
    lambda_good = np.concatenate((np.array(lambda_goodb[::-1][3::]), np.array(lambda_goodr[0:-3])))

    # Polynomial fit
    sorted_indx = np.array(np.argsort(lambda_good))
    zx = np.polyfit(np.array(lambda_good[sorted_indx]),
                    np.array(av_xref_cen - av_xref_cen_adjust_good[sorted_indx]), 2)
    zy = np.polyfit(np.array(lambda_good[sorted_indx]),
                    np.array(av_yref_cen - av_yref_cen_adjust_good[sorted_indx]), 2)
    polyval_x = np.polyval(zx, wavelengths)
    polyval_y = np.polyval(zy, wavelengths)

    # fig1, ax1 = py.subplots(figsize=(11, 4))
    # ax1.plot(wavelengths, (av_xref_cen - av_xref_cen_adjust)*ARCSEC_IN_MICRONS, 'bo', label='xdata')
    # ax1.plot(wavelengths, (av_yref_cen - av_yref_cen_adjust)*ARCSEC_IN_MICRONS, 'ro', label='ydata')
    #
    # ax1.plot(lambda_good, (av_yref_cen - av_yref_cen_adjust_good)*ARCSEC_IN_MICRONS, 'gx')
    # ax1.plot(lambda_good, (av_xref_cen - av_xref_cen_adjust_good)*ARCSEC_IN_MICRONS, 'cx')
    # ax1.plot(wavelengths, polyval_x*ARCSEC_IN_MICRONS, "b-")
    # ax1.plot(wavelengths, polyval_y*ARCSEC_IN_MICRONS, "r-")
    #
    # ax1.legend(loc='lower left', ncol=2)
    # ax1.set(xlabel='Wavelength (Ang.)', ylabel=r"$\Delta$ (ref - ref$_{\lambda}$) [microns]",
    #         title=f"Chromatic Variations in Distortion ({str(star_match['probenum'])})")
    # py.savefig(f"{os.path.basename(path_list[0])}_probename_{star_match['name']}_cubing.png", bbox_inches='tight')
    # py.close()
    # py.show()

    return [zx, zy]




def read_chunked_data(path_list, probenum, n_drop=None, n_chunk=None, sigma_clip=None):
    """
    Read flux from a list of files, chunk it and combine.
    """
    # MLPG: This is almost exactly the same as "read_chunked_data" routine in dr/fluxcal2.py
    # The difference is in the call to "get_chunks" which returns 3D data array
    if isinstance(path_list, str):
        path_list = [path_list]
    for i_file, path in enumerate(path_list):
        ifu = IFU(path, probenum, flag_name=False)
        remove_atmosphere(ifu)
        data_i, variance_i, wavelength_i = get_chunks(
            ifu, n_drop=n_drop, n_chunk=n_chunk, sigma_clip=sigma_clip)

        if i_file == 0:
            data = data_i
            variance = variance_i
            wavelength = wavelength_i
        else:
            data = np.hstack((data, data_i))
            variance = np.hstack((variance, variance_i))
            wavelength = np.vstack((wavelength, wavelength_i))

    xfibre = ifu.xpos_rel * np.cos(np.deg2rad(np.mean(ifu.ypos)))
    yfibre = ifu.ypos_rel
    # Only keep unbroken fibres
    good_fibre = (ifu.fib_type == 'P')
    chunked_data = {'data': data[good_fibre, :],
                    'variance': variance[good_fibre, :],
                    'wavelength': wavelength,
                    'xfibre': xfibre[good_fibre],
                    'yfibre': yfibre[good_fibre]}
    return chunked_data


def trim_chunked_data(chunked_data, n_trim):
    """Trim off the extreme blue end of the chunked data, because it's bad."""
    chunked_data['data'] = chunked_data['data'][:, n_trim:]
    chunked_data['variance'] = chunked_data['variance'][:, n_trim:]
    chunked_data['wavelength'] = chunked_data['wavelength'][n_trim:]
    return


def get_chunks(ifu, n_drop=None, n_chunk=None, sigma_clip=None):
    """Condence a spectrum into a number of chunks."""
    # MLPG: this is almost identical to "chunk_data" in dr/fluxcal2.py
    # Instead of taking the mean over one direction, this function returns the 3D chunk
    n_pixel = ifu.naxis1
    n_fibre = len(ifu.data)
    if n_drop is None:
        n_drop = 24
    if n_chunk is None:
        n_chunk = round((n_pixel - 2*n_drop) / 100.0)
    chunk_size = round((n_pixel - 2*n_drop) / n_chunk)
    if sigma_clip:
        good = np.isfinite(ifu.data)
        data_smooth = ifu.data.copy()
        data_smooth[~good] = np.median(ifu.data[good])
        data_smooth = median_filter(data_smooth, size=(1, 51))
        data_smooth[~good] = np.nan
        # Estimate of variance; don't trust 2dfdr values
        std_smooth = 1.4826 * np.median(np.abs(ifu.data[good] -
                                               data_smooth[good]))
        data = ifu.data
        clip = abs(data - data_smooth) > (sigma_clip * std_smooth)
        data[clip] = data_smooth[clip]
    else:
        data = ifu.data
    # Convert to integer for future compatibility.
    n_chunk = np.int(np.floor(n_chunk))
    chunk_size = np.int(np.floor(chunk_size))
    start = n_drop
    end = n_drop + n_chunk * chunk_size
    data = data[:, start:end].reshape(n_fibre, n_chunk, chunk_size)
    variance = ifu.var[:, start:end].reshape(n_fibre, n_chunk, chunk_size)
    wavelength = ifu.lambda_range[start:end].reshape(n_chunk, chunk_size)

    return data, variance, wavelength


def chunk_again(chunked_data_i, islice, n_drop=None, n_chunk=None, sigma_clip=None):
    """ Takes a given 3D chunk covering a specific wavelength range, and chunk
     to smaller pieces, again"""
    data = chunked_data_i['data'][:,islice,:]
    var = chunked_data_i['variance'][:,islice,:]
    lambda_range = chunked_data_i['wavelength'][islice,:]

    n_pixel = np.shape(data)[1]
    n_fibre = np.shape(data)[0]
    if n_drop is None:
        n_drop = 5
    if n_chunk is None:
        n_chunk = round((n_pixel - 2*n_drop) / 20.0)
    chunk_size = round((n_pixel - 2*n_drop) / n_chunk)

    # Convert to integer for future compatibility.
    n_chunk = np.int(np.floor(n_chunk))
    chunk_size = np.int(np.floor(chunk_size))
    start = n_drop
    end = n_drop + n_chunk * chunk_size
    data = data[:, start:end].reshape(n_fibre, n_chunk, chunk_size)
    variance = var[:, start:end].reshape(n_fibre, n_chunk, chunk_size)
    wavelength = lambda_range[start:end].reshape(n_chunk, chunk_size)

    data = nanmean(data, axis=2)
    variance = (np.nansum(variance, axis=2) /
                np.sum(np.isfinite(variance), axis=2)**2)
    # # Replace any remaining NaNs with 0.0; not ideal but should be very rare
    bad_data = ~np.isfinite(data)
    data[bad_data] = 0.0
    variance[bad_data] = np.inf
    wavelength = np.median(wavelength, axis=1)

    chunked_data = {'data': data,
                    'variance': variance,
                    'wavelength': wavelength,
                    'xfibre': chunked_data_i['xfibre'],
                    'yfibre': chunked_data_i['yfibre']}

    return data, variance, wavelength, chunked_data



""" Routines for fiting model fluxes """
def remove_atmosphere(ifu):
    """Remove atmospheric extinction (not tellurics) from ifu data."""
    # Read extinction curve (in magnitudes)
    wavelength_extinction, extinction_mags = read_atmospheric_extinction()
    # Interpolate onto the data wavelength grid
    extinction_mags = np.interp(ifu.lambda_range, wavelength_extinction,
                                extinction_mags)
    # Scale for observed airmass
    #airmass = calculate_airmass(ifu.zdstart, ifu.zdend)
    # no longer calculate airmass here,  instead take the value from the ifu
    # class that uses calc_eff_airmass()
    airmass = calc_eff_airmass(ifu.primary_header)
    extinction_mags *= airmass
    # Turn into multiplicative flux scaling
    extinction = 10.0 ** (-0.4 * extinction_mags)
    ifu.data /= extinction
    ifu.var /= (extinction**2)
    return

def zd2am( zenithdistance ):
    # fitting formula from Pickering (2002)
    altitude = ( 90. - zenithdistance )
    airmass = 1./ ( np.sin( ( altitude + 244. / ( 165. + 47 * altitude**1.1 )
                            ) / 180. * np.pi ) )
    return airmass

def calc_eff_airmass(header, return_zd=False):
    """Calculate the effective airmass using observatory location, coordinates
    and time.  This makes use of various astropy functions.  The input is
    a primary FITS header for a standard frame.  If return_zd = True, then
    return the effective ZD rather than airmass.
    """
    # this should really go into fluxcal, but there seems to be problems with
    # imports as this is also called from the ifu class that is within utils.
    # not sure why, but putting this in utils.other is a solution that seems to work.

    # get all the relevant header keywords:
    meanra = header['MEANRA'];
    meandec = header['MEANDEC']
    utdate = header['UTDATE'];
    utstart = header['UTSTART'];
    utend = header['UTEND']
    lat_obs = header['LAT_OBS'];
    long_obs = header['LONG_OBS'];
    alt_obs = header['ALT_OBS']
    zdstart = header['ZDSTART']

    # define observatory location:
    obs_loc = EarthLocation(lat=lat_obs * u.deg, lon=long_obs * u.deg, height=alt_obs * u.m)

    # Convert to the correct time format:
    date_formatted = utdate.replace(':', '-')
    time_start = date_formatted + ' ' + utstart
    # note that here we assume UT date start is the same as UT date end.  This works for
    # the AAT, given the time difference from UT at night, but will not for other observatories.
    time_end = date_formatted + ' ' + utend
    time1 = Time(time_start)
    time2 = Time(time_end)
    time_diff = time2 - time1
    time_mid = time1 + time_diff / 2.0

    # define coordinates using astropy coordinates object:
    coords = SkyCoord(meanra * u.deg, meandec * u.deg)

    # calculate alt/az using astropy coordinate transformations:
    altazpos1 = coords.transform_to(AltAz(obstime=time1, location=obs_loc))
    altazpos2 = coords.transform_to(AltAz(obstime=time2, location=obs_loc))
    altazpos_mid = coords.transform_to(AltAz(obstime=time_mid, location=obs_loc))

    # get altitude and remove units put in by astropy.  We need the float(), as
    # even when divided by the units, we get back a dimensionless object, not an actual
    # float.
    alt1 = float(altazpos1.alt / u.deg)
    alt2 = float(altazpos2.alt / u.deg)
    alt_mid = float(altazpos_mid.alt / u.deg)

    # convert to ZD:
    zd1 = 90.0 - alt1
    zd2 = 90.0 - alt2
    zd_mid = 90.0 - alt_mid

    # calc airmass at the start, end and midpoint:
    airmass1 = 1. / (np.sin((alt1 + 244. / (165. + 47 * alt1 ** 1.1)
                             ) / 180. * np.pi))
    airmass2 = 1. / (np.sin((alt2 + 244. / (165. + 47 * alt2 ** 1.1)
                             ) / 180. * np.pi))
    airmass_mid = 1. / (np.sin((alt_mid + 244. / (165. + 47 * alt_mid ** 1.1)
                                ) / 180. * np.pi))

    # get effective airmass by simpsons rule integration:
    airmass_eff = (airmass1 + 4. * airmass_mid + airmass2) / 6.

    # if needed get effective ZD:
    if (return_zd):
        zd_eff = (zd1 + 4. * zd_mid + zd2) / 6.

    # print('effective airmass:',airmass_eff)
    # print('ZD start:',zdstart)
    # print('ZD start (calculated):',zd1)

    # check that the ZD calculated actually agrees with the ZDSTART in the header
    d_zd = abs(zd1 - zdstart)
    if (d_zd > 0.1):
        print('WARNING: calculated ZD different from ZDSTART.  Difference:', d_zd)
        # if we have this problem, assume that the ZDSTART header keyword is correct
        # and that one or more of the other keywords has a problem.  Then set
        # the effective airmass to be based on ZDSTART:
        alt1 = 90.0 - zdstart
        airmass_eff = 1. / (np.sin((alt1 + 244. / (165. + 47 * alt1 ** 1.1)
                                    ) / 180. * np.pi))
    if (return_zd):
        return zd_eff
    else:
        return airmass_eff



def read_atmospheric_extinction(sso_extinction_table=SSO_EXTINCTION_TABLE):
    wl, ext = [], []
    #for entry in open( sso_extinction_table, 'r' ).xreadlines() :
    for entry in open( sso_extinction_table, 'r' ):
        line = entry.rstrip( '\n' )
        if not line.count( '*' ) and not line.count( '=' ):
            values = line.split()
            wl.append(  values[0] )
            ext.append( values[1] )
    wl = np.array( wl ).astype( 'f' )
    ext= np.array( ext ).astype( 'f' )
    return wl, ext


def insert_fixed_parameters(parameters_dict, fixed_parameters):
    """Insert the fixed parameters into the parameters_dict."""
    if fixed_parameters is not None:
        for key, value in fixed_parameters.items():
            parameters_dict[key] = value
    return parameters_dict


def set_fixed_parameters(path_list, model_name, probenum=None):
    """Return fixed values for certain parameters."""
    fixed_parameters = {}
    if (model_name == 'ref_centre_alpha_dist_circ' or
        model_name == 'ref_centre_alpha_dist_circ_hdratm' or
        model_name == 'ref_centre_alpha_circ_hdratm'):
        header = pf.getheader(path_list[0])
        ifu = IFU(path_list[0], probenum, flag_name=False)
        ha_offset = ifu.xpos[0] - ifu.meanra  # The offset from the HA of the field centre
        ha_start = header['HASTART'] + ha_offset
        # The header includes HAEND, but this goes very wrong if the telescope
        # slews during readout. The equation below goes somewhat wrong if the
        # observation was paused, but somewhat wrong is better than very wrong.
        ha_end = ha_start + (ifu.exptime / 3600.0) * 15.0
        ha = 0.5 * (ha_start + ha_end)
        zenith_direction = np.deg2rad(parallactic_angle(
            ha, header['MEANDEC'], header['LAT_OBS']))
        fixed_parameters['zenith_direction'] = zenith_direction
    if (model_name == 'ref_centre_alpha_dist_circ_hdratm' or
        model_name == 'ref_centre_alpha_circ_hdratm'):
        fibre_table_header = pf.getheader(path_list[0], 'FIBRES_IFU')
        temperature = fibre_table_header['ATMTEMP']
        pressure = fibre_table_header['ATMPRES'] * millibar_to_mmHg
        vapour_pressure = (fibre_table_header['ATMRHUM'] *
            saturated_partial_pressure_water(pressure, temperature))
        fixed_parameters['temperature'] = temperature
        fixed_parameters['pressure'] = pressure
        fixed_parameters['vapour_pressure'] = vapour_pressure
    if model_name == 'ref_centre_alpha_circ_hdratm':
        # Should take into account variation over course of observation
        # instead of just using the start value, which we do here:
        zd_eff = calc_eff_airmass(header,return_zd=True)
        fixed_parameters['zenith_distance'] = np.deg2rad(zd_eff)
        #fixed_parameters['zenith_distance'] = np.deg2rad(header['ZDSTART'])
    return fixed_parameters


def moffat_normalised(parameters, xfibre, yfibre, simple=False):
    """Return model Moffat flux for a single slice in wavelength."""
    if simple:
        xterm = (xfibre - parameters['xcen']) / parameters['alphax']
        yterm = (yfibre - parameters['ycen']) / parameters['alphay']
        alphax = parameters['alphax']
        alphay = parameters['alphay']
        beta = parameters['beta']
        rho = parameters['rho']
        moffat = (((beta - 1.0) /
                   (np.pi * alphax * alphay * np.sqrt(1.0 - rho ** 2))) *
                  (1.0 + ((xterm ** 2 + yterm ** 2 - 2.0 * rho * xterm * yterm) /
                          (1.0 - rho ** 2))) ** (-1.0 * beta))
        return moffat * np.pi * FIBRE_RADIUS ** 2
    else:
        n_fibre = len(xfibre)
        xfibre_sub = (np.outer(XSUB, np.ones(n_fibre)) +
                      np.outer(np.ones(N_SUB), xfibre))
        yfibre_sub = (np.outer(YSUB, np.ones(n_fibre)) +
                      np.outer(np.ones(N_SUB), yfibre))
        wt_sub = (np.outer(WSUB, np.ones(n_fibre)))

        flux_sub = moffat_normalised(parameters, xfibre_sub, yfibre_sub,
                                     simple=True)
        flux_sub = flux_sub * wt_sub

        return np.mean(flux_sub, axis=0)


def moffat_flux(parameters_array, xfibre, yfibre):
    """Return n_fibre X n_wavelength array of Moffat function flux values."""
    n_slice = len(parameters_array)
    n_fibre = len(xfibre)
    flux = np.zeros((n_fibre, n_slice))
    for i_slice, parameters_slice in enumerate(parameters_array):
        fibre_psf = moffat_normalised(parameters_slice, xfibre, yfibre)
        flux[:, i_slice] = (parameters_slice['flux'] * fibre_psf +
                            parameters_slice['background'])
    return flux


def model_flux(parameters_dict, xfibre, yfibre, wavelength, model_name):
    """Return n_fibre X n_wavelength array of model flux values."""
    parameters_array = parameters_dict_to_array(parameters_dict, wavelength,
                                                model_name)

    return moffat_flux(parameters_array, xfibre, yfibre)


def residual(parameters_vector, datatube, vartube, xfibre, yfibre,
             wavelength, model_name, fixed_parameters=None, secondary=False):
    """Return the residual in each fibre for the given model."""
    parameters_dict = parameters_vector_to_dict(parameters_vector, model_name)
    parameters_dict = insert_fixed_parameters(parameters_dict,
                                              fixed_parameters)

    model = model_flux(parameters_dict, xfibre, yfibre, wavelength, model_name)

    # 2dfdr variance is just plain wrong for fibres with little or no flux!
    # Try replacing with something like sqrt(flux), but with a floor

    if secondary:
        vartube = datatube.copy()
        cutoff = 0.05 * datatube.max()
        vartube[datatube < cutoff] = cutoff
    res = np.ravel((model - datatube) / np.sqrt(vartube))
    # Really crude way of putting bounds on the value of alpha
    if 'alpha_ref' in parameters_dict:
        if parameters_dict['alpha_ref'] < 0.5:
            res *= 1e10 * (0.5 - parameters_dict['alpha_ref'])
        elif parameters_dict['alpha_ref'] > 5.0:
            res *= 1e10 * (parameters_dict['alpha_ref'] - 5.0)
    if 'beta' in parameters_dict:
        if parameters_dict['beta'] <= 1.0:
            res *= 1e10 * (1.01 - parameters_dict['beta'])

    return res


def fit_model_flux(datatube, vartube, xfibre, yfibre, wavelength, model_name,
                   fixed_parameters=None, secondary=False):
    """Fit a model to the given datatube. slow_task 100 sec """
    par_0_dict = first_guess_parameters(datatube, vartube, xfibre, yfibre,
                                        wavelength, model_name)
    par_0_vector = parameters_dict_to_vector(par_0_dict, model_name)
    args = (datatube, vartube, xfibre, yfibre, wavelength, model_name,
            fixed_parameters, secondary)

    parameters_vector = leastsq(residual, par_0_vector, args=args)[0]
    parameters_dict = parameters_vector_to_dict(parameters_vector, model_name)

    return parameters_dict


def first_guess_parameters(datatube, vartube, xfibre, yfibre, wavelength,
                           model_name):
    """Return a first guess to the parameters that will be fitted."""
    par_0 = {}
    #weighted_data = np.sum(datatube / vartube, axis=1)
    if np.ndim(datatube)>1:
        weighted_data = np.nansum(datatube, axis=1)
        (nf, nc) = np.shape(datatube)
        # print(nf,nc)
    else:
        weighted_data = np.copy(datatube)
        (nf) = np.shape(datatube)
        nc = 1
        # print(nf)
    weighted_data[weighted_data < 0] = 0.0
    weighted_data /= np.sum(weighted_data)

    weighted_datacube = np.copy(datatube)
    weighted_datacube[weighted_datacube < 0] = 0.0
    weighted_datacube /= np.sum(weighted_datacube, axis=0)
    par_0['wavelength'] = wavelength
    par_0['flux'] = np.nansum(datatube, axis=0)
    par_0['background'] = np.zeros(len(par_0['flux']))
    par_0['xcen_ref'] = np.sum(xfibre * weighted_data)
    par_0['ycen_ref'] = np.sum(yfibre * weighted_data)
    par_0['alpha_ref'] = 1.0
    par_0['beta'] = 4.0

    return par_0

def parameters_dict_to_vector(parameters_dict, model_name):
    """Convert a parameters dictionary to a vector."""
    parameters_vector = np.hstack(
        (parameters_dict['flux'],
         parameters_dict['background'],
         parameters_dict['wavelength'],
         parameters_dict['xcen_ref'],
         parameters_dict['ycen_ref'],
         parameters_dict['alpha_ref'],
         parameters_dict['beta']))
    return parameters_vector


def parameters_vector_to_dict(parameters_vector, model_name):
    """Convert a parameters vector to a dictionary."""
    parameters_dict = {}
    n_slice = np.int((len(parameters_vector) - 10) // 3)
    parameters_dict['flux'] = parameters_vector[0:n_slice]
    parameters_dict['background'] = parameters_vector[n_slice:2*n_slice]
    parameters_dict['wavelength'] = parameters_vector[2*n_slice:3*n_slice]
    parameters_dict['xcen_ref'] = parameters_vector[-4]
    parameters_dict['ycen_ref'] = parameters_vector[-3]
    parameters_dict['alpha_ref'] = parameters_vector[-2]
    parameters_dict['beta'] = parameters_vector[-1]

    return parameters_dict


def parameters_dict_to_array(parameters_dict, wavelength, model_name):
    parameter_names = ('xcen ycen alphax alphay beta rho flux '
                       'background'.split())
    formats = ['float64'] * len(parameter_names)
    lw = np.size(wavelength)
    parameters_array = np.zeros(lw,
                                dtype={'names':parameter_names,
                                       'formats':formats})
    # Simple modelling of Chromatic Variation Dispersion effects
    parameters_array['xcen'] = parameters_dict['xcen_ref']
    parameters_array['ycen'] = parameters_dict['ycen_ref']

    parameters_array['alphax'] = (
        alpha(wavelength, parameters_dict['alpha_ref']))
    parameters_array['alphay'] = (
        alpha(wavelength, parameters_dict['alpha_ref']))
    parameters_array['beta'] = parameters_dict['beta']
    parameters_array['rho'] = np.zeros(lw)
    if len(parameters_dict['flux']) == len(parameters_array):
        parameters_array['flux'] = parameters_dict['flux']
    if len(parameters_dict['background']) == len(parameters_array):
        parameters_array['background'] = parameters_dict['background']

    return parameters_array


def alpha(wavelength, alpha_ref):
    """Return alpha at the specified wavelength(s)."""
    return alpha_ref * ((wavelength / REFERENCE_WAVELENGTH)**(-0.2))

