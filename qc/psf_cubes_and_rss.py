"""
                    ****** Quality control for checking CvD corrections *******
This function can be run independently of the hector module, the only dependence is the term colours

Authors: Madusha Gunawardhana (primary author, 2024+) -- converted to use the moffat integrated for the CvD fitting
         Yifan Mai (2024) -- MLPG included the original functions from Yifan in this script, however, they do not do
                             integration across the pixels, so need to be careful in using them

To Run: > import hector
            # To get the psf parameters from cubes
        > hector.qc.psf_cubes_and_rss.psf_check_cubes(parent_path="path_to_data", write_file="file_name_to_write")
            # To get the psf parameters from RSS frames
        > hector.qc.psf_cubes_and_rss.psf_check_rss(parent_path="path_to_data", write_file="file_name_to_write")

        # TODO list for MLPG
        TODO: Both measurements based on Cubes and RSS frames must be in the same units. Currently,
            cubes=pixel units RSS frame = arcseconds. The "fit_moffat_image" includes xfibpos, yfibpos keywords.
            Just need to propage them across to the cube routine.
            ---> SOLUTION: In writing to the *.csv file, the pixels are times by 0.5" to convert to arcseconds

        TODO: In ingrating_moffat profile check X_SUB, Y_SUB, against the definition of the grids of the same in fluxcal2.py
              to ensure the integration is performed correctly

"""

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import os

from astropy.io import fits as pf
from astropy.table import Table
import numpy as np
import pandas as pd
import string
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, leastsq

from ..utils.ifu import IFU
from ..dr import fluxcal2
from ..utils.term_colours import *


def psf_check_cubes(parent_path=None, write_file='Moffat_Circular_Cubes_gband.csv'):
    # parent_path = '/Volumes/USyd_2022v2/HECTOR DR/Busy Week September2024/drop_factor/cube_drop_v0_01/cubed_drop_1/'
    assert parent_path is not None, prRed(f"Please provide the parent path to the data cubes")

    all_fits_list = os.listdir(parent_path)
    new_list = []

    for (root, dirs, files) in os.walk(parent_path, topdown=True):
        for cube in files:
            extract_name = cube.split('_blue')[0]
            if 'blue' in cube:
                fitsname = f"{root}/{cube}"
                fitsname_other = f"{root}/{cube.replace('_blue', '_red')}"

                small_list = [fitsname, fitsname_other]
                # new_list is a 2-d list, each row is a 1-d list include red_arm_fits_path and blue_arm_fits_path
                new_list.append(small_list)

    # for dates in os.listdir(parent_path):
    #     for fitsfiles in os.listdir(f"{parent_path}/{dates}/"):
    #         for cube in os.listdir(f"{parent_path}/{dates}/{fitsfiles}/"):
    #             if 'blue' in cube:
    #                 fitsname = f"{parent_path}/{dates}/{fitsfiles}/{cube}"
    #             else:
    #                 fitsname_other = f"{parent_path}/{dates}/{fitsfiles}/{cube}"


    # create an empty dataframe, it will be used to save all moffat function parameters
    para_all = pd.DataFrame()
    error_list = []
    for list2 in new_list:
        blue_path = list2[0]
        red_path = list2[1]

        prCyan(f"----> blue/red paths: {blue_path}, {red_path} \n")

        # try:
            # df = fit_2D_Gaussian_model__get_parameters_from_cubed_data(path_list=[blue_path, red_path], n_chunk=4) # n_chunk redundant
        df = fit_integrate_moffat_func_and_get_parameters_from_cubes(path_list=[blue_path, red_path], n_chunk=4, band=[4700-200, 4700+200], elliptical=False)
        # except:
        #     prRed(f"**** {blue_path} has failed fitting due to a problem....**** \n")
        #     continue

        para_all = pd.concat([para_all, df])
        prGreen(f"{blue_path} is done.")

    para_all.to_csv(write_file)

    return


def psf_check_rss(parent_path=None, write_file='Moffat_Circular_RSS_gband.csv'):
    # parent_path = '/Volumes/USyd_2022v2/HECTOR DR/Busy Week September2024/drop_factor/cube_drop_v0_01/rss/'
    assert parent_path is not None, prRed(f"Please provide the parent path to the RSS frames")

    all_fits_list = os.listdir(parent_path)
    new_list = []
    all_fits_list_cp = all_fits_list[:]
    for fitsname in all_fits_list_cp:

        fitsname_other = fitsname

        if fitsname_other[5] == '1':
            fitsname_other = fitsname[:5] + '2' + fitsname[6:]
        elif fitsname_other[5] == '2':
            fitsname_other = fitsname[:5] + '1' + fitsname[6:]
        elif fitsname_other[5] == '3':
            fitsname_other = fitsname[:5] + '4' + fitsname[6:]
        elif fitsname_other[5] == '4':
            fitsname_other = fitsname[:5] + '3' + fitsname[6:]

        all_fits_list_cp.remove(fitsname_other)

        small_list = [fitsname, fitsname_other]
        # new_list is a 2-d list, each row is a 1-d list include red_arm_fits_path and blue_arm_fits_path
        new_list.append(small_list)

    # create an empty dataframe, it will be used to save all moffat function parameters
    para_all = pd.DataFrame()
    error_list = []

    for list2 in new_list:
        if list2[0][5] == '1' or list2[0][5] == '3':
            blue_path = list2[0]
            red_path = list2[1]
        else:
            blue_path = list2[1]
            red_path = list2[0]

        prCyan(f"----> blue/red paths: {blue_path}, {red_path} \n")

        blue_path = parent_path + blue_path
        red_path = parent_path + red_path

        _probename, _probenum = clear_probenum_list(blue_path)

        # We only need to fit a given cube
        if 'U' in _probename: probenum = _probenum[_probename=='U']
        else: probenum = _probenum[_probename=='H']
        prYellow(f"ProbeName: {probenum}, {_probename}")

        df = fit_integrated_moffat_func_and_get_parameters_from_rss(path_list=[blue_path, red_path], probenum=probenum,
                                                                 n_chunk=4, band=[4700-200, 4700+200])

        para_all = pd.concat([para_all, df])
        prGreen(f"{blue_path} is done.")

    para_all.to_csv(write_file)

    return


def clear_probenum_list(test_fits_name_blue):
    hdulist = pf.open(test_fits_name_blue)
    fibre_table = Table(hdulist['FIBRES_IFU'].data)
    hdulist.close()

    _probenum_list = np.unique([fibre['PROBENUM'] for fibre in fibre_table
                               if 'Sky' not in fibre['PROBENAME']])

    # This line did not work, as the probenum, included sky fibres....?
    # probename_list = [(fibre_table['PROBENAME'][fibre_table['PROBENUM']==probenum][0]).strip(' ') for probenum in probenum_list]
    probename_list, probenum_list = [], []
    probes = list(string.ascii_uppercase[0:21])
    for probenum in _probenum_list[1:]: # the first index corresponds to the name of tha fits Table column (in this case ""PROBENAME")
        plist = np.array(fibre_table['PROBENAME'][fibre_table['PROBENUM'] == probenum])
        i, found = 0, False
        while not found:
            if str(plist[i].strip(' ')) in probes:
                probename_list.append(plist[i].strip(' '))
                probenum_list.append(probenum)
                found = True
            else:
                i += 1

    return np.array(probename_list), np.array(probenum_list)


""" 
    DIFFERENT FUNCTIONS IN USE BELOW 
        * Elliptical_moffat_f  - Based on Yifan Mai's work
        * Elliptical_moffat_f_fit - Based on Yifan Mai's work
        * Elliptical_moffat - fluxcal.py
        * Circular_moffat - fluxcal.py
        * twoD_Gaussian 
    
"""
def elliptical_moffat_f(x, y, S0, S1, x0, y0, fwhm_1, fwhm_2, angle, beta):
    alpha_1 = 0.5 * fwhm_1 / np.sqrt(2. ** (1. / beta) - 1.)
    alpha_2 = 0.5 * fwhm_2 / np.sqrt(2. ** (1. / beta) - 1.)
    A = (np.cos(angle) / alpha_1) ** 2. + (np.sin(angle) / alpha_2) ** 2.
    B = (np.sin(angle) / alpha_1) ** 2. + (np.cos(angle) / alpha_2) ** 2.
    C = 2.0 * np.sin(angle) * np.cos(angle) * (1. / alpha_1 ** 2. - 1. / alpha_2 ** 2.)
    f = S0 + S1 / ((1. + A * ((x - x0) ** 2) + B * ((y - y0) ** 2) + C * (x - x0) * (y - y0)) ** beta)

    f_sum = np.nansum(f)

    f = f / f_sum

    return f


def elliptical_moffat_f_fit(xdata_tuple, S0, S1, x0, y0, fwhm_1, fwhm_2, angle, beta):
    """ See: http://www.aspylib.com/doc/aspylib_fitting.html """
    (x, y) = xdata_tuple

    alpha_1 = 0.5 * fwhm_1 / np.sqrt(2. ** (1. / beta) - 1.)
    alpha_2 = 0.5 * fwhm_2 / np.sqrt(2. ** (1. / beta) - 1.)
    A = (np.cos(angle) / alpha_1) ** 2. + (np.sin(angle) / alpha_2) ** 2.
    B = (np.sin(angle) / alpha_1) ** 2. + (np.cos(angle) / alpha_2) ** 2.
    C = 2.0 * np.sin(angle) * np.cos(angle) * (1. / alpha_1 ** 2. - 1. / alpha_2 ** 2.)

    f = S0 + S1 / ((1. + A * ((x - x0) ** 2) + B * ((y - y0) ** 2) + C * (x - x0) * (y - y0)) ** beta)

    return f.ravel()


def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))

    return g.ravel()


def moffat_elliptical(x, y, alpha_x, alpha_y, rho, beta, x0, y0, intensity):
    """Return an elliptical Moffat profile."""
    norm = (beta - 1) / (np.pi * alpha_x * alpha_y * np.sqrt(1 - rho**2))
    norm = norm * intensity
    x_term = (x - x0) / alpha_x
    y_term = (y - y0) / alpha_y
    moffat = norm * (1 + (x_term**2 + y_term**2 - 2*rho*x_term*y_term) /
                         (1 - rho**2))**(-beta)
    return moffat


def moffat_circular(x, y, alpha, beta, x0, y0, intensity):
    """Return a circular Moffat profile."""
    norm = (beta - 1) / (np.pi * alpha**2)
    norm = norm * intensity
    x_term = (x - x0) / alpha
    y_term = (y - y0) / alpha
    moffat = norm * (1 + (x_term**2 + y_term**2))**(-beta)
    return moffat


""" 
    INITIAL GUESSES:
        * for the 2D Gaussian function
        * elliptical_moffat_f function
"""
def init_guess_2D_Gauss_f(flux, x, y):
    # x0 = np.nanmedian(x)
    # y0 = np.nanmedian(y)
    amplitude = np.nanmax(flux)
    index_maxflux = np.where(flux == amplitude)[0][0]

    xo = x[index_maxflux]
    yo = y[index_maxflux]

    d = 1.7

    FWHM = np.sqrt(np.sum((flux > amplitude / 2.).flatten())) * d
    sigma_x = FWHM / (2. * np.sqrt(2. * np.log(2.)))
    sigma_y = FWHM / (2. * np.sqrt(2. * np.log(2.)))

    theta = 0.
    beta = 3
    if amplitude <= 0.6:
        beta = 1

    S0 = 0
    offset = 1.0

    p0 = [amplitude, xo, yo, sigma_x, sigma_y, theta, offset]

    return p0

def init_guess_ellip_moffat_f(flux, x, y):
    # x0 = np.nanmedian(x)
    # y0 = np.nanmedian(y)
    S1 = np.nanmax(flux)
    index_maxflux = np.where(flux == S1)[0][0]

    x0 = x[index_maxflux]
    y0 = y[index_maxflux]

    d = 1.7

    fwhm = np.sqrt(np.sum((flux > S1 / 2.).flatten())) * d
    fwhm_1 = fwhm
    fwhm_2 = fwhm

    angle = 0.
    beta = 3
    if S1 <= 0.6:
        beta = 1

    S0 = 0

    p0 = [S0, S1, x0, y0, fwhm_1, fwhm_2, angle, beta]
    return p0


"""
    CREATE FLUX/NOISE IMAGE by collapsing the cube over a given wavelength band
"""
def chunk_cube(flux, noise, wavelength, good=None, n_band=1, band=None):
    """Collapse a cube into a 2-d image, or a series of images."""
    # Be careful about sending in mixed red+blue cubes - in this case
    # n_band should be even
    n_pix = len(wavelength)
    wave_mask = (wavelength > band[0]) & (wavelength < band[1])
    if good is None:
        good = np.arange(flux.size)
        good.shape = flux.shape

    flux_out = np.zeros((n_band, flux.shape[1], flux.shape[2]))
    noise_out = np.zeros((n_band, flux.shape[1], flux.shape[2]))
    wavelength_out = np.zeros(n_band)
    for i_band in range(n_band):
        # start = i_band * n_pix / n_band
        # finish = (i_band+1) * n_pix / n_band
        flux_out[i_band, :, :] = (
            np.nansum(flux[wave_mask, :, :] * good[wave_mask, :, :], 0) /
            np.sum(good[wave_mask, :, :], 0))
        noise_out[i_band, :, :] = (
            np.sqrt(np.nansum(noise[wave_mask, :, :]**2 *
                              good[wave_mask, :, :], 0)) /
            np.sum(good[wave_mask, :, :], 0))
        wavelength_out = wavelength[wave_mask]
    flux_out = np.squeeze(flux_out)
    noise_out = np.squeeze(noise_out)
    wavelength_out = np.squeeze(wavelength_out)

    return flux_out, noise_out, wavelength_out


"""
    DIFFERENT MODEL FITTING ROUTINES DEALING WITH CUBES
        * most of them do not integrate over the pixels
"""
def get_cube_data(path_list, band=None):
    if band[0] < 6000.0:
        ccd_flag = 'blue'
        hdulist = pf.open(path_list[0])
        hdr = hdulist[0].header
        wavelength = get_coords(hdr, 3)
        flux_cube = hdulist[0].data
        noise_cube = np.sqrt(hdulist['VARIANCE'].data)
        hdulist.close()
    else:
        ccd_flag = 'red'
        hdulist = pf.open(path_list[1])
        hdr = hdulist[0].header
        wavelength = get_coords(hdr, 3)
        flux_cube = hdulist[0].data
        noise_cube = np.sqrt(hdulist['VARIANCE'].data)
        hdulist.close()

    return wavelength, flux_cube, noise_cube, ccd_flag, hdr


def fit_2D_Gaussian_model__get_parameters_from_cubed_data(path_list, n_chunk, band=None):
    # Get blue/red wavelength arrays
    wavelength, flux_cube, noise_cube, ccd_flag = get_cube_data(path_list, band=band)

    flux_image, noise_image, wave_image = chunk_cube(flux_cube, noise_cube, wavelength, band=band)

    # Create a meshgrid with the coordinates
    coords = np.meshgrid(np.arange(flux_image.shape[0]), np.arange(flux_image.shape[1]))

    # --> Fitting routines....
    initial_guess = init_guess_2D_Gauss_f(
        flux_image,
        np.arange(flux_image.shape[0]).astype(float),
        np.arange(flux_image.shape[1]).astype(float))
    # print("initial_guess:", initial_guess)

    x, y = np.array(coords[0]).astype(float), np.array(coords[1]).astype(float)
    popt, pcov = curve_fit(twoD_Gaussian, (x, y), flux_image.ravel(), p0=initial_guess)

    # Plotting routines...
    data_fitted = twoD_Gaussian((x, y), *popt)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(flux_image.reshape(flux_image.shape[0], flux_image.shape[1]), cmap=plt.cm.jet, origin='lower',
              extent=(x.min(), x.max(), y.min(), y.max()))
    ax.contour(x, y, data_fitted.reshape(flux_image.shape[0], flux_image.shape[1]), 8, colors='w')
    # plt.show()

    c = '/'
    index_of_slash = [pos for pos, char in enumerate(path_list[0]) if char == c]
    index = index_of_slash[-1]
    file_name = path_list[0][index + 1:]

    plt.tight_layout()
    plt.savefig(f"Gauss2D_{file_name}.png", dpi=300)
    plt.close()

    # End plotting

    if np.abs(popt[3]) > np.abs(popt[4]):
        fwhm_large = 2.* np.sqrt(2.0*np.log(2)) * np.abs(popt[3])
        fwhm_small = 2.* np.sqrt(2.0*np.log(2)) * np.abs(popt[4])
    else:
        fwhm_large = 2.* np.sqrt(2.0*np.log(2)) * np.abs(popt[4])
        fwhm_small = 2.* np.sqrt(2.0*np.log(2)) * np.abs(popt[3])

    df = pd.DataFrame({'FileName': [file_name],
                       'CCD': 'NA',
                       'Observ_num': 'NA',
                       'Amplitude': popt[0],
                       'x0': popt[1],
                       'y0': popt[2],
                       'sigma_x': popt[3],
                       'sigma_y': popt[4],
                       'fwhm_semi_major': fwhm_large,
                       'fwhm_semi_minor': fwhm_small,
                       'angle': popt[5],
                       'offset': popt[6],
                       'wavelength_centre': np.nanmedian(band)})

    return df


def fit_integrate_moffat_func_and_get_parameters_from_cubes(path_list, n_chunk,band=None, elliptical=True):
    """
        Uses the "fit_moffat_to_image" and "moffat_integrated" in the "fluxcal.py" (which MLPG copied to this script).
        This allows the fitting of a Moffat profile to an image, optionally allowing ellipticity.
        Also, a Moffat profile is integrated over pixels

        This is the routine we should be using
    """
    # Save to a pandas dataframe
    c = '/'
    index_of_slash = [pos for pos, char in enumerate(path_list[0]) if char == c]
    index = index_of_slash[-1]
    file_name = path_list[0][index + 1:]

    # Get blue/red wavelength arrays
    wavelength, flux_cube, noise_cube, ccd_flag, hdr_primary = get_cube_data(path_list, band=band)
    prPurple(f"selecting {ccd_flag} cube for the analysis to cover the lambda range {band}")

    # Get the collapsed image over the wave band
    flux_image, noise_image, wave_image = chunk_cube(flux_cube, noise_cube, wavelength, band=band)

    coords = np.meshgrid(np.arange(flux_image.shape[0]), np.arange(flux_image.shape[1]))
    x_mesh, y_mesh = np.meshgrid(np.arange(flux_image.shape[0]), np.arange(flux_image.shape[1]))

    # Using the moffat fitting function in fluxcal
    elliptical = False # Moffat model fitted is an elliptical or circular
    params, sigma = fit_moffat_to_image(flux_image, noise_image, elliptical=elliptical, background=False, elliptical_f=False)

    if elliptical:
        moffat = moffat_elliptical(x_mesh, y_mesh, *params)
        prPurple(f"Best-fitting params from the RSS frame {file_name} are alpha1={params[0]}, alpha2={params[1]}, "
                 f"rho={params[2]}, beta={params[3]}, x00={params[4]}, y00={params[5]}, intensity={params[6]}")
    else:
        moffat = moffat_circular(x_mesh, y_mesh, *params)
        prPurple(f"Best-fitting params from the RSS frame {file_name} are ALPHA={params[0]}, BETA={params[1]}, "
                 f"X00={params[2]}, Y00={params[3]}, INTENSITY={params[4]}, LAMBDA={band}")

    # Calculate the FWHM
    if elliptical:
        if np.abs(params[0]) > np.abs(params[1]):
            fwhm_large = params[0] * 2.0 * np.sqrt(2. ** (1. / params[3]) - 1.)
            fwhm_small = params[1] * 2.0 * np.sqrt(2. ** (1. / params[3]) - 1.)
        else:
            fwhm_large = params[1] * 2.0 * np.sqrt(2. ** (1. / params[3]) - 1.)
            fwhm_small = params[0] * 2.0 * np.sqrt(2. ** (1. / params[3]) - 1.)

        df = pd.DataFrame({'Date': [file_name],
                           'CCD': 'N/A',
                           'observ_num': 'N/A',
                           'probenum': 'N/A',
                           'alpha1': params[0],
                           'alpha2': params[1],
                           'rho': params[2],
                           'beta': params[3],
                           'x0': params[4],
                           'y0': params[5],
                           'intensity': params[6],
                           'fwhm_semi_major': fwhm_large,
                           'fwhm_semi_minor': fwhm_small,
                           'wavelength': np.nanmedian(band)})
    else:
        fwhm_large = params[0] * 2.0 * np.sqrt(2. ** (1. / params[1]) - 1.)

        df = pd.DataFrame({'Date': [file_name],
                           'CCD': 'N/A',
                           'observ_num': 'N/A',
                           'probenum': 'N/A',
                           'alpha1': params[0],
                           'alpha2': 'N/A',
                           'rho': 'N/A',
                           'beta': params[1],
                           'x0': params[2],
                           'y0': params[3],
                           'intensity': params[4],
                           'fwhm_semi_major_pixels': fwhm_large,
                           'fwhm_semi_major_arcsec': fwhm_large * 0.5,
                           'fwhm_semi_minor': 'N/A',
                           'wavelength': np.nanmedian(band)})

    # --> plotting
    fig, ax = plt.subplots(1, 1)
    x, y = np.array(coords[0]).astype(float), np.array(coords[1]).astype(float)
    ax.imshow(flux_image.reshape(flux_image.shape[0], flux_image.shape[1]), cmap=plt.cm.jet, origin='lower',
              extent=(x.min(), x.max(), y.min(), y.max()))
    ax.contour(x, y, moffat.reshape(flux_image.shape[0], flux_image.shape[1]), 8, colors='w')

    plt.tight_layout()
    plt.savefig(f"Cube_Moffat_elliptical_{elliptical}_integrated_{file_name}.png", dpi=300)
    plt.close()

    return df


def fit_ellip_moffat_f_model_and_get_parameters_from_cubed_data(path_list, n_chunk,band=None):
    """
    read data and fit by model elliptical moffat
    ---
    return: pd.DataFrame [S0, S1, x0, y0, fwhm_large, fwhm_small, angle, beta, wavelength
    """

    # Get blue/red wavelength arrays
    wavelength, flux_cube, noise_cube, ccd_flag = get_cube_data(path_list, band=band)
    prPurple(f"selecting {ccd_flag} cube for the analysis to cover the lambda range {band}")

    flux_image, noise_image, wave_image = chunk_cube(flux_cube,noise_cube,wavelength, band=band)

    coords = np.meshgrid( np.arange(flux_image.shape[0]), np.arange(flux_image.shape[1]) )

    x, y = np.array(coords[0]).astype(float), np.array(coords[1]).astype(float)
    popt, pcov = curve_fit(f=elliptical_moffat_f_fit, xdata=(x,y), ydata=flux_image.ravel(),
                           p0=init_guess_ellip_moffat_f(flux_image,
                                                        np.arange(flux_image.shape[0]).astype(float),
                                                        np.arange(flux_image.shape[1]).astype(float)), maxfev=5000)

    if np.abs(popt[4]) > np.abs(popt[5]):
        fwhm_large = np.abs(popt[4])
        fwhm_small = np.abs(popt[5])
    else:
        fwhm_large = np.abs(popt[5])
        fwhm_small = np.abs(popt[4])

    c = '/'
    index_of_slash = [pos for pos, char in enumerate(path_list[0]) if char == c]
    index = index_of_slash[-1]
    file_name = path_list[0][index + 1:]

    df = pd.DataFrame({'Date': [file_name],
                       'CCD': 'N/A',
                       'observ_num': 'N/A',
                       'probenum': 'N/A',
                       'S0': popt[0],
                       'S1': popt[1],
                       'x0': popt[2],
                       'y0': popt[3],
                       'fwhm_semi_major': fwhm_large,
                       'fwhm_semi_minor': fwhm_small,
                       'angle': popt[6],
                       'beta': popt[7],
                       'wavelength': np.nanmedian(band)})

    del x, y

    # Plotting
    flux = flux_image
    x, y = coords[0], coords[1] #
    x_linear, y_linear = np.arange(flux_image.shape[0]).astype(float), np.arange(flux_image.shape[1]).astype(float)
    fit_flux = elliptical_moffat_f(x,y,
                                   popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7])

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))

    max_v = np.max(flux.ravel())
    min_v = np.min(flux.ravel())

    im0 = ax[0].scatter(x.ravel(), y.ravel(), c=flux.ravel(), vmin=min_v, vmax=max_v, cmap='Oranges')
    ax[0].contour(x_linear, y_linear, fit_flux, 8, colors='w')
    cb0 = plt.colorbar(im0, ax=ax[0], fraction=0.047)
    cb0.ax.locator_params(nbins=5)
    cb0.set_label('flux')
    ax[0].set_title('data')

    im1 = ax[1].scatter(x.ravel(), y.ravel(), c=fit_flux, vmin=min_v, vmax=max_v, cmap='Oranges')
    cb1 = plt.colorbar(im1, ax=ax[1], fraction=0.047)
    cb1.ax.locator_params(nbins=5)
    cb1.set_label('flux')
    ax[1].set_title('fit_data')

    max_res = np.max((flux - fit_flux) / flux)

    im2 = ax[2].scatter(x.ravel(), y.ravel(), c=(flux - fit_flux) / flux, cmap='RdYlBu_r', vmin=-0.3, vmax=0.3)
    cb2 = plt.colorbar(im2, ax=ax[2], fraction=0.047)
    cb2.ax.locator_params(nbins=5)
    cb2.set_label('(data - fit_data)/data')
    ax[2].set_title('residual')
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.8,
                        hspace=0.4)
    plt.tight_layout()
    plt.savefig(f"Moffat_Elliptical_2D_YM_{file_name}.png", dpi=300)
    plt.close()

    return df


"""
    DIFFERENT MODEL FITTING ROUTINES DEALING WITH RSS FRAMES
        * most of them do not integrate over the pixels
"""
def fit_integrated_moffat_func_and_get_parameters_from_rss(path_list, probenum, n_chunk, band=None):
    c = '/'
    index_of_slash = [pos for pos, char in enumerate(path_list[0]) if char == c]
    index = index_of_slash[-1]
    file_name = path_list[0][index + 1:]

    # Get data and chunk
    chunked_data = read_chunked_data(path_list, probenum, n_chunk=n_chunk, band=band)
    flux_image, var_image, _x, _y = np.array(chunked_data['data']), np.array(chunked_data['variance']), \
                                  np.array(chunked_data['xfibre']), np.array(chunked_data['yfibre'])

    # Get the collapsed image over the wave band
    x, y  = np.linspace(_x.min(), _x.max(), 100), np.linspace(_y.min(), _y.max(), 100)
    x_mesh, y_mesh = np.meshgrid(x, y)

    # Using the moffat fitting function in fluxcal
    elliptical = False  # Moffat model fitted is an elliptical or circular
    params, sigma = fit_moffat_to_image(flux_image, np.sqrt(var_image), _x, _y,
                                        elliptical=elliptical, background=False,
                                        elliptical_f=False, rss=True)

    # Get the fitted model
    if elliptical:
        fit_flux = moffat_elliptical(x_mesh, y_mesh, *params)
        prPurple(f"Best-fitting params (ELLIPTICAL MOFFAT) from the RSS frame {file_name} are alpha1={params[0]}, alpha2={params[1]}, "
                 f"rho={params[2]}, beta={params[3]}, x00={params[4]}, y00={params[5]}, intensity={params[6]}")
    else:
        fit_flux = moffat_circular(x_mesh, y_mesh, *params)
        prPurple(f"Best-fitting params (CIRCULAR MOFFAT) from the RSS frame {file_name} are ALPHA={params[0]}, BETA={params[1]}, "
                 f"X00={params[2]}, Y00={params[3]}, INTENSITY={params[4]}, LAMBDA={band}")

    # Calculate the FWHM and create a data frame
    if elliptical:
        if np.abs(params[0]) > np.abs(params[1]):
            fwhm_large = params[0] * 2.0 * np.sqrt(2. ** (1. / params[3]) - 1.)
            fwhm_small = params[1] * 2.0 * np.sqrt(2. ** (1. / params[3]) - 1.)
        else:
            fwhm_large = params[1] * 2.0 * np.sqrt(2. ** (1. / params[3]) - 1.)
            fwhm_small = params[0] * 2.0 * np.sqrt(2. ** (1. / params[3]) - 1.)

        df = pd.DataFrame({'Date': [file_name],
                           'CCD': 'N/A',
                           'observ_num': 'N/A',
                           'probenum': 'N/A',
                           'alpha1': params[0],
                           'alpha2': params[1],
                           'rho': params[2],
                           'beta': params[3],
                           'x0': params[4],
                           'y0': params[5],
                           'intensity': params[6],
                           'fwhm_semi_major': fwhm_large,
                           'fwhm_semi_minor': fwhm_small,
                           'wavelength': np.nanmedian(band)})
    else:
        fwhm_large = params[0] * 2.0 * np.sqrt(2. ** (1. / params[1]) - 1.)

        df = pd.DataFrame({'Date': [file_name],
                           'CCD': 'N/A',
                           'observ_num': 'N/A',
                           'probenum': 'N/A',
                           'alpha1': params[0],
                           'alpha2': 'N/A',
                           'rho': 'N/A',
                           'beta': params[1],
                           'x0': params[2],
                           'y0': params[3],
                           'intensity': params[4],
                           'fwhm_semi_major': fwhm_large,
                           'fwhm_semi_minor': 'N/A',
                           'wavelength': np.nanmedian(band)})


    # Plotting......
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    max_v = np.max(flux_image)
    min_v = np.min(flux_image)

    ax[0].set(xlabel='Xfibre [arcsec]', ylabel='Yfibre [arcsec]', title='Data')
    im0 = ax[0].scatter(_x, _y, c=flux_image, vmin=min_v, vmax=max_v, cmap='Oranges')
    ax[0].contour(x_mesh, y_mesh, fit_flux, 4, colors='k')
    cb0 = plt.colorbar(im0, ax=ax[0], fraction=0.047)
    cb0.ax.locator_params(nbins=5)
    cb0.set_label('Flux')

    del fit_flux
    if elliptical:
        fit_flux = moffat_elliptical(_x, _y, *params)
    else:
        fit_flux = moffat_circular(_x, _y, *params)

    ax[1].set(xlabel='Xfibre [arcsec]', ylabel='Yfibre [arcsec]', title='Fitted Data')
    im1 = ax[1].scatter(_x.ravel(), _y.ravel(), c=fit_flux, vmin=min_v, vmax=max_v, cmap='Oranges')
    cb1 = plt.colorbar(im1, ax=ax[1], fraction=0.047)
    cb1.ax.locator_params(nbins=5)
    cb1.set_label('Flux')

    max_res = np.max((flux_image - fit_flux) / flux_image)

    ax[2].set(xlabel='Xfibre [arcsec]', ylabel='Yfibre [arcsec]', title='Residual')
    im2 = ax[2].scatter(_x.ravel(), _y.ravel(), c=(flux_image - fit_flux) / flux_image, cmap='RdYlBu_r', vmin=-0.3, vmax=0.3)
    cb2 = plt.colorbar(im2, ax=ax[2], fraction=0.047)
    cb2.ax.locator_params(nbins=5)
    cb2.set_label('(Data - Fit_Data)/Data')
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.8,
                        hspace=0.4)

    plt.tight_layout()
    plt.savefig(f"RSS_Moffat_elliptical_{elliptical}_integrated_{file_name}.png", dpi=300)
    plt.close()

    return df


def fit_ellip_moffat_f_model_and_get_parameters(path_list, probenum, n_chunk):
    """
    read data and fit by model elliptical moffat -- Relies on "chuck_data" in fluxcal2

    -----
    return: pd.DataFrame with [ S0, S1, x0, y0, fwhm_large, fwhm_small, angle, beta ,wavelength ]
    """

    S0_list = []
    S1_list = []
    x0_list = []
    y0_list = []
    fwhm_large_list = []
    fwhm_small_list = []
    angle_list = []
    beta_list = []
    wave_list = []

    chunked_data = read_chunked_data(path_list, probenum, n_chunk=n_chunk)

    for i in range(chunked_data['data'].shape[1]):
        xdata = np.vstack((chunked_data['xfibre'], chunked_data['yfibre']))
        ydata = chunked_data['data'][:, i]
        popt, pcov = curve_fit(f=elliptical_moffat_f_fit, xdata=xdata, ydata=ydata,
                               p0=init_guess_ellip_moffat_f(chunked_data['data'][:, i], chunked_data['xfibre'],
                                                            chunked_data['yfibre']), maxfev=5000)

        if np.abs(popt[4]) > np.abs(popt[5]):
            fwhm_large = np.abs(popt[4])
            fwhm_small = np.abs(popt[5])
        else:
            fwhm_large = np.abs(popt[5])
            fwhm_small = np.abs(popt[4])

        S0_list.append(popt[0])
        S1_list.append(popt[1])
        x0_list.append(popt[2])
        y0_list.append(popt[3])
        fwhm_large_list.append(fwhm_large)
        fwhm_small_list.append(fwhm_small)
        angle_list.append(popt[6])
        beta_list.append(popt[7])
        wave_list.append(chunked_data['wavelength'][i])

    c = '/'
    index_of_slash = [pos for pos, char in enumerate(path_list[0]) if char == c]
    index = index_of_slash[-1]
    file_name = path_list[0][index + 1:]

    df = pd.DataFrame({'Date': [file_name[:5]] * len(S1_list),
                       'CCD': [file_name[5:6]] * len(S1_list),
                       'observ_num': [file_name[6:10]] * len(S1_list),
                       'probenum': [probenum] * len(S1_list),
                       'S0': S0_list,
                       'S1': S1_list,
                       'x0': x0_list,
                       'y0': y0_list,
                       'fwhm_large': fwhm_large_list,
                       'fwhm_small': fwhm_small_list,
                       'angle': angle_list,
                       'beta': beta_list,
                       'wavelength': wave_list})

    return df


"""
    CHUNKING DATA FOR RSS FRAME FITTING FUNCTIONS
"""
def read_chunked_data(path_list, probenum, n_drop=None, n_chunk=None,
                      sigma_clip=None, band=None):
    """Read flux from a list of files, collapse over a given wavelength and combine."""
    if band[0] < 6000.0:
        prPurple(f"--> Band[0] is {band[0]}, Choosing the blue ccd: {path_list[0]}")
        path = path_list[0]
    else:
        prPurple(f"--> Band[1] is {band[1]}, Choosing the red ccd: {path_list[1]}")
        path = path_list[1]

    if isinstance(path_list, str):
        path_list = [path_list]

    ifu = IFU(path, probenum, flag_name=False)
    fluxcal2.remove_atmosphere(ifu)
    data, variance, wavelength = chunk_data(ifu, n_drop=n_drop, n_chunk=n_chunk, sigma_clip=sigma_clip, band=band)

    # if i_file == 0:
    #     data = data_i
    #     variance = variance_i
    #     wavelength = wavelength_i
    # else:
    #     data = np.hstack((data, data_i))
    #     variance = np.hstack((variance, variance_i))
    #     wavelength = np.hstack((wavelength, wavelength_i))

    xfibre = ifu.xpos_rel * np.cos(np.deg2rad(np.mean(ifu.ypos)))
    yfibre = ifu.ypos_rel
    # Only keep unbroken fibres
    good_fibre = (ifu.fib_type == 'P')
    chunked_data = {'data': data[good_fibre],
                    'variance': variance[good_fibre],
                    'wavelength': wavelength,
                    'xfibre': xfibre[good_fibre],
                    'yfibre': yfibre[good_fibre]}
    return chunked_data


def chunk_data(ifu, n_drop=None, n_chunk=None, sigma_clip=None, band=None):
    """Condence a spectrum into slice over a given wavelength."""
    n_pixel = ifu.naxis1
    n_fibre = len(ifu.data)

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

    lambda_mask = (ifu.lambda_range > band[0]) & (ifu.lambda_range < band[1])

    data = data[:, lambda_mask]
    variance = ifu.var[:, lambda_mask]
    wavelength = ifu.lambda_range[lambda_mask]

    data = np.nanmean(data, axis=1)
    variance = (np.nansum(variance, axis=1) /
                np.sum(np.isfinite(variance), axis=1)**2)
    # Replace any remaining NaNs with 0.0; not ideal but should be very rare
    bad_data = ~np.isfinite(data)
    data[bad_data] = 0.0
    variance[bad_data] = np.inf
    wavelength = np.median(wavelength, axis=0)
    return data, variance, wavelength


"""
    COMPARISON FUNCTIONS + OTHER
"""
def ellip_moffat_f_compare(flux, sigma, x, y, savePath=None):
    xdata = np.vstack((x, y))
    ydata = flux

    popt, pcov = curve_fit(f=elliptical_moffat_f_fit, xdata=xdata, ydata=ydata, sigma=sigma,
                           p0=init_guess_ellip_moffat_f(flux, x, y), maxfev=5000)

    fit_flux = elliptical_moffat_f(x, y, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7])

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))

    max_v = np.max(flux)
    min_v = np.min(flux)

    im0 = ax[0].scatter(x, y, c=flux, vmin=min_v, vmax=max_v, cmap='Oranges')
    cb0 = plt.colorbar(im0, ax=ax[0], fraction=0.047)
    cb0.ax.locator_params(nbins=5)
    cb0.set_label('flux')
    ax[0].set_title('data')

    im1 = ax[1].scatter(x, y, c=fit_flux, vmin=min_v, vmax=max_v, cmap='Oranges')
    cb1 = plt.colorbar(im1, ax=ax[1], fraction=0.047)
    cb1.ax.locator_params(nbins=5)
    cb1.set_label('flux')
    ax[1].set_title('fit_data')

    max_res = np.max((flux - fit_flux) / flux)

    im2 = ax[2].scatter(x, y, c=(flux - fit_flux) / flux, cmap='RdYlBu_r', vmin=-0.3, vmax=0.3)
    cb2 = plt.colorbar(im2, ax=ax[2], fraction=0.047)
    cb2.ax.locator_params(nbins=5)
    cb2.set_label('(data - fit_data)/data')
    ax[2].set_title('residual')
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.8,
                        hspace=0.4)
    plt.tight_layout()
    if savePath is not None:
        plt.savefig(savePath, dpi=300)

    # plt.suptitle(str(magpiid))
    plt.show()

    return


def fit_moffat_to_image(image, noise, xfibpos=None, yfibpos=None, elliptical=False, background=False, elliptical_f=False, rss=False):
    """
    Fit a Moffat profile to an image, optionally allowing ellipticity.
    MLPG (25/10/2024): adding "& (noise > 0.0)" to L672

    """
    fit_pix = np.isfinite(image) & np.isfinite(noise) & (noise > 0.0) # Updated to include noise > 0.0
    if rss:
        coords = [xfibpos, yfibpos]
        x00 = xfibpos.min() + (0.5 * (xfibpos.max() - xfibpos.min()))
        y00 = yfibpos.min() + (0.5 * (yfibpos.max() - yfibpos.min()))
    else:
        coords = np.meshgrid(np.arange(image.shape[0]),
                             np.arange(image.shape[1]))
        x00 = 0.5 * (image.shape[0] - 1)
        y00 = 0.5 * (image.shape[1] - 1)
    alpha0 = 4.0
    beta0 = 4.0
    intensity0 = np.nansum(image)
    if elliptical:
        p0 = [alpha0, alpha0, 0.0, beta0, x00, y00, intensity0]
    else:
        p0 = [alpha0, beta0, x00, y00, intensity0]

    if elliptical_f:
        p0 = init_guess_ellip_moffat_f(image, np.arange(image.shape[0]), np.arange(image.shape[1]))
    if background:
        p0.append(0.0)
    def fit_function(p):
        model = moffat_integrated(
            coords[0], coords[1], p, elliptical=elliptical,
            background=background, good=fit_pix, elliptical_f=elliptical_f)
        return (model - image[fit_pix]) / noise[fit_pix]

    result = leastsq(fit_function, p0, full_output=True)
    params = result[0]
    if result[1] is None:
        sigma = None
    else:
        reduced_chi2 = np.sum(fit_function(params)**2 / (np.sum(fit_pix) - 1))
        n_params = len(params)
        sigma = np.sqrt(result[1][np.arange(n_params), np.arange(n_params)] /
                        reduced_chi2)
    return params, sigma


def moffat_integrated(x, y, params, elliptical=False, background=False,
                      good=None, elliptical_f=False, pix_size=1.0, n_sub=10):
    """Return a Moffat profile, integrated over pixels."""
    if good is None:
        good = np.ones(x.size, bool)
        good.shape = x.shape
    n_pix = np.sum(good)
    x_flat = x[good]
    y_flat = y[good]
    delta = pix_size * (np.arange(float(n_sub)) / n_sub)
    delta -= np.mean(delta)
    x_sub = (np.outer(x_flat, np.ones(n_sub**2)) +
             np.outer(np.ones(n_pix), np.outer(delta, np.ones(n_sub))))
    y_sub = (np.outer(y_flat, np.ones(n_sub**2)) +
             np.outer(np.ones(n_pix), np.outer(np.ones(n_sub), delta)))

    if background:
        params_sub = params[:-1]
    else:
        params_sub = params

    if elliptical:
        moffat_sub = moffat_elliptical(x_sub, y_sub, *params_sub)
    elif elliptical_f:
        moffat_sub = elliptical_moffat_f(x_sub, y_sub, *params_sub)
    else:
        moffat_sub = moffat_circular(x_sub, y_sub, *params_sub)

    moffat = np.mean(moffat_sub, 1)

    if background:
        moffat += params[-1]

    return moffat


def get_coords(header, axis):
    """Return coordinates for a given axis from a header."""
    axis_str = str(axis)
    naxis = header['NAXIS' + axis_str]
    crpix = header['CRPIX' + axis_str]
    cdelt = header['CDELT' + axis_str]
    crval = header['CRVAL' + axis_str]
    coords = crval + cdelt * (np.arange(naxis) + 1.0 - crpix)
    return coords


"""
    CvD checking functions used in fluxcal2 in "derive_transfer_function"
"""
def interp_cvd_for_rss(cen_data, f_cvd, wavelength, plateCentre=None):
    # MLPG - 23/09/2023: Adding this function which interpolate the information from cvd_model
    # xfibre and yfibre positions in arcsec relative to the field centre, assumed to be at the plate centre
    if plateCentre is None: plateCentre = 0.0
    cvd_pos = cen_data + np.polyval(f_cvd, wavelength)

    return cvd_pos

def check_centroids(data, variance, wavelength, xfib_cvd, yfib_cvd, wavebin=None):
    nlambda = len(wavelength)
    lambda_bins = np.arange(0, nlambda, wavebin)

    _data, _var, _wavelength, _xfib_cvd, _yfib_cvd = [], [], [], [], []
    for ibin in range(1, len(lambda_bins)):
        start = lambda_bins[ibin-1]
        end   = lambda_bins[ibin]

        data_tmp = np.nanmean(data[:, start:end], axis=1)
        var_tmp = (np.nansum(variance[:, start:end], axis=1) /
                   np.sum(np.isfinite(variance[:, start:end]), axis=1) ** 2.0)

        # Replace any remaining NaNs with 0.0; not ideal but should be very rare
        bad_data = ~np.isfinite(data_tmp)
        data_tmp[bad_data] = 0.0
        var_tmp[bad_data] = np.inf

        _data.append( data_tmp )
        _var.append( var_tmp )
        _wavelength.append( np.median(wavelength[start:end], axis=0) )
        _xfib_cvd.append( np.nanmean(xfib_cvd[:, start:end], axis=1) )
        _yfib_cvd.append( np.nanmean(yfib_cvd[:, start:end], axis=1) )

        del data_tmp, var_tmp, start, end

    return np.array(_data).T, np.array(_var).T, np.array(_wavelength), np.array(_xfib_cvd).T, np.array(_yfib_cvd).T


def fit_integrated_moffat_func_and_get_parameters_for_primary(path_list, flux_image, var_image, wave, _x, _y, xref=None, yref=None):
    c = '/'
    index_of_slash = [pos for pos, char in enumerate(path_list[0]) if char == c]
    index = index_of_slash[-1]
    file_name = path_list[0][index + 1:]

    # Get the collapsed image over the wave band
    # x, y  = np.linspace(_x.min(), _x.max(), 100), np.linspace(_y.min(), _y.max(), 100)
    x, y  = np.linspace(xref.min(), xref.max(), 100), np.linspace(yref.min(), yref.max(), 100)
    x_mesh, y_mesh = np.meshgrid(x, y)

    # Using the moffat fitting function in fluxcal
    elliptical = False # Moffat model fitted is an elliptical or circular
    params, sigma = fit_moffat_to_image(flux_image, np.sqrt(var_image), _x, _y,
                                        elliptical=elliptical, background=False,
                                        elliptical_f=False, rss=True)
    # Get the fitted model
    if elliptical:
        fit_flux = moffat_elliptical(x_mesh, y_mesh, *params)
        prPurple(f"Best-fitting params from the RSS frame {file_name} are alpha1={params[0]}, alpha2={params[1]}, "
                 f"rho={params[2]}, beta={params[3]}, x00={params[4]}, y00={params[5]}, intensity={params[6]}")
    else:
        fit_flux = moffat_circular(x_mesh, y_mesh, *params)
        prPurple(f"Best-fitting params from the RSS frame {file_name} are ALPHA={params[0]}, BETA={params[1]}, "
                 f"X00={params[2]}, Y00={params[3]}, INTENSITY={params[4]}, LAMBDA={wave}")


    # Calculate the FWHM and create a data frame
    if elliptical:
        if np.abs(params[0]) > np.abs(params[1]):
            fwhm_large = params[0] * 2.0 * np.sqrt(2. ** (1. / params[3]) - 1.)
            fwhm_small = params[1] * 2.0 * np.sqrt(2. ** (1. / params[3]) - 1.)
        else:
            fwhm_large = params[1] * 2.0 * np.sqrt(2. ** (1. / params[3]) - 1.)
            fwhm_small = params[0] * 2.0 * np.sqrt(2. ** (1. / params[3]) - 1.)

        central_x = xref.min() + (xref.max() - xref.min())/2.0
        central_y = yref.min() + (yref.max() - yref.min())/2.0
        radius = np.sqrt( (params[4] - central_x)**2.0 + (params[5] - central_y)**2.0 )
        df = pd.DataFrame({'Date': [file_name],
                           'CCD': 'N/A',
                           'observ_num': 'N/A',
                           'probenum': 'N/A',
                           'alpha1': params[0],
                           'alpha2': params[1],
                           'rho': params[2],
                           'beta': params[3],
                           'x0': params[4],
                           'y0': params[5],
                           'intensity': params[6],
                           'fwhm_semi_major': fwhm_large,
                           'fwhm_semi_minor': fwhm_small,
                           'radius_from_centre': radius,
                           'wavelength': wave})
    else:
        fwhm_large = params[0] * 2.0 * np.sqrt(2. ** (1. / params[1]) - 1.)

        df = pd.DataFrame({'Date': [file_name],
                           'CCD': 'N/A',
                           'observ_num': 'N/A',
                           'probenum': 'N/A',
                           'alpha1': params[0],
                           'alpha2': 'N/A',
                           'rho': 'N/A',
                           'beta': params[1],
                           'x0': params[2],
                           'y0': params[3],
                           'intensity': params[4],
                           'fwhm_semi_major': fwhm_large,
                           'fwhm_semi_minor': 'N/A',
                           'wavelength': wave})


    # Plotting......
    fig, ax = plt.subplots(1, 3, figsize=(9, 3), constrained_layout = True)

    max_v = np.max(flux_image)
    min_v = np.min(flux_image)

    ax[0].set(xlabel='Xfibre [arcsec]', ylabel='Yfibre [arcsec]', title=f"Data {params[3]}")
    im0 = ax[0].scatter(xref, yref, c=flux_image, vmin=min_v, vmax=max_v, cmap='Oranges')
    ax[0].contour(x_mesh, y_mesh, fit_flux, 4, colors='k')
    cb0 = plt.colorbar(im0, ax=ax[0], fraction=0.047)
    cb0.ax.locator_params(nbins=5)
    cb0.set_label('Flux', fontsize=8)
    cb0.ax.tick_params(labelsize=8)
    ax[0].tick_params(axis = 'both', labelsize = 8)
    ax[0].xaxis.get_label().set_fontsize(8)
    ax[0].yaxis.get_label().set_fontsize(8)

    del fit_flux
    if elliptical:
        fit_flux = moffat_elliptical(xref, yref, *params)
    else:
        fit_flux = moffat_circular(xref, yref, *params)

    ax[1].set(xlabel='Xfibre [arcsec]', ylabel='Yfibre [arcsec]', title='Fitted Data')
    im1 = ax[1].scatter(xref.ravel(), yref.ravel(), c=fit_flux, vmin=min_v, vmax=max_v, cmap='Oranges')
    cb1 = plt.colorbar(im1, ax=ax[1], fraction=0.047)
    cb1.ax.locator_params(nbins=5)
    cb1.set_label('Flux', fontsize=8)
    cb1.ax.tick_params(labelsize=8)
    ax[1].tick_params(axis='both', labelsize=8)
    ax[1].xaxis.get_label().set_fontsize(8)
    ax[1].yaxis.get_label().set_fontsize(8)
    max_res = np.max((flux_image - fit_flux) / flux_image)

    ax[2].set(xlabel='Xfibre [arcsec]', ylabel='Yfibre [arcsec]', title='Residual')
    im2 = ax[2].scatter(xref.ravel(), yref.ravel(), c=(flux_image - fit_flux) / flux_image, cmap='RdYlBu_r', vmin=-0.3, vmax=0.3)
    cb2 = plt.colorbar(im2, ax=ax[2], fraction=0.047)
    cb2.ax.locator_params(nbins=5)
    cb2.set_label('(Data - Fit_Data)/Data', fontsize=8)
    cb2.ax.tick_params(labelsize=8)
    ax[2].tick_params(axis='both', labelsize=8)
    ax[2].xaxis.get_label().set_fontsize(8)
    ax[2].yaxis.get_label().set_fontsize(8)

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.8,
                        hspace=0.4)

    # plt.tight_layout()
    plt.savefig(f"Primary_Moffat_elliptical_integrated_{file_name}_{np.round(wave)}.png", dpi=300)
    plt.close()

    return df


