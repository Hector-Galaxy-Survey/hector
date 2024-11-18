"""
    The CvD model implemented by Madusha Gunawardhana (madusha.gunawardhana@sydney.edu.au)
"""

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
#from hop.hexabundle_allocation.hector import constants

from ..utils.term_colours import *
from ..utils.ifu import IFU
from ..utils import constants # # From Hector package Sam V. developed
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

import hector
hector_path = str(hector.__path__[0]) + '/'

# Get the astropy version as a tuple of integers
ASTROPY_VERSION = tuple(int(x) for x in ASTROPY_VERSION.split('.'))
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
    """
    Generate a subgrid of points within a fibre.
        Copied from the fluxcal2.py
    """
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
    if wt_profile:
        wsub = 0.5*erfc((radius-fibre_radius*0.8)*4.0)
        wnorm = float(np.size(radius))/np.sum(wsub)
        wsub = wsub * wnorm
    else:
        # or unit weighting:
        wsub = np.ones(np.size(xsub))
    return xsub, ysub, wsub


XSUB, YSUB, WSUB= generate_subgrid(FIBRE_RADIUS)
N_SUB = len(XSUB)

MICRONS_TO_ARCSEC = 1.6 / 105.0

# Robot center positions [in microns]
robot_centre_in_mic = [constants.robot_center_x * 1.0E3,
                       constants.robot_center_y * 1.0E3]
plate_radius_in_mic = constants.HECTOR_plate_radius  * 1.0E3
robotCentreX_in_mic, robotCentreY_in_mic = robot_centre_in_mic[1], robot_centre_in_mic[0]  # switch, since y is x, and x is y

# Robot center positions [in arcseconds]
plate_radius_in_arcsec = constants.HECTOR_plate_radius  * 1.0E3 * MICRONS_TO_ARCSEC
robot_centre_in_arcsec = [constants.robot_center_x * 1.0E3 * MICRONS_TO_ARCSEC,
                          constants.robot_center_y * 1.0E3 * MICRONS_TO_ARCSEC]
robotCentreX_in_arcsec, robotCentreY_in_arcsec = robot_centre_in_arcsec[1], robot_centre_in_arcsec[0]  # switch, since y is x, and x is y


def assert_sky_coord_sign_check(ifu):
    """
    From IFU class within the manager:
        [xpos_rel, ypos_rel] in arcseconds relative to the field centre
        [x_micron, y_micron] in microns relative to the robot centre.

        Below is how the signs change given the location on the plate
            # X/Y-VALUES: top-left  (x,y) = (+ve, -ve) in arcseconds / (+ve, +ve) in microns
            #             top-right (x,y) = (-ve, -ve) in arcseconds / (-ve, +ve) in microns
            #             bot-left  (x,y) = (+ve, +ve) in arcseconds / (+ve, -ve) in microns
            #             bot-right (x,y) = (-ve, +ve) in arcseconds / (-ve, -ve) in microns
    """
    plateCentre_microns = [robotCentreX_in_mic, robotCentreY_in_mic]

    mean_xfibrePos = np.nanmean( ifu.xpos_rel * np.cos(np.deg2rad(np.mean(ifu.ypos))) ) # In arcsec
    mean_yfibrePos = np.nanmean( ifu.ypos_rel )

    mean_xfibreMic = np.nanmean( ifu.x_microns ) # In microns
    mean_yfibreMic = np.nanmean( ifu.y_microns ) # In microns

    if (ifu.mean_y <= plateCentre_microns[1]) & (ifu.mean_x <= plateCentre_microns[0]): # Top-left of the plate
        assert (mean_xfibrePos > 0) & (mean_yfibrePos < 0) & \
               (mean_xfibreMic > 0) & (mean_yfibreMic > 0), prRed(f"--> arcsecond/micron coordinate mismatch: Hexabundle {ifu.hexabundle_name[10]} is on the top-left of the plate. \n"
                                                                  f"Corrdinates should be (+ve, -ve) in arcsec, (+ve, +ve) in microns")
    elif (ifu.mean_y <= plateCentre_microns[1]) & (ifu.mean_x > plateCentre_microns[0]):  # Top-right of the plate
        assert (mean_xfibrePos < 0) & (mean_yfibrePos < 0) & \
               (mean_xfibreMic < 0) & (mean_yfibreMic > 0), prRed(f"--> arcsecond/micron coordinate mismatch: Hexabundle {ifu.hexabundle_name[10]} is on the top-right of the plate. \n"
                                                                  f"Corrdinates should be (-ve, -ve) in arcsec, (-ve, +ve) in microns")
    elif (ifu.mean_y > plateCentre_microns[1]) & (ifu.mean_x <= plateCentre_microns[0]):  # bottom-left of the plate
        assert (mean_xfibrePos > 0) & (mean_yfibrePos > 0) & \
               (mean_xfibreMic > 0) & (mean_yfibreMic < 0), prRed(f"--> arcsecond/micron coordinate mismatch: Hexabundle {ifu.hexabundle_name[10]} is on the bottom-left of the plate. \n"
                                                                  f"Corrdinates should be (+ve, +ve) in arcsec, (+ve, -ve) in microns")
    elif (ifu.mean_y > plateCentre_microns[1]) & (ifu.mean_x > plateCentre_microns[0]):  # bottom-right of the plate
        assert (mean_xfibrePos < 0) & (mean_yfibrePos > 0) & \
               (mean_xfibreMic < 0) & (mean_yfibreMic < 0), prRed(f"--> arcsecond/micron coordinate mismatch: Hexabundle {ifu.hexabundle_name[10]} is on the bottom-right of the plate. \n"
                                                                  f"Corrdinates should be (-ve, +ve) in arcsec, (-ve, -ve) in microns")
    return


def get_cvd_parameters(path_list, probenum, check_against_cvd_model=False, moffat_params=None, psf_parameters_array=None, wavelength=None):
    """
    The main function to get the CvD corrections
    """

    if isinstance(path_list, str):
        path_list = [path_list]
    for i_file, path in enumerate(path_list):
        ifu = IFU(path, probenum, flag_name=False)

    dest_path = path[0:path.find('/reduced')] + '/Optical_Model'
    if not os.path.isdir(dest_path):
        os.makedirs(dest_path)

    def record_bad_data():
        fname = open(f"{dest_path}/debug_primary_standards.txt", "a")  # safer than w mode
        fname.write(f"{path_list}   (moffat faliure)\n")
        # Close opened file
        fname.close()

        raise ValueError(prRed(f"---> Moffat-fitting Faliure!!! (*** debug_cvd ***)"))


    def optical_model_function(meanX, meanY, _ifu=ifu, _check_against_cvd_model=False, _moffat_params=None, _psf_parameters_array=None, _wavelength=None):
        """
        Call the Hector Optical Model, which currently assumes that the optical_centre = physical_plate_centre
        """
        # Best-fitting plate centre from CVD modelling (originally, robotCentreX, robotCentreY assumed as the plate centre coordinates)
        # plateCentre_microns = [325508.93894568, 316861.50305806] # bestfit_plateCentre_microns
        # Coefficients relative to lambda=6000Ang.
        # distortModel_coeffs = [ -5.36299973e-44,  1.74032192e-39, -9.28129572e-36, -1.04070566e-32,
        #                         8.05548970e-29, -3.70053127e-26,  1.33691248e-21, -1.86385976e-17,
        #                         6.14872408e-14, -5.95792839e-11,  8.96376563e-07, -3.20239518e-03 ]

        # Check for coordinate mis-matches to ensure the cvd correction is applied correctly
        assert_sky_coord_sign_check(_ifu)

        plateCentre_microns = [robotCentreX_in_mic, robotCentreY_in_mic]
        # Coefficients relative to lambda=6000Ang.
        distortModel_coeffs = [ 1.92596410e-42, -2.57109951e-38,  8.41650182e-35, -1.24150088e-31,
                                1.66263366e-27, -5.46127489e-24,  3.17913580e-21, -4.46188558e-17,
                                1.52220616e-13, -7.05097402e-11,  1.05661104e-06, -3.77980140e-03]

        # Modelling wavelength range, in bins of 100A
        wave = np.arange(3000.0, 8000.0, 100.0) # Create a lambda array

        # The distortion model functional fit
        A7 = (distortModel_coeffs[0] * wave ** 2.0) + (distortModel_coeffs[1] * wave) + distortModel_coeffs[2]
        A5 = (distortModel_coeffs[3] * wave ** 2.0) + (distortModel_coeffs[4] * wave) + distortModel_coeffs[5]
        A3 = (distortModel_coeffs[6] * wave ** 2.0) + (distortModel_coeffs[7] * wave) + distortModel_coeffs[8]
        A1 = (distortModel_coeffs[9] * wave ** 2.0) + (distortModel_coeffs[10] * wave) + distortModel_coeffs[11]

        # alphar - radius from the plate Centre, which is a vector, with the top of the plate (in robot orientation) being -ve,
        xval, yval = meanX - plateCentre_microns[0], meanY - plateCentre_microns[1]
        r_plateCentre_probeCentre = np.sqrt( xval ** 2.0 + yval ** 2.0 )

        prCyan(f"\n ----> Probe (MeanX, MeanY) = ({meanX, meanY}) \n PlateCentr (X, Y) = ({plateCentre_microns})")
        if meanY < plateCentre_microns[1]:
            alphar = r_plateCentre_probeCentre * -1.0
            prLightPurple(f"--> Hexa {ifu.hexabundle_name[5]} placed on the top half of the plate (i.e. MeanY < plateCentreY) in robot coor \n")
        else:
            alphar = r_plateCentre_probeCentre
            prLightPurple(f"--> Hexa {ifu.hexabundle_name[5]} placed on the bottom half of the plate (i.e. MeanY > plateCentreY) in robot coor \n")

        Angle = np.rad2deg(np.arctan2(yval, xval)) # Angle is anti-clockwise from the plate Centre
        if meanY < plateCentre_microns[1]:
            prLightGray(f"Angle anti-clockwise from +ve x-axis about the plateCentre (i.e. -ve angle) = {Angle} \n")
        else:
            prLightGray(f"Angle clockwise from +ve x-axis about the plateCentre (i.e. +ve angle) = {Angle} \n")

        Angle = 180.0 - np.abs(Angle) # But we need the angle about the hexabundle (so, subtract from 180)
        prLightGray(f"Angle about the hexabundleCentre = {Angle} \n")


        A = (A7 * alphar ** 7.0) + (A5 * alphar ** 5.0) + (A3 * alphar ** 3.0) + (A1 * alphar)
        fractional_r = A / r_plateCentre_probeCentre

        deltaX, deltaY = np.zeros(len(A)) - 99999.9, np.zeros(len(A)) - 99999.9
        mask = A < 0
        if alphar > 0: # Bottom-half of the plate (meanY > plateCentre) in robot coordinates
            Angle = Angle * -1.0
            deltaX[mask] = np.abs(A[mask]) * np.cos(np.deg2rad(Angle))
            deltaX[~mask] = np.abs(A[~mask]) * np.cos(np.deg2rad(180.0 - np.abs(Angle)))

            deltaY[mask] = np.abs(A[mask]) * np.sin(np.deg2rad(Angle))
            deltaY[~mask] = np.abs(A[~mask]) * np.sin(np.deg2rad(180.0 - np.abs(Angle)))
        else:
            deltaX[mask] = np.abs(A[mask]) * np.cos(np.deg2rad(np.abs(Angle) - 180.0))
            deltaX[~mask] = np.abs(A[~mask]) * np.cos(np.deg2rad(Angle))

            deltaY[mask] = np.abs(A[mask]) * np.sin(np.deg2rad(np.abs(Angle) - 180.0))
            deltaY[~mask] = np.abs(A[~mask]) * np.sin(np.deg2rad(Angle))


        # ---> ************************** DEBUG PLOTS ************************** <---- #
        # DEBUG plot - plateview in microns
        if _check_against_cvd_model:
            wave = np.arange(3000.0, 8000.0, 100.0)  # Create a lambda array

            xfibre0 = ifu.xpos_rel * np.cos(np.deg2rad(np.mean(ifu.ypos)))  # In arcsec
            yfibre0 = ifu.ypos_rel
            xfibre, yfibre = yfibre0, xfibre0 # Switch x- and y-. This then matches with the rotated micron coordinates

            # Interpolate the cvd corrections [in microns]
            zx_mic = np.polyfit(np.array(wave), np.array(deltaX), 2)
            zy_mic = np.polyfit(np.array(wave), np.array(deltaY), 2)

            fig = py.figure(figsize=(10, 9.8))
            cmap = mpl.cm.get_cmap('rainbow')

            supltitle = f""
            ax = fig.add_subplot(1, 1, 1)
            ax.set_aspect('equal')

            ax.add_patch( Circle(xy=(robot_centre_in_mic[1], robot_centre_in_mic[0]),
                                 radius=plate_radius_in_mic, facecolor="#cccccc", edgecolor='#000000', zorder=-1))
            ax.plot(robot_centre_in_mic[1], robot_centre_in_mic[0], 'rx', markersize=12)

            # Draw vertical/horizontal lines across the plate
            ax.plot([robot_centre_in_mic[1]-plate_radius_in_mic, robot_centre_in_mic[1]+plate_radius_in_mic],
                    [robot_centre_in_mic[0], robot_centre_in_mic[0]], '-', color=[.5, .5, .5])
            ax.plot([robot_centre_in_mic[1], robot_centre_in_mic[1]],
                    [robot_centre_in_mic[0] - plate_radius_in_mic, robot_centre_in_mic[0] + plate_radius_in_mic], '-', color=[.5, .5, .5])

            ax.plot(meanX, meanY, 'ko', markersize=12, label='RobotCoor in mic')

            icmap = np.linspace(0, 1, len(_moffat_params['wavelength']))

            wave = _moffat_params['wavelength'].to_numpy()
            for ilambda in range(len(wave)):
                ax.plot(meanX + np.polyval(zx_mic, np.array(wave[ilambda])) * 1000.0,
                        meanY + np.polyval(zy_mic, np.array(wave[ilambda])) * 1000.0,
                        'x', color=cmap(icmap[ilambda]), markersize=12, lw=1,
                        label='CvD model correction applied to meanX/Y of bundle' if ilambda==5 else None)

            fig, (ax1, ax2) = py.subplots(2, 1, figsize=(9, 9))
            ax1.set(xlabel='x-rotated [micron]', ylabel='y-rotated [micron]',
                    title=f"{os.path.basename(path)}, Hexabundle {_ifu.hexabundle_name[5]}")
            ax2.set(xlabel='yfibre_pos [arcsec]=DEC', ylabel='xfibre_pos [arcsec]=RA')
            ax1.xaxis.get_label().set_fontsize(6); ax1.yaxis.get_label().set_fontsize(6)
            ax1.tick_params(axis='both', labelsize=6)
            ax2.xaxis.get_label().set_fontsize(6); ax2.yaxis.get_label().set_fontsize(6)
            ax2.tick_params(axis='both', labelsize=6)

            xCen_mic, yCen_mic, wave_revised, bad_data_count = [], [], [], 0
            for i in range(len(_moffat_params['wavelength'])):
                # Converts from arcseconds to microns (xref, yref) are switched since the moffat fitting is done without switching them
                tmpx, tmpy = coord_convert( _moffat_params['y0'].iloc[i],
                                            _moffat_params['x0'].iloc[i],
                                            xfibre, yfibre,  # Already switched, see L@263
                                            ifu.x_rotated, ifu.y_rotated,
                                            ifu.hexabundle_name, path_to_save=dest_path)
                # if tmpx is None: record_bad_data()
                if tmpx is None: bad_data_count += 1; continue

                ax1.plot(ifu.x_rotated, ifu.y_rotated, 'kx', ms=8)
                ax1.plot(tmpx, tmpy, 'o', ms=8, color=cmap(icmap[i]),
                         label="Direct calculation of \n centroid converted \n from arcsec to micron" if i == 5 else None)

                ax2.plot(xfibre, yfibre, 'kx', ms=8)
                ax2.plot(_moffat_params['y0'].iloc[i], _moffat_params['x0'].iloc[i], 'o', ms=8, color=cmap(icmap[i]),
                         label="Direct calculation of \n centroid converted \n [arcsec]" if i == 5 else None)

                xCen_mic.append(tmpx)
                yCen_mic.append(tmpy)
                wave_revised.append(_moffat_params['wavelength'].iloc[i])
                del tmpx, tmpy

            # If the fitting has failed for more than half of the wavelength slices considered in the moffat_params, then record the bad frame
            if bad_data_count >= len(_moffat_params['wavelength']): record_bad_data()

            ax1.legend(loc='best', prop={'size': 6}); ax2.legend(loc='best', prop={'size': 6})
            fig.tight_layout()
            figfile = f"{dest_path}/DEBUG_hexabundle_{_ifu.hexabundle_name[5]}_inmicron_{_ifu.primary_header['PLATEID']}_{os.path.basename(path)[:10]}.png"
            fig.savefig(figfile, bbox_inches='tight', pad_inches=0.3, dpi=150)
            py.close()


            xCen_mic, yCen_mic, wave_revised = np.array(xCen_mic).squeeze(), np.array(yCen_mic).squeeze(), np.array(wave_revised).squeeze()
            xref, yref = xCen_mic[48], yCen_mic[48]
            for i in range(len(wave_revised)):
                ax.plot(meanX + (xCen_mic[i] - xref) * 1000, meanY + (yCen_mic[i] - yref) * 1000,
                        'o', color=cmap(icmap[i]), markersize=4)


            # Add annuli information to the plate-view plot
            magenta_radius = 226. * 1000.
            yellow_radius = 196.05124 * 1000.
            green_radius = 147.91658 * 1000.
            blue_radius = 92.71721 * 1000.

            ax.add_patch(Circle((robot_centre_in_mic[1], robot_centre_in_mic[0]), blue_radius, color='lightblue', alpha=0.3))
            ax.add_patch(
                Wedge((robot_centre_in_mic[1], robot_centre_in_mic[0]), magenta_radius, 0, 360., width=30, color='indianred',
                      alpha=0.7))
            ax.add_patch(
                Wedge((robot_centre_in_mic[1], robot_centre_in_mic[0]), yellow_radius, 0, 360., width=30, color='gold',
                      alpha=0.7))
            ax.add_patch(
                Wedge((robot_centre_in_mic[1], robot_centre_in_mic[0]), green_radius, 0, 360., width=30, color='lightgreen',
                      alpha=0.7))

            fig.suptitle(f"Hector Optical Model: {supltitle} [micron]", fontsize=15)

            ax.invert_yaxis()
            # ax.invert_xaxis()
            ax.set_ylabel("Robot $x$ coordinate")
            ax.set_xlabel("Robot $y$ coordinate")

            py.tight_layout()
            figfile = f"{dest_path}/plateview_hexa{_ifu.hexabundle_name[5]}_inmicron_{_ifu.primary_header['PLATEID']}_compareWth_CvDmodel.png"  # save_files / f"plateViewAll_{config['file_prefix']}_Run{obs_number:04}"
            py.savefig(figfile, bbox_inches='tight', pad_inches=0.3, dpi=150)
            py.close()
            # py.show()


        # DEBUG plot - plateview in arcseconds
        if _check_against_cvd_model:
            wave = np.arange(3000.0, 8000.0, 100.0)  # Create a lambda array

            # Converts the unit of the CvD corrections to arcseconds
            zx = np.polyfit(np.array(wave), np.array(deltaX) * MICRONS_TO_ARCSEC, 2)
            zy = np.polyfit(np.array(wave), np.array(deltaY) * MICRONS_TO_ARCSEC, 2)

            xfibre0 = _ifu.xpos_rel * np.cos(np.deg2rad(np.mean(_ifu.ypos)))  # In arcsec
            yfibre0 = _ifu.ypos_rel
            # Switch x- and y- [in arcseconds]. This then matches with the rotated micron coordinates
            xfibre, yfibre = yfibre0, xfibre0

            fig = py.figure(figsize=(10, 9.8))
            cmap = mpl.cm.get_cmap('rainbow')

            supltitle = f""
            ax = fig.add_subplot(1, 1, 1)
            ax.set_aspect('equal')

            plateCentre = [0, 0]
            ax.add_patch(Circle(xy=(plateCentre[0], plateCentre[1]), radius=plate_radius_in_arcsec, facecolor="#cccccc", edgecolor='#000000', zorder=-1))
            ax.plot(plateCentre[0], plateCentre[1], 'rx', markersize=12)

            # Draw vertical/horizontal lines across the plate
            ax.plot([plateCentre[1] - plate_radius_in_arcsec, plateCentre[1] + plate_radius_in_arcsec],
                    [plateCentre[0], plateCentre[0]], '-', color=[.5, .5, .5])
            ax.plot([plateCentre[1], plateCentre[1]],
                    [plateCentre[0] - plate_radius_in_arcsec, plateCentre[0] + plate_radius_in_arcsec],
                    '-', color=[.5, .5, .5])

            # This is the definition of meanX- and meanY-micron from the ifu class
            meanX_arcsec = -1.0 * (meanX - robot_centre_in_mic[1]) * MICRONS_TO_ARCSEC
            meanY_arcsec = (robot_centre_in_mic[0] - meanY) * MICRONS_TO_ARCSEC

            # for ilambda in range(len(wave)):
                # X/Y-VALUES: top-left  (x,y) = (+ve, -ve) in arcseconds / (+ve, +ve) in microns
                #            top-right (x,y) = (-ve, -ve) in arcseconds / (-ve, +ve) in microns
                #            bot-left  (x,y) = (+ve, +ve) in arcseconds / (+ve, -ve) in microns
                #            bot-right (x,y) = (-ve, +ve) in arcseconds / (-ve, -ve) in microns

                # NOTE: x- and y- axes are swapped in the optical model (constructed in micron),
                #       i.e. the y-CvD correction needs to be applied to the xfibre positions,
                #       and x-CvD correction to the yfibre positions
                # ax.plot(np.nanmean(_xfibre) + np.polyval(zy, np.array(wave[ilambda])) * 1000.0,
                #         np.nanmean(_yfibre) + np.polyval(zx, np.array(wave[ilambda])) * 1000.0,
                #         's', color=cmap(icmap[ilambda]), markersize=8, alpha=.1, lw=1,
                #         label="Directly applying CvD model corrections (different waveArry)" if ilambda == 5 else None)
            ax.plot(meanX_arcsec, -1.0 * meanY_arcsec, 'ko', markersize=12, label='RobotCoor [converted to arcsec]')


            if _psf_parameters_array is not None:
                xcen0, ycen0 = list(_psf_parameters_array['xcen']), list(_psf_parameters_array['ycen'])
                # xcen0, ycen0 = np.array(xcen0), np.array(ycen0)
                icmap = np.linspace(0, 1, len(xcen0))
                for j in range(len(xcen0)):
                    ax.plot(np.nanmean(xcen0) + (xcen0[j] - np.nanmean(xcen0)) * 1000,
                            np.nanmean(ycen0) + (ycen0[j] - np.nanmean(ycen0)) * 1000,
                            'p', color=cmap(icmap[j]), markersize=8, alpha=.4, lw=1)

                    # prRed(f"model_based: {np.polyval(zy, wavelength[j])}, "
                    #       f"{np.polyval(zx, wavelength[j])}")
                    # prCyan(f"from manager: {xcen0[j] - np.nanmean(xcen0)}, {ycen0[j] - np.nanmean(ycen0)}")

            fig, (ax1, ax2) = py.subplots(2, 1, figsize=(9, 9))
            ax1.set(xlabel='x-rotated [micron]', ylabel='y-rotated [micron]',
                    title=f"{os.path.basename(path)}, Hexabundle {_ifu.hexabundle_name[5]}")
            ax2.set(xlabel='yfibre_pos [arcsec]=DEC', ylabel='xfibre_pos [arcsec]=RA')
            ax1.xaxis.get_label().set_fontsize(6); ax1.yaxis.get_label().set_fontsize(6); ax1.title.set_size(8)
            ax1.tick_params(axis='both', labelsize=6)
            ax2.xaxis.get_label().set_fontsize(6); ax2.yaxis.get_label().set_fontsize(6)
            ax2.tick_params(axis='both', labelsize=6)

            fig1, (ax3, ax4) = py.subplots(2, 1)
            ax3.set(xlabel='Lambda [Ang]', ylabel='DeltaX (Model - Measured)')
            ax3.tick_params(axis='both', labelsize=6)
            ax3.xaxis.get_label().set_fontsize(6); ax3.yaxis.get_label().set_fontsize(6)
            Y = np.array([[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]])
            ax3.plot( [_moffat_params['wavelength'].iloc[0], _moffat_params['wavelength'].iloc[-1] ],
                      Y.T, 'k--', lw=1)

            ax4.set(xlabel='Lambda [Ang]', ylabel='DeltaY (Model - Measured)')
            ax4.tick_params(axis='both', labelsize=6)
            ax4.xaxis.get_label().set_fontsize(6); ax4.yaxis.get_label().set_fontsize(6)
            ax4.plot([_moffat_params['wavelength'].iloc[0], _moffat_params['wavelength'].iloc[-1]],
                     Y.T, 'k--', lw=1)

            fig2, (ax5) = py.subplots(1, 1)
            ax5.set(xlabel='Lambda [Ang]', ylabel='Offset [arcsec]', title=f"{os.path.basename(path)}, Hexabundle {_ifu.hexabundle_name[5]}")
            ax5.tick_params(axis='both', labelsize=6)
            ax5.xaxis.get_label().set_fontsize(6); ax5.yaxis.get_label().set_fontsize(6); ax5.title.set_size(8)


            icmap = np.linspace(0, 1, len(_moffat_params['wavelength']))
            diffX_pcent, diffY_pcent, diff_offset, bad_data_count = [], [], [], 0
            for i in range(len(_moffat_params['wavelength'])):
                moffat_wave = _moffat_params['wavelength'].iloc[i]

                # NOTE: x- and y- axes are swapped in the optical model (constructed in micron),
                #       i.e. the y-CvD correction needs to be applied to the xfibre positions,
                #       and x-CvD correction to the yfibre positions
                ax.plot(np.nanmean(_xfibre) + np.polyval(zy, moffat_wave) * 1000.0,
                        np.nanmean(_yfibre) + np.polyval(zx, moffat_wave) * 1000.0,
                        's', color=cmap(icmap[i]), markersize=8, alpha=.4, lw=1,
                        label="Directly applying CvD model corrections" if i == 5 else None)

                # Converts from arcseconds to microns (xref, yref) are switched since the moffat fitting is done without switching them
                # STEP1: For ( mean_x [arcsec] and mean_y [arcsec] ) positions, get the respective micron positions.
                #        The input x-, y- mean arcsec needs to be switched
                xCen_mic, yCen_mic = coord_convert(np.nanmean(xfibre),
                                           np.nanmean(yfibre),
                                           xfibre, yfibre,  # Already switched (xfibre holds original yfibre postions), see L@373
                                           ifu.x_rotated, ifu.y_rotated,
                                           ifu.hexabundle_name, path_to_save=dest_path)
                prCyan(f"xCen_mic, yCen_mic = {xCen_mic}, {yCen_mic}")
                if xCen_mic is None: bad_data_count += 1; continue

                # STEP2: Add the CvD model vector magnitudes to the position in microns
                xCen_mic = xCen_mic + np.polyval(zx, np.array(moffat_wave)) * ARCSEC_IN_MICRONS
                yCen_mic = yCen_mic + np.polyval(zy, np.array(moffat_wave)) * ARCSEC_IN_MICRONS
                prPurple(f"xCen_mic, yCen_mic [cvd added] =  {xCen_mic}, {yCen_mic}")

                ax1.plot(ifu.x_rotated, ifu.y_rotated, 'kx', ms=8)
                ax1.plot(xCen_mic, yCen_mic, 'o', ms=8, color=cmap(icmap[i]),
                         label='CvD modelCrr [mic] added \n to hexaCentre \n [converted to mic from arcsec]' if i == 5 else None)

                # STEP3: Make the new positions the reference positions, and get the respective positions in arcseconds
                tmpx, tmpy = coord_convert(xCen_mic, yCen_mic,
                                           ifu.x_rotated, ifu.y_rotated,
                                           xfibre, yfibre,  # Already switched, see L@373
                                           ifu.hexabundle_name, path_to_save=dest_path)
                # if tmpx is None: record_bad_data()

                ax2.plot(xfibre, yfibre, 'kx', ms=8)
                ax2.plot(tmpx, tmpy, 'o', ms=10, color=cmap(icmap[i], alpha=.1),
                         label='CvD modelCrr [mic] added \n to hexaCentre (above fig) \n converted back to arcsec' if i == 5 else None)

                prYellow(f"x_cvdCrr, y_cvdCrr [arcseconds] = {tmpx}, {tmpy}, \n "
                         f"x_pos, y_pos [arcseconds] (swapped) = {np.nanmean(xfibre)}, {np.nanmean(yfibre)} (i.e. x_pos = y_pos from fits header)")

                # In plotting to plateView in arcseconds, we use the x-, y- fibre positions from the header as it is.
                # Here yfibre holds xpositions from the header (see L@373) and vice versa for xfibre
                ax.plot((tmpy - np.nanmean(yfibre)) * 1000.0 + np.nanmean(yfibre), # y indicated here, since it contains x values, See L@373
                        (tmpx - np.nanmean(xfibre)) * 1000.0 + np.nanmean(xfibre),
                        'x', color=cmap(icmap[i]), markersize=10, alpha=.5, lw=1,
                        label="CvD corrected using coord conversions" if i == 5 else None)
                del tmpx, tmpy
                del xCen_mic, yCen_mic

                # Overplot the centroids from direct calculations
                xref, yref, lref = _moffat_params['x0'].iloc[48],  _moffat_params['y0'].iloc[48], \
                                   _moffat_params['wavelength'].iloc[48]
                ax.plot( _moffat_params['x0'].iloc[i] + (_moffat_params['x0'].iloc[i] - xref) * 1000.0,
                         _moffat_params['y0'].iloc[i] + (_moffat_params['y0'].iloc[i] - yref) * 1000.0,
                        'o', color=cmap(icmap[i]), markersize=8, lw=1, alpha=.5,
                        label="Direct calculation of centroid" if i == 5 else None)
                ax.plot(xref, yref, 's', color='k', markersize=8, lw=1, alpha=.5)

                # below is plotted on the hexabundle that matches with the robot orientation, so x-, and y- swapped
                ax2.plot(_moffat_params['y0'].iloc[i], _moffat_params['x0'].iloc[i],
                          's', ms=8, alpha=.5, color=cmap(icmap[i]), label='Directly calculated centroid' if i == 5 else None)

                ax2.plot(np.nanmean(_yfibre) + np.polyval(zx, moffat_wave),
                         np.nanmean(_xfibre) + np.polyval(zy, moffat_wave),
                         '^', color='k', markerfacecolor=cmap(icmap[i]), markersize=4, alpha=.5, lw=2,
                         label='Directly applied CvD model' if i == 5 else None)

                # Find the magnitude difference between the model versus measured CvD corrections
                diffY_ref = ( np.nanmean(_yfibre) + np.polyval(zx, moffat_wave) ) - ( np.nanmean(_yfibre) + np.polyval(zx, lref) )
                diffX_ref = ( np.nanmean(_xfibre) + np.polyval(zy, moffat_wave) ) - ( np.nanmean(_xfibre) + np.polyval(zy, lref) )
                diffY_pcent.append( (diffY_ref - (_moffat_params['y0'].iloc[i] - yref)) / diffY_ref )
                diffX_pcent.append( (diffX_ref - (_moffat_params['x0'].iloc[i] - xref)) / diffX_ref )

                diff_offset = ( np.sqrt( (diffY_ref - (_moffat_params['y0'].iloc[i] - yref))**2.0 +
                                       (diffX_ref - (_moffat_params['x0'].iloc[i] - xref))**2.0 ) )

                ax3.plot( moffat_wave, ( diffX_ref - (_moffat_params['x0'].iloc[i] - xref) ) / diffX_ref,
                         '^', color='k', markerfacecolor=cmap(icmap[i]), markersize=8, alpha=.5, lw=2,
                          label=f"x-diff" if i==5 else None)
                ax4.plot( moffat_wave, ( diffY_ref - (_moffat_params['y0'].iloc[i] - yref) ) / diffY_ref,
                         's', color='k', markerfacecolor=cmap(icmap[i]), markersize=8, alpha=.5, lw=2,
                          label=f"y-diff" if i == 5 else None)
                ax5.plot(moffat_wave, diff_offset,
                         's', color='k', markerfacecolor=cmap(icmap[i]), markersize=8, alpha=.5, lw=2)

            # If the fitting has failed for more than half of the wavelength slices considered in the moffat_params, then record the bad frame
            if bad_data_count >= len(_moffat_params['wavelength']): record_bad_data()

            ax1.legend(loc='best', prop={'size': 6}); ax2.legend(loc='best', prop={'size': 6})
            fig.tight_layout()
            figfile = f"{dest_path}/DEBUG_hexabundle_{_ifu.hexabundle_name[5]}_inarcsec_{_ifu.primary_header['PLATEID']}_{os.path.basename(path)[:10]}.png"
            fig.savefig(figfile, bbox_inches='tight', pad_inches=0.3, dpi=150)
            py.close()

            diff_pcent = np.array(np.hstack([diffX_pcent, diffY_pcent])).squeeze()
            diff_len =len(diff_pcent)
            mask_50, mask_75, mask_100 = ((diff_pcent < 0.5) & (diff_pcent > -0.5)), \
                                         ((diff_pcent < 0.75) & (diff_pcent > -0.75)), \
                                         ((diff_pcent < 1.0) & (diff_pcent > -1.0))

            diffX_pcent, diffX_len = np.array(diffX_pcent), len(diffX_pcent)
            maskx_50, maskx_75, maskx_100 = ((diffX_pcent < 0.5) & (diffX_pcent > -0.5)), \
                                         ((diffX_pcent < 0.75) & (diffX_pcent > -0.75)), \
                                         ((diffX_pcent < 1.0) & (diffX_pcent > -1.0))

            diffY_pcent, diffY_len = np.array(diffY_pcent), len(diffY_pcent)
            masky_50, masky_75, masky_100 = ((diffY_pcent < 0.5) & (diffY_pcent > -0.5)), \
                                         ((diffY_pcent < 0.75) & (diffY_pcent > -0.75)), \
                                         ((diffY_pcent < 1.0) & (diffY_pcent > -1.0))

            ax3.set(title=f"{os.path.basename(path)}, Hexabundle {_ifu.hexabundle_name[5]} "
                          f"(<50%={np.round(len(diff_pcent[mask_50]) / diff_len, 2)}, "
                          f"<75%={np.round(len(diff_pcent[mask_75]) / diff_len, 2)}, "
                          f"<100%={np.round(len(diff_pcent[mask_100]) / diff_len, 2)}) \n"
                          f"--> x%: (<50%={np.round(len(diffX_pcent[maskx_50]) / diffX_len, 2)}, "
                          f"<75%={np.round(len(diffX_pcent[maskx_75]) / diffX_len, 2)}, "
                          f"<100%={np.round(len(diffX_pcent[maskx_100]) / diffX_len, 2)}) \n"
                          f"--> y%: (<50%={np.round(len(diffY_pcent[masky_50]) / diffY_len, 2)}, "
                          f"<75%={np.round(len(diffY_pcent[masky_75]) / diffY_len, 2)}, "
                          f"<100%={np.round(len(diffY_pcent[masky_100]) / diffY_len, 2)})")

            ax3.title.set_size(8)
            ax3.legend(loc='best', prop={'size': 6}); ax4.legend(loc='best', prop={'size': 6})
            fig1.tight_layout()
            figfile1 = f"{dest_path}/DEBUG_hexabundle_{_ifu.hexabundle_name[5]}_inarcsec_{_ifu.primary_header['PLATEID']}_centroid_variation_{os.path.basename(path)[:10]}.png"
            fig1.savefig(figfile1, bbox_inches='tight', pad_inches=0.3, dpi=150)
            py.close()

            ax5.legend(loc='best', prop={'size': 6});
            fig2.tight_layout()
            figfile2 = f"{dest_path}/DEBUG_hexabundle_{_ifu.hexabundle_name[5]}_inarcsec_{_ifu.primary_header['PLATEID']}_OFFSET_variation_{os.path.basename(path)[:10]}.png"
            fig2.savefig(figfile2, bbox_inches='tight', pad_inches=0.3, dpi=150)
            py.close()

            ax.plot(np.nanmean(_xfibre), np.nanmean(_yfibre), 'o', color='g', markersize=12, label='mean x/y [arcsec]')

            # Add annuli information to the plate-view plot
            magenta_radius = 226. * 1000. * MICRONS_TO_ARCSEC
            yellow_radius = 196.05124 * 1000. * MICRONS_TO_ARCSEC
            green_radius = 147.91658 * 1000. * MICRONS_TO_ARCSEC
            blue_radius = 92.71721 * 1000. * MICRONS_TO_ARCSEC

            ax.add_patch(Circle((0, 0), blue_radius, color='lightblue', alpha=0.3))
            ax.add_patch(
                Wedge((0, 0), magenta_radius, 0, 360., width=30, color='indianred',
                      alpha=0.7))
            ax.add_patch(
                Wedge((0, 0), yellow_radius, 0, 360., width=30, color='gold',
                      alpha=0.7))
            ax.add_patch(
                Wedge((0, 0), green_radius, 0, 360., width=30, color='lightgreen',
                      alpha=0.7))

            fig.suptitle(f"Hector Optical Model: {supltitle}", fontsize=15)
            py.legend(loc='best')

            ax.invert_yaxis()
            ax.invert_xaxis()
            ax.set_ylabel("yPos [arcsec]")
            ax.set_xlabel("xPos [arcsec]")

            py.tight_layout()
            figfile = f"{dest_path}/plateview_hexa{_ifu.hexabundle_name[5]}_inarcsec_{_ifu.primary_header['PLATEID']}_compareWth_CvDmodel.png"  # save_files / f"plateViewAll_{config['file_prefix']}_Run{obs_number:04}"
            py.savefig(figfile, bbox_inches='tight', pad_inches=0.3, dpi=150)
            py.close()
            # py.show()

            # If <60% of directly calculated x/y CvD correction don't fall within 75% direct-to-model based / model-based, then raise exception
            if np.round(len(diff_pcent[mask_75]) / diff_len, 2) < 0.6:
                raise ValueError(prRed(f"Large mismatch with the CvD model"))
            # sys.exit('---> Exit in the debugging routine @L530')


        # DEBUG plot - 1D illustration of "wave" versus X/Y CvD corrections
        """
        zx = np.polyfit(np.array(wave), np.array(deltaX), 2)
        zy = np.polyfit(np.array(wave), np.array(deltaY), 2)
        fig = py.figure()
        ax = fig.add_subplot(111)
        ax.plot(wave, deltaX, 'xr', alpha=0.5, label="X-corr, applied to y-coor")
        ax.plot(wave, deltaY, 'xb', alpha=0.5, label="Y-corr, applied to x-coor")
        ax.plot(wave, np.polyval(zx, np.array(wave)), 'r', alpha=0.5, label=" ")
        ax.plot(wave, np.polyval(zy, np.array(wave)), 'b', alpha=0.5, label=" ")
        ax.set(xlabel='Wavelength (Ang.)', ylabel='Relative flux', title=f"{os.path.basename(path)} ppxf fits")
        ax.legend(loc='best')
        py.show()
        fig.savefig(f"CvD_model_hexaC.png", bbox_inches='tight')
        py.close(fig)
        sys.exit()
        del zx, zy
        """

        return A, fractional_r, wave, deltaX, deltaY


    _xfibre = ifu.xpos_rel * np.cos(np.deg2rad(np.mean(ifu.ypos)))  # In arcsec
    _yfibre = ifu.ypos_rel

    prGreen(f"probename={ifu.hexabundle_name[5]}, (x,y) = ({np.nanmean(_xfibre)}, {np.nanmean(_yfibre)}) in arcsec from centre")
    prYellow(f"probename={ifu.hexabundle_name[5]}, (x,y) = ({-1.0 * (ifu.mean_x-robot_centre_in_mic[1]) * MICRONS_TO_ARCSEC}, "
             f"{(robot_centre_in_mic[0] - ifu.mean_y) * MICRONS_TO_ARCSEC}) in microns from centre")


    offset, frac_offset, lam, delta_X, delta_Y = optical_model_function(ifu.mean_x, ifu.mean_y,
                                                                        _ifu=ifu,
                                                                        _check_against_cvd_model=check_against_cvd_model,
                                                                        _moffat_params=moffat_params,
                                                                        _psf_parameters_array=psf_parameters_array,
                                                                        _wavelength=wavelength)
    cvdFrac = np.polyfit(np.array(lam), np.array(frac_offset), 2)
    cvdX    = np.polyfit(np.array(lam), np.array(delta_X) * MICRONS_TO_ARCSEC, 2)
    cvdY    = np.polyfit(np.array(lam), np.array(delta_Y) * MICRONS_TO_ARCSEC, 2)

    # DEBUG plot -
    # fig = py.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(lam, frac_offset, 'xk', alpha=0.5, label="best-fit")
    # ax.plot(lam, np.polyval(zf, np.array(lam)), 'b', alpha=0.5, label="best-fit")
    # ax.set(xlabel='Wavelength (Ang.)', ylabel='Relative flux', title=f"{os.path.basename(path)} ppxf fits")
    # ax.legend(loc='best')
    # py.show()

    return cvdFrac, cvdX, cvdY



def get_cvd_parameters_old(path_list, probenum, max_sep_arcsec=60.0,
                             catalogues=STANDARD_CATALOGUES,
                             model_name='ref_centre_alpha_dist_circ_hdratm',   # NEED to change the name here
                             n_trim=0, smooth='spline', molecfit_available=False,
                             molecfit_dir='', speed='', tell_corr_primary=False):


    data_chunked = read_chunked_data(path_list, probenum)
    trim_chunked_data(data_chunked, n_trim)

    psf_params_xcen, psf_params_ycen, psf_params_lambda = [], [], []
    for islice in range(np.shape(data_chunked['data'])[1]):
        chunked_vals, chunked_var, chunked_lambda, chunked_data = chunk_again(data_chunked, islice)

        # Fit the PSF
        fixed_parameters = set_fixed_parameters(
            path_list, model_name, probenum=probenum)
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


    av_xref_cen = np.nanmean(psf_params_xcen)
    av_yref_cen = np.nanmean(psf_params_ycen)
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
    prGreen(f"xcen_av={av_xref_cen}, ycen_av={av_yref_cen}")

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


def coord_convert (xref, yref, xfib_from, yfib_from, xfib_to, yfib_to, probename, path_to_save=None):
        dist = np.sqrt((xref - xfib_from) ** 2.0 + (yref - yfib_from) ** 2.0)
        idx_sorted_dist = np.argsort(dist)
        idx = np.argmin(dist)

        # Now we need to find the indcies of points that enclose (xref_cen_av, yref_cen_av)
        for j in range(1, 25):  # Should not need to go beyond 4 in range
            if (((xfib_from[idx_sorted_dist[0]] <= xref <= xfib_from[idx_sorted_dist[j]]) |
                 (xfib_from[idx_sorted_dist[0]] >= xref >= xfib_from[idx_sorted_dist[j]])) &
                    ((yfib_from[idx_sorted_dist[0]] <= yref <= yfib_from[idx_sorted_dist[j]]) |
                     (yfib_from[idx_sorted_dist[0]] >= yref  >= yfib_from[idx_sorted_dist[j]]))):

                encl_idx1, encl_idx2, encl_idx3 = idx_sorted_dist[0], idx_sorted_dist[j], idx_sorted_dist[j+1]
                break


        if 'encl_idx1' not in locals():
            fig, _ = py.subplots()  # In RA/DEC
            py.plot(xfib_from, yfib_from, 'kx', ms=8)
            py.plot(xfib_from[idx], yfib_from[idx], 'rx', ms=8)
            py.plot(xfib_from[idx_sorted_dist[0:3]], yfib_from[idx_sorted_dist[0:3]], 'g.', ms=10)
            py.plot(xref, yref, 'bo', ms=8)
            # py.show()
            py.tight_layout()
            if path_to_save is not None:
                figfile = f"{path_to_save}/DEBUG_arcsec_conversion.png"  # save_files / f"plateViewAll_{config['file_prefix']}_Run{obs_number:04}"
                py.savefig(figfile, bbox_inches='tight', pad_inches=0.3, dpi=150)
            py.close()

            fig, _ = py.subplots()  # In microns
            py.plot(xfib_to, yfib_to, 'kx', ms=8)
            py.plot(xfib_to[idx], yfib_to[idx], 'rx', ms=8)
            py.plot(xfib_to[idx_sorted_dist[0:3]], yfib_to[idx_sorted_dist[0:3]], 'g.', ms=10)
            py.show()

            py.tight_layout()
            if path_to_save is not None:
                figfile = f"{path_to_save}/DEBUG_micron_conversion.png"  # save_files / f"plateViewAll_{config['file_prefix']}_Run{obs_number:04}"
                py.savefig(figfile, bbox_inches='tight', pad_inches=0.3, dpi=150)
            py.close()

            return None, None

            # sys.exit(f"The indcies of the enclosed points of probe {probename} (i.e. points on either side of the centroid position) not found!")
        # assert 'encl_idx1' in locals(), \
        #     f"The indcies of the enclosed points of probe {probename} (i.e. points on either side of the centroid position) not found!"

        # Now we need to find the ratio: the distance from centroid position to the closest point / distance between the
        # two enclosing the closest points
        x_ratio = np.diff([xfib_from[encl_idx1], xref]) / np.diff([xfib_from[encl_idx1], xfib_from[encl_idx2]])
        y_ratio = np.diff([yfib_from[encl_idx1], yref]) / np.diff([yfib_from[encl_idx1], yfib_from[encl_idx2]])

        # Now take the micron grid,
        xref_to = xfib_to[encl_idx1] + np.diff([xfib_to[encl_idx1], xfib_to[encl_idx2]]) * x_ratio
        yref_to = yfib_to[encl_idx1] + np.diff([yfib_to[encl_idx1], yfib_to[encl_idx2]]) * y_ratio

        # Debug-figures
        # fig, ax = py.subplots()  # In RA/DEC
        # # ax.set(xlabel='X arcsec', ylabel='Y arcsec', title=probename)
        # py.plot(xfib_from, yfib_from, 'kx', ms=8)
        # py.plot(xfib_from[idx], yfib_from[idx], 'rx', ms=10)
        # py.plot(xref, yref, 'bo', ms=8)
        # py.plot(xfib_from[encl_idx1], yfib_from[encl_idx1], 'go', xfib_from[encl_idx2], yfib_from[encl_idx2], 'go', ms=6)
        # py.plot(xfib_from[idx_sorted_dist[0:3]], yfib_from[idx_sorted_dist[0:3]], 'r.', ms=8)
        # py.tight_layout()
        # figfile = 'DEBUG_hexaU_arcsec_conversion4.png'  # save_files / f"plateViewAll_{config['file_prefix']}_Run{obs_number:04}"
        # py.savefig(figfile, bbox_inches='tight', pad_inches=0.3, dpi=150)
        # py.close()
        # # #
        # fig, ax = py.subplots()  # In microns
        # # ax.set(xlabel='X microns', ylabel='Y microns', title=probename)
        # py.plot(xfib_to, yfib_to, 'kx', ms=8)
        # py.plot(xfib_to[idx], yfib_to[idx], 'rx', ms=10)
        # py.plot(xref_to, yref_to, 'bo', ms=8)
        # py.plot(xfib_to[encl_idx1], yfib_to[encl_idx1], 'go', xfib_to[encl_idx2], yfib_to[encl_idx2], 'go', ms=8)
        # py.plot(xfib_to[idx_sorted_dist[0:3]], yfib_to[idx_sorted_dist[0:3]], 'r.', ms=8)
        # # ax.invert_yaxis()
        # py.tight_layout()
        # figfile = 'DEBUG_hexaU_micron_conversion4.png'  # save_files / f"plateViewAll_{config['file_prefix']}_Run{obs_number:04}"
        # # figfile = 'DEBUG_hexaU_micron_conversion4_inverted_yaxis.png'  # save_files / f"plateViewAll_{config['file_prefix']}_Run{obs_number:04}"
        # py.savefig(figfile, bbox_inches='tight', pad_inches=0.3, dpi=150)
        # py.close()
        # sys.exit()

        return xref_to, yref_to

