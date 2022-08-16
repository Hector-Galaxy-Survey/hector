import os
import sys
import shutil
import numpy as np
import scipy as sp

import pylab as py

import astropy.io.fits as pf
from astropy.io import fits

import pandas as pd
import string
import itertools
from collections import Counter

import math as Math

# Circular patch.
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Arrow, Wedge
from matplotlib.collections import PatchCollection
from matplotlib.patches import Arc
from matplotlib.transforms import IdentityTransform, TransformedBbox, Bbox

import hector_display_utils as utils
import hector_centroid_fitting_utils as fitting_tools
# import gcam_utils as utils_tf

from hop.hexabundle_allocation.hector import constants

from importlib import reload
utils = reload(utils)
fitting_tools = reload(fitting_tools)
# utils_tf = reload(utils_tf)


# Print colours in python terminal --> https://www.geeksforgeeks.org/print-colors-python-terminal/
def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))
def prYellow(skk): print("\033[93m {}\033[00m" .format(skk))
def prLightPurple(skk): print("\033[94m {}\033[00m" .format(skk))
def prPurple(skk): print("\033[95m {}\033[00m" .format(skk))
def prCyan(skk): print("\033[96m {}\033[00m" .format(skk))
def prLightGray(skk): print("\033[97m {}\033[00m" .format(skk))
def prBlack(skk): print("\033[98m {}\033[00m" .format(skk))


def prepare_files(save_files, object_file):
    # Depending on the centroiding mode selected, create a folder within save_files
    if not os.path.exists(save_files):
        os.makedirs(save_files)

    centroid_filename = str(os.path.basename(object_file)) + ".offsets"

    # print("---> Centroid will be written to:", centroid_filename)
    # centroid_file = open(centroid_filename, "w")

    # print("#display.py probe centroids from file:", str(os.path.basename(object_file)), file=centroid_file)
    # print("# Format ready for fpcal program.", file=centroid_file)
    # print("# probenum cent_x_microns cent_y_microns # Details...", file=centroid_file)

    ## Turn this on to write offsets for Tony Farrell
    # centroid_filename2 = str(os.path.basename(object_file))[:-5] + '' + ".offsets"
    # centroid_file2 = open(centroid_filename2, "w")
    # print("# Probe RotatedX_microns RotatedY_microns FWHM MagX MagY", file=centroid_file2)

    colnames = ['Probe', 'MeanX', 'MeanY', 'RotationAngle', 'CentroidX_rotated', 'CentroidY_rotated',
                'CentroidXErr_rotated', 'CentroidYErr_rotated', 'CentroidX_COM_rotated', 'CentroidY_COM_rotated',
                'CentroidRMS_Err', 'RotationAngle_Centroid', 'RadialDist', 'RadialDistErr',
                'PDist', 'QDist', 'PDistErr', 'QDistErr', 'NDist', 'EDist', 'NDistErr',
                'EDistErr', 'RPerpenDist', 'RParallelDist', 'RPerpenDistErr',
                'RParallelDistErr', 'TelecentricAng', 'RadialDist_Plate', 'RadialDist_PlateErr']

    # Create a dataframe to hold centroiding statistics
    centroid_stat = pd.DataFrame(columns=colnames)


    return centroid_stat, colnames


def call_centroider(Probe, Probe_data, Probe_annulus, x, y, mean_x, mean_y, rotation_angle, robot_centre_in_mm, ax, centroid_stat, scat_plt, centroider=None, make_plots=None):
    scale_factor = 18
    hexabundle_tail_length = scale_factor * 1000

    quiet = False
    savecentdata = False

    # Uses GCAM centroider routine from Tony Farrall.
    if centroider:
        centroidX, centroidY, FHWM, imageX, imageY = \
            utils_tf.CentroidProbe(str(Probe).rstrip(), Probe_data, x, y, mean_x, mean_y, quiet, savecentdata)

    # Uses the SAMI/Hector module
    else:
        # gf_gauss, flux_gauss, xlin_gauss, ylin_gauss, model_gauss = \
        #     fitting_tools.centroid_gauss_fit(x, y, Probe_data, Probe, microns=True)
        # gf_gauss, gf_polyfit, rms_err, flux_gauss, xlin_gauss, ylin_gauss, model_gauss = \
        #     fitting_tools.centroid_gauss_fit_flux_weighted(x, y, Probe_data, Probe, microns=True, make_plots=make_plots)
        gf_gauss, gf_polyfit, fcom, rms_err, flux_gauss, xlin_gauss, ylin_gauss, model_gauss = \
            fitting_tools.centroid_gauss_fit_flux_weighted_main(x, y, Probe_data, Probe, microns=True,
                                                                make_plots=make_plots)

        centroidX, centroidY, FHWM, rmsErr = gf_gauss.p[1], gf_gauss.p[2], gf_gauss.p[3], rms_err
        centroidXErr0, centroidYErr0 = gf_polyfit[0], gf_polyfit[1]
        centroidXErr1, centroidYErr1 = fcom[0], fcom[1]
        imageX, imageY = np.NaN, np.NaN

        del gf_gauss, gf_polyfit, rms_err, flux_gauss, xlin_gauss, ylin_gauss, model_gauss


    if centroidX == 0 and centroidY == 0:
        prRed("Probe: " + str(Probe).rstrip() + " Centroid Failed!")
    else:
        if not quiet:
            prCyan("---> Probe " + str(Probe).rstrip() +
                  ", centroid:{0:.1f} {1:.1f}, FHWM={2:.1f} (Image X={3:.1f}, ImageY={4:.1f})".format(
                      float(centroidX), float(centroidY), float(FHWM), float(imageX), float(imageY)))
            print('\n')

        # centroidX_rotated = -1 * (np.cos(rotation_angle) * float(centroidX) - np.sin(rotation_angle) * float(centroidY))
        # centroidY_rotated = -1 * (np.sin(rotation_angle) * float(centroidX) + np.cos(rotation_angle) * float(centroidY))
        def rotate_centroids(angle_rotation, Xcentroid, Ycentroid):
            Xcentroid_rotated = 1. * (+np.cos(angle_rotation) * float(Xcentroid) + np.sin(angle_rotation) * float(Ycentroid))
            Ycentroid_rotated = 1. * (-np.sin(angle_rotation) * float(Xcentroid) + np.cos(angle_rotation) * float(Ycentroid))
            return Xcentroid_rotated, Ycentroid_rotated

        centroidX_rotated, centroidY_rotated = rotate_centroids(rotation_angle, centroidX, centroidY)

        # Calculates the errors - the error is taken as the difference between the centroid estimated from the Gaussian fits and the COM centroids
        centroidXErr0_rotated, centroidYErr0_rotated = rotate_centroids(rotation_angle, centroidXErr0, centroidYErr0)
        centroidXErr1_rotated, centroidYErr1_rotated = rotate_centroids(rotation_angle, centroidXErr1, centroidYErr1)

        # File written for Tony Farrell
        # print(
        #     "{0:d} {1:.1f} {2:.1f}\t# Probe {3:s},  FHWM={4:.1f} (Mag X={5:.1f}, Mag Y={6:.1f})".format(
        #         probenum, float(centroidX_rotated), float(centroidY_rotated),
        #         str(Probe).rstrip(), float(FHWM), float(mean_x), float(mean_y)), file=centroid_file2)

        centroid_marker = py.scatter(centroidX_rotated * scale_factor + mean_x,
                                     centroidY_rotated * scale_factor + mean_y,
                                     s=15, c='c', marker='x', zorder=300, linewidth=0.5)
        scat_plt.append(ax.add_artist(centroid_marker))

        centroid_marker = py.scatter(centroidXErr1_rotated * scale_factor + mean_x,
                                     centroidYErr1_rotated * scale_factor + mean_y,
                                     s=15, c='r', marker='x', zorder=300, linewidth=0.5)


        # -------------- DETERMINE VARIOUS OFFSETS
        # (1) RADIAL DISTANCE within the hexabundle
        def radial_distance_from_centre(rotated_Xcentroid, rotated_Ycentroid, meanX, meanY):
            """
            :param rotated_Xcentroid:
            :param rotated_Ycentroid:
            :param meanX: The centre X-coordinate
            :param meanY: The centre Y-coordinate
            :return: Radial distance
            """
            distance_radial = np.sqrt((rotated_Xcentroid - meanX) ** 2.0 + (rotated_Ycentroid - meanY) ** 2.0)
            return distance_radial

        radial_dist = radial_distance_from_centre(centroidX_rotated + mean_x, centroidY_rotated + mean_y, mean_x, mean_y)
        radial_distErr0 = radial_distance_from_centre(centroidXErr0_rotated + mean_x, centroidYErr0_rotated + mean_y, mean_x, mean_y)
        radial_distErr1 = radial_distance_from_centre(centroidXErr1_rotated + mean_x, centroidYErr1_rotated + mean_y, mean_x, mean_y)

        # Find the maximum of the radial distances calculated using the two Err centroids estimates.
        # Then To calculate the error, we take the error centroid measurement that give the maximum radial separation
        radial_argmax = np.nanargmax([radial_distErr0, radial_distErr1])
        centroidXErr_rotated = locals()['centroidXErr' + str(radial_argmax) + '_rotated']
        centroidYErr_rotated = locals()['centroidYErr' + str(radial_argmax) + '_rotated']
        radial_distErr = locals()['radial_distErr' + str(radial_argmax)]

        withErr = np.abs(radial_dist - radial_distErr)/2.0
        if radial_distErr >= radial_dist: radial_dist_withErr = radial_dist + withErr
        else: radial_dist_withErr = radial_dist - withErr
        ax.text(mean_x, mean_y + scale_factor * 970 * 2,
                r'$\Delta$ ' + "{:.1f}".format(radial_dist_withErr) + r" ($\pm${:.2f}".format(np.abs(radial_dist - radial_distErr)/2.) + ')',
                verticalalignment="bottom", horizontalalignment='center')

        # Centroid angle = the angle from East (to the right) to the centroid location within the probe -- +ve anti-clockwise
        centroid_angle = Math.atan2((centroidY_rotated + mean_y) - mean_y,
                                    (centroidX_rotated + mean_x) - mean_x)  # np.rad2deg(Math.atan2(y - cy, x - cx))

        """  (2) RADIAL DISTANCE from the plate centre (assuming the plate centre is at the origin) to the centroid """
        plate_x, plate_y = robot_centre_in_mm[1], robot_centre_in_mm[0]
        radial_dist_plate = radial_distance_from_centre(centroidX_rotated + mean_x, centroidY_rotated + mean_y, plate_x, plate_y)
        radial_dist_plateErr = radial_distance_from_centre(centroidXErr_rotated + mean_x, centroidYErr_rotated + mean_y, plate_x, plate_y)

        del plate_x, plate_y

        """  (3) P-DIRECTION (parallel to ferral axis) and Q-DIRECTION (perpendicular to ferral axis) VECTORS
        # In Robot-coordinate system, x-axis increasing downwards, and y-axis is increasing to the right.
        # So in plotting in python, we need to reverse the y-axis (in python plot window, which is the x-axis in
        # robot's frame of reference). In reversing the axis leads to the reversal of the sign of Qdist. Doesn't
        # affect Pdist sign (see evernote) but Qdist sign is defined to be 90deg, anti-clockwise from ferral. So
        # only that sign is affected.
        """
        ferral_axis, Q_dist, Q_sign, P_dist, P_sign = fitting_tools.Ps_and_Qs \
            (mean_x, mean_y, rotation_angle, hexabundle_tail_length, centroidX_rotated, centroidY_rotated,
             robot_coor=True)
        _, Q_distErr, Q_signErr, P_distErr, P_signErr = fitting_tools.Ps_and_Qs \
            (mean_x, mean_y, rotation_angle, hexabundle_tail_length, centroidXErr_rotated, centroidYErr_rotated,
             robot_coor=True)

        """  (4) CALCULATE N- AND E- VECTORS -- North is down, east to the left
        # The sign of E-, which is defined as 90deg. anti-clockwise from north (pointed down), will have to
        # change in robot coordinate system
        """
        N_dist, N_sign, E_dist, E_sign = fitting_tools.Ns_and_Es \
            (mean_x, mean_y, rotation_angle, hexabundle_tail_length, centroidX_rotated, centroidY_rotated,
             robot_coor=True)
        N_distErr, N_signErr, E_distErr, E_signErr = fitting_tools.Ns_and_Es \
            (mean_x, mean_y, rotation_angle, hexabundle_tail_length, centroidXErr_rotated, centroidYErr_rotated,
             robot_coor=True)

        """  (5) CALCULATE THE PARALLEL AND PERPENDICULAR DISTANCES from the centroid position to the Radial axis -
        # the radial axis is defined as the axis from the hexabundle centre to the plate centre. The hexabundle
        # centre to plate centre direction is negative. For the axis orthogonal to the radial axis, 90deg
        # anti-clockwise from hexa-centre to plate-centre vector is negative.
        """
        plate_x, plate_y = robot_centre_in_mm[1], robot_centre_in_mm[0]
        hexaCen_x, hexaCen_y = mean_x, mean_y
        angle_hexaCen_plateCen = Math.atan2(plate_x - hexaCen_x, plate_y - hexaCen_y)  # np.rad2deg(Math.atan2(y - cy, x - cx))
        angle_hexaCen_plateCen = angle_hexaCen_plateCen

        total_len = np.sqrt((plate_x - hexaCen_x) ** 2 + (plate_y - hexaCen_y) ** 2.0)
        line_hexabundle_tail4 = [(mean_x, mean_y), (mean_x + total_len * np.sin(angle_hexaCen_plateCen),
                                                    mean_y + total_len * np.cos(angle_hexaCen_plateCen))]
        ax.plot(*zip(*line_hexabundle_tail4), c='c', linewidth=1, zorder=1, alpha=0.1)

        radial_axis, Rperpendi_dist, Rperpendi_sign, Rparallel_dist, Rparallel_sign = \
            fitting_tools.perpendicular_and_parallel_to_RadialAxis \
                (mean_x, mean_y, angle_hexaCen_plateCen, hexabundle_tail_length, centroidX_rotated, centroidY_rotated,
                 robot_coor=True)
        _, Rperpendi_distErr, Rperpendi_signErr, Rparallel_distErr, Rparallel_signErr = \
            fitting_tools.perpendicular_and_parallel_to_RadialAxis \
                (mean_x, mean_y, angle_hexaCen_plateCen, hexabundle_tail_length, centroidXErr_rotated,
                 centroidYErr_rotated, robot_coor=True)


        # ----------- RECORD CENTROID STATS FOR PLOTS ON VARIOUS OFFSETS
        # In getting the colours of the probes, note that for some probes, (e.g. R) have a few fibres with
        # different colour. So the hexabundle colour is assigned to be the colour common to most fibres.
        occurrence = Counter(Probe_annulus)
        scale_radial_dist_plate = 1.E3  # arbitary scaling factor

        # for coln in centroid_stat_colnames:
        #     centroid_stat[coln] = locals()[coln]

        centroid_stat = centroid_stat.append(
            {'Probe': Probe, 'MeanX': mean_x, 'MeanY': mean_y, 'RotationAngle': rotation_angle,
             'CentroidX_rotated': centroidX_rotated, 'CentroidY_rotated': centroidY_rotated,
             'CentroidXErr_rotated': centroidXErr_rotated, 'CentroidYErr_rotated': centroidYErr_rotated,
             'CentroidX_COM_rotated': centroidXErr1_rotated, 'CentroidY_COM_rotated': centroidYErr1_rotated,
             'CentroidRMS_Err': rmsErr, 'RotationAngle_Centroid': centroid_angle,
             'RadialDist': radial_dist, 'RadialDistErr': radial_distErr,
             'PDist': P_dist, 'QDist': Q_dist, 'PDistErr': P_distErr, 'QDistErr': Q_distErr,
             'NDist': N_dist, 'EDist': E_dist, 'NDistErr': N_distErr, 'EDistErr': E_distErr,
             'RPerpenDist': Rperpendi_dist, 'RParallelDist': Rparallel_dist,
             'RPerpenDistErr': Rperpendi_distErr, 'RParallelDistErr': Rparallel_distErr,
             'TelecentricAng': [key for key, _ in occurrence.most_common()][0].lower()[0],
             'RadialDist_Plate': radial_dist_plate / scale_radial_dist_plate,
             'RadialDist_PlateErr': radial_dist_plateErr / scale_radial_dist_plate}, ignore_index=True)

        del centroidXErr0, centroidYErr0, centroidXErr1, centroidYErr1, centroidY, centroidX
        del centroidXErr_rotated, centroidYErr_rotated, centroidXErr0_rotated, centroidYErr0_rotated, centroidXErr1_rotated, centroidYErr1_rotated, centroidY_rotated, centroidX_rotated
        del ferral_axis, radial_dist, P_dist, Q_sign, Q_dist, N_sign, N_dist, E_sign, E_dist, radial_dist_plate
        del radial_distErr, P_distErr, Q_signErr, Q_distErr, N_signErr, N_distErr, E_signErr, E_distErr, radial_dist_plateErr
        del radial_axis, Rperpendi_dist, Rperpendi_sign, Rparallel_dist, Rparallel_sign
        del Rperpendi_distErr, Rperpendi_signErr, Rparallel_distErr, Rparallel_signErr
        del occurrence, rmsErr, withErr

    return centroid_stat, scat_plt


def make_figures(centroid_statFinal, save_files, robot_centre_in_mm, plate_radius, supltitle, obs_number, hexabundle_tail_length, scale_factor, config):

    fig_stat, (axes) = plt.subplots(3, 3, figsize=(14, 10))
    axes[0, 2].set_axis_off()
    fig_stat.suptitle(f"Centroiding Stats: {supltitle}", fontsize=15)
    fig_stat.subplots_adjust(left=0.05,
                             bottom=0.06,
                             right=0.99,
                             top=0.93,
                             wspace=0.2,
                             hspace=0.2)

    def plot_hist(datFrame, name, ax0, label, colr, with_error=None):
        """ plot a histogram for a give set of values.
            Calls 'autolabel' attach text labels
        """
        datFrame[name] = datFrame[name].astype(float)
        datFrame[name + 'Err'] = datFrame[name + 'Err'].astype(float)

        if with_error:
            data_from_frame = datFrame[name].to_numpy()
            dataerr_from_frame = datFrame[name + 'Err'].to_numpy()
            dataFrame = data_from_frame + (dataerr_from_frame - data_from_frame) / 2.0
        else:
            dataFrame = datFrame[name].to_numpy()

        width = 25.0
        # nbins = np.ceil((datFrame[name].values.max() - datFrame[name].values.min()) / width)
        # datFrame[name] = datFrame[name].astype(float)

        nbins = np.ceil((dataFrame.max() - dataFrame.min()) / width)

        n1, bins1, patchs = ax0.hist(dataFrame, bins=np.int(np.array(nbins).squeeze()), histtype='bar',
                                     color=colr, edgecolor='black', label=label, alpha=0.5)

        autolabel(n1, bins1, patchs, datFrame, name, ax0, with_error=with_error)
        ax0.set_xlabel(label)

        return

    def autolabel(n, bins, rects, datFrame, name, ax0, with_error=None):
        """ Attach a text label above each bar in *rects*, displaying its content.
        """
        if with_error:
            DistErr = datFrame[name + 'Err'].to_numpy()
            Dist = datFrame[name].to_numpy()
            withErr = (DistErr - Dist) / 2.0
            Dist = Dist + withErr
        else:
            Dist = datFrame[name].to_numpy()

        for irect in range(len(rects)):
            if irect == 0:
                indx = np.where((Dist >= np.floor(bins[irect])) & (Dist < bins[irect + 1]))
            elif irect == len(rects) - 1:
                indx = np.where((Dist >= bins[irect]) & (Dist <= np.ceil(bins[irect + 1])))
            else:
                indx = np.where((Dist >= bins[irect]) & (Dist < bins[irect + 1]))
            indx = np.array(indx).squeeze()

            rect = rects[irect]
            height = rect.get_height()
            if not indx.size == 0:
                assert indx.size == n[irect], 'index size must match n items in bins'

                delta = height / 10.
                for i in range(indx.size):
                    if indx.size == 1:
                        loc = indx
                    else:
                        loc = indx[i]
                    ax0.annotate('{}'.format(datFrame['Probe'].iloc[loc]),
                                 xy=(rect.get_x() + rect.get_width() / 2, height - height / 8 - delta),
                                 xytext=(0, 0),  # 3 points vertical offset
                                 textcoords="offset points",
                                 ha='center', va='bottom',
                                 color=datFrame['TelecentricAng'].iloc[loc])
                    if with_error:
                        ax0.errorbar(rect.get_x() + rect.get_width() / 2, height - height / 8 - delta, xerr = np.abs(withErr[loc]), lw = 1,
                                    capsize = 1, capthick = 1, color = "black")


                    delta += height / 10.
            del indx, height
        return

    # Call 'plot_hist' with various offsets
    plot_hist(centroid_statFinal, 'RadialDist', axes[0, 0], "Radial offset " r'[$\mu $m]', 'black', with_error=True)
    plot_hist(centroid_statFinal, 'RadialDist_Plate', axes[0, 1],
              'Radial distance from the plate centre' r'[$10^3 \mu $m]', 'Chartreuse', with_error=True)

    plot_hist(centroid_statFinal, 'PDist', axes[1, 0], "P-dir Offset " r'[$\mu $m]', 'black', with_error=True)
    plot_hist(centroid_statFinal, 'QDist', axes[2, 0], "Q-dir Offset " r'[$\mu $m]', 'black', with_error=True)

    plot_hist(centroid_statFinal, 'NDist', axes[1, 1], "Compass (N-S) Offset " r'[$\mu $m]', 'black', with_error=True)
    plot_hist(centroid_statFinal, 'EDist', axes[2, 1], "Compass (E-W) Offset " r'[$\mu $m]', 'black', with_error=True)

    plot_hist(centroid_statFinal, 'RParallelDist', axes[1, 2], r'$\parallel$ to radial axis [$\mu $m]', 'black', with_error=True)
    plot_hist(centroid_statFinal, 'RPerpenDist', axes[2, 2], r'$\bot$ to radial axis [$\mu $m]', 'black', with_error=True)

    # plt.tight_layout()
    figfile = save_files / f"CentroidingDist_{config['file_prefix']}_Run{obs_number:04}"
    plt.savefig(figfile, bbox_inches='tight', pad_inches=0.3)
    plt.show()

    #
    fig2 = plt.figure(figsize=(10, 9.8))

    fig2.suptitle(f"Hector raw data: {supltitle}", fontsize=15)

    ax_vec = fig2.add_subplot(1, 1, 1)
    ax_vec.set_aspect('equal')

    ax_vec.add_patch(
        Circle(xy=(robot_centre_in_mm[1], robot_centre_in_mm[0]), radius=plate_radius * 1.0E3, facecolor="#cccccc",
               edgecolor='#000000', zorder=-1))
    ax_vec.plot(robot_centre_in_mm[1], robot_centre_in_mm[0], 'rx', markersize=12)

    def add_vectors(datFrame, tail_length):
        """
            Show P- and Q- and radial offsets as vectors
        """
        datFrame['MeanX'], datFrame['MeanY'] = datFrame['MeanX'].astype(float), datFrame['MeanY'].astype(float)
        datFrame['RotationAngle'] = datFrame['RotationAngle'].astype(float)
        datFrame['PDist'] = datFrame['PDist'].astype(float)

        meanX, meanY = datFrame['MeanX'].to_numpy(), datFrame['MeanY'].to_numpy()
        centrX_rot, centrY_rot = datFrame['CentroidX_rotated'].to_numpy(), datFrame['CentroidY_rotated'].to_numpy()
        centroid_rotation = datFrame['RotationAngle_Centroid'].to_numpy()

        radial_distance = datFrame['RadialDist'].to_numpy()
        angle_rotation = datFrame['RotationAngle'].to_numpy()

        hexabundle_tail = [(meanX, meanY), (meanX + tail_length * np.sin(angle_rotation),
                                            meanY + tail_length * np.cos(angle_rotation))]

        ax_vec.plot(*zip(*hexabundle_tail), c='k', linewidth=10, zorder=1, alpha=0.3)

        ax_vec.plot(*zip(*hexabundle_tail), c='k', linewidth=10, zorder=1, alpha=0.3)

        for iprobe in range(np.array(meanX).size):
            ax_vec.text(meanX[iprobe], meanY[iprobe] + scale_factor * 750 * 2,
                        'Probe {}'.format(datFrame['Probe'].iloc[iprobe]), verticalalignment="bottom",
                        horizontalalignment='center')

        # Add radial vectors -- I am switching from [x + Rsin(theta)], [y + Rcos(theta)], which takes the angle
        # from north (pointing upwards), to [x + Rcos(theta)], [y + Rsin(theta)], which takes the angle from east -- +ve anti-clockwise
        hexabundle_RDir = [(meanX, meanY), (meanX + radial_distance * 100 * np.cos(centroid_rotation),
                                            meanY + radial_distance * 100 * np.sin(centroid_rotation))]
        plot_arrows(hexabundle_RDir, ax_vec, color='g', fill=True, width=500, head_width=5000)

        # Add P- and Q-vectors
        for type_dist in ['PDist', 'QDist']:
            distance = datFrame[type_dist].to_numpy()

            indx1 = np.array(np.where(distance >= 0.0)).squeeze()  # +ve ?-distance (blue arrow)
            indx2 = np.array(np.where(distance < 0.0)).squeeze()  # -ve ?-distance (red arrow)
            plot1, plot2 = False, False

            if type_dist is 'PDist':
                if indx1.size > 0:
                    plot1, colr1 = True, 'b'
                    dist1 = np.abs(distance[indx1]) * 100.  # 100 is a scaling factor
                    ang_adust1 = angle_rotation[indx1] + np.pi  # 180 deg. from the centre-to-ferral axis

                if indx2.size > 0:
                    plot2, colr2 = True, 'r'
                    dist2 = np.abs(distance[indx2]) * 100.
                    ang_adust2 = angle_rotation[indx2]  # direction of the centre-to-ferral axis

            elif type_dist is 'QDist':
                if indx1.size > 0:
                    plot1, colr1 = True, 'b'
                    dist1 = np.abs(distance[indx1]) * 100.
                    ang_adust1 = angle_rotation[indx1] - np.pi / 2  # 90 deg. clockwise from the centre-to-ferral axis

                if indx2.size > 0:
                    plot2, colr2 = True, 'r'
                    dist2 = np.abs(distance[indx2]) * 100.
                    ang_adust2 = angle_rotation[
                                     indx2] + np.pi / 2  # 90 deg. anti-clockwise from the centre-to-ferral axis

            if plot1:
                hexabundle_Dir = [(meanX[indx1], meanY[indx1]), (meanX[indx1] + dist1 * np.sin(ang_adust1),
                                                                 meanY[indx1] + dist1 * np.cos(ang_adust1))]
                plot_arrows(hexabundle_Dir, ax_vec, color=colr1, fill=True, width=500, head_width=5000)

                del hexabundle_Dir, dist1, ang_adust1

            if plot2:
                hexabundle_Dir = [(meanX[indx2], meanY[indx2]), (meanX[indx2] + dist2 * np.sin(ang_adust2),
                                                                 meanY[indx2] + dist2 * np.cos(ang_adust2))]
                plot_arrows(hexabundle_Dir, ax_vec, color=colr2, fill=True, width=500, head_width=5000)

                del hexabundle_Dir, dist2, ang_adust2

            del distance, indx1, indx2, plot1, plot2
        return

    def plot_arrows(verts, ax1, **kw_args):
        """
            Arrow plotting routine.
        """
        x_tmp, y_tmp = zip(*verts)
        x0, y0 = x_tmp[0], y_tmp[0]
        x1, y1 = x_tmp[1], y_tmp[1]

        for xy0, xy1 in zip(np.vstack([x0, y0]).T, np.vstack([x1, y1]).T):
            ax1.arrow(*xy0, *(xy1 - xy0), **kw_args)
        return

    add_vectors(centroid_statFinal, hexabundle_tail_length)

    # And add some N/E arrows
    # ax_vec = utils.add_NE_arrows(ax_vec)

    ax_vec.invert_yaxis()
    ax_vec.set_ylabel("Robot $x$ coordinate")
    ax_vec.set_xlabel("Robot $y$ coordinate")

    plt.setp(ax_vec.get_xticklabels(), visible=True)
    plt.setp(ax_vec.get_yticklabels(), visible=True)

    plt.tight_layout()
    figfile = save_files / f"CentroidingOffsets_{config['file_prefix']}_Run{obs_number:04}"
    plt.savefig(figfile, bbox_inches='tight', pad_inches=0.3)
    plt.show()


def individual_hexabundle_plots(x_rotated, y_rotated, mean_x, mean_y, scale_factor, Probe_data, Probe, save_files, obs_number):

    fig_single_hexas = plt.figure()
    fig_single_hexas.suptitle(f"Probe: {Probe}", fontsize=15)
    #
    ax_single_hexas = fig_single_hexas.add_subplot(1, 1, 1)
    ax_single_hexas.set_aspect('equal')

    ax_single_hexas.add_collection(utils.display_ifu(x_rotated, y_rotated, mean_x, mean_y, scale_factor, Probe_data), autolim=True)

    x_shift = np.max(np.abs(x_rotated)) * scale_factor * 1.5
    y_shift = np.max(np.abs(y_rotated)) * scale_factor * 1.5
    ax_single_hexas.set_xlim([mean_x-x_shift, mean_x+x_shift])
    ax_single_hexas.set_ylim([mean_y-y_shift, mean_y+y_shift])
    ax_single_hexas.set_facecolor('#cccccc')

    ax_single_hexas.invert_yaxis()
    ax_single_hexas.set_ylabel("Robot $x$ coordinate")
    ax_single_hexas.set_xlabel("Robot $y$ coordinate")

    plt.tight_layout()
    figfile = save_files / f"Run{obs_number:04}_Hexabundle_{Probe}.png"
    plt.savefig(figfile, bbox_inches='tight', pad_inches=0.3)
    plt.close()

    return
