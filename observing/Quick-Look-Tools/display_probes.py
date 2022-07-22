"""
This module is a modified version of sami_display intended for a quick view of Hector data.

Edited by Jesse van de Sande Dec-2021
Edited by Sam Vaughan Jan 2021

Edited by Madusha Gunawardhana Jan-2021, Feb-2021 and since Feb-2021. Edits include:
    - terminal colours
    - centroid-fitting routines for use with (raw data frame + raw flat), (reduced data frame) and (raw data frame + reduced flat tlm)
    - centroid offset plotting routines
    - included 'hector_centroid_fitting_utils'
    - included 'gcam_utils', a routine that calls on the GCAM centroider -cpp executable provided by Tony Farrell

"""
import os
import sys
import shutil
import datetime
import numpy as np
import scipy as sp

import pylab as py

import astropy.io.fits as pf
from astropy.io import fits

import pandas as pd
# import string
# import itertools
# from collections import Counter

import math as Math

# Circular patch.
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Arrow, Wedge
from matplotlib.collections import PatchCollection
# from matplotlib.patches import Arc
# from matplotlib.transforms import IdentityTransform, TransformedBbox, Bbox

import hector_display_utils as utils
import hector_centroid_fitting_utils as fitting_tools
import hector_centroider as hector_centroider
# import gcam_utils as utils_tf

from hop.hexabundle_allocation.hector import constants

from importlib import reload
utils = reload(utils)
fitting_tools = reload(fitting_tools)
# utils_tf = reload(utils_tf)


from termcolor import colored, cprint


# Print colours in python terminal --> https://www.geeksforgeeks.org/print-colors-python-terminal/
def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))
def prYellow(skk): print("\033[93m {}\033[00m" .format(skk))
def prLightPurple(skk): print("\033[94m {}\033[00m" .format(skk))
def prPurple(skk): print("\033[95m {}\033[00m" .format(skk))
def prCyan(skk): print("\033[96m {}\033[00m" .format(skk))
def prLightGray(skk): print("\033[97m {}\033[00m" .format(skk))
def prBlack(skk): print("\033[98m {}\033[00m" .format(skk))

def month_to_num(month):
    return {'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'may': '05', 'jun': '06', 'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'}[month]


if __name__ == "__main__":
    

    import yaml
    from pathlib import Path
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("CCD")
    parser.add_argument("file_number", type=int, help='The object frame number you want to display')
    parser.add_argument("flat_number", type=int, help='The flat frame number corresponding TLM file')
    parser.add_argument("--file_prefix", default=None, type=str, help='The date stamp on the file, e.g. 28jun')
    parser.add_argument("--robot_file_name", default=None, type=str, help='The robot file name')

    parser.add_argument("--config-file", default=None, help="A .yaml file which contains parameters and filenames for the code. See hector_display_config.yaml for an example")
    parser.add_argument("--outfile", help='Filename to save the plot to. If not given, display the plot instead')
    parser.add_argument("-sigma", "--sigma_clip", action='store_true', help='Turn on sigma clipping. Can also be set in the config file')
    parser.add_argument("-data", "--data_type", type=str, help='The type of the data frame to be processed (options "raw" or "reduced")')

    args = parser.parse_args()

    obs_number = args.file_number
    flat_obs_number = args.flat_number
    # flat_obs_number = config['flat_obs_number']


    config_filename = 'hector_display_config.yaml'
    if args.config_file is not None:
        config_filename = args.config_file

    try:
        with open(config_filename, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {config_filename} does not exist!")

    # Date the observations are taken
    # file_prefix = config['file_prefix']
    if args.file_prefix is not None:
        config['file_prefix'] = args.file_prefix

    # Use file_prefix to extract the year-month-day information to append to the data_dir
    datestamp = str(datetime.date.today().year)[-2::] + str(month_to_num(config['file_prefix'][-3::])) + config['file_prefix'][0:2]

    # Get the robot file name
    if args.robot_file_name is not None:
        config['robot_file_name'] = args.robot_file_name

    # Do we need to make plots looking at individual fits
    make_plots = config['make_plots']

    # -------------------------------- Load some options
    # - data_type (options "raw" or "tramline_map")
    data_type = config['data_type']
    if not data_type:
        data_type = args.data_type

    # - sigma_clipping
    sigma_clip = config['sigma_clip']
    # if not sigma_clip:
    #     sigma_clip = args.sigma_clip

    # - Centroid-fitting related keywords
    centroid = config['centroid']
    # if not centroid:
    #     centroid = args.centroid

    centroider = config['centroider']
    sami_centroider = config['sami_centroider']
    savestatdata = config['savestatdata']


    # Read-in the robot file
    robot_file = Path(config['data_dir']) / f"raw/{datestamp}" / Path(config['robot_file_name'])
    robot_fname = str(Path(config['robot_file_name']))

    # Read-in the tile file
    object_tile_file = Path(config['data_dir']) / f"raw/{datestamp}" / Path(config['robot_file_name'].replace("Robot", "Tile"))
    if not object_tile_file.exists():
        raise FileNotFoundError(f"The Hector Tile file seems to not exist: {object_tile_file} not found")

    tile_file = open(object_tile_file, "rt")
    uttime_config = tile_file.readlines()[7].strip('UTTIME, #Target observing time\n') # Extract UT TIME from the header of the tile file.
    tile_file.close()

    # Read-in the fibre position information file from Julia Bryant.
    fibre_pos = pd.read_csv('Fibre_slitInfo_final_updated03Mar2022for_swapped_fibres_BrokenHexaM.csv')

    if config['red_or_blue'] == 'blue':
        hector_ccd, aaomega_ccd = [3], [1]

    elif config['red_or_blue'] == 'red':
        hector_ccd, aaomega_ccd = [4], [2]

    elif config['red_or_blue'] == 'both':
        hector_ccd, aaomega_ccd = [3,4], [1,2]

    else:
        raise NameError(f"The red_or_blue value must be either 'red' or blue' or 'both': currently {config['red_or_blue']}")

    if data_type == "reduced": method_to_call = getattr(utils, 'get_alive_fibres_reduced_frames')
    elif data_type == "tramline_map": method_to_call = getattr(utils, 'get_alive_fibres_from_tlm')
    else: method_to_call = getattr(utils, 'get_alive_fibres')

    object_spec_Asum, object_spec_Hsum, flats = [], [], ""
    for i in range(np.size(hector_ccd)):
        flat_file_Hector = Path(config['data_dir']) / f"reduced/{datestamp}/1/1_F0/calibrators" / f"ccd_{hector_ccd[i]}" / f"{config['file_prefix']}{hector_ccd[i]}{flat_obs_number:04}.fits"
        flat_file_AAOmega = Path(config['data_dir']) / f"reduced/{datestamp}/1/1_F0/calibrators" / f"ccd_{aaomega_ccd[i]}" / f"{config['file_prefix']}{aaomega_ccd[i]}{flat_obs_number:04}.fits"

        flats = flats + f"{str(flat_file_AAOmega)[-15:-5]} {str(flat_file_Hector)[-15:-5]} "

        if data_type == "reduced":
            object_file_Hector = Path(config['data_dir']) / f"reduced/{datestamp}" / f"ccd_{hector_ccd[i]}" / f"{config['file_prefix']}{hector_ccd[i]}{obs_number:04}red.fits"
            object_file_AAOmega = Path(config['data_dir']) / f"reduced/{datestamp}" / f"ccd_{aaomega_ccd[i]}" / f"{config['file_prefix']}{aaomega_ccd[i]}{obs_number:04}red.fits"
        else:
            object_file_Hector = Path(config['data_dir']) / f"raw/{datestamp}" / f"ccd_{hector_ccd[i]}" / f"{config['file_prefix']}{hector_ccd[i]}{obs_number:04}.fits"
            object_file_AAOmega = Path(config['data_dir']) / f"raw/{datestamp}" / f"ccd_{aaomega_ccd[i]}" / f"{config['file_prefix']}{aaomega_ccd[i]}{obs_number:04}.fits"

        if not object_file_Hector.exists():
            raise FileNotFoundError(f"The Hector file seems to not exist: {object_file_Hector} not found")
        if not object_file_AAOmega.exists():
            raise FileNotFoundError(f"The AAomega file seems to not exist: {object_file_AAOmega} not found")

        # Get the fibre tables and find the tramlines:
        object_header_A, object_fibtab_A, object_guidetab_A, object_robottab_A, object_spec_A, spec_id_alive_A = method_to_call(flat_file_AAOmega, object_file_AAOmega, robot_file, sigma_clip=sigma_clip, IFU="unknown", log=True, pix_waveband=100, pix_start="unknown", plot_fibre_trace = False)
        object_header_H, object_fibtab_H, object_guidetab_H, object_robottab_H, object_spec_H, spec_id_alive_H = method_to_call(flat_file_Hector, object_file_Hector, robot_file, sigma_clip=sigma_clip, IFU="unknown", log=True, pix_waveband=100, pix_start="unknown", plot_fibre_trace = False)

        object_spec_Asum.append(object_spec_A)
        object_spec_Hsum.append(object_spec_H)

        del object_spec_A, object_spec_H


    # Create a directory to store images and other data files, but first check if it exists
    save_files = Path(config['data_dir']) / f"Outputs/{config['file_prefix']}Run{obs_number:04}/{data_type}/"
    if not os.path.exists(save_files):
        os.makedirs(save_files)

    # Prepare for centroid-fitting
    if centroid:
        if centroider: mode = 'GCAM'
        elif sami_centroider: mode = 'Hector'
        else: sys.exit(prRed('Centroid fitting is set to True - must provide a centroid-fitting method'))

        prLightPurple(str("---> Centroid will be fitted...using {} Centroider".format(mode)))

        save_files = save_files / f"{mode}_fit/"
        centroid_stat, colnames = hector_centroider.prepare_files(save_files, object_file_AAOmega)


    # Plot the data
    print("---> Plotting...")
    print("--->")

    scale_factor = 18
    hexabundle_tail_length = scale_factor * 1000

    fig = plt.figure(figsize=(10,9.8))
    supltitle = f"{config['file_prefix']} frame {obs_number} (flats: {flats}) \n" \
                f"Robot file: {str(robot_fname)} \n" \
                f"UTconfigured: {uttime_config}, UTobserved: {object_header_A['UTSTART']}, ZD: {object_header_A['ZDSTART']}" \

    if data_type == "tramline_map": fig.suptitle(f"Hector data (TLM): {supltitle}",fontsize=15)
    else: fig.suptitle(f"Hector data ({data_type}): {supltitle}",fontsize=15)

    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')

    robot_centre_in_mm = [constants.robot_center_x * 1.0E3, constants.robot_center_y * 1.0E3]
    plate_radius = constants.HECTOR_plate_radius

    ax.add_patch(Circle(xy=(robot_centre_in_mm[1], robot_centre_in_mm[0]), radius=plate_radius * 1.0E3, facecolor="#cccccc", edgecolor='#000000', zorder=-1))
    ax.plot(robot_centre_in_mm[1], robot_centre_in_mm[0], 'rx', markerSize=12)
    scat_plt = []


    plist = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U']
    for Probe in plist:  # list(string.ascii_uppercase[:21]):

        if Probe in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            object_header, object_fibtab, object_guidetab, object_robottab, object_spec, spec_id_alive = object_header_A, object_fibtab_A, object_guidetab_A, object_robottab_A, np.mean(object_spec_Asum, axis=0), spec_id_alive_A
        else:
            object_header, object_fibtab, object_guidetab, object_robottab, object_spec, spec_id_alive = object_header_H, object_fibtab_H, object_guidetab_H, object_robottab_H, np.mean(object_spec_Hsum, axis=0), spec_id_alive_H

        mask = (object_fibtab.field('TYPE')=="P") & (object_fibtab.field('SPAX_ID')==Probe)

        if (data_type == "reduced") or (data_type == "tramline_map"): Probe_data = object_spec[mask]
        else: Probe_data = object_spec[spec_id_alive[mask]]

        Probe_annulus = object_fibtab.field('CIRCMAG')[mask] # telecentricity of the probe, given by the circmag colour

        fibnum = object_fibtab.field('FIBNUM')[mask]


        fibnum_from_file_probe = fibre_pos.loc[(fibre_pos["Bundle/plate"] == Probe)]
        fibnum_from_file = fibnum_from_file_probe['Fibre_number'].to_numpy()
        Bundle_Xc_from_file = fibnum_from_file_probe['Bundle_Xc'].to_numpy()
        Bundle_Yc_from_file = fibnum_from_file_probe['Bundle_Yc'].to_numpy()
        x, y, fiber_number =[], [], []
        for num in fibnum:
            min_idx = np.array(np.where((fibnum_from_file - num) == 0.0)).squeeze()
            assert np.size(min_idx) == 1, 'Error'

            x.append(Bundle_Xc_from_file[min_idx])
            y.append(Bundle_Yc_from_file[min_idx])
            fiber_number.append(fibnum_from_file[min_idx])

        del fibnum_from_file_probe, fibnum_from_file, Bundle_Xc_from_file, Bundle_Yc_from_file

        x = np.array(x)
        y = np.array(y)

        # For each hexabundle, get its circular and rectangular magnet (comes directly from the robot file)
        cm = object_robottab.loc[(object_robottab['Hexabundle'] == Probe) & (object_robottab['#Magnet'] == 'circular_magnet')]
        rm = object_robottab.loc[(object_robottab['Hexabundle'] == Probe) & (object_robottab['#Magnet'] == 'rectangular_magnet')]

        mean_x = cm.Center_y
        mean_x = mean_x[mean_x.index[0]] * 1.0E3
        mean_y = cm.Center_x
        mean_y = mean_y[mean_y.index[0]] * 1.0E3


        # The angle of the rectangular magnet- 270 minus the robot holding angle minus the robot placing angle
        rotation_angle = np.radians(270.0 - rm.rot_holdingPosition - rm.rot_platePlacing)
        rotation_angle = rotation_angle[rotation_angle.index[0]]

        # Rotation of axes: see Evernote HECTOR/probe rotations for more information. The 'minus' sign, I think, accounts
        # for the fact that North is down (and East to the right) on the plate
        x_rotated = 1 * (+np.cos(rotation_angle) * x + np.sin(rotation_angle) * y)
        y_rotated = 1 * (-np.sin(rotation_angle) * x + np.cos(rotation_angle) * y)


        # Hexabundle tail or ferral direction
        line_hexabundle_tail = [(mean_x, mean_y), (mean_x + hexabundle_tail_length * np.sin(rotation_angle),
                                                   mean_y + hexabundle_tail_length * np.cos(rotation_angle))]
        ax.plot(*zip(*line_hexabundle_tail), c='k', linewidth=2, zorder=1, alpha=0.5)

        ax.add_collection(utils.display_ifu(x_rotated, y_rotated, mean_x, mean_y, scale_factor, Probe_data))
        # for ix in range(len(x_rotated)):
        #     ax.plot(x_rotated[ix]*scale_factor+mean_x, y_rotated[ix]*scale_factor+mean_y, '.', c='k')
        #     ax.text(x_rotated[ix]*scale_factor+mean_x, y_rotated[ix]*scale_factor+mean_y, str(fibnum[ix]), color='k')

        line_hexabundle_tail4 = [(mean_x, mean_y), (mean_x + hexabundle_tail_length * np.sin(0.0),
                                                    mean_y + hexabundle_tail_length * np.cos(0.0))]
        ax.plot(*zip(*line_hexabundle_tail4), c='g', linewidth=2, zorder=1, alpha=0.1)


        # ------------------------------------- CENTROID - FITTING -----------------------------------------------------
        # Determine the centroid of the bundle (field plate coordinates relative to the bundle centre.)
        if centroid:
            centroid_stat, scat_plt = hector_centroider.call_centroider(Probe, Probe_data, Probe_annulus, x, y, mean_x, mean_y, rotation_angle, robot_centre_in_mm, ax, centroid_stat, scat_plt, centroider=centroider, make_plots=make_plots)


        # Add annuli information
        magenta_radius = 226. * 1000.
        yellow_radius = 196.05124 * 1000.
        green_radius = 147.91658 * 1000.
        blue_radius = 92.71721 * 1000.

        ax.add_patch(Circle((robot_centre_in_mm[1], robot_centre_in_mm[0]), blue_radius, color='lightblue', alpha=0.3))
        ax.add_patch(Wedge((robot_centre_in_mm[1], robot_centre_in_mm[0]), magenta_radius, 0, 360., width=10, color='indianred', alpha=0.3))
        ax.add_patch(Wedge((robot_centre_in_mm[1], robot_centre_in_mm[0]), yellow_radius, 0, 360., width=10, color='gold', alpha=0.3))
        ax.add_patch(Wedge((robot_centre_in_mm[1], robot_centre_in_mm[0]), green_radius, 0, 360., width=10, color='lightgreen', alpha=0.3))

        plt.setp(ax.get_xticklabels(), visible=True)
        plt.setp(ax.get_yticklabels(), visible=True)
        ax.text(mean_x, mean_y + scale_factor*750*2, "Probe " + str(Probe),\
                verticalalignment="bottom", horizontalalignment='center')

    # Add the guides
    # if object_guidetab is not None:
    #     ax = utils.display_guides(ax, object_guidetab, scale_factor=scale_factor, tail_length=hexabundle_tail_length)

    # And add some N/E arrows
    # ax = utils.add_NE_arrows(ax)

    ax.invert_yaxis()
    ax.set_ylabel("Robot $x$ coordinate")
    ax.set_xlabel("Robot $y$ coordinate")

    plt.tight_layout()
    plt.show()

    print("---> END")

    if args.outfile is not None:
        fig.savefig(Path(args.outfile), bbox_inches='tight')
    else:
        # plt.show()
        if make_plots: os.system('mv Probe_* ' + str(save_files))
        figfile = save_files / f"plateViewAll_{config['file_prefix']}_Run{obs_number:04}"
        plt.savefig(figfile, bbox_inches='tight', pad_inches=0.3)
        plt.show(block=False) # This bit of the code should have shown the plot before, asking for user input, but had to add 'pause' to get it to show
        plt.pause(0.1)


    # Create separate set of figures to show the centroiding statistics
    if centroid:
        median_RMS = np.median(centroid_stat['CentroidRMS_Err'].to_numpy())
        if median_RMS < 300.0: status = [2, "acceptable"]
        else: status = [1, "somewhat high!"]

        radialErr = (centroid_stat['RadialDist'] - centroid_stat['RadialDistErr'])/2.
        nobs = radialErr[radialErr.abs() > 25.0].size

        ## User to examine and provide input on the centroid fitting. If there are bad fits, specify the Probe name to be excluded from the analysis
        prLightPurple("\t\t ----------------------------------------------------------  \n"
                      "\t\t  SOME EXTERNAL INPUT IS REQUIRED BEFORE FITTING CENTROIDS   \n"
                      "\t\t ----------------------------------------------------------  \n\n"
                      "The RMS scatter from Gaussian fitting, and the centre-of-mass vs Gaussian fitted centroid differences are used to identify bad fits. \n"
                      "\t \033[9{}m {}\033[00m \n\n".format(str(status[0]), str(nobs) + ' Probes with centroiding errors > 25 microns'))
                      # "\t The median RMS for this frame is \033[9{}m {}\033[00m \n "
                      # "\t \033[9{}m {}\033[00m \n\n".format(str(status[0]), status[1], str(status[0]), str(nobs) + ' Probes with centroiding errors > 25 microns'))

        prLightPurple("\t NOTE: The RMS scatter is not always a good indicator of the quality of the fits. For example, under very good seeing conditions, the flux will be mostly concentrated on one fibre or so. In these cases, "
                      "the calculated centroid will be accurate, but the RMS from Gaussian fitting will be misleading. Also, the fits will be affected if there is more than one object in the bundle\n "
                      "So do check the figures produced up to this point to identify whether the data show evidence of smearing out due to bad seeing. \n \n")
        userExclude = str(input("Examine the figure, and specify names of the probes with bad fits to be excluded in a single line (e.g. ABCD) and enter: "))

        ## Start by saving the dataFrame to a CSV file, after excluding the bad data
        exclude_indx = []
        if len(userExclude) > 0:
            for iexclude in range(len(userExclude)):
                exclude_indx.append(np.squeeze(np.where(centroid_stat['Probe'] == userExclude[iexclude])[0]))

            for iexclude in range(len(exclude_indx)):
                iremove = exclude_indx[iexclude]
                xy = np.delete(scat_plt[iremove].get_offsets(), 0, axis=0)
                scat_plt[iremove].set_offsets(xy)
                plt.draw()

            figfile = save_files + f"plateView_{config['file_prefix']}_Run{obs_number:04}"
            plt.savefig(figfile, bbox_inches='tight', pad_inches=0.3)

        centroid_statFinal = centroid_stat.drop(exclude_indx)
        plt.show()


        if savestatdata:
            if obs_number < 10: obs_number_str = '0' + str(obs_number)
            else: obs_number_str = obs_number
            centroid_statFinal.to_csv(str(save_files) + f"/CentroidingStats_{config['file_prefix']}_Run{obs_number_str}" + ".csv", index=True, index_label='Index')

        hector_centroider.make_figures(centroid_statFinal, save_files, robot_centre_in_mm, plate_radius, supltitle, obs_number, hexabundle_tail_length, scale_factor, config)

