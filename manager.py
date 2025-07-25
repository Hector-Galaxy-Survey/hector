"""
Code for organising and reducing Hector (and SAMI) data.
Instructions are overally correct but some are not yet comparable to Hector yet.
=============  

Instructions on how to use this module are given in the docstring for the
Manager class. The following describes some of the under-the-hood details.

This module contains two classes: Manager and FITSFile. The Manager stores
information about an observing run, including a list of all raw files. Each
FITSFile object stores information about a particular raw file. Note that
neither object stores whether or not a file has been reduced; this is checked
on the fly when necessary.

When a Manager object is initiated, it makes an empty list to store the raw
files. It will then inspect given directories to find raw files, with names of
like 01jan10001.fits. It will reject duplicate filenames. Each valid filename
is used to initialise a FITSFile object, which is added to the Manager's file
list. The file itself is also moved into a suitable location in the output
directory structure.

Each FITSFile object stores basic information about the file, such as the path
to the raw file and to the reduced file. The plate and field IDs are
determined automatically from the FITS headers. A check is made to see if the
telescope was pointing at the field listed in the MORE.FIBRES_IFU extension.
If not, the user is asked to give a name for the pointing, which will
generallname of whatever object was being observed. This name is then
added to an "extra" list in the Manager, so that subsequent observations at
the same position will be automatically recognised.

The Manager also keeps lists of the different dark frame exposure lengths (as
both string and float), as well as a list of directories that have been
recently reduced, and hence should be visually checked.

2dfdr is controlled via the tdfdr module. Almost all data reduction steps are
run in parallel, creating a Pool as it is needed.

As individual files are reduced, entries are added to the checklist of
directories to visually inspect. There are some functions for loading up 2dfdr
in the relevant directories, but the user must select and plot the individual
files themself. This whole system is a bit clunky and needs overhauling.

There are a few generators for useful items, most notably Manager.files. This
iterates through all entries in the internal file list and yields those that
satisfy a wide range of optional parameters.

The Manager class can be run in demo mode, in which no actual data reduction
is done. Instead, the pre-calculated results are simply copied into the output
directories. This is useful for demonstrating how to use the Manager without
waiting for the actual data reduction to happen.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from typing import List, Tuple, Dict, Sequence

import shutil
import os
import re
import multiprocessing

import warnings
from functools import wraps
from contextlib import contextmanager
from collections import defaultdict, Counter
from getpass import getpass
from time import sleep
from glob import glob
from pydoc import pager
import itertools
import traceback
import datetime
import csv
import pandas as pd

import six
from six.moves import input

# Set up logging
from . import slogging
log = slogging.getLogger(__name__)
log.setLevel(slogging.WARNING)
# log.enable_console_logging()

import astropy.coordinates as coord
from astropy import units
import astropy.io.fits as pf
from astropy.table import Table
from astropy import __version__ as ASTROPY_VERSION
import numpy as np
from pathlib import Path
import hector
hector_path = str(hector.__path__[0])+'/'
#np.set_printoptions(threshold=np.inf)

try:
    import pysftp

    PYSFTP_AVAILABLE = True
except ImportError:
    PYSFTP_AVAILABLE = False
try:
    from mock import patch

    PATCH_AVAILABLE = True
except ImportError:
    PATCH_AVAILABLE = False

# MLPG: remove my path to molecfit
# MF_BIN_DIR = '/Users/madusha/' +'molecfit_install/bin'  # hector_path[0:-7]+'molecfit_install/bin'
MF_BIN_DIR = hector_path[0:-7]+'molecfit_install/bin'

if not os.path.exists(os.path.join(MF_BIN_DIR,'molecfit')):
        warnings.warn('molecfit not found from '+MF_BIN_DIR+'; Disabling improved telluric subtraction')
        MOLECFIT_AVAILABLE = False
else:
        # print("Note to MLPG: Remove my path to molecfit")
        MOLECFIT_AVAILABLE = True

from .utils.other import find_fibre_table, gzip, ungzip, der_snr
from .utils import IFU
from .utils.term_colours import *
from .general.cubing import dithered_cubes_from_rss_list, get_object_names
from .general.cubing import dithered_cube_from_rss_wrapper
from .general.cubing import scale_cube_pair, scale_cube_pair_to_mag
from .general.align_micron import find_dither
from .dr import fluxcal2, telluric, check_plots, tdfdr, dust, binning
from .dr.throughput import make_clipped_thput_files
from .qc.fluxcal import stellar_mags_cube_pair, stellar_mags_frame_pair
from .qc.fluxcal import throughput, get_sdss_stellar_mags, identify_secondary_standard
from .qc.sky import sky_residuals
from .qc.arc import bad_fibres
from .dr.fflat import correct_bad_fibres
from .dr.twilight_wavecal import wavecorr_frame, wavecorr_av, apply_wavecorr
from .dr.fit_arc_model import arc_model_2d
from .observing.arc_calculate_fwhm import calculate_fwhm

# Temporary edit. Prevent bottleneck 1.0.0 being used.
try:
    import bottleneck.__version__ as BOTTLENECK_VERSION
except ImportError:
    BOTTLENECK_VERSION = ''

if BOTTLENECK_VERSION == '1.0.0':
    raise ImportError('Bottleneck {} has a Blue Whale sized bug. Please update your library NOW')

# Get the astropy version as a tuple of integers
ASTROPY_VERSION = tuple(int(x) for x in ASTROPY_VERSION.split('.'))
if ASTROPY_VERSION[:2] == (0, 2):
    ICRS = coord.ICRSCoordinates
    warnings.warn('Support for astropy {} is being phased out. Please update your software!'.format(ASTROPY_VERSION))
elif ASTROPY_VERSION[:2] == (0, 3):
    ICRS = coord.ICRS
    warnings.warn('Support for astropy {} is being phased out. Please update your software!'.format(ASTROPY_VERSION))
else:
    def ICRS(*args, **kwargs):
        return coord.SkyCoord(*args, frame='icrs', **kwargs)

#below are for SAMI
#IDX_FILES_SLOW = {'580V': 'sami580V_v1_7.idx',
#                  '1500V': 'sami1500V_v1_5.idx',
#                  '1000R': 'sami1000R_v1_7.idx',
#IDX_FILES_FAST = {'580V': 'sami580V.idx',
#                  '1500V': 'sami1500V.idx',
#                  '1000R': 'sami1000R.idx',

IDX_FILES_SLOW = {'580V': 'hector1_v4.idx',
                  '1000R': 'hector2_v4.idx',
                  'SPECTOR1':'hector3_v4.idx',
                  'SPECTOR2':'hector4_v4.idx'}

IDX_FILES_FAST = {'580V': 'hector1_v1.idx',
                  '1000R': 'hector2_v1.idx',
                  'SPECTOR1':'hector3_v1.idx',
                  'SPECTOR2':'hector4_v1.idx'}

                  
IDX_FILES = {'fast': IDX_FILES_FAST,
             'slow': IDX_FILES_SLOW}

GRATLPMM = {'580V': 582.0,
            '1500V': 1500.0,
            '1000R': 1001.0,
            'SPECTOR1': 1099.0,
            'SPECTOR2': 1178.0}

CATALOG_PATH = "./gama_catalogues/"

# This list is used for identifying field numbers in the pilot data.
PILOT_FIELD_LIST = [
    {'plate_id': 'run_6_star_P1', 'field_no': 1,
     'coords': '18h01m54.38s -22d46m49.1s'},
    {'plate_id': 'run_6_star_P1', 'field_no': 2,
     'coords': '21h12m25.06s +04d14m59.6s'},
    {'plate_id': 'run_6_P1', 'field_no': 1,
     'coords': '00h41m35.46s -09d40m29.9s'},
    {'plate_id': 'run_6_P1', 'field_no': 2,
     'coords': '01h13m02.16s +00d26m42.2s'},
    {'plate_id': 'run_6_P1', 'field_no': 3,
     'coords': '21h58m30.77s -08d09m23.9s'},
    {'plate_id': 'run_6_P2', 'field_no': 2,
     'coords': '01h16m01.24s +00d03m23.4s'},
    {'plate_id': 'run_6_P2', 'field_no': 3,
     'coords': '21h55m37.75s -07d40m58.3s'},
    {'plate_id': 'run_6_P3', 'field_no': 2,
     'coords': '01h16m19.66s +00d17m46.9s'},
    {'plate_id': 'run_6_P3', 'field_no': 3,
     'coords': '21h56m37.34s -07d32m16.2s'},
    {'plate_id': 'run_7_star_P1', 'field_no': 1,
     'coords': '20h04m08.32s +07d16m40.6s'},
    {'plate_id': 'run_7_star_P1', 'field_no': 2,
     'coords': '23h14m36.57s +12d45m20.6s'},
    {'plate_id': 'run_7_star_P1', 'field_no': 3,
     'coords': '02h11m46.77s -08d56m09.0s'},
    {'plate_id': 'run_7_star_P1', 'field_no': 4,
     'coords': '05h32m00.40s -00d17m56.9s'},
    {'plate_id': 'run_7_P1', 'field_no': 1,
     'coords': '21h58m27.59s -07d43m50.7s'},
    {'plate_id': 'run_7_P1', 'field_no': 2,
     'coords': '00h40m12.73s -09d31m47.5s'},
    {'plate_id': 'run_7_P2', 'field_no': 1,
     'coords': '21h56m27.49s -07d12m02.4s'},
    {'plate_id': 'run_7_P2', 'field_no': 2,
     'coords': '00h40m33.40s -09d04m21.6s'},
    {'plate_id': 'run_7_P3', 'field_no': 1,
     'coords': '21h56m27.86s -07d46m17.1s'},
    {'plate_id': 'run_7_P3', 'field_no': 2,
     'coords': '00h41m25.78s -09d17m14.4s'},
    {'plate_id': 'run_7_P4', 'field_no': 1,
     'coords': '21h57m48.55s -07d23m40.6s'},
    {'plate_id': 'run_7_P4', 'field_no': 2,
     'coords': '00h42m34.09s -09d12m08.1s'}]

# Things that should be visually checked
# Priorities: Lower numbers (more negative) should be done first
# Each key ('TLM', 'ARC',...) matches to a check method named 
# check_tlm, check_arc,...
CHECK_DATA = {
    'BIA': {'name': 'Bias',
            'ndf_class': 'BIAS',
            'spectrophotometric': None,
            'priority': -3,
            'group_by': ('ccd', 'date')},
    'DRK': {'name': 'Dark',
            'ndf_class': 'DARK',
            'spectrophotometric': None,
            'priority': -2,
            'group_by': ('ccd', 'exposure_str', 'date')},
    'LFL': {'name': 'Long-slit flat',
            'ndf_class': 'LFLAT',
            'spectrophotometric': None,
            'priority': -1,
            'group_by': ('ccd', 'date')},
    'TLM': {'name': 'Tramline map',
            'ndf_class': 'MFFFF',
            'spectrophotometric': None,
            'priority': 0,
            'group_by': ('date', 'ccd', 'field_id')},
    'ARC': {'name': 'Arc reduction',
            'ndf_class': 'MFARC',
            'spectrophotometric': None,
            'priority': 1,
            'group_by': ('date', 'ccd', 'field_id')},
    'FLT': {'name': 'Flat field',
            'ndf_class': 'MFFFF',
            'spectrophotometric': None,
            'priority': 2,
            'group_by': ('date', 'ccd', 'field_id')},
    'SKY': {'name': 'Twilight sky',
            'ndf_class': 'MFSKY',
            'spectrophotometric': None,
            'priority': 3,
            'group_by': ('date', 'ccd', 'field_id')},
    'OBJ': {'name': 'Object frame',
            'ndf_class': 'MFOBJECT',
            'spectrophotometric': None,
            'priority': 4,
            'group_by': ('date', 'ccd', 'field_id', 'name')},
    'FLX': {'name': 'Flux calibration',
            'ndf_class': 'MFOBJECT',
            'spectrophotometric': True,
            'priority': 5,
            'group_by': ('date', 'field_id', 'name')},
    'TEL': {'name': 'Telluric correction',
            'ndf_class': 'MFOBJECT',
            'spectrophotometric': None,
            'priority': 6,
            'group_by': ('date', 'ccd', 'field_id')},
    'ALI': {'name': 'Alignment',
            'ndf_class': 'MFOBJECT',
            'spectrophotometric': False,
            'priority': 7,
            'group_by': ('field_id',)},
    'CUB': {'name': 'Cubes',
            'ndf_class': 'MFOBJECT',
            'spectrophotometric': False,
            'priority': 8,
            'group_by': ('field_id',)}}
# Extra priority for checking re-reductions
PRIORITY_RECENT = 100

STELLAR_MAGS_FILES = [
    (hector_path+'standards/secondary/APMCC_0917_STARS.txt', 'ATLAS',
     (0.076, 0.059, 0.041, 0.030, 0.023)),
    (hector_path+'standards/secondary/Abell_3880_STARS.txt', 'ATLAS',
     (0.064, 0.050, 0.034, 0.025, 0.019)),
    (hector_path+'standards/secondary/Abell_4038_STARS.txt', 'ATLAS',
     (0.081, 0.063, 0.044, 0.033, 0.024)),
    (hector_path+'standards/secondary/EDCC_0442_STARS.txt', 'ATLAS',
     (0.071, 0.052, 0.038, 0.029, 0.020)),
    (hector_path+'standards/secondary/Abell_0085.fstarcat.txt', 'SDSS_cluster',
     (0.0, 0.0, 0.0, 0.0, 0.0)),
    (hector_path+'standards/secondary/Abell_0119.fstarcat.txt', 'SDSS_cluster',
     (0.0, 0.0, 0.0, 0.0, 0.0)),
    (hector_path+'standards/secondary/Abell_0168.fstarcat.txt', 'SDSS_cluster',
     (0.0, 0.0, 0.0, 0.0, 0.0)),
    (hector_path+'standards/secondary/Abell_2399.fstarcat.txt', 'SDSS_cluster',
     (0.0, 0.0, 0.0, 0.0, 0.0)),
    (hector_path+'standards/secondary/sdss_stellar_mags.csv', 'SDSS_GAMA',
     (0.0, 0.0, 0.0, 0.0, 0.0))]


def stellar_mags_files():
    """Yield details of each stellar magnitudes file that can be found."""
    # MLPG: Commenting out the SAMI extra file path, and adding the Hector extra file downloaded by user path
    for mags_file in STELLAR_MAGS_FILES:
        # The pre-determined ones listed above
        yield mags_file
    # for path in glob(hector_path+'standards/secondary/sdss_stellar_mags_*.csv'):
        # Extra files that have been downloaded by the user
        # yield (path, 'SDSS_GAMA', (0.0, 0.0, 0.0, 0.0, 0.0))
    for path in glob(hector_path+'standards/secondary/Hector_tiles/Hector_secondary_standards_shortened.csv'):
        # Extra files that have been downloaded by the user
        yield (path, 'Hector_mags', (0.0, 0.0, 0.0, 0.0, 0.0))


class Manager:
    """Object for organising and reducing Hector (and SAMI) data.

    Initial setup
    =============

    You start a new manager by creating an object, and telling it where
    to put its data, e.g.:

    >>> import hector
    >>> mngr = hector.manager.Manager('130305_130317')

    The directory name you give it should normally match the dates of the
    observing run that you will be reducing.

    IMPORTANT: When starting a new manager, the directory you give it should be 
    non-existent or empty. Inside that directory it will create "raw" and
    "reduced" directories, and all the subdirectories described on the SAMI wiki
    (http://sami-survey.org/wiki/staging-disk-file-structure). You do not need
    to create the subdirectories yourself.

    Note that the manager carries out many I/O tasks that will fail if you
    change directories during your python session. If you do need to change
    directory, make sure you change back before running any further manager
    tasks. This limitation may be addressed in a future update, if enough people
    want it. You may be able to avoid it by providing an absolute path when you
    create the manager, but no guarantee is made.

    By default, the manager will perform full science-grade reductions, which
    can be quite slow. If you want quick-look reductions only (e.g. if you are
    at the telescope, or for testing purposes), then set the 'fast' keyword:

    >>> mngr = hector.manager.Manager('130305_130317', fast=True)

    At this point the manager is not aware of any actual data - skip to
    "Importing data" and carry on from there.

    Deriving the tram-line maps from the twilight sky frames (blue arm only)
    ========================================================================

    The keyword `use_twilight_tlm_blue` instructs the manager to use the
    twilight sky frames to derive the tram-line maps (default value is `False`).
    For the blue arm, using tram-line maps derived from the twilight sky frames
    reduces the noise at the blue end of the spectra.

    When this keyword is set to `True`, two sets of tram-line maps are derived:
    one contains the tram-line maps from each twilight sky frame, and one
    contains the tram-line maps derived from each dome flat frame. The tram-line
    maps derived from the dome flats are used: a) for the red arm in *any* case 
    b) for the blue arm if no twilight frame was available to derive the
    tram-line maps.

    >>> mngr = hector.manager.Manager('130305_130317',use_twilight_tlm_blue=True)

    The reductions will search for a twilight from the same plate to use as
    a TLM file.  If one from the same plate cannot be found, a twilight from another
    plate (or another night) will be used in preference to a dome flat.  The current
    default is use_twilight_tlm_blue=False until full testing has been completed.

    For the on site data reduction, it might be advisable to use `False`
    (default), because this requires less time.
    
    Improving the blue arm wavelength calibration using twilight sky frames
    =======================================================================
    
    The keyword 'improve_blue_wavecorr' instructs the manager to determine
    an improved blue arm wavelength solution from the twilight sky frames.
    
    When this keyword is set to true, the reduced twilight sky frame spectra
    are compared to a high-resolution solar spectrum (supplied as part of the
    SAMI package) to determine residual wavelength shifts. An overall
    fibre-to-fibre wavelength offset is derived by averaging over all twilight
    sky frames in a run. This average offset is stored in a calibration file in
    the root directory of the run. These shifts are then applied to all arc frames. 

    Applying telluric correction to primary standards before flux calibration
    =========================================================================

    The keyword 'telluric_correct_primary' instructs the manager to use molecfit
    to telluric correct the primary standard stars before determining the flux
    calibration transfer function. This is only applied if molecfit is installed.

    This keyword should only be set if the reference spectra for the primary
    standard stars have themselves been telluric corrected. If they have not this
    will result in highly unphysical transfer functions.

    Continuing a previous session
    =============================

    If you quit python and want to get back to where you were, just restart
    the manager on the same directory (after importing hector):

    >>> mngr = hector.manager.Manager('130305_130317')

    It will search through the subdirectories and restore its previous
    state. By default it will restore previously-assigned object names that
    were stored in the headers. To re-assign all names:

    >>> mngr = hector.manager.Manager('data_directory', trust_header=False)

    As before, set the 'fast' keyword if you want quick-look reductions.

    Importing data
    ==============

    After creating the manager, you can import data into it:

    >>> mngr.import_dir('path/to/raw/data')

    It will copy the data into the data directory you defined earlier,
    putting it into a neat directory structure. You will typically need to
    import all of your data before you start to reduce anything, to ensure
    you have all the bias, dark and lflat frames.

    When importing data, the manager will do its best to work out what the
    telescope was pointing at in each frame. Sometimes it wont be able to and
    will ask you for the object name to go with a particular file. Depending on
    the file, you should give an actual object name - e.g. HR7950 or NGC2701 -
    or a more general description - e.g. SNAFU or blank_sky. If the telescope
    was in fact pointing at the field specified by the configuration .csv file -
    i.e. the survey field rather than some extra object - then enter main. It
    will also ask you which of these objects should be used as
    spectrophotometric standards for flux calibration; simply enter y or n as
    appropriate.

    Importing at the AAT
    ====================

    If you're at the AAT and connected to the network there, you can import
    raw data directly from the data directories there:

    >>> mngr.import_aat()

    The first time you call this function you will be prompted for a username
    and password, which are saved for future times. By default the data from
    the latest date is copied; if you want to copy earlier data you can
    specify the date:

    >>> mngr.import_aat(date='140219')

    Only new files are copied over, so you can use this function to update
    your manager during the night. An ethernet connection is strongly
    recommended, as the data transfer is rather slow over wifi.

    Reducing bias, dark and lflat frames
    ====================================

    The standard procedure is to reduced all bias frames and combine them,
    then reduce all darks and combine them, then reduce all long-slit flats
    and combine them. To do this:

    >>> mngr.reduce_bias()
    >>> mngr.combine_bias()
    >>> mngr.reduce_dark()
    >>> mngr.combine_dark()
    >>> mngr.reduce_lflat()
    >>> mngr.combine_lflat()

    The manager will put in symbolic links as it goes, to ensure the
    combined files are available wherever they need to be.

    If you later import more of the above frames, you'll need to re-run the
    above commands to update the combined frames.

    In these (and later) commands, by default nothing happens if the
    reduced file already exists. To override this behaviour, set the
    overwrite keyword, e.g.

    >>> mngr.reduce_bias(overwrite=True)

    If you later import more data that goes into new directories, you'll
    need to update the links:

    >>> mngr.link_bias()
    >>> mngr.link_dark()
    >>> mngr.link_lflat()

    Reducing fibre flat, arc and offset sky frames
    ==============================================

    2dfdr works by creating a tramline map from the fibre flat fields, then
    reducing the arc frames, then re-reducing the fibre flat fields. After
    this the offset skies (twilights) can also be reduced.

    >>> mngr.make_tlm()
    >>> mngr.reduce_arc()
    >>> mngr.reduce_fflat()
    >>> mngr.reduce_sky()

    In any of these commands, keywords can be used to restrict the files
    that get reduced, e.g.

    >>> mngr.reduce_arc(ccd='ccd_1', date='130306',
                        field_id='Y13SAR1_P002_09T004')

    will only reduce arc frames for the blue CCD that were taken on the
    6th March 2013 for the field ID Y13SAR1_P002_09T004. Allowed keywords
    and examples are:
    
        date            '130306'
        plate_id        'Y13SAR1_P002_09T004_12T006'
        plate_id_short  'Y13SAR1_P002'
        field_no        0
        field_id        'Y13SAR1_P002_09T004'
        ccd             'ccd_1'
        exposure_str    '10'
        min_exposure    5.0
        max_exposure    20.0
        reduced_dir     ('130305_130317/reduced/130305/'
                         'Y13SAR1_P001_09T012_15T001/Y13SAR1_P001_09T012/'
                         'calibrators/ccd_1')

    Note the reduced_dir is a single long string, in this example split
    over several lines.

    Reducing object frames
    ======================

    Once all calibration files have been reduced, object frames can be
    reduced in a similar manner:

    >>> mngr.reduce_object())

    This function takes the same keywords as described above for fibre
    flats etc.

    Flux calibration
    ================

    The flux calibration requires a series of steps. First, a transfer
    function is derived for each observation of a standard star:

    >>> mngr.derive_transfer_function()

    Next, transfer functions from related observations are combined:

    >>> mngr.combine_transfer_function()

    The combined transfer functions can then be applied to the data to
    flux calibrate it:

    >>> mngr.flux_calibrate()

    Finally, the telluric correction is derived and applied with a
    single command:

    >>> mngr.telluric_correct()

    As before, keywords can be set to restrict which files are processed,
    and overwrite=True can be set to force re-processing of files that
    have already been done.

    A option is to flux calibrate from the secondary stars.  This can 
    have several advantages.  For example, it means that any residual 
    extinction variations can be removed on a frame-by-frame basis.  It can
    also provide better calibration when the pirmary standards have too much
    scattered light in the blue.  This can cause PSF fitting problems and so
    lead to some systematics at trhe far blue end (can be 10-20%).  To fit
    using the secondaries we need to match the star to a model (Kurucz 
    theoretical models) and we do this using ppxf to find the best model. 
    The model used is a linear combination of the 4 templates closest in
    Teff and [Fe/H] to the observed star.  The best template is estimated
    from all the frames in the field.  This is then use to estimate
    the transfer function for inditivual frames.  There is also an option
    to average the TF across all the frames used.  The model star spectrum
    is also scaled to the SDSS photometry so that application of the TF 
    also does the equivalent of the scale_frames() proceedure, so this
    should not need to be done if fluxcal_secondary() is used.

    >>> mngr.fluxcal_secondary()

    of if averaging:

    >>> mngr.fluxcal_secondary(use_av_tf_sec=True)  (default)

    >>> mngr.fluxcal_secondary(use_av_tf_sec=False)  (or not)

    Scaling frames
    ==============

    To take into account variations in atmospheric transmission over the
    course of a night (or between nights), the flux level in each frame is
    scaled to set the standard star flux to the catalogue level. First the
    manager needs to find out what that catalogue level is:

    >>> mngr.get_stellar_photometry()

    This will check the stars in your observing run against the inbuilt
    catalogue. If any are missing you will be prompted to download the
    relevant data from the SDSS website.

    Once the stellar photometry is complete, you can continue with scaling
    each frame:

    >>> mngr.scale_frames()
    
    Cubing
    ======

    Before the frames can be combined their relative offsets must be
    measured:

    >>> mngr.measure_offsets()

    The individual calibrated spectra can then be turned into datacubes:

    >>> mngr.cube()

    A final rescaling is done, to make sure everything is on the correct
    flux scale:

    >>> mngr.scale_cubes()

    If you have healpy installed and the relevant dust maps downloaded, you
    can record the E(B-V) and attenuation curves in the datacubes:

    >>> mngr.record_dust()

    It's safe to leave this step out if you don't have the necessary
    components.

    Finally, the cubes can be gzipped to save space/bandwidth. You might
    want to leave this until after the output checking (see below), to
    improve read times.

    >>> mngr.gzip_cubes()

    Checking outputs
    ================

    As the reductions are done, the manager keeps a record of reduced
    files that need to be plotted to check that the outputs are ok. These 
    are grouped in sets of related files. To print the checks that need to 
    be done:

    >>> mngr.print_checks()

    Separate lists are returned for checks that have never been done, and
    those that haven't been done since the last re-reduction. You can
    specify one or the other by passing 'ever' or 'recent' as an argument
    to print_checks().

    To perform the next check:

    >>> mngr.check_next_group()

    The manager will either load the 2dfdr GUI and give you a list of
    files to check, or will make some plots in python. Check the things it
    tells you to, keeping a note of any files that need to be disabled or
    examined further (if you don't know what to do about a file, ask a
    friendly member of the data reduction working group).

    If 2dfdr was loaded, no more commands can be entered until you close it. 
    When you do so, or immediately for python-based plots, you will be asked
    whether the files can be removed from the checklist. Enter 'y'
    if you have checked all the files, 'n' otherwise.

    You can also perform a particular check by specifying its index in the
    list:

    >>> mngr.check_group(3)

    The index numbers are given when mngr.print_checks() is called.

    Checking the outputs is a crucial part of data reduction, so please
    make sure you do it thoroughly, and ask for assistance if you're not
    sure about anything.

    Disabling files
    ===============

    If there are any problems with some files, either because an issue is
    noted in the log or because they wont reduce properly, you can disable
    them, preventing them from being used in any later reductions:

    >>> mngr.disable_files(['06mar10003', '06mar20003.fits', '06mar10047'])

    If you only have one file you want to disable, you still need the
    square brackets. The filenames can be with or without the extension (.fits)
    but must be without the directory. You can disable lots of files at a time
    using the files generator:

    >>> mngr.disable_files(mngr.files(
                date='130306', field_id='Y13SAR1_P002_09T004'))

    This allows the same keywords as described earlier, as well as:

        ndf_class           'MFFFF'
        reduced             False
        tlm_created         False
        flux_calibrated     True
        telluric_corrected  True
        name                'LTT2179'

    For example, specifying the first three of these options as given
    would disable all fibre flat fields that had not yet been reduced and
    had not yet had tramline maps created. Specifying the last three
    would disable all observations of LTT2179 that had already been flux
    calibrated and telluric corrected.

    To re-enable files:

    >>> mngr.enable_files(['06mar10003', '06mar20003', '06mar10047'])

    This function follows exactly the same syntax as disable_files.

    Summarising results
    ===================

    At any time you can print out a summary of the object frames observed,
    including some basic quality control metrics:

    >>> mngr.qc_summary()

    The QC values are not filled in until certain steps of the data
    reduction have been done; you need to get as far as mngr.scale_frames()
    to see everything. While observing, keep an eye on the seeing and the
    transmission. If the seeing is above 3" or the transmission is below
    about 0.7, the data are unlikely to be of much use.

    Changing object names and spectrophotometric flags
    ==================================================

    If you want to change the object names for one or more files, or change
    whether they should be used as spectrophotometric standards, use the
    following commands:

    >>> mngr.update_name(['06mar10003', '06mar20003'], 'new_name')
    >>> mngr.update_spectrophotometric(['06mar10003', '06mar20003'], True)

    In the above example, the given files are set to have the name
    'new_name' and they are listed as spectrophotometric standards. The
    options for spectrophotometric flags must be entered as True or
    False (without quote marks, with capital letter). You can use the
    same file generator syntax as for disabling/enabling files
    (above), so for example if you realise that on importing some
    files you entered LTT2197 instead of LTT2179 you can correct all
    affected files at once:

    >>> mngr.update_name(mngr.files(name='LTT2197'), 'LTT2179')

    Changing speed/accuracy of the reductions
    =========================================

    If you want to switch between fast and slow (rough vs accurate) reductions:

    >>> mngr.change_speed()

    Or to ensure you end up with a particular speed, specify 'fast' or 'slow':

    >>> mngr.change_speed('slow')

    Reducing everything in one go
    =============================

    You can perform all of the above reduction tasks with a single command:

    >>> mngr.reduce_all()

    You should only do this for re-reducing data that you have previously
    checked and are confident that nothing will go wrong, and after
    disabling all unwanted files. Otherwise, you can easily have a tramline
    map go haywire (for example) and wreck everything that follows.

    Parallel processing
    ===================

    You can make use of multi-core machines by setting the number of CPUs to
    use when the manager is made, e.g.:

    >>> mngr = hector.manager.Manager('130305_130317', n_cpu=4)

    Note that you cannot run multiple instances of 2dfdr in the same
    directory, so you wont always be able to use all your cores. To keep
    track of where 2dfdr is running, the manager makes empty directories
    called '2dfdrLockDir'. These will normally be cleaned up when 2dfdr
    completes, but after a bad crash they may be left behind and block
    any following reductions. In this case, you can force their removal:

    >>> mngr.remove_directory_locks()

    Including data from other runs
    ==============================

    If a field has been observed over more than one run, the manager will
    need to be made aware of the pre-existing data to make combined
    datacubes. Note that this is only necessary for the final data
    reduction, so observers do not need to worry about this.

    To combine the data, first create a manager for each run (you may
    already have done this):

    >>> mngr = hector.manager.Manager('2014_04_24-2014_05_04')
    >>> mngr_old = hector.manager.Manager('2014_05_23-2014_06_01')

    Then create the link:

    >>> mngr.link_manager(mngr_old)

    Now `mngr` will include files from `mngr_old` when necessary, i.e. for
    these steps:

    >>> mngr.measure_offsets()
    >>> mngr.cube()
    >>> mngr.scale_cubes()
    >>> mngr.bin_cubes()

    For all previous steps the two managers still act independently, so
    you need to follow through up to scale_frames() for each manager
    individually.

    Other functions
    ===============

    The other functions defined probably aren't useful to you.
    """

    # Task list provides the list of standard reduction tasks in the necessary
    # order. This is used by `reduce_all`, and also by each reduction step to provide instructions on the next step to run.
    task_list = (
        ('reduce_bias', True),
        ('combine_bias', False),
        ('reduce_dark', True),
        ('combine_dark', False),
        ('reduce_lflat', True),
        ('combine_lflat', False),
        ('make_tlm', True),
        ('reduce_arc', True),
        ('reduce_fflat', True),
        ('reduce_sky', True),
        ('reduce_object', True),
        ('derive_transfer_function', True),
        ('combine_transfer_function', True),
        ('flux_calibrate', True),
        ('telluric_correct', True),
        ('fluxcal_secondary',True),
        ('scale_frames', True),
        ('measure_offsets', True),
        ('cube', True),
        ('scale_cubes', True),
        ('bin_cubes', True),
        ('record_dust', True),
        ('bin_aperture_spectra', True),
        ('gzip_cubes', True)
    )

    def __init__(self, root, copy_files=False, move_files=False, fast=False,
                 gratlpmm=GRATLPMM, n_cpu=1,demo_data_source='demo',
                 use_twilight_tlm_blue=False, use_twilight_flat_blue=False,
                 improve_blue_wavecorr=False, telluric_correct_primary=False,
                debug=False, dummy=False, use_twilight_tlm_all=False, use_twilight_flat_all=False,
                fit_arc_model_2d=False):
        if fast:
            self.speed = 'fast'
        else:
            self.speed = 'slow'
        self.idx_files = IDX_FILES[self.speed]
        # define the internal flag that allows twilights to be used for
        # making tramline maps:
        self.use_twilight_tlm_blue = use_twilight_tlm_blue

        # define the internal flag that allows twilights to be used for
        # fibre flat fielding:
        self.use_twilight_flat_blue = use_twilight_flat_blue

        # define the internal flag that allows twilights to be used for *all* ccds
        self.use_twilight_tlm_all = use_twilight_tlm_all
        self.use_twilight_flat_all = use_twilight_flat_all
        if(self.use_twilight_tlm_all):
            self.use_twilight_tlm_blue = False
        if(self.use_twilight_flat_all):
            self.use_twilight_flat_blue = False

        # define the internal flag that specifies the improved twlight wavelength
        # calibration step should be applied
        self.improve_blue_wavecorr = improve_blue_wavecorr
        self.fit_arc_model_2d = fit_arc_model_2d
        # Internal flag to set telluric correction for primary standards
        self.telluric_correct_primary = telluric_correct_primary
        self.gratlpmm = gratlpmm
        self.n_cpu = n_cpu
        self.root = root
        self.abs_root = os.path.abspath(root)
        self.tmp_dir = os.path.join(self.abs_root, 'tmp')
        # Match objects within 1'
        if ASTROPY_VERSION[0] == 0 and ASTROPY_VERSION[1] == 2:
            self.matching_radius = coord.AngularSeparation(
                0.0, 0.0, 0.0, 1.0, units.arcmin)
        else:
            self.matching_radius = coord.Angle('0:1:0 degrees')
        self.file_list = []
        self.extra_list = []
        self.dark_exposure_str_list = []
        self.dark_exposure_list = []
        self.linked_managers = []
        self.cwd = os.getcwd()
        if 'IMP_SCRATCH' in os.environ:
            self.imp_scratch = os.environ['IMP_SCRATCH']
        else:
            self.imp_scratch = None
        self.scratch_dir = None
        self.min_exposure_for_throughput = 900.0
        self.min_exposure_for_sky_wave = 900.0
        self.min_exposure_for_5577pca = 599.0
        self.aat_username = None
        self.aat_password = None
        self.inspect_root(copy_files, move_files)
        if self.find_directory_locks():
            print('Warning: directory locks in place!')
            print('If this is because you killed a crashed manager, clean them')
            print('up using mngr.remove_directory_locks()')

        if use_twilight_tlm_blue or use_twilight_tlm_all:
            print('Using twilight frames to derive TLM and profile map')
        else:
            print('NOT using twilight frames to derive TLM and profile map')

        if use_twilight_flat_blue or use_twilight_flat_all:
            print('Using twilight frames for fibre flat field')
        else:
            print('NOT using twilight frames for fibre flat field')

        if improve_blue_wavecorr:
            print('Applying additional twilight-based wavelength calibration step')
        else:
            print('NOT applying additional twilight-based wavelength calibration step')
        if fit_arc_model_2d:
            print('Arc solutions are improved based on 2D arc modelling')

        if telluric_correct_primary:
            print('Applying telluric correction to primary standard stars before flux calibration')
            print('WARNING: Only do this if the reference spectra for the primary standards have good telluric correction')
        else:
            print('NOT applying telluric correction to primary standard stars')

        self.dummy = dummy
        self._debug = False
        self.debug = debug

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, value):
        if not isinstance(value, bool):
            raise ValueError("debug must be set to a boolean value.")
        if not value == self._debug:
            if value:
                log.setLevel(slogging.DEBUG)
                tdfdr.log.setLevel(slogging.DEBUG)
            else:
                log.setLevel(slogging.WARNING)
                tdfdr.log.setLevel(slogging.WARNING)
            self._debug = value

    def next_step(self, step, print_message=False):
        task_name_list = list(map(lambda x: x[0], self.task_list))
        current_index = task_name_list.index(step)
        if current_index + 1 < len(task_name_list):
            next_step = task_name_list[current_index + 1]
        else:
            # nothing left
            next_step = None
        if print_message:
            print("'{}' step complete. Next step is '{}'".format(step, next_step))
        return next_step

    def __repr__(self):
        return "HectorManagerInstance at {}".format(self.root)

    def map(self, function, input_list):
        """Map inputs to a function, using built-in map or multiprocessing."""
        if not input_list:
            # input_list is empty. I expected the map functions to deal with
            # this issue, but in one case it hung on aatmacb, so let's be
            # absolutely sure to avoid the issue
            print('empty input_list, returning...')
            return []
        # if asyncio.iscoroutinefunction(function):
        #
        #     result_list = []
        #
        #     # loop = asyncio.new_event_loop()
        #     loop = asyncio.get_event_loop()
        #     # Break up the overall job into chunks that are n_cpu in size:
        #     for i in range(0, len(input_list), self.n_cpu):
        #         print("{} jobs total, running {} to {} in parallel".format(len(input_list), i, min(i+self.n_cpu, len(input_list))))
        #         # Create an awaitable object which can be used as a future.
        #         # This is the job that will be run in parallel.
        #         @asyncio.coroutine
        #         def job():
        #             tasks = [function(item) for item in input_list[i:i+self.n_cpu]]
        #             # for completed in asyncio.as_completed(tasks):  # print in the order they finish
        #             #     await completed
        #             #     # print(completed.result())
        #             sub_results = yield from asyncio.gather(*tasks, loop=loop)
        #             result_list.extend(sub_results)
        #
        #         loop.run_until_complete(job())
        #     # loop.close()
        #
        #     return np.array(result_list)
        #
        # else:
        # Fall back to using multiprocessing for non-coroutine functions
        if self.n_cpu == 1:
            result_list = list(map(function, input_list))
        else:
            pool = multiprocessing.Pool(self.n_cpu)
            result_list = pool.map(function, input_list, chunksize=1)
            pool.close()
            pool.join()
        return result_list

    def inspect_root(self, copy_files, move_files, trust_header=True):
        """Add details of existing files to internal lists."""
        files_to_add = []
        for dirname, subdirname_list, filename_list in os.walk(os.path.join(self.abs_root, "raw")):
            for filename in filename_list:
                if self.file_filter(filename):
                    full_path = os.path.join(dirname, filename)
                    files_to_add.append(full_path)
        files_to_add = sorted(files_to_add)

        assert len(set(files_to_add)) == len(files_to_add), "Some files would be duplicated on manager startup."

        if self.n_cpu == 1:
            fits_list = list(map(FITSFile, files_to_add))
        else:
            pool = multiprocessing.Pool(self.n_cpu)
            fits_list = pool.map(FITSFile, files_to_add, chunksize=20)
            pool.close()
            pool.join()

        if os.path.exists(self.abs_root):
            f = open(self.abs_root+'/filelist_'+str(self.abs_root)[-13:]+'.txt', 'w')
            f.write('#filename  ndfclass  field  object  standrad  disabled  exposure\n')
            f.close()

        for fits in fits_list:
            self.import_file(fits,
                             trust_header=trust_header,
                             copy_files=copy_files,
                             move_files=move_files)

            #modify header or fibre table following modified_frames.txt
            if os.path.exists(str(self.abs_root)+'/modified_frames_'+str(self.abs_root)[-13:]+'.txt'): 
                moditem = np.loadtxt(str(self.abs_root)+'/modified_frames_'+str(self.abs_root)[-13:]+'.txt',delimiter=',',dtype={'names': ('frame','type','key','fibre','value','comment','reason'), 'formats': ('U10','U10','U20','U20','U100','U100','U100')})
                if moditem.ndim == 0:
                    moditem = np.array([moditem], dtype=moditem.dtype)
                if (fits.filename[0:10] in moditem['frame']): #the name is on it
                    sub = np.where(moditem['frame'] == fits.filename[0:10])
                    for ind in sub[0]:
                        mtype = moditem['type'][ind]; mkey = moditem['key'][ind]; mvalue = moditem['value'][ind]; mcomment = moditem['comment'][ind];mfibre= moditem['fibre'][ind]
                        if(mtype.strip() == 'header'): #change primary header
                            if(mkey == 'PLATE_ID'):
                                mkey = 'PLATEID'
                            fits.add_header_item(mkey,mvalue.strip(),mcomment)
                            print('  Change the header keyword '+mkey+' to '+mvalue)
                        if(mtype.strip() == 'ftable'): #change fibre table
                            hdulist=pf.open(fits.raw_path,'update')
                            tab=hdulist['MORE.FIBRES_IFU'].data
                            if mfibre.isdigit():
                                tab[mkey][int(mfibre)-1] = mvalue
                                print('  Change the '+mkey+' of fibre '+mfibre+' to '+mvalue)
                            elif mfibre.isalpha():
                                tab[mkey][np.where(np.char.strip(tab['PROBENAME'])==mfibre)] = mvalue
                                print('  Change the '+mkey+' of all fibre in bundle '+mfibre+' to '+mvalue)
                            hdulist.flush(); hdulist.close()


    def file_filter(self, filename):
        """Return True if the file should be added."""
        # Match filenames of the form 01jan10001.fits
        return (re.match(r'[0-3][0-9]'
                         r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)'
                         r'[1-4][0-9]{4}\.(fit|fits|FIT|FITS)$',
                         filename)
                and (self.fits_file(filename) is None))

    def import_file(self, source,
                    trust_header=True, copy_files=True, move_files=False):
        """Add details of a file to the manager"""
        if not isinstance(source, FITSFile):
            # source_path = os.path.join(dirname, filename)
            # Initialize an instance of the FITSFile:
            filename = os.path.basename(source)
            fits = FITSFile(source)
        else:
            filename = source.filename
            fits = source
        if fits.copy:
            # print 'this is a copy, do not import:',dirname,filename
            # This is a copy of a file, don't add it to the list
            return
        if fits.ndf_class not in [
            'BIAS', 'DARK', 'LFLAT', 'MFFFF', 'MFARC', 'MFSKY',
            'MFOBJECT']:
            print('Unrecognised NDF_CLASS for {}: {}'.format(
                filename, fits.ndf_class))
            print('Skipping this file')
            return
        if fits.ndf_class == 'DARK':
            if fits.exposure_str not in self.dark_exposure_str_list:
                self.dark_exposure_str_list.append(fits.exposure_str)
                self.dark_exposure_list.append(fits.exposure)
        self.set_raw_path(fits)

        if os.path.abspath(fits.source_path) != os.path.abspath(fits.raw_path):
            if copy_files:
                print('Copying file:', filename)
                self.update_copy(fits.source_path, fits.raw_path)
            if move_files:
                print('Moving file: ', filename)
                self.move(fits.source_path, fits.raw_path)
            if not copy_files and not move_files:
                print('Warning! Adding', filename, 'in unexpected location')
                fits.raw_path = fits.source_path
            self.set_name(fits, trust_header=trust_header)
        else:
            self.set_name(fits, trust_header=trust_header)
            print('Adding file: ', filename, fits.ndf_class, fits.plate_id, fits.name, fits.exposure_str, fits.lamp, "\033[91m{}\033[00m".format('*disabled*') if fits.do_not_use else '')

            f = open(self.abs_root+'/filelist_'+str(self.abs_root)[-13:]+'.txt', 'a')
            #f.write(filename+' '+fits.ndf_class+' '+(fits.plate_id or 'None')+' '+(fits.name or 'None')+' '+(str(fits.spectrophotometric) or 'None')+' '+(str(fits.do_not_use) or 'None')+' '+fits.exposure_str+' '+(str(fits.lamp) or 'None')+'\n')
            f.write(filename+' '+fits.ndf_class+' '+(fits.plate_id or 'None')+' '+(fits.name or 'None')+' '+(str(fits.spectrophotometric) or 'None')+' '+(str(fits.do_not_use) or 'None')+' '+fits.exposure_str+'\n')
            f.close()
            
        fits.set_check_data()
        self.set_reduced_path(fits)

        if fits.ndf_class == 'MFOBJECT' and fits.adc == 'Idle':
                prRed(f"  WARNING: ADC status for {filename} is idle !")

        #Sree: ccd_4 data from 26, 27 Apr 2023, where overscan region is different but header does not point the right pixels
        if (fits.ccd == 'ccd_4' and fits.header['RO_PORTS'] == 'B'):
            fits.add_header_item('DETECXS', 41,'First column of detectorm, set by Hector manager')
            fits.add_header_item('DETECXE', 4136,'Last column of detector, set by Hector manager')
            fits.add_header_item('WINDOXS1', 41,'First column of window 1, set by Hector manager')
            fits.add_header_item('WINDOXE1', 4136,'Last column of window 1, set by Hector manager')
            fits.add_header_item('WINDOXS2', 1,'First column of window 2, set by Hector manager')
            fits.add_header_item('WINDOXE2', 40,'Last column of window 2, set by Hector manager')
            print('    DETECX? and WINDOX?? keywords have been modified having RO_PORTS=B from',fits.filename)

        #Sree (Feb 2024): A bias column is artificially added in ccd_4 at x = 2048 for Hector
        #It has been permanantly fixed from 2024 data
        #It has been temporarily returned during 17-20 July 2025 due to UPS failure?
        try:
            biascol_modified = fits.header['BIASCOL']
        except KeyError:
            biascol_modified = 'F'
        if ( (((fits.epoch>2022.0) and (fits.epoch<2023.99)) or ((fits.epoch>2025.540) and (fits.epoch<2025.551))) and (fits.ccd == 'ccd_4') and biascol_modified != 'T'):
            new_path = os.path.join(fits.raw_dir, (fits.filename[:10]+'_original.fits'))
            if not os.path.exists(new_path):
                shutil.copy2(fits.raw_path, new_path)
            # relocate the bias column x=2048 to the end of the image and shift the image to the left by 1 pixel from x=2049
            hdulist = pf.open(fits.raw_path, 'update'); image = hdulist[0].data
            bias_col = image[:,2048].copy(); image[:,2048:-1] = image[:, 2049:]; image[:, -1] = bias_col
            hdulist.flush(); hdulist.close()
            fits.add_header_item('BIASCOL', 'T','BIAS column at x=2048 is removed by manager')
            print('  remove bias colume at x=2048 from '+fits.filename)

        if not fits.do_not_use:
            fits.make_reduced_link()
        if fits.grating in self.gratlpmm: #This helps wavelength solution of SAMI. TODO: need to check GRATLPMM for Spector
            try:
                fits.add_header_item('GRATLPMM', self.gratlpmm[fits.grating],'Grating Lines per mm, set by Hector manager')
            except IOError:
                pass
        if fits.grating not in self.idx_files:
            # Without an idx file we would have no way to reduce this file
            self.disable_files([fits])
        if fits.lamp == 'Helium+CuAr+FeAr+CuNe': #Dither script had a wrong arc lamp settings until July 2024 run. 
            self.disable_files([fits])
            print('  diasble this file with a wrong arc lamp: ',fits.lamp)

        self.file_list.append(fits)
        return

    def set_raw_path(self, fits):
        """Set the raw path for a FITS file."""
        if fits.ndf_class == 'BIAS':
            rel_path = os.path.join('bias', fits.ccd, fits.date)
        elif fits.ndf_class == 'DARK':
            rel_path = os.path.join('dark', fits.ccd, fits.exposure_str,
                                    fits.date)
        elif fits.ndf_class == 'LFLAT':
            rel_path = os.path.join('lflat', fits.ccd, fits.date)
        else:
            rel_path = os.path.join(fits.date, fits.ccd)
        fits.raw_dir = os.path.join(self.abs_root, 'raw', rel_path)
        fits.raw_path = os.path.join(fits.raw_dir, fits.filename)
        return

    def update_copy(self, source_path, dest_path):
        """Copy the file, unless a more recent version exists."""
        dest_dir = os.path.dirname(dest_path)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        elif os.path.exists(dest_path):
            if os.path.getmtime(source_path) <= os.path.getmtime(dest_path):
                # File has already been copied and no update to be done
                return
        shutil.copy2(source_path, dest_path)
        return

    def move(self, source_path, dest_path):
        """Move the file."""
        dest_dir = os.path.dirname(dest_path)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        shutil.move(source_path, dest_path)
        return

    def move_reduced_files(self, filename_root, old_reduced_dir, reduced_dir):
        """Move all reduced files connected to the given root."""
        for filename in os.listdir(old_reduced_dir):
            if filename.startswith(filename_root):
                self.move(os.path.join(old_reduced_dir, filename),
                          os.path.join(reduced_dir, filename))
        # If there is nothing useful left in the old directory, delete it.
        if not self.check_reduced_dir_contents(old_reduced_dir):
            # There's nothing useful in the old directory, so move any
            # remaining files to the new directory and then delete it
            for filename in os.listdir(old_reduced_dir):
                self.move(os.path.join(old_reduced_dir, filename),
                          os.path.join(reduced_dir, filename))
            os.removedirs(old_reduced_dir)
        return

    def set_name(self, fits, trust_header=True):
        """Set the object name for a FITS file."""
        fits.name = None
        fits.spectrophotometric = None
        if fits.ndf_class != 'MFOBJECT':
            # Don't try to set a name for calibration files
            return
        # Check if there's already a name in the header
        try:
            name_header = pf.getval(fits.raw_path, 'MNGRNAME')
        except KeyError:
            name_header = None
        try:
            spectrophotometric_header = pf.getval(fits.raw_path, 'MNGRSPMS')
        except KeyError:
            spectrophotometric_header = None
        # Check if the telescope was pointing in the right direction
        fits_coords = ICRS(
            ra=fits.coords['ra'],
            dec=fits.coords['dec'],
            unit=fits.coords['unit'])
        fits_cfg_coords = ICRS(
            ra=fits.cfg_coords['ra'],
            dec=fits.cfg_coords['dec'],
            unit=fits.cfg_coords['unit'])
        if fits_coords.separation(fits_cfg_coords) < self.matching_radius:
            # Yes it was
            name_coords = 'main'
            spectrophotometric_coords = False
        else:
            # No it wasn't
            name_coords = None
            spectrophotometric_coords = None
        # See if it matches any previous fields
        name_extra = None
        spectrophotometric_extra = None
        for extra in self.extra_list:
            if (fits_coords.separation(extra['coords']) < self.matching_radius):
                # Yes it does
                name_extra = extra['name']
                spectrophotometric_extra = extra['spectrophotometric']
                break
        # Now choose the best name
        if name_header and trust_header:
            best_name = name_header
        elif name_coords:
            best_name = name_coords
        elif name_extra:
            best_name = name_extra
        else:
            # As a last resort, ask the user
            best_name = None
            while best_name is None:
                try:
                    best_name = input('Enter object name for file ' +
                                      fits.filename + '\n > ')
                except ValueError as error:
                    print(error)
        # If there are any remaining bad characters (from an earlier version of
        # the manager), just quietly replace them with underscores
        best_name = re.sub(r'[\\\[\]*/?<>|;:&,.$ ]', '_', best_name)
        fits.update_name(best_name)
        # Now choose the best spectrophotometric flag
        if spectrophotometric_header is not None and trust_header:
            fits.update_spectrophotometric(spectrophotometric_header)
        elif spectrophotometric_coords is not None:
            fits.update_spectrophotometric(spectrophotometric_coords)
        elif spectrophotometric_extra is not None:
            fits.update_spectrophotometric(spectrophotometric_extra)
        else:
            # Ask the user whether this is a spectrophotometric standard
            yn = input('Is ' + fits.name + ' in file ' + fits.filename +
                       ' a spectrophotometric standard? (y/n)\n > ')
            spectrophotometric_input = (yn.lower()[0] == 'y')
            fits.update_spectrophotometric(spectrophotometric_input)
        # If the field was new and it's not a "main", add it to the list
        if name_extra is None and name_coords is None:
            self.extra_list.append(
                {'name': fits.name,
                 'coords': fits_coords,
                 'spectrophotometric': fits.spectrophotometric,
                 'fitsfile': fits})
        return

    def update_name(self, file_iterable, name):
        """Change the object name for a set of FITSFile objects."""
        for fits in file_iterable:
            if isinstance(fits, str):
                fits = self.fits_file(fits)
            # Update the name
            try:
                fits.update_name(name)
            except ValueError as error:
                print(error)
                return
            # Update the extra list if necessary
            for extra in self.extra_list:
                if extra['fitsfile'] is fits:
                    extra['name'] = name
            # Update the path for the reduced files
            if fits.do_not_use is False:
                old_reduced_dir = fits.reduced_dir
                self.set_reduced_path(fits)
                if fits.reduced_dir != old_reduced_dir:
                    # The path has changed, so move all the reduced files
                    self.move_reduced_files(fits.filename_root, old_reduced_dir,
                                            fits.reduced_dir)
        return

    def update_spectrophotometric(self, file_iterable, spectrophotometric):
        """Change the spectrophotometric flag for FITSFile objects."""
        for fits in file_iterable:
            if isinstance(fits, str):
                fits = self.fits_file(fits)
            # Update the flag
            fits.update_spectrophotometric(spectrophotometric)
            # Update the extra list if necessary
            for extra in self.extra_list:
                if extra['fitsfile'] is fits:
                    extra['spectrophotometric'] = spectrophotometric
        return

    def set_reduced_path(self, fits):
        """Set the reduced path for a FITS file."""
        if fits.ndf_class == 'BIAS':
            rel_path = os.path.join('bias', fits.ccd, fits.date)
        elif fits.ndf_class == 'DARK':
            rel_path = os.path.join('dark', fits.ccd, fits.exposure_str,
                                    fits.date)
        elif fits.ndf_class == 'LFLAT':
            rel_path = os.path.join('lflat', fits.ccd, fits.date)
        elif fits.ndf_class in ['MFFFF', 'MFARC', 'MFSKY']:
            rel_path = os.path.join(fits.date, fits.plate_id, fits.field_id,
                                    'calibrators', fits.ccd)
        else:
            rel_path = os.path.join(fits.date, fits.plate_id, fits.field_id,
                                    fits.name, fits.ccd)
        fits.reduced_dir = os.path.join(self.abs_root, 'reduced', rel_path)
        fits.reduced_link = os.path.join(fits.reduced_dir, fits.filename)
        fits.reduced_path = os.path.join(fits.reduced_dir,
                                         fits.reduced_filename)
        # set the tlm_path for MFSKY frames that can be used as a TLM:
        if fits.ndf_class == 'MFSKY':
            fits.tlm_path = os.path.join(fits.reduced_dir, fits.tlm_filename)
        if fits.ndf_class == 'MFFFF':
            fits.tlm_path = os.path.join(fits.reduced_dir, fits.tlm_filename)
        elif fits.ndf_class == 'MFOBJECT':
            fits.fluxcal_path = os.path.join(fits.reduced_dir,
                                             fits.fluxcal_filename)
            fits.telluric_path = os.path.join(fits.reduced_dir,
                                              fits.telluric_filename)
        return

    def import_dir(self, source_dir, trust_header=True):
        """Import the contents of a directory and all subdirectories."""
        for dirname, subdirname_list, filename_list in os.walk(source_dir):
            for filename in filename_list:
                if self.file_filter(filename):
                    tmp_path = os.path.join(self.tmp_dir, filename)
                    self.update_copy(os.path.join(dirname, filename),
                                     tmp_path)
                    self.import_file(
                        os.path.join(self.tmp_dir, filename),
                        trust_header=trust_header,
                        copy_files=False, move_files=True)
                    if os.path.exists(tmp_path):
                        # The import was abandoned; delete the temporary copy
                        os.remove(tmp_path)
                if 'Tile' in filename: #Tile is stored in the raw/date/ directory. 
                    src_path = os.path.join(dirname, filename)
                    dest_path = os.path.join(self.abs_root, 'raw', filename)
                    shutil.copy(src_path, dest_path)
                # (SMC 11/06/24) now that we have guider files, also copy these across.  
                # they cannot be imported like regular files as they don't
                # have all the right header keywords.  So we just copy them:
                if 'guide' in dirname:
                    src_path = os.path.join(dirname, filename)
                    # guide folder name and the date to put into the dest_path
                    folders = dirname.split('/')
                    dest_folder = os.path.join(self.abs_root, 'raw',folders[-2],folders[-1])
                    if not os.path.exists(dest_folder):
                        os.mkdir(dest_folder)
                    dest_path = os.path.join(dest_folder,filename)
                    shutil.copy(src_path, dest_path)
        
                    
        if os.path.exists(self.tmp_dir) and len(os.listdir(self.tmp_dir)) == 0:
            os.rmdir(self.tmp_dir)
        return


    def import_aat(self, username=None, password=None, date=None,
                   server='aatlxa', path='/data_liu/aatobs/OptDet_data'):
        """Import from the AAT data disks."""
        current_run_file =  self.abs_root
        start_date, end_date = current_run_file.split('/')[-1].split('_')
        if os.path.exists(path):
            # Assume we are on a machine at the AAT which has direct access to
            # the data directories
            if date is None:
                whole_survey = [s for s in os.listdir(path)
                                if (re.match(r'\d{6}', s) and
                                    os.path.isdir(os.path.join(path, s)))]
                date_options = [x for x in whole_survey for y in range(int(start_date),int(end_date)+1) if str(y) in x]
            else:
                # (SMC 11/06/24 - bug fix) need to put the actual data passed to the function into
                # date options, otherwise this will not work:
                date_options = [date]
            for date in date_options:
                self.import_dir(os.path.join(path, date))
            return
 
        # Otherwise, it is necessary to SCP!
        with self.connection(server=server, username=username,
                             password=password) as srv:
            if srv is None:
                return
            if date is None:
                whole_survey = [s for s in srv.listdir(path)
                                if re.match(r'\d{6}', s)]
                date_options = [x for x in whole_survey for y in range(int(start_date),int(end_date)+1) if str(y) in x]
            else:
                date_options = [date]
                
            if not os.path.exists(self.tmp_dir):
                os.makedirs(self.tmp_dir)
            for date in date_options:
                for ccd in ['ccd_1', 'ccd_2','ccd_3', 'ccd_4']:
                    dirname = os.path.join(path, date, ccd)
                    filename_list = sorted(srv.listdir(dirname))
                    for filename in filename_list:
                        if self.file_filter(filename):
                            srv.get(os.path.join(dirname, filename),
                                    localpath=os.path.join(self.tmp_dir, filename))
                            self.import_file(
                                os.path.join(self.tmp_dir, filename),
                                trust_header=False, copy_files=False,
                                move_files=True)
        if os.path.exists(self.tmp_dir) and len(os.listdir(self.tmp_dir)) == 0:
            os.rmdir(self.tmp_dir)
        return


    def fits_file(self, filename, include_linked_managers=False):
        """Return the FITSFile object that corresponds to the given filename."""
        filename_options = [filename, filename + '.fit', filename + '.fits',
                            filename + '.FIT', filename + '.FITS']
        if include_linked_managers:
            # Include files from linked managers too
            file_list = itertools.chain(
                self.file_list,
                *[mngr.file_list for mngr in self.linked_managers])
        else:
            file_list = self.file_list
        for fits in file_list:
            if fits.filename in filename_options:
                return fits
        return None

    def check_reduced_dir_contents(self, reduced_dir):
        """Return True if any FITSFile objects point to reduced_dir."""
        for fits in self.file_list:
            if (fits.do_not_use is False and
                    os.path.samefile(fits.reduced_dir, reduced_dir)):
                # There is still something in this directory
                return True
        # Failed to find anything
        return False

    def disable_files(self, file_iterable=None):
        """Disable (delete links to) files in provided list (or iterable)."""
        if file_iterable is None:
            if os.path.exists(str(self.abs_root)+'/disable.txt'):  #disable files
                id = np.loadtxt(str(self.abs_root)+'/disable.txt',dtype='U'); self.disable_files(id) 
                print('disable files listed in '+str(self.abs_root)+'/disable.txt')
        else:
            if isinstance(file_iterable, str):
                raise ValueError("disable_files must be passed a list of files, e.g., ['07mar10032.fits']")
            for fits in file_iterable:
                print(fits)
                if isinstance(fits, str):
                    fits = self.fits_file(fits)
                fits.update_do_not_use(True)
                # Delete the reduced directory if it's now empty
                try:
                    os.removedirs(fits.reduced_dir)
                except OSError:
                    # It wasn't empty - no harm done
                    pass
        return

    def enable_files(self, file_iterable):
        """Enable files in provided list (or iterable)."""
        if isinstance(file_iterable, str):
            raise ValueError("enable_files must be passed a list of files, e.g., ['07mar10032.fits']")
        for fits in file_iterable:
            if isinstance(fits, str):
                fits = self.fits_file(fits)
            fits.update_do_not_use(False)
        return

    def link_manager(self, manager):
        """Include data from specified manager when cubing."""
        if manager not in self.linked_managers:
            self.linked_managers.append(manager)
        else:
            print('Already including that manager!')
        return

    def unlink_manager(self, manager):
        """Remove specified manager from list to include when cubing."""
        if manager in self.linked_managers:
            self.linked_managers.remove(manager)
        else:
            print('Manager not in linked list!')
        return

    def bias_combined_filename(self):
        """Return the filename for BIAScombined.fits"""
        return 'BIAScombined.fits'

    def dark_combined_filename(self, exposure_str):
        """Return the filename for DARKcombined.fits"""
        return 'DARKcombined' + exposure_str + '.fits'

    def lflat_combined_filename(self):
        """Return the filename for LFLATcombined.fits"""
        return 'LFLATcombined.fits'

    def bias_combined_path(self, ccd):
        """Return the path for BIAScombined.fits"""
        return os.path.join(self.abs_root, 'reduced', 'bias', ccd,
                            self.bias_combined_filename())

    def dark_combined_path(self, ccd, exposure_str):
        """Return the path for DARKcombined.fits"""
        return os.path.join(self.abs_root, 'reduced', 'dark', ccd, exposure_str,
                            self.dark_combined_filename(exposure_str))

    def lflat_combined_path(self, ccd):
        """Return the path for LFLATcombined.fits"""
        return os.path.join(self.abs_root, 'reduced', 'lflat', ccd,
                            self.lflat_combined_filename())

    def reduce_calibrator(self, calibrator_type, overwrite=False, check=None,
                          **kwargs):
        """Reduce all biases, darks of lflats."""
        self.check_calibrator_type(calibrator_type)
        file_iterable = self.files(ndf_class=calibrator_type.upper(),
                                   do_not_use=False, **kwargs)
        reduced_files = self.reduce_file_iterable(
            file_iterable, overwrite=overwrite, check=check)
        return reduced_files

    def combine_calibrator(self, calibrator_type, overwrite=False):
        """Produce and link necessary XXXXcombined.fits files."""
        for ccd, exposure_str, filename, path in self.combined_filenames_paths(
                calibrator_type, do_not_use=False):
            if overwrite and os.path.exists(path):
                # Delete the existing file
                os.remove(path)
            if not os.path.exists(path):
                self.run_2dfdr_combine(
                    self.files(ccd=ccd, exposure_str=exposure_str,
                               ndf_class=calibrator_type.upper(),
                               reduced=True, do_not_use=False),
                    path)
                dirname = os.path.dirname(path)
        self.link_calibrator(calibrator_type, overwrite)
        return

    def link_calibrator(self, calibrator_type, overwrite=False):
        """Make necessary symbolic links for XXXXcombined.fits files."""
        if calibrator_type.lower() == 'bias':
            dir_type_list = ['dark', 'lflat', 'calibrators', 'object',
                             'spectrophotometric']
        elif calibrator_type.lower() == 'dark':
            dir_type_list = ['lflat', 'calibrators', 'object',
                             'spectrophotometric']
        elif calibrator_type.lower() == 'lflat':
            dir_type_list = ['calibrators', 'object', 'spectrophotometric']
        for ccd, exposure_str, filename, path in self.combined_filenames_paths(
                calibrator_type, do_not_use=False):
            for dir_type in dir_type_list:
                for link_dir in self.reduced_dirs(dir_type, ccd=ccd,
                                                  do_not_use=False):
                    link_path = os.path.join(link_dir, filename)
                    if overwrite and (os.path.exists(link_path) or
                                      os.path.islink(link_path)):
                        os.remove(link_path)
                    if (not os.path.exists(link_path)) and os.path.exists(path):
                        os.symlink(os.path.relpath(path, link_dir),
                                   link_path)
        return

    def check_calibrator_type(self, calibrator_type):
        """Raise an exception if that's not a real calibrator type."""
        if calibrator_type.lower() not in ['bias', 'dark', 'lflat']:
            raise ValueError(
                'calibrator type must be "bias", "dark" or "lflat"')
        return

    def reduce_bias(self, overwrite=False, **kwargs):
        """Reduce all bias frames."""
        self.reduce_calibrator(
            'bias', overwrite=overwrite, check='BIA', **kwargs)
        self.next_step('reduce_bias', print_message=True)
        return

    def combine_bias(self, overwrite=False): 
        """Produce and link necessary BIAScombined.fits files."""
        self.combine_calibrator('bias', overwrite=overwrite) 
        self.next_step('combine_bias', print_message=True)
        return

    def link_bias(self, overwrite=False):
        """Make necessary symbolic links for BIAScombined.fits files."""
        self.link_calibrator('bias', overwrite=overwrite)
        return

    def reduce_dark(self, overwrite=False, **kwargs):
        """Reduce all dark frames."""
        self.reduce_calibrator(
            'dark', overwrite=overwrite, check='DRK', **kwargs)
        self.next_step('reduce_dark', print_message=True)
        return

    def combine_dark(self, overwrite=False):
        """Produce and link necessary DARKcombinedXXXX.fits files."""
        self.combine_calibrator('dark', overwrite=overwrite)
        self.next_step('combine_dark', print_message=True)
        return

    def link_dark(self, overwrite=False):
        """Make necessary symbolic links for DARKcombinedXXXX.fits files."""
        self.link_calibrator('dark', overwrite=overwrite)
        return

    def reduce_lflat(self, overwrite=False, **kwargs):
        """Reduce all lflat frames."""
        self.reduce_calibrator(
            'lflat', overwrite=overwrite, check='LFL', **kwargs)
        self.next_step('reduce_lflat', print_message=True)
        return

    def combine_lflat(self, overwrite=False):
        """Produce and link necessary LFLATcombined.fits files."""
        self.combine_calibrator('lflat', overwrite=overwrite)
        self.next_step('combine_lflat', print_message=True)
        return

    def link_lflat(self, overwrite=False):
        """Make necessary symbolic links for LFLATcombined.fits files."""
        self.link_calibrator('lflat', overwrite=overwrite)
        return

    def make_tlm(self, overwrite=False, leave_reduced=False, this_only=None, **kwargs):
        """Make TLMs from all files matching given criteria.
        If the use_twilight_tlm_blue keyword is set to True in the manager
        (when the manager is initialized), then we will also
        attempt to get a tramline map from twilight frames.  This is done
        by copying them to a different file that has class MFFFF using the
        copy_as function."""

        # check if ccd keyword argument is set, as we need to account for the
        # fact that it is also set for the twilight reductions (ccd1 only).
#        self.disable_files() #do not use diabled file listed in disable.txt
        do_twilight = True
        if ('ccd' in kwargs):
            if ((kwargs['ccd'] == 'ccd_1') or (kwargs['ccd'] == 'ccd_3')):
                do_twilight = True
            else:
                do_twilight = False
            # make a copy of the keywords, but remove the ccd flag for the
            # reduction of twilights, as it is set again in the call below.
            kwargs_copy = dict(kwargs)
            del kwargs_copy['ccd']
        else:
            kwargs_copy = dict(kwargs)

        if (self.use_twilight_tlm_blue and do_twilight):
            fits_twilight_list = []
            print('Processing twilight frames to get TLM')
            # for each twilight frame use the copy_as() function to
            # make a copy with file type MFFFF.  The copied files are
            # placed in the list fits_twilight_list and then can be
            # processed as normal MFFFF files.
            ccdlist = ['ccd_1','ccd_3']
            if this_only:
                filelist = self.files(ndf_class='MFSKY', do_not_use=False, ccd=ccdlist, filename=[this_only], **kwargs_copy)
            else:
                filelist = self.files(ndf_class='MFSKY', do_not_use=False, ccd=ccdlist, **kwargs_copy)
            for fits in filelist:
                fits_twilight_list.append(self.copy_as(fits, 'MFFFF', overwrite=overwrite))
            # use the iterable file reducer to loop over the copied twilight list and
            # reduce them as MFFFF files to make TLMs.
            self.reduce_file_iterable(fits_twilight_list, overwrite=overwrite, tlm=True, leave_reduced=leave_reduced,
                                      check='TLM')

        if self.use_twilight_tlm_all:
            print('Processing twilight frames for *all* ccds to get TLM')
            fits_twilight_list = []
            if this_only:
                filelist = self.files(ndf_class='MFSKY', do_not_use=False, filename=[this_only], **kwargs_copy)
            else:
                filelist = self.files(ndf_class='MFSKY', do_not_use=False, **kwargs_copy)
            for fits in filelist: #marie
                fits_twilight_list.append(self.copy_as(fits, 'MFFFF', overwrite=overwrite))
            self.reduce_file_iterable(fits_twilight_list, overwrite=overwrite, tlm=True, leave_reduced=leave_reduced,
                                      check='TLM')

        # now we will process the normal MFFFF files
        # this currently only allows TLMs to be made from MFFFF files
        file_iterable = self.files(ndf_class='MFFFF', do_not_use=False,
                                   **kwargs)

        if this_only:
            if '.fits' not in this_only:
                this_only = this_only+'.fits'
            file_iterable = self.files(ndf_class='MFFFF', do_not_use=False, filename=[this_only], **kwargs)

        self.reduce_file_iterable(
            file_iterable, overwrite=overwrite, tlm=True,
            leave_reduced=leave_reduced, check='TLM')
        self.next_step('make_tlm', print_message=True)
        return

    def check_tramline(self, check_focus=False, overwrite=False, **kwargs):
        """Automatically detect tramline failure. It will properly work after running make_tlm(),reduce_arc(),and reduce_fflat()."""
        self.disable_files() #do not use diabled file listed in disable.txt
        file_iterable = self.files(ndf_class='MFFFF', do_not_use=False, reduced=True,**kwargs)

        f = open(self.abs_root+'/tlm_failure.txt', 'w') 
        f.write('This is an output of hector.manager.check_tramline() generated on {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())+'\n\n')
        f.write('Automatic detection of tramline failure using the following criteria:\n')
        f.write('  1. failed tlm maps may have active fibres allocated where there is no signal\n')
        f.write('  2. failed tlm maps may have inactive fibres allocated where there is a signal\n')
        f.write('  3. failed tlm maps sometimes show tlms not in numerical order\n')
        f.write('  4. issue on slit mounting may bring tlms lie outside the detector dimensions\n\n')
        f.write('  Find '+hector_path+'observing/check_tramline_example.pdf for examples.\n\n')

        typical_contrast = [5700,10500,5000,15000]
        #f.write('Also, check_tramline(check_focus=True) automatically checks the spectral focus based on the contrast btw the gap and signal at the centre of each flat.\n')
        #f.write('Note that the typical contrast for each ccds are ccd1(60s):'+str(typical_contrast[0])+' ccd2(60s):'+str(typical_contrast[1])+' ccd3(50s):'+str(typical_contrast[2])+' ccd4(25s):'+str(typical_contrast[3])+'\n\n')

        nfail = 0; nfile = 0; nfocus = 0
        for fits in file_iterable:
            nfile = nfile+1
            exfile = pf.open(fits.reduced_path[0:-8]+'ex.fits'); ex = exfile['PRIMARY'].data
            ccd = int(fits.reduced_path[-13])
            
            #examine contrast for checking focusing 
            if check_focus:
                with pf.open(fits.reduced_path[0:-8] + '.fits') as flatfile:
                    flathdr = flatfile['PRIMARY'].header
                    data = flatfile['PRIMARY'].data
#                    y_start, y_end = 1000, -1000
                    x_center = round(flathdr['NAXIS1'] / 2); y_center = round(flathdr['NAXIS2'] / 2)
                    x_start, x_end = x_center - 100, x_center + 100; y_start, y_end = y_center - 200, y_center + 200
                    ystrip = data[y_start:y_end, x_start:x_end]
                midcount = np.median(ystrip)
                maxcount = np.median(ystrip[np.where(ystrip > midcount)]); maxcount = np.median(ystrip[np.where(ystrip > maxcount)])
                contrast = maxcount - np.median(ystrip[np.where((ystrip < midcount) & (ystrip > 700.))])

            fibtab = pf.getdata(fits.reduced_path[0:-8]+'.fits', 'MORE.FIBRES_IFU')
            fib_spec_id = fibtab.field('SPEC_ID'); fib_type = fibtab.field('TYPE'); fib_name = fibtab.field('SPAX_ID')
            thput = np.nanmedian(ex,1); thput = thput/np.nanmedian(thput[np.where((fib_type == 'P') | (fib_type == 'S'))])
            thput_active = 0.08; thput_inactive=thput_active
            if self.speed == 'fast':
                thput_inactive = thput_inactive + 0.1  

            tlmfail = np.zeros(len(fib_spec_id))
            fib_weak = [51,181,202,210] #know weak fibres
            tlmfail[np.where( ((fib_type == 'P') | (fib_type == 'S')) & (thput < thput_active) & (~np.isin(fib_spec_id, fib_weak)) )] = 1 # active (P, S) fibres with no signal
            tlmfail[np.where( ((fib_type == 'N') | (fib_type == 'U')) & (thput > thput_inactive))] = 2 # inactive (N, U) fibres with a signal
            tlmfile = pf.open(fits.reduced_path[0:-8]+'tlm.fits'); tlm = tlmfile['PRIMARY'].data #tlm position
        #    tlm[0,-1] = -10 # ****** only for testing cutoff detection
            tlm_orig_left = (tlm[:,0]); tlm_orig_right= (tlm[:,-1]) # Y position of tlm at the left most and the right most pixel.
            if (fits.ccd == 'ccd_1') | (fits.ccd == 'ccd_2'):
                fib_start = 2; fib_end = 819; ysize=4098-2 # the first and the last active fibres of AAOmega. This info is used for checking tlm cutoff
            else:
                fib_start = 1; fib_end = 855; ysize=4112-2 # Spector
            if(tlm_orig_left[fib_start-1] < 2):
                tlmfail[np.where(fib_spec_id == fib_start)] = 41; cutoff_pos = 'bottom left'; cutoff_fib = str(fib_start) # tlm cutoff detected at the bottom left corner
            if(tlm_orig_right[fib_start-1] < 2):
                tlmfail[np.where(fib_spec_id == fib_start)] = 42; cutoff_pos = 'bottom right'; cutoff_fib = str(fib_start) # bottom right corner
            if(tlm_orig_left[fib_end-1] > ysize-1):
                tlmfail[np.where(fib_spec_id == fib_end)] = 43; cutoff_pos = 'top left'; cutoff_fib = str(fib_end) # top left corner
            if(tlm_orig_right[fib_end-1] > ysize-1):
                tlmfail[np.where(fib_spec_id == fib_end)] = 44; cutoff_pos = 'top right'; cutoff_fib = str(fib_end) # top right corner

            tlm_shift_left  = np.insert(tlm_orig_left[0:-1],0,0) ; tlm_diff_left  = tlm_orig_left  - tlm_shift_left
            tlm_shift_right = np.insert(tlm_orig_right[0:-1],0,0); tlm_diff_right = tlm_orig_right - tlm_shift_right
            tlmfail[np.where((tlmfail < 40.) & (tlm_diff_left < 0)  & (fib_spec_id != 1))] = 31 #tlms not in numerical order in the left end
            tlmfail[np.where((tlmfail < 40.) & (tlm_diff_right < 0) & (fib_spec_id != 1))] = 32 #tlms not in numerical order in the right end

            otherfits = self.other_arm(fits); othername = ''
            if otherfits is not None:
                otherfilename=otherfits.filename #filename of the other arm

            sub = np.where(tlmfail != 0)
            if len(fib_spec_id[sub]) > 0: #tlm failure is detected
                rawfile = pf.open(fits.reduced_path[0:-8]+'.fits'); raw = rawfile['PRIMARY'].data
                f.write('\n========== '+str(fits.filename)+': Failure detected ==========\n')
                prRed(f'\n========== '+str(fits.filename)+': tlm Failure detected ==========\n')
                if len(fib_spec_id[sub]) > 10:
                    f.write('     * Catastrophic failure. Tramline fails for more than 10 fibres. \n')
                else:
                    for i in sub[0]:
                        if tlmfail[i] == 1:
                            f.write('     * Fibre '+str(fib_spec_id[i])+'(Hexa '+fib_name[i]+') is allocated where there is no signal (thput='+str(thput[i])+').\n')
                        if tlmfail[i] == 2:
                            f.write('     * Fibre '+str(fib_spec_id[i])+'('+fib_name[i]+') is an inactive fibre but allocated where there is a signal (thput='+str(thput[i])+').\n')
                        if ((fib_type[i] == 'S') | (fib_type[i] == 'N')) & ((tlmfail[i] == 1) | (tlmfail[i] == 2)) & (str(fib_name[i]) != ''): #check point: wrong sky position
                            f.write('     Check point: This is a sky fibre '+str(fib_name[i])+'. Go to the top-end and check the sky fibre is correctly positioned. \n')
                            f.write('       If the position was wrong, disable the frames adding '+str(fits.filename)[0:-5]+' and '+otherfilename[0:-5]+' on '+str(self.abs_root)+'/disable.txt\n')
                            f.write('       Take new dome flat with corrected sky fibre position and repeat this analysis. \n')
                            f.write('       If it is too late to take dome flat of the plate, do not add the file to the disable list but add a comment by following the command below. \n')
                            f.write('       If you took arc and object frames with a wrong fibre position, add a comment to the frames: \n')
                            f.write('       In the ipython shell    In [1]: mngr.add_comment(["filename"])         e.g. mngr.add_comment(["'+str(fits.filename)+'"])\n')
                            f.write('         Please enter a comment (type n to abort): \n')
                            f.write('         position error: sky fibre '+str(fib_name[i])+' SPEC_ID='+str(fib_spec_id[i])+' TYPE='+str(fib_type[i])+'\n')
                            f.write('       If the sky fibre position was correct, continue to the next check point\n')
                        if (str(tlmfail[i])[0] == '3'): # not in numerical order
                            f.write('     * Fibre '+str(fib_spec_id[i]-1)+' and '+str(fib_spec_id[i])+'(Hexa '+fib_name[i]+') are not in numerical order.\n')
                        if (str(tlmfail[i])[0] == '4'): # tlm cut off 
                            f.write('     * Fibre '+str(fib_spec_id[i])+'(Hexa '+fib_name[i]+') shows cut off of tramline at the '+cutoff_pos+' corner. \n')
                if(np.max(raw) > 65534.): #check point: saturation
                    f.write('     Check point: Max count: '+str(np.max(raw))+'   Number of saturated pixels: '+str(len(raw[np.where(raw > 65534)]))+'\n')
                    f.write('       Max count reaches to the saturation level. Is it saturated or just cosmic ray?\n')
                    f.write('       Visually check the frame: hector@aatlxe:~$ ds9 '+fits.reduced_dir+'/'+str(fits.filename)+' -scale limits 65000 65100 -zoom to fit&\n')
                    f.write('       If satureted, you should find white horizental line(s).\n')
                    f.write('       If saturated, disable '+str(fits.filename)[0:-5]+'\n')
                    f.write('       Take a new dome flat with shorter exposure time. No need to continue to the next check point. \n')
                else:
                    f.write('     Check point: This frame is not saturated. Continue to the next check point.\n')
                f.write('     Check point: check how the other arm ('+otherfilename[0:-5]+') is doing. You may find a solution there. \n')
                f.write('     Check point: visually check tlm:\n') # load drcontrol qui to visually check tlm
                f.write('       In the ipython shell    In [1]: mngr.load_2dfdr_gui("'+fits.reduced_dir+'")\n')
                f.write('       When the window pops up, click the triangle symbol by the filename ('+str(fits.filename)+')\n')
                f.write('       Select '+str(fits.filename)[0:-5]+'tlm.fits, and click 2D Plot button on the middle column.\n')
                f.write('       In the new window, place your cursor to any y-axis value, click and drag down to zoom in\n')
                f.write('       Find the fibres listed above and confirm the failure.\n')
                f.write('       Note that only active fibres (P, S) are supposed to have a signal except Fibre 51. Inactive fibres (N, U) should not have a signal.\n')
                if(np.max(tlmfail) > 40.): #tlm cut-off issue
                    f.write('       Also, visually check the '+cutoff_pos+' corner. The Y position of the tramline of Fibre '+cutoff_fib+' should be between 2 and '+str(ysize)+'. \n')
                    f.write('       If this is not a result of tramline failure, and the cut-off of tramline is confirmed, ask site staff immediately to adjust the mounting of the slit on '+str(fits.instrument)[0:7]+' '+str(fits.ccd)+'. \n')
                    f.write('       They stay until 4 pm. If it is too late, resolve this the next day and add a comment to the '+str(fits.ccd)+' frames of the day: \n')
                    f.write('       In the ipython shell    In [1]: mngr.add_comment(["filename"])         e.g. mngr.add_comment(["'+str(fits.filename)+'"])\n')
                    f.write('         Please enter a comment (type n to abort): \n')

                    f.write('       Once the failure is confirmed, you immediately report this to Sree Oh (sree.oh@anu.edu.au), Madusha Gunawardhana (madusha.gunawardhana@sydney.edu.au), and Scott Croom (scott.croom@sydney.edu.au) and also send the raw file ('+str(fits.filename)+') and this text file (flm_failure.txt).\n')
                    f.write('       It may require a modification of 2dfdr, which takes some time. You may take another dome flat with different exposure time and telescope pointing. If failed again, try a flap flat.\n')
                nfail = nfail + 1
            else:
                f.write('\n'+str(fits.filename)+'  checked. No tlm failure found.')
                print(str(fits.filename)+'  checked. No tlm failure found.')

            if check_focus:
                if contrast < typical_contrast[ccd-1]:
                    nfocus = 1
                    prYellow(f'  ** Check the focus!!! Expected contrast is above '+str(typical_contrast[ccd-1])+', and the contrast of this frame is '+str(contrast))
                    f.write('\n  ** Check the focus of this frame!!! Expected contrast btw the gap and signal is '+str(typical_contrast[ccd-1])+', and the contrast of this frame is '+str(contrast)+'\n')
                    f.write('       If the contrast is way too below the expected one, consider refocusing. \n')
                    f.write('       Visually check the focus: hector@aatlxe:~$ ds9 '+fits.reduced_dir+'/'+str(fits.filename)+' -zoom 4 -zscale&\n')
                    f.write('       If it is out of focus, disable it and the file from the other arm adding '+str(fits.filename)[0:-5]+' and '+otherfilename[0:-5]+' on '+str(self.abs_root)+'/disable.txt\n')
                    f.write('       You need to add all other (arc, object) frames that are affected. Do focussing again.\n')
                else:
                    f.write(' It also shows a reasonable focus (contrast='+str(contrast)+').')
#                    print(' It also shows a reasonable focus (contrast='+str(contrast)+').\n')

        print('\nNote that this task properly works after running mngr.make_tlm(), mngr.reduce_arc(), and mngr.reduce_fflat()')
        if nfail > 0:
            f.write('\n=======\nUnfortunately tramline failures are detected. Follow the steps.\n')
            f.write('Once you go through all the check points, you run mngr.check_tramline() again. \n')
            print('\nUnfortunately tramline failures are detected. \nOpen '+str(self.abs_root)+'/tlm_failure.txt and follow the steps.\n')
        elif nfile == 0:
            f.write('\nNo tlm files found. Do mngr.make_tlm();mngr.reduce_arc();mngr.reduce_fflat() \n')
            print('\nNo tlm files found. Do mngr.make_tlm();mngr.reduce_arc();mngr.reduce_fflat()')
        else:
            print('\nNo tramline failures are detected.\nFind list of frames checked: '+str(self.abs_root)+'/tlm_failure.txt\n')
            f.write('\n=======\nNo tramline failures are detected.\n')

        if nfocus > 0:
            print('You may need to check focus of frame(s). Find the details from tlm_failure.txt\n')
            f.write('\n=======\nSome issue on focus are detected. Fine the details above.\n')
        f.close()
        #self.next_step('make_tlm', print_message=True)
        return

    def reduce_arc(self, overwrite=False, check_focus=None, this_only=None,  **kwargs):
        """Reduce all arc frames matching given criteria.
        check_focus is for checking focus estimating FWHM of arc lines.
        Give the name of an arc frame to check their FWHM. It will check for all ccds for the given frame.
        e.g. check_focus='06mar10003'
        """

        file_iterable = list(self.files(ndf_class='MFARC', do_not_use=False,
                                   **kwargs))
        if this_only:
            if '.fits' not in this_only:
                this_only = this_only+'.fits'
            file_iterable = self.files(ndf_class='MFARC', do_not_use=False, filename=[this_only], **kwargs)

        if check_focus: #make it reduces necessary frames only
            if '.fits' in check_focus:
                check_focus = check_focus[0:10]
            date=check_focus[0:5];frame=check_focus[6:10]+'.fits'
            frames = [date+'1'+frame, date+'2'+frame, date+'3'+frame, date+'4'+frame]
            file_iterable = list(self.files(ndf_class='MFARC', do_not_use=False, filename=frames,**kwargs))
            frames_reduced = [x.reduced_path for x in file_iterable]

        reduced_files = self.reduce_file_iterable(
            file_iterable, overwrite=overwrite, check='ARC')
        for fits in reduced_files:
            bad_fibres(fits.reduced_path, save=True)

        # Run Sam's 2D arc modelling for better wavelength solutions
        if self.fit_arc_model_2d:
            input_list=[]
            if len(reduced_files) == 0:
                reduced_list=[]
                for fits in file_iterable:
                    tdfdr_options = tuple(self.tdfdr_options(fits))
                    reduced_list.append((fits, self.idx_files[fits.grating], tdfdr_options, self.dummy, self.abs_root))
                reduced_files = [item[0] for item in reduced_list]

            for fits in reduced_files:
                tdfdr_options = tuple(self.tdfdr_options(fits,verbose=False))
                arc_reduced = fits.reduced_path
                arcfit_name = os.path.join(fits.reduced_dir,os.path.basename(fits.filename)[0:10]+'_outdir/arcfits.dat')
                tlm_name = os.path.join(fits.reduced_dir,tdfdr_options[-1])
                global_fit = True
                N_x = 4 if fits.instrument == 'AAOMEGA-HECTOR' else 6
                N_y = 2 
                if global_fit:
                    N_y = N_x
                with pf.open(arc_reduced, mode='readonly') as hdul:
                    applied = any(hdu.name == 'OLDWAVELA'  for hdu in hdul) #to avoid applying it multiple times
                if not applied and (fits.lamp[0:16] == 'Helium+CuAr+FeAr'): #only correct wavelength solutions with the right lamp
                    input_list.append((arc_reduced,arcfit_name,tlm_name,N_x,N_y,global_fit))
            print(input_list)
            if input_list:
                self.map(fit_arc_model_wrapper, input_list)
            else:
                print(' Empty list for arc modelling. Nothing to do.')

        #This will measure FWHM of specified frames and plot them over time
        if check_focus:
            outdir = os.path.join(self.abs_root,'qc_plots/')
            if not (os.path.isdir(outdir)):
                os.makedirs(outdir)
            print('Calculating FWHM for the following frames:')
            print(frames_reduced)
            calculate_fwhm(frames_reduced,outdir)
            print('Check the results from ', outdir)

        self.next_step('reduce_arc', print_message=True)
        return

    def reduce_fflat(self, overwrite=False, twilight_only=False, this_only=None, twilight_qc=True, **kwargs):
        """Reduce all fibre flat frames matching given criteria."""

        # check if ccd keyword argument is set, as we need to account for the
        # fact that it is also set for the twilight reductions (ccd1 only).
        if this_only:
            if '.fits' not in this_only:
                this_only = this_only+'.fits'

        do_twilight = True
        if ('ccd' in kwargs):
            if ((kwargs['ccd'] == 'ccd_1') or (kwargs['ccd'] == 'ccd_3')): #marie: I think this should modify it to apply to ccd_3
                do_twilight = True
            else:
                do_twilight = False

            # make a copy of the keywords, but remove the ccd flag for the
            # reduction of twilights, as it is set again in the call below.
            kwargs_copy = dict(kwargs)
            del kwargs_copy['ccd']
        else:
            kwargs_copy = dict(kwargs)

        if ((self.use_twilight_flat_blue and do_twilight) or self.use_twilight_flat_all):
            if(self.use_twilight_flat_blue and do_twilight):
                ccdlist = ['ccd_1','ccd_3']
                print('Processing twilight frames for ccd1 and ccd3 to get fibre flat field')
            if(self.use_twilight_flat_all):
                ccdlist = ['ccd_1','ccd_2','ccd_3','ccd_4']
                print('Processing twilight frames for *all* ccds to get fibre flat field for testing')

            # The twilights should already have been copied as MFFFF
            # at the make_tlm stage, but we can do this again here without
            # any penalty (easier to sue the same code).  So for   
            # each twilight frame use the copy_as() function to
            # make a copy with file type MFFFF.  The copied files are
            # placed in the list fits_twilight_list and then can be
            # processed as normal MFFFF files.
            fits_twilight_list = []
            file_iterable = self.files(ndf_class='MFSKY', do_not_use=False, ccd=ccdlist, **kwargs)
            if this_only:
                this_only = this_only[:6]+'0'+this_only[7:]
                file_iterable = self.files(ndf_class='MFSKY', do_not_use=False, ccd=ccdlist, filename=[this_only], **kwargs)
            for fits in file_iterable:
                #TODO: does it need to be **kwargs or **kwargs_copy??? SAMI has **kwargs here. Why did I change this?
                fits_twilight_list.append(self.copy_as(fits, 'MFFFF', overwrite=overwrite))

            # use the iterable file reducer to loop over the copied twilight list and
            # reduce them as MFFFF files:
            reduced_twilights = self.reduce_file_iterable(fits_twilight_list, overwrite=overwrite, check='FLT')

            for nccd in ccdlist:
                reduced_twilights_ccd = [fits for fits in reduced_twilights if fits.ccd == nccd]
                # Identify bad fibres and replace with an average over all other twilights
                if len(reduced_twilights_ccd) >= 3:
                    path_list = [os.path.join(fits.reduced_dir, fits.filename) for fits in reduced_twilights_ccd]
                    correct_bad_fibres(path_list)

        #Sree: reduce one twilight frame per tile for qc purposes.
        #It will be used with hector.qc.calibration.flat_spectral()
        if twilight_qc and not this_only:
            file_iterable = self.files(ndf_class='MFSKY', do_not_use=False, reduced=False, **kwargs)
            fits_twilight_list = []; indexes = []
            for fits in file_iterable:
                index = fits.ccd+fits.plate_id
                if(index not in indexes):
                    fits_twilight_list.append(self.copy_as(fits, 'MFFFF', overwrite=overwrite))
                indexes.append(index)
            self.reduce_file_iterable(fits_twilight_list, overwrite=overwrite, tlm=True, leave_reduced=False,check='TLM')
            self.reduce_file_iterable(fits_twilight_list, overwrite=overwrite, check='FLT')

        # now we will process the normal MFFFF files
        if (not twilight_only):
            file_iterable = self.files(ndf_class='MFFFF', do_not_use=False,
                                       **kwargs)
            if this_only:
                file_iterable = self.files(ndf_class='MFFFF', do_not_use=False, filename=[this_only], **kwargs)
            self.reduce_file_iterable(
                file_iterable, overwrite=overwrite, check='FLT')

        self.next_step('reduce_fflat', print_message=True)
        return

    def reduce_sky(self, overwrite=False, fake_skies=True, fake_skies_all=False, **kwargs):
        """Reduce all offset sky frames matching given criteria."""
        groups = self.group_files_by(
            ('field_id', 'plate_id', 'date', 'ccd'),
            ndf_class='MFSKY', do_not_use=False, **kwargs)
        file_list = []
        for files in groups.values():
            file_list.extend(files)
        self.reduce_file_iterable(
            file_list, overwrite=overwrite, check='SKY')

        # Average the throughput values in each group
        for files in groups.values():
            path_list = [fits.reduced_path for fits in files]
            make_clipped_thput_files(
                path_list, overwrite=overwrite, edit_all=True, median=True)

        # Send all the sky frames to the improved wavecal routine then
        # apply correction to all the blue arcs
        if  self.improve_blue_wavecorr:
           # ccdlist = ['ccd_1','ccd_3']
            ccdlist = ['ccd_1'] #TODO:Sree: the correction make it worse for Spector. Now it is turned off for Spector but I will revisit this.
            for nccd in ccdlist:
                file_list_tw = []
                for f in file_list:
                    if f.ccd == nccd:
                        file_list_tw.append(f)
                #input_list = zip(file_list_tw,[overwrite]*len(file_list_tw))
                #self.map(wavecorr_frame,input_list)
                if len(file_list_tw) > 0:
                    input_list = list(zip(file_list_tw,[overwrite]*len(file_list_tw)))
                    self.map(wavecorr_frame,input_list)
                    wavecorr_av(file_list_tw,self.root)
            
                kwargs_tmp = kwargs.copy()
                if 'ccd' in kwargs_tmp:
                    del kwargs_tmp['ccd']

                arc_file_iterable = self.files(ndf_class='MFARC', ccd = nccd,
                                        do_not_use=False, **kwargs_tmp)
               
                arc_paths = [fits.reduced_path for fits in arc_file_iterable]
                for arc_path in arc_paths:
                    apply_wavecorr(arc_path,self.root)           
            
        if fake_skies or fake_skies_all:
            if fake_skies:
                no_sky_list = self.fields_without_skies(**kwargs)

            if fake_skies_all:
                #make fake sky frames for *all* dome flats (MFFFF)
                #useful to generate throughput files for all dome flat frames
                keys = ('field_id', 'plate_id', 'date', 'ccd')
                no_sky_list = self.group_files_by(
                    keys, ndf_class='MFFFF', do_not_use=False,
                    lamp='Dome', **kwargs).keys()

            # Certain parameters will already have been set so don't need
            # to be passed (and passing will cause duplicate kwargs)
            for key in ('field_id', 'plate_id', 'date', 'ccd'):
                if key in kwargs:
                    del kwargs[key]
            fits_sky_list = []
            for (field_id, plate_id, date, ccd) in no_sky_list:
                # This field has no MFSKY files, so copy the dome flats
                for fits in self.files(
                        ndf_class='MFFFF', do_not_use=False, lamp='Dome',
                        field_id=field_id, plate_id=plate_id, date=date,
                        ccd=ccd, **kwargs):
                    print('Fake list:',fits)
                    fits_sky_list.append(
                        self.copy_as(fits, 'MFSKY', overwrite=overwrite))
            # Now reduce the fake sky files from all fields
            reduced_file = self.reduce_file_iterable(fits_sky_list, overwrite=overwrite, check='SKY')
            # Rereduce it having many failures due to memory issue when multiprocessing
            rereduced_file = self.reduce_file_iterable(fits_sky_list, overwrite=overwrite, check='SKY')
            reduced_files = reduced_file + rereduced_file

            # Average the throughput values in each group. 
            if self.speed == 'slow':
                for fits in reduced_files:
                    path_list = [fits.reduced_path]
                    make_clipped_thput_files(
                        path_list, overwrite=overwrite, edit_all=True, median=True)

        self.next_step('reduce_sky', print_message=True)
        return

    def fields_without_skies(self, **kwargs):
        """Return a list of fields that have a dome flat but not a sky."""
        keys = ('field_id', 'plate_id', 'date', 'ccd')
        field_id_list_dome = self.group_files_by(
            keys, ndf_class='MFFFF', do_not_use=False,
            lamp='Dome', **kwargs).keys()
        field_id_list_sky = self.group_files_by(
            keys, ndf_class='MFSKY', do_not_use=False,
            **kwargs).keys()
        no_sky = [field for field in field_id_list_dome
                  if field not in field_id_list_sky]
        return no_sky

    def copy_as(self, fits, ndf_class, overwrite=False):
        """Copy a fits file and change its class. Return a new FITSFile."""
        old_num = int(fits.filename[6:10])
        key_num = 7
        if(ndf_class == 'MFFFF'):
            key_num = 8
        if(ndf_class == 'MFSKY'):
            key_num = 9
        new_num = old_num + 1000 * (key_num - (old_num // 1000))
        new_filename = (
                fits.filename[:6] + '{:04d}'.format(int(new_num)) + fits.filename[10:])
        new_path = os.path.join(fits.reduced_dir, new_filename)
        if os.path.exists(new_path) and overwrite:
            os.remove(new_path)
        if not os.path.exists(new_path):
            # Make the actual copy
            shutil.copy2(fits.raw_path, new_path)
            # Open up the file and change its NDF_CLASS
            hdulist = pf.open(new_path, 'update')
            hdulist[0].header['NDFCLASS'] = (ndf_class, 'Data Reduction class name (NDFCLASS)')
            hdulist[0].header['MNGRCOPY'] = (
                True, 'True if this is a copy created by a HECTOR Manager')
            hdulist.flush()
            hdulist.close()

        new_fits = FITSFile(new_path)

        #        print('new_fits:',new_fits
        # Add paths to the new FITSFile instance.
        # Don't use Manager.set_reduced_path because the raw location is
        # unusual
        new_fits.raw_dir = fits.reduced_dir
        new_fits.raw_path = new_path
        new_fits.reduced_dir = fits.reduced_dir
        new_fits.reduced_link = new_path
        new_fits.reduced_path = os.path.join(fits.reduced_dir, new_fits.reduced_filename)
        # as this file has not been imported normally, we need to also set the check_data:
        new_fits.set_check_data()
        # if the new class is MFFFF, then add tlm_path to the FITSfile instance as this
        # is also usually done by set_reduced_path.
        if ndf_class == 'MFFFF':
            new_fits.tlm_path = os.path.join(new_fits.reduced_dir, new_fits.tlm_filename)
            # Do we also set the lamp? Probably not.

        return new_fits

    def copy_path(self, path, ndf_class):
        """Return the path for a copy of the specified file."""
        directory = os.path.dirname(path)
        old_filename = os.path.basename(path)
        old_num = int(old_filename[6:10])
        key_num = 7
        if(ndf_class == 'MFFFF'):
            key_num = 9
        if(ndf_class == 'MFSKY'):
            key_num = 8
        new_num = old_num + 1000 * (key_num - (old_num // 1000))
        new_filename = (
                old_filename[:6] + '{:04d}'.format(int(new_num)) + old_filename[10:])
        new_path = os.path.join(directory, new_filename)
        return new_path

    def reduce_object(self, overwrite=False, recalculate_throughput=True,
                      sky_residual_limit=0.025, this_only=None, check_failure=True, **kwargs):
        """Reduce all object frames matching given criteria."""
        # Reduce long exposures first, to make sure that any required
        # throughput measurements are available
        file_iterable_long = self.files(
            ndf_class='MFOBJECT', do_not_use=False,
            min_exposure=self.min_exposure_for_throughput, **kwargs)
        if this_only:
            if '.fits' not in this_only:
                this_only = this_only+'.fits'
            file_iterable_long = self.files(ndf_class='MFOBJECT', do_not_use=False, 
                    min_exposure=self.min_exposure_for_throughput, filename=[this_only], **kwargs)
        print( '\n Reduction for long exposure frames')
        reduced_files = self.reduce_file_iterable(
            file_iterable_long, overwrite=overwrite, check='OBJ')

        # Sree: double check if there was any failure mostly due to force quit, memory shortage etc.
        print( '\n Checking reduction failures')
        rereduce=[]
        if check_failure:
            file_iterable_check = list(self.files(
                ndf_class='MFOBJECT', do_not_use=False,
                min_exposure=self.min_exposure_for_throughput, reduced=True, **kwargs))
        else:
            file_iterable_check = reduced_files
        rereduce = [fits for fits in file_iterable_check if fits.reduce_options() is None]
        for fits in file_iterable_check:
            hdu = pf.open(fits.reduced_path)
            all_nan = np.all(np.isnan(hdu[0].data))
            if all_nan:
                rereduce.append(fits)
                if os.path.exists(fits.fluxcal_path):
                    os.remove(fits.fluxcal_path)
                if os.path.exists(fits.telluric_path):
                    os.remove(fits.telluric_path)
        if len(rereduce) >= 1:
            print( '\n Re-reduction for failures')
            self.reduce_file_iterable(
            rereduce, overwrite=True, check='OBJ')

        # Check how good the sky subtraction was
        print('\n Reduction for frames with bad sky residuals')
        for fits in reduced_files:
            self.qc_sky(fits)
        if sky_residual_limit is not None:
            # Switch to sky line throughputs if the sky residuals are bad
            fits_list = self.files_with_bad_dome_throughput(
                reduced_files, sky_residual_limit=sky_residual_limit)
            # Only keep them if they actually have a sky line to use 
            # - except for Y15SAR3_P006_12T097 which is problematic due to a v. bright galaxy
            fits_list = [fits for fits in fits_list if (fits.has_sky_lines() and 
                                                        (fits.field_id != 'Y15SAR3_P006_12T097'))]
            self.reduce_file_iterable(
                fits_list, throughput_method='skylines',
                overwrite=True, check='OBJ')
            bad_fields = set([fits.field_id for fits in fits_list])
        else:
            bad_fields = []
        if recalculate_throughput and (self.speed == 'slow'):
            # Average over individual throughputs measured from sky lines
            print('\n Reduction for frames with bad throughputs')
            extra_files = self.correct_bad_throughput(
                overwrite=overwrite, **kwargs)
            for fits in extra_files:
                if fits not in reduced_files:
                    reduced_files.append(fits)
        # Now reduce the short exposures, which might need the long
        # exposure reduced above
        if 'max_exposure' in kwargs:
            upper_limit = (1.*kwargs['max_exposure'] - 
                           np.finfo(1.*kwargs['max_exposure']).epsneg)
            del kwargs['max_exposure']
        else:
            upper_limit = (self.min_exposure_for_throughput -
                       np.finfo(self.min_exposure_for_throughput).epsneg)
        file_iterable_short = self.files(
            ndf_class='MFOBJECT', do_not_use=False,
            max_exposure=upper_limit, **kwargs)
        if this_only:
            file_iterable_short = self.files(
                ndf_class='MFOBJECT', do_not_use=False, max_exposure=upper_limit, filename=[this_only], **kwargs)
        file_iterable_sky_lines = []
        file_iterable_default = []
        for fits in file_iterable_short:
            if fits.field_id in bad_fields:
                file_iterable_sky_lines.append(fits)
            else:
                file_iterable_default.append(fits)
        # Although 'skylines' is requested, these will be throughput calibrated
        # by matching to long exposures, because they will be recognised as
        # short
        print('\n Reduction for short exposure frames')
        print('file_iterable_sky_lines',file_iterable_sky_lines,overwrite)
        print('file_iterable_default',file_iterable_default,overwrite)
        reduced_files.extend(self.reduce_file_iterable(
            file_iterable_sky_lines, overwrite=overwrite,
            throughput_method='skylines', check='OBJ'))
        # These will be throughput calibrated using dome flats
        reduced_files.extend(self.reduce_file_iterable(
            file_iterable_default, overwrite=overwrite, check='OBJ'))
        # Check how good the sky subtraction was
        for fits in reduced_files:
            self.qc_sky(fits)

        self.next_step('reduce_object', print_message=True)
        return


    def reduce_file_iterable(self, file_iterable, throughput_method='default',
                             overwrite=False, tlm=False, leave_reduced=True,
                             check=None):
        """Reduce all files in the iterable."""
        # First establish the 2dfdr options for all files that need reducing
        # Would be more memory-efficient to construct a generator

        input_list = []  # type: List[Tuple[FITSFile, str, Sequence]]
        for fits in file_iterable:
            if (overwrite or
                    not os.path.exists(self.target_path(fits, tlm=tlm))):
                tdfdr_options = tuple(self.tdfdr_options(fits, throughput_method=throughput_method, tlm=tlm))
                input_list.append(
                    (fits, self.idx_files[fits.grating], tdfdr_options, self.dummy, self.abs_root))
        reduced_files = [item[0] for item in input_list]

        # Send the items out for reducing. Keep track of which ones were done.
        while input_list:
            print(len(input_list), 'files remaining.')
            finished = np.array(self.map(
                run_2dfdr_single_wrapper, input_list))

            # Mark finished files as requiring checks
            if check:
                for fin, reduction_tuple in zip(finished, input_list):
                    fits = reduction_tuple[0]
                    if fin:
                        update_checks(check, [fits], False)
                        # Add time stamp to the header
                        fits.add_header_item('REDTIME','{:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now()), ' Time this file is reduced')

                input_list = [item for i, item in enumerate(input_list)
                          if not finished[i]]

        # Delete unwanted reduced files
        for fits in reduced_files:
            if (fits.ndf_class == 'MFFFF' and tlm and not leave_reduced and os.path.exists(fits.reduced_path)):
                os.remove(fits.reduced_path)

        # Create dummy output if pipeline is being run in dummy mode
        if self.dummy:
            create_dummy_output(reduced_files, tlm=tlm, overwrite=overwrite)

        # Sree: Data with error or we could not reduced within timeout of 600s
        tdfail = str(self.abs_root)+'/tdfdr_failure.txt'
        if os.path.exists(tdfail):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ftimeout = open(str(self.abs_root)+'/tdfdr_failure.txt')
                timeout = ftimeout.read()
                if len(timeout) > 0:
                    print('Could not reduce files listed in '+str(self.abs_root)+'/tdfdr_failure.txt')
                ftimeout.close()

        # Return a list of fits objects that were reduced
        return reduced_files

    def target_path(self, fits, tlm=False):
        """Return the path of the file we want 2dfdr to produce."""
        if fits.ndf_class == 'MFFFF' and tlm:
            target = fits.tlm_path
        else:
            target = fits.reduced_path
        return target

    def files_with_bad_dome_throughput(self, fits_list,
                                       sky_residual_limit=0.025):
        """Return list of fields with bad residuals that used dome flats."""
        # Get a list of all throughput files used
        thput_file_list = np.unique([
            fits.reduce_options().get('THPUT_FILENAME', '')
            for fits in fits_list])
        # Keep only the dome flats
        thput_file_list = [
            filename for filename in thput_file_list if
            filename and not self.fits_file(filename) and
            not filename.startswith('thput')]
        file_list = []
        for fits in self.files(
                ndf_class='MFOBJECT', do_not_use=False, reduced=True):
            try:
        #        print(fits.reduced_path) #to check the file has a problematic header. remove this file. 
                residual = pf.getval(fits.reduced_path, 'SKYMNCOF', 'QC')
            except KeyError:
                # The QC measurement hasn't been done
                continue
            file_list.append(
                (fits,
                 fits.reduce_options().get('THPUT_FILENAME', ''),
                 residual))
        bad_files = []
        for thput_file in thput_file_list:
            # Check all files, not just the ones that were just reduced
            matching_files = [
                (fits, sky_residual) for
                (fits, thput_file_match, sky_residual) in file_list
                if thput_file_match == thput_file]
            mean_sky_residual = np.mean([
                sky_residual
                for (fits, sky_residual) in matching_files
                if fits.exposure >= self.min_exposure_for_throughput])
            if mean_sky_residual >= sky_residual_limit:
                bad_files.extend([fits for (fits, _) in matching_files])
        return bad_files

    def correct_bad_throughput(self, overwrite=False, **kwargs):
        """Create thput files with bad values replaced by mean over field."""
        rereduce = []
        # used_flats = self.fields_without_skies(**kwargs)
        for group in self.group_files_by(
                ('date', 'field_id', 'ccd'), ndf_class='MFOBJECT',
                do_not_use=False,
                min_exposure=self.min_exposure_for_throughput, reduced=True,
                **kwargs).values():
            # done = False
            # for (field_id_done, plate_id_done, date_done,
            #         ccd_done) in used_flats:
            #     if (date == date_done and field_id == field_id_done and
            #             ccd == ccd_done):
            #         # This field has been taken care of using dome flats
            #         done = True
            #         break
            # if done:
            #     continue
            # Only keep files that used sky lines for throughput calibration
            group = [fits for fits in group
                     if fits.reduce_options()['TPMETH'] in
                     ('SKYFLUX(MED)', 'SKYFLUX(COR)')]
            if len(group) <= 1:
                # Can't do anything if there's only one file available, or none
                continue
            path_list = [fits.reduced_path for fits in group]
            edited_list = make_clipped_thput_files(
                path_list, overwrite=overwrite, edit_all=True)
            for fits, edited in zip(group, edited_list):
                if edited:
                    rereduce.append(fits)
        reduced_files = self.reduce_file_iterable(
            rereduce, throughput_method='external', overwrite=True, check='OBJ')
        return reduced_files

    def derive_transfer_function(self,
                                 overwrite=False, model_name='ref_centre_alpha_circ_hdr_cvd',
                                 smooth='spline', this_only =None, **kwargs):
        """Derive flux calibration transfer functions and save them."""
        # modified model name to be the version that takes ZD from header values, not
        # fitted.  This is because the fitting is not always robust for ZD.
        # MLPG: modified model_name = ref_centre_alpha_circ_hdr_cvd
        # such that DAR correction is removed/and chromatic variation distortion added
        # CVD model name: ref_centre_alpha_circ_hdr_cvd
        # Original model name: ref_centre_alpha_circ_hdratm
        hector.manager.read_hector_tiles(abs_root=self.abs_root)
        inputs_list = []
        ccdlist = ['ccd_1','ccd_3']
#        ccdlist = ['ccd_1']
        for fits in self.files(ndf_class='MFOBJECT', do_not_use=False,
                               spectrophotometric=True, ccd=ccdlist, **kwargs):
            if not overwrite:
                hdulist = pf.open(fits.reduced_path)
                try:
                    hdu = hdulist['FLUX_CALIBRATION']
                except KeyError:
                    # Hasn't been done yet. Close the file and carry on.
                    hdulist.close()
                else:
                    # Has been done. Close the file and skip to the next one.
                    hdulist.close()
                    continue
            fits_2 = self.other_arm(fits)
            path_pair = (fits.reduced_path, fits_2.reduced_path)
            log.info(path_pair)
            if fits.epoch < 2013.0:
                # SAMI v1 had awful throughput at blue end of blue, need to trim that data
                n_trim = 3 #TODO: how about Hector ccd3?
            else:
                n_trim = 0
            #inputs_list.append({'path_pair': path_pair, 'n_trim': n_trim,
            #                    'model_name': model_name, 'smooth': smooth,
            #                    'speed':self.speed,'tellcorprim':self.telluric_correct_primary})
            if not this_only or fits.reduced_path[-18:-8] == this_only[:10]:
                inputs_list.append({'path_pair': path_pair, 'n_trim': n_trim,
                        'model_name': model_name, 'smooth': smooth,
                        'speed':self.speed,'tellcorprim':self.telluric_correct_primary})

        self.map(derive_transfer_function_pair, inputs_list)
        self.next_step('derive_transfer_function', print_message=True)
        return

    def combine_transfer_function(self, overwrite=False, use_median_TF=False, **kwargs):
        """Combine and save transfer functions from multiple files."""

        # First sort the spectrophotometric files into date/field/CCD/name
        # groups. Grouping by name is not strictly necessary and could be
        # removed, which would cause results from different stars to be
        # combined.
        #groups = self.group_files_by(('date', 'field_id', 'ccd', 'name'),
        #                             ndf_class='MFOBJECT', do_not_use=False,
        #                             spectrophotometric=True, **kwargs)
        # revise grouping of standards, so that we average over all the
        # standard star observations in the run.  Only group based on ccd.
        # (SMC 17/10/2019)
        # Sree(22/Mar/2024): it now automatically calculate median TF 
        # within 1 year window and applies median TF when current TF/median TF > 1.05
        # Or users can specify use_median_TF=True
        # Diagnostic figures are shown derive_TF/median_TF_(basename)_ccdX.fits

        import matplotlib.pyplot as plt

        groups = self.group_files_by(('ccd'),
                                     ndf_class='MFOBJECT', do_not_use=False,
                                     spectrophotometric=True, **kwargs)
        # Now combine the files within each group
        for fits_list in groups.values():
            path_list = [fits.reduced_path for fits in fits_list]
            path_out = os.path.join(os.path.dirname(path_list[0]),
                                    'TRANSFERcombined.fits')
            #if overwrite:
            #    os.remove(path_out)
            if overwrite or not os.path.exists(path_out):
                print('\nCombining files to create', path_out)
                fluxcal2.combine_transfer_functions(path_list, path_out)
                # Run the QC throughput measurement
                print(os.path.exists(path_out))
                if os.path.exists(path_out):
                    print('here')
                    self.qc_throughput_spectrum(path_out)

                # Since we've now changed the transfer functions, mark these as needing checks.
                update_checks('FLX', fits_list, False)

            #Sree (Mar 2024): compare the current TF with median TF over the past year
            #If current TF/median TF > 1.1, we use median TF instead of the current TF from the run
            #TODO: this is an additional task having issues on extracting standard star flux.
            #this step might be skipped, if we can confirm our flux extraction is accurate enough.
            if use_median_TF:
                median_tf, use_median = fluxcal2.calculate_mean_transfer(path_out,self.abs_root)
            else:
                use_median = False
                
            if(('ccd_4' in path_out) and ('220914_220925' in path_out)): #the median TF is dominated by Aug 2023 where thput is ridicularsly high in ccd4 blue end
                use_median = False
            if('230809_230814' in path_out): #thput is ridicularsly high in ccd4 blue end in Aug 2023 run
                use_median = False

            if (use_median and use_median_TF):
                with pf.open(path_out, mode='update') as hdul:
                    #replace the TF with median TF
                    print('Use median TF for ',path_out)
                    hdul[0].data = median_tf
                    hdul[0].header.set('MEDTF',True,'This is replaced by the median TF')
                    hdul.pop() #remove existing THROUGHPUT extension
                    hdul[1].header['ORIGFILE'] = os.path.basename(path_list[0])
                    hdul.writeto(path_out, overwrite=True)
                #update the QC throughput measurement too, so it does not affect thput calculation
                self.qc_throughput_spectrum(path_out)

            # Copy the file into all required directories for each standard stars (e.g. LTT1788)
            paths_with_copies = [os.path.dirname(path_list[0])]
            for path in path_list:
                if os.path.dirname(path) not in paths_with_copies:
                    path_copy = os.path.join(os.path.dirname(path),
                                             'TRANSFERcombined.fits')
                    if overwrite or not os.path.exists(path_copy):
                        print('Copying combined file to', path_copy)
                        shutil.copy2(path_out, path_copy)
                    paths_with_copies.append(os.path.dirname(path_copy))
        self.next_step('combine_transfer_function', print_message=True)
        return

    def flux_calibrate(self, overwrite=False, **kwargs):
        """Apply flux calibration to object frames."""
        for fits in self.files(ndf_class='MFOBJECT', do_not_use=False,
                               spectrophotometric=False, **kwargs):
            fits_spectrophotometric = self.matchmaker(fits, 'fcal')
            if fits_spectrophotometric is None:
                # Try again with less strict criteria
                fits_spectrophotometric = self.matchmaker(fits, 'fcal_loose')
                if fits_spectrophotometric is None:
                    raise MatchException('No matching flux calibrator found ' +
                                         'for ' + fits.filename)
            if overwrite or not os.path.exists(fits.fluxcal_path):
                print('Flux calibrating file:', fits.reduced_path)
                if os.path.exists(fits.fluxcal_path):
                    os.remove(fits.fluxcal_path)
                path_transfer_fn = os.path.join(
                    fits_spectrophotometric.reduced_dir,
                    'TRANSFERcombined.fits')
                fluxcal2.primary_flux_calibrate(
                    fits.reduced_path,
                    fits.fluxcal_path,
                    path_transfer_fn)
        self.next_step('flux_calibrate', print_message=True)
        return

    def telluric_correct(self, overwrite=False, model_name=None, name='main',
                         **kwargs):
        """Apply telluric correction to object frames."""
        # First make the list of file pairs to correct
        # MLPG: adding new model name = ref_centre_alpha_circ_hdr_cvd
        inputs_list = []
        ccdlist = ['ccd_2','ccd_4']
        for nccd in ccdlist:
             for fits_2 in self.files(ndf_class='MFOBJECT', do_not_use=False,
                                      spectrophotometric=False, ccd=nccd,
                                      name=name, **kwargs):
                 fits_1 = self.other_arm(fits_2)
                 if os.path.exists(fits_2.telluric_path) and os.path.exists(fits_1.telluric_path) and not overwrite:
                     # Already been done; skip to the next file
                     continue

                 if (fits_2.epoch < 2013.0) or ((fits_2.epoch > 2021.0) and (fits_2.epoch < 2022.75)):
                     # SAMI v1 had awful throughput at blue end of blue, need to
                     # trim that data.
                     n_trim = 3
                     # Also get telluric shape from primary standard
                     use_PS = True
                     fits_spectrophotometric = self.matchmaker(fits_2, 'fcal')
                     if fits_spectrophotometric is None:
                         # Try again with less strict criteria
                         fits_spectrophotometric = self.matchmaker(
                             fits_2, 'fcal_loose')
                         if fits_spectrophotometric is None:
                             raise MatchException('No matching flux calibrator ' +
                                                  'found for ' + fits_2.filename)
                     PS_spec_file = os.path.join(
                         fits_spectrophotometric.reduced_dir,
                         'TRANSFERcombined.fits')
                     # For September 2012, secondary stars were often not in the
                     # hexabundle at all, so use the theoretical airmass scaling
                     # The same happened for Hector until Sep 2022.
                     if (fits_2.epoch < 2012.75) or ((fits_2.epoch > 2021.0) and (fits_2.epoch < 2022.75)):
                         scale_PS_by_airmass = True
                     else:
                         scale_PS_by_airmass = False
                     # Also constrain the zenith distance in fitting the star
                     if model_name is None:
                         model_name_out = 'ref_centre_alpha_circ_hdr_cvd' # old_model = 'ref_centre_alpha_circ_hdratm'
                     else:
                         model_name_out = model_name
                 else:
                     # These days everything is hunkydory. Haven't checked for Hector though. 
                     n_trim = 0
                     use_PS = False
                     PS_spec_file = None
                     scale_PS_by_airmass = False
                     if model_name is None:
#                         model_name_out = 'ref_centre_alpha_dist_circ_hdratm'
                         # in some case the fit of the model does not do a good job of
                         # getting the zenith distance.  A more reliable fit is
                         # obtained when we instead use the ZD based on the atmosphere
                         # fully, not just the direction:
                         model_name_out = 'ref_centre_alpha_circ_hdr_cvd' # old model_name = 'ref_centre_alpha_circ_hdratm'
                     else:
                         model_name_out = model_name
                 inputs_list.append({
                     'fits_1': fits_1,
                     'fits_2': fits_2,
                     'n_trim': n_trim,
                     'use_PS': use_PS,
                     'scale_PS_by_airmass': scale_PS_by_airmass,
                     'PS_spec_file': PS_spec_file,
                     'model_name': model_name_out,
                     'speed':self.speed})
        # Now send this list to as many cores as we are using
        # Limit this to 10, because of semaphore issues I don't understand
        old_n_cpu = self.n_cpu
        if old_n_cpu > 10:
            self.n_cpu = 10
        done_list = self.map(telluric_correct_pair, inputs_list)

        # Mark files as needing visual checks:
        for item in inputs_list:
            update_checks('TEL', [item["fits_2"]], False)

        self.n_cpu = old_n_cpu
        for inputs in [inputs for inputs, done in
                       zip(inputs_list, done_list) if done]:
            # Copy the FWHM measurement to the QC header
            self.qc_seeing(inputs['fits_1'])
            self.qc_seeing(inputs['fits_2'])
        self.next_step('telluric_correct', print_message=True)
        return

    def _get_missing_stars(self, catalogue=None):
        """Return lists of observed stars missing from the catalogue."""
        name_list = []
        coords_list = []
        for fits_list in self.group_files_by(
                'field_id', ndf_class='MFOBJECT', reduced=True).values():
            fits = fits_list[0]
            path = fits.reduced_path
            try:
                star = identify_secondary_standard(path)
            except ValueError:
                # A frame didn't have a recognised star. Just skip it.
                continue
            if catalogue and star['name'] in catalogue:
                continue
            fibres = pf.getdata(path, 'FIBRES_IFU')
            fibre = fibres[fibres['NAME'] == star['name']][0]
            name_list.append(star['name'])
            coords_list.append((fibre['GRP_MRA'], fibre['GRP_MDEC']))
        return name_list, coords_list

    def get_stellar_photometry(self, refresh=False, automatic=True):
        """Get photometry of stars, with help from the user."""
        if refresh:
            catalogue = None
        else:
            catalogue = read_stellar_mags()

        name_list, coords_list = self._get_missing_stars(catalogue=catalogue)
        new = get_sdss_stellar_mags(name_list, coords_list, catalogue=catalogue, automatic=automatic)
        # Note: with automatic=True, get_sdss_stellar_mags will try to download
        # the data and return it as a string
        if isinstance(new, bool) and not new:
            # No new magnitudes were downloaded
            return
        idx = 1
        path_out = hector_path+'standards/secondary/sdss_stellar_mags_{}.csv'.format(idx)
        while os.path.exists(path_out):
            idx += 1
            path_out = (
                hector_path+'standards/secondary/sdss_stellar_mags_{}.csv'.format(idx))
        if isinstance(new, bool) and new:
            # get_sdss_stellar_mags could not do an automatic retrieval.
            path_in = input('Enter the path to the downloaded file:\n')
            shutil.move(path_in, path_out)
        else:
            with open(path_out, 'w') as f:
                f.write(new)
        return

    def fluxcal_secondary(self, overwrite=False, use_av_tf_sec = True,force=False,verbose=False, minexp=600.0,**kwargs):
        """derive a flux calibration for individual frames based on the secondary std stars.
        This is done with fits to stellar models using ppxf to get the correct stellar model.
        If force=True, we will apply the correction even if the SECCOR keyword is set to 
        True."""
        # MLPG: ccds array is added, and that's assigned to 'ccd' keyword
        # assertion added after "assign_true_mag" call to ensure the standard star is in the catalogue

        # Generate a list of frames to do fit the stellar models to.
        # these are only for ccd_1 (not enough features in the red arm to
        # make it useful).  The fits are done one frame at a time for all
        # object frames and the best templates are written to the header
        # of the frame.  Also write a header keyword to signify that the
        # secondary correction has been done - keyword is SECCOR.
        #os.remove(str(hector.__path__[0])+'/standards/secondary/Hector_tiles/Hector*.csv')

        hector.manager.read_hector_tiles(abs_root=self.abs_root)
        inputs_list = []
        prGreen('Fitting models to star observations')
        ccdlist = ['ccd_1','ccd_3']
        for fits_1 in self.files(ndf_class='MFOBJECT', do_not_use=False,
                                 spectrophotometric=False, ccd=ccdlist,
                                 name='main',**kwargs):
            if ((not overwrite and 'SECCOR' in
                pf.getheader(fits_1.telluric_path)) | 
                (not os.path.exists(fits_1.telluric_path))):
                # Already been done; skip to the next file
                continue
            if fits_1.telluric_path not in inputs_list:
                inputs_list.append(fits_1.telluric_path)
            
        # fit each of the frames indivdually using ppxf, store the results
        # (the best template(s) and possibly weights) in the headers.
        # call to actually do the fitting:
        self.map(fit_sec_template, inputs_list)
 
        # group the data by field and/or std star (probably field).  Average
        # The best templates or weights to determine the best model for the
        # star in each field.
        prGreen('Averaging models to determine best calibration template')
        groups = self.group_files_by(('date', 'field_id', 'ccd'),
                                     ndf_class='MFOBJECT', do_not_use=False,
                                     ccd=ccdlist,name='main',
                                     spectrophotometric=False, **kwargs)

        prGreen('Deriving and applying secondary transfer functions')
        for fits_list in groups.values():
            #fits_1 = self.other_arm(fits_2)
            #inputs_list.append((fits_1.telluric_path, fits_2.telluric_path))
            # get the path list for all the ccd_1 frames in this group:
            path_list = [fits.telluric_path for fits in fits_list if os.path.exists(fits.telluric_path)]
            if self.speed == 'fast': #not to repeat every single frame while reducing them at the telescope
                filtered_path_list = [path for path in path_list if 'CATMAGU' not in pf.getheader(path, extname='FLUX_CALIBRATION')]
                path_list = filtered_path_list
            if len(path_list) == 0:
                continue
            # also get the equivalent list for the ccd_2 frames:
            path_list2 = [self.other_arm(fits).telluric_path for fits in fits_list]
            path_out = os.path.join(os.path.dirname(path_list[0]),
                                    'TRANSFER2combined.fits')
            path_out2 = os.path.join(os.path.dirname(path_list2[0]),
                                    'TRANSFER2combined.fits')
            if overwrite or not os.path.exists(path_out):
                print('combined template weight into', path_out)
                # now actually call the routine to combine the weights:
                fluxcal2.combine_template_weights(path_list, path_out)
   
            # for each frame (red and blue) use the best template (gal extinction corrected)
            # to derive a transfer function.  Write the transfer function to the data frame
            # as a separate extension - FLUX_CALIBRATION2.  Also grouped by field, average
            # the indivdual secondary calibrations to derive one per field.  This may be
            # optional depending on how good invididual fits are.  Write the combined secondary
            # TF to a separate file for each field.
            
            fluxcal2.derive_secondary_tf(path_list,path_list2,path_out,verbose=verbose,minexp=minexp)

            # by group now correct the spectra by applying the TF.  This can be done on a
            # frame by frame basis, or by field.
            for index, path1 in enumerate(path_list):
                path2 = path_list2[index]
                fluxcal2.apply_secondary_tf(path1,path2,path_out,path_out2,use_av_tf_sec=use_av_tf_sec,force=force,minexp=minexp)
                
                # put the actual SDSS/VST mags for the secondary star into the FLUX_CALIBRATION
                # HDU.  This is also done in the scale_frames() function, but as scale_frames()
                # is not used when doing secondary calibration, we do it here instead.
                star = pf.getval(path1, 'STDNAME', 'FLUX_CALIBRATION')
                found = assign_true_mag([path1,path2], star, catalogue=None, hdu='FLUX_CALIBRATION')

                assert found, prRed(f"{star} is not found in the catalogue")

                # also write the SDSS/VST mags to the TRANSFER2combined.fits if this is the first frame,
                # don't need to repeat every time.  For completeness, do this for red and blue arms,
                # although really only need it on one of them.
                if index == 0:
                    stdname = pf.getval(path1,'STDNAME',extname='FLUX_CALIBRATION')
                    hdulist1 = pf.open(path_out, 'update')
                    hdulist2 = pf.open(path_out2, 'update')
                    hdu = 'TEMPLATE_OPT'
                    for band in 'ugriz':
                        starmag = pf.getval(path1,'CATMAG'+ band.upper(),extname='FLUX_CALIBRATION')
                        hdulist1[hdu].header['CATMAG' + band.upper()] = (starmag, band + ' mag from catalogue')
                        hdulist2[hdu].header['CATMAG' + band.upper()] = (starmag, band + ' mag from catalogue')
                    hdulist1[hdu].header['STDNAME'] = (stdname,'Name of standard star')
                    hdulist2[hdu].header['STDNAME'] = (stdname,'Name of standard star')
                    hdulist1.flush()
                    hdulist2.flush()
                    hdulist1.close()
                    hdulist2.close()


                umag = pf.getval(path1,'CATMAGU',extname='FLUX_CALIBRATION')
                gmag = pf.getval(path1,'CATMAGG',extname='FLUX_CALIBRATION')
                rmag = pf.getval(path1,'CATMAGR',extname='FLUX_CALIBRATION')
                imag = pf.getval(path1,'CATMAGI',extname='FLUX_CALIBRATION')
                zmag = pf.getval(path1,'CATMAGZ',extname='FLUX_CALIBRATION')

        # TODO: possibly set some QC stuff here...?

        self.next_step('fluxcal_secondary', print_message=True)
        
        return
        
    
    def scale_frames(self, overwrite=False, apply_scale=False, **kwargs):
        """Scale individual RSS frames to the secondary standard flux.
        If we only want to calculate the scaling but not apply it, then
        use apply_scale=False.  Typically this will be because we have 
        already done the scaling, for example by flux calibrating to the 
        secondary standard stars.  However, it is still useful to derive
        the scaling for QC purposes, so even if apply_scale=False we 
        calculate the value and write to the header.  Note that in this case
        the scaling is based on the secondary flux star in the frame, which 
        is extracted before secondary flux calibration.  Therefore if calculated
        after secondary flux cal, the rescale value gives a good estimate for
        the relative system throughput, but should not be applied to the data
        as it has aleady been rescaled."""
        # First make the list of file pairs to scale
        inputs_list = []
        frames_list = []

        for fits_2 in self.files(ndf_class='MFOBJECT', do_not_use=False,
                                 spectrophotometric=False, ccd=['ccd_2','ccd_4'],
                                 telluric_corrected=True, name='main',
                                 **kwargs):
            if (not overwrite and 'RESCALE' in
                    pf.getheader(fits_2.telluric_path, 'FLUX_CALIBRATION')):
                # Already been done; skip to the next file
                continue
            
            fits_1 = self.other_arm(fits_2)
            path_pair = (fits_1.telluric_path, fits_2.telluric_path)
            # here we make input list an set of iterable lists so that map will
            # work properly on it:
            inputs_list.append({'path_pair': path_pair,'apply_scale': apply_scale})
            frames_list.append(path_pair)

        self.map(scale_frame_pair, inputs_list)
        # Measure the relative atmospheric transmission
        for (path_1, path_2) in frames_list:
            self.qc_throughput_frame(path_1)
            self.qc_throughput_frame(path_2)

        self.next_step('scale_frames', print_message=True)
        return

    def measure_offsets(self, overwrite=False, min_exposure=599.0, name='main',
                        ccd='both', **kwargs):
        """Measure the offsets between dithered observations."""
        if ccd == 'both':
            ccd_measure = ['ccd_2','ccd_4']
            copy_to_other_arm = True
        else:
            ccd_measure = ccd
            copy_to_other_arm = False

        for nccd in ccd_measure:
            print('measure nccd',nccd)
            print(min_exposure, name, nccd)
            print(self.files(ndf_class='MFOBJECT', do_not_use=False,
                reduced=True, min_exposure=min_exposure, name=name, ccd=nccd,
                include_linked_managers=True, **kwargs))
            groups = self.group_files_by(
                'field_id', ndf_class='MFOBJECT', do_not_use=False,
                reduced=True, min_exposure=min_exposure, name=name, ccd=nccd,
                include_linked_managers=True, **kwargs)
            print('done')
            complete_groups = []
            for key, fits_list in groups.items():
                fits_list_other_arm = [self.other_arm(fits, include_linked_managers=True)
                                       for fits in fits_list]
                if overwrite:
                    complete_groups.append(
                        (key, fits_list, copy_to_other_arm, fits_list_other_arm))
                    print('add1',fits_list)
                    for fits in fits_list:
                        hdulist_this_arm = pf.open(best_path(fits),'update')
                        try:
                            del hdulist_this_arm['ALIGNMENT']
                            print(hdulist_this_arm)
                        except KeyError:
                        # Nothing to delete; no worries
                            pass
                        hdulist_this_arm.flush()
                        hdulist_this_arm.close()
                    continue
                for fits in fits_list:
                    # This loop checks each fits file and adds the group to the
                    # complete list if *any* of them are missing the ALIGNMENT HDU
                    try:
                        pf.getheader(best_path(fits), 'ALIGNMENT')
                    except KeyError:
                        # No previous measurement, so we need to do this group
                        complete_groups.append(
                            (key, fits_list, copy_to_other_arm,
                             fits_list_other_arm))
                        print('add2')

                        # Also mark this group as requiring visual checks:
                        update_checks('ALI', fits_list, False)
                        break
            print(complete_groups)
            self.map(measure_offsets_group, complete_groups)

        self.next_step('measure_offsets', print_message=True)
        return

    def cube(self, overwrite=False, min_exposure=1499.0, name='main',
             star_only=False, drop_factor=None, tag='', update_tol=0.02,
             size_of_grid=80, output_pix_size_arcsec=0.5, clip_throughput=False,
             min_transmission=0.333, max_seeing=4.0, min_frames=6, ndither=None,
             tileid=None, objid=None, qc_only=False, **kwargs):
        """Make datacubes from the given RSS files.
        Sree: specify tileid or objid if want to generate cubes only for the subset.
        qc_only=True checks only galaxy IDs intended for cubing.
        """
        groups = self.group_files_by(
            ['field_id', 'ccd'], ndf_class='MFOBJECT', do_not_use=False,
            reduced=True, name=name, include_linked_managers=True, **kwargs)
        # Add in the root path as well, so that cubing puts things in the 
        # right place
        cubed_root = os.path.join(self.root, 'cubed')
        inputs_list = []

        failed_qc_file = os.path.join(self.root, 'failed_qc_fields.txt')
        with open(failed_qc_file, "w+") as infile:
            failed_fields = [line.rstrip() for line in infile]
            passed_obj = []

            for (field_id, ccd), fits_list in groups.items():
                good_fits_list = self.qc_for_cubing(
                    fits_list, min_transmission=min_transmission,
                    max_seeing=max_seeing, min_exposure=min_exposure)
                path_list = [best_path(fits) for fits in good_fits_list]
                if len(path_list) < min_frames:
                    # Not enough good frames to bother making the cubes
                    objects = ''
                    if field_id not in failed_fields:
                        failed_fields.append(field_id)
                elif star_only:
                    objects = [pf.getval(path_list[0], 'STDNAME', 'FLUX_CALIBRATION')]
                elif (tileid is not None) and (field_id != tileid+'_F0'):
                    objects = ''
                else:
                    objects = get_object_names(path_list[0])
                    if field_id in failed_fields:
                        failed_fields.remove(field_id)
                    # qc_passed_objects
                    nccd = 'BL' if ccd in ['ccd_1', 'ccd_3'] else 'RD'
                    ccd_objects = [nccd+str(s) for s in objects]
                    passed_obj.extend(ccd_objects)

                if drop_factor is None:
                    drop_factor = 0.75 # The decision has been made for Hector in Nov 2024 
                #    if fits_list[0].epoch < 2013.0:
                #        # Large pitch of pilot data requires a larger drop size
                #        drop_factor = 0.75
                #    else:
                #        drop_factor = 0.5
                
                gridsize = size_of_grid
                if((ccd == 'ccd_3') or ((ccd == 'ccd_4'))):
                    gridsize = 60
                

                for objname in objects:
                    if objid:
                        if objname == objid:
                            inputs_list.append(
                                (field_id, ccd, path_list, objname, cubed_root, drop_factor,
                                tag, update_tol, gridsize, output_pix_size_arcsec, clip_throughput,
                                ndither, overwrite))
                    else:
                        inputs_list.append(
                            (field_id, ccd, path_list, objname, cubed_root, drop_factor,
                            tag, update_tol, gridsize, output_pix_size_arcsec, clip_throughput,
                            ndither, overwrite))

        with open(failed_qc_file, "w") as outfile:
            failed_fields = [field + '\n' for field in failed_fields]
            outfile.writelines(failed_fields)
        with open(os.path.join(self.root, 'qc_passed_objects_'+str(self.abs_root)[-13:]+'.txt'), "w") as outfile:
            # qc_passed_objects with both blue and red spectra
            stripped_obj = sorted(set([s[2:] for s in passed_obj]))
            filtered_obj = [item for item in stripped_obj if 'BL'+item in passed_obj and 'RD'+item in passed_obj]
            passed_objects = [obj + '\n' for obj in filtered_obj]
            outfile.writelines(passed_objects)

        if qc_only:
            print('qc check '+failed_qc_file+' if there are any failed tile')
            print('pass '+os.path.join(self.root, 'qc_passed_objects_'+str(self.abs_root)[-13:]+'.txt')+' to TS team')
            return

        # Send the cubing tasks off to multiple CPUs
        cubed_list = self.map(cube_object, inputs_list)
        
        # Mark cubes as not checked. Only mark the first file in each input set
        for inputs, cubed in zip(inputs_list, cubed_list):
            if cubed:
                # Select the first fits file from this run (not linked runs)
                path_list = inputs[2]  # From inputs_list above.
                for path in path_list:
                    fits = self.fits_file(os.path.basename(path)[:10])
                    if fits:
                        break
                if fits:
                    update_checks('CUB', [fits], False)


        # Sree: there was a mistake on tile file which swap bundles B & T in the catalogue
        # 901006735001769 should have been allocated to T in the observed tile file
        # TODO: move this to post processing code for data release.. 
        # Sree (July2025): cube orientation is earlier matched with the input information and does it is not correct 
        # I disable all swapped fibres for now not saving them but maybe later this can be revisit.
        # The list of swapped fibre can be found here:
        # https://docs.google.com/spreadsheets/d/1GvQVK3rFZjZzgr-8Z2ViVs9HwJrZGeCYc6wJy7m5UFA/edit?gid=2048352866#gid=2048352866
        #swap_id = ['901006735001769','901006999103632']; swap_bundle = ['T', 'B']
        #if(str(self.abs_root)[-13:] == '230710_230724'):
        #    if(os.path.exists(cubed_root+'/'+swap_id[0]+'/')):
        #        frames0 = glob(cubed_root + '/'+swap_id[0]+'/*')
        #        frames1 = glob(cubed_root + '/'+swap_id[1]+'/*')
#
#                if(pf.getheader(frames0[0])['IFUPROBE'] != 'T'):
#                    shutil.move(frames0[0], cubed_root+'/'+swap_id[1]+'/')
#                    shutil.move(frames0[1], cubed_root+'/'+swap_id[1]+'/')
#                    shutil.move(frames1[0], cubed_root+'/'+swap_id[0]+'/')
#                    shutil.move(frames1[1], cubed_root+'/'+swap_id[0]+'/')
#                    prYellow('swap directories of 901006735001769 and 901006999103632')
#
#                for sid in swap_id: 
#                    for k in glob(cubed_root + '/'+sid+'/*'):
#                        tmp = k.split('/')[-1].split('_'); tmp[0] = sid
#                        newname = cubed_root + '/'+sid+'/'+'_'.join(tmp)
#                        if k != newname:
#                            prYellow('move '+k+' to '+newname)
#                            shutil.move(k,newname)
#
#                frames0 = glob(cubed_root + '/'+swap_id[0]+'/*')
#                frames1 = glob(cubed_root + '/'+swap_id[1]+'/*')
#                if(pf.getheader(frames0[0])['NAME'] != swap_id[0]):
#                    prYellow('modify header of 901006735001769 and 901006999103632')
#                    for i in [0,1]:
#                        hdulist0 = pf.open(frames0[i], 'update', do_not_scale_image_data=True)
#                        hdulist1 = pf.open(frames1[i], 'update', do_not_scale_image_data=True)
#                        crval1  = [hdulist1[0].header['CRVAL1'],hdulist0[0].header['CRVAL1']]
#                        crval2  = [hdulist1[0].header['CRVAL2'],hdulist0[0].header['CRVAL2']]
#                        catara  = [hdulist1[0].header['CATARA'],hdulist0[0].header['CATARA']]
#                        catadec = [hdulist1[0].header['CATADEC'],hdulist0[0].header['CATADEC']]
#                        hdulist0[0].header['CRVAL1']  = crval1[0] ; hdulist1[0].header['CRVAL1']  = crval1[1]
#                        hdulist0[0].header['CRVAL2']  = crval2[0] ; hdulist1[0].header['CRVAL2']  = crval2[1]
#                        hdulist0[0].header['CATARA']  = catara[0] ; hdulist1[0].header['CATARA']  = catara[1]
#                        hdulist0[0].header['CATADEC'] = catadec[0]; hdulist1[0].header['CATADEC'] = catadec[1]
#                        hdulist0[0].header['NAME'] = swap_id[0]; hdulist1[0].header['NAME'] = swap_id[1]
#                        hdulist0.flush();hdulist0.close()
#                        hdulist1.flush();hdulist1.close()


        print('Start resizing the cubes...') #Susie's code resizing cubes
        #Resize the cropped blue and red cubes of the same object to get equal dimension
        #Be careful of cubes of identical objects being produced by different tiles!
        #Here we need to add the suffix of tile while searching for cubes before resizing
        #Sree: I also added codes for triming wavelengths of Spector to prevent wavelength overlaps btw ccd3 and ccd4 

        for k in glob(cubed_root + '/*/'):
            k2 = []
            for k1 in glob(k + '*'):
                name00 = ((k1.split('/')[-1]).split('.')[0]).split('_')[2:]
                if (len(name00) > 0):
                    name0x = name00[0]    
                    for n in range(len(name00))[1:]:
                        name0y = name0x + '_' + name00[n]
                        name0x = name0y
                    k2.append(name0x)
                
            for k3 in np.unique(k2):
                k0 = glob(k + '*' + k3 + '*')
                print(k0)
                axis_b = pf.getheader(k0[0])['NAXIS1']
                axis_r = pf.getheader(k0[1])['NAXIS1']

                #Delete cubes with zero dimension (from inactive hexabundles)
                #print(axis_b, axis_r, pf.getheader(k0[0])['IFUPROBE'], k0)
                if (axis_b == 0) or (axis_r == 0):
                    print('Delete zero dimemsion cube: ',pf.getheader(k0[0])['NAME'],' Tile ',pf.getheader(k0[0])['PLATEID'],' Hexabundle ',pf.getheader(k0[0])['IFUPROBE'])
                    for i in range(len(k0)):
                        os.remove(k0[i])
                
                #Non-zero dimension, so continue the process
                else:
                    #Set the condition to follow the larger values
                    if axis_b > axis_r:
                        axis = axis_b
                    elif axis_b < axis_r:
                        axis = axis_r
                    else:
                        axis = axis_b
                    for i in range(2):
                        cfile = pf.open(k0[i]) #open file
                        length = cfile[0].header['NAXIS1']
                        if (length==axis): #no need to change
                            print(cfile[0].header['NAME'],cfile[0].header['SPECTID'],'Unchanged...')
                            cfile.flush(); cfile.close()
                            pass
                        else:  #insert NaN row to fulfil the data
                            print(cfile[0].header['NAME'],cfile[0].header['SPECTID'],'Resized...')
                            for j0 in range(4):
                                x = cfile[j0].data; header = cfile[j0].header
                                print('Before resizing file',j0,np.shape(x))
                                if j0==3:
                                    n0 = int(3); n1 = int(4)
                                else:
                                    n0 = int(1); n1 = int(2)
                                j1 = 0
                                while j1 < int((axis-length)/2):
                                    x = np.insert(x, (0,(np.shape(x)[n0])), np.nan , axis=n0)
                                    j1 = j1 + 1
                                j2 = 0
                                while j2 < int((axis-length)/2):
                                    x = np.insert(x, (0,(np.shape(x)[n1])), np.nan , axis=n1)
                                    j2 = j2 + 1

                                print('After resizing file',j0,np.shape(x))
                                print('Writing a new .fits file...')

                                if j0==0:
                                    header['EXTNAME'] = ('PRIMARY','extension name')
                                    header['NAXIS1'] = axis
                                    header['NAXIS2'] = axis
                                    header['CRPIX1'] = (header['CRPIX1'] + int((axis-length)/2),'Pixel coordinate of reference point')
                                    header['CRPIX2'] = (header['CRPIX2'] + int((axis-length)/2),'Pixel coordinate of reference point')
                                    p0 = pf.PrimaryHDU(x, header)
                                elif j0==1:
                                    header['EXTNAME'] = ('VARIANCE','extension name')
                                    p1 = pf.ImageHDU(x , header)
                                elif j0==2:
                                    header['EXTNAME'] = ('WEIGHT','extension name')
                                    p2 = pf.ImageHDU(x , header)
                                else:
                                    header['EXTNAME'] = ('COVAR','extension name')
                                    p3 = pf.ImageHDU(x , header)

                            #unchanged throughout the process
                            header4 = cfile[4].header
                            header4['EXTNAME'] = ('QC','extension name')
                            p4 = pf.BinTableHDU(cfile[4].data , header4)

                            #combine HDUList
                            hdul = pf.HDUList([p0,p1,p2,p3,p4])

                            #name of file to be written
                            filename = k0[i]

	                    #Overwrite the files
                            cfile.flush(); cfile.close()
                            hdul.writeto(filename,overwrite=True); hdul.close()

                        #Sree: wavelength cropping 
                        #TODO: remove this if the truncation is made at earlier stage (e.g. 2dfdr idx options)
                        if pf.getheader(k0[i])['INSTRUME'].strip() == 'SPECTOR':
                            crval3 = pf.getheader(k0[i])['CRVAL3']; cdelt3 = pf.getheader(k0[i])['CDELT3']; crpix3 = pf.getheader(k0[i])['CRPIX3']; naxis3 = pf.getheader(k0[i])['NAXIS3']
                            x=np.arange(naxis3)+1; lam=(crval3-crpix3*cdelt3) + x*cdelt3 #wavelength
                            if pf.getheader(k0[i])['SPECTID'].strip() == 'BL':
                                mask = lam > 5810.
                            else:
                                mask = lam < 5770.
                            hdul = pf.open(k0[i], mode='update')
                            new_data = hdul[0].data
                            new_data[mask,:,:] = np.nan
                            hdul[0].data = new_data
                            hdul.close()

        #Delete the directory if there is no item in it
        print('\n')
        print('Identifying empty directories...')
        for k in glob(cubed_root + '/*/'):
            if len(glob(k + '*'))==0:
                print('No cubes for the directory ',k.split('/')[-2])
                print('Delete the directory...')
                os.rmdir(k)  
            else:
                pass

        print('Finish resizing all cubes')

                   
        self.next_step('cube', print_message=True)
        return

    def qc_for_cubing(self, fits_list, min_transmission=0.333, max_seeing=4.0,
                      min_exposure=599.0):
        """Return a list of fits files from the inputs that pass basic QC."""
        good_fits_list = []
        for fits in fits_list:
            # Check that the file pair meets basic QC requirements.
            # We check both files and use the worst case, so that
            # either both are used or neither.
            transmission = np.inf
            seeing = 0.0
            fits_pair = (
                fits, self.other_arm(fits, include_linked_managers=True))
            for fits_test in fits_pair:
                try:
                    transmission = np.minimum(
                        transmission,
                        pf.getval(best_path(fits_test), 'TRANSMIS', 'QC'))
                except KeyError:
                    # Either QC HDU doesn't exist or TRANSMIS isn't there
                    pass
                try:
                    seeing = np.maximum(
                        seeing,
                        pf.getval(best_path(fits_test), 'FWHM', 'QC'))
                except KeyError:
                    # Either QC HDU doesn't exist or FWHM isn't there
                    pass
            if (transmission >= min_transmission
                    and seeing <= max_seeing
                    and fits.exposure >= min_exposure):
                good_fits_list.append(fits)
            #print('qc_for_cubing:',fits, transmission, seeing, fits.exposure)
        return good_fits_list

    def scale_cubes(self, overwrite=False, min_exposure=599.0, name='main',
                    min_transmission=0.333, max_seeing=4.0, tag=None, ndither=None,
                    **kwargs):
        """Scale datacubes based on the stellar g magnitudes."""

        ccdlist = ['ccd_1','ccd_3']
        secondary_standards_path = hector_path + 'standards/secondary/Hector_tiles/Hector_secondary_standards_shortened.csv'
        secondary_standards = pd.read_csv(secondary_standards_path)
        starname_set = set(secondary_standards['ID'])

        for nccd in ccdlist:
             groups = self.group_files_by(
                 'field_id', ccd=nccd, ndf_class='MFOBJECT', do_not_use=False,
                 reduced=True, name=name, include_linked_managers=True, **kwargs)
             input_list = []
             for (field_id,), fits_list in groups.items():
                 table = pf.getdata(fits_list[0].reduced_path, 'FIBRES_IFU')
                 objects = table['NAME'][table['TYPE'] == 'P']
                 objects = np.unique(objects).tolist()
                 objects = [obj.strip() for obj in objects]  # Stripping whitespace from object names
                 for objname in objects:
                     print('Checking ',objname)
                     if objname in starname_set: #Sree: this is new for Hector 
                    #if telluric.is_star(objname): #Sree: this is for SAMI
                         print('   '+objname+' is a star.')   
                         break
                 else:
                     print('No star found in field, skipping: ' + field_id)
                     continue
                 star = objname
                 objects.remove(star)
                 star_path_pair = [
                     self.cubed_path(star, arm, fits_list, field_id,
                                     exists=True, min_exposure=min_exposure,
                                     min_transmission=min_transmission,
                                     max_seeing=max_seeing, tag=tag, ndither=ndither)
                     for arm in ('blue', 'red')]
                 if ((star_path_pair[0] is None) or (star_path_pair[1] is None) or 
                     ('.gz' in star_path_pair[0]) or ('.gz' in star_path_pair[1])):
                     continue
                 if not overwrite:
                     # Need to check if the scaling has already been done
                     try:
                         [pf.getval(path, 'RESCALE') for path in star_path_pair]
                     except KeyError:
                         pass
                     else:
                         continue
                     # Skip scaling for SS failures, TODO: Sree: SECCOR should be presented to the cube header first
                     #seccor_values = [pf.getval(path, 'SECCOR') for path in star_path_pair]
                     #if any(seccor is False for seccor in seccor_values):
                     #    print("Skipping scaling due to issue on secondary standard star", star_path_pair)
                     #    continue
                 object_path_pair_list = [
                     [self.cubed_path(objname, arm, fits_list, field_id,
                                      exists=True, min_exposure=min_exposure,
                                      min_transmission=min_transmission,
                                      max_seeing=max_seeing, tag=tag, gzipped=False)
                      for arm in ('blue', 'red')]
                     for objname in objects]
                 object_path_pair_list = [
                     pair for pair in object_path_pair_list if None not in pair]
                 input_list.append((star_path_pair, object_path_pair_list, star))
             self.map(scale_cubes_field, input_list)
        self.next_step('scale_cubes', print_message=True)
        return

    def bin_cubes(self, overwrite=False, min_exposure=599.0, name='main',
                  min_transmission=0.333, max_seeing=4.0, tag=None, ndither=None, verbose=False, **kwargs):
        """Apply default binning schemes to datacubes."""
        ccdlist = ['ccd_1','ccd_3']
        for nccd in ccdlist:
            path_pair_list = []
            groups = self.group_files_by(
                'field_id', ccd=nccd, ndf_class='MFOBJECT', do_not_use=False,
                reduced=True, name=name, include_linked_managers=True, **kwargs)
            for (field_id,), fits_list in groups.items():
                if verbose:
                    print(fits_list)
                table = pf.getdata(fits_list[0].reduced_path, 'FIBRES_IFU')
                objects = table['NAME'][table['TYPE'] == 'P']
                objects = np.unique(objects).tolist()
                objects = [obj.strip() for obj in objects]  # Strip whitespace from object names
                for objname in objects:
                    if verbose:
                        print(objname,ndither)
                    path_pair = [
                        self.cubed_path(objname, arm, fits_list, field_id,
                                        exists=True, min_exposure=min_exposure,
                                        min_transmission=min_transmission,
                                        max_seeing=max_seeing, tag=tag, ndither=ndither,gzipped=False)
                        for arm in ('blue', 'red')]
                    if verbose:
                        print(path_pair)
                    if path_pair[0] and path_pair[1]:
                        if ('.gz' in path_pair[0]) or ('.gz' in path_pair[1]):
                            skip = True
                            continue
                        skip = False
                        if not overwrite:
                            hdulist_blue = pf.open(path_pair[0])
                            hdulist_red  = pf.open(path_pair[1])
                            check_ext = 'BINNED_VARIANCE_SECTORS' #Sree:now it skips when the last extension is found
                            for hdu_blue, hdu_red in zip(hdulist_blue, hdulist_red): #Sree: check both cubes
                                if verbose:
                                    print(path_pair)
                                    print(hdu_blue.name.startswith(check_ext),hdu_red.name.startswith(check_ext))
                                if hdu_blue.name.startswith(check_ext) and hdu_red.name.startswith(check_ext):
                                    skip = True
                                    break
                        if not skip:
                            path_pair_list.append(path_pair)
            if verbose:
                print(path_pair_list, len(path_pair_list))
            self.map(bin_cubes_pair, path_pair_list)
        self.check_extensions(n=14) 
        self.next_step('bin_cubes', print_message=True)
        return

    def check_extensions(self,cubed_root=None,n=14):    
        '''
        Susie: Check the number of extensions in a cube - V02 (31st May 2023)
        Contact Susie Tuntipong (stun4076@uni.sydney.edu.au) for details.
        cubed_root is the directory of cubed files, i.e. '*/test/cubed/
        n is the expected number of extensions for complete cubes
        n = 14 before 'record_dust' ('DUST' extension has not been produced yet)
        n = 15 after 'record_dust' ('DUST' extension has already been produced)
        '''

        print('\nChecking the number of extension...')
        if cubed_root is None:
            cubed_root = self.abs_root+'/cubed'
        data = glob(cubed_root + '/*/*')
    
        zero_dimension = []
        other_problems = []
        nwrong = 0

        f0 = open(cubed_root[0:-6] + '/missing_cube_extensions.txt', 'w')
        f0.write('# cube  number_of_extension  expected_number_of_extension  plate_id  probe\n')
        for d in data:
            x = pf.open(d)
            #count the number of extensions
            extension = len(np.array(x.info(0),dtype=object))

            #check the condition
            if extension==n:
                #full extensions satified
                #print('Complete extensions')
                f0.write(d.split('/')[-1] +' '+str(extension)+' '+str(n)+' '+pf.getheader(d)['PLATEID']+' '+pf.getheader(d)['IFUPROBE']+'\n')
                pass
            else:
                #incomplete extensions, can be divided into two cases
                nwrong=nwrong+1
                if pf.getheader(d)['NAXIS1']==0:
                    #case 1 - zero dimension
                    print(' '+d.split('/')[-1]+' ** Zero dimensions')
                    f0.write(d.split('/')[-1] +' 0 '+str(n)+' '+pf.getheader(d)['PLATEID']+' '+pf.getheader(d)['IFUPROBE']+'\n')
                else:
                    #case 2 - non-zero dimension, may suffer from other issues
                    print(' '+d.split('/')[-1]+' ** Incomplete dimensions, # of dimensions='+str(extension)+'  Hexa bundle '+pf.getheader(d)['IFUPROBE'])
                    f0.write(d.split('/')[-1] +' '+str(extension)+' '+str(n)+' '+pf.getheader(d)['PLATEID']+' '+pf.getheader(d)['IFUPROBE']+'\n')
        f0.close()

        if(nwrong > 0):
            print('Incomplete cubes are detected. Should check missing_cube_extensions.txt')
        else:
            print('Checking the number of extensions done. No missing extensions.')
        return


    def bin_aperture_spectra(self, overwrite=False, min_exposure=599.0, name='main',
                             min_transmission=0.333, max_seeing=4.0, tag=None, 
                             include_gzipped=False, **kwargs):
        """Create aperture spectra."""
        print('Producing aperture spectra')
        ccdlist = ['ccd_1','ccd_3']
        for nccd in ccdlist:
            path_pair_list = []
            groups = self.group_files_by(
                'field_id', ccd=nccd, ndf_class='MFOBJECT', do_not_use=False,
                reduced=True, name=name, include_linked_managers=True, **kwargs)
            for (field_id,), fits_list in groups.items():
                table = pf.getdata(fits_list[0].reduced_path, 'FIBRES_IFU')
                objects = table['NAME'][table['TYPE'] == 'P']
                objects = np.unique(objects)
                objects = [obj.strip() for obj in objects]
                for objname in objects:
                    path_pair = [
                        self.cubed_path(objname.strip(), arm, fits_list, field_id,
                                        exists=True, min_exposure=min_exposure,
                                        min_transmission=min_transmission,
                                        max_seeing=max_seeing, tag=tag, gzipped=include_gzipped)
                        for arm in ('blue', 'red')]
                    if path_pair[0] and path_pair[1]:
                        if (('.gz' in path_pair[0]) or ('.gz' in path_pair[1])) and (include_gzipped == False):
                            continue
                        path_pair_list.append(path_pair)

            inputs_list = []

            for path_pair in path_pair_list:
                inputs_list.append([path_pair,overwrite])
            self.map(aperture_spectra_pair, inputs_list)
        self.next_step('bin_aperture_spectra', print_message=True)

        return

    def record_dust(self, overwrite=False, min_exposure=599.0, name='main',
                    min_transmission=0.333, max_seeing=4.0, tag=None, **kwargs):
        """Record information about dust in the output datacubes."""
        ccdlist = ['ccd_1','ccd_3']
        for nccd in ccdlist:
            groups = self.group_files_by(
                'field_id', ccd=nccd, ndf_class='MFOBJECT', do_not_use=False,
                reduced=True, name=name, include_linked_managers = True, **kwargs)
            for (field_id,), fits_list in groups.items():
                table = pf.getdata(fits_list[0].reduced_path, 'FIBRES_IFU')
                objects = table['NAME'][table['TYPE'] == 'P']
                objects = np.unique(objects).tolist()
                objects = [obj.strip() for obj in objects]
                for objname in objects:
                    for arm in ('blue', 'red'):
                        path = self.cubed_path(
                            objname, arm, fits_list, field_id,
                            exists=True, min_exposure=min_exposure,
                            min_transmission=min_transmission,
                            max_seeing=max_seeing, tag=tag, gzipped=False)
                        if path:
                            if '.gz' not in path:
                                dust.dustCorrectHectorCube(path, overwrite=overwrite)
        self.check_extensions(n=15)
        self.next_step('record_dust',print_message=True)
        return

    def data_release(self, version=None, date_start='221001', date_finish=None, move=False, moveback=False, check_data=True):
        """
        Sree:this generates release catalogue and move the generated cubes. It also change the name of cubes in a regular format.
        First import hector and manager with one of the any runs in the reduction directory for the current version.
        The format of date_start and date_finish is 'YYMMDD'
        if date_start and date_finish are not specified, it condisers runs between Aug 2022 and today
        if move=False, it generates a catalogue, but doesn't move the cubes to the release directory
        if move=True, it moves all cubes to the release directory.
        if moveback=True, it moves back the cubes to the working directory.
        The catalogue can be generated only when cubes are in working directories. Therefore, should moveback the cubes to working
        directories first, to generate the catalogue again.
        if check_data=True, it skipped cubes with extremly low S/N
        mngr.data_release(version='0_01')
        """

        import string

        if version is None or not bool(re.match(r'^\d{1}_\d{2}$',version)):
            print("\nI expect to receive a version in a format of 'x_xx' where x is a number. Specify the version correctly.")
            print("e.g. mngr.data_release(version='0_01')\n")
            sys.exit(0)
        if date_finish is None: # til now
            now = datetime.datetime.now()
            date_finish = now.strftime('%y%m%d')

        print(f'\nPreparing release cubes and catalogue for version v{version} from {date_start} to {date_finish}')
        dir_version = os.path.join(os.path.dirname(os.path.dirname(self.abs_root)),'v'+version)
        dir_release = dir_version+'/release_v'+version
        list_dir = np.array(os.listdir(dir_version))
        run_list = [x for x in list_dir if bool(re.match(r'^2\d{5}_2\d{5}$',x))]
        runid = sorted([x for x in run_list if int(x[:6]) >= int(date_start) and int(x[7:13]) <= int(date_finish)],key=lambda x: (int(x[:6])))
        print(' Directory for the version: ',dir_version)
        print(' List of runs to be included: ',runid)
        try:
            os.makedirs(dir_release)
        except OSError:
            pass

        if not moveback:
            tempname = os.path.join(dir_version, f'release_v{version}_temp.csv') #list all available cubes
            skipname = os.path.join(dir_release, f'release_skip_v{version}.csv') #list all skipped cubes
            all_data = []; skip_data = []
            if os.path.exists(tempname):
                os.remove(tempname)
            if os.path.exists(skipname):
                os.remove(skipname)
            for i in range(len(runid)):
                print(' Identifying cubes from '+os.path.join(dir_version, runid[i], "cubed"))
                os.system(f'find {os.path.join(dir_version, runid[i], "cubed")} -name "*.fits*" > list.txt')
                with open('list.txt', 'r') as list_file:
                    listcube = list_file.readlines()
                    for j in range(len(listcube)):
                        listcube[j] = listcube[j].strip()
                        with pf.open(listcube[j]) as hdulist:
                            hdr = hdulist[0].header
                            name = hdr['NAME'].strip()
                            infile = listcube[j][listcube[j].rfind(name):]
                            ccd = infile[infile.find('_') + 1]
                            if ccd == 'b':
                                ccd = 'BL'
                            if ccd == 'r':
                                ccd = 'RD'
                            ra   = hdr['CATARA']; dec  = hdr['CATADEC']
                            xcen = hdr['CRPIX1']; ycen = hdr['CRPIX2']
                            texp = hdr['TOTALEXP']; ndither = hdr['NDITHER']
                            inst = hdr['INSTRUME'].strip(); probe = hdr['IFUPROBE'].strip()
                            plate = hdr['PLATEID'].strip()
                            try:
                                label = hdr['LABEL'].strip()
                            except KeyError:
                                label = 1
                            source = 'S'
                            if (name[0] == 'C') or (plate[0] == 'A'):
                                source = 'C'; type = 'G'
                            if (name[0] == 'W') or (plate[0] == 'H') or (plate[0] == 'G') or (plate == 'Commissioning_SAMI_low_mass_T001') or (plate == 'Commissioning_SAMI_MANGA_overlap_T001'):
                                source = 'W'; type = 'G'
                            if (name[0] == 'S') or (source == 'S'):
                                type = 'S'
                            pname = name[:]
                            if pname.isnumeric(): # add a prefix if it is not already included in the id; obj name was numeric and didn't have prefix in early data
                                if (probe == 'H') or (probe == 'U'): #H and U were always for secondary stars at that time when id was numeric; it can be another bundle since 2025 
                                    type = 'S'
                                prefix = 'S'
                                if (source=='C') and (type=='G'):
                                    prefix = 'C'
                                if (source=='W') and (type=='G'):
                                    prefix = 'W'
                                pname = prefix+pname
                            qc_data = hdulist['QC'].data
                            fwhm = np.nanmedian(qc_data['FWHM'])
                            trans = np.nanmedian(qc_data['TRANSMIS'])
                            data = {
                                'name': name,'pname': pname,'ccd': ccd,'runid': runid[i],'infile': infile, 'version': version,
                                'ra': ra,'dec': dec,'xcen': xcen,'ycen': ycen, 'texp': texp,'ndither': ndither,
                                'inst': inst,'probe':probe,'tile':plate,'field':label,'type':type,
                                'fwhm':fwhm,'trans':trans}
                            if check_data:
                                hdu = hdulist[0].data
                                sn = der_snr(np.nanmedian(hdu, axis=(1, 2)))
                                print(' S/N of ',listcube[j],' is ' ,sn)
                                if sn < 0.05: #Sree: S/N < 0.05, this is potentially a broken bundle
                                    print('  skip ',listcube[j])
                                    skip_data.append(data)
                                    continue
                            all_data.append(data)
            df = pd.DataFrame(all_data)
            df.to_csv(tempname, mode='w', header=True, index=False)

            if check_data and len(skip_data)>0:
                df_skip = pd.DataFrame(skip_data)
                df_skip.to_csv(skipname, mode='w', header=True, index=False)

            cols = pd.read_csv(tempname)
            #cols['merge'] = cols.apply(lambda row: '_'.join(row.drop(['ccd', 'infile', 'fwhm', 'trans']).astype(str)), axis=1)
            cols['merge'] = cols.apply(lambda row: '_'.join(row.drop(['ccd', 'infile', 'trans']).astype(str)), axis=1)

            letters = list(string.ascii_uppercase); letter_counter = 0; cols['identifier'] = None

            merge_identifiers = {}  
            for gname, group in cols.groupby('pname'):
                letter_counter = 0
                for index, row in group.iterrows():
                    merge_value = row['merge']
                    if merge_value not in merge_identifiers:
                        identifier = letters[letter_counter % len(letters)]
                        merge_identifiers[merge_value] = identifier
                        letter_counter += 1
                    else:
                        identifier = merge_identifiers[merge_value]
                    cols.loc[index, 'identifier'] = identifier

            print(cols[['name', 'ccd', 'infile', 'identifier']])
            cols['filename'] = cols['pname'].astype(str)+'_'+cols['ccd']+'_'+cols['identifier']+'_v'+cols['version']+'.fits'
            if('.gz' in cols['infile'][0]):
                cols['filename'] = cols['filename']+'.gz'
            data = pd.DataFrame({
                'name': cols['pname'], 'filename': cols['filename'], 'ccd': cols['ccd'], 'identifier': cols['identifier'], 'version': cols['version'], 'infile': cols['infile'],
                'runid': cols['runid'], 'ra': cols['ra'], 'dec': cols['dec'], 'xcen': cols['xcen'], 'ycen': cols['ycen'], 'texp': cols['texp'], 'ndither': cols['ndither'],
                'inst': cols['inst'],'probe': cols['probe'],'tile': cols['tile'], 'field': cols['field'], 'type': cols['type'],
                'fwhm': cols['fwhm'], 'trans': cols['trans']})

            outname = os.path.join(dir_release, f'release_catalogue_v{version}.csv')
            if os.path.exists(outname):
                os.remove(outname)
            if os.path.exists(outname):
                data.to_csv(outname, mode='a', header=False, index=False)
            else:
                data.to_csv(outname, mode='w', header=True, index=False)
            print(' Writing release catalogue to ',os.path.join(dir_release, f'release_v{version}.csv'))

            for index, row in cols.iterrows():
                oldpath = os.path.join(dir_version, row['runid'], "cubed", str(row['name']), row['infile'])
                newpath = os.path.join(dir_release, row['filename'])
                if (os.path.exists(newpath)) or (os.path.exists(newpath+'.gz')):
                        print(newpath+' is already exist. Exit')
                        sys.exit(0)
                #if os.path.isfile(oldpath+'.gz'):
                #    oldpath = oldpath+'.gz'; newpath = newpath+'.gz'
                #print(f"shutil.move('{oldpath}', '{newpath}')")
                if move:
                    shutil.move(oldpath, newpath)
            if move:
                prRed(' Moved cubes to the release directory')
            else:
                prRed(' Cubes have not yet moved. Make move=True to move the cubes to the release directory')

        else:  #return the cube to the reduction directory
            cols = pd.read_csv(os.path.join(dir_release, f'release_catalogue_v{version}.csv'))
            for index, row in cols.iterrows():
                cname = str(row['infile']); cubedir = cname.split('_')[0]
                oldpath = os.path.join(dir_version, row['runid'], "cubed", cubedir, row['infile'])
                newpath = os.path.join(dir_release, row['filename'])
                if (os.path.exists(oldpath)) or (os.path.exists(oldpath+'.gz')):
                    print(oldpath+' is already exist. Exit')
                    sys.exit(0)
                #if os.path.isfile(newpath+'.gz'):
                #    oldpath = oldpath+'.gz'; newpath = newpath+'.gz'

                print(f"shutil.move('{newpath}', '{oldpath}')")
                if not os.path.isdir(os.path.dirname(oldpath)):
                    os.makedirs(os.path.dirname(oldpath)) 
                shutil.move(newpath, oldpath)
            prRed(' Cubes are moved back to original working directories')

    def prepare_new_version(self, version_old=None, version_new=None, move=False, date_start='220801', date_finish=None):
        """
        Sree: this prepares reduction directories when starting a new version of reductions. 
        First import hector and manager with one of the any runs in the reduction directory for the old version.

        Inputs:
        version_old = a version of the current reduction (e.g. 0_01)
        version_new = a version of the new reduction (e.g. 0_02)
        if move=False, it copies raw data to the new directory
        if move=True, it moves raw frames from the old directory to the new directory, to save storage.
        date_start and date_finish specify the starting and end date to be considered. No need to specify in most cases.
        mngr.prepare_new_version(version_old='0_01',version_new='0_02')
        """

        if (version_old is None) or (not bool(re.match(r'^\d{1}_\d{2}$',version_old))):
            print("\nI expect to receive version_old in a format of 'x_xx' where x is a digit number. Specify the version correctly.")
            print("e.g. mngr.data_release(version_old='0_01')\n")
            sys.exit(0)
        if (version_new is None) or (not bool(re.match(r'^\d{1}_\d{2}$',version_new))):
            print("\nI expect to receive version_new in a format of 'x_xx' where x is a digit number. Specify the version correctly.")
            print("e.g. mngr.data_release(version_new='0_01')\n")
            sys.exit(0)
        if date_finish is None: # til now
            now = datetime.datetime.now()
            date_finish = now.strftime('%y%m%d')

        dir_version_old = os.path.join(os.path.dirname(os.path.dirname(self.abs_root)),'v'+version_old)
        dir_version_new = os.path.join(os.path.dirname(os.path.dirname(self.abs_root)),'v'+version_new)
        print(' Old directory: ',dir_version_old)
        print(' New directory: ',dir_version_new)

        #if not os.path.isdir(dir_version_new):
        os.makedirs(dir_version_new,exist_ok=True)

        list_dir = np.array(os.listdir(dir_version_old))
        run_list = [x for x in list_dir if bool(re.match(r'^2\d{5}_2\d{5}$',x))]
        runs = sorted([x for x in run_list if int(x[:6]) >= int(date_start) and int(x[7:13]) <= int(date_finish)],key=lambda x: (int(x[:6])))
        print(' List of runs to be included: ',runs)

        for run in runs:
            #if not os.path.isdir(dir_version_new,run):
            dir_run_old = os.path.join(dir_version_old,run)
            dir_run_new = os.path.join(dir_version_new,run)
            os.makedirs(dir_run_new,exist_ok=True)
            if move:
                os.rename(os.path.join(dir_run_old,'raw'), os.path.join(dir_run_new,'raw'))
            else:
                print(' Copying raw frames from '+dir_run_old+' to '+dir_run_new)
                shutil.copytree(os.path.join(dir_run_old,'raw'), os.path.join(dir_run_new,'raw'),dirs_exist_ok = True)

    def gzip_cubes(self, overwrite=False, min_exposure=599.0, name='main',
                   star_only=False, min_transmission=0.333, max_seeing=4.0,
                   tag=None, **kwargs):
        """Gzip the final datacubes. TODO: we may not need this anymore.
        Sree: It is outdated.
        
        """
        groups = self.group_files_by(
            ['field_id', 'ccd'], ndf_class='MFOBJECT', do_not_use=False,
            reduced=True, name=name, include_linked_managers = True, **kwargs)
        input_list = []
        for (field_id, ccd), fits_list in groups.items():
            if (ccd == 'ccd_1') | (ccd == 'ccd_3'):
                arm = 'blue'
            else:
                arm = 'red'
            if star_only:
                objects = [pf.getval(fits_list[0].fcal_path, 'STDNAME',
                                     'FLUX_CALIBRATION')]
            else:
                table = pf.getdata(fits_list[0].reduced_path, 'FIBRES_IFU')
                objects = table['NAME'][table['TYPE'] == 'P']
                objects = np.unique(objects).tolist()
                objects = [obj.strip() for obj in objects]
            for obj in objects:
                input_path = self.cubed_path(
                    obj, arm, fits_list, field_id,
                    exists=True, min_exposure=min_exposure,
                    min_transmission=min_transmission,
                    max_seeing=max_seeing, tag=tag)
                if input_path:
                    if input_path.endswith('.gz'):
                        # Already gzipped, and no non-gzipped version exists
                        continue
                    output_path = input_path + '.gz'
                    if os.path.exists(output_path) and overwrite:
                        os.remove(output_path)
                    if not os.path.exists(output_path):
                        input_list.append(input_path)
        self.map(gzip_wrapper, input_list)
        self.next_step('gzip_cubes', print_message=True)
        return

    def ungzip_cubes(self, overwrite=False, min_exposure=599.0, name='main',
                   star_only=False, min_transmission=0.333, max_seeing=4.0,
                   tag=None, **kwargs):
        """Gzip the final datacubes."""
        groups = self.group_files_by(
            ['field_id', 'ccd'], ndf_class='MFOBJECT', do_not_use=False,
            reduced=True, name=name, include_linked_managers = True, **kwargs)
        input_list = []
        for (field_id, ccd), fits_list in groups.items():
            if ccd == 'ccd_1':
                arm = 'blue'
            else:
                arm = 'red'
            if star_only:
                objects = [pf.getval(fits_list[0].fcal_path, 'STDNAME',
                                     'FLUX_CALIBRATION')]
            else:
                table = pf.getdata(fits_list[0].reduced_path, 'FIBRES_IFU')
                objects = table['NAME'][table['TYPE'] == 'P']
                objects = np.unique(objects).tolist()
                objects = [obj.strip() for obj in objects]
            for obj in objects:
                input_path = self.cubed_path(
                    obj, arm, fits_list, field_id,
                    exists=True, min_exposure=min_exposure,
                    min_transmission=min_transmission,
                    max_seeing=max_seeing, tag=tag)
                if input_path:
                    if input_path.endswith('.fits'):
                        # Already ungzipped, and no non-gzipped version exists
                        continue
                    output_path = input_path[:-3]
                    if os.path.exists(output_path) and overwrite:
                        os.remove(output_path)
                    if not os.path.exists(output_path):
                        input_list.append(input_path)
        self.map(ungzip_wrapper, input_list)

        return

    def reduce_all(self, start=None, finish=None, overwrite=False, **kwargs):
        """Reduce everything, in order. Don't use unless you're sure."""

        task_list = self.task_list

        # Check for valid inputs:
        if start is None:
            start = task_list[0][0]
        if finish is None:
            finish = task_list[-1][0]

        task_name_list = list(map(lambda x: x[0], task_list))
        if start not in task_name_list:
            raise ValueError("Invalid start step! Must be one of: {}".format(", ".join(task_name_list)))
        if finish not in task_name_list:
            raise ValueError("Invalid finish step! Must be one of: {}".format(", ".join(task_name_list)))

        started = False
        for task, include_kwargs in task_list:
            if not started and task != start:
                # Haven't yet reached the first task to do
                continue
            started = True
            method = getattr(self, task)
            print("Starting reduction step '{}'".format(task))
            if include_kwargs:
                method(overwrite, **kwargs)
            else:
                method(overwrite)
            if task == finish:
                # Do not do any further tasks
                break
        # self.reduce_bias(overwrite, **kwargs)
        # self.combine_bias(overwrite)
        # self.reduce_dark(overwrite, **kwargs)
        # self.combine_dark(overwrite)
        # self.reduce_lflat(overwrite, **kwargs)
        # self.combine_lflat(overwrite)
        # self.make_tlm(overwrite, **kwargs)
        # self.reduce_arc(overwrite, **kwargs)
        # self.reduce_fflat(overwrite, **kwargs)
        # self.reduce_sky(overwrite, **kwargs)
        # self.reduce_object(overwrite, **kwargs)
        # self.derive_transfer_function(overwrite, **kwargs)
        # self.combine_transfer_function(overwrite, **kwargs)
        # self.flux_calibrate(overwrite, **kwargs)
        # self.telluric_correct(overwrite, **kwargs)
        # self.scale_frames(overwrite, **kwargs)
        # self.measure_offsets(overwrite, **kwargs)
        # self.cube(overwrite, **kwargs)
        # self.scale_cubes(overwrite, **kwargs)
        # self.bin_cubes(overwrite, **kwargs)
        return

    def ensure_qc_hdu(self, path, name='QC'):
        """Ensure that the file has a QC HDU."""
        hdulist = pf.open(path, 'update')
        try:
            hdulist[name]
        except KeyError:
            # QC HDU doesn't exist, so make one
            hdu = pf.ImageHDU(name=name)
            hdulist.append(hdu)
            hdulist.flush()
        hdulist.close()
        return

    def qc_seeing(self, fits):
        """Copy the FWHM over the QC header."""
        print(fits.telluric_path)
        self.ensure_qc_hdu(fits.telluric_path)
        hdulist = pf.open(fits.telluric_path, 'update')
        source_header = hdulist['FLUX_CALIBRATION'].header
        header = hdulist['QC'].header
        header['FWHM'] = source_header['FWHM'], source_header.comments['FWHM']
        hdulist.flush()
        hdulist.close()

    def qc_sky(self, fits):
        """Run QC check on sky subtraction accuracy and save results."""
        results = sky_residuals(fits.reduced_path)
        for key, value in results.items():
            if not np.isfinite(value):
                results[key] = -9999
        self.ensure_qc_hdu(fits.reduced_path)
        hdulist = pf.open(fits.reduced_path, 'update')
        header = hdulist['QC'].header
        header['SKYMDCOF'] = (
            results['med_frac_skyres_cont'],
            'Median continuum fractional sky residual')
        header['SKYMDLIF'] = (
            results['med_frac_skyres_line'],
            'Median line fractional sky residual')
        header['SKYMDCOA'] = (
            results['med_skyflux_cont'],
            'Median continuum absolute sky residual')
        header['SKYMDLIA'] = (
            results['med_skyflux_line'],
            'Median line absolute sky residual')
        header['SKYMNCOF'] = (
            results['mean_frac_skyres_cont'],
            'Mean continuum fractional sky residual')
        header['SKYMNLIF'] = (
            results['mean_frac_skyres_line'],
            'Mean line fractional sky residual')
        header['SKYMNCOA'] = (
            results['mean_skyflux_cont'],
            'Mean continuum absolute sky residual')
        header['SKYMNLIA'] = (
            results['mean_skyflux_line'],
            'Mean line absolute sky residual')
        hdulist.flush()
        hdulist.close()
        return

    def qc_throughput_spectrum(self, path):
        """Save the throughput function for a TRANSFERcombined file."""
        absolute_throughput, data, probename, stdname = throughput(path)
        # Check the CCD and date for this file
        file_input = pf.getval(path, 'ORIGFILE', 1)
        #print(path, file_input)
        path_input = os.path.join(
            self.fits_file(file_input[:10]).reduced_dir, file_input)
        detector = pf.getval(path_input, 'DETECTOR')
        epoch = pf.getval(path_input, 'EPOCH')
        # Load mean throughput function for that CCD
        path_list = (glob(hector_path+'standards/throughput/mean_throughput_' +
                          detector + '.fits') +
                     glob(hector_path+'standards/throughput/mean_throughput_' +
                          detector + '_*.fits'))
        for path_mean in path_list:
            hdulist_mean = pf.open(path_mean)
            header = hdulist_mean[0].header
            if (('DATESTRT' not in header or
                 epoch >= header['DATESTRT']) and
                    ('DATEFNSH' not in header or
                     epoch <= header['DATEFNSH'])):
                # This file is suitable for use
                found_mean = True
                mean_throughput = hdulist_mean[0].data
                hdulist_mean.close()
                print(path_mean+' has been used')
                break
            hdulist_mean.close()
        else:
            prRed('  Warning: No valid mean throughput file found for QC checks.')
            print('     epoch of '+file_input+': ',epoch)
            print('     go to '+hector_path+'standards/throughput/')
            print('     check DATESTRT, DATEFNSH of the current mean throughput frame')
            print('     if necessary change DATESTRT, DATEFNSH or generate the new mean througput (readme.txt)')
            found_mean = False
        if found_mean:
            relative_throughput = absolute_throughput / mean_throughput
            data = np.vstack((absolute_throughput, relative_throughput))
            median_relative_throughput = np.nanmedian(relative_throughput)
            if not np.isfinite(median_relative_throughput):
                median_relative_throughput = -1.0
        else:
            data = absolute_throughput
        hdulist = pf.open(path, 'update')
        hdulist.append(pf.ImageHDU(data, name='THROUGHPUT'))
        if found_mean:
            hdulist['THROUGHPUT'].header['MEDRELTH'] = (
                median_relative_throughput, 'Median relative throughput')
            hdulist['THROUGHPUT'].header['PATHMEAN'] = (
                path_mean, 'File used to define mean throughput')
        hdulist.flush()
        hdulist.close()
        return

    def qc_throughput_frame(self, path):
        """Calculate and save the relative throughput for an object frame."""

        print('qc_throughput_frame',path,(pf.getval(path, 'FCALFILE')))
        try:
            median_relative_throughput = (
                pf.getval(pf.getval(path, 'FCALFILE'),
                          'MEDRELTH', 'THROUGHPUT'))
            print('median_relative_throughput1',median_relative_throughput)

        except KeyError:
            # Not all the data is available
            print("Warning: 'combine_transfer_function' required to calculate transmission.")
            return
        try:
            median_relative_throughput /= (
                pf.getval(path, 'RESCALE', 'FLUX_CALIBRATION'))
            print('median_relative_throughput2',median_relative_throughput, pf.getval(path, 'RESCALE', 'FLUX_CALIBRATION'))

        except KeyError:
            # Not all the data is available
            print('Warning: Flux calibration required to calculate transmission.')
            return

        # Sree (June 2025): AAOmega transmission is currently unreliable due to extremely low throughput in bundle H.
        # For now, we set the transmission to the higher value between AAOmega and Spector.
        # TODO: Revisit this once bundle H has been properly fixed.

        if pf.getval(path, 'EPOCH') > 2024.84  and pf.getval(path, 'INSTRUME') == 'AAOMEGA-HECTOR':
            if 'ccd_1' in path:
                path_spector = path.replace('ccd_1', 'ccd_3')[:-13]+'3'+path[-12:]
            elif 'ccd_2' in path:
                path_spector = path.replace('ccd_2', 'ccd_4')[:-13]+'4'+path[-12:]
            try:
                median_relative_throughput_spector = (
                    pf.getval(pf.getval(path_spector, 'FCALFILE'),'MEDRELTH', 'THROUGHPUT'))
            except KeyError:
                median_relative_throughput_spector = -1.
            try:
                median_relative_throughput_spector /= (
                    pf.getval(path_spector, 'RESCALE', 'FLUX_CALIBRATION'))
            except KeyError:
                pass
            median_relative_throughput = max(median_relative_throughput, median_relative_throughput_spector)

        if not np.isfinite(median_relative_throughput):
            median_relative_throughput = -1.0
        self.ensure_qc_hdu(path)
        hdulist = pf.open(path, 'update')
        header = hdulist['QC'].header
        header['TRANSMIS'] = (
            median_relative_throughput, 'Relative transmission')
        hdulist.flush()
        hdulist.close()
        return

    def qc_summary(self, min_exposure=599.0, ccd='ccd_2', **kwargs):
        """Print a summary of the QC information available."""
        text = 'QC summary table\n'
        text += '=' * 75 + '\n'
        text += 'Use space bar and cursor keys to move up and down; q to quit.\n'
        text += 'Disabled ccds are marked with T, and usable ccds with F.\n'
        text += '   If any of ccds are disabled, it is marked with a *\n'
        text += '   E.g., when ccd_1 and ccd_4 are disabled, it will be shown as TFFT\n'
        text += 'FWHM, Transmission, Sky_residual, Sky_brightness are shown for both AAOmega and Spector.\n'
        text += '   The first values are estimates from AAOmega, while the values after / are from Spector.\n'
        text += '\n'

        text += 'This table is saved to '+self.abs_root+'/qc_summary_'+self.abs_root[-13:]+'.txt\n'
        text += '\n'

        #
        #  Summarize shared calibrations
        #
        text += "Summary of shared calibrations\n"
        text += "-" * 75 + "\n"

        # Get grouped lists and restructure to be a dict of dicts.
        by_ndf_class = defaultdict(dict)
        for k, v in self.group_files_by(["ndf_class", "date", "ccd"]).items():
            if k[2] == ccd:
                by_ndf_class[k[0]].update({k[1]: v})

        # Print info about basic cals
        for cal_type in ("BIAS", "DARK", "LFLAT"):
            if cal_type in by_ndf_class:
                text += "{} Frames:\n".format(cal_type)
                total_cals = 0
                for date in sorted(by_ndf_class[cal_type].keys()):
                    n_cals = len(by_ndf_class[cal_type][date])
                    text += "  {}: {} frames\n".format(date, n_cals)
                    total_cals += n_cals
                text += "  TOTAL {}s: {} frames\n".format(cal_type, total_cals)

        # Gather info about flux standards
        text += "Flux standards\n"
        flux_standards = defaultdict(dict)
        for k, v in self.group_files_by(["date", "name", "spectrophotometric", "ccd"]).items():
            if k[3] == ccd and k[2]:
                flux_standards[k[0]].update({k[1]: v})

        # Print info about flux standards
        total_all_stds = 0
        for date in flux_standards:
            text += "  {}:\n".format(date)
            total_cals = 0
            for std_name in sorted(flux_standards[date].keys()):
                n_cals = len(flux_standards[date][std_name])
                text += "    {}: {} frames\n".format(std_name, n_cals)
                total_cals += n_cals
            text += "    Total: {} frames\n".format(total_cals)
            total_all_stds += total_cals
        text += "  TOTAL Flux Standards: {} frames\n".format(total_all_stds)
        text += "\n"

        # Summarize field observations. Start with AAOmega
        for (field_id,), fits_list in self.group_files_by(
                'field_id', ndf_class='MFOBJECT', min_exposure=min_exposure,
                ccd=ccd, **kwargs).items():
            text += '+' * 75 + '\n'
            text += field_id + '\n'
            text += '-' * 75 + '\n'
            text += 'File       Disabled   Exposure     FWHM(")        Transmission      Sky_residual     Sky_brightness\n'
            for fits in sorted(fits_list, key=lambda f: f.filename):
                fwhm = '       -'
                transmission = sky_residual = mean_sky_brightness = '              -'
                if fits is not None:
                    try:
                        header = pf.getheader(best_path(fits), 'QC')
                    except (IOError, KeyError):
                        pass
                    else:
                        if 'FWHM' in header:
                            fwhm = '{:8.2f}'.format(header['FWHM'])
                        if 'TRANSMIS' in header:
                            transmission = '{:10.3f}'.format(header['TRANSMIS'])
                        if 'SKYMDCOF' in header:
                            sky_residual = '{:10.3f}'.format(header['SKYMDCOF'])
                        with pf.open(best_path(fits)) as hdul:
                            if 'SKY' in [hdu.name for hdu in hdul]:
                                mean_sky_brightness='{:8.1f}'.format(np.median(hdul['SKY'].data))
                # Disabled flag
                    fits_2 = self.other_arm(fits)
                disabled_flag = 'T' if (fits is not None and fits.do_not_use) else 'F'
                disabled_flag += 'T' if (fits_2 is not None and fits_2.do_not_use) else 'F'

                # Sree: Estimations for Spector 
                if fits.instrument == 'AAOMEGA-HECTOR':
                    fits_3 = self.other_inst(fits)
                    fits_4 = self.other_inst(fits_2)
                    if fits_3 is None:
                        fwhm, transmission, sky_residual, mean_sky_brightness = [x + '/-' for x in [fwhm, transmission, sky_residual, mean_sky_brightness]]
                    else:
                        try:
                            header = pf.getheader(best_path(fits_3), 'QC')
                        except (IOError, KeyError):
                            fwhm, transmission, sky_residual, mean_sky_brightness = [x + '/-' for x in [fwhm, transmission, sky_residual, mean_sky_brightness]]
                            pass
                        else:
                            fwhm += '/{:.2f}'.format(header['FWHM']) if 'FWHM' in header else '/-'
                            transmission += '/{:.3f}'.format(header['TRANSMIS']) if 'TRANSMIS' in header else '/-'
                            sky_residual += '/{:.3f}'.format(header['SKYMDCOF']) if 'SKYMDCOF' in header else '/-'
                            with pf.open(best_path(fits_3)) as hdul:
                                mean_sky_brightness += '/{:.1f}'.format(np.median(hdul['SKY'].data)) if any(hdu.name == 'SKY' for hdu in hdul) else '/-'
                    disabled_flag += 'T' if (fits_3 is not None and fits_3.do_not_use) else 'F'
                    disabled_flag += 'T' if (fits_4 is not None and fits_4.do_not_use) else 'F'
                disabled_flag += '*' if 'T' in disabled_flag else ' '

                text += '{}X{}   {}  {:8d}  {}  {}  {}  {}\n'.format(
                    fits.filename[:5], fits.filename[6:10], disabled_flag,
                    int(fits.exposure), fwhm, transmission, sky_residual, mean_sky_brightness)
            text += '+' * 75 + '\n'
            text += '\n'
        pager(text)

        f = open(self.abs_root+'/qc_summary_'+self.abs_root[-13:]+'.txt', 'w')
        f.write(text)
        f.close()
        return


    def tdfdr_options(self, fits, throughput_method='default', tlm=False, verbose=True):
        """Set the 2dfdr reduction options for this file."""
        options = []

        # Define what the best choice is for a TLM:
        if (self.use_twilight_tlm_all or (self.use_twilight_tlm_blue and ((fits.ccd == 'ccd_1') or (fits.ccd == 'ccd_3')) and (fits.plate_id_short != 'Y14SAR4_P007'))):
            best_tlm = 'tlmap_mfsky'
        else:
            best_tlm = 'tlmap'

        # Define what the best choice is for a FFLAT, in particular
        # if we are going to use a twilight flat:
        if (self.use_twilight_flat_all or (self.use_twilight_flat_blue and ((fits.ccd == 'ccd_1') or (fits.ccd == 'ccd_3'))) ):	            
            best_fflat = 'fflat_mfsky'
        else:
            best_fflat = 'fflat'

        # only do skyscrunch for longer exposures (both CCDs):
        if fits.exposure < self.min_exposure_for_sky_wave:
            options.extend(['-SKYSCRUNCH', '0'])
        #else:
            # Adjust wavelength calibration of red frames using sky lines; This will be specified by idx files.
        #    options.extend(['-SKYSCRUNCH', '1'])

        # only do improved 5577 PCA for longer exposures (CCD_1 & CCD_3):
        if ((fits.ccd == 'ccd_1') or (fits.ccd == 'ccd_3')) and (fits.exposure <= self.min_exposure_for_5577pca):
            options.extend(['-PCASKY','0'])
                
        # add options for just CCD_2:
        if fits.ccd == 'ccd_2':
            # Turn off bias and dark subtraction
            if fits.detector == 'E2V3':
                options.extend(['-USEBIASIM', '0','-BIAS_FILENAME', '', '-USEDARKIM', '0','-DARK_FILENAME', ''])
            elif fits.detector == 'E2V3A':
                options.extend(['-USEBIASIM', '0','-BIAS_FILENAME', ''])

        # turn off bias and dark for new CCD. These are named
        # E2V2A (blue) and E2V3A (red).  The old ones are E2V2 (blue
        # and E2V3 (red).
        if fits.detector == 'E2V2A':
            options.extend(['-USEBIASIM', '0','-BIAS_FILENAME', '', '-USEDARKIM', '0','-DARK_FILENAME', ''])

        if fits.ndf_class == 'BIAS':
            files_to_match = []
        elif fits.ndf_class == 'DARK':
            files_to_match = ['bias']
        elif fits.ndf_class == 'LFLAT':
            files_to_match = ['bias', 'dark']
        elif fits.ndf_class == 'MFFFF' and tlm:
            files_to_match = ['bias', 'dark', 'lflat']
            #Sree (19May2025): due to fibre damanges of bundle H, gauss extraction mode frequently fails for tlm mapping at the telescope with fast mode
            #It may take more time but observers do not need to switch between the mode while observing
            #TODO: Revisit this once bundle H has been repaired. 
            #if self.speed == 'fast': 
            #    options.extend(['-EXTR_OPERATION', 'OPTEX'])
            #    options.extend(['-SCATSUB', 'TRACE'])
            #    options.extend(['-SKYFITORDER', '1'])
            #    options.extend(['-SKYSCRUNCHSM', 'TRUE'])
            #    options.extend(['-TPMETH', 'OFFSKY'])
            #    #options.extend(['-COSRAY_MTHD', 'LACOSMIC'])
        elif fits.ndf_class == 'MFARC':
            files_to_match = ['bias', 'dark', 'lflat', best_tlm]
            # Arc frames can't use optimal extraction because 2dfdr screws up
            # and marks entire columns as bad when it gets too many saturated
            # pixels
            # Sree: Hector uses optimal extraction from v0_02
            #options.extend(['-EXTR_OPERATION', 'GAUSS'])
        elif fits.ndf_class == 'MFFFF' and not tlm:
            #f self.speed == 'fast': #TODO: remove this?
            #    options.extend(['-EXTR_OPERATION', 'OPTEX'])
            #    options.extend(['-SCATSUB', 'TRACE'])
            #    options.extend(['-SKYFITORDER', '1'])
            #    options.extend(['-SKYSCRUNCHSM', 'TRUE'])
            #    options.extend(['-TPMETH', 'OFFSKY'])
            #    #options.extend(['-COSRAY_MTHD', 'LACOSMIC'])
            # new version of 2dfdr aaorun (2dfdr 7.0) also needs to pass the TLMAP_FILENAME argument
            # when reducing flat fields.  As a result we need to add this to the arguments.
            # We need to be careful (see discussion below) that it is the right filename,
            # i.e. just the tlm for the frame being reduced (SMC 14/06/2018).
            options.extend(['-TLMAP_FILENAME',fits.tlm_filename])
            # if not reduced for a TLM, then assume it has already been done, and set
            # flag to not repeat TLM.  The only reason to do this is save time in the
            # reductions.  Otherwise the result should be the same:
            options.extend(['-DO_TLMAP','0'])

            #Sree (7/11/2024): apply bspline smoothing for twilight flats
            if fits.filename[6] == '8': #twilight flats faked as MFFFF
                options.extend(['-BSSMOOTH', '1'])

            if fits.lamp == 'Flap':
                # Flap flats should use their own tramline maps, not those
                # generated by dome flats.  Do we want this to happen, even
                # if the best TLM could be from a twilight frame?  For now
                # leave it as this, but it may be that the twilight tlm (if
                # available) is better, at least in regard to the measurement
                # of the fibre profile widths.
                # files_to_match = ['bias', 'dark', 'lflat', 'tlmap_flap',
                #                  'wavel']
                # 2dfdr always remakes a TLM for an MFFFF, so don't set the
                # tlmap for these anyway:
                files_to_match = ['bias', 'dark', 'lflat', 'wavel']
            else:
                # if this is an MFFFF then always assure that we are using the
                # TLM that came from that file.  The main reason for this is that
                # if we pass a different TLM file, then a new TLM will be generated
                # anyway, but overwritten into the filename that is passed (e.g.
                # a twilight TLM could be overwritten by a dome flat TLM): 
                #files_to_match = ['bias', 'dark', 'lflat', 'tlmap', 'wavel']
                files_to_match = ['bias', 'dark', 'lflat','wavel']

        elif fits.ndf_class == 'MFSKY':
            files_to_match = ['bias', 'dark', 'lflat', best_tlm, 'wavel',
                              best_fflat]
        elif fits.ndf_class == 'MFOBJECT':
            if throughput_method == 'default':
                files_to_match = ['bias', 'dark', 'lflat', best_tlm, 'wavel',
                                  best_fflat, 'thput']
                options.extend(['-TPMETH', 'OFFSKY'])
            elif throughput_method == 'external':
                files_to_match = ['bias', 'dark', 'lflat', best_tlm, 'wavel',
                                  best_fflat]
                options.extend(['-TPMETH', 'OFFSKY'])
                options.extend(['-THPUT_FILENAME',
                                'thput_' + fits.reduced_filename])
            elif throughput_method == 'skylines':
                if (fits.exposure >= self.min_exposure_for_throughput and
                        fits.has_sky_lines()):
                    files_to_match = ['bias', 'dark', 'lflat', best_tlm, 'wavel',
                                      best_fflat]
                    options.extend(['-TPMETH', 'SKYFLUX(MED)'])
                else:
                    files_to_match = ['bias', 'dark', 'lflat', best_tlm, 'wavel',
                                      best_fflat, 'thput_object']
                    options.extend(['-TPMETH', 'OFFSKY'])
        else:
            raise ValueError('Unrecognised NDF_CLASS: ' + fits.ndf_class)
        # Remove unnecessary files from files_to_match
        if 'bias' in files_to_match and '-USEBIASIM' in options:
            if options[options.index('-USEBIASIM') + 1] == '0':
                files_to_match.remove('bias')
        if 'dark' in files_to_match and '-USEDARKIM' in options:
            if options[options.index('-USEDARKIM') + 1] == '0':
                files_to_match.remove('dark')
        if 'lflat' in files_to_match and '-USEFLATIM' in options:
            if options[options.index('-USEFLATIM') + 1] == '0':
                files_to_match.remove('lflat')
        # Disable bias/dark/lflat if they're not being used
        # If you don't, 2dfdr might barf
        if 'bias' not in files_to_match and '-USEBIASIM' not in options:
            options.extend(['-USEBIASIM', '0','-BIAS_FILENAME', ''])
        if 'dark' not in files_to_match and '-USEDARKIM' not in options:
            options.extend(['-USEDARKIM', '0','-DARK_FILENAME', ''])
        if 'lflat' not in files_to_match and '-USEFLATIM' not in options:
            options.extend(['-USEFLATIM', '0','-LFLAT_FILENAME', ''])
        for match_class in files_to_match:
            # this is the main call to the matching routine:
            filename_match = self.match_link(fits, match_class)
            if filename_match is None:
                # What to do if no match was found
                if match_class == 'bias':
                    if verbose:
                        print('Warning: Bias frame not found. '
                          'Turning off bias subtraction for ' + fits.filename)
                    options.extend(['-USEBIASIM', '0','-BIAS_FILENAME', ''])
                    continue
                elif match_class == 'dark':
                    if verbose:
                        print('Warning: Dark frame not found. '
                          'Turning off dark subtraction for ' + fits.filename)
                    options.extend(['-USEDARKIM', '0','-DARK_FILENAME', ''])
                    continue
                elif match_class == 'lflat':
                    if verbose:
                        print('Warning: LFlat frame not found. '
                          'Turning off LFlat division for ' + fits.filename)
                    options.extend(['-USEFLATIM', '0','-LFLAT_FILENAME', ''])
                    continue
                elif match_class == 'thput':
                    # Try to find a fake MFSKY made from a dome flat
                    filename_match = self.match_link(fits, 'thput_fflat')
                    if filename_match is None:
                        if (fits.exposure < self.min_exposure_for_throughput or
                                not fits.has_sky_lines()):
                            # Try to find a suitable object frame instead
                            filename_match = self.match_link(
                                fits, 'thput_object')
                            # Really run out of options here
                            if filename_match is None:
                                # Still nothing
                                if verbose:
                                    print('Warning: Offsky (or substitute) frame '
                                      'not found. Turning off throughput '
                                      'calibration for ' + fits.filename)
                                options.extend(['-THRUPUT', '0'])
                                continue
                        else:
                            # This is a long exposure, so use the sky lines
                            options[options.index('-TPMETH') + 1] = (
                                'SKYFLUX(MED)')
                elif match_class == best_tlm:
                    # If we are using twilights, then go through the 3 different
                    # twilight options first.  If they are not found, then default
                    # back to the normal tlmap route.
                    found = 0
                    if (self.use_twilight_tlm_blue and ((fits.ccd == 'ccd_1') or (fits.ccd == 'ccd_3')) and 
                        (fits.plate_id_short != 'Y14SAR4_P007')):
                        filename_match = self.match_link(fits, 'tlmap_mfsky')
                        if filename_match is None:
                            filename_match = self.match_link(fits, 'tlmap_mfsky_loose')
                            if filename_match is None:
                                filename_match = self.match_link(fits, 'tlmap_mfsky_any')
                                if filename_match is None:
                                    if verbose:
                                        print('Warning: no matching twilight frames found for TLM. '
                                          'Will default to using flat field frames instead '
                                          'for ' + fits.filename)
                                else:
                                    if verbose:
                                        print('Warning: No matching twilight found for TLM. '
                                          'Using a twilight frame from a different night '
                                          'for ' + fits.filename)
                                    found = 1
                            else:
                                if verbose:
                                    print('Warning: No matching twilight found for TLM.'
                                      'Using a twilight frame from the same night'
                                      'for ' + fits.filename)
                                found = 1
                        else:
                            if verbose:
                                print('Found matching twilight for TLM '
                                  'for ' + fits.filename)
                            found = 1

                    # if we haven't already found a matching TLM above (i.e. if found = 0), then
                    # go through the options with the flats:
                    if (found == 0):
                        # Try with normal TLM from flat:
                        filename_match = self.match_link(fits, 'tlmap')
                        if filename_match is None:
                            # Try with looser criteria
                            filename_match = self.match_link(fits, 'tlmap_loose')
                            if filename_match is None:
                                # Try using a flap flat instead
                                filename_match = self.match_link(fits, 'tlmap_flap')
                                if filename_match is None:
                                    # Try with looser criteria
                                    filename_match = self.match_link(
                                        fits, 'tlmap_flap_loose')
                                    if filename_match is None:
                                        # Still nothing. Raise an exception
                                        raise MatchException(
                                            'No matching tlmap found for ' +
                                            fits.filename)
                                    else:
                                        if verbose:
                                            print('Warning: No good flat found for TLM. '
                                              'Using flap flat from different field '
                                              'for ' + fits.filename)
                                else:
                                    if verbose:
                                        print('Warning: No dome flat found for TLM. '
                                          'Using flap flat instead for ' + fits.filename)
                            else:
                                if verbose:
                                    print('Warning: No matching flat found for TLM. '
                                      'Using flat from different field for ' +
                                      fits.filename)
                        else:
                            if verbose:
                                print('Warning: No matching twilight found for TLM. '
                                  'Using a dome flat instead ' +
                                  fits.filename)

                elif match_class == best_fflat:
                    # If we are using twilights, then go through the 3 different
                    # twilight options first.  If they are not found, then default
                    # back to the normal fflat route (this is a copy of the version
                    # for the TLM above - with minor changes):
                    found = 0
                    if (self.use_twilight_flat_blue) and ((fits.ccd == 'ccd_1') or (fits.ccd == 'ccd_3')): #marie adds ccd_3
                        filename_match = self.match_link(fits, 'fflat_mfsky')
                        if filename_match is None:
                            filename_match = self.match_link(fits, 'fflat_mfsky_loose')
                            if filename_match is None:
                                filename_match = self.match_link(fits, 'fflat_mfsky_any')
                                if filename_match is None:
                                    if verbose:
                                        print('Warning: no matching twilight frames found for FFLAT.'
                                          'Will default to using flat field frames instead'
                                          'for ' + fits.filename)
                                else:
                                    if verbose:
                                        print('Warning: No matching twilight found for FFLAT.'
                                          'Using a twilight frame from a different night'
                                          'for ' + fits.filename)
                                    found = 1
                            else:
                                if verbose:
                                    print('Warning: No matching twilight found for FFLAT.'
                                      'Using a twilight frame from the same night'
                                      'for ' + fits.filename)
                                found = 1
                        else:
                            if verbose:
                                print('Found matching twilight for FFLAT '
                                  'for ' + fits.filename)
                            found = 1

                    # if we haven't already found a matching FFLAT above (i.e. if found = 0), then
                    # go through the options with the flats:
                    if (found == 0):
                        options.extend(['-TRUNCFLAT', '1'])
                        # Try with normal FFLAT from flat:
                        filename_match = self.match_link(fits, 'fflat')
                        if filename_match is None:
                            # Try with looser criteria
                            filename_match = self.match_link(fits, 'fflat_loose')
                            if filename_match is None:
                                # Try using a flap flat instead
                                filename_match = self.match_link(fits, 'fflat_flap')
                                if filename_match is None:
                                    # Try with looser criteria
                                    filename_match = self.match_link(
                                        fits, 'fflat_flap_loose')
                                    if filename_match is None:
                                        # Still nothing. Raise an exception
                                        raise MatchException(
                                            'No matching tlmap found for ' +
                                            fits.filename)
                                    else:
                                        print('Warning: No good flat found for FFLAT. '
                                              'Using flap flat from different field '
                                              'for ' + fits.filename)
                                else:
                                    print('Warning: No dome flat found for FFLAT. '
                                          'Using flap flat instead for ' + fits.filename)
                            else:
                                print('Warning: No matching flat found for FFLAT. '
                                      'Using flat from different field for ' +
                                      fits.filename)
                        else:
                            print('Warning: No matching twilight found for FFLAT. '
                                  'Using a dome flat instead ' +
                                  fits.filename)

                ## elif match_class == 'fflat':
                ##     # Try with looser criteria
                ##     filename_match = self.match_link(fits, 'fflat_loose')
                ##     if filename_match is None:
                ##         # Try using a flap flat instead
                ##         filename_match = self.match_link(fits, 'fflat_flap')
                ##         if filename_match is None:
                ##             # Try with looser criteria
                ##             filename_match = self.match_link(
                ##                 fits, 'fflat_flap_loose')
                ##             if filename_match is None:
                ##                 # Still nothing. Raise an exception
                ##                 raise MatchException(
                ##                     'No matching fflat found for ' + 
                ##                     fits.filename)
                ##             else:
                ##                 print ('Warning: No good flat found for '
                ##                     'flat fielding. '
                ##                     'Using flap flat from different field '
                ##                     'for ' + fits.filename)
                ##         else:
                ##             print ('Warning: No dome flat found for flat '
                ##                 'fielding. '
                ##                 'Using flap flat instead for ' + fits.filename)
                ##     else:
                ##         print ('Warning: No matching flat found for flat '
                ##             'fielding. '
                ##             'Using flat from different field for ' + 
                ##             fits.filename)
                elif match_class == 'wavel':
                    # Try with looser criteria
                    filename_match = self.match_link(fits, 'wavel_loose')
                    if filename_match is None:
                        # Still nothing. Raise an exception
                        raise MatchException('No matching wavel found for ' +
                                             fits.filename)
                    else:
                        print('Warning: No good arc found for wavelength '
                              'solution. Using arc from different field '
                              'for ' + fits.filename)
                else:
                    # Anything else missing is fatal
                    raise MatchException('No matching ' + match_class +
                                         ' found for ' + fits.filename)
            if filename_match is not None:
                # Note we can't use else for the above line, because
                # filename_match might have changed
                # Make sure that 2dfdr gets the correct option names
                # We have added the tlmap_mfsky option here.
                if match_class == 'tlmap_flap':
                    match_class = 'tlmap'
                elif match_class == 'tlmap_mfsky':
                    match_class = 'tlmap'
                elif match_class == 'tlmap_mfsky_loose':
                    match_class = 'tlmap'
                elif match_class == 'tlmap_mfsky_any':
                    match_class = 'tlmap'
                elif match_class == 'thput_object':
                    match_class = 'thput'
                elif match_class == 'fflat_flap':
                    match_class = 'fflat'
                elif match_class == 'fflat_loose':
                    match_class = 'fflat'
                elif match_class == 'fflat_mfsky':
                    match_class = 'fflat'
                elif match_class == 'fflat_mfsky_loose':
                    match_class = 'fflat'
                elif match_class == 'fflat_mfsky_any':
                    match_class = 'fflat'

                options.extend(['-' + match_class.upper() + '_FILENAME',
                                filename_match])
        return options

    def determine_tlm_shift(self,fits,twilight_fits,flat_fits):
        twilight_fits = os.path.join(fits.reduced_dir,twilight_fits)
        flat_fits = os.path.join(fits.reduced_dir,flat_fits)
        
        twilight_tlm = pf.getdata(twilight_fits,'PRIMARY')
        flat_tlm = pf.getdata(flat_fits,'PRIMARY')
        
        tlm_offset = np.mean(twilight_tlm-flat_tlm)
        return tlm_offset

    def run_2dfdr_combine(self, file_iterable, output_path): 
        """Use 2dfdr to combine the specified FITS files."""

   #     file_iterable, file_iterable_copy,copy = itertools.tee(file_iterable,3)
        file_iterable, file_iterable_copy, file_iterable_dummy = itertools.tee(file_iterable,3)

        input_path_list = [fits.reduced_path for fits in file_iterable]

        if not input_path_list:
            print('No reduced files found to combine!')
            return
        # Following line uses the last FITS file, assuming all are the same CCD
        grating = next(file_iterable_copy).grating
        idx_file = self.idx_files[grating]
        print('Combining files to create', output_path)
        tdfdr.run_2dfdr_combine(input_path_list, output_path, idx_file, self.dummy)

        # Create dummy output if pipeline is being run in dummy mode
        if self.dummy:
            reduced_files = [fits for fits in file_iterable_dummy]
            create_dummy_combine(input_path_list[0], output_path, reduced_files[0].ndf_class)
        return

    def files(self, ndf_class=None, date=None, plate_id=None,
              plate_id_short=None, field_no=None, field_id=None,
              ccd=None, exposure_str=None, do_not_use=None,
              min_exposure=None, max_exposure=None,
              reduced_dir=None, reduced=None, copy_reduced=None,
              tlm_created=None, flux_calibrated=None, telluric_corrected=None,
              spectrophotometric=None, name=None, lamp=None, min_fluxlev=None,
              max_fluxlev=None,
              central_wavelength=None, include_linked_managers=False, filename=None):
        """Generator for FITS files that satisfy requirements."""
        if include_linked_managers:
            # Include files from linked managers too
            file_list = itertools.chain(
                self.file_list,
                *[mngr.file_list for mngr in self.linked_managers])
        else:
            file_list = self.file_list  # type: List[FITSFile]

        for fits in file_list:
            if fits.ndf_class is None:
                continue
            if ((ndf_class is None or fits.ndf_class in ndf_class) and
                    (date is None or fits.date in date) and
                    (plate_id is None or fits.plate_id in plate_id) and
                    (plate_id_short is None or
                     fits.plate_id_short in plate_id_short) and
                    (field_no is None or fits.field_no == field_no) and
                    (field_id is None or fits.field_id in field_id) and
                    (ccd is None or fits.ccd in ccd) and
                    (exposure_str is None or
                     fits.exposure_str in exposure_str) and
                    (do_not_use is None or fits.do_not_use == do_not_use) and
                    (min_exposure is None or fits.exposure >= min_exposure) and
                    (max_exposure is None or fits.exposure <= max_exposure) and
                    (min_fluxlev is None or fits.fluxlev >= min_fluxlev) and  # add flux level limits
                    (max_fluxlev is None or fits.fluxlev <= max_fluxlev) and
                    (reduced_dir is None or
                     os.path.realpath(reduced_dir) ==
                     os.path.realpath(fits.reduced_dir)) and
                    (reduced is None or
                     (reduced and os.path.exists(fits.reduced_path)) or
                     (not reduced and
                      not os.path.exists(fits.reduced_path))) and
                    (copy_reduced is None or
                     (copy_reduced and os.path.exists(
                         self.copy_path(fits.reduced_path,fits.ndf_class))) or
                     (not copy_reduced and not os.path.exists(
                         self.copy_path(fits.reduced_path,fits.ndf_class)))) and
                    (tlm_created is None or
                     (tlm_created and hasattr(fits, 'tlm_path') and
                      os.path.exists(fits.tlm_path)) or
                     (not tlm_created and hasattr(fits, 'tlm_path') and
                      not os.path.exists(fits.tlm_path))) and
                    (flux_calibrated is None or
                     (flux_calibrated and hasattr(fits, 'fluxcal_path') and
                      os.path.exists(fits.fluxcal_path)) or
                     (not flux_calibrated and hasattr(fits, 'fluxcal_path') and
                      not os.path.exists(fits.fluxcal_path))) and
                    (telluric_corrected is None or
                     (telluric_corrected and hasattr(fits, 'telluric_path') and
                      os.path.exists(fits.telluric_path)) or
                     (not telluric_corrected and
                      hasattr(fits, 'telluric_path') and
                      not os.path.exists(fits.telluric_path))) and
                    (spectrophotometric is None or
                     (hasattr(fits, 'spectrophotometric') and
                      (fits.spectrophotometric == spectrophotometric))) and
                    (name is None or
                     (fits.name is not None and fits.name in name)) and
                    (filename is None or
                     (fits.filename is not None and fits.filename in filename)) and
                    (lamp is None or fits.lamp == lamp) and
                    (central_wavelength is None or
                     fits.central_wavelength == central_wavelength)):
                yield fits
        return

    def group_files_by(self, keys, require_this_manager=True, **kwargs):
        """Return a dictionary of FITSFile objects grouped by the keys."""
        if isinstance(keys, six.string_types):
            keys = [keys]
        groups = defaultdict(list)
        for fits in self.files(**kwargs):
            combined_key = []
            for key in keys:
                combined_key.append(getattr(fits, key))
            combined_key = tuple(combined_key)
            groups[combined_key].append(fits)
        if require_this_manager:
            # Check that at least one of the files from each group has come
            # from this manager
            for combined_key, fits_list in list(groups.items()):
                for fits in fits_list:
                    if fits in self.file_list:
                        break
                else:
                    # None of the files are from this manager
                    del groups[combined_key]
        return groups

    def ccds(self, do_not_use=False):
        """Generator for ccd names in the data."""
        ccd_list = []
        for fits in self.files(do_not_use=do_not_use):
            if fits.ccd not in ccd_list:
                ccd_list.append(fits.ccd)
                yield fits.ccd
        return

    def reduced_dirs(self, dir_type=None, **kwargs):
        """Generator for reduced directories containing particular files."""
        reduced_dir_list = []
        if dir_type is None:
            ndf_class = None
            spectrophotometric = None
        else:
            ndf_class = {'bias': 'BIAS',
                         'dark': 'DARK',
                         'lflat': 'LFLAT',
                         'calibrators': ['MFFFF', 'MFARC', 'MFSKY'],
                         'object': 'MFOBJECT',
                         'mffff': 'MFFFF',
                         'mfarc': 'MFARC',
                         'mfsky': 'MFSKY',
                         'mfobject': 'MFOBJECT',
                         'spectrophotometric': 'MFOBJECT'}[dir_type.lower()]
            if dir_type == 'spectrophotometric':
                spectrophotometric = True
            elif ndf_class == 'MFOBJECT':
                spectrophotometric = False
            else:
                spectrophotometric = None
        for fits in self.files(ndf_class=ndf_class,
                               spectrophotometric=spectrophotometric,
                               **kwargs):
            if fits.reduced_dir not in reduced_dir_list:
                reduced_dir_list.append(fits.reduced_dir)
                yield fits.reduced_dir
        return

    def dark_exposure_strs(self, ccd, do_not_use=False):
        """Generator for dark exposure strings for a given ccd name."""
        exposure_str_list = []
        for fits in self.files(ndf_class='DARK', ccd=ccd,
                               do_not_use=do_not_use):
            if fits.exposure_str not in exposure_str_list:
                exposure_str_list.append(fits.exposure_str)
                yield fits.exposure_str
        return

    def combined_filenames_paths(self, calibrator_type, do_not_use=False):
        """Generator for filename and path of XXXXcombined.fits files."""
        self.check_calibrator_type(calibrator_type)
        for ccd in self.ccds(do_not_use=do_not_use):
            if calibrator_type.lower() == 'bias':
                yield (ccd,
                       None,
                       self.bias_combined_filename(),
                       self.bias_combined_path(ccd))
            elif calibrator_type.lower() == 'dark':
                for exposure_str in self.dark_exposure_strs(
                        ccd, do_not_use=do_not_use):
                    yield (ccd,
                           exposure_str,
                           self.dark_combined_filename(exposure_str),
                           self.dark_combined_path(ccd, exposure_str))
            elif calibrator_type.lower() == 'lflat':
                yield (ccd,
                       None,
                       self.lflat_combined_filename(),
                       self.lflat_combined_path(ccd))
        return

    def other_arm(self, fits, include_linked_managers=False):
        """Return the FITSFile from the other arm of the spectrograph."""
        if fits.ccd == 'ccd_1':
            other_number = '2'
        elif fits.ccd == 'ccd_2':
            other_number = '1'
        elif fits.ccd == 'ccd_3':
            other_number = '4'
        elif fits.ccd == 'ccd_4':
            other_number = '3'
        else:
            raise ValueError('Unrecognised CCD: ' + fits.ccd)
        other_filename = fits.filename[:5] + other_number + fits.filename[6:]
        other_fits = self.fits_file(
            other_filename, include_linked_managers=include_linked_managers)
        return other_fits

    def other_inst(self, fits, include_linked_managers=False):
        """Return the FITSFile from the other spectrograph."""
        if fits.ccd == 'ccd_1':
            other_number = '3'
        elif fits.ccd == 'ccd_2':
            other_number = '4'
        elif fits.ccd == 'ccd_3':
            other_number = '1'
        elif fits.ccd == 'ccd_4':
            other_number = '2'
        else:
            raise ValueError('Unrecognised CCD: ' + fits.ccd)
        other_filename = fits.filename[:5] + other_number + fits.filename[6:]
        other_fits = self.fits_file(
            other_filename, include_linked_managers=include_linked_managers)
        return other_fits



    def cubed_path(self, name, arm, fits_list, field_id, gzipped=False,
                   exists=False, tag=None, ndither=None, **kwargs):
        """Return the path to the cubed file."""
        n_file = len(self.qc_for_cubing(fits_list, **kwargs))
        if ndither:
            n_file = ndither
        path = os.path.join(
            self.abs_root, 'cubed', name,
            name + '_' + arm + '_' + str(n_file) + '_' + field_id)
        if tag:
            path += '_' + tag
        path += '.fits'
        if gzipped:
            path = path + '.gz'
        if exists:
            if not os.path.exists(path):
                path = self.cubed_path(name, arm, fits_list, field_id,
                                       gzipped=(not gzipped), exists=False,
                                       tag=tag, ndither=ndither, **kwargs)
                if not os.path.exists(path):
                    return None
        return path

    def matchmaker(self, fits, match_class):
        """Return the file that should be used to help reduce the FITS file.

        match_class is one of the following:
        tlmap_mfsky      -- Find a tramline map from twilight flat fields
        tlmap_mfsky_loose-- Find a tramline map from any twilight flat field on a night
        tlmap_mfsky_any  -- Find a tramline map from any twilight flat field in a manager set 
        tlmap            -- Find a tramline map from the dome lamp
        tlmap_loose      -- As tlmap, but with less strict criteria
        tlmap_flap       -- As tlmap, but from the flap lamp
        tlmap_flap_loose -- As tlmap_flap, but with less strict criteria
        wavel            -- Find a reduced arc file
        wavel_loose      -- As wavel, but with less strict criteria
        fflat_mfsky      -- Find a reduced fibre flat field from a twilight flat
        fflat_mfsky_loose-- Find a reduced fibre flat field from any twilight flat field on a night
        fflat_mksky_any  -- Find a reduced fibre flat field from any twilight flat field in a manager set 
        fflat            -- Find a reduced fibre flat field from the dome lamp
        fflat_loose      -- As fflat, but with less strict criteria
        fflat_flap       -- As fflat, but from the flap lamp
        fflat_flap_loose -- As fflat_flap, but with less strict criteria
        thput            -- Find a reduced offset sky (twilight) file
        thput_fflat      -- Find a dome flat that's had a copy made as MFSKY
        thput_sky        -- As thput, but find long-exposure object file
        bias             -- Find a combined bias frame
        dark             -- Find a combined dark frame
        lflat            -- Find a combined long-slit flat frame
        fcal             -- Find a reduced spectrophotometric standard star
        fcal_loose       -- As fcal, but with less strict criteria

        The return type depends on what is asked for:
        tlmap, wavel, fflat, thput, fcal and related 
                                -- A FITS file object
        bias, dark, lflat       -- The path to the combined file
        """
        fits_match = None
        # The following are the things that could potentially be matched
        date = None
        plate_id = None
        field_id = None
        ccd = None
        exposure_str = None
        min_exposure = None
        max_exposure = None
        reduced_dir = None
        reduced = None
        copy_reduced = None
        tlm_created = None
        flux_calibrated = None
        telluric_corrected = None
        spectrophotometric = None
        lamp = None
        central_wavelength = None
        # extra match criteria that is the amount of flux in the
        # frame, based on the FLXU90P value (9-95th percentile value
        # of raw frame).  This is for twilights used as flats for
        # TLMs.  If a frame is a twilight, then this paramater is
        # set on initialization of the FITSFile object.  Then we
        # have easy access to the value.
        min_fluxlev = None
        max_fluxlev = None
        # Define some functions for figures of merit
        time_difference = lambda fits, fits_test: (
            abs(fits_test.epoch - fits.epoch))
        recent_reduction = lambda fits, fits_test: (
                -1.0 * os.stat(fits_test.reduced_path).st_mtime)
        copy_recent_reduction = lambda fits, fits_test: (
                -1.0 * os.stat(self.copy_path(fits_test.reduced_path,fits_test.ndf_class)).st_mtime)
        # merit function that returns the best fluxlev value.  As the
        # general f-o-m selects objects if the f-o-m is LESS than other values
        # we should just multiple fluxlev by -1:
        flux_level = lambda fits, fits_test: (
                -1.0 * fits_test.fluxlev)

        def time_difference_min_exposure(min_exposure):
            def retfunc(fits, fits_test):
                if fits_test.exposure <= min_exposure:
                    return np.inf
                else:
                    return time_difference(fits, fits_test)

            return retfunc

        def determine_tlm_shift_fits(twilight_fits,flat_fits):

            twilight_tlm = pf.getdata(twilight_fits.tlm_path,'PRIMARY')
            flat_tlm = pf.getdata(flat_fits.tlm_path,'PRIMARY')

            tlm_offset = np.mean(twilight_tlm-flat_tlm)
            return tlm_offset

        def flux_level_shift(fits,fits_test):
            
            fits_comp = self.matchmaker(fits,'tlmap')
            if fits_comp == None:
                fits_comp = self.matchmaker(fits,'tlmap_loose')
            shift = determine_tlm_shift_fits(fits_test,fits_comp)

            if np.abs(shift) >= 1:
                    return np.inf
            else:
                return flux_level(fits, fits_test)

        # Determine what actually needs to be matched, depending on match_class
        #
        # this case is where we want to use a twilight sky frame to derive the
        # tramline maps, rather than a flat field, as the flat can often have too
        # little flux in the far blue to do a good job.  The order of matching for
        # the twilights should be:
        # 1) The brightest twilight frame of the same field (needs to be brighter than
        #    some nominal level, say FLUX90P>500) - tlmap_mfsky.
        # 2) The brightest twilight frame from the same night (same constraint on
        #    brightness) - tlmap_mfsky_loose.
        # 3) The brightest twilight frame from a different night (same constraint on
        #    brightness) - tlmap_mfsky_any.
        if match_class.lower() == 'tlmap_mfsky':
            # allow MFSKY to be used:
            ndf_class = 'MFSKY'
            date = fits.date
            plate_id = fits.plate_id
            field_id = fits.field_id
            min_fluxlev = 1000.0
            max_fluxlev = 40000.0  # use a max_fluxlev to reduce the chance of saturated twilights
            ccd = fits.ccd
            tlm_created = True
            fom = flux_level
        elif match_class.lower() == 'tlmap_mfsky_loose':
            # this is the case where we take the brightest twilight on the same
            # night, irrespective of whether its from the same plate.
            ndf_class = 'MFSKY'
            date = fits.date
            min_fluxlev = 1000.0
            max_fluxlev = 40000.0  # use a max_fluxlev to reduce the chance of saturated twilights
            ccd = fits.ccd
            tlm_created = True
            fom = flux_level_shift
        elif match_class.lower() == 'tlmap_mfsky_any':
            # in this case find the best (brightest) twilight frame from anywhere
            # during the run.
            ndf_class = 'MFSKY'
            min_fluxlev = 1000.0
            max_fluxlev = 40000.0  # use a max_fluxlev to reduce the chance of saturated twilights
            ccd = fits.ccd
            tlm_created = True
            fom = flux_level_shift
        elif match_class.lower() == 'tlmap':
            ndf_class = 'MFFFF'
            date = fits.date
            plate_id = fits.plate_id
            field_id = fits.field_id
            ccd = fits.ccd
            tlm_created = True
            lamp = 'Dome'
            fom = time_difference
        elif match_class.lower() == 'tlmap_loose':
            # Find a tramline map with looser criteria
            ndf_class = 'MFFFF'
            ccd = fits.ccd
            tlm_created = True
            lamp = 'Dome'
            fom = time_difference
        elif match_class.lower() == 'tlmap_flap':
            # Find a tramline map from a flap flat
            ndf_class = 'MFFFF'
            date = fits.date
            plate_id = fits.plate_id
            field_id = fits.field_id
            ccd = fits.ccd
            tlm_created = True
            lamp = 'Flap'
            fom = time_difference
        elif match_class.lower() == 'tlmap_flap_loose':
            # Tramline map from flap flat with looser criteria
            ndf_class = 'MFFFF'
            ccd = fits.ccd
            tlm_created = True
            lamp = 'Flap'
            fom = time_difference
        elif match_class.lower() == 'wavel':
            # Find a reduced arc field
            ndf_class = 'MFARC'
            date = fits.date
            plate_id = fits.plate_id
            field_id = fits.field_id
            ccd = fits.ccd
            reduced = True
            fom = time_difference
        elif match_class.lower() == 'wavel_loose':
            # Find a reduced arc field, with looser criteria
            ndf_class = 'MFARC'
            ccd = fits.ccd
            reduced = True
            fom = time_difference
        # options for using twilight frame as flibre flat:
        elif match_class.lower() == 'fflat_mfsky':
            # allow MFSKY to be used:
            ndf_class = 'MFSKY'
            date = fits.date
            plate_id = fits.plate_id
            field_id = fits.field_id
            min_fluxlev = 1000.0
            max_fluxlev = 40000.0  # use a max_fluxlev to reduce the chance of saturated twilights
            ccd = fits.ccd
            copy_reduced = True
            fom = flux_level
        elif match_class.lower() == 'fflat_mfsky_loose':
            # this is the case where we take the brightest twilight on the same
            # night, irrespective of whether its from the same plate.
            ndf_class = 'MFSKY'
            date = fits.date
            min_fluxlev = 1000.0
            max_fluxlev = 40000.0  # use a max_fluxlev to reduce the chance of saturated twilights
            ccd = fits.ccd
            copy_reduced = True
            fom = flux_level_shift
        elif match_class.lower() == 'fflat_mfsky_any':
            # in this case find the best (brightest) twilight frame from anywhere
            # during the run.
            ndf_class = 'MFSKY'
            min_fluxlev = 1000.0
            max_fluxlev = 40000.0  # use a max_fluxlev to reduce the chance of saturated twilights
            ccd = fits.ccd
            copy_reduced = True
            fom = flux_level_shift
        elif match_class.lower() == 'fflat':
            # Find a reduced fibre flat field from the dome lamp
            ndf_class = 'MFFFF'
            date = fits.date
            plate_id = fits.plate_id
            field_id = fits.field_id
            ccd = fits.ccd
            reduced = True
            lamp = 'Dome'
            fom = time_difference
        elif match_class.lower() == 'fflat_loose':
            # Find a reduced fibre flat field with looser criteria
            ndf_class = 'MFFFF'
            ccd = fits.ccd
            reduced = True
            lamp = 'Dome'
            fom = time_difference
        elif match_class.lower() == 'fflat_flap':
            # Find a reduced flap fibre flat field
            ndf_class = 'MFFFF'
            date = fits.date
            plate_id = fits.plate_id
            field_id = fits.field_id
            ccd = fits.ccd
            reduced = True
            lamp = 'Flap'
            fom = time_difference
        elif match_class.lower() == 'fflat_flap_loose':
            # Fibre flat field from flap lamp with looser criteria
            ndf_class = 'MFFFF'
            ccd = fits.ccd
            reduced = True
            lamp = 'Flap'
            fom = time_difference
        elif match_class.lower() == 'thput':
            # Find a reduced offset sky field
            ndf_class = 'MFSKY'
            date = fits.date
            plate_id = fits.plate_id
            field_id = fits.field_id
            ccd = fits.ccd
            reduced = True
            fom = recent_reduction
        elif match_class.lower() == 'thput_fflat':
            # Find a dome flat that's had a fake sky copy made
            ndf_class = 'MFFFF'
            date = fits.date
            plate_id = fits.plate_id
            field_id = fits.field_id
            ccd = fits.ccd
            copy_reduced = True
            fom = copy_recent_reduction
        elif match_class.lower() == 'thput_object':
            # Find a reduced object field to take the throughput from
            ndf_class = 'MFOBJECT'
            date = fits.date
            plate_id = fits.plate_id
            field_id = fits.field_id
            ccd = fits.ccd
            reduced = True
            fom = time_difference_min_exposure(
                self.min_exposure_for_throughput)
        elif match_class.lower() == 'fcal':
            # Find a spectrophotometric standard star
            ndf_class = 'MFOBJECT'
            date = fits.date
            plate_id = fits.plate_id
            field_id = fits.field_id
            ccd = fits.ccd
            reduced = True
            spectrophotometric = True
            central_wavelength = fits.central_wavelength
            fom = time_difference
        elif match_class.lower() == 'fcal_loose':
            # Spectrophotometric with less strict criteria
            ndf_class = 'MFOBJECT'
            ccd = fits.ccd
            reduced = True
            spectrophotometric = True
            central_wavelength = fits.central_wavelength
            fom = time_difference
        elif match_class.lower() == 'bias':
            # Just return the standard BIAScombined filename
            filename = self.bias_combined_filename()
            if os.path.exists(os.path.join(fits.reduced_dir, filename)):
                return filename
            else:
                return None
        elif match_class.lower() == 'dark':
            # This works a bit differently. Return the filename of the
            # combined dark frame with the closest exposure time.
            best_fom = np.inf
            exposure_str_match = None
            for exposure_str in self.dark_exposure_strs(ccd=fits.ccd):
                test_fom = abs(float(exposure_str) - fits.exposure)
                if test_fom < best_fom:
                    exposure_str_match = exposure_str
                    best_fom = test_fom
            if exposure_str_match is None:
                return None
            filename = self.dark_combined_filename(exposure_str_match)
            if os.path.exists(os.path.join(fits.reduced_dir, filename)):
                return filename
            else:
                return None
        elif match_class.lower() == 'lflat':
            # Just return the standard LFLATcombined filename
            filename = self.lflat_combined_filename()
            if os.path.exists(os.path.join(fits.reduced_dir, filename)):
                return filename
            else:
                return None
        else:
            raise ValueError('Unrecognised match_class')
        # Perform the match
        best_fom = np.inf
        for fits_test in self.files(
                ndf_class=ndf_class,
                date=date,
                plate_id=plate_id,
                field_id=field_id,
                ccd=ccd,
                exposure_str=exposure_str,
                min_exposure=min_exposure,
                max_exposure=max_exposure,
                reduced_dir=reduced_dir,
                reduced=reduced,
                copy_reduced=copy_reduced,
                tlm_created=tlm_created,
                flux_calibrated=flux_calibrated,
                telluric_corrected=telluric_corrected,
                spectrophotometric=spectrophotometric,
                lamp=lamp,
                min_fluxlev=min_fluxlev,
                max_fluxlev=max_fluxlev,
                do_not_use=False,
        ):
            test_fom = fom(fits, fits_test)
            if test_fom < best_fom:
                fits_match = fits_test
                best_fom = test_fom
        #        exit()
        if (best_fom == np.inf) & (('tlmap_mfsky' in match_class.lower()) | ('fflat_mfsky' in match_class.lower())):
            return None
        return fits_match

    def match_link(self, fits, match_class):
        """Match and make a link to a file, and return the filename."""
        #        print 'started match_link: ',match_class,fits.filename
        fits_match = self.matchmaker(fits, match_class)

        if fits_match is None:
            # No match was found, send the lack of match onwards
            return None
        if match_class.lower() in ['bias', 'dark', 'lflat']:
            # matchmaker returns a filename in these cases; send it straight on
            filename = fits_match
        elif match_class.lower().startswith('tlmap'):
            filename = fits_match.tlm_filename
            raw_filename = fits_match.filename
            raw_dir = fits_match.raw_dir
            # add im file as this is needed for tlm offset estimate:
            imfilename = fits_match.im_filename
        elif match_class.lower() == 'thput':
            thput_filename = 'thput_' + fits_match.reduced_filename
            thput_path = os.path.join(fits_match.reduced_dir, thput_filename)
            if os.path.exists(thput_path):
                filename = thput_filename
            else:
                filename = fits_match.reduced_filename
            raw_filename = fits_match.filename
            raw_dir = fits_match.raw_dir
        elif match_class.lower() == 'thput_fflat':
            filename = self.copy_path(fits_match.reduced_filename,fits_match.ndf_class)
            raw_filename = self.copy_path(fits_match.filename,fits_match.ndf_class)
            raw_dir = fits_match.reduced_dir
        elif match_class.lower().startswith('fflat_mfsky'):
            # case of using twilight frame for fibre flat.  In this case
            # we need to use the copy_reduced_filename, that is the one
            # with the leading 9 in the file name:
            filename = fits_match.copy_reduced_filename
            raw_filename = fits_match.filename
            raw_dir = fits_match.raw_dir
        else:
            filename = fits_match.reduced_filename
            raw_filename = fits_match.filename
            raw_dir = fits_match.raw_dir
        # These are the cases where we do want to make a link
        require_link = [
            'tlmap_mfsky', 'tlmap_mfsky_loose', 'tlmap_mfsky_any',
            'tlmap', 'tlmap_loose', 'tlmap_flap', 'tlmap_flap_loose',
            'fflat', 'fflat_loose', 'fflat_flap', 'fflat_flap_loose',
            'wavel', 'wavel_loose', 'thput', 'thput_fflat', 'thput_object',
            'tlmap_mfsky', 'fflat_mfsky', 'fflat_mfsky_loose', 'fflat_mfsky_any']
        if match_class.lower() in require_link:
            link_path = os.path.join(fits.reduced_dir, filename)
            source_path = os.path.join(fits_match.reduced_dir, filename)
            raw_link_path = os.path.join(fits.reduced_dir, raw_filename)
            raw_source_path = os.path.join(raw_dir, raw_filename)
            # If the link path is occupied by a link, delete it
            # Leave actual files in place
            if os.path.islink(link_path):
                os.remove(link_path)
            if os.path.islink(raw_link_path):
                os.remove(raw_link_path)
            # Make a link, unless the file is already there
            if not os.path.exists(link_path):
                os.symlink(os.path.relpath(source_path, fits.reduced_dir),
                           link_path)
            if not os.path.exists(raw_link_path):
                os.symlink(os.path.relpath(raw_source_path, fits.reduced_dir),
                           raw_link_path)
            # add links to im files if we are looking for a TLM:
            if match_class.lower().startswith('tlmap'):
                im_link_path = os.path.join(fits.reduced_dir, imfilename)
                im_source_path = os.path.join(fits_match.reduced_dir, imfilename)
                if os.path.islink(im_link_path):
                    os.remove(im_link_path)
                if not os.path.exists(im_link_path):
                    os.symlink(os.path.relpath(im_source_path, fits.reduced_dir),
                               im_link_path)
        return filename

    def change_speed(self, speed=None):
        """Switch between fast and slow reductions."""
        if speed is None:
            if self.speed == 'fast':
                speed = 'slow'
            else:
                speed = 'fast'
        if speed not in ('fast', 'slow'):
            raise ValueError("Speed must be 'fast' or 'slow'.")
        self.speed = speed
        self.idx_files = IDX_FILES[self.speed]
        prRed(speed+' mode reduction starts')
        return

    @contextmanager
    def connection(self, server='aatlxa', username=None, password=None):
        """Make a secure connection to a remote server."""
        if not PYSFTP_AVAILABLE:
            print("You must install the pysftp package to do that!")
        if username is None:
            if self.aat_username is None:
                username = input('Enter AAT username: ')
                self.aat_username = username
            else:
                username = self.aat_username
        if password is None:
            if self.aat_password is None:
                password = getpass('Enter AAT password: ')
                self.aat_password = password
            else:
                password = self.aat_password
        try:
            srv = pysftp.Connection(server, username=username, password=password)
        except pysftp.paramiko.AuthenticationException:
            print('Authentication failed! Check username and password.')
            self.aat_username = None
            self.aat_password = None
            yield None
        else:
            try:
                yield srv
            finally:
                srv.close()

    def load_2dfdr_gui(self, fits_or_dirname):
        """Load the 2dfdr GUI in the required directory."""
        if isinstance(fits_or_dirname, FITSFile):
            # A FITS file has been provided, so go to its directory
            dirname = fits_or_dirname.reduced_dir
            idx_file = self.idx_files[fits_or_dirname.grating]
        else:
            # A directory name has been provided
            dirname = fits_or_dirname
            if dirname.find('ccd_') > 0:
                # Specify what idx file to use using the given dirname
                readccd = int(dirname[dirname.find('ccd_')+4:dirname.find('ccd_')+5])-1
                ngrating = ['580V','1000R','SPECTOR1','SPECTOR2']
                idx_file = IDX_FILES_FAST[ngrating[readccd]]
            else: 
                idx_file = None #Let the GUI sort out what idx file to use
        tdfdr.load_gui(dirname, idx_file=idx_file)
        return

    def find_directory_locks(self, lock_name='2dfdrLockDir'):
        """Return a list of directory locks that currently exist."""
        lock_list = []
        for dirname, subdirname_list, _ in os.walk(self.abs_root):
            if lock_name in subdirname_list:
                lock_list.append(os.path.join(dirname, lock_name))
        return lock_list

    def remove_directory_locks(self, lock_name='2dfdrLockDir'):
        """Remove all 2dfdr locks from directories."""
        for path in self.find_directory_locks(lock_name=lock_name):
            os.rmdir(path)
        return

    def list_checks(self, recent_ever='both', *args, **kwargs):
        """Return a list of checks that need to be done."""
        if 'do_not_use' not in kwargs:
            kwargs['do_not_use'] = False
        # Each element in the list will be a tuple, where
        # element[0] = key from below
        # element[1] = list of fits objects to be checked
        if recent_ever == 'both':
            complete_list = []
            complete_list.extend(self.list_checks('ever', *args, **kwargs))
            # Should ditch the duplicate checks, but will work anyway
            complete_list.extend(self.list_checks('recent', *args, **kwargs))
            return complete_list
        # The keys for the following defaultdict will be tuples, where
        # key[0] = 'TLM' (or similar)
        # key[1] = tuple according to CHECK_DATA group_by
        # key[2] = 'recent' or 'ever'
        check_dict = defaultdict(list)
        for fits in self.files(*args, **kwargs):
            if recent_ever == 'ever':
                items = fits.check_ever.items()
            elif recent_ever == 'recent':
                items = fits.check_recent.items()
            else:
                raise KeyError(
                    'recent_ever must be "both", "ever" or "recent"')

            # Iterate over checks which have been explicitly been marked as `False`
            for key in [key for key, value in items if value is False]:
                check_dict_key = []
                for attribute_to_group_by in CHECK_DATA[key]['group_by']:
                    check_dict_key.append(getattr(fits, attribute_to_group_by))
                check_dict_key = (key, tuple(check_dict_key), recent_ever)
                check_dict[check_dict_key].append(fits)
        # Now change the dictionary into a sorted list
        key_func = lambda item: CHECK_DATA[item[0][0]]['priority']
        check_list = sorted(check_dict.items(), key=key_func)
        return check_list

    def print_checks(self, *args, **kwargs):
        """Print the list of checks to be done."""
        check_list = self.list_checks(*args, **kwargs)
        for index, (key, fits_list) in enumerate(check_list):
            check_data = CHECK_DATA[key[0]]
            print('{}: {}'.format(index, check_data['name']))
            if key[2] == 'ever':
                print('Never been checked')
            else:
                print('Not checked since last re-reduction')
            for group_by_key, group_by_value in zip(
                    check_data['group_by'], key[1]):
                print('   {}: {}'.format(group_by_key, group_by_value))
            for fits in fits_list:
                print('      {}'.format(fits.filename))
        return

    def check_next_group(self, *args, **kwargs):
        """Perform required checks on the highest priority group."""
        if len(self.list_checks(*args, **kwargs)) == 0:
            print("Yay! no more checks to do.")
            return
        self.check_group(0, *args, **kwargs)

    def check_group(self, index, *args, **kwargs):
        """Perform required checks on the specified group."""
        try:
            key, fits_list = self.list_checks(*args, **kwargs)[index]
            print(fits_list)
        except IndexError:
            print("Check group '{}' does not exist.\n"
                  + "Try mngr.print_checks() for a list of "
                  + "available checks.").format(index)
            return
        check_method = getattr(self, 'check_' + key[0].lower())
        check_method(fits_list)
        print('Have you finished checking all the files? (y/n)')
        print('If yes, the check will be removed from the list.')
        y_n = input(' > ') + "n"
        finished = (y_n.lower()[0] == 'y')
        if finished:
            print('Removing this test from the list.')
            for fits in fits_list:
                fits.update_checks(key[0], True)
        else:
            print('Leaving this test in the list.')
        print('\nIf any files need to be disabled, use commands like:')
        print(">>> mngr.disable_files(['" + fits_list[0].filename + "'])")
        print('To add comments to a specifc file, use commands like:')
        print(">>> mngr.add_comment(['" + fits_list[0].filename + "'])")
        return

    def check_2dfdr(self, fits_list, message, filename_type='reduced_filename'):
        """Use 2dfdr to perform a check of some sort."""
        print('Use 2dfdr to plot the following files.')
        print('You may need to click on the triangles to see reduced files.')
        print('If the files are not listed, use the plot commands in the 2dfdr menu.')
        for fits in fits_list:
            print('   ' + getattr(fits, filename_type))
        print(message)
        self.load_2dfdr_gui(fits_list[0])
        return

    def check_bia(self, fits_list):
        """Check a set of bias frames."""
        # Check the individual bias frames, and then the combined file
        message = 'Check that the bias frames have no more artefacts than normal.'
        self.check_2dfdr(fits_list, message)
        combined_path = self.bias_combined_path(fits_list[0].ccd)
        if os.path.exists(combined_path):
            check_plots.check_bia(combined_path)
        return

    def check_drk(self, fits_list):
        """Check a set of dark frames."""
        # Check the individual dark frames, and then the combined file
        message = 'Check that the dark frames are free from any stray light.'
        self.check_2dfdr(fits_list, message)
        combined_path = self.dark_combined_path(fits_list[0].ccd,
                                                fits_list[0].exposure_str)
        if os.path.exists(combined_path):
            check_plots.check_drk(combined_path)
        return

    def check_lfl(self, fits_list):
        """Check a set of long-slit flats."""
        # Check the individual long-slit flats, and then the combined file
        message = 'Check that the long-slit flats have smooth illumination.'
        self.check_2dfdr(fits_list, message)
        combined_path = self.lflat_combined_path(fits_list[0].ccd)
        if os.path.exists(combined_path):
            check_plots.check_lfl(combined_path)
        return

    def check_tlm(self, fits_list):
        """Check a set of tramline maps."""
        message = 'Zoom in to check that the red fitted tramlines go through the data.'
        filename_type = 'tlm_filename'
        self.check_2dfdr(fits_list, message, filename_type)
        return

    def check_arc(self, fits_list):
        """Check a set of arc frames."""
        message = 'Zoom in to check that the arc lines are vertically aligned.'
        self.check_2dfdr(fits_list, message)
        return

    def check_flt(self, fits_list):
        """Check a set of flat field frames."""
        message = 'Check that the output varies smoothly with wavelength.'
        self.check_2dfdr(fits_list, message)
        return

    def check_sky(self, fits_list):
        """Check a set of offset sky (twilight) frames."""
        message = 'Check that the spectra are mostly smooth, and any features are aligned.'
        self.check_2dfdr(fits_list, message)
        return

    def check_obj(self, fits_list):
        """Check a set of reduced object frames."""
        message = 'Check that there is flux in each hexabundle, with no bad artefacts.'
        self.check_2dfdr(fits_list, message)
        return

    def check_flx(self, fits_list):
        """Check a set of spectrophotometric frames."""
        check_plots.check_flx(fits_list)
        return

    def check_tel(self, fits_list):
        """Check a set of telluric corrections."""
        check_plots.check_tel(fits_list)
        return

    def check_ali(self, fits_list):
        """Check the alignment of a set of object frames."""
        check_plots.check_ali(fits_list)
        return

    def check_cub(self, fits_list):
        """Check a set of final datacubes."""
        check_plots.check_cub(fits_list)
        return

    def _add_comment_to_file(self, fits_file_name, user_comment):
        """Add a comment to the FITS file corresponding to the name (with path)
        ``fits_file_name``.
        """

        try:
            hdulist = pf.open(fits_file_name, 'update',
                              do_not_scale_image_data=True)
            hdulist[0].header['COMMENT'] = user_comment
            hdulist.close()
        except IOError:
            return

    def add_comment(self, fits_list):
        """Add a comment to the FITS header of the files in ``fits_list``."""

        # Separate file names from vanilla names.
        # Run one thing on vanilla names
        # Run the other thing on file names.

        user_comment = input('Please enter a comment (type n to abort):\n')

        # If ``user_comment`` is equal to ``'n'``, skip updating the FITS
        # headers and jump to the ``return`` statement.
        if user_comment != 'n':

            time_stamp = 'Comment added by Hector Observer on '
            time_stamp += '{:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
            user_comment += ' (' + time_stamp + ')'

            fits_file_list, FITSFile_list = [], []
            for fits_file in fits_list:
                if os.path.isfile(fits_file):
                    fits_file_list.append(fits_file)
                elif isinstance(self.fits_file(fits_file), FITSFile):
                    FITSFile_list.append(self.fits_file(fits_file))
                else:
                    error_message = "'{}' must be a valid file name".format(
                        fits_file)
                    error_message += "Please use the full path for combined "
                    error_message += "products, and simple filenames for raw "
                    error_message += "files."

                    raise ValueError(error_message)

            # Add the comments to each FITSFile instance.
            list(map(FITSFile.add_header_item,
                     FITSFile_list,
                     ['COMMENT' for _ in FITSFile_list],
                     [user_comment for _ in FITSFile_list]))

            # Add the comments to each instance of pyfits.
            list(map(Manager._add_comment_to_file,
                     [self for _ in fits_file_list],
                     fits_file_list,
                     [user_comment for _ in fits_file_list]))

            comments_file = os.path.join(self.root, 'observer_comments.txt')
            with open(comments_file, "a") as infile:
                comments_list = [
                    '{}: '.format(fits_file) \
                    + user_comment + '\n'
                    for fits_file in FITSFile_list]
                comments_list += [
                    '{}: '.format(fits_file) \
                    + user_comment + '\n'
                    for fits_file in fits_file_list]
                infile.writelines(comments_list)

        return


class FITSFile:
    """Holds information about a FITS file to be copied."""

    def __init__(self, input_path):
        self.input_path = input_path
        self.source_path = os.path.realpath(input_path)
        self.filename = os.path.basename(self.source_path)
        self.filename_root = self.filename[:self.filename.rfind('.')]
        try:
            self.hdulist = pf.open(self.source_path)
        except IOError:
            self.ndf_class = None
            return
        self.header = self.hdulist[0].header
        self.set_ndf_class()
        self.set_reduced_filename()
        self.set_copy_reduced_filename()
        self.set_date()
        if self.ndf_class and self.ndf_class not in ['BIAS', 'DARK', 'LFLAT']:
            self.set_fibres_extno()
        else:
            self.fibres_extno = None
        self.set_coords()
        if self.ndf_class and self.ndf_class not in ['BIAS', 'DARK', 'LFLAT']:
            self.set_plate_id()
            self.set_plate_id_short()
            self.set_field_no()
            self.set_field_id()
        else:
            self.plate_id = None
            self.plate_id_short = None
            self.field_no = None
            self.field_id = None
        self.set_instrument()
        self.set_ccd()
        self.set_detector()
        self.set_grating()
        self.set_exposure()
        self.set_adc()
        self.set_epoch()
        # define the fluxlev property that is the 5-95th percentile range for the
        # raw frame.  This is a reasonable metric to use for assessing whether
        # twilight frames have sufficient flux:
        if self.ndf_class == 'MFSKY':
            flux = self.hdulist[0].data
            p05 = np.nanpercentile(flux, 5.0)
            p95 = np.nanpercentile(flux, 95.0)
            self.fluxlev = p95 - p05
            log.debug('%s: 5th,95th flux percentile: %s, %s, range:%s', self.filename, p05, p95, self.fluxlev)

        self.set_lamp()
        self.set_central_wavelength()
        self.set_do_not_use() #TODO: marie: this should be activated when SPECTOR has the keyword of 'SPEED' in their header
#        self.do_not_use = False #marie: this should be removed when SPECTOR has the keyword of 'SPEED' in their header

        self.set_coords_flags()
        self.set_copy()
        self.hdulist.close()
        del self.hdulist

    def __repr__(self):
        return "FITSFile {}, type: {}".format(self.filename, self.ndf_class)

    def set_ndf_class(self):
        """Save the NDF_CLASS of an AAT fits file."""
        # Change DFLAT to LFLAT

        #prYellow(f"Next {self.filename}")
        if self.header['NDFCLASS'] == 'DFLAT':
            self.header['NDFCLASS'] = 'LFLAT'

        # Ask the user if SFLAT should be changed to MFSKY
        if self.header['NDFCLASS'] == 'SFLAT':
            print('NDF_CLASS of SFLAT (OFFSET FLAT) found for ' + self.filename)
            print('Change to MFSKY (OFFSET SKY)? (y/n)')
            y_n = input(' > ')
            if y_n.lower()[0] == 'y':
                self.header['NDFCLASS'] = 'MFSKY'

        if 'NDFCLASS' in self.header:
            self.ndf_class = self.header['NDFCLASS']
        else:
            self.ndf_class = None

#        for hdu in self.hdulist:
#            if ('EXTNAME' in hdu.header.keys() and
#                    (hdu.header['EXTNAME'] == 'STRUCT.MORE.NDF_CLASS' or
#                     hdu.header['EXTNAME'] == 'NDF_CLASS')):
#                # It has a class
#                self.ndf_class = hdu.data['NAME'][0]
#                # Change DFLAT to LFLAT
#                if self.ndf_class == 'DFLAT':
#                    self.overwrite_ndf_class('LFLAT')
#                # Ask the user if SFLAT should be changed to MFSKY
#                if self.ndf_class == 'SFLAT':
#                    print('NDF_CLASS of SFLAT (OFFSET FLAT) found for ' +
#                          self.filename)
#                    print('Change to MFSKY (OFFSET SKY)? (y/n)')
#                    y_n = input(' > ')
#                    if y_n.lower()[0] == 'y':
#                        self.overwrite_ndf_class('MFSKY')
#                break
#        else:
#            self.ndf_class = None

    def set_reduced_filename(self):
        """Set the filename for the reduced file."""
        self.reduced_filename = self.filename_root + 'red.fits'
        # also set the intermediate im.fits filename as this can
        # be used for some processing steps, for example cross-matching
        # to get the best TLM offset:
        self.im_filename = self.filename_root + 'im.fits'
        if self.ndf_class == 'MFFFF':
            self.tlm_filename = self.filename_root + 'tlm.fits'
        # If the object is an MFSKY, then set the name of the
        # tlm_filename to be the copy of the MFSKY that is reduced
        # as a MFFFF:
        elif self.ndf_class == 'MFSKY':
            old_num = int(self.filename_root[6:10])
            key_num = 8
            new_num = old_num + 1000 * (key_num - (old_num // 1000))
            new_filename_root = (self.filename_root[:6] + '{:04d}'.format(int(new_num)) + self.filename_root[10:])
            self.tlm_filename = new_filename_root + 'tlm.fits'
            # If the file is a copy, then we'll also need to set the copy name as
            # the im_filename, as this is the one that will be looked for when
            # doing the TLM offset measurements:
            self.im_filename = new_filename_root + 'im.fits'

        elif self.ndf_class == 'MFOBJECT':
            self.fluxcal_filename = self.filename_root + 'fcal.fits'
            self.telluric_filename = self.filename_root + 'sci.fits'
        return

    def set_copy_reduced_filename(self):
        """Set the filename for the reduced version of a copied file.
        This will be numbered 06mar19001red.fits for fake flat
        and 06mar18001red.fits for fake twilight rather than 06mar10001red.fits"""
        key_num = 7
        if(self.ndf_class == 'MFFFF'):
            key_num = 9
        if(self.ndf_class == 'MFSKY'):
            key_num = 8
        self.copy_reduced_filename = self.filename_root + 'red.fits'
        old_num = int(self.filename_root[6:10])
        new_num = old_num + 1000 * (key_num - (old_num // 1000))
        new_filename_root = (self.filename_root[:6] + '{:04d}'.format(int(new_num)) + self.filename_root[10:])
        self.copy_reduced_filename = new_filename_root + 'red.fits'

        return

    def set_date(self):
        """Save the observation date as a 6-digit string yymmdd."""
        try:
            file_orig = self.header['FILEORIG']
            finish = file_orig.rfind('/', 0, file_orig.rfind('/'))
            self.date = file_orig[finish - 6:finish]
        except KeyError:
            self.date = None
        return

    def set_fibres_extno(self):
        """Save the extension number for the fibre table."""
        self.fibres_extno = find_fibre_table(self.hdulist)

    def set_plate_id(self):
        """Save the plate ID."""
        try:
            # First look in the primary header
            self.plate_id = self.header['PLATEID']
        except KeyError:
            # Check in the fibre table instead
            header = self.hdulist[self.fibres_extno].header
            self.plate_id = header['PLATEID']
            match = re.match(r'(^run_[0-9]+_)(P[0-9]+$)', self.plate_id) #This is for SAMI
            comment = 'Plate ID (from config file)'
            if match and 'star plate' in header['LABEL']:
                # This is a star field; adjust the plate ID accordingly
                self.plate_id = match.group(1) + 'star_' + match.group(2)
                comment = 'Plate ID (edited by manager)'
            # Also save it in the primary header, for future reference
            try:
                self.add_header_item('PLATEID', self.plate_id, comment,
                                     source=True)
                self.add_header_item('LABEL', header['LABEL'], 'LABEL (edited by manager)',
                                     source=True)
            except IOError:
                # This probably means we don't have write access to the
                # source file. Ideally we would still edit the copied file,
                # but that doesn't actually exist yet.
                pass
        if self.header['EPOCH'] > 2021. and self.plate_id == '1': # Hector commissioning data (Dec 2021 - Apr 2023) have PLATEID=1 
            # TODO: adjust epoch when the primary header has a proper PLATEID
            tile = self.hdulist[self.fibres_extno].header['FILENAME'] # the tile name is truncated in the primary header and get it from fibre table
            start = max(tile.find('Tile_FinalFormat_')+17,tile.find('Tile_')+5,0)
            end = np.array([tile.find('_CONFIGURED_correct_header.csv'),tile.find('.csv')])
            end = end[np.where(end > -1)]; end = min(end)
            self.plate_id = tile[start:end]
            try: # Save this to the primary header
                self.add_header_item('PLATEID', self.plate_id, 'Plate ID (edited by manager)',
                                     source=True)
                self.add_header_item('LABEL', '1', 'LABEL (edited by manager)',
                                     source=True)
            except IOError:
                pass
        if self.plate_id == '':
            self.plate_id = 'none'
        return

    def set_plate_id_short(self):
        """Save the shortened plate ID. For Hector, plate_id_short = plate_id"""
        finish = self.plate_id.find('_', self.plate_id.find('_') + 1)
        first_sections = self.plate_id[:finish]
        if self.plate_id == 'none':
            self.plate_id_short = 'none'
        elif (re.match(r'^Y[0-9]{2}S(A|B)R[0-9]+_P[0-9]+$', first_sections) or
              re.match(r'^A[0-9]+_P[0-9]+$', first_sections)):
            self.plate_id_short = first_sections
        else:
            self.plate_id_short = self.plate_id
        return

    def set_field_no(self):
        """Save the field number. For Hector, plate id is unique and therefore field id is not required and set to be 0"""
        if int(self.date) < 130101:
            # SAMIv1. Can only get the field number by cross-checking the
            # config file RA and Dec.
            for pilot_field in PILOT_FIELD_LIST:
                cfg_coords = ICRS(
                    ra=self.cfg_coords['ra'],
                    dec=self.cfg_coords['dec'],
                    unit=self.cfg_coords['unit'])
                if (cfg_coords.separation(
                        ICRS(pilot_field['coords'])).arcsecs < 1.0
                        and self.plate_id == pilot_field['plate_id']):
                    self.field_no = pilot_field['field_no']
                    break
            else:
                raise RuntimeError('Field is from pilot data but cannot find'
                                   ' it in list of known pilot fields: ' +
                                   self.filename)
        else:
            # SAMIv2 and Hector. Should be in the fibre table header somewhere
            header = self.hdulist[self.fibres_extno].header
            # First, see if it's included in the LABEL keyword
            match = re.search(r'(field )([0-9]+)', header['LABEL'])
            if match:
                # Yes, it is
                self.field_no = int(match.group(2))
            else:
                # The field number should be included in the filename of
                # the config file.
                match = re.search(r'(.*_f)([0-9]+)', header['FILENAME'])
                if match:
                    self.field_no = int(match.group(2))
                else:
                    # Nothing found. Default to 0.
                    self.field_no = 0
        return

    def set_field_id(self):
        """Save the field ID."""
        if self.plate_id == 'none':
            self.field_id = 'none'
        else:
            # First check if the LABEL keyword is of the correct form
            expr = (r'(.*?)( - )(Run [0-9]+ .*? plate [0-9]+)'
                    r'( - )(field [0-9]+)')
            header = self.hdulist[self.fibres_extno].header
            match = re.match(expr, header['LABEL'])
            if match:
                # It is, so just copy the field ID directly
                self.field_id = match.group(1)
            elif (self.plate_id.startswith('run_') or
                  re.match(r'[0-9]+S[0-9]+', self.plate_id)):
                # Pilot and commissioning data. No field ID in the plate ID, so
                # append one.
                self.field_id = self.plate_id + '_F' + str(self.field_no)
            elif (re.match(r'^Y[0-9]{2}S(A|B)R[0-9]+_P[0-9]+$',
                           self.plate_id_short) or
                  re.match(r'^A[0-9]+_P[0-9]+$',
                           self.plate_id_short)):
                # Main survey or early cluster data. Field ID is stored within the 
                # plate ID.
                start = len(self.plate_id_short)
                for i in range(self.field_no):
                    start = self.plate_id.find('_', start) + 1
                finish = self.plate_id.find('_', start)
                if finish == -1:
                    field_id = self.plate_id[start:]
                else:
                    field_id = self.plate_id[start:finish]
                self.field_id = self.plate_id_short + '_' + field_id
            elif re.match(r'^A[0-9]+T[0-9]+_A[0-9]+T[0-9]+$', self.plate_id):
                # Cluster data. Field ID is one segment of the plate ID.
                start = 0
                for i in range(self.field_no - 1):
                    start = self.plate_id.find('_', start) + 1
                finish = self.plate_id.find('_', start)
                if finish == -1:
                    field_id = self.plate_id[start:]
                else:
                    field_id = self.plate_id[start:finish]
                self.field_id = field_id
            else:
                # Unrecognised form for the plate ID. That is for Hector
                self.field_id = self.plate_id + '_F' + str(self.field_no)
        return

    def set_coords(self):
        """Save the RA/Dec and config RA/Dec."""
        if self.ndf_class and self.ndf_class not in ['BIAS', 'DARK', 'LFLAT']:
            header = self.hdulist[self.fibres_extno].header
            self.cfg_coords = {'ra': header['CFGCENRA'],
                               'dec': header['CFGCENDE'],
                               'unit': (units.radian, units.radian)}
            if self.ndf_class == 'MFOBJECT':
                self.coords = {'ra': header['CENRA'],
                               'dec': header['CENDEC'],
                               'unit': (units.radian, units.radian)}
            else:
                self.coords = None
        else:
            self.cfg_coords = None
        return

    def set_detector(self):
        """Set the specific detector name, e.g. E2V2A etc.  This is different from
        the ccd name as ccd is just whether ccd_1 (blue) or ccd_2 (red).  We need
        to know which detector as some reduction steps can be different, e.g. treatment
        of bias and dark frames."""
        if self.ndf_class:
            detector_id = self.header['DETECTOR']
            self.detector = detector_id
        else:
            self.detector = None
        return

    def set_instrument(self):
        """Set the instrument name, either Spector or AAOmega"""
        if self.ndf_class:
            instrument = self.header['INSTRUME']
            if instrument == 'AAOMEGA-SAMI':
                self.instrument = 'AAOMEGA-SAMI'
            elif instrument == 'AAOMEGA-HECTOR':
                self.instrument = 'AAOMEGA-HECTOR'
            elif instrument == 'SPECTOR':
                self.instrument = 'SPECTOR-HECTOR'
            else:
                self.instrument = 'unknown_instrument'
        else:
            self.instrument = None
        return
            
    def set_ccd(self):
        """Set the CCD name."""
        if (self.ndf_class and (self.instrument is not None)
            and (self.instrument != 'unknown_instrument')):
            spect_id = self.header['SPECTID']
            if (spect_id == 'BL') and ((self.instrument == 'AAOMEGA-SAMI') or
                                        (self.instrument == 'AAOMEGA-HECTOR')):
                self.ccd = 'ccd_1'
            elif (spect_id == 'RD') and ((self.instrument == 'AAOMEGA-SAMI') or
                                        (self.instrument == 'AAOMEGA-HECTOR')):
                self.ccd = 'ccd_2'
            elif (spect_id == 'BL') and (self.instrument == 'SPECTOR-HECTOR'):
                self.ccd = 'ccd_3'
            elif (spect_id == 'RD') and (self.instrument == 'SPECTOR-HECTOR'):
                self.ccd = 'ccd_4'
            else:
                self.ccd = 'unknown_ccd'
        else:
            self.ccd = None
        return

    def set_grating(self):
        """Set the grating name."""
        if self.ndf_class:
            self.grating = self.header['GRATID']
        else:
            self.grating = None
        if (self.grating == 3) or (self.grating == 'VPH-1099-484'):
            self.grating = 'SPECTOR1'
        if (self.grating == 4) or (self.grating == 'VPH-1178-679'):
            self.grating = 'SPECTOR2'

        return

    def set_exposure(self):
        """Set the exposure time."""
        if self.ndf_class:
            self.exposure = self.header['EXPOSED']
            self.exposure_str = '{:d}'.format(int(np.round(self.exposure)))
        else:
            self.exposure = None
            self.exposure_str = None
        return

    def set_adc(self):
        """ADC status"""
        try:
            self.adc = self.header['ADCSTATS'].strip()
        except KeyError:
            self.adc = None
        return

    def set_epoch(self):
        """Set the observation epoch."""
        if self.ndf_class:
            self.epoch = self.header['EPOCH']
        else:
            self.epoch = None
        return

    def set_lamp(self):
        """Set which lamp was on, if any."""
        if self.ndf_class == 'MFARC':
            lamp = self.header['LAMPNAME']
            if lamp == '':
                # No lamp set. Most likely that the CuAr lamp was on but the
                # control software screwed up. Patch the headers assuming this
                # is the case, but warn the user.
                lamp = 'CuAr'
                hdulist_write = pf.open(self.source_path, 'update')
                hdulist_write[0].header['LAMPNAME'] = lamp
                hdulist_write[0].header['OBJECT'] = 'ARC - ' + lamp
                hdulist_write.flush()
                hdulist_write.close()
                print('No arc lamp specified for ' + self.filename)
                print('Updating LAMPNAME and OBJECT keywords assuming a ' +
                      lamp + ' lamp')
            self.lamp = lamp
        elif self.ndf_class == 'MFFFF':
            if self.exposure >= 17.5:
                # Assume longer exposures are dome flats
                self.lamp = 'Dome'
            else:
                self.lamp = 'Flap'
        else:
            self.lamp = None
        return

    def set_central_wavelength(self):
        """Set what the requested central wavelength of the observation was."""
        if self.ndf_class:
            if 'LAMBDCR' in self.header:
                self.central_wavelength = self.header['LAMBDCR']
            else: # spector (ccd3 & ccd4) have different keyward for the central wavelength
                self.central_wavelength = self.header['LAMBDAC']        
        else:
            self.central_wavelength = None
        return

    def set_do_not_use(self):
        """Set whether or not to use this file."""
        try:
            self.do_not_use = self.header['DONOTUSE']
        except KeyError:
            # By default, don't use fast readout files
            if self.header['INSTRUME'] == 'SPECTOR':
                self.do_not_use = (self.header['SPEED'] != 'MEDIUM')
            else:
                self.do_not_use = (self.header['SPEED'] != 'NORMAL')
        return

    def set_coords_flags(self):
        """Set whether coordinate corrections have been done."""
        try:
            self.coord_rot = self.header['COORDROT']
        except KeyError:
            self.coord_rot = None
        try:
            self.coord_rev = self.header['COORDREV']
        except KeyError:
            self.coord_rev = None
        return

    def set_copy(self):
        """Set whether this is a copy of a file."""
        try:
            self.copy = self.header['MNGRCOPY']
        except KeyError:
            self.copy = False
        return

    def relevant_check(self, check):
        """Return True if a visual check is relevant for this file."""
        return (self.ndf_class == check['ndf_class'] and
                (check['spectrophotometric'] is None or
                 check['spectrophotometric'] == self.spectrophotometric))

    def set_check_data(self):
        """Set whether the relevant checks have been done."""
        self.check_ever = {}
        self.check_recent = {}
        for key in [key for key, check in CHECK_DATA.items()
                    if self.relevant_check(check)]:
            try:
                check_done_ever = self.header['MNCH' + key]
            except KeyError:
                check_done_ever = None
            self.check_ever[key] = check_done_ever
            try:
                check_done_recent = self.header['MNCH' + key + 'R']
            except KeyError:
                check_done_recent = None
            self.check_recent[key] = check_done_recent
        return

    def make_reduced_link(self):
        """Make the link in the reduced directory."""
        if not os.path.exists(self.reduced_dir):
            os.makedirs(self.reduced_dir)
        if not os.path.exists(self.reduced_link):
            os.symlink(os.path.relpath(self.raw_path, self.reduced_dir),
                       self.reduced_link)
        return

    def reduce_options(self):
        """Return a dictionary of options used to reduce the file."""
        if not os.path.exists(self.reduced_path):
            return None
        with pf.open(self.reduced_path) as hdul:
            # Check if 'REDUCTION_ARGS' extension exists
            if 'REDUCTION_ARGS' in [hdu.name for hdu in hdul]:
                #print(self.reduced_path)
                return dict(pf.getdata(self.reduced_path, 'REDUCTION_ARGS'))
            else:
                print('No REDUCTION_ARGS extensions: ',self.reduced_path) #activate this for the error, KeyError: "Extension ('REDUCTION_ARGS', 1) not found."
                return None

    def update_name(self, name):
        """Change the object name assigned to this file."""
        if self.name != name:
            if re.match(r'.*[\\\[\]*/?<>|;:&,.$ ].*', name):
                raise ValueError(r'Invalid character in name; '
                                 r'do not use any of []\/*?<>|;:&,.$ or space')
            # Update the FITS header
            self.add_header_item('MNGRNAME', name,
                                 'Object name set by Hector manager')
            # Update the object
            self.name = name
        return

    def update_spectrophotometric(self, spectrophotometric):
        """Change the spectrophotometric flag assigned to this file."""
        if self.spectrophotometric != spectrophotometric:
            # Update the FITS header
            self.add_header_item('MNGRSPMS', spectrophotometric,
                                 'Flag set if a spectrophotometric star')
            # Update the object
            self.spectrophotometric = spectrophotometric
        return

    def update_do_not_use(self, do_not_use):
        """Change the do_not_use flag assigned to this file."""
        if self.do_not_use != do_not_use:
            # Update the FITS header
            self.add_header_item('DONOTUSE', do_not_use,
                                 'Do Not Use flag for Hector manager')
            # Update the object
            self.do_not_use = do_not_use
            # Update the file system
            if do_not_use:
                if os.path.exists(self.reduced_link):
                    os.remove(self.reduced_link)
            else:
                self.make_reduced_link()
        return

    def update_checks(self, key, value, force=False):
        """Update both check flags for this key for this file."""
        self.update_check_recent(key, value)
        # Don't set the "ever" check to False unless forced, or there
        # is no value set yet
        if value or force or self.check_ever[key] is None:
            self.update_check_ever(key, value)
        return

    def update_check_recent(self, key, value):
        """Change one of the check flags assigned to this file."""
        if self.check_recent[key] != value:
            # Update the FITS header
            self.add_header_item('MNCH' + key + 'R', value,
                                 CHECK_DATA[key]['name'] +
                                 ' checked since re-reduction')
            # Update the object
            self.check_recent[key] = value
        return

    def update_check_ever(self, key, value):
        """Change one of the check flags assigned to this file."""
        if self.check_ever[key] != value:
            # Update the FITS header
            self.add_header_item('MNCH' + key, value,
                                 CHECK_DATA[key]['name'] +
                                 ' checked ever')
            # Update the object
            self.check_ever[key] = value
        return

    def add_header_item(self, key, value, comment=None, source=False):
        """Add a header item to the FITS file."""
        if comment is None:
            value_comment = value
        else:
            value_comment = (value, comment)
        if source:
            path = self.source_path
        else:
            path = self.raw_path
        # old_header = pf.getheader(path)
        old_header = self.header
        # Only update if necessary
        if (key not in self.header or
                self.header[key] != value or
                type(self.header[key]) != type(value) or
                (comment is not None and self.header.comments[key] != comment)):
            with pf.open(path, 'update', do_not_scale_image_data=True) as hdulist:
                hdulist[0].header[key] = value_comment
                self.header = hdulist[0].header
        return

#    def overwrite_ndf_class(self, new_ndf_class):
#        """Change the NDF_CLASS value in the FITS file and in the object."""
#        hdulist_write = pf.open(self.source_path, 'update')
#        for hdu_name in ('STRUCT.MORE.NDF_CLASS', 'NDF_CLASS'):
#            try:
#                hdu = hdulist_write[hdu_name]
#                break
#            except KeyError:
#                pass
#        else:
#            # No relevant extension found
#            raise KeyError('No NDF_CLASS extension found in file')
#        hdu.data['NAME'][0] = new_ndf_class
#        hdulist_write.flush()
#        hdulist_write.close()
#        self.ndf_class = new_ndf_class
#        return

    def has_sky_lines(self):
        """Return True if there are sky lines in the wavelength range."""
        # Coverage taken from http://ftp.aao.gov.au/2df/aaomega/aaomega_gratings.html
        # add coverage for Spector (SMC 17/08/22) 
        coverage_dict = {
            '1500V': 750,
            '580V': 2100,
            '1000R': 1100,
            'SPECTOR1': 2200,
            'SPECTOR2': 2100,
        }
        coverage = coverage_dict[self.grating]
        wavelength_range = (
             self.central_wavelength - 0.5 * coverage, #marie
             self.central_wavelength + 0.5 * coverage #marie
        )
#        wavelength_range = (
#            self.header['LAMBDCR'] - 0.5 * coverage, #marie: remove this when it is sure for the below modification
#            self.header['LAMBDCR'] + 0.5 * coverage #marie
#        )
        # Highly incomplete list! May give incorrect results for high-res
        # red gratings
        useful_lines = (5577.338, 6300.309, 6553.626, 6949.066, 7401.862,
                        7889.680, 8382.402, 8867.605, 9337.874, 9799.827, 9972.357)
        for line in useful_lines:
            if wavelength_range[0] < line < wavelength_range[1]:
                # This sky line is within the observed wavelength range
                return True
        # No sky lines were within the observed wavelength range
        return False

def update_checks(key, file_iterable, value, force=False):
    """Set flags for whether the files have been manually checked."""
    for fits in file_iterable:
        fits.update_checks(key, value, force)
    return


def safe_for_multiprocessing(function):
    @wraps(function)
    def safe_function(*args, **kwargs):
        try:
            result = function(*args, **kwargs)
        except KeyboardInterrupt:
            print("Handling KeyboardInterrupt in worker process")
            print("You many need to press Ctrl-C multiple times")
            result = None
        return result

    return safe_function


@safe_for_multiprocessing
def derive_transfer_function_pair(inputs):
    """Derive transfer function for a pair of fits files."""
    path_pair = inputs['path_pair']
    n_trim = inputs['n_trim']
    model_name = inputs['model_name']
    smooth = inputs['smooth']

    try:
        fluxcal2.derive_transfer_function(
            path_pair, n_trim=n_trim, model_name=model_name, smooth=smooth,
            molecfit_available = MOLECFIT_AVAILABLE, molecfit_dir = MF_BIN_DIR,
            speed=inputs['speed'],tell_corr_primary=inputs['tellcorprim'])
    except ValueError:
        print('  Warning: No star found in dataframe, skipping ' +
              os.path.basename(path_pair[0]))
        return
    good_psf = pf.getval(path_pair[0], 'GOODPSF', 'FLUX_CALIBRATION')
    if not good_psf:
        print('  Warning: Bad PSF fit in ' + os.path.basename(path_pair[0]) +
              '; will skip this one in combining.')
    return


@safe_for_multiprocessing
def telluric_correct_pair(inputs):
    """Telluric correct a pair of fits files."""
    fits_1 = inputs['fits_1']
    fits_2 = inputs['fits_2']
    n_trim = inputs['n_trim']
    use_PS = inputs['use_PS']
    scale_PS_by_airmass = inputs['scale_PS_by_airmass']
    PS_spec_file = inputs['PS_spec_file']
    model_name = inputs['model_name']
    use_probe = None

    if fits_1 is None or not os.path.exists(fits_1.fluxcal_path):
        print('Matching blue arm not found for ' + fits_2.filename +
              '; skipping this file.')
        return
    path_pair = (fits_1.fluxcal_path, fits_2.fluxcal_path)
    print('Deriving telluric correction for ' + fits_1.filename +
          ' and ' + fits_2.filename)
    try:
        prCyan("The inputs to telluric.derive_transfer_function:")
        debug = (fits_1.epoch < 2025.) or (fits_1.instrument != 'AAOMEGA-HECTOR') #TODO:Sree (may2025): make debug=True once H bundle is fixed
        if (fits_1.epoch > 2025.5) and (fits_1.instrument == 'AAOMEGA-HECTOR'):
            use_probe = 'G'  #TODO: Sree (July2025): Bundle 'G' will be used for secondary standard stars from the 17 July 2025 run, until bundle 'H' is recovered.
            debug = True
        
        print(path_pair,PS_spec_file,use_PS,n_trim,scale_PS_by_airmass,model_name,MOLECFIT_AVAILABLE, MF_BIN_DIR, debug, use_probe)
        telluric.derive_transfer_function(
            path_pair, PS_spec_file=PS_spec_file, use_PS=use_PS, n_trim=n_trim,
            scale_PS_by_airmass=scale_PS_by_airmass, model_name=model_name, use_probe=use_probe,
            molecfit_available = MOLECFIT_AVAILABLE, molecfit_dir = MF_BIN_DIR,speed=inputs['speed'],debug=debug)
    except ValueError as err:
        if err.args[0].startswith('No star identified in file:'):
            # No standard star found; probably a star field
            print(err.args[0])
            print('Skipping telluric correction for file:', fits_2.filename)
            return False
        else:
            # Some other, unexpected error. Re-raise it.
            raise err
    for fits in (fits_1, fits_2):
        print('Telluric correcting file:', fits.filename)
        if os.path.exists(fits.telluric_path):
            os.remove(fits.telluric_path)
        telluric.apply_correction(fits.fluxcal_path,
                                  fits.telluric_path)
    return True


@safe_for_multiprocessing
def measure_offsets_group(group):
    """Measure offsets between a set of dithered observations."""
    # MLPG - 23/09/2023: Adding "do_cvd_correct=True" to find_dither function
    field, fits_list, copy_to_other_arm, fits_list_other_arm = group
    print('Measuring offsets for field ID: {}'.format(field[0]))
    path_list = [best_path(fits) for fits in fits_list]
    print('These are the files:')
    for path in path_list:
        print('  ', os.path.basename(path))
    if len(path_list) < 2:
        # Can't measure offsets for a single file
        print('Only one file so no offsets to measure!')
        return

    prLightPurple(f"PlateCentre in CvD modelling is assumed to be at (0.0, 0.0), i.e. at phyiscal centre given by robot coordinates")
    find_dither(path_list, path_list[0], centroid=True,
                remove_files=True, do_dar_correct=False, do_cvd_correct=True)

    if copy_to_other_arm:
        for fits, fits_other_arm in zip(fits_list, fits_list_other_arm):
            hdulist_this_arm = pf.open(best_path(fits))
            hdulist_other_arm = pf.open(best_path(fits_other_arm), 'update')
            try:
                del hdulist_other_arm['ALIGNMENT']
            except KeyError:
                # Nothing to delete; no worries
                pass
            hdulist_other_arm.append(hdulist_this_arm['ALIGNMENT'])
            hdulist_other_arm.flush()
            hdulist_other_arm.close()
            hdulist_this_arm.close()
    return


@safe_for_multiprocessing
def cube_group(group):
    """Cube a set of RSS files."""
    field, fits_list, root, overwrite, star_only = group
    print('Cubing field ID: {}, CCD: {}'.format(field[0], field[1]))
    path_list = [best_path(fits) for fits in fits_list]
    print('These are the files:')
    for path in path_list:
        print('  ', os.path.basename(path))
    if star_only:
        objects = [pf.getval(path_list[0], 'STDNAME', 'FLUX_CALIBRATION')]
    else:
        objects = 'all'
    if fits_list[0].epoch < 2013.0:
        # Large pitch of pilot data requires a larger drop size
        drop_factor = 0.75
    else:
        drop_factor = 0.5
    dithered_cubes_from_rss_list(
        path_list, suffix='_' + field[0], size_of_grid=50, write=True,
        nominal=True, root=root, overwrite=overwrite, do_dar_correct=True,
        objects=objects, clip=False, do_clip_by_fibre=True, drop_factor=drop_factor)
    return


@safe_for_multiprocessing
def cube_object(inputs):
    """Cube a single object in a set of RSS files."""
    (field_id, ccd, path_list, name, cubed_root, drop_factor, tag,
     update_tol, size_of_grid, output_pix_size_arcsec, clip_throughput, ndither, overwrite) = inputs
    print('\nCubing {} in field ID: {}, CCD: {}'.format(name, field_id, ccd))
    #if(len(path_list)>28): #when generate cubes using a spericifed number of dithers
    #        path_list = path_list[:7]
    if ndither:
        path_list = path_list[:ndither]
    print('{} files available'.format(len(path_list)))
    suffix = '_' + field_id
    if tag:
        suffix += '_' + tag
    return dithered_cube_from_rss_wrapper(
        path_list, name, suffix=suffix, write=True, nominal=True,
        root=cubed_root, overwrite=overwrite, do_dar_correct=False, do_cvd_correct=True, clip=False,
        do_clip_by_fibre=True, drop_factor=drop_factor, update_tol=update_tol,
        size_of_grid=size_of_grid, clip_throughput=clip_throughput,
        output_pix_size_arcsec=output_pix_size_arcsec, plateCentre=None)

def best_path(fits):
    """Return the best (most calibrated) path for the given file."""
    if os.path.exists(fits.telluric_path):
        path = fits.telluric_path
    elif os.path.exists(fits.fluxcal_path):
        path = fits.fluxcal_path
    else:
        path = fits.reduced_path
    return path


@safe_for_multiprocessing
def run_2dfdr_single_wrapper(group):
    """Run 2dfdr on a single file."""
    fits, idx_file, options, dummy, root = \
        group
    try:
        tdfdr.run_2dfdr_single(fits, idx_file, root=root, options=options, dummy=dummy)
    except tdfdr.LockException:
        message = ('Postponing ' + fits.filename +
                   ' while other process has directory lock.')
        print(message)
        return False
    return True

@safe_for_multiprocessing
def fit_arc_model_wrapper(input_list):
    """fit 2d arc modelling"""
    arc_reduced, arcfit_name, tlm_name, Nx, Ny, global_fit = input_list
    param_filename = Path(arc_reduced).parent / (Path(arc_reduced).stem + "_2dfit_params.nc")
    param_filename = None 
        # To overwrite the arc completely, uncomment this line
    print(' Performing a 2D wavelength fit of the arc frames.',arc_reduced)
    arc_model_2d(arc_reduced, arcfit_name, tlm_name, N_x=Nx, N_y=Ny, save_params=param_filename,global_fit=global_fit, verbose=False)
    return


@safe_for_multiprocessing
def scale_cubes_field(group):
    """Scale a field to the correct magnitude."""
    star_path_pair, object_path_pair_list, star = group
    print('Scaling field with star', star)
    stellar_mags_cube_pair(star_path_pair, save=True)
    # Copy the PSF data to the galaxy datacubes
    star_header = pf.getheader(star_path_pair[0])
    for object_path_pair in object_path_pair_list:
        for object_path in object_path_pair:
            with pf.open(object_path,mode='update') as hdulist_write:
                for key in ('PSFFWHM', 'PSFALPHA', 'PSFBETA'):
                    try:
                        hdulist_write[0].header[key] = star_header[key]
                    except KeyError:
                        pass
                hdulist_write.flush()
                hdulist_write.close()
    # Previously tried reading the catalogue once and passing it, but for
    # unknown reasons that was corrupting the data when run on aatmacb.

    # Temporarily commenting out the below step to scale the cube fluxes. Now
    # only determines and writes the PSF info to the cube.

    #found = assign_true_mag(star_path_pair, star, catalogue=None)
    #if found:
    #    scale = scale_cube_pair_to_mag(star_path_pair, 3)
    #    for object_path_pair in object_path_pair_list:
    #        scale_cube_pair(object_path_pair, scale)
    #else:
    #    print('No photometric data found for', star)
    return


@safe_for_multiprocessing
def fit_sec_template(path):
    """Fit theoretical templates to secondary calibration stars that have been
    selected to be halo F-stars.  This uses ppxf and save the best template and
    weight to the fits header."""
    # MLPG: setting dplot=True and turning on verbose

    # call the main template fitting routine for the given file:
    fluxcal2.fit_sec_template_ppxf(path) # , doplot=True,verbose=True)
    
    
    return

@safe_for_multiprocessing
def scale_frame_pair(inputs):
    """Scale a pair of RSS frames to the correct magnitude."""

    # get inputs:
    path_pair = inputs['path_pair']
    apply_scale = inputs['apply_scale']
    if (apply_scale):
        print('Scaling RSS files to give star correct magnitude: %s' %
          str((os.path.basename(path_pair[0]), os.path.basename(path_pair[1]))))
    else:
        print('Calculating scaling for RSS files to give star correct magnitude, but NOT applying: %s' %
          str((os.path.basename(path_pair[0]), os.path.basename(path_pair[1]))))
    print(path_pair)
        
    stellar_mags_frame_pair(path_pair, save=True)
    star = pf.getval(path_pair[0], 'STDNAME', 'FLUX_CALIBRATION')
    # Previously tried reading the catalogue once and passing it, but for
    # unknown reasons that was corrupting the data when run on aatmacb.
    found = assign_true_mag(path_pair, star, catalogue=None,
                            hdu='FLUX_CALIBRATION')
    if found:
        scale_cube_pair_to_mag(path_pair, 1, hdu='FLUX_CALIBRATION',band='g',apply_scale=apply_scale)
    else:
        print('No photometric data found for', star)
    return


@safe_for_multiprocessing
def bin_cubes_pair(path_pair):
    """Bin a pair of datacubes using each of the default schemes."""
    # TODO: Allow the user to specify name/kwargs pairs. Will require
    # coordination with Manager.bin_cubes() [JTA 23/9/2015]
    path_blue, path_red = path_pair
    print('Binning datacubes:')
    print(os.path.basename(path_blue), os.path.basename(path_red))
    binning_settings = (
        ('adaptive', {'mode': 'adaptive'}),
        ('annular', {'mode': 'prescriptive', 'sectors': 1}),
        ('sectors', {'mode': 'prescriptive'}))
    for name, kwargs in binning_settings:
        binning.bin_cube_pair(path_blue, path_red, name=name, **kwargs)
    return


@safe_for_multiprocessing
def aperture_spectra_pair(input_list, overwrite=False):
    """Create aperture spectra for a pair of data cubes using default apertures."""
    path_pair,overwrite = input_list
    path_blue, path_red = path_pair
    global CATALOG_PATH

    try:
        print('Processing: ' + path_blue + ', ' + path_red)
        binning.aperture_spectra_pair(path_blue, path_red, CATALOG_PATH, overwrite)
    except Exception as e:
        print("ERROR on pair %s, %s:\n %s" % (path_blue, path_red, e))
        traceback.print_exc()
    return

@safe_for_multiprocessing
def ungzip_wrapper(path):
    """Gzip a single file."""
    print('Ungzipping file: ' + path)
    ungzip(path)
    return

@safe_for_multiprocessing
def gzip_wrapper(path):
    """Gzip a single file."""
    print('Gzipping file: ' + path)
    gzip(path)
    return


# @safe_for_multiprocessing
# def test_function(variable):
#     import time
#     print("starting", variable)
#     start_time = time.time()
#     current_time = time.time()
#     while current_time < (start_time + 5):
#         print("waiting...")
#         time.sleep(1); current_time = time.time()
#     print("finishing", variable)

def assign_true_mag(path_pair, name, catalogue=None, hdu=0):
    """Find the magnitudes in a catalogue and save them to the header."""
    # MLPG: added a warning to indicate the cases where the name of the standard star
    # is not in the catalogue
    if catalogue is None:
        catalogue = read_stellar_mags()
        # in some cases the catalogue keys can end up at bytes, rather than as strings.
        # this is due to a numpy bug that is fixed in later versions.  It is certainly
        # fixed by numpy vesion 1.14.2.
        # this bug can lead to a failure of matching in the lines below (matched against "name").
    if name not in catalogue:
        prRed(f"WARNING: {name} is not in the catalogue")
        return False
    line = catalogue[name]
    print(name, line)
    print(line['u'],line['g'],line['r'],line['i'],line['z'])
    for path in path_pair:
        hdulist = pf.open(path, 'update')
        for band in 'ugriz':
            hdulist[hdu].header['CATMAG' + band.upper()] = (
                line[band], band + ' mag from catalogue')
        hdulist.flush()
        hdulist.close()
    return True

def read_hector_tiles(abs_root=None):
    """ reads in a hector tiling file, and extracts the magnitude information """
    # MLPG: A new function added to extract the magnitude information from tiling files.
    # From June 2023, Tile files are stored in the raw directory, which are automatically copied into
    # hector/standards/secondary/Hector_tiles/.
    # The user may be requried to download the hector tile files before June 2023 from the data central could
    # /Hector/DR/DR_pipeline_resources/tile to the "Hector_tiles" folder.
    # This function is automated, being called by fluxcal_secondary().
    # How to manually run: import hector
    #                      hector.manager.read_hector_tiles()
    # MLPG - 24/09/2023: Changed the tile/robot file location to "hector/Tiles/"
    # TODO: Sree: do we want to place Hector_tiles directory in the hector git folder??
    from pathlib import Path

    headerList = ['#probe', 'ID', 'x', 'y', 'rads', 'angs', 'azAngs', 'angs_azAng', 'RA', 'DEC', 'g_mag', 'r_mag',
                  'i_mag', 'z_mag', 'y_mag', 'GAIA_g_mag', 'GAIA_bp_mag', 'GAIA_rp_mag', 'Mstar', 'Re', 'z',
                  'GAL_MU_E_R', 'pmRA', 'pmDEC', 'priority', 'MagnetX_noDC', 'MagnetY_noDC', 'type', 'MagnetX',
                  'MagnetY', 'SkyPosition', 'fibre_type', 'Magnet', 'Label', 'order', 'Pickup_option', 'Index',
                  'Hexabundle', 'probe_orientation', 'rectMag_inputOrientation', 'Magnet_C', 'Label_C', 'order_C',
                  'Pickup_option_C', 'offset_P', 'offset_Q']
    headerNew = ['ID', 'x', 'y', 'rads', 'angs', 'azAngs', 'angs_azAng', 'RA', 'DEC', 'u_mag', 'g_mag', 'r_mag',
                 'i_mag', 'z_mag', 'GAIA_g_mag', 'GAIA_bp_mag', 'GAIA_rp_mag']


    # setup file paths
    tile_path = Path(hector_path) / f"Tiles/Tile_files"
    robot_path = Path(hector_path) / f"Tiles/Robot_files"
    base_path = Path(hector_path) / f"standards/secondary/Hector_tiles"
    file_names = ["Hector_tiles.csv", "Hector_secondary_standards.csv", "Hector_secondary_standards_shortened.csv"]

    # Check if the directory exists, if not create
    if not os.path.isdir(base_path):
        os.makedirs(base_path)
    if not os.path.isdir(tile_path):
        os.makedirs(tile_path)
    if not os.path.isdir(robot_path):
        os.makedirs(robot_path)

    # grab new Hector tiles from the raw folder
    update = False
    if abs_root is not None:
        tile_dir = abs_root + f'/raw/'
        for root, dirs, files in os.walk(tile_dir):
            for file in files:
                if (('Tile' in file) and (file[-3:] == 'csv')):
                    src_path = os.path.join(root, file)
                    dest_tile_path = os.path.join(tile_path, file)
                    dest_robot_path = os.path.join(robot_path, file.replace('Tile', 'Robot'))
                    if not os.path.exists(dest_tile_path):
                        update = True
                        shutil.copy(src_path, dest_tile_path)
                        shutil.copy(src_path.replace('Tile', 'Robot'), dest_robot_path)


    # Check if the files holding the tile list and secondary standards exists. If not, create.
#    if not os.path.exists(f"{base_path}/{file_names[0]}"):
    if update:
        with open(f"{base_path}/{file_names[0]}", 'w') as filelist:
            filelist.write("Hector_Tile_List \n")  # create hector-tile-list file

        with open(f"{base_path}/{file_names[1]}", 'w') as filelist:
            dw = csv.DictWriter(filelist, delimiter=',', fieldnames=headerList)
            dw.writeheader() # create hector standard star list file. This contains all information in the tile file
            del dw
        with open(f"{base_path}/{file_names[2]}", 'w') as filelist:
            dw = csv.DictWriter(filelist, delimiter=',', fieldnames=headerNew)
            dw.writeheader() # create hector standard star list file. This contains a subset of information in the tile file
            del dw

    prRed(f"Please ensure that the relevant tile file is present in {tile_path} \n "
          f"If not, add the file run hector.manager.read_hector_tiles(), again")

    # Loop through and check whether the tile files present in the directory are all listed in "Hector_Tile_List".
    # If not add them to "Hector_Tile_List", as well as keep a record in "not_in_list"
    filenames = [file_name for file_name in os.listdir(tile_path) if "Tile" in file_name]
    not_in_list = []
    with open(f"{base_path}/{file_names[0]}", 'r+') as file:
        content = file.read() # read all content from a file using read()
        for afile in filenames:
            if afile not in content:
                file.write(afile + "\n")
                not_in_list.append(f"{tile_path}/{afile}")

    if len(not_in_list):
        # Check to see if the number of columns and their names match by comparing the headers
        ss_header = pd.read_csv(f"{base_path}/{file_names[1]}", nrows=0).columns
        for abs_path in not_in_list:
            print(abs_path)
            pd.read_csv(abs_path, nrows=0, header=11).columns
        cols = [pd.read_csv(abs_path, nrows=0, header=11).columns for abs_path in not_in_list]
        cols_identical = [all(ss_header == colx) for colx in cols]
        assert all(cols_identical), f"Tile file header mis-match. The required column names are: {headerList}"

        for afile in not_in_list:
            hector_tile = pd.read_csv(afile, header=11, index_col="Hexabundle")
            hector_tile['u_mag'] = np.repeat(-99.0, hector_tile.shape[0])
            hector_tile.loc[["H", "U"]].to_csv(f"{base_path}/{file_names[1]}", mode='a', header=False, index=False)
            hector_tile.loc[["H", "U"]].to_csv(f"{base_path}/{file_names[2]}", mode='a', header=False, index=False, columns=headerNew)
    return

def read_stellar_mags():
    """Read stellar magnitudes from the various catalogues available.  Note that
    for python 3 some versions of numpy will have a problem with loadtxt() not
    converting the strings from byte to str.  This is fixed in later versions of
    numpy, so make sure to update your numpy."""
    # MLPG: elif added to take "Hector_mags" into account, which comes from Tiling files
    data_dict = {}
    for (path, catalogue_type, extinction) in stellar_mags_files():
        if catalogue_type == 'ATLAS':
            names = ('PHOT_ID', 'ra', 'dec', 'u', 'g', 'r', 'i', 'z',
                     'sigma', 'radius')
            formats = ('U20', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                       'f8', 'f8')
            skiprows = 2
            delimiter = None
            name_func = lambda d: d['PHOT_ID']
        elif catalogue_type == 'SDSS_cluster':
            names = ('obj_id', 'ra', 'dec', 'u', 'g', 'r', 'i', 'z',
                     'priority')
            formats = ('U30', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'i')
            skiprows = 0
            delimiter = None
            name_func = lambda d: '888' + d['obj_id'][-9:]
        elif catalogue_type == 'SDSS_GAMA':
            names = ('name', 'obj_id', 'ra', 'dec', 'type', 'u', 'sig_u',
                     'g', 'sig_g', 'r', 'sig_r', 'i', 'sig_i', 'z', 'sig_z')
            formats = ('U20', 'U30', 'f8', 'f8', 'U10', 'f8', 'f8',
                       'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
            skiprows = 1
            delimiter = ','
            name_func = lambda d: d['name']
        elif catalogue_type == "Hector_mags":
            names = ('name', 'x', 'y', 'rads', 'angs', 'azAngs', 'angs_azAng', 'ra', 'dec', 'u', 'g', 'r',
                  'i', 'z', 'GAIA_g_mag', 'GAIA_bp_mag', 'GAIA_rp_mag')
            formats = ('U30', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                       'f8', 'f8', 'f8', 'f8', 'f8')
            skiprows = 1
            delimiter = ','
            name_func = lambda d: d['name']
        data = np.genfromtxt(path, skip_header=skiprows, delimiter=delimiter,
                          dtype={'names': names, 'formats': formats})
        data = data[~np.isnan(data['g'])] #Sree: early Hector tiles for clusters do not have g mag and cause issues later on

        if data.shape == ():
            # Ensure data is iterable in case of only a single line
            data.shape = (1,)
        data['u'] += extinction[0]
        data['g'] += extinction[1]
        data['r'] += extinction[2]
        data['i'] += extinction[3]
        data['z'] += extinction[4]
        #Sree: cluster tile files have nan for u,i,z bands. TODO: catalogue should have all mags!
        data['u'][np.isnan(data['u'])] = 0.0; data['i'][np.isnan(data['i'])] = 0.0; data['z'][np.isnan(data['z'])] = 0.0
        new_data_dict = {name_func(line): line for line in data}
        data_dict.update(new_data_dict)
    return data_dict

def create_dummy_output(reduced_files, tlm=False, overwrite=False):
    # Loop over all reduced files and create mock
    # output for the appropriate file type and size
    #TODO: remove this

    for reduced_file in reduced_files:
        if reduced_file.ndf_class == 'BIAS' or reduced_file.ndf_class == 'DARK' or reduced_file.ndf_class == 'LFLAT' or reduced_file.ndf_class == 'MFFFF' or reduced_file.ndf_class == 'MFARC' or reduced_file.ndf_class == 'MFOBJECT':
        #probably we would not require the above *if* condition

            # specify a demension of each ccd
            tmpfile = pf.open(reduced_file.raw_path)
            sz1 = 4096; sz2 = 2048
            if reduced_file.ccd == 'ccd_3' or reduced_file.ccd == 'ccd_4': # Spector
                sz2 = 4096 # Should check the dimension for Spector is really 4096x4096
            if reduced_file.ndf_class == 'MFARC' or reduced_file.ndf_class == 'MFOBJECT' or (reduced_file.ndf_class == 'MFFFF' and not tlm): # arc, flat, object
                sz1 = 819
                if reduced_file.ccd == 'ccd_3' or reduced_file.ccd == 'ccd_4':
                    sz1 = 819 # Should update the dimension for Spector
            tmpfile['PRIMARY'].data = tmpfile['PRIMARY'].data[0:sz1,0:sz2]


            # Insert Variance HDU
            try:
                tmpfile['VARIANCE']
            except KeyError:
                hdu = pf.ImageHDU(tmpfile['PRIMARY'].data)
                hdu.header['EXTNAME'] = ('VARIANCE')
                tmpfile.append(hdu)

            # Insert essential HDUs to object frames
            if reduced_file.ndf_class == 'MFOBJECT':
                try: 
                    tmpfile['SKY']
                except KeyError:
                    hdu = pf.ImageHDU(np.zeros(sz2))
                    hdu.header['EXTNAME'] = ('SKY')
                    tmpfile.append(hdu)

                try: 
                    tmpfile['REDUCTION_ARGS']
                except KeyError:
                    hdu = pf.BinTableHDU(Table([['THPUT_FILENAME','TPMETH'],['thput_dummy.fits','OFFSKY']],names=['ARG_NAME','ARG_VALUE'], dtype=['S','S']))
                    hdu.header['EXTNAME'] = ('REDUCTION_ARGS')
                    tmpfile.append(hdu)

            # WCS is required for ARC and object frames 
            if (reduced_file.ndf_class == 'MFARC' or reduced_file.ndf_class == 'MFOBJECT') and 'CRPIX1' not in tmpfile['PRIMARY'].header:
                tmpfile['PRIMARY'].header['CRPIX1'] = (1.024000000000E+03,'Reference pixel along axis 1')
                tmpfile['PRIMARY'].header['CDELT1'] = (1.050317537860E+00,'Co-ordinate increment along axis 1')
                tmpfile['PRIMARY'].header['CRVAL1'] = (4.724474841231E+03,'Co-ordinate value of axis 1')

            if reduced_file.ndf_class == 'MFFFF' and tlm: # make_tlm()
                out_path = reduced_file.tlm_path
            else:
                out_path = reduced_file.reduced_path

            if os.path.exists(out_path) and overwrite:
                os.remove(out_path)

            tmpfile.writeto(out_path)
            tmpfile.close()


def create_dummy_combine(input_file, output_file, class_dummy):
    # Simply copy one of the reduced calibration files to combined one
    #TODO: remove this
    if class_dummy == 'BIAS' or class_dummy == 'DARK' or class_dummy == 'LFLAT':
        shutil.copy2(input_file, output_file)


class MatchException(Exception):
    """Exception raised when no matching calibrator is found."""
    pass
