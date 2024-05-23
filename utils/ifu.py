"""
This module contains the IFU class, used extensively throughout sami.

An IFU instance contains data from a single IFU's observation. As well as
the observed flux, it stores the variance and a lot of metadata. See the
code for everything that's copied.

One quirk to be aware of: the data on disk are stored in terms of total
counts, but the IFU object automatically scales this by exposure time to
get a flux.
"""
import sys

import numpy as np
import re
import glob
import pandas as pd

# astropy fits file io (replacement for pyfits)
import astropy.io.fits as pf
# extra astropy bits to calculate airmass
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

from .fluxcal2_io import read_model_parameters
from .term_colours import *

# MLPG: importing hector, and adding hector_path
import hector
hector_path = str(hector.__path__[0])+'/'

PLATECENTRE_IN_ROBOT_COOR = [324470.0, 297834.0]

class IFU:

    def __init__(self, rss_filename, probe_identifier, flag_name=True):
        """A class containing data and other information from a single file pertaining to a particular object or
        probe."""
        
        self.infile=rss_filename

        # Open the file (should I really be doing this here?)
        hdulist=pf.open(rss_filename)

        data_in=hdulist['PRIMARY'].data
        variance_in=hdulist['VARIANCE'].data
        
        # Load all necessary metadata
        self.primary_header = hdulist['PRIMARY'].header
        self.fibre_table_header = hdulist['FIBRES_IFU'].header
        try:
            self.reduction_arguments = hdulist['REDUCTION_ARGS'].data
        except KeyError:
            raise IOError('No REDUCTION_ARGS extension found: '+rss_filename)

        #TEMP - store full headers (Nic)
        self.primary_header = hdulist['PRIMARY'].header
        self.fibre_table_header = hdulist['FIBRES_IFU'].header
        self.reduction_arguments = hdulist['REDUCTION_ARGS'].header

        fibre_table=hdulist['FIBRES_IFU'].data

        # Some useful stuff from the header
        self.exptime = self.primary_header['EXPOSED']
        self.crval1 = self.primary_header['CRVAL1']
        self.cdelt1 = self.primary_header['CDELT1']
        self.crpix1 = self.primary_header['CRPIX1']
        self.naxis1 = self.primary_header['NAXIS1']

        # Field centre values (not bundle values!)
        self.meanra = self.primary_header['MEANRA']
        self.meandec = self.primary_header['MEANDEC']

        self.instrument = self.primary_header['INSTRUME'] 
        self.epoch = self.primary_header['EPOCH']

        # Determine and store which spectrograph ARM this is from (red/blue)

        if (self.primary_header['SPECTID'] == 'BL'):
            self.spectrograph_arm = 'blue'
        elif (self.primary_header['SPECTID'] == 'RD'):
            self.spectrograph_arm = 'red'

        self.gratid = self.primary_header['GRATID']
        self.gain = self.primary_header['RO_GAIN']

        self.zdstart=self.primary_header['ZDSTART']
        self.zdend=self.primary_header['ZDEND']

        # Wavelength range
        x=np.arange(self.naxis1)+1
        
        L0=self.crval1-self.crpix1*self.cdelt1 #Lc-pix*dL
        
        self.lambda_range=L0+x*self.cdelt1

        # Based on the given information (probe number or object name) find the other piece of information. NOTE - this
        # will fail for unassigned probes which will have empty strings as a name.
        if flag_name==True:
            if len(probe_identifier)>0:
                self.name=probe_identifier # Flag is true so we're selecting on object name.
                msk0=fibre_table.field('NAME')==self.name # First mask on name.
                table_find=fibre_table[msk0] 

                # Find the IFU name from the find table.
                self.ifu=np.unique(table_find.field('PROBENUM'))[0]

            else:
                # Write an exception error in here?
                pass
            
        else:
            self.ifu=probe_identifier # Flag is not true so we're selecting on probe (IFU) number.
            
            msk0=fibre_table.field('PROBENUM')==self.ifu # First mask on probe number.
            table_find=fibre_table[msk0]

            # Pick out the place in the table with object names, rejecting SKY and empty strings.
            object_names_nonsky = [s for s in table_find.field('NAME') if s.startswith('SKY')==False and s.startswith('Sky')==False and len(s)>0]
            #print np.shape(object_names_nonsky)

            self.name=list(set(object_names_nonsky))[0]
            
        mask=np.logical_and(fibre_table.field('PROBENUM')==self.ifu, fibre_table.field('NAME')==self.name)
        table_new=fibre_table[mask]


        ##
        # for i in ['C', 'H']:
        #     del mask, table_new
        #     mask = fibre_table.field('PROBENAME')==i
        #     table_new = fibre_table[mask]
        #     print("Probe=", i, 'probenum=', table_new.field('PROBENUM'))
        #     print(np.mean(table_new.field('FIBPOS_X')), np.mean(table_new.field('FIBPOS_Y')))
        #     print(np.mean(table_new.field('XPOS')), np.mean(table_new.field('YPOS')))
        # sys.exit()
        ##

        # Mean RA of probe centre, degrees
        self.ra = table_new.field('GRP_MRA')[0]
        # Mean Dec of probe centre, degrees
        self.dec = table_new.field('GRP_MDEC')[0]


        #X and Y positions of fibres in absolute degrees.
        self.xpos=table_new.field('FIB_MRA') #RA add -1*
        self.ypos=table_new.field('FIB_MDEC') #Dec

        # Positions in arcseconds relative to the field centre
        self.xpos_rel=table_new.field('XPOS')
        self.ypos_rel=table_new.field('YPOS')
 
        # Fibre number - used for tests.
        self.n=table_new.field('FIBNUM')
    
        # Fibre designation.
        self.fib_type=table_new.field('TYPE')
        
        # Probe Name
        self.hexabundle_name=table_new.field('PROBENAME')
        
        # Adding for tests only - LF 05/04/2012
        # self.x_microns=-1*table_new.field('FIBPOS_X') # To put into on-sky frame
        # self.x_microns=table_new.field('FIBPOS_X')  # MLPG: removed the -1.0 sign in the above line here
        # self.y_microns=table_new.field('FIBPOS_Y')


        # Name of object
        name_tab=table_new.field('NAME')
        self.name=name_tab[0]
        
        # indices of the corresponding spectra (SPEC_ID counts from 1, image counts from 0)
        ind=table_new.field('SPEC_ID')-1
        
        self.data=data_in[ind,:]/self.exptime
        self.var=variance_in[ind,:]/(self.exptime*self.exptime)

        # Master sky spectrum:
        self.sky_spectra = hdulist['SKY'].data
        #    TODO: It would be more useful to have the sky spectrum subtracted from
        #    each fibre which requires the RWSS file/option in 2dfdr

        # 2dfdr determined fibre througput corrections
        try:
            self.fibre_throughputs = hdulist['THPUT'].data[ind]
        except KeyError:
            # None available; never mind.
            pass

        # Added for Iraklis, might need to check this.
        self.fibtab=table_new

        # TEMP -  object RA & DEC (Nic)
        self.obj_ra=table_new.field('GRP_MRA')
        self.obj_dec=table_new.field('GRP_MDEC')

        # Pre-measured offsets, if available
        try:
            offsets_table = hdulist['ALIGNMENT'].data
        except KeyError:
            # Haven't been measured yet; never mind
            pass
        else:
            line_number = np.where(offsets_table['PROBENUM'] == self.ifu)[0][0]
            offsets = offsets_table[line_number]
            self.x_cen = -1 * offsets['X_CEN'] # Following sign convention for x_microns above
            self.y_cen = offsets['Y_CEN']
            self.x_refmed = -1 * offsets['X_REFMED']
            self.y_refmed = offsets['Y_REFMED']
            self.x_shift = -1 * offsets['X_SHIFT']
            self.y_shift = offsets['Y_SHIFT']


        # Fitted DAR parameters, if available
        try:
            hdu_fluxcal = hdulist['FLUX_CALIBRATION']
        except KeyError:
            # Haven't been measured yet; never mind
            pass
        else:
            self.atmosphere = read_model_parameters(hdu_fluxcal)[0]
            del self.atmosphere['flux']
            del self.atmosphere['background']

        # Object RA & DEC
        self.obj_ra=table_new.field('GRP_MRA')
        self.obj_dec=table_new.field('GRP_MDEC')

        def get_fibbundle_Xc_Yc_and_robotPos(Probename, fibrenum, object_robot_tab):
            """
            Extracts the x,y positions of fibres in a give hexabundle, and rectangular magnet x, y positions from the
            robot file.
            The rotation of the positions based on the angle of rotation of the rectangular magnet is also returned

            :param Probename: Name of the hexabundle
            :param fibrenum: fiber numbers for that hexabundle
            :param object_robot_tab: pandas csv tab storing the information contained in the file input to robot
            :return: x_pos, y_pos, fiber_number: x/y positions in microns and fibre no. read from the fibre slitlet file from Julia Bryant
                     mean_magX, mean_magY: circular magnet x/y position in microns from the robot file
                     rotation_angle_magnet : The angle of the rectangular magnet --> 270 minus the robot holding angle minus the robot placing angle
                     x_rotated_pos, y_rotated_pos: The rotated x/y positions
            """
            # MLPG: adding a brand-new functions to do hexabundle rotations

            # Read-in the fibre position information file from Julia Bryant.
            fibre_pos = pd.read_csv(hector_path + 'utils/Fibre_slitInfo_final_updated03Mar2022for_swapped_fibres_BrokenHexaM.csv')

            fibnum_from_file_probe = fibre_pos.loc[(fibre_pos["Bundle/plate"] == Probename)]
            fibnum_from_file = fibnum_from_file_probe['Fibre_number'].to_numpy()
            Bundle_Xc_from_file = fibnum_from_file_probe['Bundle_Xc'].to_numpy()
            Bundle_Yc_from_file = fibnum_from_file_probe['Bundle_Yc'].to_numpy()
            x_pos, y_pos, fiber_number = [], [], []
            for num in fibrenum:
                min_idx = np.array(np.where((fibnum_from_file - num) == 0.0)).squeeze()
                assert np.size(min_idx) == 1, 'Error'

                x_pos.append(Bundle_Xc_from_file[min_idx])
                y_pos.append(Bundle_Yc_from_file[min_idx])
                fiber_number.append(fibnum_from_file[min_idx])

            del fibnum_from_file_probe, fibnum_from_file, Bundle_Xc_from_file, Bundle_Yc_from_file

            x_pos = np.array(x_pos)
            y_pos = np.array(y_pos)

            # For each hexabundle, get its circular and rectangular magnet (comes directly from the robot file)
            cm = object_robot_tab.loc[(object_robot_tab['Hexabundle'] == Probename) & (object_robot_tab['#Magnet'] == 'circular_magnet')]
            rm = object_robot_tab.loc[(object_robot_tab['Hexabundle'] == Probename) & (object_robot_tab['#Magnet'] == 'rectangular_magnet')]

            mean_magX = cm.Center_y
            mean_magX = mean_magX[mean_magX.index[0]] * 1.0E3  # Converts to microns
            mean_magY = cm.Center_x
            mean_magY = mean_magY[mean_magY.index[0]] * 1.0E3  # Converts to microns

            # The angle of the rectangular magnet -- 270 minus the robot holding angle minus the robot placing angle
            rotation_angle_magnet = np.radians(270.0 - rm.rot_holdingPosition - rm.rot_platePlacing)
            rotation_angle_magnet = rotation_angle_magnet[rotation_angle_magnet.index[0]]

            # Rotation of axes: see Evernote HECTOR/probe rotations for more information. The 'minus' sign, I think, accounts
            # for the fact that North is down (and East to the right) on the plate
            x_rotated_pos = -1 * (+np.cos(rotation_angle_magnet) * x_pos - np.sin(rotation_angle_magnet) * y_pos)
            y_rotated_pos = -1 * (-np.sin(rotation_angle_magnet) * x_pos - np.cos(rotation_angle_magnet) * y_pos)

            return x_pos, y_pos, fiber_number, mean_magX, mean_magY, rotation_angle_magnet, x_rotated_pos, y_rotated_pos

        # MLPG: Take information from the Tile/Robot files
        # TODO MLPG: automatically download the robot/tile files if they are not within the specified file path
        dash_locs = [m.start() for m in re.finditer('/', self.primary_header['CFG_FILE'])]
        #tile_file = self.primary_header['CFG_FILE'][dash_locs[-1] + 1::]
        tile_file = self.primary_header['PLATEID']
        #robot_file = glob.glob(hector_path + 'Tiles/Robot_files/' + tile_file.replace('Tile', 'Robot') + '*')
        robot_file = glob.glob(hector_path + 'Tiles/Robot_files/*' + tile_file + '*')
        if(len(robot_file) == 0):
            print('Missing robot file: ',robot_file,' Tile file:',tile_file)
            sys.exit()
        object_robottab = pd.read_csv(robot_file[0], skiprows=6)

        self.probe_annulus = table_new.field('CIRCMAG')  # telecentricity of the probe, given by the circmag colour

        # Gets the x,y positions of the fibres, the rotation angle of the rectangular magnet, and the x,y rotated positions
        self.x, self.y, self.fnum, self.mean_x, self.mean_y, self.rotation_angle, self.x_rotated, self.y_rotated = \
            get_fibbundle_Xc_Yc_and_robotPos(self.hexabundle_name[0], self.n, object_robottab)

        self.x_microns = -1.0 * ( (self.mean_x + self.x_rotated) - PLATECENTRE_IN_ROBOT_COOR[1] ) # Robot y- is x-coordinate, and multipllied by -1.0 to put it in sky-coor
        self.y_microns = PLATECENTRE_IN_ROBOT_COOR[0] - (self.mean_y + self.y_rotated)  # Robot x- is y-coordinate
        # self.x_microns = (self.mean_x + self.x) - PLATECENTRE_IN_ROBOT_COOR[1]  # Robot y- is x-coordinate
        # self.y_microns = (self.mean_y + self.y) - PLATECENTRE_IN_ROBOT_COOR[0]  # Robot x- is y-coordinate

        # print("--->", self.hexabundle_name[0])
        # print(self.mean_x, self.mean_y)
        # print("robot coor= ", PLATECENTRE_IN_ROBOT_COOR[1], PLATECENTRE_IN_ROBOT_COOR[0])
        # # print("self.x_microns=", self.x_microns)
        # # print("self.x=", self.x)
        # # print("self.x_rotated=", self.x_rotated)
        # print("My_mean_microns= ", np.mean(self.x_microns), np.mean(self.y_microns))
        # print("FIBPOS_mean_microns= ", np.mean(table_new.field('FIBPOS_X')), np.mean(table_new.field('FIBPOS_Y')))
        # print("FIBPOS_mean_notrot= ", np.mean(self.x_microns), np.mean(self.y_microns))
        # print(np.mean(table_new.field('XPOS')), np.mean(table_new.field('YPOS')))
        # print("mean RA, DEC= ", self.ra, self.dec)
        # print((table_new.field('XPOS')), (table_new.field('YPOS')))

        # print("")
        # del mask, table_new
        # mask = fibre_table.field('PROBENAME') == 'F'
        # table_new = fibre_table[mask]
        # print('mean xpos, ypos (relative to field centre, arcsec)=', np.mean(table_new.field('XPOS')), np.mean(table_new.field('YPOS')))
        # # print(np.mean(table_new.field('XPOS') * np.cos(np.deg2rad(table_new.field('XPOS')))))
        # print("mean RA, DEC= ", table_new.field('GRP_MRA')[0], table_new.field('GRP_MDEC')[0])
        # fieldC = HMS2deg(ra='14 51 44.86', dec='+02 15 54.5')
        # print("Field Centre = ", fieldC)
        # print("xpos, ypos calculated = ", 3600*(table_new.field('GRP_MRA')[0] - float(fieldC[0])), 3600*(table_new.field('GRP_MDEC')[0] - float(fieldC[1])))
        # print("table FIBPOS X, Y [microns]=", np.mean(table_new.field('FIBPOS_X')), np.mean(table_new.field('FIBPOS_Y')))
        # print("")
        #
        # del mask, table_new
        # mask = fibre_table.field('PROBENAME') == 'G'
        # table_new = fibre_table[mask]
        # print('mean xpos, ypos (relative to field centre, arcsec)=', np.mean(table_new.field('XPOS')),
        #       np.mean(table_new.field('YPOS')))
        # # print(np.mean(table_new.field('XPOS') * np.cos(np.deg2rad(table_new.field('XPOS')))))
        # print("mean RA, DEC= ", table_new.field('GRP_MRA')[0], table_new.field('GRP_MDEC')[0])
        # fieldC = HMS2deg(ra='14 51 44.86', dec='+02 15 54.5')
        # print("Field Centre = ", fieldC)
        # print("xpos, ypos calculated = ", 3600 * (table_new.field('GRP_MRA')[0] - float(fieldC[0])),
        #       3600 * (table_new.field('GRP_MDEC')[0] - float(fieldC[1])))
        # print("table FIBPOS X, Y [microns]=", np.mean(table_new.field('FIBPOS_X')), np.mean(table_new.field('FIBPOS_Y')))
        #

        # Cleanup to reduce memory footprint
        del hdulist
