"""
This version of the code is implemented to Hector pipeline.
Sree implemented Sam Vaughan's 2D arc modelling code into Hector repo.
Visit https://dev.aao.org.au/datacentral/hector/Hector-Wave-Cal for further details.
arc_model_2d is modularized one of Hector-Wave-Cal/workflow/scripts/fit_arc_model_from_command_line.py
All other modules are from Hector-Wave-Cal/workflow/scripts/utils.py
"""

import hector.dr.fit_arc_model_utils as utils
import argparse
from pathlib import Path
from astropy.io import fits
import numpy as np

import pandas as pd
from tqdm import tqdm
import scipy.interpolate as interpolate
from astropy.table import Table
import numpy.polynomial.chebyshev as cheb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import xarray as xr
import os


def arc_model_2d(reduced_arc_filename, arcdata_filename, tlm_filename, plot_residuals=False, save_params=None, 
        no_update_arc=False, saved_file_suffix='_wavela_updated', intensity_cut=20, N_x=8, N_y=4, alpha=1e-3, verbose=False, debug=False):
    """
    Run the 2D wavelength fitting from the command line. 
    Take a reduced arc file, an arc data file and a TLM map and run the 2D wavelength fitting code on them. 
    Update the WAVELA and SHIFTS extensions of the reduced arc filename. 
    The original extensions are copied to "OLD_WAVELA" and "OLD_SHIFTS" extensions.
    The updated arc is then written to a file which has the same filename as the input 
    with a "_wavela_updated" suffix (which can be specified on the command line). 
    Optionally, plot the residuals and save the fitting parameters.
    """

    arcdata_filename = Path(arcdata_filename)
    tlm_filename = Path(tlm_filename)
    reduced_arc_filename = Path(reduced_arc_filename)

    if not no_update_arc:
        # To add an "_wavela_updated" to the arc filename, use this line
        #output_filename = reduced_arc_filename.parent / (
        #    reduced_arc_filename.stem + saved_file_suffix + ".fits")
        # To overwrite the arc completely, uncomment this line
         output_filename = reduced_arc_filename

    arc_name = Path(reduced_arc_filename).stem.strip("red")
    N_params_per_slitlet = (N_x + 1) * (N_y + 1)
    ccd_number = Path(reduced_arc_filename).stem[5]

    if verbose:
        print(
        f"""
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Performing a 2D wavelength fit of the arc file {reduced_arc_filename}, with TLM map {tlm_filename}.
        Using a polynomial of order {N_x} in the x direction and {N_y} in the y direction for each slitlet.
        The Ridge Regression regularisation parameter alpha is {alpha}.
        Ignoring arc lines with an "INTENSITY" value less than {intensity_cut}.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        """
    )

    # Read in the Arc Data. The df_full pandas data frame has properties such as  "x_pixel",
    # "y_pixel", "wave" etc.  See read_arc() for details.
    if verbose:    
        print("Reading in the Arc Data...\n")
    df_full,N_pixels,s_min,s_max,slitlet_info = utils.read_arc(arcdata_filename, tlm_filename, reduced_arc_filename,verbose=verbose)


    # Set up the fitting
    df_fitting, wave_standardised, X2 = utils.set_up_arc_fitting(
        df_full, N_x=N_x, N_y=N_y, N_pixels=N_pixels, s_min=s_min, s_max=s_max, intensity_cut=intensity_cut, verbose=verbose
    )
    wavelengths = df_fitting.wave.values

    # Do the fitting
    model = utils.fit_model(X=X2, y=wave_standardised, alpha=alpha)

    if (verbose):
        print('Model:')
        print(model.coef_)
        print('X2:')
        print(X2)

    # Get the predictions and the mean squared error
    predictions = utils.get_predictions(model, X2, wavelengths)
    mse = utils.calculate_MSE(model, X2, wavelengths)
    if verbose:
        print(f"The MSE is {mse:.3f} A\n")

    # Save the parameters
    if save_params is not None:
        parameters = utils.save_parameters(
            save_params,
            df_fitting,
            model,
            N_params_per_slitlet,
            mse,
            arc_name,
        )

    # Optionally make a plot of the residuals
    if plot_residuals:
        print('Plotting residuals...')
        plt.ion()
        fig, axs = utils.plot_residuals(df_fitting, predictions, wavelengths, debug=debug)
        fig.show()

    # If we are going to update the arc...
    if not no_update_arc:
        # Update the shifts and WAVELA array of the input arc file
        if verbose:
            print("Predicting new values for the WAVELA array... (This may take a while)")
        df_predict, wave_standardised, X2_predict, N_pixels_x, N_fibres_total = (
            utils.set_up_WAVELA_predictions(tlm_filename, ccd_number, N_x, N_y,slitlet_info,verbose=verbose)
        )

        WAVELA_predictions = utils.get_predictions(model, X2_predict, wavelengths)

        # output test values if we have switched on debug:
        nn = len(df_predict)
        if (debug):
            print('fib 188 (WAVELA predict):')
            for i in range(nn):
                fb = df_predict.iloc[i]['fibre_number']
                if (fb==188):
                    xp = df_predict.iloc[i]['x_pixel']
                    yp = df_predict.iloc[i]['y_pixel']
                    if (((xp>210) & (xp<230)) | ((xp>1125) & (xp < 1145)) | ((xp>1802) & (xp<1822))):
                        print(i,xp,yp,WAVELA_predictions[i],df_predict.iloc[i]['fibre_number'],df_predict.iloc[i]['slitlet'],df_predict.iloc[i]['y_slitlet'])

        # Convert the WAVELA into nanometres from microns
        new_wavela = WAVELA_predictions.reshape(N_fibres_total, N_pixels_x) / 10

        if verbose:
            print("\nSorting out the output file...")
        reduced_arc_hdu = fits.open(reduced_arc_filename)
        new_shifts_array = np.zeros_like(reduced_arc_hdu["SHIFTS"].data)
        new_shifts_array[1, :] = 1.0

        if verbose:
            print("\tCopying previous WAVELA and SHIFTS extensions to OLDWAVELA and OLDSHIFTS")
        # Copy the old shifts and WAVELA array to old values
        arc_hdu = fits.open(reduced_arc_filename)
        old_WAVELA_array = arc_hdu["WAVELA"].data
        old_WAVELA_header = arc_hdu["WAVELA"].header
        old_SHIFTS_array = arc_hdu["SHIFTS"].data
        old_SHIFTS_header = arc_hdu["SHIFTS"].header

        if 'OLDWAVELA' not in arc_hdu:
            copied_WAVELA_extension = fits.ImageHDU(
                old_WAVELA_array, header=old_WAVELA_header, name="OLDWAVELA")
            arc_hdu.append(copied_WAVELA_extension)
        if 'OLDSHIFTS' not in arc_hdu:
            copied_SHIFTS_extension = fits.ImageHDU(
                old_SHIFTS_array, header=old_SHIFTS_header, name="OLDSHIFTS")
            arc_hdu.append(copied_SHIFTS_extension)

        arc_hdu["WAVELA"].data = new_wavela
        arc_hdu["SHIFTS"].data = new_shifts_array

        #if verbose:
        arc_hdu.writeto(output_filename, overwrite=True)
        print(f"\tWriting to {output_filename}")
