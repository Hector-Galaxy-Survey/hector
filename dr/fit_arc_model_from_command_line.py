import utils
import argparse
from pathlib import Path
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt


description_string = """
Run the 2D wavelength fitting from the command line. 

Take a reduced arc file, an arc data file and a TLM map and run the 2D wavelength fitting code on them. 

Update the WAVELA and SHIFTS extensions of the reduced arc filename. The original extensions are copied to "OLD_WAVELA" and "OLD_SHIFTS" extensions.

The updated arc is then written to a file which has the same filename as the input with a "_wavela_updated" suffix (which can be specified on the command line). 

Optionally, plot the residuals and save the fitting parameters.
"""
parser = argparse.ArgumentParser(description=description_string)
parser.add_argument(
    "reduced_arc_filename",
    help="The filename of the reduced arc file, which usually ends in red.fits",
)
parser.add_argument(
    "dat_filename",
    help="The filename of the arc data file, which usually ends in .dat",
)
parser.add_argument("tlm_filename", help="The filename of the TLM map")
parser.add_argument(
    "--intensity-cut",
    type=int,
    default=20,
    help="Ignore arc lines with an intensity lower than this value",
)
parser.add_argument(
    "--N_x",
    type=int,
    default=6,
    help="Polynomial order in the x direction",
)
parser.add_argument(
    "--N_y",
    type=int,
    default=2,
    help="Polynomial order in the y direction",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=1e-3,
    help="Ridge regression regularisation term",
)
parser.add_argument(
    "--plot_residuals",
    action="store_true",
    help="Whether to display a plot of the residuals or not",
)
parser.add_argument(
    "--verbose",
    action="store_true",
    default=False,
    help="Whether to output diagnostics to the terminal or not",
)
parser.add_argument(
    "--clip",
    action="store_true",
    default=False,
    help="Whether to do a 5 sigma clipping of the data after first fit attempt",
)
parser.add_argument(
    "--global_fit",
    action="store_true",
    default=False,
    help="Whether to do a global fit to the whole detector at once, rather than by slitlet",
)
parser.add_argument(
    "--save-params",
    required=False,
    help="The netcdf filename of the paramters. Must end in .nc",
)
parser.add_argument(
    "--no-update-arc",
    required=False,
    action="store_true",
    help="If given, do not update the WAVELA and SHIFTS extensions of the reduced arc file",
)
parser.add_argument(
    "--saved-file-suffix",
    required=False,
    default="_wavela_updated",
    type=str,
    help="The suffix to add to the reduced arc filename (before the .fits). Set to an empty string to **silently** overwrite the original file",
)


args = parser.parse_args()

arcdata_filename = Path(args.dat_filename)
tlm_filename = Path(args.tlm_filename)
reduced_arc_filename = Path(args.reduced_arc_filename)

print(args)

# flag for debugging:
debug=False

# Optional Parameters
plot_residuals = args.plot_residuals
verbose = args.verbose
clip = args.clip
global_fit = args.global_fit
save_params = args.save_params
no_update_arc = args.no_update_arc
saved_file_suffix = args.saved_file_suffix
intensity_cut = args.intensity_cut
N_x = args.N_x
N_y = args.N_y
alpha = args.alpha

if not no_update_arc:
    # To add an "_wavela_updated" to the arc filename, use this line
    output_filename = reduced_arc_filename.parent / (
        reduced_arc_filename.stem + saved_file_suffix + ".fits"
    )
    # To overwrite the arc completely, uncomment this line
    # output_filename = reduced_arc_filename


arc_name = Path(reduced_arc_filename).stem.strip("red")
N_params_per_slitlet = (N_x + 1) * (N_y + 1)
ccd_number = Path(reduced_arc_filename).stem[5]

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

# Read in the Arc Data.  The df_full pandas data frame has properties such as  "x_pixel",
# "y_pixel", "wave" etc.  See read_arc() for details. 
print("Reading in the Arc Data...\n")
df_full,N_pixels,s_min,s_max,slitlet_info = (utils.read_arc(arcdata_filename, tlm_filename, reduced_arc_filename,verbose=verbose,global_fit=global_fit))

# get the pixel scale (just for plottign and diagnostics:
pixscale = fits.getval(reduced_arc_filename, 'CDELT1', ext=0)
print('Pixel scale is ',pixscale,' ang/pix')

# Set up the fitting

print(df_full)
df_full_c = df_full.copy()


df_fitting, wave_standardised, X2 = utils.set_up_arc_fitting(
    df_full, N_x=N_x, N_y=N_y, N_pixels=N_pixels, s_min=s_min, s_max=s_max, intensity_cut=intensity_cut, verbose=verbose)

print('X2 shape:',np.shape(X2))
print('wave_standardize shape:',np.shape(wave_standardised))

wavelengths = df_fitting.wave.values
print("\tDone!")

# Do the fitting
model = utils.fit_model(X=X2, y=wave_standardised, alpha=alpha)

if (verbose):
    print('Model:')
    print(model.coef_)
    print('X2:')
    print(X2)
    
# Get the predictions and the mean squared error
predictions = utils.get_predictions(model, X2, wavelengths)
rms = utils.calculate_RMS(model, X2, wavelengths)
print(f"The RMS is {rms:.3f} A\n")
residuals = wavelengths - predictions

# clip the data and refit:
if (clip):
    # generate clipped data for fitting:
    X2_clip, wave_standardised_clip,wavelengths_clip = utils.clip_data(X2,wave_standardised,wavelengths,predictions,rms,verbose=verbose)

    # fitted the clipped data:
    model = utils.fit_model(X=X2_clip, y=wave_standardised_clip, alpha=alpha)
    #model = utils.fit_model(X=X2, y=wave_standardised, alpha=alpha)
    predictions_clip = utils.get_predictions(model, X2_clip, wavelengths_clip)
    #predictions_clip = utils.get_predictions(model, X2, wavelengths)
    rms_clip = utils.calculate_RMS(model, X2_clip, wavelengths_clip)
    #rms_clip = utils.calculate_RMS(model, X2, wavelengths)
    residuals_clip = wavelengths_clip - predictions_clip
    # for some reason the utils.calculate_RMS() routine does not return the correct number
    # for the clipped RMS.  Don't know why.  Instead, estimate it directly here:
    rms_clip = residuals_clip.std()
    print(f"The RMS after clipping is {rms_clip:.3f} A (ignoring clipped data)\n")

    if (verbose):
        print('Model:')
        print(model.coef_)
        print('X2:')
        print(X2)

    # recalculate predictions for all data based on new model
    predictions = utils.get_predictions(model, X2, wavelengths)
    rms = utils.calculate_RMS(model, X2, wavelengths)
    print(f"The RMS after clipping for all data is {rms:.3f} A\n")


# Save the parameters
if save_params is not None:
    parameters = utils.save_parameters(
        save_params,
        df_fitting,
        model,
        N_params_per_slitlet,
        rms,
        arc_name,
    )

# Optionally make a plot of the residuals
if plot_residuals:
    print('Plotting residuals...')
    plt.ion()
    fig, axs = utils.plot_residuals(df_fitting, predictions, wavelengths,debug=debug,pixscale=pixscale)
    fig.show()

# If we are going to update the arc...
if not no_update_arc:
    # Update the shifts and WAVELA array of the input arc file
    print("Predicting new values for the WAVELA array... (This may take a while)")
    df_predict, wave_standardised, X2_predict, N_pixels_x, N_fibres_total = (
        utils.set_up_WAVELA_predictions(tlm_filename, ccd_number, N_x, N_y,slitlet_info,verbose=verbose,global_fit=global_fit)
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

    print("\nSorting out the output file...")
    reduced_arc_hdu = fits.open(reduced_arc_filename)
    new_shifts_array = np.zeros_like(reduced_arc_hdu["SHIFTS"].data)
    new_shifts_array[1, :] = 1.0

    print("\tCopying previous WAVELA and SHIFTS extensions to OLDWAVELA and OLDSHIFTS")
    # Copy the old shifts and WAVELA array to old values
    arc_hdu = fits.open(reduced_arc_filename)
    old_WAVELA_array = arc_hdu["WAVELA"].data
    old_WAVELA_header = arc_hdu["WAVELA"].header

    old_SHIFTS_array = arc_hdu["SHIFTS"].data
    old_SHIFTS_header = arc_hdu["SHIFTS"].header

    copied_WAVELA_extension = fits.ImageHDU(
        old_WAVELA_array, header=old_WAVELA_header, name="OLDWAVELA"
    )
    copied_SHIFTS_extension = fits.ImageHDU(
        old_SHIFTS_array, header=old_SHIFTS_header, name="OLDSHIFTS"
    )

    arc_hdu["WAVELA"].data = new_wavela
    arc_hdu["SHIFTS"].data = new_shifts_array

    print(f"\tWriting to {output_filename}")
    arc_hdu.writeto(output_filename, overwrite=True)
    print("\tDone!")
