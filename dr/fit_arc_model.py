"""
Sree implemented Sam Vaughan's 2D arc modelling code into Hector repo.
Visit https://dev.aao.org.au/datacentral/hector/Hector-Wave-Cal for further details.
arc_model_2d is modularized one of Hector-Wave-Cal/workflow/scripts/fit_arc_model_from_command_line.py
All other modules are from Hector-Wave-Cal/workflow/scripts/utils.py
"""

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
        no_update_arc=False, saved_file_suffix='_wavela_updated', intensity_cut=20, N_x=8, N_y=4, alpha=1e-3, verbose=False):
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

    # Read in the Arc Data
    if verbose:    
        print("Reading in the Arc Data...\n")
    df_full = read_arc(arcdata_filename, tlm_filename, reduced_arc_filename)


    # Set up the fitting
    df_fitting, wave_standardised, X2 = set_up_arc_fitting(
        df_full, N_x=N_x, N_y=N_y, intensity_cut=intensity_cut
    )
    wavelengths = df_fitting.wave.values

    # Do the fitting
    model = fit_model(X=X2, y=wave_standardised, alpha=alpha)

    # Get the predictions and the mean squared error
    predictions = get_predictions(model, X2, wavelengths)
    mse = calculate_MSE(model, X2, wavelengths)
    if verbose:
        print(f"The MSE is {mse:.3f} A\n")

    # Save the parameters
    if save_params is not None:
        parameters = save_parameters(
            save_params,
            df_fitting,
            model,
            N_params_per_slitlet,
            mse,
            arc_name,
        )

    # Optionally make a plot of the residuals
    if plot_residuals:
        fig, axs = plot_residuals(df_fitting, predictions, wavelengths)
        fig.show()

    # If we are going to update the arc...
    if not no_update_arc:
        # Update the shifts and WAVELA array of the input arc file
        if verbose:
            print("Predicting new values for the WAVELA array... (This may take a while)")
        df_predict, wave_standardised, X2_predict, N_pixels_x, N_fibres_total = (
            set_up_WAVELA_predictions(tlm_filename, ccd_number, N_x, N_y)
        )

        WAVELA_predictions = get_predictions(model, X2_predict, wavelengths)
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

def load_tlm_map(tlm_filename):
    """
    Given a TLM map, load the data and return the number of fibres and the number of x pixels

    Args:
        tlm_filename (str): Filename of a tramline map fits file

    Returns:
        tuple: a tuple of the TLM data, the number of fibres and the number of x pixels
    """
    # Load the tlm map
    tlm_map = fits.open(tlm_filename)
    tlm = tlm_map["PRIMARY"].data
    N_fibres, N_pixels = tlm.shape

    return tlm, N_fibres, N_pixels


def read_arc(arcdata_filename, tlm_filename, reduced_arc_filename, return_column_subset=True):
    """
    Make a pandas dataframe of the data in an Arc file.
    Read in the arc data file, the tlm map and the reduced arc itself, then
    load this all into a pandas dataframe with columns which include x_pixel,
    y_pixel and wave. This is the data we'll during the fitting.

    This is a high level function which:
        - Read in the 'FIBRES_IFU' table of the reduced arc filename, which
        has the slitlet information for each fibre.
        - Reads in the Arc Data file, which contains the wavelength information
        of each arc
        - Interpolates the TLM map, so we can find the y pixel value of each arc line.
        The reduced arc only has x-pixel and fibre number information
        - This is all stored in a pandas dataframe

    Args:
        arcdata_filename (str): Filename of the .dat file
        tlm_filename (str): Filename of the TLM map
        reduced_arc_filename (str): Filename of the reduced Arc
        return_column_subset (bool): If True, only return the columns we need for the fitting.

    Returns:
        pd.DataFrame:
    """

    # Load the arcfits.dat file
    datafile = Path(arcdata_filename)

    # Load the tlm map
    tlm, N_fibres, N_pixels = load_tlm_map(tlm_filename)

    # Load the arc file
    arc_frame = fits.open(reduced_arc_filename)
    fibre_df = Table(arc_frame["FIBRES_IFU"].data).to_pandas()

    # Get the file ID
    stem = datafile.stem
    file_id = os.path.basename(reduced_arc_filename)[0:10]
    ccd = int(file_id[5])

    # Load the arc .dat file
    data, fibre_numbers, N_arc_lines, column_names = load_arc_data_file(datafile)
    df = pd.DataFrame(data, columns=column_names)

    df["fibre_number"] = np.repeat(fibre_numbers, N_arc_lines)
    df["CCD"] = ccd
    df["file_id"] = file_id

    # Now make the Y values
    y_values = interpolate_tlm_map(
        N_fibres=N_fibres,
        N_pixels=N_pixels,
        fibre_numbers=df.fibre_number.values,
        ORIGPIX=df.ORIGPIX.values,
        tlm_data=tlm,
    )

    # Add these to the pandas dataframe
    df["y_pixel"] = y_values
    df["x_pixel"] = df.ORIGPIX
    df["wave"] = df.ORIGWAVES

    # Now add the slitlet number
    df = pd.merge(
        df,
        fibre_df.loc[:, ["SPEC_ID", "SLITLET"]],
        left_on="fibre_number",
        right_on="SPEC_ID",
        how="left",
    )

    # Finally, do some renaming and only keep certain columns
    df = df.rename(
        dict(INTEN="intensity", LNEWID="linewidth", SLITLET="slitlet", CCD="ccd"),
        axis=1,
    )

    if return_column_subset:
        df = df.loc[
            :,
            [
                "x_pixel",
                "y_pixel",
                "wave",
                "fibre_number",
                "intensity",
                "linewidth",
                "slitlet",
                "ccd",
                "file_id",
            ],
        ]

    return df


def load_arc_from_db(con, arc_name, verbose=False):
    """Read in the arc data from a database

    Args:
        con (sqlite3 connection): SQLite database connection
        arc_name (str): _description_
        verbose (bool, optional): Print extra information. Defaults to True.

    Returns:
        pd.DataFrame:
    """

    if verbose:
        print("Loading the measurements from the database...")
    df_full = pd.read_sql(
        f"select x_pixel, y_pixel, wave, fibre_number, intensity, linewidth, slitlet, ccd, file_ID from arc_data where arc_data.file_ID = '{arc_name}'",
        con,
    )
    if verbose:
        print("\tDone!")

    df_full = df_full.rename(dict(CCD="ccd"), axis=1)
    return df_full


def interpolate_tlm_map(N_fibres, N_pixels, fibre_numbers, ORIGPIX, tlm_data):
    """
    Interpolate a TLM map to get the y pixel value for each arc line, given its
    fibre number and x pixel location.

    Args:
        N_fibres (int): Number of fibres in the data
        N_pixels (int): Number of x pixels in the data
        fibre_numbers (np.ndarray): The fibre number values of each arc line
        ORIGPIX (np.ndarray): The x pixel locations of each arc line
        tlm_data (np.ndarray): The TLM data

    Returns:
        np.ndarray: The y pixel values of each arc line
    """

    # Now interpolate the tlm map to get the Y value for each pixel
    xx = np.arange(1, N_fibres + 1)
    yy = np.arange(1, N_pixels + 1)
    interp = interpolate.RegularGridInterpolator((xx, yy), values=tlm_data)
    y_values = interp(np.c_[fibre_numbers, ORIGPIX])

    return y_values


def load_arc_data_file(datafile):
    """
    Read in the .dat file which contains the results of the Arc fitting.
    The format is a little funky, so we have to read it in with this custom code.

    This function returns a tuple of:
        - The data in each row of the file
        - A list of fibre numbers in the file
        - The number of arc lines found in each fibre
        - The column names of the file

    Args:
        datafile (str): Arc .dat filename

    Returns:
        tuple: See above
    """

    fibre_numbers = []
    data = []
    N_arc_lines = []

    # Reading the Arc Data file
    with open(datafile, "r") as f:
        for line in tqdm(f):
            if line.startswith(" # FIBNO:"):
                fibre_numbers.append(int(line.split()[2]))
                continue
            elif line.startswith(" # fit parameters: "):
                N_arc_lines.append(int(line.split()[3]))
                continue
            elif line.startswith(
                " # I LNECHAN INTEN LNEWID CHANS WAVES FIT DEV ORIGPIX ORIGWAVES"
            ):
                column_names = line.lstrip("#").split()[1:]
                continue
            else:
                data.append([float(value) for value in line.split()])

    return data, fibre_numbers, N_arc_lines, column_names


def set_up_arc_fitting(df_full, N_x, N_y, intensity_cut=10, verbose=False):
    """
    Set up all the arrays and things we need for the arc fitting.

    Args:
        df_full (pd.DataFrame): A Dataframe loaded from either read_arc or load_arc_from_db.
        N_x (int): Polynomial order in the x direction.
        N_y (int): Polynomial order in the y direction.
        intensity_cut (int): Ignore all arc lines fainter than this value. Default is 10.

    Returns:
        tuple:
    """
    ccd_number = str(df_full.ccd.unique()[0])
    ccd_name, N_slitlets_total, N_fibres_total, N_fibres_per_slitlet = get_info(
        ccd_number
    )

    # Ignore any duplicated columns we may have- e.g if an arc was added twice
    df_full = df_full.drop_duplicates()

    # Ignore NaNs
    s_max = df_full.groupby("slitlet")["y_pixel"].max()[df_full.slitlet]
    s_min = df_full.groupby("slitlet")["y_pixel"].min()[df_full.slitlet]

    # These are y values **within a slitlet**
    df_full["y_slitlet"] = (
        2 * (df_full["y_pixel"] - s_max.values) / (s_max.values - s_min.values)
    ) + 1
    df_full = df_full.loc[df_full.intensity > intensity_cut]
    df = df_full.dropna()  # .sample(frac=0.1)

    N = len(df)
    # N_alive_slitlets = len(np.unique(df.slitlet))
    N_alive_fibres = len(np.unique(df.fibre_number))

    # N_missing_fibres = N_fibres_total - N_alive_fibres
    # N_missing_slitlets = N_slitlets_total - N_alive_slitlets
    if verbose:
        print(f"\nWe have {N} measured arc lines in {N_alive_fibres} fibres.\n")

    wavelengths = df.loc[:, "wave"].values
    slitlet_numbers = df.slitlet.astype(int)
    fibre_numbers = df.fibre_number.astype(int)

    # # Find the fibres which are missing/turned off/broken/etc
    # missing_fibre_numbers = list(set(np.arange(N_fibres_total)) - set(fibre_numbers))
    # missing_slitlet_numbers = list(
    #     set(np.arange(N_slitlets_total)) - set(slitlet_numbers)
    # )

    # Standardise the X and Wavelength values
    x = df.x_pixel
    x_standardised = standardise(x)
    wave_standardised = (wavelengths - wavelengths.mean()) / wavelengths.std()
    # The y values have already been standardised per slitlet
    y_standardised = df.y_slitlet

    # Make the constants- we have a different constant for every fibre
    constants = np.array(
        [(fibre_numbers == i).astype(int) for i in range(1, N_fibres_total + 1)]
    ).T

    # Get the Chebyshev polynomial columns
    X = cheb.chebvander2d(x_standardised, y_standardised, [N_x, N_y])
    # And now make these on a per-slitlet basis, so all the coefficients we measure are per-slitlet
    X_values_per_slitlet = np.column_stack(
        [
            np.where(slitlet_numbers.values[:, None] == i, X, 0)
            for i in range(1, N_slitlets_total + 1)
        ]
    )

    # We now have one constant term per fibre (~720 terms) and (n_x + 1)(n_y + 1) Chebyshev polynomial coefficients per slitlet
    X2 = np.c_[constants, X_values_per_slitlet]

    return df, wave_standardised, X2


def fit_model(X, y, alpha=1e-3, fit_intercept=False):
    """
    Perform the fitting using an SkLearn Ridge model.

    Args:
        X (np.ndarray): The Design Matrix of row vectors used in the fitting.
        y (np.ndarray): The vector of observed wavelengths (which have been rescaled)
        alpha (float, optional): Ride regression regularisation value. Defaults to 1e-3.
        fit_intercept (bool, optional): Whether or not to fit the intercept term. Defaults to False, and should probably always be False...

    Returns:
        Sklearn.Model: the Sklearn model instance
    """
    #print("Doing the fitting...")

    model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    model.fit(X, y)
    #print("\tDone!\n")
    return model


def standardise(array):
    """Take an array and standardise the values to lie between -1 and 1

    Args:
        array (np.ndarray): Array of values to rescale.

    Returns:
        array: The rescaled array
    """
    return 2 * (array - array.max()) / (array.max() - array.min()) + 1


def get_predictions(model, X, wavelengths):
    """
    For a given set of coeffients, dot with a design matrix to make predictions.
    Multiply these predictions by the mean and standard deviation of the observed wavelengths.

    Args:
        model (Sklearn.Model): A fitted model, with a .coef_ attribute.
        X (np.ndarray): A design matrix
        wavelengths (np.ndarray): The original wavelength array. The predictions will be multiplied by the standard deviation of this array and the mean of this array will be added.

    Returns:
        np.ndarray: An array of wavelength predictions.
    """
    beta_hat = model.coef_

    predictions = (X @ beta_hat) * wavelengths.std() + wavelengths.mean()

    return predictions


def calculate_MSE(model, X, wavelengths):
    """
    Given a model, a design matrix and a set of observed wavelengths, calculate the mean-squared error of the fit.

    Args:
        model (Sklearn.Model): An Sklearn fitted model.
        X (np.ndarray): A design matrix.
        wavelengths (np.ndarray): Observed wavelength values.

    Returns:
        float: The Mean Squared Error of the predictions.
    """
    predictions = get_predictions(model, X, wavelengths)
    mse = np.sqrt(
        mean_squared_error(
            y_true=wavelengths,
            y_pred=predictions,
        )
    )
    return mse


def get_info(ccd_number):
    """
    Get various bits of information about an exposure, e.g. how many slitlets there are, how many fibres there are, etc.

    Args:
        ccd_number (str): The CCD number of the arc exposure. Will be converted to a str.

    Raises:
        NameError: If the CCD number isn't one of "1", "2", "3", "4"

    Returns:
        tuple: A tuple of ccd_name, N_slitlets_total, N_fibres_total, N_fibres_per_slitlet
    """
    ccd_number = str(ccd_number)
    if ccd_number in ["3", "4"]:
        ccd_name = "SPECTOR"
        N_slitlets_total = 19
        N_fibres_total = 855
        N_fibres_per_slitlet = 45
    elif ccd_number in ["1", "2"]:
        ccd_name = "AAOmega"
        N_slitlets_total = 13
        N_fibres_total = 819
        N_fibres_per_slitlet = 63
    else:
        raise NameError(
            f"CCD number must be '1', '2', '3', '4' and of type string. Currently {ccd_number}, {type(ccd_number)}"
        )
    return ccd_name, N_slitlets_total, N_fibres_total, N_fibres_per_slitlet


def plot_residuals(df, predictions, wavelengths):
    """Plot the residuals between a fit and the original data

    Args:
        df (pd.DataFrame): The dataframe from set_up_arc_fitting.
        predictions (np.ndarray): Array of predictions from the model.
        wavelengths (np.ndarray): Original observed wavelength array.

    Returns:
        tuple: A tuple of fig, ax
    """
    # Get the residuals
    residuals = wavelengths - predictions
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(13, 7), constrained_layout=True)

    axs[0, 0].hist(residuals, bins="fd")
    axs[0, 0].set_title(rf"$\sigma$(Wavelength) = {residuals.std():.3f} A")
    axs[0, 0].set_xlabel("Residuals ($\\mathrm{\\AA}$)")

    axs[0, 1].scatter(
        df.x_pixel, df.y_pixel, c=residuals, vmin=-0.5, vmax=0.5, rasterized=True
    )
    axs[0, 1].set_xlabel("Detector x pixel")
    axs[0, 1].set_ylabel("Detector y pixel")

    axs[1, 0].scatter(df.x_pixel, residuals, c=df.slitlet.astype(int), cmap="prism")
    axs[1, 0].set_xlabel("Detector x pixel")
    axs[1, 0].set_ylabel("Residuals ($\\mathrm{\\AA}$)")

    axs[1, 1].scatter(df.y_slitlet, residuals, c=df.slitlet.astype(int), cmap="prism")
    axs[1, 1].set_xlabel("Detector y pixel")
    axs[1, 1].set_ylabel("Residuals ($\\mathrm{\\AA}$)")

    # fig.savefig(plot_filename, bbox_inches="tight")
    return fig, axs


def save_parameters(output_file, df, model, N_params_per_slitlet, mse, arc_name):
    """
    Save the fitted parameters from a model to a netcdf output file, using the xarray package.

    Args:
        output_file (str): A file to save the results to. Must end in .nc
        df (pd.DataFrame): A dataframe from set_up_arc_fitting
        model (sklearn.Model): A fitted model. Must have a .coef_ attribute.
        N_params_per_slitlet (int): Number of params per slitlet (N_x + 1) * (N_y + 1)
        mse (float): The mean-squared error of the fit
        arc_name (str): The name of the arc file

    Returns:
        xr.dataset: The parameters, saved as an xarray dataset.
    """
    ccd_number = str(df.ccd.unique()[0])
    ccd_name, N_slitlets_total, N_fibres_total, N_fibres_per_slitlet = get_info(
        ccd_number
    )

    # Get the coefficients
    beta_hat = model.coef_

    # Save the outputs as an xarray array
    fibre_constants = xr.DataArray(
        beta_hat[:N_fibres_total],
        dims=("fibre"),
        coords=dict(fibre=np.arange(1, N_fibres_total + 1)),
        name="fibre constant",
    )
    slitlet_params = xr.DataArray(
        beta_hat[N_fibres_total:].reshape(N_slitlets_total, N_params_per_slitlet),
        dims=("slitlet", "polynomial_parameter"),
        coords=dict(
            slitlet=np.arange(1, N_slitlets_total + 1),
            polynomial_parameter=np.arange(N_params_per_slitlet),
        ),
        name="slitlet parameters",
    )
    mse_values = xr.DataArray(mse, name="MSE")
    dataset = xr.Dataset(
        data_vars=dict(
            fibre_constants=fibre_constants,
            slitlet_params=slitlet_params,
            mse=mse_values,
        ),
        coords=dict(arc_ID=arc_name, ccd=int(ccd_number)),
    )
    dataset.to_netcdf(output_file)

    return dataset


def set_up_WAVELA_predictions(tlm_filename, ccd_number, N_x, N_y):
    """
    Set up the needed arrays in order to make predictions to create a new WAVELA array

    Args:
        tlm_filename (str): The filename of the tramline map
        ccd_number (str): The CCD the data was observed in. Should be 1, 2, 3 or 4
        N_x (int): The polynomial order in the x direction.
        N_y (int): The polynomial order in the y direction.
    """
    # Load the tlm map
    tlm, N_fibres_total, N_pixels_x = load_tlm_map(tlm_filename)

    ccd_name, N_slitlets_total, N_fibres_total, N_fibres_per_slitlet = get_info(
        ccd_number
    )

    y_values = tlm.ravel().astype(float)  # Casting to float64 is important
    x_values = np.tile(np.arange(N_pixels_x), N_fibres_total).ravel()
    fibre_numbers = np.arange(1, N_fibres_total + 1).repeat(N_pixels_x)
    slitlet_numbers = (
        N_slitlets_total - np.ceil(fibre_numbers / N_fibres_per_slitlet) + 1
    )

    df_predict = pd.DataFrame(
        data=dict(
            x_pixel=x_values,
            y_pixel=y_values,
            fibre_number=fibre_numbers,
            slitlet=slitlet_numbers,
        )
    )

    # Add some columns to the predict dataframe
    df_predict["ccd"] = ccd_number
    df_predict["intensity"] = 100
    df_predict["wave"] = 0
    df_predict.loc[:10, "wave"] = 10

    df_predict, wave_standardised, X2 = set_up_arc_fitting(
        df_predict, N_x=N_x, N_y=N_y, intensity_cut=0
    )

    return df_predict, wave_standardised, X2, N_pixels_x, N_fibres_total
