"""
Sree implemented Sam Vaughan's 2D arc modelling code into Hector repo.
Visit https://dev.aao.org.au/datacentral/hector/Hector-Wave-Cal for further details.
fit_arc_model is the modularized one of Hector-Wave-Cal/workflow/scripts/fit_arc_model_from_command_line.py
fit_arc_model_utils is a copy of Hector-Wave-Cal/workflow/scripts/utils.py
"""

import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path
import scipy.interpolate as interpolate
from astropy.io import fits
from astropy.table import Table
import numpy.polynomial.chebyshev as cheb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages


def load_tlm_map(tlm_filename):
    """
    Given a TLM map, load the data and return the number of fibres and the number of x pixels

    Args:
        tlm_filename (str): Filename of a tramline map fits file

    Returns:
        tuple: a tuple of the TLM data, the number of fibres and the number of x pixels
    """
    # Load the tlm maps
    tlm_map = fits.open(tlm_filename)
    tlm = tlm_map["PRIMARY"].data
    N_fibres, N_pixels = tlm.shape

    return tlm, N_fibres, N_pixels


def read_arc(
    arcdata_filename, tlm_filename, reduced_arc_filename, return_column_subset=True, verbose=True, global_fit=True):
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
        global_fit (bool): if True, then force data to act as one slitlet and fit the entire
                           frame in one go.

    Returns:
        pd.DataFrame: data frame that contains the following columns:
                "x_pixel",
                "y_pixel",
                "wave",
                "fibre_number",
                "intensity",
                "linewidth",
                "slitlet",
                "ccd",
                "file_id",
        Mostly taken from arclist.dat file, arc frame and tlm (or calculations from these).

    """

    # Load the arcfits.dat file
    datafile = Path(arcdata_filename)

    # Load the tlm map
    tlm, N_fibres, N_pixels = load_tlm_map(tlm_filename)

    if (verbose):
        print('Number of fibres and number of pixels:',N_fibres,N_pixels)
    
    # Load the arc file
    arc_frame = fits.open(reduced_arc_filename)
    fibre_df = Table(arc_frame["FIBRES_IFU"].data).to_pandas()

    if (verbose):
        print('Fibre table:')
        print(fibre_df)
    
    # Get the file ID
    #stem = datafile.stem
    #print(datafile)
    #print(stem)
    # this version to get file ID and CCD only seems to work for specific
    # directory structures:
    #file_id = stem.split("_")[1]
    #ccd = int(file_id[5])
    # instead, base on standard arc filename:
    arcfile = Path(reduced_arc_filename)
    file_id = arcfile.stem
    # remove "red" from file stem:
    file_id = file_id.replace('red','')
    ccd = int(file_id[5])
    if (verbose):
        print(file_id,', CCD number:',ccd)

    # Load the arc .dat file
    data, fibre_numbers, column_names = load_arc_data_file(datafile)
    df = pd.DataFrame(data, columns=column_names)
    
    #df["fibre_number"] = np.repeat(fibre_numbers, N_arc_lines)
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

    # copy slitlet IDs to a second column in case we overwrite them for
    # calculating the global fit:
    df['slitlet_orig'] = df['SLITLET']
    
    # force there to be just one slitlet for a global fit:
    if (global_fit):
        df['SLITLET'] = 1
    
    # do some renaming and only    certain columns
    df = df.rename(
        dict(INTEN="intensity", LNEWID="linewidth", SLITLET="slitlet", CCD="ccd"),
        axis=1,
    )

    # now use the tlm to find the min and max y values for each slitlet, so that
    # we can rescale them.  We do this based on the tlm as it has to be the same for both
    # the arclines list and the WAVELA file.  Here we need to pass the data from the full
    # frame (including bad fibres), as if the number of fibres change (e.g. some fibres
    # not used), then the range will be different:

    # get the slitlet info, based on the fibre table and tlm:
    slitlet_info = get_slitlet_info(fibre_df,tlm,verbose=verbose)
    # now assign slitlet info for each row in the data:
    s_min,s_max = get_s_minmax(df,slitlet_info,verbose=verbose)


    # only return some cols:
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
                "slitlet_orig"
            ],
        ]

    return df,N_pixels,s_min,s_max,slitlet_info


def load_arc_from_db(con, arc_name, verbose=True):
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

def get_slitlet_info(df,tlm,verbose=True):
    """calculate the min and max y values for each slitlet to enable scaling
    of the slitlet y locations.  This calculation is done based on the tlm and 
    fibre table so it is a consistent scaling for both the arc data and the 
    WAVELA prediction.

    Args:
       df: Pandas data frame of the fibre table.
       tlm: 2D array of tramline map
       verbose: logical flag to control output to terminal

    Returns:
       df_slitlet_info_u: a data frame with a row for each slitlet 
    """

    if (verbose):
        print('Getting slitlet info...')

    # get the first and last fibre for each slitlet.  Output of this
    # is a pandas series (a single column, but with additional axis labels).
    # note that the fibre number is "SPEC_ID" in the fibres tabe:
    fib_min = df.groupby("SLITLET")["SPEC_ID"].min()[df.SLITLET]
    fib_max = df.groupby("SLITLET")["SPEC_ID"].max()[df.SLITLET]
    
    #generate versions with just unique values to speed up calc:
    fib_min_u = fib_min.drop_duplicates()
    fib_max_u = fib_max.drop_duplicates()

    #convert the unique fib min/max pandas series to a data frame so we
    # can add the locations to it:
    df_fib_min_u = fib_min_u.to_frame()
    df_fib_max_u = fib_max_u.to_frame()
    # rename col to unique name:
    df_fib_min_u.rename(columns={"SPEC_ID": "fib_min"},inplace=True)
    df_fib_max_u.rename(columns={"SPEC_ID": "fib_max"},inplace=True)
    
    # now generate a data frame that can hold all the unique data, including
    # the empty columns that we will fill with the s_min and s_max values:
    df_slitlet_info_u = df_fib_min_u.copy()
    df_slitlet_info_u['fib_max'] = df_fib_max_u['fib_max']
    df_slitlet_info_u['s_min'] = ""
    df_slitlet_info_u['s_max'] = ""

    # rename index to be consitent with rest of code:
    df_slitlet_info_u.index.names = ['slitlet']
        
    # calculate the s_min and s_max for each slitlet:
    for index,row in df_slitlet_info_u.iterrows():
        # offset by 1 for python array indexing:
        fib1 = row['fib_min']-1
        fib2 = row['fib_max']-1
        # get the min and max from the tlm data:
        df_slitlet_info_u.at[index,'s_min'] = np.nanmin(tlm[fib1:fib2,:])
        df_slitlet_info_u.at[index,'s_max'] = np.nanmax(tlm[fib1:fib2,:])
        
    if (verbose):
        print('unique slitlet min and max from get_slitlet_info:')
        print(df_slitlet_info_u)
        
    
    return df_slitlet_info_u

def get_s_minmax(df,df_slitlet,verbose=True):
    """Get the min and max y values for each entry in adata frame, based
    on the slitlet info that was previously calculated by get_slitlet_info()
    
    Args:
       df: Pandas data frame of the fibre data.
       df_slitlet: Pandas data frame of slitlet info
       verbose: logical flag to print to screen or not.

    Returns:
       tuple: a tuple of the smin and smax data
    """

    # get length of data frame:
    ndf = len(df.index)
    if (verbose):
        print('number of rows in df:',ndf)

    # final output arrays:
    s_min = np.empty(ndf)
    s_max = np.empty(ndf)

    # now populate the per-fibre arrays with the s_min/s_max values
    for i in range(ndf):
        sl = df['slitlet'][i]
        
        #print(i,sl,df_fib_info_u.loc[sl,'fib_min'],df_fib_info_u.loc[sl,'fib_max'],df_fib_info_u.loc[sl,'s_min'],df_fib_info_u.loc[sl,'s_max'])
        s_min[i] = df_slitlet.loc[sl,'s_min']
        s_max[i] = df_slitlet.loc[sl,'s_max']

    if (verbose):
        print('s_min: ',s_min)
        print('s_max: ',s_max)
        
    return s_min,s_max
    
       
def get_s_minmax_old(df,tlm,ccd_number):
    """NOT USED!!! calculate the min and max y values for each slitlet to enable scaling
    of the slitlet y locations.  This calculation is done based on the tlm so 
    it is a consistent scaling for both the arc data and the WAVELA prediction.

    Args:
       df: Pandas data frame of the fibre data.
       tlm: 2D array of tramline map
       ccd: CCD number

    Returns:
       tuple: a tuple of the smin and smax data
    """

    print('Calculating s_min and s_max')

    nfib,npix = np.shape(tlm)
    print('npix,nfib for TLM: ',npix,nfib)
    ndf = len(df.index)
    print('number of rows in df:',ndf)

    ccd_name, N_slitlets_total, N_fibres_total, N_fibres_per_slitlet = get_info(
        ccd_number)
    
    # get the first and last fibre for each slitlet.  Output of this
    # is a pandas series (a single column, but with additional axis labels: 
    fib_min = df.groupby("slitlet")["fibre_number"].min()[df.slitlet]
    fib_max = df.groupby("slitlet")["fibre_number"].max()[df.slitlet]
    
    #generate versions with just unique values to speed up calc:
    fib_min_u = fib_min.drop_duplicates()
    fib_max_u = fib_max.drop_duplicates()
    nu = np.size(fib_min_u)

    #convert the unique fib min/max pandas series to a data frame so we
    # can add the locations to it:
    df_fib_min_u = fib_min_u.to_frame()
    # rename col to unique name:
    df_fib_min_u.rename(columns={"fibre_number": "fib_min"},inplace=True)
    df_fib_max_u = fib_max_u.to_frame()
    df_fib_max_u.rename(columns={"fibre_number": "fib_max"},inplace=True)

    # now generate a data frame that can hold all the unique data:
    df_fib_info_u = df_fib_min_u.copy()
    df_fib_info_u['fib_max'] = df_fib_max_u['fib_max']
    df_fib_info_u['s_min'] = ""
    df_fib_info_u['s_max'] = ""
    
    # final output arrays:
    s_min = np.empty(ndf)
    s_max = np.empty(ndf)

    # for each fibre, calculate the s_min and s_max for that slitlet:
    for index,row in df_fib_info_u.iterrows():
        # offset by 1 for python array indexing:
        fib1 = row['fib_min']-1
        fib2 = row['fib_max']-1
        # calculate the first and last fibre number based on the CCD info.
        # we need to do this as sometimes fibres are not used and the 'groupby'
        # method above will not work in this case.  Note -1 for python array indexes:
        fib1a = int((N_slitlets_total-index)*N_fibres_per_slitlet+1)-1
        fib2a = int((N_slitlets_total-index+1)*N_fibres_per_slitlet)-1
        # use the fixed up fibre ranges:
        df_fib_info_u.at[index,'s_min'] = np.nanmin(tlm[fib1a:fib2a,:])
        df_fib_info_u.at[index,'s_max'] = np.nanmax(tlm[fib1a:fib2a,:])
        print('slitlet: ',index,fib1+1,fib2+1,fib1a+1,fib2a+1,fib2-fib1+1,fib2a-fib1a+1,df_fib_info_u.at[index,'s_min'],df_fib_info_u.at[index,'s_max'])
        
    print('min of tlm of fib 1:',np.nanmin(tlm[0,:]))

    print('unique slitlet min and max:')
    print(df_fib_info_u)

    
    # now populate the per-fibre arrays with the s_min/s_max values
    for i in range(ndf):
        sl = df['slitlet'][i]
        
        #print(i,sl,df_fib_info_u.loc[sl,'fib_min'],df_fib_info_u.loc[sl,'fib_max'],df_fib_info_u.loc[sl,'s_min'],df_fib_info_u.loc[sl,'s_max'])
        s_min[i] = df_fib_info_u.loc[sl,'s_min']
        s_max[i] = df_fib_info_u.loc[sl,'s_max']
        
    return s_min,s_max
    
       

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
        - The number of arc lines found in each fibre (no longer returned)
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
                # reset nlines and fibno for this fibre:
                nlines = 0
                fibno = int(line.split()[2])
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
                # get row data and also append the fibre number to the end of the row.
                # do this as sometimes the fibres have no data so the N_arc_lines
                # gets messed up.  if there are no lines, then the "# fit parameters..."
                # line is not in the file for that fibre
                row = [float(value) for value in line.split()]
                row.append(fibno)
                #data.append([float(value) for value in line.split()])
                data.append(row)

    # append fibre_number label:
    column_names.append("fibre_number")
                
    return data, fibre_numbers, column_names


def set_up_arc_fitting(df_full, N_x, N_y, N_pixels, s_min, s_max,intensity_cut=10,verbose=True,firstpass=True):
    """
    Set up all the arrays and things we need for the arc fitting.

    Args:
        df_full (pd.DataFrame): A Dataframe loaded from either read_arc or load_arc_from_db.
        N_x (int): Polynomial order in the x direction.
        N_y (int): Polynomial order in the y direction.
        N_pixels (int): Number of pixels in the x direction (from TLM).
        intensity_cut (int): Ignore all arc lines fainter than this value. Default is 10.
        verbose: Boolian flag controlling output to terminal
        firstpass: Boolian flag to say whether this is the first time we are running
                   the routine (so normalization is needed), or its a later pass, that
                   is used for clipping and doesn't need the rescaling

    Returns:
        tuple:
    """
    ccd_number = str(df_full.ccd.unique()[0])
    ccd_name, N_slitlets_total, N_fibres_total, N_fibres_per_slitlet = get_info(
        ccd_number
    )

    # make a copy of 
    
    # Ignore any duplicated columns we may have- e.g if an arc was added twice
    df_full = df_full.drop_duplicates()

    # Ignore NaNs
    # these s_min, s_max values that are used for the scaling of the slitlets
    # needs to be the same for both the input data and the full WAVELA predictions
    # so for both need to base this on the TLM. 
    #s_max = df_full.groupby("slitlet")["y_pixel"].max()[df_full.slitlet]
    #s_min = df_full.groupby("slitlet")["y_pixel"].min()[df_full.slitlet]


    # These are y values **within a slitlet**, normalized to the min and max values
    # within each slit:
    # these numbers are passed from an earlier routine to make sure they are consistent:
    df_full["y_slitlet"] = (
        2 * (df_full["y_pixel"] - s_max) / (s_max - s_min)
    ) + 1
    df_full = df_full.loc[df_full.intensity > intensity_cut]
    df = df_full.dropna()  # .sample(frac=0.1)

    N = len(df)
    # N_alive_slitlets = len(np.unique(df.slitlet))
    N_alive_fibres = len(np.unique(df.fibre_number))

    # N_missing_fibres = N_fibres_total - N_alive_fibres
    # N_missing_slitlets = N_slitlets_total - N_alive_slitlets
    if (verbose):
        print(f"\nWe have {N} measured arc lines in {N_alive_fibres} fibres.\n")

    wavelengths = df.loc[:, "wave"].values
    slitlet_numbers = df.slitlet.astype(int)
    fibre_numbers = df.fibre_number.astype(int)

    # # Find the fibres which are missing/turned off/broken/etc
    # missing_fibre_numbers = list(set(np.arange(N_fibres_total)) - set(fibre_numbers))
    # missing_slitlet_numbers = list(
    #     set(np.arange(N_slitlets_total)) - set(slitlet_numbers)
    # )

    # Standardise the X and Wavelength values.   Standardise means scale between -1 and +1.
    # see standardise() function for details.
    x = df.x_pixel
    #x_standardised = standardise(x)
    # need to use this version to standardise, otherwise the fit and the WAVELA calc can
    # be different:
    x_standardised = standardise_minmax(x,0.0,float(N_pixels-1))
    if (verbose):
        print('x and x_standardized:')
        print(x)
        print(x_standardised)
    # SMC not clear why wavelengths is standardizd by dividing by std()?
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
    print("Doing the fitting...")

    model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    model.fit(X, y)
    print("\tDone!\n")
    return model


def standardise(array):
    """Take an array and standardise the values to lie between -1 and 1

    Args:
        array (np.ndarray): Array of values to rescale.

    Returns:
        array: The rescaled array
    """
    return 2 * (array - array.max()) / (array.max() - array.min()) + 1

def standardise_minmax(array,minval,maxval):
    """Take an array and standardise the values to lie between -1 and 1.  For this version use input values to
    define max and min to normalize to, so that we can have the same normalization as other arrays.


    Args:
        array (np.ndarray): Array of values to rescale.
        minval (float): 
        maxval (float):

    Returns:
        array: The rescaled array
    """
    return 2 * (array - maxval) / (maxval - minval) + 1


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


def calculate_RMS(model, X, wavelengths):
    """
    Given a model, a design matrix and a set of observed wavelengths, calculate the root mean-squared error of the fit.

    Args:
        model (Sklearn.Model): An Sklearn fitted model.
        X (np.ndarray): A design matrix.
        wavelengths (np.ndarray): Observed wavelength values.

    Returns:
        float: The Mean Squared Error of the predictions.
    """
    predictions = get_predictions(model, X, wavelengths)
    
    rmse = np.sqrt(
        mean_squared_error(
            y_true=wavelengths,
            y_pred=predictions,
        )
    )
    return rmse


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


def plot_residuals(df, predictions, wavelengths,plot_filename='residual_plots.pdf',debug=False,pixscale=None):
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
    xp = np.array(df.x_pixel)
    yp = np.array(df.y_pixel)

    # get the unique wavelength values:
    wav_uni = np.unique(wavelengths)
    print('Unique wavelengths in arclist (individual lines):')
    nuni = np.size(wav_uni)
    res_mean = np.zeros(nuni)
    res_med = np.zeros(nuni)
    res_std = np.zeros(nuni)
    res_n = np.zeros(nuni)
    print('Number of unique wavelengths: ',nuni)
    # now loop over unique lines to se what the stats look like for each one:
    print('Residuals per line.  Last col is how many sigma the mean is from zero:')
    print(' lam     mean   median  stdev   n   nsig x_mean')
    for i in range(nuni):
        lam = wav_uni[i]
        idx = np.where(wavelengths==lam)
        res_mean[i] = np.nanmean(residuals[idx])
        res_med[i] = np.nanmedian(residuals[idx])
        res_std[i] = np.nanstd(residuals[idx])
        res_n[i] = np.size(residuals[idx])
        mean_x = np.nanmean(xp[idx])
        if ((res_std[i] > 0.0) & (res_n[i] > 1)):
            nsig = res_mean[i]/(res_std[i]/np.sqrt(res_n[i]))
        else:
            nsig = 0.0
        print('{0:7.2f} {1:7.4f} {2:7.4f} {3:7.4f} {4:4d} {5:7.2f} {6:6.1f}'.format(lam,res_mean[i],res_med[i],res_std[i],int(res_n[i]),nsig,mean_x))
    

    
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(13, 7), constrained_layout=True)

    stdev = residuals.std()
    aresiduals = np.abs(residuals)

    # set range to plot:
    r1 = -1.0*np.nanpercentile(aresiduals,99.0)
    r2 = -1.0*r1
    
    axs[0, 0].hist(residuals, bins="fd")
    if (pixscale):
        title = "$\sigma$(Wavelength) = {0:.3f} A.  1 pixel is {1:4.2f} A.".format(residuals.std(),pixscale)
    else:
        title = "$\sigma$(Wavelength) = {0:.3f} A.".format(residuals.std())
    axs[0, 0].set_title(title)
    axs[0, 0].set_xlabel("Residuals ($\mathrm{\AA}$)")
    axs[0, 0].set(xlim=[-1.0,1.0])
    axs[0, 0].axvline(0.0,linestyle='--',color='k')
    if (pixscale):
        axs[0, 0].axvline(pixscale,linestyle=':',color='k')
        axs[0, 0].axvline(-1.0*pixscale,linestyle=':',color='k')
        axs[0, 0].axvline(0.1*pixscale,linestyle=':',color='k')
        axs[0, 0].axvline(-0.1*pixscale,linestyle=':',color='k')

    plot1 = axs[0, 1].scatter(
        df.x_pixel, df.y_pixel, c=residuals, vmin=r1, vmax=r2, s=2, rasterized=True, cmap="RdYlBu")
    axs[0, 1].set_xlabel("Detector x pixel")
    axs[0, 1].set_ylabel("Detector y pixel")
    #axs[0,1].set(xlim=[1000,1060],ylim=[930,990])
    plt.colorbar(plot1,ax=axs[0,1])

    # output values to test:
    if (debug):
        nn = len(df)
        print(df)
        print('fibre 188:')
        for i in range(nn):
            fb = df.iloc[i]['fibre_number']
            #if ((xp>1000) & (xp<1060) & (yp>930) & (yp<990)):
            if (fb==188):
                xp = df.iloc[i]['x_pixel']
                yp = df.iloc[i]['y_pixel']
                print(i,xp,yp,wavelengths[i],predictions[i],residuals[i],df.iloc[i]['fibre_number'],df.iloc[i]['slitlet'],df.iloc[i]['y_slitlet'])

    
    axs[1, 0].scatter(df.x_pixel, residuals, c=df.slitlet.astype(int), cmap="prism",s=2)
    axs[1, 0].set_xlabel("Detector x pixel")
    axs[1, 0].set_ylabel("Residuals ($\mathrm{\AA}$)")
    axs[1, 0].set(ylim=[r1,r2])

    axs[1, 1].scatter(df.y_pixel, residuals, c=df.slitlet.astype(int), cmap="prism",s=2)
    axs[1, 1].set_xlabel("Detector y pixel")
    axs[1, 1].set_ylabel("Residuals ($\mathrm{\AA}$)")
    axs[1, 1].set(ylim=[r1,r2])

    # break up image into bins and calc mean/median:
    # calc boundaries:
    nbx= 16
    nby= 16
    x_grid = np.zeros((nbx,nby))
    y_grid = np.zeros((nbx,nby))
    mean_grid = np.zeros((nbx,nby))
    stder_grid = np.zeros((nbx,nby))
    xmin = np.nanmin(xp)
    xmax = np.nanmax(xp)
    ymin = np.nanmin(yp)
    ymax = np.nanmax(yp)
    xstep = (xmax-xmin)/float(nbx)
    ystep = (ymax-ymin)/float(nby)
    for j in range(nbx):
        x1 = xmin + xstep*j
        x2 = x1 + xstep
        # plot lines for grid:
        for k in range(nby):
            y1 = ymin + ystep*k
            y2 = y1 + ystep
            idxb = np.where(((xp > x1) & (xp < x2) & (yp > y1) & (yp < y2)))
            nb = np.size(residuals[idxb])
            #med = np.nanmedian(residuals[idxb])
            # 1 sigma percentile ranges:
            # 0.1587 and 0.8413
            #per1 = np.nanpercentile(residuals[idxb],15.87)
            #per2 = np.nanpercentile(residuals[idxb],84.13)
            #medstd = np.abs(per2-per1)
            #medstder = medstd/np.sqrt(nb)
            if (nb > 0):
                mean = np.nanmean(residuals[idxb])
                std = np.nanstd(residuals[idxb])
                stder = std/np.sqrt(nb)
                mean_grid[j,k] = mean
                stder_grid[j,k] = stder
            else:
                mean_grid[j,k] = np.nan
                stder_grid[j,k] = np.nan
                    
            x_grid[j,k] = (x1+x2)/2
            y_grid[j,k] = (y1+y2)/2

    # plot the mean points in the grid on plots:
    for j in range(nby):
        lab = 'y={0:4d}'.format(int(y_grid[0,j]))
        axs[1, 0].errorbar(x_grid[:,j],mean_grid[:,j],yerr=stder_grid[:,j],linestyle='-',marker='x',label=lab)
    for j in range(nbx):
        lab = 'x={0:4d}'.format(int(x_grid[j,0]))
        axs[1, 1].errorbar(y_grid[j,:],mean_grid[j,:],yerr=stder_grid[j,:],linestyle='-',marker='x',label=lab)
        
    axs[1, 0].axhline(0.0,color='k',linestyle=':')
    axs[1, 1].axhline(0.0,color='k',linestyle=':')
    if (pixscale):
        axs[1, 0].axhline(0.1*pixscale,color='k',linestyle=':')
        axs[1, 0].axhline(-0.1*pixscale,color='k',linestyle=':')
        axs[1, 0].text(0.0,0.11*pixscale,'0.1 pix',horizontalalignment='left',color='k')
        axs[1, 1].axhline(0.1*pixscale,color='k',linestyle=':')
        axs[1, 1].axhline(-0.1*pixscale,color='k',linestyle=':')
        axs[1, 1].text(0.0,0.11*pixscale,'0.1 pix',horizontalalignment='left',color='k')

    axs[1, 0].legend(loc='upper center',bbox_to_anchor=(0.5, 1.05),fontsize=6,ncols=nby/4)

    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #      fancybox=True, shadow=True, ncol=5)
    axs[1, 1].legend(loc='upper center',bbox_to_anchor=(0.5, 1.05),fontsize=6,ncols=nby/4)
    #axs[1, 1].legend()
        
    # add a colourbar for the slitlet number?
    
    fig.savefig(plot_filename, bbox_inches="tight")

    # now generate a multi-page plots that show a zoom in for the
    
    pdf = PdfPages('slitlet_residuals.pdf')
    #
    nslitlet = np.size(np.unique(df.slitlet_orig))
    #nslitlet=13
    print(df)
    slitlet = np.array(df.slitlet_orig)
    # number of block to sub-divide slitlet into for averages:
    nbx= 5
    nby= 3

    usemedian = True
    
    for i in range(nslitlet):
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(1,1,1)
        idx = np.where(slitlet==(i+1))
        idxs = np.where(((slitlet==(i+1)) & (aresiduals>5.0*stdev)))
        # plot residuals
        plot1 = ax1.scatter(xp[idx],yp[idx], c=residuals[idx], vmin=-0.5, vmax=0.5, rasterized=True,cmap="RdYlBu",s=5)
        # mark residuals that are more than 5 sigma:
        plot2 = ax1.plot(xp[idxs],yp[idxs],'x',color='r')
        # calculate std dev for this slitlet:
        stdev_slitlet = residuals[idx].std()
        ax1.set_xlabel("Detector x pixel")
        ax1.set_ylabel("Detector y pixel")
        ax1.set_title("Slitlet "+str(i+1)+" std dev = "+f"{stdev_slitlet:7.4f}")
        plt.colorbar(plot1,ax=ax1)
        # break up slitlet into sections and calculate median/mean in each section:
        # calc boundaries:
        xmin = np.nanmin(xp[idx])
        xmax = np.nanmax(xp[idx])
        ymin = np.nanmin(yp[idx])
        ymax = np.nanmax(yp[idx])
        xstep = (xmax-xmin)/float(nbx)
        ystep = (ymax-ymin)/float(nby)
        for j in range(nbx):
            x1 = xmin + xstep*j
            x2 = x1 + xstep
            # plot lines for grid:
            ax1.axvline(x1,linestyle=':',color='k')
            for k in range(nby):
                y1 = ymin + ystep*k
                y2 = y1 + ystep
                # plot lines for grid (for first loop in x):
                if (j == 0):
                    ax1.axhline(y1,linestyle=':',color='k')
                    
                idxb = np.where(((slitlet==(i+1)) & (xp > x1) & (xp < x2) & (yp > y1) & (yp < y2)))
                nb = np.size(residuals[idxb])
                med = np.nanmedian(residuals[idxb])
                # 1 sigma percentile ranges:
                # 0.1587 and 0.8413
                per1 = np.nanpercentile(residuals[idxb],15.87)
                per2 = np.nanpercentile(residuals[idxb],84.13)
                medstd = np.abs(per2-per1)
                medstder = medstd/np.sqrt(nb)
                mean = np.nanmean(residuals[idxb])
                std = np.nanstd(residuals[idxb])
                stder = std/np.sqrt(nb)
                if (usemedian):
                    nsig = np.abs(mean/stder)
                    label = '{0:7.4f}\n+-{1:6.4f}'.format(med,medstder)
                else:
                    nsig = np.abs(med/stder)
                    label = '{0:7.4f}\n+-{1:6.4f}'.format(mean,stder)

                if (nsig < 2):
                    col = 'k'
                elif ((nsig <3) & (nsig>2)):
                    col = 'b'
                else:
                    col = 'r'
                
                ax1.text((x1+x2)/2,(y1+y2)/2,label,horizontalalignment='center',color=col)
        ax1.axvline(xmax,linestyle=':',color='k')
        ax1.axhline(ymax,linestyle=':',color='k')
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

    pdf.close()


    
    return fig, axs


def save_parameters(output_file, df, model, N_params_per_slitlet, rms, arc_name):
    """
    Save the fitted parameters from a model to a netcdf output file, using the xarray package.

    Args:
        output_file (str): A file to save the results to. Must end in .nc
        df (pd.DataFrame): A dataframe from set_up_arc_fitting
        model (sklearn.Model): A fitted model. Must have a .coef_ attribute.
        N_params_per_slitlet (int): Number of params per slitlet (N_x + 1) * (N_y + 1)
        rms (float): The root mean-squared error of the fit
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
    rms_values = xr.DataArray(rms, name="RMS")
    dataset = xr.Dataset(
        data_vars=dict(
            fibre_constants=fibre_constants,
            slitlet_params=slitlet_params,
            rms=rms_values,
        ),
        coords=dict(arc_ID=arc_name, ccd=int(ccd_number)),
    )
    dataset.to_netcdf(output_file)

    return dataset


def set_up_WAVELA_predictions(tlm_filename, ccd_number, N_x, N_y, slitlet_info,verbose=True,global_fit=False):
    """
    Set up the needed arrays in order to make predictions to create a new WAVELA array

    Args:
        tlm_filename (str): The filename of the tramline map
        ccd_number (str): The CCD the data was observed in. Should be 1, 2, 3 or 4
        N_x (int): The polynomial order in the x direction.
        N_y (int): The polynomial order in the y direction.
        slitlet_info: Data frame with info on each slitlet for y scaling.
    """
    # Load the tlm map
    tlm, N_fibres_total, N_pixels_x = load_tlm_map(tlm_filename)

    ccd_name, N_slitlets_total, N_fibres_total, N_fibres_per_slitlet = get_info(
        ccd_number
    )

    # if a global fit (treating as one slitlet) then update the details above to handle this.
    if (global_fit):
        N_slitlets_total = 1
        N_fibres_per_slitlet = N_fibres_total
    
    y_values = tlm.ravel().astype(float)  # Casting to float64 is important
    # get x values.  Note that we need to use arange 1 to N_pixels_x+1 due
    # python starting at zero:  
    x_values = np.tile(np.arange(1,N_pixels_x+1), N_fibres_total).ravel()
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

    # get the s_min, s_max values:
    s_min,s_max = get_s_minmax(df_predict,slitlet_info,verbose=verbose)
    
    df_predict, wave_standardised, X2 = set_up_arc_fitting(
        df_predict, N_x=N_x, N_y=N_y, N_pixels=N_pixels_x, s_min=s_min, s_max=s_max, intensity_cut=0,verbose=verbose
    )

    return df_predict, wave_standardised, X2, N_pixels_x, N_fibres_total

def clip_data(X2,wave_standardised,wavelengths,predictions,rms,verbose=False):
    """
    Clip outliers from the fits from the data, so we can refit.

    Args:
      X2 (np.ndarray): A design matrix
      wave_standardised (np.ndarray): original wavelengths normalized for fitting.
      wavelengths (np.ndarray): The original wavelength array. The predictions will be multiplied by the standard deviation of this array and the mean of this array will be added.
      predictions (np.ndarray): array of model predictions
      rms (float): root mean square error of residuals.
    """

    # calculate absolute residuals:
    residuals = np.abs(wavelengths - predictions)

    # size of arrays:
    n = np.size(residuals)

    # make an array flagging good/bad values:
    good = np.ones((n), dtype=bool)
    good = np.where((residuals>5.0*rms),False,True)
    idx = np.where(good)
    ngood = good.sum()

    # use squeeze here, as for some reason the command hear adds an extra size 1 dimension
    X2_c = np.squeeze(X2[idx,:].copy())
    wave_standardised_c = wave_standardised[idx].copy()
    wavelengths_c = wavelengths[idx].copy()

    if (verbose):
        print('Done clipping...')
        print('Number of points:',n)
        print('Number of good points:',ngood)
        print('Percentage rejected:', 100*float(n-ngood)/float(n),' %')
    
    return X2_c, wave_standardised_c,wavelengths_c
    
    
    
