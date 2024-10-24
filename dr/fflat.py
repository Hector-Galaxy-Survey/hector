"""
This module addresses issues with fibre flat field frames

correct_bad_fibres() is used to identify fibres with 'bad' flat field values,
typical due to the presence of star in a twilight flat or an uncorrected
cosmic ray.

correct_bad_fibres takes as input a group of flats observed on the same field,
constructs a mean flat frame then calculates the residuals of each input frame
on a fibre-by-fibre basis (summing over the wavelength axis). Any fibres
identified as 'bad' (i.e. differing by more than 3 sigma from the mean) are
replaced with the average fibre profile.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import astropy.io.fits as pf

import numpy as np
import astropy.io.fits as pf

def correct_bad_fibres(path_list, debug=False):
    """Replace bad fibre flat values with an average
        Sree(9sep2024): corrected this script to be more efficient, fast and straightforward.
    """

    # Define array to store fibre flats
    n_file = len(path_list)
    n_fibre = pf.getval(path_list[0], 'NAXIS2', 'PRIMARY')
    n_spec = pf.getval(path_list[0], 'NAXIS1', 'PRIMARY')

    # Read in fibre flats from fits files to array
    fflats = np.array([pf.getdata(path, 'PRIMARY') for path in path_list])

    # Calculate the average fibre flat, residual images, and fibre residuals
    avg_fflat = np.nanmean(fflats, axis=0)
    residual_fflats = fflats - avg_fflat
    fibre_residuals = np.sqrt(np.nanmean(residual_fflats**2, axis=2))

    # Identify all 'bad' fibres that lie more than 3 sigma from the mean
    mean_residual = np.nanmean(fibre_residuals)
    sig_residual = np.nanstd(fibre_residuals)
    threshold = mean_residual + 3 * sig_residual

    # Initial identification of bad fibres
    bad_fibres_mask = fibre_residuals > threshold
    bad_fibres_index = np.argwhere(bad_fibres_mask)

    # Replace bad fibres iteratively until no bad fibres remain
    fflats_fixed = np.copy(fflats)
    while len(bad_fibres_index) > 0:
        for file_index, fibre_index in bad_fibres_index:
            fflats_fixed[file_index, fibre_index, :] = np.nan

        # Recompute average and residuals after replacing bad fibres
        avg_fixed = np.nanmean(fflats_fixed, axis=0)
        residual_fflats = fflats_fixed - avg_fixed
        fibre_residuals = np.sqrt(np.nanmean(residual_fflats**2, axis=2))

        # Update bad fibres mask and index
        bad_fibres_mask = fibre_residuals > (mean_residual + 4 * sig_residual)
        bad_fibres_index = np.argwhere(bad_fibres_mask)

    # Replace all bad fibres with the average for that fibre
    for file_index, fibre_index in np.argwhere(np.isnan(fflats_fixed)):
        fflats_fixed[file_index, fibre_index, :] = avg_fixed[fibre_index, :]

    # Write new fibre flat field values to file
    edited_files = np.unique(np.argwhere(np.isnan(fflats_fixed))[:, 0])
    for index in edited_files:
        with pf.open(path_list[index], mode='update') as hdulist:
            hdulist['PRIMARY'].data = fflats_fixed[index, :, :]
            hdulist.flush()

    if debug:
        for val in edited_files:
            ww = np.where(bad_fibres_index[:,0] == val)
            print(path_list[val])
            print(np.array(bad_fibres_index)[ww,1])
            print('--------')

    return
