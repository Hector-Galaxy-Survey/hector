import numpy as np
import scipy as sp
import os, sys
import matplotlib.pyplot as plt

from scipy.optimize import leastsq
import astropy.io.fits as fits
from astropy.table import Table
import math as Math

try:
    from bottleneck import median
    from bottleneck import nansum
except:
    from numpy import median
    from numpy import nansum

    print("Not Using bottleneck: Speed will be improved if you install bottleneck")


def comxyz(x, y, z):
    """Centre of mass given x, y and z vectors (all same size). x,y give position which has value z."""

    Mx = 0
    My = 0
    mass = 0

    for i in range(len(x)):
        Mx = Mx + x[i] * z[i]
        My = My + y[i] * z[i]
        mass = mass + z[i]

    com = (Mx / mass, My / mass)
    return com


class BundleFitter:
    """ Fits a 2d Gaussian with PA and ellipticity. Params in form (amplitude, mean_x, mean_y, sigma_x, sigma_y,
    rotation, offset). Offset is optional. To fit a circular Gaussian use (amplitude, mean_x, mean_y, sigma, offset),
    again offset is optional. To fit a Moffat profile use (amplitude, mean_x, mean_y, alpha, beta, offset), with
    offset optional. """

    def __init__(self, p, x, y, z, model='', weights=None):
        self.p_start = p
        self.p = p
        self.x = x
        self.y = y
        self.z = z
        self.model = model

        if weights is None:
            self.weights = sp.ones(len(self.z))
        else:
            self.weights = weights

        self.perr = 0.
        self.var_fit = 0.

        if model == 'gaussian_eps':
            # 2d elliptical Gaussian with offset.
            self.p[0] = abs(self.p[0])  # amplitude should be positive.
            self.fitfunc = self.f1

        elif model == 'gaussian_eps_simple':
            # 2d elliptical Gaussian witout offset.
            self.p[0] = abs(self.p[0])  # amplitude should be positive.
            self.fitfunc = self.f2

        elif model == 'gaussian_circ':
            # 2d circular Gaussian with offset.
            self.p[0] = abs(self.p[0])  # amplitude should be positive.
            self.fitfunc = self.f3

        elif model == 'gaussian_circ_simple':
            # 2d circular Gaussian without offset
            self.p[0] = abs(self.p[0])
            self.fitfunc = self.f4

        elif model == 'moffat':
            # 2d Moffat profile
            self.p[0] = abs(self.p[0])
            self.fitfunc = self.f5

        elif model == 'moffat_simple':
            # 2d Moffat profile without offset
            self.p[0] = abs(self.p[0])
            self.fitfun = self.f6

        else:
            raise Exception

    def f1(self, p, x, y):
        # f1 is an elliptical Gaussian with PA and a bias level.

        rot_rad = p[5] * sp.pi / 180  # convert rotation into radians.

        rc_x = p[1] * sp.cos(rot_rad) - p[2] * sp.sin(rot_rad)
        rc_y = p[1] * sp.sin(rot_rad) + p[2] * sp.cos(rot_rad)

        return p[0] * sp.exp(-(((rc_x - (x * sp.cos(rot_rad) - y * sp.sin(rot_rad))) / p[3]) ** 2 \
                               + ((rc_y - (x * sp.sin(rot_rad) + y * sp.cos(rot_rad))) / p[4]) ** 2) / 2) + p[6]

    def f2(self, p, x, y):
        # f2 is an elliptical Gaussian with PA and no bias level.

        rot_rad = p[5] * sp.pi / 180  # convert rotation into radians.

        rc_x = p[1] * sp.cos(rot_rad) - p[2] * sp.sin(rot_rad)
        rc_y = p[1] * sp.sin(rot_rad) + p[2] * sp.cos(rot_rad)

        return p[0] * sp.exp(-(((rc_x - (x * sp.cos(rot_rad) - y * sp.sin(rot_rad))) / p[3]) ** 2 \
                               + ((rc_y - (x * sp.sin(rot_rad) + y * sp.cos(rot_rad))) / p[4]) ** 2) / 2)

    def f3(self, p, x, y):
        # f3 is a circular Gaussian, p in form (amplitude, mean_x, mean_y, sigma, offset).
        return p[0] * sp.exp(-(((p[1] - x) / p[3]) ** 2 + ((p[2] - y) / p[3]) ** 2) / 2) + p[4]

    def f4(self, p, x, y):
        # f4 is a circular Gaussian as f3 but without an offset
        return p[0] * sp.exp(-(((p[1] - x) / p[3]) ** 2 + ((p[2] - y) / p[3]) ** 2) / 2)

    def f5(self, p, x, y):
        # f5 is a circular Moffat profile
        return p[0] * ((p[4] - 1.0) / np.pi / p[3] / p[3]) * (
                    1 + (((x - p[1]) ** 2 + (y - p[2]) ** 2) / p[3] / p[3])) ** (-1 * p[4]) + p[5]

    def f6(self, p, x, y):
        # f6 is a circular Moffat profile but without an offset
        return p[0] * ((p[4] - 1.0) / np.pi / p[3] / p[3]) * (
                    1 + (((x - p[1]) ** 2 + (y - p[2]) ** 2) / p[3] / p[3])) ** (-1 * p[4])

    def errfunc(self, p, x, y, z, weights):
        # If Moffat alpha of beta become unphysical return very large residual
        if (self.model == 'moffat') or (self.model == 'moffat_simple'):
            if (p[4] <= 0) or (p[3] <= 0):
                return np.ones(len(weights)) * 1e99
        return weights * (self.fitfunc(p, x, y) - z)

    def fit(self):

        self.p, self.cov_x, self.infodict, self.mesg, self.success = \
            leastsq(self.errfunc, self.p, \
                    args=(self.x, self.y, self.z, self.weights), full_output=1)

        var_fit = (self.errfunc(self.p, self.x, \
                                self.y, self.z, self.weights) ** 2).sum() / (len(self.z) - len(self.p))

        self.var_fit = var_fit

        if self.cov_x is not None:
            self.perr = sp.sqrt(self.cov_x.diagonal()) * self.var_fit

        if not self.success in [1, 2, 3, 4]:
            print("Fit Failed...")
            # raise ExpFittingException("Fit failed")

    def fwhm(self):
        if (self.model == 'moffat') or (self.model == 'moffat_simple'):
            psf = 2 * self.p[3] * np.sqrt(2 ** (1 / self.p[4]) - 1)
        elif (self.model == 'gaussian_circ') or (self.model == 'gaussian_circ_simple'):
            psf = self.p[3] * 2 * np.sqrt(2 * np.log(2))
        else:
            print('Unknown model, no PSF measured')
            psf = 0.0

        return psf

    def __call__(self, x, y):
        return self.fitfunc(self.p, x, y)


def fibre_integrator(fitter, diameter, pixel=False):
    """Edits a fitter's fitfunc so that it integrates over each SAMI fibre."""

    # Save the diameter; not used here but could be useful later
    fitter.diameter = diameter

    # Define the subsampling points to use
    n_pix = 5  # Number of sampling points across the fibre
    # First make a 1d array of subsample points
    delta_x = np.linspace(-0.5 * (diameter * (1 - 1.0 / n_pix)),
                          0.5 * (diameter * (1 - 1.0 / n_pix)),
                          num=n_pix)
    delta_y = delta_x
    # Then turn that into a 2d grid of (delta_x, delta_y) centred on (0, 0)
    delta_x = np.ravel(np.outer(delta_x, np.ones(n_pix)))
    delta_y = np.ravel(np.outer(np.ones(n_pix), delta_y))
    if pixel:
        # Square pixels; keep everything
        n_keep = n_pix ** 2
    else:
        # Round fibres; only keep the points within one radius
        keep = np.where(delta_x ** 2 + delta_y ** 2 < (0.5 * diameter) ** 2)[0]
        n_keep = np.size(keep)
        delta_x = delta_x[keep]
        delta_y = delta_y[keep]

    old_fitfunc = fitter.fitfunc

    def integrated_fitfunc(p, x, y):
        # The fitter's fitfunc will be replaced by this one
        n_fib = np.size(x)
        x_sub = (np.outer(delta_x, np.ones(n_fib)) +
                 np.outer(np.ones(n_keep), x))
        y_sub = (np.outer(delta_y, np.ones(n_fib)) +
                 np.outer(np.ones(n_keep), y))
        return np.mean(old_fitfunc(p, x_sub, y_sub), 0)

    # Replace the fitter's fitfunc
    fitter.fitfunc = integrated_fitfunc

    return


def centroid_gauss_fit(x, y, flux, Probe, microns=False, premask=False):
    # if str(Probe).rstrip() == 'A':
    #     flux[np.argmax(flux)] = np.median(flux)
    #     flux[np.argmax(flux)] = np.median(flux)
    # flux = flux / np.nanmax(flux)
    # com = comxyz(x, y, flux)  # **use good data within masking # centre-of-mass guess
    ind = np.argmax(flux, axis=None) # maximum flux point
    com = [x[ind], y[ind]]

    # Peak height guess could be closest fibre to com position.
    dist = (x - com[0]) ** 2 + (y - com[1]) ** 2  # distance between com and all fibres.

    # First guess at width of Gaussian - diameter of a core in degrees/microns.
    if microns:
        sigx = 105.0
        core_diam = 105.0
    else:
        sigx = 4.44e-4 * 3600
        core_diam = 4.44e-4 * 3600

    # Fit circular 2D Gaussians.
    p0 = [np.mean(flux[np.where(dist == np.min(dist))]), com[0], com[1], sigx, 0.0]
    gf = BundleFitter(p0, x, y, flux, model='gaussian_circ')
    fibre_integrator(gf, core_diam)
    gf.fit()
    # print(gf.perr)
    # print(gf.p[0], gf.p[1], gf.p[2], gf.p[3], gf.p[4])

    # if Probe == 'A':
    #     fig = plt.figure()
    #     plt.clf()
    #     ax = plt.subplot(111)
    #     ax.set_aspect('equal')
    #
    #     # Set up color scaling for fibre fluxes
    #     cm = plt.cm.get_cmap('RdYlBu')
    #     plt.scatter(x, y, c=flux / np.nanmax(flux))
    #     plt.plot(gf.p[1], gf.p[2], 'xr', ms=40)
    #     plt.plot(com[0], com[1], 'xg', ms=40)
    #     plt.show()

    # Make a linear grid to reconstruct the fitted Gaussian over.
    x_0 = np.min(x)
    y_0 = np.min(y)

    # dx should be 1/10th the fibre diameter (in whatever units)
    dx = sigx / 10.0

    xlin = x_0 + np.arange(100) * dx  # x axis
    ylin = y_0 + np.arange(100) * dx  # y axis

    # Reconstruct the model
    model = np.zeros((len(xlin), len(ylin)))
    # Reconstructing the Gaussian over the proper grid.
    for ii in range(len(xlin)):
        xval = xlin[ii]
        for jj in range(len(ylin)):
            yval = ylin[jj]
            model[ii, jj] = gf.fitfunc(gf.p, xval, yval)


    if (np.min(x) < gf.p[1] < np.max(x)) & (np.min(y) < gf.p[2] < np.max(y)):
        print('CentroidX = {}, CentroidY = {}, FWHM = {}, Offset = {}'.format(gf.p[1], gf.p[2], gf.p[3], gf.p[4]))
    else:
        # print('Centroid is outside the x-/y-range of the data')
        gf.p[1] = 0.0
        gf.p[2] = 0.0

    # calculate the sum of squares error
    square_of_diff = 0.0
    Nobs = 0.0
    for xi in range(len(x)):
        actual = flux[xi]
        predicted = gf.fitfunc(gf.p, x[xi], y[xi])

        square_of_diff = square_of_diff + (actual - predicted)**2.0
        Nobs += 1.0

    err = np.sqrt(square_of_diff / Nobs)
    print('err:')
    print(err)

    # Plot the test data as a 2D image and the fit as overlaid contours.   # RIGHT ONE TO KEEP
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # fig.suptitle(f"Probe: {Probe}, Error (RMS): {err}", fontsize=15)
    # plt.scatter(x, y, c=flux, cmap=plt.get_cmap('RdYlBu'))
    # plt.plot(gf.p[1], gf.p[2], 'xr', ms=40)
    # ax.contour(xlin, ylin, np.array(model).transpose(), colors='k')
    # # plt.show()
    # fig.savefig(f"Probe_{Probe}", bbox_inches='tight')
    # plt.close()

    # if Probe == 'A':
    #     print('model_fits:', gf.p[1], gf.p[2])
    #     fig12 = plt.figure()
    #     ax = plt.subplot(111)
    #     ax.set_aspect('equal')
    #     c = ax.pcolor(xlin, ylin, np.array(model).transpose(), cmap='RdBu')
    #     plt.plot(gf.p[1], gf.p[2], 'xk', ms=40)
    #     plt.plot(com[0], com[1], 'xr', ms=40)
    #     ax.set_title('pcolor1')
    #     fig12.colorbar(c, ax=ax)
    #     plt.show()
    #     sys.exit()

    return gf, flux, xlin, ylin, model


def centroid_gauss_fit_flux_weighted(x, y, flux, Probe, microns=False, premask=False, make_plots=False):
    from scipy.interpolate import griddata
    from photutils.centroids import centroid_com, centroid_quadratic, centroid_1dg, centroid_2dg

    """
    photutils.centroids provides several functions to calculate the centroid of a single source:

    centroid_com(): Calculates the object “center of mass” from 2D image moments.
    centroid_quadratic(): Calculates the centroid by fitting a 2D quadratic polynomial to the data.
    centroid_1dg(): Calculates the centroid by fitting 1D Gaussians to the marginal x and y distributions of the data.
    centroid_2dg(): Calculates the centroid by fitting a 2D Gaussian to the 2D distribution of the data.
    
    Masks can be input into each of these functions to mask bad pixels. Error arrays can be input into the two Gaussian fitting methods to weight the fits.
    """

    # Create a circular mask of fixed radius (of 3 fibres), based on the centre pixel with maximum flux

    # iter_one = True
    # # if iter_one:
    # # Get the fibre with the maximum flux
    # ind = np.argmax(flux, axis=None)  # maximum flux point
    # xcen, ycen = x[ind], y[ind]
    #
    # mask = ((x > (xcen - 250)) & (x < (xcen + 250)) & (y > (ycen - 250)) & (y < (ycen + 250)))
    # print(mask)
    #
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    #
    # ax1.scatter(x[mask], y[mask], c=flux[mask], cmap=plt.get_cmap('RdYlBu'))
    # plt.show()
    #
    # sys.exit()


    # The target grid to interpolate to
    xi = np.arange(np.min(x), np.max(x) + 1, 1)
    yi = np.arange(np.min(y), np.max(y) + 1, 1)
    xi, yi = np.meshgrid(xi, yi)

    background = np.median(flux) #

    # Interpolate
    # zi = griddata((x,y),data,(xi,yi),method='linear', fill_value=background, rescale=False)
    ##zi = griddata((x,y),data,(xi,yi),method='nearest', fill_value=background, rescale=False)
    zi = griddata((x, y), flux, (xi, yi), method='cubic', fill_value=background, rescale=False)   # background or NaN???????

    xycen1 = centroid_com(zi)
    xycen2 = centroid_quadratic(zi)
    xycen3 = centroid_1dg(zi)
    xycen4 = centroid_2dg(zi)
    xycens = [xycen1, xycen2, xycen3, xycen4]

    # com = comxyz(x, y, flux)  # **use good data within masking # centre-of-mass guess
    ind = np.argmax(flux, axis=None)  # maximum flux point
    com = [x[ind], y[ind]]
    # Peak height guess could be the closest fibre to Centre-of-mass (com) position.
    dist = (x - com[0]) ** 2 + (y - com[1]) ** 2  # distance between com and all fibres.

    # First guess at width of Gaussian - diameter of a core in degrees/microns.
    if microns:
        sigx = 105.0
        core_diam = 105.0
    else:
        sigx = 4.44e-4 * 3600
        core_diam = 4.44e-4 * 3600

    # Fit circular 2D Gaussians.
    p0 = [np.mean(flux[np.where(dist == np.min(dist))]), com[0], com[1], sigx, 0.0]
    gf = BundleFitter(p0, x, y, flux, model='gaussian_circ')
    fibre_integrator(gf, core_diam)
    gf.fit()

    # Make a linear grid to reconstruct the fitted Gaussian over.
    x_0 = np.min(x)
    y_0 = np.min(y)

    # dx should be 1/10th the fibre diameter (in whatever units)
    dx = sigx / 10.0

    xlin = x_0 + np.arange(100) * dx  # x axis
    ylin = y_0 + np.arange(100) * dx  # y axis

    # Reconstruct the model
    model = np.zeros((len(xlin), len(ylin)))
    # Reconstructing the Gaussian over the proper grid.
    for ii in range(len(xlin)):
        xval = xlin[ii]
        for jj in range(len(ylin)):
            yval = ylin[jj]
            model[ii, jj] = gf.fitfunc(gf.p, xval, yval)


    if (np.min(x) < gf.p[1] < np.max(x)) & (np.min(y) < gf.p[2] < np.max(y)):
        print('CentroidX = {}, CentroidY = {}, FWHM = {}, Offset = {}'.format(gf.p[1], gf.p[2], gf.p[3], gf.p[4]))
    else:
        # print('Centroid is outside the x-/y-range of the data')
        gf.p[1] = 0.0
        gf.p[2] = 0.0

    # calculate the sum of squares error
    square_of_diff = 0.0
    Nobs = 0.0
    for xi in range(len(x)):
        actual = flux[xi]
        predicted = gf.fitfunc(gf.p, x[xi], y[xi])

        square_of_diff = square_of_diff + (actual - predicted) ** 2.0
        Nobs += 1.0

    err = np.sqrt(square_of_diff[0] / Nobs)
    # print('RMS Err:', err)

    if make_plots:

        # Plot the flux data as a 2D image and the fit as overlaid contours.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))

        polyfitCenX, polyfitCenY = xycen2[0]+x_0, xycen2[1]+y_0
        # Calculates the radial difference between the centroid calculated from the Gaussian and polyfit centroid
        distance = np.sqrt( (gf.p[1] - polyfitCenX)**2.0 + (gf.p[2] - polyfitCenY)**2.0 )

        fig.suptitle(f"Probe: {Probe}, Error (RMS): {err} \n GaussFit (X, Y)={np.round(gf.p[1],2), np.round(gf.p[2],2)}, "
                     f"Centroid (X, Y)={np.round(polyfitCenX,2), np.round(polyfitCenY,2)}, "
                     f"$\Delta r$ = {np.round(distance,2)}", fontsize=15)
        ax1.scatter(x, y, c=flux, cmap=plt.get_cmap('RdYlBu'))
        ax1.plot(gf.p[1], gf.p[2], 'xr', ms=40)
        ax1.plot(com[0], com[1], 'xk', ms=40)

        ax1.contour(xlin, ylin, np.array(model).transpose(), colors='k')


        ax2.imshow(zi, cmap=plt.get_cmap('RdYlBu'), vmin=zi.min(), vmax=zi.max(),
                      extent=[x.min(), x.max(), y.min(), y.max()],
                      interpolation='nearest', origin='lower')
        ax2.plot(gf.p[1], gf.p[2], 'xr', ms=40)
        ax2.plot(com[0], com[1], 'xk', ms=40)

        marker = '+'
        ms, mew = 15, 1.
        colors = ('white', 'black', 'red', 'blue')
        for xycen, color in zip(xycens, colors):
            xcen, ycen = xycen
            # plt.plot(*xycen, color=color, marker=marker, ms=ms, mew=mew)
            ax2.plot(xcen+x_0, ycen+y_0, color=color, marker=marker, ms=ms, mew=mew)

        # if Probe == "B":
        #     sys.exit()

        fig.savefig(f"Probe_{Probe}", bbox_inches='tight')
        plt.close()

    return gf, [xycen2[0]+x_0, xycen2[1]+y_0], err, flux, xlin, ylin, model


def centroid_gauss_fit_flux_weighted_main(x_main, y_main, flux_main, Probe_main, microns=False, premask=False, make_plots=False):
    from scipy.interpolate import griddata
    from photutils.centroids import centroid_com, centroid_quadratic, centroid_1dg, centroid_2dg

    import warnings
    warnings.filterwarnings('error', '.*The fit may be unsuccessful.*', ) # Converts warnings to raise exceptions

    """
    photutils.centroids provides several functions to calculate the centroid of a single source:

    centroid_com(): Calculates the object “center of mass” from 2D image moments.
    centroid_quadratic(): Calculates the centroid by fitting a 2D quadratic polynomial to the data.
    centroid_1dg(): Calculates the centroid by fitting 1D Gaussians to the marginal x and y distributions of the data.
    centroid_2dg(): Calculates the centroid by fitting a 2D Gaussian to the 2D distribution of the data.

    Masks can be input into each of these functions to mask bad pixels. Error arrays can be input into the two Gaussian fitting methods to weight the fits.
    """

    def centroid_gauss_fit_flux_weighted_com(x, y, flux, gf1, microns=microns, iterate=None):
        mask = ((x > (gf1[0] - 250.)) & (x < (gf1[0] + 250.)) & (y > (gf1[1] - 250.)) & (y < (gf1[1] + 250.)))
        x_mask = x[mask]
        y_mask = y[mask]
        flux_mask = flux[mask]

        com = comxyz(x_mask, y_mask, flux_mask)  # **use good data within masking # centre-of-mass guess
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        # ax1.scatter(x[mask], y[mask], c=flux[mask], cmap=plt.get_cmap('RdYlBu'))
        # plt.show()

        # Peak height guess could be the closest fibre to Centre-of-mass (com) position.
        dist = (x_mask - com[0]) ** 2 + (y_mask - com[1]) ** 2  # distance between com and all fibres.

        # First guess at width of Gaussian - diameter of a core in degrees/microns.
        if microns:
            sigx = 105.0
            core_diam = 105.0
        else:
            sigx = 4.44e-4 * 3600
            core_diam = 4.44e-4 * 3600

        # Fit circular 2D Gaussians.
        p0 = [np.mean(flux_mask[np.where(dist == np.min(dist))]), com[0], com[1], sigx, 0.0]
        gf = BundleFitter(p0, x_mask, y_mask, flux_mask, model='gaussian_circ')
        fibre_integrator(gf, core_diam)
        gf.fit()

        if (np.min(x_mask) < gf.p[1] < np.max(x_mask)) & (np.min(y_mask) < gf.p[2] < np.max(y_mask)):
            if not iterate:
                print('CentroidX = {}, CentroidY = {}, FWHM = {}, Offset = {}'.format(gf.p[1], gf.p[2], gf.p[3], gf.p[4]))
        else:
            # print('Centroid is outside the x-/y-range of the data')
            gf.p[1] = 0.0
            gf.p[2] = 0.0

        # Make a linear grid to reconstruct the fitted Gaussian over.
        x_0 = np.min(x_mask)
        y_0 = np.min(y_mask)

        # dx should be 1/10th the fibre diameter (in whatever units)
        dx = sigx / 10.0

        xlin = x_0 + np.arange(100) * dx  # x axis
        ylin = y_0 + np.arange(100) * dx  # y axis

        # Reconstruct the model
        model = np.zeros((len(xlin), len(ylin)))
        # Reconstructing the Gaussian over the proper grid.
        for ii in range(len(xlin)):
            xval = xlin[ii]
            for jj in range(len(ylin)):
                yval = ylin[jj]
                model[ii, jj] = gf.fitfunc(gf.p, xval, yval)

        # calculate the sum of squares error
        square_of_diff = 0.0
        Nobs = 0.0
        for xi in range(len(x_mask)):
            actual = flux_mask[xi]
            predicted = gf.fitfunc(gf.p, x_mask[xi], y_mask[xi])

            square_of_diff = square_of_diff + (actual - predicted) ** 2.0
            Nobs += 1.0

        err = np.sqrt(square_of_diff[0] / Nobs)
        # print('RMS Err:', err)

        return gf, com, err, flux, xlin, ylin, model


    def centroid_gauss_fit_flux_weighted_photoUtils(x, y, flux, gf2, microns=microns, iterate=None):
        # print(gf2)
        mask = ((x > (gf2[0] - 250.)) & (x < (gf2[0] + 250.)) & (y > (gf2[1] - 250.)) & (y < (gf2[1] + 250.)))
        x_mask = x
        y_mask = y
        flux_mask = flux

        # print(mask)
        # print(x_mask)

        # The target grid to interpolate to
        xi = np.arange(np.min(x_mask), np.max(x_mask) + 1, 1)
        yi = np.arange(np.min(y_mask), np.max(y_mask) + 1, 1)
        xi, yi = np.meshgrid(xi, yi)

        background = np.median(flux_mask)  #

        # Interpolate
        # zi = griddata((x,y),data,(xi,yi),method='linear', fill_value=background, rescale=False)
        ##zi = griddata((x,y),data,(xi,yi),method='nearest', fill_value=background, rescale=False)
        zmodel = griddata((x_mask, y_mask), flux_mask, (xi, yi), method='cubic', fill_value=background,
                      rescale=False)  # background or NaN???????

        x_0 = np.min(x_mask)
        y_0 = np.min(y_mask)

        xycen1 = centroid_com(zmodel)
        try:
            # xycen2 = centroid_quadratic(zmodel, xpeak=gf2[0], ypeak=gf2[1])
            xycen2 = centroid_quadratic(zmodel)
            # xycen3 = centroid_1dg(zmodel)
            # xycen4 = centroid_2dg(zmodel)
        except AstropyUserWarning:
            xycen2 = [np.NaN, np.NaN]

        # xycens = [xycen1, xycen2, xycen3, xycen4]
        xycens = [xycen1, xycen2]

        return xycens, zmodel, [x_0, y_0], x_mask, y_mask

    # Create a circular mask of fixed radius (of 3 fibres), based on the centre pixel with maximum flux
    niters, first_iter, iterate = 3, True, True
    for iteration in range(niters):
        if iteration == niters - 1:
            iterate = False

        if first_iter:
            # Get the fibre with the maximum flux
            ind = np.argmax(flux_main, axis=None)  # maximum flux point
            cenfX, cenfY = x_main[ind], y_main[ind]
            gfunc1, _, _, _, _, _, _ = centroid_gauss_fit_flux_weighted_com(x_main, y_main, flux_main, [cenfX, cenfY], microns=microns, iterate=iterate)
            gfunc2, _, _, _, _ = centroid_gauss_fit_flux_weighted_photoUtils(x_main, y_main, flux_main, [cenfX, cenfY], microns=microns, iterate=iterate)
            first_iter = False
        else:
            gfun1 = gfunc1.p[1], gfunc1.p[2]
            gfun2 = gfunc2[1] # centroid_quadratic()
            gfunc1, com1, err1, flux1, xlin1, ylin1, model1 = centroid_gauss_fit_flux_weighted_com(x_main, y_main, flux_main, gfun1, microns=microns, iterate=iterate)
            gfunc2, zi, zero_points, xpoints, ypoints = centroid_gauss_fit_flux_weighted_photoUtils(x_main, y_main, flux_main, gfun2, microns=microns, iterate=iterate)


    x0, y0 = zero_points
    gfun2 = gfunc2[1]
    if make_plots:
        # Plot the flux data as a 2D image and the fit as overlaid contours.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

        polyfitCenX, polyfitCenY = gfun2[0] + x0, gfun2[1] + y0
        # Calculates the radial difference between the centroid calculated from the Gaussian and polyfit centroid
        distance = np.sqrt((gfunc1.p[1] - polyfitCenX) ** 2.0 + (gfunc1.p[2] - polyfitCenY) ** 2.0)

        fig.suptitle(
            f"Probe: {Probe_main}, Error (RMS): {err1} \n GaussFit (X, Y)={np.round(gfunc1.p[1], 2), np.round(gfunc1.p[2], 2)}, "
            f"Centroid (X, Y)={np.round(polyfitCenX, 2), np.round(polyfitCenY, 2)}, "
            f"$\Delta r$ = {np.round(distance, 2)}", fontsize=15)
        ax1.scatter(x_main, y_main, c=flux_main, cmap=plt.get_cmap('RdYlBu'))
        ax1.plot(gfunc1.p[1], gfunc1.p[2], 'xr', ms=40)
        ax1.plot(com1[0], com1[1], 'xk', ms=40)

        ax1.contour(xlin1, ylin1, np.array(model1).transpose(), colors='k')

        ax2.imshow(zi, cmap=plt.get_cmap('RdYlBu'), vmin=zi.min(), vmax=zi.max(),
                   extent=[xpoints.min(), xpoints.max(), ypoints.min(), ypoints.max()],
                   interpolation='nearest', origin='lower')
        ax2.plot(gfunc1.p[1], gfunc1.p[2], 'xr', ms=40)
        ax2.plot(com1[0], com1[1], 'xk', ms=40)

        marker = '+'
        ms, mew = 15, 1.
        colors = ('white', 'black', 'red', 'blue') # photoutils (com, quadratc, 1dg, 2dg)
        for xycen, color in zip(gfunc2, colors):
            xcen, ycen = xycen
            ax2.plot(xcen + x0, ycen + y0, color=color, marker=marker, ms=ms, mew=mew)

        # plt.show()
        # if Probe_main == "A":
        #     sys.exit()
        fig.savefig(f"Probe_{Probe_main}", bbox_inches='tight')
        plt.close()

    # The three centroiding estimates returned are: Gaussian fit based on COM estimate, PhotoUtils quadratic COM, and COM
    return gfunc1, [gfun2[0] + x0, gfun2[1] + y0], com1, err1, flux1, xlin1, ylin1, model1


def centroid_fit(x, y, flux, Probe, microns=False, premask=False, do_moffat=True):
    com = comxyz(x, y, flux)  # **use good data within masking

    # Peak height guess could be closest fibre to com position.
    dist = (x - com[0]) ** 2 + (y - com[1]) ** 2  # distance between com and all fibres.

    # First guess at width of Gaussian - diameter of a core in degrees/microns.
    if microns:
        sigx = 105.0
        core_diam = 105.0
    else:
        sigx = 4.44e-4 * 3600
        core_diam = 4.44e-4 * 3600

    # Fit circular 2D Gaussians.
    p0 = [np.mean(flux[np.where(dist == np.min(dist))]), com[0], com[1], sigx, 0.0]
    gf = BundleFitter(p0, x, y, flux, model='gaussian_circ')
    fibre_integrator(gf, core_diam)
    gf.fit()

    # Refit using initial values from Gaussian fit
    if do_moffat:
        p0 = [gf.p[0], gf.p[1], gf.p[2], gf.p[3] * np.sqrt(2 * np.log(2)), 1.0, gf.p[4]]
        gf = BundleFitter(p0, x, y, flux, model='moffat')
        fibre_integrator(gf, core_diam)
        gf.fit()
    print('moffat gf.p[1], gf.p[2]=', gf.p[1], gf.p[2])

    # Make a linear grid to reconstruct the fitted Gaussian over.
    x_0 = np.min(x)
    y_0 = np.min(y)

    # dx should be 1/10th the fibre diameter (in whatever units)
    dx = sigx / 10.0

    xlin = x_0 + np.arange(100) * dx  # x axis
    ylin = y_0 + np.arange(100) * dx  # y axis

    # Reconstruct the model
    model = np.zeros((len(xlin), len(ylin)))
    # Reconstructing the Gaussian over the proper grid.
    for ii in range(len(xlin)):
        xval = xlin[ii]
        for jj in range(len(ylin)):
            yval = ylin[jj]
            model[ii, jj] = gf.fitfunc(gf.p, xval, yval)

    return gf, flux, xlin, ylin, model


def hector_circle(x, xc, yc, radius):
    return yc + np.sqrt(radius ** 2 - (x - xc) ** 2)


def rotation_fit(file_list, plot_fit=False):
    """
    Fit for bundle rotation, returning rotation centroid and radius for all bundles. Take
    a list of >3 fitted centroid positions for a set of bundles, then determine a rotation
    centre by fitting a simple circle to the input data. Loop over all bundles in input file
    then print results to screen.

    Required inputs: file_list - list of strings with paths to centroid input files as
                        output by hector_utils.main. NB all files should have data for
                        the same set of probes
    """

    # Check if required input has been provided
    if len(file_list) < 3:
        print('Please provide a minimum of three input files, otherwise rotation cannot be constrained')
        return

    # Open the first input file to get list of probes
    tab = Table.read(file_list[0], format='ascii.commented_header')
    Probe_list = tab['Probe'].data

    def calc_R(x, y, xc, yc):
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f(c, x, y):
        Ri = calc_R(x, y, *c)
        return Ri - Ri.mean()

    # Loop over probes in Probe_list
    for Probe in Probe_list:

        # Read in x,y coordinates of centroid for 1 probe from all input files
        xdat, ydat = [], []
        for file in file_list:
            tab = Table.read(file, format='ascii.commented_header')
            index = np.where(tab['Probe'] == Probe)
            xdat.append(tab['X_mic'][index].data[0])
            ydat.append(tab['Y_mic'][index].data[0])

        # Fit a circle

        # p0 = [np.mean(xdat),np.mean(ydat),500.]
        # popt, pcov = curve_fit(hector_circle,xdat,ydat)#,p0=p0)

        p0 = np.mean(xdat), np.mean(ydat)
        center_2, ier = leastsq(f, p0, args=(xdat, ydat))
        xc_2, yc_2 = center_2
        Ri_2 = calc_R(xdat, ydat, *center_2)
        R_2 = Ri_2.mean()
        popt = [xc_2, yc_2, R_2]

        print(p0)
        print(popt)

        print('Probe: {}, Xrot: {}, Yrot: {}, Radrot: {}'.format(Probe, popt[0], popt[1], popt[2]))
        if plot_fit:
            fig = plt.figure()
            plt.clf()
            ax = plt.subplot(111)
            ax.set_aspect('equal')
            theta = np.arange(1000) / 1000 * 2 * np.pi
            x = popt[0] + popt[2] * np.cos(theta)
            y = popt[1] + popt[2] * np.sin(theta)
            ax.plot(x, y, 'k-', lw=3)
            ax.plot(xdat, ydat, 'rx', ms=5)

    return


def Ps_and_Qs(cen_x, cen_y, rotation_angle, tail_len, centroid_x, centroid_y, robot_coor=False):
    adj_ang = np.pi # mark the point 180 deg., from the ferral to extend the ferral axis across the probe
    rotation_angle = np.array(rotation_angle).squeeze()
    centroid_x = np.array(centroid_x).squeeze()
    centroid_y = np.array(centroid_y).squeeze()

    # Treat (cen_x, cen_y) of the probe as the origin to calculate P and Q
    cen_x, cen_y = 0.0, 0.0

    points_axferral = [(cen_x + tail_len * np.sin(rotation_angle + adj_ang),
                        cen_y + tail_len * np.cos(rotation_angle + adj_ang)),
                       (cen_x, cen_y),
                       (cen_x + tail_len * np.sin(rotation_angle),
                        cen_y + tail_len * np.cos(rotation_angle))] # This last two points, points to ferral direction

    grad = ((tail_len * np.cos(rotation_angle)) / (tail_len * np.sin(rotation_angle)))
    slope, intercept = np.polyfit(*zip(*points_axferral), 1)

    eqn_axferral = grad * centroid_x + intercept  # equation of the ferral axis, with centroid_x substituted for variable x

    # Deal with the cases, where equation of the line is vertical i.e. infinity gradient
    if np.isinf(grad) or np.abs(grad) > 1.0E9: grad = np.inf
    if np.isinf(slope) or np.abs(slope) > 1.0E9: slope = np.inf

    assert np.round(grad, 2) == np.round(slope, 2), 'check: gradients check failed!'

    Q_dist = calculate_Qs(points_axferral, centroid_x, centroid_y)

    P_dist = calculate_Ps(points_axferral, rotation_angle, centroid_x, centroid_y, grad, intercept, Q_dist, tail_len)


    # --------------------- Now determine the directions for Q- and P-directions ---------------------------------------
    # The Q-direction - For the axis orthogonal to the ferral axis (i.e. Q direction), 90 degree anti-clockwise from the
    # centre-to-ferral direction is assumed as the -ve direction
    # The P-direction - For the ferral axis, centre-to-ferral direction is taken to be -ve
    Q_sign, P_sign = is_angle_between(points_axferral, centroid_x, centroid_y)

    if Q_sign: Q_dist = Q_dist * (-1.0)
    if P_sign: P_dist = P_dist * (-1.0)

    # in robot-coordinate system y-axis is flipped. Pdist sign is fine, but Qdist sign will be switched.
    if robot_coor: Q_dist = Q_dist * (-1.0)


    return points_axferral, Q_dist, Q_sign, P_dist, P_sign


def perpendicular_distance_point_to_line(points_axisLine, point_x, point_y):
    """
    # ---------------------------- Perpendicular distance from a point to a line ---------------------------------------
    # Calculates the shortest distance (i.e. perpendicular distance) from a point (x0, y0),
    # to a line defined by two points; P1(x1, y1) and P2(x2, y2)
    # (ref: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line)
    #
    # If the line passes through two points P1 = (x1, y1) and P2 = (x2, y2) then the distance of (x0, y0) from the line is:
    # distance(P1, P2, (x0, y0)) = |(x2 - x1)(y1 - y0) - (x1 - x0)(y2 - y1)|    --- comes from the Cross Product
    #                              -----------------------------------------
    #                                   sqrt((x2 - x1)^2 + (y2 - y1)^2)
    # The above expression comes about from the formula for calculating the area of a parallelagram
    # (see evernote: HECTOR/calculating P and Q values)
    """
    P1 = np.array(points_axisLine[0])
    P2 = np.array(points_axisLine[2])
    P3 = np.array([point_x, point_y])

    # Perpendicular distance (or Q values) from P3 (or centroid position) to the line between P1 and P2 (or ferral axis).
    # The magnitude of this cross product gives the area of the parallelogram with sides given by the vector P1/P3 and P1/P2.
    # Divide Area/side-length to get height (or in our case, the distance)
    dist = np.linalg.norm(np.cross(P2 - P1, P1 - P3)) / np.linalg.norm(P2 - P1)

    # Perform a quick mathematical check
    x1, y1 = P1
    x2, y2 = P2
    x0, y0 = P3
    check_dist = np.abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))

    assert np.round(dist) == np.round(check_dist), \
        'Perpendicular distance consistency check FAILED!: shortest distances based on the two methods must match!'

    return dist


def calculate_Qs(points_axferral, centroid_x, centroid_y):
    """
    # ---------------- Q-DISTANCE  or perpendicular distance to the ferral axis ----------------------------------------
    """

    Qdist = perpendicular_distance_point_to_line(points_axferral, centroid_x, centroid_y)

    return Qdist


def calculate_Ps(points_axferral, rotation_angle, centroid_x, centroid_y, grad, intercept, Qdist, tail_len):
    """
    # ------------- P - DISTANCE or Parallel distance along the ferral axis --------------------------------------------
    # Here I used three methods to calculate the P-distance. In essence, if there is a bug in P- or Q-distance
    # calculations, the comparison between the three methods should show
    #
    # Method - I
    # Calculate the coordinates of the point (on the ferral axis), which makes a 90 degree angle with line connecting
    # that point and the centroid - this point is at a distance of 'Qdist' from the centroid, and at an angle of theta
    # dx = distance * cos(theta)  -- theta measured counter-clockwise from due east
    # dy = distance * sin(theta)
    #
    # If theta is measured clockwise from due north (for example, compass bearings), the calculation for dx and dy is
    # slightly different:
    # dx = R*sin(theta)  ; theta measured clockwise from due north
    # dy = R*cos(theta)  ; dx, dy same units as R
    #
    # Method - II
    # Calculate the coordinates of the point on the ferral axis by using the intersection between the line
    # describing the ferral axis, and the line perpendicular to the ferral axis going through the
    # centroid point
    #
    # Method - III
    # Q is the distance perpendicular to P- (or ferral axis)
    # P is the distance perpendicular to Q-axis
    # So call the "perpendicular_distance_point_to_line" with the inputs: centroid position as the point,
    # and the points describing the Q-axis
    """

    cen_x, cen_y = points_axferral[1][0], points_axferral[1][1]
    eqn_axferral = grad * centroid_x + intercept  # equation of the ferral axis, with centroid_x substituted for variable x

    ## Method-I - using the angle that the centroid forms with the x-axis and Qdist
    rotation_angle2 = rotation_angle
    # print(np.rad2deg(rotation_angle))
    if np.abs(rotation_angle) > np.pi:
        if rotation_angle > 0: rotation_angle2 = -(2*np.pi) + rotation_angle
        else: rotation_angle2 = (2*np.pi) + rotation_angle

    # Deal with the infinite gradients --
    if not np.isinf(grad):
        if rotation_angle2 > 0 and centroid_y > eqn_axferral: # rotation_angle has positive sign
            # print('here0')
            phi = -1.0 * rotation_angle2 # -np.pi + rotation_angle2
        elif rotation_angle2 > 0 and centroid_y < eqn_axferral:
            # print('here1')
            phi = np.pi - rotation_angle2  # rotation_angle2
        elif rotation_angle2 >= 0:  # centroid_y == eqn_axferral
            phi = np.pi
            print('on the ferral axis. This is probably a rare occurence (rotation angle=180 or 0 deg.). Needs to implement')
        elif rotation_angle2 < 0 and centroid_y > eqn_axferral: # rotation_angle has negative sign
            # print('here3')
            phi = np.pi + (-1.*rotation_angle2) # rotation_angle2
        elif rotation_angle2 < 0 and centroid_y < eqn_axferral:
            # print('here4')
            phi = -1. * rotation_angle2 # np.pi + rotation_angle2
        elif rotation_angle2 <= 0:
            print('on the ferral axis. This is probably a rare occurence (rotation angle=-180 deg.). Needs to implement')

    else: # if the gradient of the ferral axis is infinite, that means the rotation_angle is either 0 or 180.
        if (rotation_angle2 == 0.0 or np.abs(rotation_angle2) == np.pi) and centroid_x > 0.0:
            phi = -1. * np.pi/2.   # np.pi
        elif (rotation_angle2 == 0.0 or np.abs(rotation_angle2) == np.pi) and centroid_x < 0.0:
            phi = np.pi/2. # 0.0
        else:
            print('unlikely to happen, but check @L561')

    # phi = rotation_angle - np.pi # np.pi + rotation_angle
    # print(np.rad2deg(phi))
    # print(centroid_y, eqn_axferral)
    X2 = centroid_x + Qdist * np.cos(phi)
    Y2 = centroid_y + Qdist * np.sin(phi)


    ## Method-II -- Calculate the coordinates of the point on the ferral axis by considering the intersection between
    # the line describing the ferral axis, and the line perpendicular to the ferral axis going through the
    # centroid point
    coeffs = np.array([[-1. * grad, 1], [1. * 1./grad, 1]])
    b = np.array([intercept, 1. * 1./grad * centroid_x + centroid_y])
    C = np.linalg.solve(coeffs,b)
    # print(X2, Y2)
    # print(C[0], C[1])

    assert np.round(X2) == np.round(C[0]) and np.round(Y2) == np.round(C[1]), \
        'P distance consistency check FAILED!: two coordinate calculations methods (I and II) must match!'


    # Method-III -- using parallegrams by calling "perpendicular_distance_point_to_line"
    del points_axferral
    adj_ang = np.pi
    points_axferral = [(cen_x + tail_len * np.sin(rotation_angle + adj_ang + np.pi/2.),
                        cen_y + tail_len * np.cos(rotation_angle + adj_ang + np.pi/2.)),
                      (cen_x, cen_y),
                      (cen_x + tail_len * np.sin(rotation_angle + np.pi/2.),
                       cen_y + tail_len * np.cos(rotation_angle + np.pi/2.))]  # This last two points, points to ferral direction
    P_dist_test = perpendicular_distance_point_to_line(points_axferral, centroid_x, centroid_y)

    # Now calculate the P-distance - distance from the centre of the probe (taken to be the origin) to the interction
    # between the perpendicular line through the centroid and the ferral axis.
    Pdist = np.sqrt((X2 - cen_x)**2.0 + (Y2 - cen_y)**2.0)

    assert np.round(np.round(P_dist_test, 3)) == np.round(np.round(Pdist, 3)), \
        'P distance consistency check FAILED!: two coordinate calculations methods (I and III) must match'

    return Pdist


def is_angle_between(points_axferral, centroid_x, centroid_y):
    """
    ## Determines the signs for P and Q distances
    # Q-direction
    # For the axis orthogonal to the ferral axis (i.e. Q direction), 90/180 degree anti-clockwise from the
    # centre-to-ferral direction is assumed as the -ve direction
    #
    # P-direction
    # For the ferral axis, centre-to-ferral direction is taken to be -ve
    #
    """

    # ------------------------------------------- Detremine the Psign --------------------------------------------------
    cx, cy = points_axferral[1][0], points_axferral[1][1]
    x, y   = centroid_x, centroid_y
    xf, yf = points_axferral[2][0], points_axferral[2][1]

    # Centroid angle is the angle from East (to the right) to the centroid location -- +ve anti-clockwise
    centroid_angle = Math.atan2(y - cy, x - cx)  # np.rad2deg(Math.atan2(y - cy, x - cx))

    # Ferral angle is the angle from East to the ferral point -- +ve anti-clockwise
    ferral_angle = Math.atan2(yf - cy, xf - cx)
    # print('ferral_angle=', np.rad2deg(ferral_angle), 'centroid_angle=', np.rad2deg(centroid_angle))
    angle_between = ferral_angle - centroid_angle

    Psign = False # Positive Pdist
    if min(np.abs(angle_between), 2*np.pi - np.abs(angle_between)) <= np.pi/2.: Psign = True # negative Pdist

    # ------------------------------------------ Determine the Qsign ---------------------------------------------------
    # For the axis orthogonal to the ferral axis (i.e. Q direction), 90/180 degree anti-clockwise from the
    # centre-to-ferral direction is assumed as the -ve direction
    Qsign = False  #  ==> +ve Q-direction
    if ferral_angle <= 0: # if ferral angle is negative, then adding +180 always makes below statement true
        if ferral_angle <= centroid_angle <= (ferral_angle + np.pi):
            Qsign = True # ==> -ve Q-direction
        else:
            Qsign = False
    # if ferral angle is positive, then +180 takes it to angles>180 deg, i.e. 225, 315 etc, and centroid angle could have negative angles (i.e. -135, which is 225 deg rotated anti-clockwise from East)
    else:
        if centroid_angle < 0: centroid_angle = 2*np.pi + centroid_angle
        if ferral_angle <= centroid_angle <= (ferral_angle + np.pi):
            Qsign = True # ==> -ve Q-direction
        else:
            Qsign = False


    return Qsign, Psign


def Ns_and_Es(cen_x, cen_y, rotation_angle, tail_len, centroid_x, centroid_y, robot_coor=False):
    """
    North is always down on the plate ==> +ve
    East is always to the right ==> +ve
    """
    adj_ang = np.pi # mark the point 180 deg., from either North or East
    rotation_angle = 0.0 # I am re-adjusting the angle to zero. So I am treating this case as if the ferral axis is on the North axis
    centroid_x = np.array(centroid_x).squeeze()
    centroid_y = np.array(centroid_y).squeeze()

    # Treat (cen_x, cen_y) of the probe as the origin to calculate E and W vectors
    cen_x, cen_y = 0.0, 0.0

    points_axferral = [(cen_x + tail_len * np.sin(rotation_angle + adj_ang), cen_y + tail_len * np.cos(rotation_angle + adj_ang)),
                  (cen_x, cen_y),
                  (cen_x + tail_len * np.sin(rotation_angle), cen_y + tail_len * np.cos(rotation_angle))] # This last two points, points to north direction

    E_dist = perpendicular_distance_point_to_line(points_axferral, centroid_x, centroid_y)

    del points_axferral

    points_axferral = [(cen_x + tail_len * np.sin(rotation_angle + adj_ang + np.pi/2.), cen_y + tail_len * np.cos(rotation_angle + adj_ang + np.pi/2.)),
        (cen_x, cen_y),
        (cen_x + tail_len * np.sin(rotation_angle + np.pi/2.), cen_y + tail_len * np.cos(rotation_angle + np.pi/2.))]

    N_dist = perpendicular_distance_point_to_line(points_axferral, centroid_x, centroid_y)

    # determine the signs for E_dist and N_dist
    cx, cy = points_axferral[1][0], points_axferral[1][1]
    x, y = centroid_x, centroid_y

    # Centroid angle is the angle from East (to the right on the plate) to the centroid location -- +ve anti-clockwise
    centroid_angle = Math.atan2(y - cy, x - cx)  # np.rad2deg(Math.atan2(y - cy, x - cx))
    # print('centroid_angle:', np.rad2deg(centroid_angle))

    E_sign = False # +ve sign
    if np.abs(centroid_angle) > np.pi/2.: E_sign = True  # Esign = -ve

    N_sign = False # +ve sign
    if centroid_angle > 0: N_sign = True # = -ve

    if N_sign: N_dist = N_dist * (-1.0)
    if E_sign: E_dist = E_dist * (-1.0)

    if robot_coor: E_dist = E_dist * (-1.0)  # reverse the sign again

    return N_dist, N_sign, E_dist, E_sign


def perpendicular_and_parallel_to_RadialAxis(cen_x, cen_y, rotation_angle, tail_len, centroid_x, centroid_y, robot_coor=False):
    """
    Radial axis --> from heaxbundle centre-to-plate centre direction is taken to be negative
    Axis perpendicular to radial axis --> 90 degrees anti-clockwise from the heaxbundle centre-to-plate centre direction
                    is taken to be negative

    IMPORTANT: Note that the "rotation_angle" in this function defines the angle from North to the direction of the
    plate centre, i.e. the angle the radial axis of a given hexabundle
    """
    adj_ang = np.pi  # mark the point 180 deg., from radial axis direction
    centroid_x = np.array(centroid_x).squeeze()
    centroid_y = np.array(centroid_y).squeeze()

    # Treat (cen_x, cen_y) of the probe as the origin to calculate the vectors parallel and perpendicular to the radial axis
    cen_x, cen_y = 0.0, 0.0

    points_axradial1 = [
        (cen_x + tail_len * np.sin(rotation_angle + adj_ang), cen_y + tail_len * np.cos(rotation_angle + adj_ang)),
        (cen_x, cen_y),
        (cen_x + tail_len * np.sin(rotation_angle), cen_y + tail_len * np.cos(rotation_angle))
        ]  # This last two points, points to hexabundle centre-to-plate centre direction

    points_axradial = points_axradial1  # hold for the input to 'is_angle_between'

    # Perpendicular distance from the centroid position to the radial axis
    Rperpendi_dist = perpendicular_distance_point_to_line(points_axradial1, centroid_x, centroid_y) # equivalent of 'Q' for Radial axis

    del points_axradial1

    points_axradial2 = [
        (cen_x + tail_len * np.sin(rotation_angle + adj_ang + np.pi / 2.),
         cen_y + tail_len * np.cos(rotation_angle + adj_ang + np.pi / 2.)),
        (cen_x, cen_y),
        (cen_x + tail_len * np.sin(rotation_angle + np.pi/2.), cen_y + tail_len * np.cos(rotation_angle + np.pi/2.))
        ]  #

    # Perpendicular distance from the centroid position to the axis that is orthogonal to the radial axis
    # Or parallel distance along the radial axis
    Rparallel_dist = perpendicular_distance_point_to_line(points_axradial2, centroid_x, centroid_y) # equivalent of 'P' for Radial axis

    del points_axradial2

    # ---------- Now determine the directions for perpendi_toRax- and parallelDist_toRax-directions -----------
    # The Rperpendi_dist-direction - For the axis orthogonal to the radial axis, 90 degree anti-clockwise from the
    # hexabndle centre-to-plate centre direction is assumed as the -ve direction
    # The Rparallel_dist-direction - For the radial axis, hexabndle centre-to-plate centre direction is taken to be -ve

    # We can use "is_angle_between" function for this task, which written for the determination of P-/Q- signs. If using
    # 'is_angle_between', need to re-define "point_axradial" with the data point between the hexabundle centre and
    # plate centre being the last point in the array

    Rperpendi_sign, Rparallel_sign = is_angle_between(points_axradial, centroid_x, centroid_y)

    if Rperpendi_sign: Rperpendi_dist = Rperpendi_dist * (-1.0)
    if Rparallel_sign: Rparallel_dist = Rparallel_dist * (-1.0)

    if robot_coor: Rperpendi_dist = Rperpendi_dist * (-1.0) # reverse sign for the robot coordinates

    return points_axradial, Rperpendi_dist, Rperpendi_sign, Rparallel_dist, Rparallel_sign

