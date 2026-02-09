import os
import sys
import shutil
import numpy as np
import scipy as sp

import pylab as py
from scipy.stats import gaussian_kde

import astropy.io.fits as pf
from astropy.io import fits
from matplotlib import pyplot as plt

import pandas as pd
import glob


# Print colours in python terminal --> https://www.geeksforgeeks.org/print-colors-python-terminal/
def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))
def prYellow(skk): print("\033[93m {}\033[00m" .format(skk))
def prLightPurple(skk): print("\033[94m {}\033[00m" .format(skk))
def prPurple(skk): print("\033[95m {}\033[00m" .format(skk))
def prCyan(skk): print("\033[96m {}\033[00m" .format(skk))
def prLightGray(skk): print("\033[97m {}\033[00m" .format(skk))
def prBlack(skk): print("\033[98m {}\033[00m" .format(skk))


def make_figures(centroid_statFinal, supltitle):

    fig_stat, (axes) = plt.subplots(3, 3, figsize=(14, 10))
    axes[0, 2].set_axis_off()
    fig_stat.suptitle(f"Centroiding Stats: {supltitle}", fontsize=15)
    fig_stat.subplots_adjust(left=0.05,
                             bottom=0.06,
                             right=0.99,
                             top=0.93,
                             wspace=0.2,
                             hspace=0.2)

    def plot_hist(datFrame, name, ax0, label, colr, with_error=None, stacked=False):
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
        if stacked:
            b, bins, patches = ax0.hist(
                [dataFrame[datFrame['TelecentricAng'] == 'b'],
                 dataFrame[datFrame['TelecentricAng'] == 'g'],
                 dataFrame[datFrame['TelecentricAng'] == 'y'],
                 dataFrame[datFrame['TelecentricAng'] == 'm']],
                bins=np.int(np.array(nbins).squeeze()),
                stacked=True,
                label=[],
                edgecolor='white',
                color=['b', 'g', 'y', 'm'])
            ax0.set_xlabel(label)
        else:
            if name == 'RadialDist_Plate':
                def calculate_point_density(x, y):
                    # Calculate point density
                    xy = np.vstack([x, y])
                    density = gaussian_kde(xy)(xy)

                    # Sort points by density so dense points plot on top
                    idx = density.argsort()
                    x, y, density = x[idx], y[idx], density[idx]

                    return x, y, density


                ax0.plot(dataFrame[datFrame['TelecentricAng'] == 'b'],
                         datFrame['RPerpenDist'][datFrame['TelecentricAng'] == 'b'], 'o', ms=2, markeredgecolor='b', markerfacecolor= 'none',
                         alpha=0.1)
                ax0.plot(dataFrame[datFrame['TelecentricAng'] == 'g'],
                         datFrame['RPerpenDist'][datFrame['TelecentricAng'] == 'g'], 'o', ms=2, markeredgecolor='g', markerfacecolor= 'none',
                         alpha=0.1)
                ax0.plot(dataFrame[datFrame['TelecentricAng'] == 'y'],
                         datFrame['RPerpenDist'][datFrame['TelecentricAng'] == 'y'], 'o', ms=2, markeredgecolor='y', markerfacecolor= 'none',
                         alpha=0.1)
                ax0.plot(dataFrame[datFrame['TelecentricAng'] == 'm'],
                         datFrame['RPerpenDist'][datFrame['TelecentricAng'] == 'm'], 'o', ms=2, markeredgecolor='m', markerfacecolor= 'none',
                         alpha=0.1)


                xb, yb, densityb = calculate_point_density(dataFrame[datFrame['TelecentricAng'] == 'b'],
                                            datFrame['RPerpenDist'][datFrame['TelecentricAng'] == 'b'].to_numpy())
                ax0.scatter(xb, yb, c=densityb, s=10, cmap='Blues')
                del xb, yb, densityb

                xg, yg, densityg = calculate_point_density(dataFrame[datFrame['TelecentricAng'] == 'g'],
                                            datFrame['RPerpenDist'][datFrame['TelecentricAng'] == 'g'].to_numpy())
                ax0.scatter(xg, yg, c=densityg, s=10, cmap='Greens')
                del xg, yg, densityg

                xy, yy, densityy = calculate_point_density(dataFrame[datFrame['TelecentricAng'] == 'y'],
                                            datFrame['RPerpenDist'][datFrame['TelecentricAng'] == 'y'].to_numpy())
                ax0.scatter(xy, yy, c=densityy, s=10, cmap='Oranges')
                del xy, yy, densityy

                xm, ym, densitym = calculate_point_density(dataFrame[datFrame['TelecentricAng'] == 'm'],
                                            datFrame['RPerpenDist'][datFrame['TelecentricAng'] == 'm'].to_numpy())
                ax0.scatter(xm, ym, c=densitym, s=10, cmap='Purples')
                del xm, ym, densitym


                coefficients = np.polyfit(dataFrame, datFrame['RPerpenDist'], 1)
                coefficients_noMagenta = np.polyfit(dataFrame[datFrame['TelecentricAng'] != 'm'], \
                                                    datFrame['RPerpenDist'][datFrame['TelecentricAng'] != 'm'], 2)
                print("Linear Fit Coefficients:", coefficients)
                print("Poly Fit Coefficients:", coefficients_noMagenta)

                # Linear fit used currently in the robot code
                coeffs = [-5.10579247e-03, -9.08942022e-01, 1.94332118e+02]
                ax0.plot(np.arange(0,250,10), np.poly1d(coeffs)(np.arange(0,250,10)), ':', label=f"Linear Fit Used", color='blue')

                # Create polynomial function
                p = np.poly1d(coefficients)
                p_noMagenta  = np.poly1d(coefficients_noMagenta)

                ax0.plot(dataFrame, p(dataFrame), label='Linear Fit', color='red')
                ax0.plot(np.arange(0,250,10), p_noMagenta(np.arange(0,250,10)), label='Linear Fit (without magenta)', color='blue')


                ax0.set_xlabel(label)
                ax0.set_ylabel(r'$\bot$ to radial axis [$\mu $m]')
                ax0.legend(loc='best', frameon=False)
            else:
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
    plot_hist(centroid_statFinal, 'RadialDist', axes[0, 0], "Radial offset " r'[$\mu $m]', 'black', with_error=True, stacked=True)
    plot_hist(centroid_statFinal, 'RadialDist_Plate', axes[0, 1],
              'Radial distance from the plate centre' r'[$10^3 \mu $m]', 'Chartreuse', with_error=True, stacked=False)

    plot_hist(centroid_statFinal, 'PDist', axes[1, 0], "P-dir Offset " r'[$\mu $m]', 'black', with_error=True, stacked=True)
    plot_hist(centroid_statFinal, 'QDist', axes[2, 0], "Q-dir Offset " r'[$\mu $m]', 'black', with_error=True, stacked=True)

    plot_hist(centroid_statFinal, 'NDist', axes[1, 1], "Compass (N-S) Offset " r'[$\mu $m]', 'black', with_error=True, stacked=True)
    plot_hist(centroid_statFinal, 'EDist', axes[2, 1], "Compass (E-W) Offset " r'[$\mu $m]', 'black', with_error=True, stacked=True)

    plot_hist(centroid_statFinal, 'RParallelDist', axes[1, 2], r'$\parallel$ to radial axis [$\mu $m]', 'black', with_error=True, stacked=True)
    plot_hist(centroid_statFinal, 'RPerpenDist', axes[2, 2], r'$\bot$ to radial axis [$\mu $m]', 'black', with_error=True, stacked=True)



if __name__ == "__main__":
    import yaml
    from pathlib import Path


    ccds = np.arange(1, 5).astype(str)

    config_filename = 'hector_display_config.yaml'
    try:
        with open(config_filename, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {config_filename} does not exist!")


    # Read all the csv files in the quick-look folder
    csv_files = list(Path(config['data_dir']).glob('Outputs/*/tramline_map/Hector_fit/*.csv'))

    try:
        disabled = pd.read_csv(Path(config['data_dir']) / f"disable.txt", header=None, names=['disabled_files'])
        disabled_files = disabled['disabled_files'].to_numpy()
    except FileNotFoundError:
        disabled = None

    dfs = []
    for index, csv in enumerate(csv_files):
        run    = csv.name.replace('.csv', '').split('_')
        ccd_no = [run[1] + i + f"{run[2].replace('Run', ''):>04}" for i in ccds] # CCD run IDs contributing to the quick-look plot
        prGreen(f"{run[1]+run[2]}....")

        if disabled is not None:
            intersect = set.intersection(*[set(x) for x in [disabled_files, ccd_no]]) # Find the intersection between the disabled list and the contributing run IDs

            if intersect:
                prYellow(f"{intersect} files are disabled"); continue # if intersect (i.e. 'true') == file in 'disabled' list... skip

        centroid_stats = pd.read_csv(csv, skiprows=1,
                                     names=["Index", "Probe", "MeanX", "MeanY", "RotationAngle", "CentroidX_rotated",
                                            "CentroidY_rotated", "CentroidXErr_rotated", "CentroidYErr_rotated",
                                            "CentroidX_COM_rotated", "CentroidY_COM_rotated", "CentroidRMS_Err",
                                            "RotationAngle_Centroid", "RadialDist", "RadialDistErr", "PDist", "QDist",
                                            "PDistErr", "QDistErr", "NDist", "EDist", "NDistErr", "EDistErr", "RPerpenDist",
                                            "RParallelDist", "RPerpenDistErr", "RParallelDistErr", "TelecentricAng",
                                            "RadialDist_Plate", "RadialDist_PlateErr"])

        dfs.append(centroid_stats)
    centroid_statsFinal = pd.concat(dfs, ignore_index=True)


    make_figures(centroid_statsFinal, f"{Path(config['data_dir']).name}")

    figfile = Path(config['data_dir']) / f"plateViewSummary_{Path(config['data_dir']).name}"
    plt.savefig(figfile, bbox_inches='tight', pad_inches=0.3)
    plt.show()



