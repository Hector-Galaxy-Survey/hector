import numpy as np
from astropy.table import Table
import astropy.io.fits as pf
from glob import glob
import os
import datetime
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

##############################
#
#Functions for Gaussian fitting of Hector Arcframes
#work together with mngr.reduce_arc()
#first version: 10th July 2024
#version 0.1: 11th July 2024 - bug fixed
#version 0.2: 12th July 2024 - bug fixed
#version 0.3: 13th July 2024 - at a plot for all-CCDs FWHM 
#Susie Tuntipong
#stun4076@uni.sydney.edu.au
#
##############################

def read_wavelengths(reduced_arc_path):
    '''Read wavelength range of a fits file
    reduced_arc_path is the directory of the arcframe
    return an array of wavelengths and the pixel width in Angstorm'''
    cube = pf.open(reduced_arc_path)
    hd0 = cube[0].header
    wavelength_per_pixel = hd0['CDELT1']
    wavelengths = hd0['CRVAL1']+(np.arange(0,hd0['NAXIS1'])+1-hd0['CRPIX1'])*wavelength_per_pixel
    return wavelengths

def select_fibres(reduced_arc_path):
    '''Randomly select ~100 fibres from a reduced arcframe
    reduced_arc_path is the directory of the arcframe'''

    #read the reduce arcframe
    x = pf.open(reduced_arc_path)
    fibres_ifu = (x['FIBRES_IFU'].data)

    #QC ifu - pick up only active fibres
    fibres_ifu_type = (fibres_ifu)['TYPE']
    fibres_order = np.arange(len(fibres_ifu))
    pos_active = np.where(np.logical_or(fibres_ifu_type=='P',fibres_ifu_type=='S'))[0]
    fibres_order_active = fibres_order[pos_active]
    fibres_ifu_probename_active = (fibres_ifu)['PROBENAME'][pos_active]
    
    #grouping by probename
    probename, probenum = np.unique(fibres_ifu_probename_active,return_counts=True)
    fibres_investigated_0 = []
    for j in range(len(probename)):
        #get the fibre order for each probename
        pos_probename = np.where(fibres_ifu_probename_active==probename[j])[0]
        #if only one fibre appear, pick it anyway
        if len(pos_probename)==1:
            fibres_investigated_0.append(fibres_order_active[pos_probename])
        else:
            #randomly select fibres ~10% of the total number of fibres for each probename
            num_fibres = int(round(0.1*len(pos_probename),0))
            fibres_selected = np.random.choice(fibres_order_active[pos_probename],size=num_fibres,replace=False)
            fibres_investigated_0.append(fibres_selected)
    #Combine and sort the fibres  
    fibres_investigated = np.sort(np.concatenate(fibres_investigated_0))
    
    return fibres_investigated
    
    
def Gaussian_func(x_measured,multiple,x_standard,sigma):
    '''Gaussian function'''
    return (multiple*((np.exp(-(((x_measured-x_standard)/sigma)**2)/2))/(sigma*np.sqrt(2*np.pi))))


def FWHM(sigma):
    '''Calculate FWHM from sigma in Gaussian function'''
    return 2*np.sqrt(2*np.log(2))*sigma


def fitting_Gaussian(wavelength_all,x_test,wavelength_test,x_subtract,variance_measured,wavelength_expected,sigma_initial_guess):
    '''Gaussian fitting'''
    try:
        popt, pcov = curve_fit(Gaussian_func, wavelength_test,x_subtract,check_finite=True,
                bounds=([0,min(wavelength_test),0],[1e+06,max(wavelength_test),5.0]),
                sigma=np.sqrt(variance_measured),p0=[1,wavelength_expected,sigma_initial_guess],method='trf')
    except:
        #good data but fail to fit
        fwhm = np.nan
    else:
        #if the fitting works, collect the data and shift cutoff
        central_line = popt[1]
        diff_wavelength = abs(central_line-wavelength_all)
        pos_central = np.where(diff_wavelength==np.nanmin(diff_wavelength))[0][0]
        x_test_ydata = x_test[pos_central - 4:pos_central + 4 + 1]
                                
        #for any lines at the edges, if nan within the peak range, record as nan
        if np.any(np.isnan(x_test_ydata))==True:
            fwhm = np.nan
        else:
            sigma = popt[2]
            fwhm = FWHM(sigma)
            
    return fwhm
    
    
def fitting(reduced_arc_path, outdir):
    ''' 
    
    Fitting Gaussian functions for all peaks
    
    reduced_arc_path is the directory of the reduced arcframes, must in in the pattern of 
    reduced_arc_path = ['/path/to/file/ddmmm1xxxxred.fits','/path/to/file/ddmmm2xxxxred.fits',
                        '/path/to/file/ddmmm3xxxxred.fits','/path/to/file/ddmmm4xxxxred.fits']
    
    outdir is the directory of outputs, must be in the pattern of
    outdir = '/data/hector/reduction/YYMMDD_YYMMDD/qc_plots/' 
    
    '''

    print('Calculating FWHM...')
    
    #Information of fitting
    
    #Expected wavelengths from CuAr-FeAr-He
    wavelength_standard_CCD = [[4103.9121, 4259.3619, 4481.8107, 4510.7332, 4579.3495,4589.8978, 4609.5673, 4726.8683, 4764.8646, 4806.0205, 4847.8095,4965.0795, 5105.541 , 5606.733 , 5700.24],[6384.7169, 6538.112 , 6604.8534, 6643.6976, 6684.2929,6752.8335, 6766.6117, 6871.2891, 6937.6642, 7030.2514, 7107.4778,7125.82  , 7158.8387, 7206.9804, 7281.349 , 7311.7159, 7316.005 ,7353.293 , 7372.1184],[3859.9114, 3961.727 , 4026.19  , 4103.9121, 4131.7235, 4158.5905,4164.1795, 4181.8836, 4237.2198, 4259.3619, 4266.2864, 4272.1689,4348.064 , 4379.6668, 4471.48  , 4481.8107, 4510.7332, 4545.0519,4579.3495, 4589.8978, 4609.5673, 4637.2328, 4713.15  , 4726.8683,4732.0532, 4735.9058, 4764.8646, 4806.0205, 4847.8095, 4879.8635,4933.2091, 4965.0795, 5105.541 , 5141.7827, 5162.2846, 5187.7462,5558.702 , 5606.733 , 5700.24],[5875.661 , 5912.0853, 6032.1274, 6043.2233, 6059.3725, 6384.7169, 6416.3071, 6538.112 , 6604.8534, 6643.6976, 6684.2929,6752.8335, 6766.6117, 6871.2891, 6879.5824, 6937.6642, 6951.4776,7030.2514, 7107.4778, 7125.82  , 7158.8387, 7206.9804, 7281.349 ,7311.7159, 7316.005 , 7353.293 , 7372.1184, 7392.9801, 7412.3368,7425.2942]]

    #Positions of the expected peaks, calculated from the median of all fibres
    wavelength_standard_peakpos_CCD = [[432,  580,  792,  819,  885,  895,  914, 1025, 1061, 1101,
       1140, 1252, 1386, 1863, 1952],[243,  500,  612,  677,  746,  860,  884, 1059, 1171, 1326,1456, 1486, 1542, 1622, 1747, 1798, 1806, 1868, 1900],[ 329,  515,  633,  775,  826,  875,  885,  918, 1019, 1059, 1072,1082, 1221, 1279, 1447, 1466, 1519, 1581, 1644, 1663, 1699, 1750,1889, 1914, 1922, 1930, 1983, 2058, 2135, 2193, 2291, 2349, 2606,2672, 2710, 2756, 3435, 3522, 3693],[ 357,  427,  658,  679,  710, 1336, 1396, 1630, 1758, 1833,1911, 2043, 2069, 2271, 2287, 2399, 2425, 2577, 2725, 2761, 2824,2917, 3060, 3118, 3126, 3198, 3234, 3274, 3311, 3336]]
    
    #initial guess for Gaussian fitting sigmas
    sigma_initial_guess_CCD = np.array([1.0,0.6,0.6,0.5])

    #write the output directory if not existing
    if len(glob(outdir)) == 0:
        os.mkdir(outdir)
    else:
        pass

    #write the text file if not existing
    if len(glob(outdir + 'qc_values.txt'))==0:
        with open(outdir + 'qc_values.txt', 'w') as f:
            f.write('Arcframe FWHM TELFOC TILTSPAT TILTSPEC PISTON')
            f.write('\n')
    else:
        pass
    
    for r in range(len(reduced_arc_path)):
        #state the framename
        framename = (reduced_arc_path[r].split('/')[-1]).split('red')[0]

        #prevent repeating calculation
        data = Table.read(outdir + 'qc_values.txt',format='ascii',delimiter=' ')
        data_framename = np.array(data['Arcframe'])
        data_fwhm = np.array(data['FWHM'])
        pos_framename = np.where(framename==data_framename)[0]
        if len(pos_framename)!=0:
            print(framename,' FWHM is ',data_fwhm[pos_framename[0]])
        else:
            #calculation is not done yet
            
            #Identify CCD
            n_ccd = int(framename[5]) - 1
    
            #expected wavelengths, peak positions and initial guess
            wavelength_standard = wavelength_standard_CCD[n_ccd]
            wavelength_standard_peakpos = wavelength_standard_peakpos_CCD[n_ccd]
            sigma_initial_guess = sigma_initial_guess_CCD[n_ccd]
       
            #read wavelengths
            wavelength_all = read_wavelengths(reduced_arc_path[r])

            #randomly select fibres
            fibres_investigated = select_fibres(reduced_arc_path[r])
    
            #intensty and variance
            x = pf.open(reduced_arc_path[r])
            x_test = x[0].data
            variance_test = x['VARIANCE'].data
    
            #read focuses
            x_header = np.array([x[0].header[h0] for h0 in ['TELFOC','TILTSPAT','TILTSPEC','PISTON']])
    
            fwhm_all = []
            for n_fibre in fibres_investigated: 
                for n_lines in range(len(wavelength_standard)):
                    #peak position
                    pos_peak_0 = wavelength_standard_peakpos[n_lines]

                    if (n_lines < 2) or (n_lines > len(wavelength_standard)-3):
                        pixel_surr_0 = 10
                        pos_peak_range_0 = np.arange(pos_peak_0-pixel_surr_0,pos_peak_0+pixel_surr_0+1,1)
                        x_measured_0 = x_test[n_fibre][pos_peak_range_0]
                        wavelength_test_0 = wavelength_all[pos_peak_range_0]
                        pos_peak_array = np.where(x_test[n_fibre]==np.nanmax(x_measured_0))[0]
                        if len(pos_peak_array)!=0:
                            pos_peak = pos_peak_array[0]
                        else:
                            pos_peak = pos_peak_0
                    else:
                        pos_peak = pos_peak_0
                
                    pos_peak_range = np.arange(pos_peak-4,pos_peak+5,1)
                    wavelength_expected = wavelength_standard[n_lines]
                    x_measured = x_test[n_fibre][pos_peak_range]
                    variance_measured = variance_test[n_fibre][pos_peak_range]
                    wavelength_test = wavelength_all[pos_peak_range]
            
                    if (np.any(np.isnan(x_measured))==True) or (np.all(x_measured==0)==True) or (np.any(np.isnan(variance_measured))==True) or (np.any(variance_measured < 0)==True) or np.logical_and(np.nanmin(x_measured) < 0,abs(np.nanmin(x_measured)/np.nanmax(x_measured)) > 0.1):
                        #data unavailable for fitting, record as nan
                        fwhm_all.append(np.nan)
                    else:
                        #then continue the job
                        #getting background
                        pos_background_lower = pos_peak - 20
                        pos_background_upper = pos_peak + 20
                        #using 25th percentile
                        background = np.nanpercentile([j for j in x_test[n_fibre][pos_background_lower:pos_background_upper+1]],25)
                        #background subtraction
                        x_subtract = x_measured - background
                        fwhm = fitting_Gaussian(wavelength_all,x_test[n_fibre],wavelength_test,x_subtract,variance_measured,wavelength_expected,sigma_initial_guess)
                        fwhm_all.append(fwhm)

            median_fwhm = np.nanmedian(fwhm_all)
            print(framename,' FWHM is ',median_fwhm)

            #Collect data onto a table
            with open(outdir + 'qc_values.txt', 'a') as f:
                f.write(framename+' '+str(median_fwhm))
                for xh in range(len(x_header)):
                    f.write(' '+str(x_header[xh]))
                f.write('\n')

        
def plot_fwhm(outdir):
    '''
    
    Plot FWHM as a function of time
    outdir is the directory of outputs, must be in the pattern of
    outdir = '/data/hector/reduction/YYMMDD_YYMMDD/qc_plots/' 
    
    '''
    
    print('Plotting...')
    data = Table.read(outdir + 'qc_values.txt',format='ascii',delimiter=' ')
    data_header = ['Arcframe','FWHM','TELFOC','TILTSPAT','TILTSPEC','PISTON']

    n_ccd = np.array([int(np.array(data[data_header[0]])[xi][5]) for xi in range(len(data))])
    
    if outdir[-1]!='/':
        outdir = outdir + '/'
    else:
        pass
    duration = outdir.split('/')[-3]
    
    colour = ['b','r','b','r']
    lower_lim = [np.array([2.0,1.0,1.0,1.0]),[35,35,35,35],[1500,600,-0.5,-0.5],[2700,2000,-0.5,-0.5],[100,400,2700,2200]]
    upper_lim = [np.array([4.0,2.0,2.0,2.0]),[45,45,45,45],[2500,2500,0.5,0.5],[3600,3100,0.5,0.5],[200,600,3000,2500]]
    median_all = np.array([2.6409,1.5598,1.4461,1.2554])
    
    #Plot for FWHM only
    nrow = 2
    ncol = 2
    fig0 = plt.figure(figsize=((ncol + 7.0), (nrow + 7.0)))
    ax0 = fig0.add_gridspec(nrow, ncol, wspace=0.3,hspace=0.3,top=1. - 0.25 / (nrow + 1), bottom=0.4 / (nrow + 1),
            left=0.3 / (ncol + 1), right=1 - 0.2 / (ncol + 1))
    fig0.suptitle('FWHM for all CCDs, run '+duration)
    
    for l in range(4):
        print('CCD ',l+1)
        pos = np.where(n_ccd==l+1)[0]
        x_plt0 = np.array(data[data_header[0]])[pos]
        
        #skip the plot if there's no datapoint
        if len(x_plt0)==0:
            print('No datapoints, skip plotting')
        else:
            #continue working
            #sorted by date
            date = np.array([datetime.date(int('20'+duration[:2]),datetime.datetime.strptime(x_plt0[i][2:5], '%b').month,int(x_plt0[i][:2])) for i in range(len(x_plt0))])
            x_plt = np.concatenate([np.sort(x_plt0[np.where(date==d)[0]]) for d in np.sort(np.unique(date))])
            pos_date = np.array([np.where(x_plt[i]==x_plt0)[0][0] for i in range(len(x_plt))])

            #convert x_plt into numbers
            x_plt_order = np.arange(len(x_plt))
            x_plt_order_min = min(x_plt_order) - 1
            x_plt_order_max = max(x_plt_order) + 1
            
            #FWHM plot
            axs0 = fig0.add_subplot(ax0[l],frameon=True)
            y_plt0 = np.array(data[data_header[1]])[pos][pos_date]
            axs0.scatter(x_plt_order,y_plt0,c='k',s=15)
            axs0.plot(np.linspace(x_plt_order_min,x_plt_order_max,10),
                    [median_all[l]]*10,c=colour[l],ls='dashed',
                 label='Median FWHM = '+'{:.4f}'.format(median_all[l])+r' $\AA$')
            axs0.fill_between(np.linspace(x_plt_order_min,x_plt_order_max,10),
                    [median_all[l]*0.95]*10,[median_all[l]]*10,color=colour[l],alpha=0.2,
                 label='-5% = '+'{:.4f}'.format(median_all[l]*0.95)+r' $\AA$')
            axs0.fill_between(np.linspace(x_plt_order_min,x_plt_order_max,10),
                    [median_all[l]*1.05]*10,[median_all[l]]*10,color=colour[l],alpha=0.2,
                 label='+5% = '+'{:.4f}'.format(median_all[l]*1.05)+r' $\AA$')
            axs0.legend(loc='upper right',fontsize=10)
            axs0.grid(color = 'k', linestyle = '--', linewidth = 0.5, alpha=0.4, which='major')
            axs0.grid(color = 'k', linestyle = ':', linewidth = 0.5, alpha=0.1, which='minor')
            axs0.minorticks_on()
            axs0.tick_params(axis='both',which='major',direction='in',length=6,pad=3,
                       top=True, right=True,bottom=True,left=True,
                     labeltop=False, labelright=False,labelbottom=True,labelleft=True)
            axs0.tick_params(axis='both',which='minor',direction='in',length=3,
                       top=True, right=True,bottom=True,left=True,
                     labeltop=False, labelright=False,labelbottom=True,labelleft=True)
            axs0.legend(loc='upper right',fontsize=6.5)
            axs0.set_title('CCD '+str(l+1))
            axs0.set_ylim(lower_lim[0][l],upper_lim[0][l])
            axs0.set_xlim(x_plt_order_min,x_plt_order_max)
            axs0.set_xticks(x_plt_order,x_plt,rotation = 90)
            axs0.tick_params(axis='x',labelsize=5)
            axs0.tick_params(axis='y',labelsize=8)
            axs0.set_ylabel(r'FWHM$_{Gaussian}$ / $\AA$')

            #Plot all 5 parameters
            nrow = 2
            ncol = 3
            fig = plt.figure(figsize=(1.5*(ncol + 7.0), (nrow + 7.0)))
            ax = fig.add_gridspec(nrow, ncol, wspace=0.3,hspace=0.3,top=1. - 0.25 / (nrow + 1), bottom=0.4 / (nrow + 1),
                left=0.3 / (ncol + 1), right=1 - 0.2 / (ncol + 1))
            fig.suptitle('CCD '+str(l+1)+', run '+duration+', QC plots')
        
            for p in range(len(data_header)-1):
                axs = fig.add_subplot(ax[p],frameon=True)
                y_plt = np.array(data[data_header[p+1]])[pos][pos_date]
                axs.scatter(x_plt_order,y_plt,c='k',s=15)
                if p==0:
                    axs.plot(np.linspace(x_plt_order_min,x_plt_order_max,10),[median_all[l]]*10,c=colour[l],ls='dashed',
                 label='Median FWHM = '+'{:.4f}'.format(median_all[l])+r' $\AA$')
                    axs.fill_between(np.linspace(x_plt_order_min,x_plt_order_max,10),
                        [median_all[l]*0.95]*10,[median_all[l]]*10,color=colour[l],alpha=0.2,
                     label='-5% = '+'{:.4f}'.format(median_all[l]*0.95)+r' $\AA$')
                    axs.fill_between(np.linspace(x_plt_order_min,x_plt_order_max,10),
                        [median_all[l]*1.05]*10,[median_all[l]]*10,color=colour[l],alpha=0.2,
                     label='+5% = '+'{:.4f}'.format(median_all[l]*1.05)+r' $\AA$')
                    axs.legend(loc='upper right',fontsize=10)
                else:
                    pass
                
                axs.grid(color = 'k', linestyle = '--', linewidth = 0.5, alpha=0.4, which='major')
                axs.grid(color = 'k', linestyle = ':', linewidth = 0.5, alpha=0.1, which='minor')
                axs.minorticks_on()
                axs.tick_params(axis='both',which='major',direction='in',length=6,pad=3,
                       top=True, right=True,bottom=True,left=True,
                     labeltop=False, labelright=False,labelbottom=True,labelleft=True)
                axs.tick_params(axis='both',which='minor',direction='in',length=3,
                       top=True, right=True,bottom=True,left=True,
                     labeltop=False, labelright=False,labelbottom=True,labelleft=True)
                axs.set_title(data_header[p+1])
                axs.set_ylim(lower_lim[p][l],upper_lim[p][l])
                axs.set_xlim(x_plt_order_min,x_plt_order_max)
                axs.set_xticks(x_plt_order,x_plt,rotation = 90)
                axs.tick_params(axis='x',labelsize=5)
                axs.tick_params(axis='y',labelsize=8)
        
            #save figure for all parameters
            figname = 'CCD'+str(l+1)+'_qc_plots'
            fig.savefig(outdir + figname +'.png',dpi=300)
    
    #save figure for FWHM only
    figname0 = 'FWHM_qc_plots'
    fig0.savefig(outdir + figname0 +'.png',dpi=300)
            
    print('Finish the process')
        

def calculate_fwhm(reduced_arc_path, outdir):
    '''function that allows everything in this script to work'''
    
    fitting(reduced_arc_path, outdir)
    plot_fwhm(outdir)
    

