Sree Jun 2025
There was a significant reduce in througput of the AAOmega including bundle H since Nov 2024.
I generated median throughput with Nov 2024, Apr, May 2025 data for AAOmega. 
mean_throughput_E2V2A_sinceNov2024.fits with start date = 204.84 (Nov 2024)
mean_throughput_E2V3A_sinceNov2024.fits
and modified start, finish date of the old mean thput files as follows:
In [11]: with fits.open("mean_throughput_E2V2A.fits", mode='update') as hdul:
    ...:   header = hdul[0].header
    ...:   header['DATESTRT'] = 2022.0
    ...:   header['DATEFNSH'] = 2024.84
    ...:   hdul.flush()
In [14]: with fits.open("mean_throughput_E2V3A.fits", mode='update') as hdul:
    ...:   header = hdul[0].header
    ...:   header['DATESTRT'] = 2022.0
    ...:   header['DATEFNSH'] = 2024.84
    ...:   hdul.flush()



The new median file will be used for qc till instrument treats happen. 
See also manager.qc_throughput_spectrum()
The mean thput should be regenerated once the low thput bundles are improved. 


Sree May 2024
Follow the below to generate mean throughput files using the reduced primary standatd star frames 
The median throughputs are then used to calculate transmission for qc in the pipeline
Should update this when there are any significant changes in the bundle thputs for standard stars (H, U)
	> cd ~/local/software/hector/standards/throughput/
	> import hector;from glob import glob
	> root_list = glob('/storage2/sree/hector/dr/v2/??????_??????')
	  modify root_list to only include necessary runs (e.g. recent 1 yr data etc.)
	> mngr_list = [hector.manager.Manager(root) for root in root_list]
	  the above line takes super long time loading all the data from many runs. Be patient or reduce runs to consider in root_list
	> hector.qc.fluxcal.calculate_mean_throughput('mean_throughput_E2V2A.fits', mngr_list, 'E2V2A', date_start=2023.0)
	  repeat for E2V2A, E2V3A, E2V_15111-10-01, E2V_17352-10-01



