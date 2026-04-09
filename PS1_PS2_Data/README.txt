The description of the SDC3 inference submission and scoring method is contained in the relevant sections of the SDC3a description document, which can be accessed at https://sdc3.skao.int/challenges/inference/data.

This folder contains the PS1 ans PS2 datasets.  


PS1 data: 
Pk_PS1_*.txt = power spectra containing the EoR signal + instrumental noise for a 100h observation and a low-level of instrumental systematics.
err_Pk_PS1*.txt = the error on the power spectra due to cosmic variance. 

PS2 data: 
Pk_PS2*.txt = power spectra containing the EoR signal + instrumental noise for a 100h observation and a low-level of instrumental systematics.
err_Pk_PS2*.txt = the error on the power spectra due to cosmic variance. 

Ancillary data (common to PS1 and PS2)

bins_frequency.txt = list of frequency bins within which the 2D power spectra have been computed. The file contains minimum and maximum frequency for each bin.
bins_kpar.txt = adopted binning in k parallel. The file contains the bin centres.
bins_kper.txt = adopted binning in k perpendicular. The file contains the bin centres.
Pk_PS_averaged_noise_*.txt = the averaged power spectrum calcuated using 10 noise realisations.


Arrays of power spectrum values/errors are represented in the files as rows of constant k∥ with individual values in that row spanning the full range of k⟂, as specified by the provided binning scheme. The noise should be either subtracted or modelled with Pk_PS_averaged_noise_*.txt before performing inference. 
















