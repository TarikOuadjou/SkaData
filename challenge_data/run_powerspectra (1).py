# 25/7/24
# currebtly using conda env /data-cold/home/p.hartley/anaconda3/envs/ps_eor_0.29.3
# modified to produce the submission for test dataset
# process one file at the time, take input and output from arguments



import os,sys
import glob
import itertools
import numpy as np
import matplotlib.pyplot as plt
import astropy.time as atime
from astropy.coordinates import SkyCoord
from astropy.io import fits
from ps_eor import datacube, pspec, psutil, simu, fitutil, fgfit, flagger, ml_gpr, obssimu
import astropy.constants as const
import matplotlib as mpl
from matplotlib import colors

from astropy.cosmology import FlatLambdaCDM
psutil.set_cosmology(FlatLambdaCDM(67.66, Om0=0.30964))

mpl.rcParams['image.cmap'] = 'Spectral_r'
mpl.rcParams['image.origin'] = 'lower'
mpl.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['axes.grid'] = True


# read input file with the list of cubes and the list of PS names

#files=np.loadtxt('cube_names.txt',dtype='str')
#files=np.loadtxt('cube_names_PS3.txt',dtype='str')
files=np.loadtxt('cube_names_PS3.txt',dtype='str')

#print(files)
#exit()
#cube_names=files[:,0]
#PS_tags=files[:,1]

#dir='/data-archive/sdc/SDC3/inference/submission_Oct24/'
dir='/data-archive/sdc/SDC3/inference/submission_Feb25/'

output_path=dir+'spectra/'
input_path=dir+'maps/'


def get_ps_gen_square(cube,  **kargs):
    ''' This function return a ps_gen object which will return a power-spectra with the SDC3a binning scheme '''

    z = psutil.freq_to_z(cube.freqs.mean()) 
    print ('z', z)
    du = psutil.k_to_l(0.05, z) / (2 * np.pi) 
    umin = du / 2.
    umax = psutil.k_to_l(0.5, z) / (2 * np.pi) + du / 2 
    M = int((1 / psutil.k_to_delay(0.05, z)) / 0.1e6)
    print (du, umin, umax)
    print('M', M)
    ps_builder = pspec.PowerSpectraBuilder()
    ps_gen = ps_builder.get(cube,  du=du, umin=umin, umax=umax, uniform_u_bins=True, **kargs)
    ps_gen.eor.M = M
    ps_gen._compute_delays()

    return ps_gen




psf_file=[input_path+'All.msn_psf.fits'] # use this file for all maps
nulimits=np.loadtxt('ps_nurange.txt')
nulimits = np.array([[151,165.9], [166,180.9], [181,195.9] ]) 
min_freqs=nulimits[:,0]
max_freqs=nulimits[:,1]


for i in range(0,len(min_freqs)):

    print ()
    print ()
    print ('**** processing freq bin', min_freqs[i], 'to', max_freqs[i], '****')

    fmin = (min_freqs[i])
    fmax = (max_freqs[i])
    fmin_st=str(fmin)
    fmax_st=str(fmax)
    fmin = (min_freqs[i])*1e6
    fmax = (max_freqs[i])*1e6
    f_ref = (fmax-fmin)/2
    z = psutil.freq_to_z((fmin+f_ref)) 
    print ('z', z)
    kpar_min = 0.05
    mydu = psutil.k_to_l(kpar_min, z) / (2 * np.pi)
    myumin = mydu /2
    myumax = psutil.k_to_l(0.5, z) / (2 * np.pi) + mydu/2
    fmin_base = fmin
    print ('mydu, myumin, myumax', mydu, myumin, myumax)


    #processed_cubes = []
    for file in files:
        print(file)
        cube_name=input_path+file[0]
        PS_tag=file[1]
        #img_file=[cube_name+'.msn_dimage.fits'] #dimage is the deconvolved image. 
        img_file=[cube_name+'.msn_image.fits']
        #psf_file=[cube_name+'.msn_psf.fits']
        print('now doing',img_file,psf_file)
        
        cube_observed = datacube.CartDataCube.load_from_fits_image_and_psf(img_file, psf_file,myumin, myumax, np.radians(4), int_time=10, total_time=3600 * 4, window_function=datacube.WindowFunction('boxcar'))
        
        cube_observed_zbin = cube_observed.get_slice(fmin, fmax)
        #processed_cubes.append(cube_observed_zbin)
        print ('got freq slice')

    
        ps_gen_zbin = get_ps_gen_square(cube_observed_zbin, ft_method='nudft', window_fct='hann', primary_beam='ska_low', rmean_freqs=False)



        #fig, axs = plt.subplots(dpi=120, figsize=(10, 8), ncols=1, nrows=1, sharex=True, sharey=True)
        #((ax1)) = axs
        
        ps1 = ps_gen_zbin.get_ps2d(cube_observed_zbin)

        #print (ps1.data)



        nbins = 10
        kpar_bins = ps1.k_par[:nbins]
        kper_bins = ps1.k_per
        ps = ps1.data[:nbins]
        err = ps1.err[:nbins]

        print('ps=')
        print(ps)

        print('err=')
        print(err)
        
        #exit()
        #fig = plt.figure(figsize=(10,8))
        #plt.subplot(111)
        #plt.pcolormesh(kper_bins,kpar_bins,(ps1.data[:nbins].T),cmap="Spectral_r", norm=colors.LogNorm(vmin=1e-3, vmax=1e-0))
        #plt.title('PS(EoR_obs)')
        #plt.colorbar()
        #plt.xlabel('kper', fontdict=None, labelpad=None)
        #plt.ylabel('kpar', fontdict=None, labelpad=None)
        output_dir = output_path
        file1 = output_dir+"Pk_"+PS_tag+"_"+fmin_st+"_"+fmax_st+".txt"
        file2 = output_dir+"kpar_"+PS_tag+"_"+fmin_st+"_"+fmax_st+".txt"
        file3 = output_dir+"kper_"+PS_tag+"_"+fmin_st+"_"+fmax_st+".txt"
        file4 = output_dir+"err_Pk_"+PS_tag+"_"+fmin_st+"_"+fmax_st+".txt"
        
        print ('kper bins', kper_bins)
        print ('kpar_bins', kpar_bins)
        #print (ps1.data[:nbins])
        np.savetxt(file2, kpar_bins)#, fmt='%e')
        np.savetxt(file3, kper_bins)#, fmt='%e')
        np.savetxt(file1, ps, fmt='%e')
        np.savetxt(file4, err, fmt='%e')


        #fmt='%1.3f'
