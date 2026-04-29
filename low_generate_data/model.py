
from glob import glob

from .create_input import compute_input_params
import py21cmfast as p21c
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import tools21cm as t2c
import os
p21c.config['HALO_CATALOG_MEM_FACTOR'] = 2.0
cache_path = "/gpfs/workdir/ouadjout/cache"
os.makedirs(cache_path, exist_ok=True)
cache = p21c.OutputCache(cache_path)

def adding_instrumental_noise(ps2d_list):
    files_noise = sorted(glob("PS1_PS2_Data/Pk_PS_averaged_noise_*.txt"))
    for i, f in enumerate(files_noise):
        data_average_noise = np.loadtxt(f)
        ps2d_list[i] += data_average_noise
    return ps2d_list

def model(theta,box_len=256, hiidim=128,cache_dir=cache_path,seed = 1234):
    inputs = compute_input_params(theta, seed=seed, box_len=box_len, hiidim=hiidim)
    lcn = p21c.RectilinearLightconer.between_redshifts(
        min_redshift=6.25,
        max_redshift=8.41,
        quantities=("brightness_temp", "density", "neutral_fraction"),
        resolution=inputs.simulation_options.cell_size,
        # index_offset=0,   
    )

    lightcone = p21c.run_lightcone(
        lightconer=lcn,
        inputs=inputs,  
        #cache=cache,
        progressbar=False,
        write = False,
    )

    # Tes paliers de redshift
    z_bounds = [8.41, 7.56, 6.85, 6.25]

    ps2d_list = []
    kperp_list = []
    kpar_list = []
    x_HI_mean_list = []
    for i in range(len(z_bounds) - 1):
        z_max, z_min = z_bounds[i], z_bounds[i+1]
        mask = (lightcone.lightcone_redshifts <= z_max) & (lightcone.lightcone_redshifts >= z_min)
        indices = np.where(mask)[0]
        sub_box = lightcone.lightcones["brightness_temp"][:, :, indices]
        sub_box = sub_box/1000.0  # Convert to K
        dist_max = lightcone.lightcone_distances[indices[0]]
        dist_min = lightcone.lightcone_distances[indices[-1]]
        depth_mpc = np.abs(dist_max - dist_min).value
        h = lightcone.cosmo_params.hlittle  
        box_dims = (lightcone.simulation_options.BOX_LEN*h, lightcone.simulation_options.BOX_LEN*h, depth_mpc*h)
        k_edges = np.arange(0.025, 0.575, 0.05)
        ps2d, kperp, kpar = t2c.power_spectrum_2d(
                    sub_box,
                    kbins=[k_edges, k_edges],
                    box_dims=box_dims)
        ps2d_list.append(ps2d)
        kperp_list.append(kperp)
        kpar_list.append(kpar)
        xH_box = lightcone.lightcones["neutral_fraction"][:, :, indices]
        x_HI_mean = np.mean(1 - xH_box)
        print(f"PS calculé pour z={z_min:.2f}-{z_max:.2f} (Profondeur: {depth_mpc:.1f} Mpc), x_HI_mean={x_HI_mean:.3f})")
        x_HI_mean_list.append(x_HI_mean)
    ps2d_list = adding_instrumental_noise(ps2d_list)
    return np.array(ps2d_list), x_HI_mean_list

