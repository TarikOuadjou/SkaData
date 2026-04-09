import os
import numpy as np
import py21cmfast as p21c
import tools21cm as t2c
from matplotlib.colors import LogNorm
import glob
import matplotlib.pyplot as plt
import time


SLURM_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
p21c.config['HALO_CATALOG_MEM_FACTOR'] = 2.0
# Utilisation d'un dossier temporaire pour le cache
cache_path = "/gpfs/workdir/ouadjout/cache"
os.makedirs(cache_path, exist_ok=True)
cache = p21c.OutputCache(cache_path)

prior_dic = {
    'ALPHA_STAR': (-0.5, 1.0),
    'F_STAR10': (-3.0, 0.0),
    'F_ESC10': (-3.0, 1.0),
    'ALPHA_ESC': (-1.0, 0.5),
    'M_TURN': (8.0, 10.0),  
    't_STAR': (0.0, 1.0),
    'X_RAY_SPEC_INDEX': (0.5, 2.0),
}

def compute_input_params(theta, seed = 1234,box_len=700, hiidim=350):
    user_params = p21c.SimulationOptions(
        BOX_LEN=box_len,
        HII_DIM=hiidim,
        DIM=hiidim*3,
        N_THREADS=16,
        SAMPLER_MIN_MASS= 1e10,
        DEXM_OPTIMIZE_MINMASS=3e11,
    )

    astro_options = p21c.AstroOptions(
        USE_TS_FLUCT=False,
        USE_X_RAY_HEATING=False,
        INHOMO_RECO=False,
        M_MIN_in_Mass=True
    )

    cosmo_params = p21c.CosmoParams(
        hlittle=0.6766,
        OMm=0.30964,
        OMb=0.04897,
        POWER_INDEX=0.9665,
        SIGMA_8=0.8102
    )
    astro_params = p21c.AstroParams(
        ALPHA_STAR=theta['ALPHA_STAR'], 
        F_STAR10=theta['F_STAR10'],
        F_ESC10=theta['F_ESC10'],
        ALPHA_ESC=theta['ALPHA_ESC'],
        M_TURN=theta['M_TURN'],
        t_STAR=theta['t_STAR'],
        X_RAY_SPEC_INDEX = theta['X_RAY_SPEC_INDEX'],
        NU_X_BAND_MAX=2000.0, # Max of the band, don't moove this one ! 
    )
    inputs = p21c.InputParameters(
        cosmo_params=cosmo_params,
        astro_options=astro_options,
        astro_params=astro_params,
        simulation_options=user_params,
        random_seed=seed,
        node_redshifts=p21c.wrapper.inputs.get_logspaced_redshifts(
            min_redshift=6.00,
            z_step_factor=1.02,
            max_redshift=8.5,))

    return inputs


def log_prior(theta):
    for key, value in theta.items():
        a, b = prior_dic[key]
        
        # hors des bornes → proba nulle
        if value < a or value > b:
            return -np.inf
    
    # constante (optionnelle)
    logp = 0.0
    for (a, b) in prior_dic.values():
        logp += -np.log(b - a)
    
    return logp


def model(theta,box_len=700, hiidim=350):
    output_dir = "results_simu"
    os.makedirs(output_dir, exist_ok=True)
    inputs = compute_input_params(theta, seed=1234, box_len=box_len, hiidim=hiidim)
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
        cache=cache,
        progressbar=True
    )

    # Tes paliers de redshift
    z_bounds = [8.41, 7.56, 6.85, 6.25]

    ps2d_list = []
    kperp_list = []
    kpar_list = []
    x_HI_mean_list = []
    for i in range(len(z_bounds) - 1):
        z_max, z_min = z_bounds[i], z_bounds[i+1]
        # 1. Trouver les indices des tranches dans le lightcone
        mask = (lightcone.lightcone_redshifts <= z_max) & (lightcone.lightcone_redshifts >= z_min)
        indices = np.where(mask)[0]
        
        # 2. Extraire le sous-volume (box)
        # La forme est (HII_DIM, HII_DIM, N_slices)
        sub_box = lightcone.lightcones["brightness_temp"][:, :, indices]
        sub_box = sub_box/1000.0  # Convert to K
        # 3. Calculer la profondeur physique du morceau (en Mpc)
        # C'est crucial pour que les unités de k_parallèle soient correctes
        dist_max = lightcone.lightcone_distances[indices[0]]
        dist_min = lightcone.lightcone_distances[indices[-1]]
        depth_mpc = np.abs(dist_max - dist_min).value
        h = lightcone.cosmo_params.hlittle  
        # Dimensions de la boîte pour tools21cm (L_x, L_y, L_z)
        box_dims = (lightcone.simulation_options.BOX_LEN*h, lightcone.simulation_options.BOX_LEN*h, depth_mpc*h)
        k_edges = np.arange(0.025, 0.575, 0.05)

        ps2d, kperp, kpar = t2c.power_spectrum_2d(
                    sub_box,
                    kbins=[k_edges, k_edges],
                    box_dims=box_dims)
        ps2d_list.append(ps2d)
        kperp_list.append(kperp)
        kpar_list.append(kpar)
        plt.figure(figsize=(8, 7))
        pcm = plt.pcolormesh(kperp, kpar, ps2d, 
                            norm=LogNorm(vmin=ps2d.min(), vmax=ps2d.max()),
                            cmap='viridis')

        #plt.colorbar(pcm, label=r'$P(k_\perp, k_\parallel)$ [$K^2 h^{-3} Mpc^3$]')
        plt.xlabel(r'$k_\perp$ [$h$/Mpc]')
        plt.ylabel(r'$k_\parallel$ [$h$/Mpc]')
        plt.title(f'Power Spectrum 2D pour z={z_min:.2f}-{z_max:.2f}')
        #plt.show()
        xH_box = lightcone.lightcones["neutral_fraction"][:, :, indices]
        np.savez(f"{output_dir}/data_z_{z_min:.2f}-{z_max:.2f}.npz", ps2d=ps2d, kperp=kperp, kpar=kpar)
        # fraction neutre moyenne
        x_HI_mean = np.mean(1-xH_box)
        print(f"PS calculé pour z={z_min:.2f}-{z_max:.2f} (Profondeur: {depth_mpc:.1f} Mpc), x_HI_mean={x_HI_mean:.3f})")
        x_HI_mean_list.append(x_HI_mean)
    # Adding Noise
    files_noise = sorted(glob.glob("PS1_PS2_Data/Pk_PS_averaged_noise_*.txt"))
    for i, f in enumerate(files_noise):
        data_average_noise = np.loadtxt(f)
        ps2d_list[i] += data_average_noise
    return np.array(ps2d_list), x_HI_mean_list

def compute_sigma():
    files = sorted(glob.glob("PS1_PS2_Data/err_Pk_PS2_*.txt"))  
    ps2_true_err_list = []
    ps2_average_noise_list = []
    for i, f in enumerate(files):
        data_std = np.loadtxt(f)
        ps2_true_err_list.append(data_std)
    return np.array(ps2_true_err_list).flatten()

def log_likelihood(y_true, y_pred):
    sigma = compute_sigma()
    chi_square_vec = y_true - y_pred
    norm = 1.0 / np.sqrt((2 * np.pi) ** len(sigma) * np.prod(sigma ** 2))
    exponent = -0.5 * np.sum((chi_square_vec / sigma) ** 2)
    return exponent   

def log_posterior(theta, y_true):
    y_pred, x_h = model(theta)  # ton modèle
    y_pred = y_pred.flatten()
    likelihood = log_likelihood(y_true, y_pred)
    prior = log_prior(theta)
    print(f"Log-likelihood: {likelihood:.2f}, Log-prior: {prior:.2f}, Log-posterior: {likelihood + prior:.2f}")
    return (likelihood + prior), x_h

def compute_ps2true():
    files = sorted(glob.glob("PS1_PS2_Data/Pk_PS2_*.txt"))
    ps2_true_list = []
    for i, f in enumerate(files):
        data = np.loadtxt(f)
        ps2_true_list.append(data)
    return np.array(ps2_true_list).flatten()

def run_convergence_test(theta_fixed):
    # Defining configurations to test
    configs = [
        (250, 128),
        (500, 250),
        (700, 350),
    ]
    ref_key = (700, 350)
    z_labels = ["z 8.41→7.56", "z 7.56→6.85", "z 6.85→6.25"]

    # ── Data Collection ─────────────────────────────────────
    results = {}
    timings = {}
    for bl, hd in configs:
        print(f"\n=== box_len={bl}, hiidim={hd} ===")
        t0 = time.time()
        # Assuming model() is your simulation wrapper
        ps2d_list, xHI = model(theta_fixed, box_len=bl, hiidim=hd)
        elapsed = time.time() - t0
        timings[(bl, hd)] = elapsed
        results[(bl, hd)] = {"ps2d": np.array(ps2d_list), "xHI": xHI}
        print(f" {bl}:{hd}   → Duration: {elapsed/60:.1f} min ({elapsed:.0f} s)")

    ref_ps = results[ref_key]["ps2d"]  # shape (3, Nkperp, Nkpar)

    # ── Relative Errors ─────────────────────────────────────
    errors = {}
    errors_max = {}
    for key, val in results.items():
        if key == ref_key:
            errors[key] = np.zeros(3)
            errors_max[key] = np.zeros(3)
        else:
            ps = val["ps2d"]
            # Calculate mean and max relative error per redshift bin
            errors[key] = np.array([
                np.nanmean(np.where(
                    np.abs(ref_ps[i]) > 1e-10,
                    np.abs(ps[i] - ref_ps[i]) / np.abs(ref_ps[i]),
                    np.nan
                ))
                for i in range(3)
            ])
            errors_max[key] = np.array([
                np.nanmax(np.where(
                    np.abs(ref_ps[i]) > 1e-10,
                    np.abs(ps[i] - ref_ps[i]) / np.abs(ref_ps[i]),
                    np.nan
                ))
                for i in range(3)
            ])

    # ── Figure: Relative Error in % ──────────────────────────
    non_ref = [c for c in configs if c != ref_key]
    n_rows = len(non_ref)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 4 * n_rows), constrained_layout=True)
    
    # Handle single row case for consistency
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, (bl, hd) in enumerate(non_ref):
        key = (bl, hd)
        ps  = results[key]["ps2d"]
        t_str = f"{timings[key]/60:.1f} min"
        
        for zi in range(3):
            ax = axes[row, zi]
            # Convert to percentage
            err_map = np.abs(ps[zi] - ref_ps[zi]) / np.abs(ref_ps[zi]) * 100 
            
            im = ax.pcolormesh(err_map, cmap="YlOrRd", vmin=0, vmax=err_map.max())
            plt.colorbar(im, ax=ax, label="Relative Error [%]")
            
            ax.set_title(
                f"bl={bl}, hd={hd} | {z_labels[zi]} | {t_str}\n"
                f"mean={errors[key][zi]:.1%}  max={errors_max[key][zi]:.1%}",
                fontsize=10
            )
            ax.set_xlabel(r"$k_\perp$ bin")
            ax.set_ylabel(r"$k_\parallel$ bin")

    ref_t = f"{timings[ref_key]/60:.1f} min"
    fig.suptitle(
        f"Relative Error vs Reference Power Spectrum (bl=700, hd=350, {ref_t})",
        fontsize=13
    )
    
    plt.savefig("convergence_3configs.pdf", bbox_inches="tight")
    plt.show()
    print("\nFigure saved: convergence_3configs.pdf")

def main():
    theta_fixed = {
    'ALPHA_STAR': 0.5, 
    'F_STAR10': -1.3,
    'F_ESC10': -1,
    'ALPHA_ESC': -0.5,
    'M_TURN': 8.69, 
    't_STAR': 0.5,
    'X_RAY_SPEC_INDEX': 1.0,     
    }
    run_convergence_test(theta_fixed)

if __name__ == "__main__":
    main()



