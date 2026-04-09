import os
import numpy as np
import py21cmfast as p21c
import tools21cm as t2c
SLURM_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
p21c.config['HALO_CATALOG_MEM_FACTOR'] = 2.0
# Utilisation d'un dossier temporaire pour le cache
cache_path = "/gpfs/workdir/ouadjout/cache"
os.makedirs(cache_path, exist_ok=True)
cache = p21c.OutputCache(cache_path)

user_params = p21c.SimulationOptions(
    BOX_LEN=700,
    HII_DIM=350,
    DIM=1050,
    N_THREADS=16,
    SAMPLER_MIN_MASS= 1e10,
    DEXM_OPTIMIZE_MINMASS=3e11,
)

astro_params = p21c.AstroParams(
    ALPHA_STAR=0.5,
    F_STAR10=-1.3,
    F_ESC10=-1,
    ALPHA_ESC=-0.5,
    M_TURN=8.69, # 5e10 M_sun
    t_STAR=0.5,
    L_X=40.5,    # L_X per SFR, in log10(erg/s/M_sun/yr)         
    NU_X_THRESH=500.0,     #E_0    
    NU_X_BAND_MAX=2000.0, # Max of the band, don't moove this one ! 
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

inputs = p21c.InputParameters(
    cosmo_params=cosmo_params,
    astro_options=astro_options,
    astro_params=astro_params,
    simulation_options=user_params,
    random_seed=1234
)

def main():
    redshifts = [6.54, 7.19, 7.96]
    output_dir = "results_simu"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Démarrage de la simulation séquentielle (Optimisation RAM)...")
    coevals = p21c.run_coeval(
            out_redshifts=redshifts,
            inputs=inputs,
            progressbar=False,
            cache = cache
    )
    for coeval in coevals:
        box = coeval.brightness_temp
        box_dims = coeval.inputs.simulation_options.BOX_LEN
        k_edges = np.arange(0.025, 0.575, 0.05)
        
        # Calcul
        ps2d, kperp, kpar = t2c.power_spectrum_2d(
            box,
            kbins=[k_edges, k_edges],
            box_dims=box_dims
        )
        z = coeval.redshift
        # Sauvegarde immédiate sur disque
        np.savez(f"{output_dir}/data_z_{z:.2f}.npz", ps2d=ps2d, kperp=kperp, kpar=kpar)
        
        print(f"Z={z:.2f} | Ionisation: {coeval.ionized_box.global_xH:.4f}")

    print("Simulation terminée. Fichiers sauvegardés individuellement.")

if __name__ == "__main__":
    main()