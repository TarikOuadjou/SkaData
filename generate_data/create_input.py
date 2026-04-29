import os

import py21cmfast as p21c
SLURM_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

def compute_input_params(theta, seed = 1234,box_len=700, hiidim=350):
    user_params = p21c.SimulationOptions(
        BOX_LEN=box_len,
        HII_DIM=hiidim,
        DIM=hiidim*3,
        N_THREADS=SLURM_CPUS,
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