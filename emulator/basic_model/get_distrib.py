import torch
import numpy as np 
import glob
from emulator.basic_model.model import Emulator21cm,run_inference
from emulator.data_loader import PARAM_NAMES, Z_BINS, Z_MIDS, N_Z, PARAM_RANGES
CHECKPOINT_DEFAULT = "checkpoints/emulator.pt"
PARAM_NAMES  = ['ALPHA_STAR', 'F_STAR10', 'F_ESC10', 'ALPHA_ESC', 'M_TURN', 't_STAR']
PRIOR_BOUNDS = np.array([
    (-0.5,  1.0),   # ALPHA_STAR
    (-3.0,  0.0),   # F_STAR10
    (-3.0,  1.0),   # F_ESC10
    (-1.0,  0.5),   # ALPHA_ESC
    ( 8.0, 10.0),   # M_TURN
    ( 0.0,  1.0),   # t_STAR
])


PRIOR_BOUNDS = np.array([
    (0.0, 1.0)
    for param in PARAM_NAMES
])

def load_emulator():
    model = Emulator21cm(n_params=6, n_redshifts=N_Z)
    model.load_state_dict(torch.load('emulator/basic_model/' + CHECKPOINT_DEFAULT, map_location="cpu"))
    return model

def log_prior(theta: np.ndarray) -> float:
    lo, hi = PRIOR_BOUNDS[:, 0], PRIOR_BOUNDS[:, 1]
    return 0.0 if np.all(theta >= lo) and np.all(theta <= hi) else -np.inf
 
def log_likelihood(y_obs: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray) -> float:
    return -0.5 * float(np.sum(((y_obs - y_pred) / sigma) ** 2))

def run_mcmc(
    model,scalers,
    y_obs:        np.ndarray,
    sigma:        np.ndarray,
    n_steps:      int   = 10_000,
    burn_in:      int   = 2_000,
    proposal_std: float = 0.1,     
    theta_init:   np.ndarray = None,
    seed:         int   = 42,
) -> dict:
    rng    = np.random.default_rng(seed)
    widths = PRIOR_BOUNDS[:, 1] - PRIOR_BOUNDS[:, 0]
 
    if theta_init is None:
        theta_init = 0.5 * (PRIOR_BOUNDS[:, 0] + PRIOR_BOUNDS[:, 1])
 
    
    chain     = np.zeros((n_steps, len(PARAM_NAMES)))
    xhi_chain = np.zeros((n_steps, 3))
    lp_chain  = np.zeros(n_steps)
 
    # Evaluate initial point
    theta_cur = theta_init.copy()
    ps2d_cur, xhi_cur = run_inference(model, theta_cur,scalers=scalers)
    ps2d_cur = ps2d_cur.flatten()
    lp_cur = log_prior(theta_cur) + log_likelihood(y_obs, ps2d_cur, sigma)
 
    n_accept = 0
 
    for i in range(n_steps):
 
        # ── Propose ────────────────────────────────────────────────────────
        theta_prop = theta_cur + proposal_std * widths * rng.standard_normal(len(PARAM_NAMES))
 
        # ── Evaluate ────────────────────────────────────────────────────────
        lp_prior_prop = log_prior(theta_prop)
        if np.isfinite(lp_prior_prop):
            ps2d_prop, xhi_prop = run_inference(model, theta_prop, scalers=scalers)
            ps2d_prop = ps2d_prop.flatten()
            lp_prop = lp_prior_prop + log_likelihood(y_obs, ps2d_prop, sigma)
        else:
            lp_prop = -np.inf
 
        # ── Accept / Reject ─────────────────────────────────────────────────
        if np.log(rng.uniform()) < (lp_prop - lp_cur):
            theta_cur, xhi_cur, lp_cur = theta_prop, xhi_prop, lp_prop
            n_accept += 1
 
        chain[i]     = theta_cur
        xhi_chain[i] = xhi_cur
        lp_chain[i]  = lp_cur
 
        # ── Progress ────────────────────────────────────────────────────────
        if (i + 1) % 500 == 0:
            print(f"  step {i+1:>6}/{n_steps}  |  "
                  f"accept rate: {n_accept/(i+1):.2%}  |  "
                  f"log-post: {lp_cur:.2f}")
 
    return {
        "chain":         chain,
        "posterior":     chain[burn_in:],
        "xhi_posterior": xhi_chain[burn_in:],
        "log_post":      lp_chain[burn_in:],
        "accept_rate":   n_accept / n_steps,
    }

def main():
    model   = load_emulator()
    checkpoint_dir = "emulator/basic_model/checkpoints"
    scalers = np.load(f"{checkpoint_dir}/scalers.npz")
    ## Generating obs
    split = torch.load("emulator/basic_model/checkpoints/dataset_split.pt")
    test_thetas = split["test_thetas"]
    test_ps2d = split["test_ps2d"]
    test_xhi = split["test_xhi"]

    theta_obs = test_thetas[0].numpy()
    test_ps2d_obs = test_ps2d[0].numpy().flatten()
    sigma = np.concatenate([
    np.loadtxt(f).flatten()
    for f in sorted(glob.glob("PS1_PS2_Data/err_Pk_PS2_*.txt"))
    ])
    print(theta_obs)
    theta_init = theta_obs.copy()
    results = run_mcmc(model, scalers, test_ps2d_obs, sigma, n_steps=10_000, burn_in=2_000,theta_init=theta_init, seed=42)
    np.savez('emulator/basic_model/mcmc_results.npz', **results)

if __name__ == "__main__":
    main()