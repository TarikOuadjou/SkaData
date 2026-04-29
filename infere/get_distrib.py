import torch
from emulator.model import Emulator21cm
import numpy as np 
import glob
def run_emulator(model, theta: np.ndarray, device: str):
    """
    Args:
        theta : (6,) array in physical units
    Returns:
        ps2d_flat : (300,) flattened  [3 redshifts × 10 × 10]
        xhi       : (3,)  neutral fractions
    """
    t = torch.tensor(theta, dtype=torch.float32, device=device).unsqueeze(0)
 
    with torch.no_grad():
        ps2d, xhi = model(t)
 
    ps2d_flat = ps2d.squeeze(0).cpu().numpy().flatten()   # (300,)
    xhi_out   = xhi.squeeze(0).cpu().numpy()              # (3,)
    return ps2d_flat, xhi_out

def log_prior(theta: np.ndarray) -> float:
    lo, hi = PRIOR_BOUNDS[:, 0], PRIOR_BOUNDS[:, 1]
    return 0.0 if np.all(theta >= lo) and np.all(theta <= hi) else -np.inf
 
 
def log_likelihood(y_obs: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray) -> float:
    return -0.5 * float(np.sum(((y_obs - y_pred) / sigma) ** 2))

def run_mcmc(
    model,
    y_obs:        np.ndarray,
    sigma:        np.ndarray,
    device:       str   = "cpu",
    n_steps:      int   = 10_000,
    burn_in:      int   = 2_000,
    proposal_std: float = 0.05,     # fraction of prior width per step
    theta_init:   np.ndarray = None,
    seed:         int   = 42,
) -> dict:
    """
    Metropolis-Hastings MCMC.
 
    Returns a dict with keys:
        chain          : (n_steps, 6)          full chain
        posterior      : (n_steps-burn_in, 6)  post burn-in parameters
        xhi_posterior  : (n_steps-burn_in, 3)  post burn-in neutral fractions
        log_post       : (n_steps-burn_in,)    log-posterior values
        accept_rate    : float
    """
    rng    = np.random.default_rng(seed)
    widths = PRIOR_BOUNDS[:, 1] - PRIOR_BOUNDS[:, 0]
 
    # Starting point: prior centre if not provided
    if theta_init is None:
        theta_init = 0.5 * (PRIOR_BOUNDS[:, 0] + PRIOR_BOUNDS[:, 1])
 
    # Storage
    chain     = np.zeros((n_steps, len(PARAM_NAMES)))
    xhi_chain = np.zeros((n_steps, 3))
    lp_chain  = np.zeros(n_steps)
 
    # Evaluate initial point
    theta_cur = theta_init.copy()
    ps2d_cur, xhi_cur = run_emulator(model, theta_cur, device)
    lp_cur = log_prior(theta_cur) + log_likelihood(y_obs, ps2d_cur, sigma)
 
    n_accept = 0
 
    for i in range(n_steps):
 
        # ── Propose ────────────────────────────────────────────────────────
        theta_prop = theta_cur + proposal_std * widths * rng.standard_normal(len(PARAM_NAMES))
 
        # ── Evaluate ────────────────────────────────────────────────────────
        lp_prior_prop = log_prior(theta_prop)
        if np.isfinite(lp_prior_prop):
            ps2d_prop, xhi_prop = run_emulator(model, theta_prop, device)
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

PARAM_NAMES  = ['ALPHA_STAR', 'F_STAR10', 'F_ESC10', 'ALPHA_ESC', 'M_TURN', 't_STAR']
PRIOR_BOUNDS = np.array([
    (-0.5,  1.0),   # ALPHA_STAR
    (-3.0,  0.0),   # F_STAR10
    (-3.0,  1.0),   # F_ESC10
    (-1.0,  0.5),   # ALPHA_ESC
    ( 8.0, 10.0),   # M_TURN
    ( 0.0,  1.0),   # t_STAR
])

ckpt = torch.load("emulator/checkpoints/emulator.pt", map_location="cpu")
print(type(ckpt))
print(ckpt.keys() if isinstance(ckpt, dict) else "→ pas un dict, c'est le modèle directement")


model = Emulator21cm(n_params=6, n_redshifts=3)

model.load_state_dict(ckpt)  
model.eval()  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
point_number = "0002"
y_obs = np.load(f"low_generate_data/results/point_{point_number}/ps2d.npy", allow_pickle = True) 
theta_obs = np.load(f"low_generate_data/results/point_{point_number}/theta.npy", allow_pickle = True)
xHI_obs = np.load(f"low_generate_data/results/point_{point_number}/xHI.npy", allow_pickle = True)  
