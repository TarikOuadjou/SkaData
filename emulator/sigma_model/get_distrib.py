from scipy.optimize import minimize
import torch
import numpy as np
import glob
import emcee
import math
from emulator.sigma_model.model_prob import Emulator21cm, run_inference
from emulator.data_loader import PARAM_NAMES, Z_BINS, Z_MIDS, N_Z, PARAM_RANGES
CHECKPOINT_DEFAULT = "checkpoints/emulator.pt"
PARAM_NAMES = ['ALPHA_STAR', 'F_STAR10', 'F_ESC10', 'ALPHA_ESC', 'M_TURN', 't_STAR']

PRIOR_BOUNDS = np.array([
    (0.0, 1.0)
    for param in PARAM_NAMES
])

N_DIM = len(PARAM_NAMES)

def diagnose_map(model, scalers, y_obs, theta_true, theta_map):
    _, _, _, mu_true, sig_true = run_inference(model, theta_true, scalers=scalers)
    _, _, _, mu_map,  sig_map  = run_inference(model, theta_map,  scalers=scalers)

    ll_true = log_likelihood_lognormal_log10(y_obs, mu_true.flatten(), sig_true.flatten())
    ll_map  = log_likelihood_lognormal_log10(y_obs, mu_map.flatten(),  sig_map.flatten())

    print(f"ll at true theta : {ll_true:.2f}")
    print(f"ll at MAP theta  : {ll_map:.2f}")
    print(f"delta            : {ll_map - ll_true:.2f}  (should be > 0)")

def load_emulator():
    model = Emulator21cm(n_params=6, n_redshifts=N_Z)
    model.load_state_dict(torch.load('emulator/sigma_model/' + CHECKPOINT_DEFAULT, map_location="cpu"))
    return model

def log_prior(theta: np.ndarray) -> float:
    lo, hi = PRIOR_BOUNDS[:, 0], PRIOR_BOUNDS[:, 1]
    return 0.0 if np.all(theta >= lo) and np.all(theta <= hi) else -np.inf


def log_likelihood_lognormal_log10(
    x_obs: np.ndarray,
    mu_log10: np.ndarray,
    sigma_log10: np.ndarray,
) -> float: 
    x_obs       = np.asarray(x_obs,                        dtype=np.float64)
    mu_log10    = np.asarray(mu_log10.detach().cpu(),      dtype=np.float64)
    sigma_log10 = np.asarray(sigma_log10,                  dtype=np.float64)

    mu_nat    = mu_log10    * np.log(10)
    sigma_nat = sigma_log10 * np.log(10)

    return float(np.sum(
        - np.log(x_obs)
        - np.log(sigma_nat)
        - 0.5 * ((np.log(x_obs) - mu_nat) / sigma_nat) ** 2
    ))

def log_prob(theta, model, scalers, y_obs):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    _, _, _, ps_mu_log10, ps_sigma_log10 = run_inference(model, theta, scalers=scalers)
    ll = log_likelihood_lognormal_log10(y_obs, ps_mu_log10.flatten(), ps_sigma_log10.flatten())
    return lp + ll

def neg_log_prob(theta, model, scalers, y_obs):
    """Negative log-posterior (minimized to find the MAP estimate)."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return np.inf
    _, _, _, ps_mu_log10, ps_sigma_log10 = run_inference(model, theta, scalers=scalers)
    ll = log_likelihood_lognormal_log10(
        y_obs, ps_mu_log10.flatten(), ps_sigma_log10.flatten()
    )
    return -(lp + ll)

def find_map_estimate(
    model,
    scalers,
    y_obs: np.ndarray,
    n_restarts: int = 40,
    seed: int = 42,
) -> np.ndarray:
    rng     = np.random.default_rng(seed)
    bounds  = list(zip(PRIOR_BOUNDS[:, 0], PRIOR_BOUNDS[:, 1]))
    y_obs_t = torch.tensor(y_obs, dtype=torch.float32)

    best_val, best_theta = np.inf, None

    for i in range(n_restarts):
        theta0 = rng.uniform(PRIOR_BOUNDS[:, 0], PRIOR_BOUNDS[:, 1])
        result = minimize(
            neg_log_prob_with_grad,
            theta0,
            args=(model, scalers, y_obs_t),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
        )
        if result.fun < best_val:
            best_val, best_theta = result.fun, result.x
        print(f"  restart {i+1:>2}/{n_restarts}  ll={-result.fun:.2f}  success={result.success}")

    print(f"\nMAP : {best_theta}")
    print(f"ll  : {-best_val:.4f}")
    return best_theta

def run_mcmc(
    model,
    scalers,
    y_obs:        np.ndarray,
    n_walkers:    int   = 32,
    n_steps:      int   = 5_000,
    burn_in:      int   = 1_000,
    seed:         int   = 42,
    theta_init:   np.ndarray = None,
    init_noise_pct: float = 0.02,   # Gaussian spread as % of prior width
) -> dict:
    rng = np.random.default_rng(seed)

    # ── Default: MAP estimate via gradient descent ───────────────────────────
    if theta_init is None:
        print("No theta_init supplied — running MAP optimization...")
        theta_init = find_map_estimate(model, scalers, y_obs, seed=seed)

    # ── Scatter walkers around theta_init (Gaussian, a few % of prior width) -
    prior_width = PRIOR_BOUNDS[:, 1] - PRIOR_BOUNDS[:, 0]
    sigma       = init_noise_pct * prior_width           # per-parameter std
    p0          = theta_init + sigma * rng.standard_normal((n_walkers, N_DIM))
    p0          = np.clip(p0, PRIOR_BOUNDS[:, 0] + 1e-6, PRIOR_BOUNDS[:, 1] - 1e-6)

    # ── Sampler ─────────────────────────────────────────────────────────────
    sampler = emcee.EnsembleSampler(
        n_walkers, N_DIM, log_prob,
        args=(model, scalers, y_obs),
        moves=emcee.moves.StretchMove(a=1.5),  # default 2.0 → trop grand
    )

    print(f"Running burn-in ({burn_in} steps, {n_walkers} walkers)...")
    state = sampler.run_mcmc(p0, burn_in, progress=True)
    sampler.reset()

    print(f"Running production ({n_steps} steps)...")
    sampler.run_mcmc(state, n_steps, progress=True)

    flat_chain = sampler.get_chain(flat=True)
    log_post   = sampler.get_log_prob(flat=True)

    try:
        tau = sampler.get_autocorr_time(quiet=True)
        print(f"Autocorrelation times: {dict(zip(PARAM_NAMES, tau.round(1)))}")
    except emcee.autocorr.AutocorrError as e:
        print(f"Warning: autocorrelation estimate did not converge — {e}")

    mean_accept = np.mean(sampler.acceptance_fraction)
    print(f"Mean acceptance fraction: {mean_accept:.2%}")

    return {
        "chain":       sampler.get_chain(),
        "posterior":   flat_chain,
        "log_post":    log_post,
        "accept_rate": mean_accept,
        "sampler":     sampler,
    }

def run_inference_differentiable(
    model: Emulator21cm,
    theta: torch.Tensor,   # (6,) avec requires_grad=True
    scalers,
):
    """
    Same as run_inference but WITHOUT torch.no_grad() — keeps the
    computation graph intact so autograd can differentiate through theta.
    """
    if theta.dim() == 1:
        theta = theta.unsqueeze(0)

    ps_mean = torch.tensor(scalers["ps_mean"], dtype=theta.dtype)
    ps_std  = torch.tensor(scalers["ps_std"],  dtype=theta.dtype)

    # model.eval() but NO torch.no_grad()
    ps2d_mu, ps2d_sigma, xhi_mu = model(theta)

    ps_mu_log10    = ps2d_mu    * ps_std + ps_mean
    ps_sigma_log10 = ps2d_sigma * ps_std

    return ps_mu_log10, ps_sigma_log10


def neg_log_prob_with_grad(theta_np, model, scalers, y_obs_t):
    theta = torch.tensor(theta_np, dtype=torch.float32, requires_grad=True)

    ps_mu_log10, ps_sigma_log10 = run_inference_differentiable(model, theta, scalers)

    mu_nat    = ps_mu_log10.flatten()    * math.log(10)
    sigma_nat = ps_sigma_log10.flatten() * math.log(10)

    ll = (
        - torch.log(y_obs_t)
        - torch.log(sigma_nat)
        - 0.5 * ((torch.log(y_obs_t) - mu_nat) / sigma_nat) ** 2
    ).sum()

    (-ll).backward()

    return (-ll).item(), theta.grad.numpy().astype(np.float64)

def main():
    model          = load_emulator()
    checkpoint_dir = "emulator/sigma_model/checkpoints"
    scalers        = np.load(f"{checkpoint_dir}/scalers.npz")

    split       = torch.load("emulator/sigma_model/checkpoints/dataset_split.pt")
    test_thetas = split["test_thetas"]
    test_ps2d   = split["test_ps2d"]

    theta_obs        = test_thetas[2].numpy()
    test_ps2d_obs    = test_ps2d[2].numpy().flatten()

    print("True theta:", theta_obs)
    print("Log Likelihood at true theta:", log_prob(theta_obs, model, scalers, test_ps2d_obs))
    # MAP estimate is found automatically inside run_mcmc when theta_init=None
    
    results = run_mcmc(
        model, scalers, test_ps2d_obs,
        n_walkers=32,
        n_steps=20_000,
        burn_in=2_000,
        seed=412,
        init_noise_pct=0.02,   # ±2 % of prior width per parameter
    )

    np.savez(
        "emulator/sigma_model/mcmc_results.npz",
        chain=results["chain"],
        posterior=results["posterior"],
        log_post=results["log_post"],
        accept_rate=results["accept_rate"],
    )

if __name__ == "__main__":
    main()