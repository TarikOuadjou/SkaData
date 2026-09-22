from scipy.optimize import minimize
import torch
import numpy as np
import emcee
import math
from emulator.sigma_model_no_log.model_prob import Emulator21cm, run_inference
from emulator.data_loader import PARAM_NAMES, N_Z, PARAM_RANGES

CHECKPOINT_DEFAULT = "checkpoints/emulator.pt"
PARAM_NAMES = ['ALPHA_STAR', 'F_STAR10', 'F_ESC10', 'ALPHA_ESC', 'M_TURN', 't_STAR']

PRIOR_BOUNDS = np.array([
    (0.0, 1.0)
    for _ in PARAM_NAMES
])

N_DIM = len(PARAM_NAMES)


def load_emulator():
    model = Emulator21cm(n_params=6, n_redshifts=N_Z)
    model.load_state_dict(torch.load('emulator/sigma_model_no_log/' + CHECKPOINT_DEFAULT, map_location="cpu"))
    return model


def log_prior(theta: np.ndarray) -> float:
    lo, hi = PRIOR_BOUNDS[:, 0], PRIOR_BOUNDS[:, 1]
    return 0.0 if np.all(theta >= lo) and np.all(theta <= hi) else -np.inf


def log_likelihood_gaussian(
    x_obs:  np.ndarray,
    mu:     np.ndarray,
    sigma:  np.ndarray,
) -> float:
    """
    Gaussian log-likelihood:
      ll = -0.5 * sum[ log(2π) + 2*log(σ) + ((x - μ)/σ)² ]
    """
    x_obs = np.asarray(x_obs, dtype=np.float64)
    mu    = np.asarray(mu,    dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    return float(np.sum(
        - 0.5 * np.log(2 * np.pi)
        #- np.log(sigma)
        - 0.5 * ((x_obs - mu) / sigma) ** 2
    ))


def log_prob(theta, model, scalers, y_obs):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ps2d_pred, _, ps2d_sigma_pred = run_inference(model, theta, scalers=scalers)
    ll = log_likelihood_gaussian(
        y_obs,
        ps2d_pred.numpy().flatten(),
        ps2d_sigma_pred.numpy().flatten(),
    )
    return lp + ll


def diagnose_map(model, scalers, y_obs, theta_true, theta_map):
    ps_true, _, ps_sig_true = run_inference(model, theta_true, scalers=scalers)
    ps_map,  _, ps_sig_map  = run_inference(model, theta_map,  scalers=scalers)

    ll_true = log_likelihood_gaussian(y_obs, ps_true.numpy().flatten(), ps_sig_true.numpy().flatten())
    ll_map  = log_likelihood_gaussian(y_obs, ps_map.numpy().flatten(),  ps_sig_map.numpy().flatten())

    print(f"ll at true theta : {ll_true:.2f}")
    print(f"ll at MAP theta  : {ll_map:.2f}")
    print(f"delta            : {ll_map - ll_true:.2f}  (should be > 0)")


# ── Differentiable forward pass (keeps grad graph for L-BFGS-B) ──────────────

def run_inference_differentiable(model, theta, scalers):
    """
    Same logic as run_inference but without torch.no_grad(),
    so autograd can differentiate through theta.
    """
    if theta.dim() == 1:
        theta = theta.unsqueeze(0)

    ps_mean = torch.tensor(scalers["ps_mean"], dtype=theta.dtype)
    ps_std  = torch.tensor(scalers["ps_std"],  dtype=theta.dtype)

    ps2d_mu, ps2d_sigma, _ = model(theta)

    ps2d_pred       = ps2d_mu    * ps_std + ps_mean
    ps2d_sigma_pred = ps2d_sigma * ps_std

    return ps2d_pred.flatten(), ps2d_sigma_pred.flatten()


def neg_log_prob_with_grad(theta_np, model, scalers, y_obs_t):
    """Returns (scalar, gradient) for scipy L-BFGS-B."""
    theta = torch.tensor(theta_np, dtype=torch.float32, requires_grad=True)

    mu, sigma = run_inference_differentiable(model, theta, scalers)

    ll = (
        - 0.5 * math.log(2 * math.pi)
        #- torch.log(sigma)
        - 0.5 * ((y_obs_t - mu) / sigma) ** 2
    ).sum()

    (-ll).backward()

    return (-ll).item(), theta.grad.numpy().astype(np.float64)


def neg_log_prob(theta, model, scalers, y_obs):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return np.inf
    ps2d_pred, _, ps2d_sigma_pred = run_inference(model, theta, scalers=scalers)
    ll = log_likelihood_gaussian(
        y_obs,
        ps2d_pred.numpy().flatten(),
        ps2d_sigma_pred.numpy().flatten(),
    )
    return -(lp + ll)


# ── MAP ───────────────────────────────────────────────────────────────────────

def find_map_estimate(model, scalers, y_obs, n_restarts=40, seed=42):
    rng    = np.random.default_rng(seed)
    bounds = list(zip(PRIOR_BOUNDS[:, 0], PRIOR_BOUNDS[:, 1]))
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


# ── MCMC ──────────────────────────────────────────────────────────────────────

def run_mcmc(
    model,
    scalers,
    y_obs,
    n_walkers=32,
    n_steps=5_000,
    burn_in=1_000,
    seed=42,
    theta_init=None,
    init_noise_pct=0.02,
):
    rng = np.random.default_rng(seed)

    if theta_init is None:
        print("No theta_init supplied — running MAP optimization...")
        theta_init = find_map_estimate(model, scalers, y_obs, seed=seed)

    prior_width = PRIOR_BOUNDS[:, 1] - PRIOR_BOUNDS[:, 0]
    sigma = init_noise_pct * prior_width
    p0 = theta_init + sigma * rng.standard_normal((n_walkers, N_DIM))
    p0 = np.clip(p0, PRIOR_BOUNDS[:, 0] + 1e-6, PRIOR_BOUNDS[:, 1] - 1e-6)

    sampler = emcee.EnsembleSampler(
        n_walkers, N_DIM, log_prob,
        args=(model, scalers, y_obs),
        moves=emcee.moves.StretchMove(a=1.5),
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

    print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.2%}")

    return {
        "chain":       sampler.get_chain(),
        "posterior":   flat_chain,
        "log_post":    log_post,
        "accept_rate": np.mean(sampler.acceptance_fraction),
        "sampler":     sampler,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    model          = load_emulator()
    checkpoint_dir = "emulator/sigma_model_no_log/checkpoints"
    scalers        = np.load(f"{checkpoint_dir}/scalers.npz")
    split       = torch.load(f"{checkpoint_dir}/dataset_split.pt")
    test_thetas = split["test_thetas"]
    test_ps2d   = split["test_ps2d"]
    theta_obs     = np.array([0.86666667, 0.73333333, 0.575     , 0.46666666, 0.        ,0.30000001])
    y_obs, xhi_cur, ps2d_sigma_cur = run_inference(model, theta_obs, scalers=scalers)
    y_obs = y_obs.numpy().flatten()
    print(xhi_cur.numpy())
    print("True theta:", theta_obs)
    print("Log-likelihood at true theta:", log_prob(theta_obs, model, scalers, y_obs))
    results = run_mcmc(
        model, scalers, y_obs,
        n_walkers=8,
       n_steps=10_000,
        burn_in=1_000,
        seed=412,
        init_noise_pct=0.1,
    )

    np.savez(
        "emulator/sigma_model_no_log/mcmc_results.npz",
        chain=results["chain"],
        posterior=results["posterior"],
        log_post=results["log_post"],
        accept_rate=results["accept_rate"],
    )


if __name__ == "__main__":
    main()