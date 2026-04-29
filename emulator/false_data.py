import numpy as np
import torch

# ── Parameter space ──────────────────────────────────────────────────────────

PARAM_NAMES  = ["ALPHA_STAR", "F_STAR10", "F_ESC10",
                "ALPHA_ESC",  "M_TURN",   "t_STAR"]

PARAM_RANGES = [
    (-0.5,  1.0),   # ALPHA_STAR  — stellar mass – UV slope
    (-3.0,  0.0),   # F_STAR10   — log10 star formation efficiency
    (-3.0,  1.0),   # F_ESC10    — log10 escape fraction
    (-1.0,  0.5),   # ALPHA_ESC  — escape fraction slope
    ( 8.0, 10.0),   # M_TURN     — log10 turnover mass [M_sun]
    ( 0.0,  1.0),   # t_STAR     — star formation time-scale
]

# Each redshift bin is defined by its bounds [z_high, z_low]
Z_BOUNDS = [8.41, 7.56, 6.85, 6.25]
Z_BINS   = [(Z_BOUNDS[i], Z_BOUNDS[i+1]) for i in range(len(Z_BOUNDS) - 1)]
# Z_BINS = [(8.41, 7.56), (7.56, 6.85), (6.85, 6.25)]
Z_MIDS   = [(a + b) / 2 for a, b in Z_BINS]
# Z_MIDS = [7.985, 7.205, 6.55]
N_Z      = len(Z_BINS)   # 3
N_PARAMS  = len(PARAM_NAMES)

LO = np.array([r[0] for r in PARAM_RANGES])   # shape (6,)
HI = np.array([r[1] for r in PARAM_RANGES])   # shape (6,)


# ── Sampling ─────────────────────────────────────────────────────────────────

def sample_latin_hypercube(n: int, seed: int = 0) -> np.ndarray:
    """
    Latin Hypercube Sampling in [0, 1]^N_PARAMS.
    Returns unit_cube of shape (n, N_PARAMS).
    """
    rng    = np.random.default_rng(seed)
    result = np.zeros((n, N_PARAMS))
    for i in range(N_PARAMS):
        cuts        = np.linspace(0.0, 1.0, n + 1)
        u           = rng.uniform(cuts[:-1], cuts[1:])
        result[:, i] = rng.permutation(u)
    return result


def unit_to_physical(unit_cube: np.ndarray) -> np.ndarray:
    """[0, 1]^6  →  physical parameter space."""
    return LO + unit_cube * (HI - LO)


def physical_to_unit(thetas_phys: np.ndarray) -> np.ndarray:
    """Physical parameter space  →  [0, 1]^6."""
    return (thetas_phys - LO) / (HI - LO)


# ── Toy simulators ───────────────────────────────────────────────────────────

def simulate_ps2d(unit_cube: np.ndarray) -> np.ndarray:
    """
    Toy 21cm power spectrum simulator.

    Input  : unit_cube  (N, 6)  normalised parameters in [0, 1]
    Output : ps2d       (N, N_Z, 10, 10)  [mK^2 Mpc^3]

    Physics couplings (approximate):
      - amplitude  ∝ F_ESC10 (idx 2) × F_STAR10 (idx 1)
      - slope      driven by ALPHA_STAR (idx 0) and ALPHA_ESC (idx 3)
      - suppression at high M_TURN (idx 4) — fewer small haloes
      - redshift evolution: power fades as reionisation progresses
    """
    rng = np.random.default_rng(42)
    N   = unit_cube.shape[0]

    k_par  = np.linspace(0.1, 1.0, 10)
    k_perp = np.linspace(0.1, 1.0, 10)
    KP, KV = np.meshgrid(k_perp, k_par)       # (10, 10)
    K       = np.sqrt(KP**2 + KV**2)           # total k

    alpha_star = unit_cube[:, 0][:, None, None]
    f_star10   = unit_cube[:, 1][:, None, None]
    f_esc10    = unit_cube[:, 2][:, None, None]
    alpha_esc  = unit_cube[:, 3][:, None, None]
    m_turn     = unit_cube[:, 4][:, None, None]

    ps2d = np.zeros((N, N_Z, 10, 10), dtype=np.float32)

    # z_decay : higher redshift → IGM more neutral → stronger 21cm signal
    z_decays = [(z - Z_MIDS[-1]) / (Z_MIDS[0] - Z_MIDS[-1]) * 0.75 + 0.25
                for z in Z_MIDS]
    # z_decays ≈ [1.0, 0.625, 0.25] for Z_MIDS = [7.985, 7.205, 6.55]

    for zi, z_decay in enumerate(z_decays):
        amplitude = (
            (1.0 + f_esc10)
            * (1.0 + 0.5 * f_star10)
            * (1.0 - 0.3 * m_turn)   # high M_TURN → fewer sources → less power
            * z_decay
        )
        slope = 2.5 + 0.4 * alpha_star - 0.3 * alpha_esc

        signal = amplitude * K[None] ** (-slope)
        noise  = 0.03 * rng.standard_normal((N, 10, 10))
        ps2d[:, zi] = (signal + noise).astype(np.float32)

    return ps2d


def simulate_xhi(unit_cube: np.ndarray) -> np.ndarray:
    """
    Toy mean neutral fraction simulator.

    Input  : unit_cube  (N, 6)  normalised parameters in [0, 1]
    Output : xhi        (N, N_Z)  in [0, 1]

    Physics couplings:
      - high F_ESC10  → faster reionisation → lower x_HI
      - high F_STAR10 → more photons        → lower x_HI
      - high M_TURN   → fewer sources       → higher x_HI
      - earlier redshifts (higher z index)  → more neutral
    """
    rng = np.random.default_rng(99)
    N   = unit_cube.shape[0]

    f_esc10  = unit_cube[:, 2]
    f_star10 = unit_cube[:, 1]
    m_turn   = unit_cube[:, 4]

    # Effective ionisation efficiency in [0, 1]
    ion_eff = 0.5 * f_esc10 + 0.3 * f_star10 - 0.2 * m_turn

    xhi = np.zeros((N, N_Z), dtype=np.float32)

    # baseline x_HI : higher z → more neutral → higher baseline
    # mapped linearly from Z_MIDS[0] (most neutral) to Z_MIDS[-1] (least neutral)
    baselines = [(z - Z_MIDS[-1]) / (Z_MIDS[0] - Z_MIDS[-1]) * 0.65 + 0.20
                 for z in Z_MIDS]
    # baselines ≈ [0.85, 0.525, 0.20] for Z_MIDS = [7.985, 7.205, 6.55]

    for zi, baseline in enumerate(baselines):
        raw     = baseline - 0.5 * ion_eff + 0.03 * rng.standard_normal(N)
        xhi[:, zi] = np.clip(raw, 0.0, 1.0).astype(np.float32)

    return xhi


# ── Dataset builder ──────────────────────────────────────────────────────────

def make_dataset(n: int, seed: int = 0):
    """
    Generate a synthetic dataset of size n.

    Returns
    -------
    thetas  : torch.Tensor  (n, 6)       normalised parameters in [0, 1]
    ps2d    : torch.Tensor  (n, 3, 10, 10)
    xhi     : torch.Tensor  (n, 3)
    thetas_phys : np.ndarray (n, 6)      physical parameter values (for reference)
    """
    unit_cube   = sample_latin_hypercube(n, seed=seed)   # (n, 6) in [0,1]
    thetas_phys = unit_to_physical(unit_cube)            # (n, 6) physical

    ps2d = simulate_ps2d(unit_cube)   # (n, 3, 10, 10)
    xhi  = simulate_xhi(unit_cube)    # (n, 3)

    return (
        torch.from_numpy(unit_cube.astype(np.float32)),
        torch.from_numpy(ps2d),
        torch.from_numpy(xhi),
        thetas_phys,
    )


# ── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    N_TRAIN, N_VAL = 20000, 400

    print("Generating training set...")
    train_thetas, train_ps2d, train_xhi, train_phys = make_dataset(N_TRAIN, seed=0)

    print("Generating validation set...")
    val_thetas, val_ps2d, val_xhi, val_phys = make_dataset(N_VAL, seed=1)

    print(f"  Redshift bins  : {Z_BINS}")
    print(f"  Bin midpoints  : {[round(z, 3) for z in Z_MIDS]}")
    print()
    print(f"  train thetas : {train_thetas.shape}   min={train_thetas.min():.3f} max={train_thetas.max():.3f}")
    print(f"  train ps2d   : {train_ps2d.shape}")
    print(f"  train xhi    : {train_xhi.shape}    min={train_xhi.min():.3f} max={train_xhi.max():.3f}")
    print()
    print(f"  val thetas   : {val_thetas.shape}")
    print(f"  val ps2d     : {val_ps2d.shape}")
    print(f"  val xhi      : {val_xhi.shape}")
    print()
    print("Physical parameter ranges in training set:")
    for i, name in enumerate(PARAM_NAMES):
        lo_obs = train_phys[:, i].min()
        hi_obs = train_phys[:, i].max()
        print(f"  {name:<12}  [{lo_obs:+.3f}, {hi_obs:+.3f}]   prior [{PARAM_RANGES[i][0]}, {PARAM_RANGES[i][1]}]")
