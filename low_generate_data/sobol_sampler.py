# sobol_manager.py
import numpy as np
from scipy.stats import qmc
import os

prior_dic = {
    'ALPHA_STAR':       (-0.5, 1.0),
    'F_STAR10':         (-3.0, 0.0),
    'F_ESC10':          (-3.0, 1.0),
    'ALPHA_ESC':        (-1.0, 0.5),
    'M_TURN':           (8.0, 10.0),
    't_STAR':           (0.0,  1.0),
}

KEYS = list(prior_dic.keys())
BOUNDS = np.array([prior_dic[k] for k in KEYS])
SAMPLES_FILE = "low_generate_data/sobol_samples.npy"
STATE_FILE = "low_generate_data/sobol_state.npy"  # just a single integer
SEED = 42


def add_points(n: int) -> list[int]:
    """Draw n new points continuing from where we left off."""
    current = int(np.load(STATE_FILE)) if os.path.exists(STATE_FILE) else 0
    samples = np.load(SAMPLES_FILE, allow_pickle=True).tolist() if os.path.exists(SAMPLES_FILE) else []

    # Rebuild sampler and skip already-drawn points
    sampler = qmc.Sobol(d=len(KEYS), scramble=True, seed=SEED)
    if current > 0:
        sampler.fast_forward(current)

    unit = sampler.random(n)
    scaled = qmc.scale(unit, BOUNDS[:, 0], BOUNDS[:, 1])
    new_points = [dict(zip(KEYS, row)) for row in scaled]

    samples.extend(new_points)
    np.save(SAMPLES_FILE, samples)
    np.save(STATE_FILE, current + n)

    new_ids = list(range(current, current + n))
    print(f"Added {n} points → job IDs {current}–{current + n - 1} (total: {current + n})")
    return new_ids


if __name__ == "__main__":
    import sys
    new_ids = add_points(int(sys.argv[1]))
    print(",".join(map(str, new_ids)))