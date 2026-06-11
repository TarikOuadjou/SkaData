# emulator/data_loader.py
import numpy as np
import os
import torch
from emulator.scale_theta import scale_thetas
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

def load_dataset(results_dir="results"):
    """
    Walk through results/point_XXXX/ folders and load all successful points.
    Returns tensors ready for training.
    """
    thetas, ps2d_list, xhi_list = [], [], []

    point_dirs = sorted([
        d for d in os.listdir(results_dir)
        if d.startswith("point_") and os.path.exists(os.path.join(results_dir, d, "done.flag"))
    ])
    point_dirs_failed = sorted([
        d for d in os.listdir(results_dir)
        if d.startswith("point_") and not os.path.exists(os.path.join(results_dir, d, "done.flag"))
    ])

    for d in point_dirs_failed:
        print(f"[Warning] Skipping {d}: done.flag not found")

    for d in point_dirs:
        point_dir = os.path.join(results_dir, d)
        try:
            theta  = np.load(os.path.join(point_dir, "theta.npy"),  allow_pickle=True).item()
            ps2d   = np.load(os.path.join(point_dir, "ps2d.npy"),   allow_pickle=True)
            xhi    = np.load(os.path.join(point_dir, "xHI.npy"),    allow_pickle=True)

            # theta is a dict → convert to ordered vector
            keys = ['ALPHA_STAR', 'F_STAR10', 'F_ESC10', 'ALPHA_ESC',
                    'M_TURN', 't_STAR']
            thetas.append([theta[k] for k in keys])
            ps2d_list.append(ps2d)
            xhi_list.append(xhi)

        except Exception as e:
            print(f"[Warning] Skipping {d}: {e}")

    last_loading_dir = os.path.join(results_dir, "last_loading")
    os.makedirs(last_loading_dir, exist_ok=True)

    np.save(os.path.join(last_loading_dir, "theta.npy"), np.array(thetas, dtype=np.float32))
    np.save(os.path.join(last_loading_dir, "ps2d_list.npy"), np.array(ps2d_list, dtype=np.float32))
    np.save(os.path.join(last_loading_dir, "xhi_list.npy"), np.array(xhi_list, dtype=np.float32))

    print(f"Loaded {len(thetas)} successful points from {results_dir}/")
    print(f"Saved last loaded arrays to {last_loading_dir}/")

    return (
        torch.tensor(scale_thetas(np.array(thetas)), dtype=torch.float32),  # (N, 6)
        torch.tensor(np.array(ps2d_list), dtype=torch.float32),  # (N, ...)
        torch.tensor(np.array(xhi_list),  dtype=torch.float32),  # (N, ...)
    )

if __name__ == "__main__":
    thetas, ps2d, xhi = load_dataset(results_dir="low_generate_data/results")
    print(f"thetas shape : {thetas.shape}")
    print(f"ps2d shape   : {ps2d.shape}")
    print(f"xhi shape    : {xhi.shape}")
    print(f"theta[0]     : {thetas[0]}")