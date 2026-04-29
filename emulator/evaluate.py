"""
evaluate.py
-----------
Loads a trained Emulator21cm and produces:
  1. Scalar metrics  — RMSE, MAE, R² for PS2D and xHI (per redshift bin)
  2. Predicted vs true scatter plots  (one per z-bin, PS2D + xHI side by side)
  3. Residual distributions           (histogram of (pred - true) / true)
  4. One PS2D example map             (true / predicted / residual 2D grid)

Usage
-----
    python evaluate.py                  # generates test set on the fly
    python evaluate.py --checkpoint checkpoints/emulator.pt
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from model      import Emulator21cm
from false_data import make_dataset, PARAM_NAMES, Z_BINS, Z_MIDS, N_Z

N_TEST = 1_000
CHECKPOINT_DEFAULT = "checkpoints/emulator.pt"


# ── Metrics ──────────────────────────────────────────────────────────────────

def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))

def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))

def r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

def relative_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return (pred - true) / (np.abs(true) + 1e-12)


# ── Inference ────────────────────────────────────────────────────────────────

def run_inference(model: Emulator21cm, thetas: torch.Tensor):
    model.eval()
    with torch.no_grad():
        ps2d_pred, xhi_pred = model(thetas)
    return ps2d_pred.numpy(), xhi_pred.numpy()


# ── Print metrics table ───────────────────────────────────────────────────────

def print_metrics(ps2d_pred, ps2d_true, xhi_pred, xhi_true):
    z_labels = [f"z=[{a:.2f},{b:.2f}]" for a, b in Z_BINS]

    print("\n── PS2D metrics (flattened 10×10 per sample) ─────────────────────")
    print(f"  {'bin':<18}  {'RMSE':>10}  {'MAE':>10}  {'R²':>8} {'MAE (%)':>10}")
    for zi, lbl in enumerate(z_labels):
        p = ps2d_pred[:, zi].flatten()
        t = ps2d_true[:, zi].flatten()
        print(f"  {lbl:<18}  {rmse(p,t):>10.5f}  {mae(p,t):>10.5f}  {r2(p,t):>8.4f}  {np.mean(relative_error(p,t))*100:>8.4f}%")

    print("\n── xHI metrics ───────────────────────────────────────────────────")
    print(f"  {'bin':<18}  {'RMSE':>10}  {'MAE':>10}  {'R²':>8} {'MAE (%)':>10}")
    for zi, lbl in enumerate(z_labels):
        p = xhi_pred[:, zi]
        t = xhi_true[:, zi]
        print(f"  {lbl:<18}  {rmse(p,t):>10.5f}  {mae(p,t):>10.5f}  {r2(p,t):>8.4f}  {np.mean(relative_error(p,t))*100:>8.4f}%")


# ── Plot 1 : predicted vs true scatter ───────────────────────────────────────

def plot_pred_vs_true(ps2d_pred, ps2d_true, xhi_pred, xhi_true, out="eval_scatter.png"):
    fig, axes = plt.subplots(2, N_Z, figsize=(5 * N_Z, 9))
    z_labels  = [f"z = [{a:.2f}, {b:.2f}]" for a, b in Z_BINS]

    for zi in range(N_Z):
        # ── PS2D ──
        ax = axes[0, zi]
        p  = ps2d_pred[:, zi].flatten()
        t  = ps2d_true[:, zi].flatten()
        vmin, vmax = min(t.min(), p.min()), max(t.max(), p.max())
        ax.scatter(t, p, s=1, alpha=0.3, rasterized=True, color="steelblue")
        ax.plot([vmin, vmax], [vmin, vmax], "r--", lw=1, label="1:1")
        ax.set_xlabel("True PS2D")
        ax.set_ylabel("Predicted PS2D")
        ax.set_title(f"PS2D  {z_labels[zi]}\nR²={r2(p,t):.4f}")
        ax.legend(fontsize=8)

        # ── xHI ──
        ax = axes[1, zi]
        p  = xhi_pred[:, zi]
        t  = xhi_true[:, zi]
        ax.scatter(t, p, s=4, alpha=0.5, rasterized=True, color="darkorange")
        ax.plot([0, 1], [0, 1], "r--", lw=1, label="1:1")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("True x_HI")
        ax.set_ylabel("Predicted x_HI")
        ax.set_title(f"x_HI  {z_labels[zi]}\nR²={r2(p,t):.4f}")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Plot 2 : residual distributions ──────────────────────────────────────────

def plot_residuals(ps2d_pred, ps2d_true, xhi_pred, xhi_true, out="eval_residuals.png"):
    fig, axes = plt.subplots(2, N_Z, figsize=(5 * N_Z, 8))
    z_labels  = [f"z = [{a:.2f}, {b:.2f}]" for a, b in Z_BINS]

    for zi in range(N_Z):
        # ── PS2D relative error ──
        ax  = axes[0, zi]
        err = relative_error(ps2d_pred[:, zi].flatten(),
                             ps2d_true[:, zi].flatten())
        ax.hist(err, bins=60, color="steelblue", edgecolor="none", alpha=0.8)
        ax.axvline(0, color="red", lw=1.5, ls="--")
        ax.set_xlabel("(pred − true) / |true|")
        ax.set_ylabel("Count")
        ax.set_title(f"PS2D residuals  {z_labels[zi]}\nmedian={np.median(err)*100:.2f}%")

        # ── xHI absolute error ──
        ax  = axes[1, zi]
        err = xhi_pred[:, zi] - xhi_true[:, zi]
        ax.hist(err, bins=40, color="darkorange", edgecolor="none", alpha=0.8)
        ax.axvline(0, color="red", lw=1.5, ls="--")
        ax.set_xlabel("pred − true")
        ax.set_ylabel("Count")
        ax.set_title(f"x_HI residuals  {z_labels[zi]}\nMAE={mae(xhi_pred[:,zi], xhi_true[:,zi]):.4f}")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Plot 3 : PS2D example map (true / pred / residual) ────────────────────────

def plot_ps2d_maps(ps2d_pred, ps2d_true, sample_idx=0, out="eval_ps2d_maps.png"):
    """
    For a single test sample, show the 10×10 PS2D map at each redshift:
    true | predicted | (pred-true)/true
    """
    fig = plt.figure(figsize=(4 * N_Z, 9))
    gs  = gridspec.GridSpec(3, N_Z, figure=fig, hspace=0.45, wspace=0.35)
    z_labels = [f"z=[{a:.2f},{b:.2f}]" for a, b in Z_BINS]
    row_titles = ["True", "Predicted", "Relative error"]

    for zi in range(N_Z):
        true_map = ps2d_true[sample_idx, zi]          # (10, 10)
        pred_map = ps2d_pred[sample_idx, zi]
        rel_map  = (pred_map - true_map) / (np.abs(true_map) + 1e-12)

        maps   = [true_map, pred_map, rel_map]
        cmaps  = ["viridis", "viridis", "RdBu_r"]
        shared = dict(vmin=true_map.min(), vmax=true_map.max())

        for row, (m, cmap) in enumerate(zip(maps, cmaps)):
            ax  = fig.add_subplot(gs[row, zi])
            kw  = shared if row < 2 else dict(vmin=-0.2, vmax=0.2)
            im  = ax.imshow(m, origin="lower", aspect="auto", cmap=cmap, **kw)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlabel("k⊥ bin")
            ax.set_ylabel("k∥ bin")
            ax.set_title(f"{row_titles[row]}\n{z_labels[zi]}", fontsize=9)

    fig.suptitle(f"PS2D maps — test sample #{sample_idx}", y=1.01)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT)
    parser.add_argument("--n_test",     type=int, default=N_TEST)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    # ── Load model ──
    print(f"Loading checkpoint: {args.checkpoint}")
    model = Emulator21cm(n_params=6, n_redshifts=N_Z)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    # ── Generate test set (unseen seed) ──
    print(f"Generating {args.n_test} test samples (seed={args.seed})...")
    test_thetas, test_ps2d, test_xhi, _ = make_dataset(args.n_test, seed=args.seed)

    # ── Run inference ──
    ps2d_pred, xhi_pred = run_inference(model, test_thetas)
    ps2d_true = test_ps2d.numpy()
    xhi_true  = test_xhi.numpy()

    # ── Metrics ──
    print_metrics(ps2d_pred, ps2d_true, xhi_pred, xhi_true)

    # ── Plots ──
    print("Generating plots...")
    plot_pred_vs_true(ps2d_pred, ps2d_true, xhi_pred, xhi_true)
    plot_residuals   (ps2d_pred, ps2d_true, xhi_pred, xhi_true)
    plot_ps2d_maps   (ps2d_pred, ps2d_true, sample_idx=0)

    print("\nDone.")


if __name__ == "__main__":
    main()