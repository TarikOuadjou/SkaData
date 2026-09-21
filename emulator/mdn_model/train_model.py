import torch
import numpy as np
from model_mdn import train, run_inference
from emulator.data_loader import load_dataset
from evaluate import print_metrics, plot_pred_vs_true, plot_residuals, plot_ps2d_maps, plot_loss_history

CHECKPOINT_DIR = "emulator/mdn_model/checkpoints"
METRICS_DIR    = "emulator/mdn_model/metrics"
DATASET_PATH   = f"{CHECKPOINT_DIR}/dataset_split.pt"


def main():
    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading dataset...")
    try:
        data = torch.load(DATASET_PATH)
        train_thetas = data["train_thetas"]
        train_ps2d   = data["train_ps2d"]
        train_xhi    = data["train_xhi"]
        test_thetas  = data["test_thetas"]
        test_ps2d    = data["test_ps2d"]
        test_xhi     = data["test_xhi"]
        print(f"Loaded cached split from {DATASET_PATH}")
    except FileNotFoundError:
        print("No cached split found, generating dataset...")
        import os
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        thetas, ps2d, xhi = load_dataset(results_dir="low_generate_data/results")
        N   = thetas.shape[0]
        idx = torch.randperm(N)
        thetas, ps2d, xhi = thetas[idx], ps2d[idx], xhi[idx]

        train_size   = int(0.8 * N)
        train_thetas = thetas[:train_size]
        test_thetas  = thetas[train_size:]
        train_ps2d   = ps2d[:train_size]
        test_ps2d    = ps2d[train_size:]
        train_xhi    = xhi[:train_size]
        test_xhi     = xhi[train_size:]

        torch.save({
            "train_thetas": train_thetas,
            "train_ps2d":   train_ps2d,
            "train_xhi":    train_xhi,
            "test_thetas":  test_thetas,
            "test_ps2d":    test_ps2d,
            "test_xhi":     test_xhi,
        }, DATASET_PATH)
        print(f"Dataset split saved → {DATASET_PATH}")

    # ── Training ──────────────────────────────────────────────────────────────
    print("\nTraining MDN emulator (K=3)...")
    model, history = train(
        train_thetas, train_ps2d, train_xhi,
        val_thetas=test_thetas,
        val_ps2d=test_ps2d,
        val_xhi=test_xhi,
        epochs=1500, batch_size=256, lr=1e-3,
        w_ps=1.0, w_xhi=1.0,
        K=3,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    plot_loss_history(history)

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    results = run_inference(model, test_thetas, checkpoint_dir=CHECKPOINT_DIR)

    ps2d_pred  = results["ps2d_mean"].numpy()     # (N, Z, H, W)  mixture mean
    ps2d_sigma = results["ps2d_std"].numpy()      # (N, Z, H, W)  mixture std
    xhi_pred   = results["xhi_mu"].numpy()        # (N, Z)
    ps2d_mu = results["ps2d_mu"].numpy()          # (N, Z, K, H, W) mixture component means
    ps2d_true  = test_ps2d.numpy()
    xhi_true   = test_xhi.numpy()

    # Diagnostic: check for mode collapse (mean weight per component, all pixels)
    ps2d_pi = results["ps2d_pi"]                       # (N, Z, K, H, W)
    mean_pi = ps2d_pi.mean(dim=(0, 1, 3, 4))           # (K,)
    
    print(f"\nMixture weights (mean over batch/pixels): {mean_pi.numpy().round(3)}")
    print("  → If one weight ≈ 1.0 and others ≈ 0, mode collapse detected.\n")

    import os
    ps2d_pi     = ps2d_pi.numpy()                       # (N, Z, K, H, W)
    mu_mix     = ps2d_pred[:, :, np.newaxis, :, :]   # (N, Z, 1, H, W)
    ps2d_sigma_mix  = np.sqrt(
        (ps2d_pi * (ps2d_sigma**2 + (ps2d_mu - mu_mix)**2)).sum(axis=2)
    )     
    os.makedirs(METRICS_DIR, exist_ok=True)

    
    plot_pred_vs_true(
        ps2d_pred, ps2d_true, xhi_pred, xhi_true,
        out=f"{METRICS_DIR}/pred_vs_true.png",
    )
    plot_residuals(
        ps2d_pred, ps2d_true, xhi_pred, xhi_true,
        out=f"{METRICS_DIR}/residuals.png",
    )
    plot_ps2d_maps(
        ps2d_pred, ps2d_true, sample_idx=0,
        out=f"{METRICS_DIR}/ps2d_maps.png",
    )
    print_metrics(
            ps2d_pred, ps2d_true,
            xhi_pred,  xhi_true,
            ps2d_sigma=ps2d_sigma_mix,
        )
    print("\nDone.")


if __name__ == "__main__":
    main()