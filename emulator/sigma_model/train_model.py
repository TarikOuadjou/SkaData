import os
import torch
from model_prob import train, run_inference
from emulator.data_loader import load_dataset
from evaluate import print_metrics, plot_pred_vs_true, plot_residuals, plot_ps2d_maps, plot_loss_history


def main():
    # ── Data ──────────────────────────────────────────────────────────────────
    dataset_path = "emulator/sigma_model_no_log/checkpoints/dataset_split.pt"

    print("Generating datasets...")
    if not os.path.exists(dataset_path):
        thetas, ps2d, xhi = load_dataset(results_dir="/gpfs/workdir/ouadjout/results")
        N = thetas.shape[0]
        idx = torch.randperm(N)
        thetas, ps2d, xhi = thetas[idx], ps2d[idx], xhi[idx]
        train_size = int(0.8 * N)

        train_thetas, test_thetas = thetas[:train_size], thetas[train_size:]
        train_ps2d,   test_ps2d   = ps2d[:train_size],   ps2d[train_size:]
        train_xhi,    test_xhi    = xhi[:train_size],    xhi[train_size:]

        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        torch.save({
            "train_thetas": train_thetas,
            "train_ps2d":   train_ps2d,
            "train_xhi":    train_xhi,
            "test_thetas":  test_thetas,
            "test_ps2d":    test_ps2d,
            "test_xhi":     test_xhi,
        }, dataset_path)
    else:
        data = torch.load(dataset_path)
        train_thetas, train_ps2d, train_xhi = data["train_thetas"], data["train_ps2d"], data["train_xhi"]
        test_thetas,  test_ps2d,  test_xhi  = data["test_thetas"],  data["test_ps2d"],  data["test_xhi"]

    print("\nTraining emulator...")
    model, history = train(
        train_thetas, train_ps2d, train_xhi,
        val_thetas=test_thetas,
        val_ps2d=test_ps2d,
        val_xhi=test_xhi,
        epochs=300, batch_size=256, lr=1e-3,
        w_ps=1.0, w_xhi=1.0,
        checkpoint_dir="emulator/sigma_model_no_log/checkpoints",
    )

    plot_loss_history(history)

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    ps2d_pred, xhi_pred, ps2d_sigma_pred = run_inference(
        model, test_thetas,
        checkpoint_dir="emulator/sigma_model_no_log/checkpoints"
    )

    ps2d_true       = test_ps2d.numpy()
    xhi_true        = test_xhi.numpy()
    ps2d_pred       = ps2d_pred.numpy()
    xhi_pred        = xhi_pred.numpy()
    ps2d_sigma_pred = ps2d_sigma_pred.numpy()

    print_metrics(ps2d_pred, ps2d_true, xhi_pred, xhi_true, ps2d_sigma=ps2d_sigma_pred)
    plot_pred_vs_true(ps2d_pred, ps2d_true, xhi_pred, xhi_true, out="emulator/sigma_model_no_log/metrics/pred_vs_true.png")
    plot_residuals   (ps2d_pred, ps2d_true, xhi_pred, xhi_true, out="emulator/sigma_model_no_log/metrics/residuals.png")
    plot_ps2d_maps   (ps2d_pred, ps2d_true, sample_idx=0,        out="emulator/sigma_model_no_log/metrics/ps2d_maps.png")

    print("\nDone.")


if __name__ == "__main__":
    main()