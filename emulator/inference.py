import torch
from train      import train
from data_loader import load_dataset
from evaluate  import print_metrics, plot_pred_vs_true, plot_residuals, run_inference, plot_ps2d_maps

def main():
    # ── Data ──────────────────────────────────────────────────────────────────
    print("Generating datasets...")
    thetas, ps2d, xhi = load_dataset(results_dir="low_generate_data/results")
    
    N = thetas.shape[0]
    train_size = int(0.8 * N)
    train_thetas, test_thetas = thetas[:train_size], thetas[train_size:]
    train_ps2d,   test_ps2d   = ps2d[:train_size], ps2d[train_size:]
    train_xhi,    test_xhi    = xhi[:train_size],  xhi[train_size:]

    # ── Training ──────────────────────────────────────────────────────────────
    print("\nTraining emulator...")
    model = train(
        train_thetas, train_ps2d, train_xhi,
        epochs=300, batch_size=256, lr=1e-3,
        w_ps=1.0, w_xhi=1.0,
        checkpoint_dir="emulator/checkpoints",
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    ps2d_pred, xhi_pred = run_inference(model, test_thetas)
    ps2d_true = test_ps2d.numpy()
    xhi_true  = test_xhi.numpy()

    print_metrics(ps2d_pred, ps2d_true, xhi_pred, xhi_true)
    plot_pred_vs_true(ps2d_pred, ps2d_true, xhi_pred, xhi_true)
    plot_residuals   (ps2d_pred, ps2d_true, xhi_pred, xhi_true)
    plot_ps2d_maps   (ps2d_pred, ps2d_true, sample_idx=0)

    print("\nDone.")

if __name__ == "__main__":
    main()