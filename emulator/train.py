import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import Emulator21cm, mse_loss
def train(
    train_thetas : torch.Tensor,   # (N, 6)
    train_ps2d   : torch.Tensor,   # (N, 3, 10, 10)
    train_xhi    : torch.Tensor,   # (N, 3)
    epochs       : int   = 300,
    batch_size   : int   = 256,
    lr           : float = 1e-3,
    w_ps         : float = 1.0,
    w_xhi        : float = 1.0,
    checkpoint_dir: str  = "emulator/checkpoints",
) -> Emulator21cm:

    os.makedirs(checkpoint_dir, exist_ok=True)

    model     = Emulator21cm(n_params=6, n_redshifts=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loader = DataLoader(
        TensorDataset(train_thetas, train_ps2d, train_xhi),
        batch_size=batch_size,
        shuffle=True,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for theta_b, ps_b, xhi_b in loader:
            ps_pred, xhi_pred = model(theta_b)
            loss = mse_loss(ps_pred, ps_b, xhi_pred, xhi_b, w_ps, w_xhi)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        if epoch % 50 == 0:
            avg = epoch_loss / len(loader)
            print(f"epoch {epoch:>4}/{epochs}  loss={avg:.6f}")

    torch.save(model.state_dict(), f"{checkpoint_dir}/emulator.pt")
    print(f"Model saved → {checkpoint_dir}/emulator.pt")
    return model