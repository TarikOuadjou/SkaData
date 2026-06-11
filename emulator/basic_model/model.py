import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import glob

class Emulator21cm(nn.Module):
    """
    Deterministic emulator for 21cm observables.

    Input  : theta  (B, 6)  — 6 astrophysical parameters
    Outputs:
        ps2d : (B, 3, 10, 10)  — 2-D power spectra at 3 redshifts
        xhi  : (B, 3)          — mean neutral fractions at 3 redshifts
    """

    N_REDSHIFTS = 3
    PS_GRID     = 10  

    def __init__(self, n_params: int = 6, n_redshifts: int = N_REDSHIFTS):
        super().__init__()
        self.n_redshifts = n_redshifts

        # ── Shared encoder ───────────────────────────────────────────────────
        self.shared = nn.Sequential(
            nn.Linear(n_params, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 512),      nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256),      nn.LayerNorm(256), nn.GELU(),
        )

        # ── PS2D decoder ─────────────────────────────────────────────────────
        self.ps2d_fc = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, n_redshifts * 16 * 5 * 5),
        )
        self.ps2d_cnn = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16,  1, kernel_size=3, padding=1),
        ) 

        # ── xHI head ─────────────────────────────────────────────────────────
        self.xhi_head = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64,  32), nn.GELU(),
            nn.Linear(32, n_redshifts),
            nn.Sigmoid(),  
        )

    def forward(self, theta: torch.Tensor):
        """
        theta : (B, 6)
        returns
            ps2d : (B, 3, 10, 10)
            xhi  : (B, 3)
        """
        B = theta.size(0)
        h = self.shared(theta)                                  # (B, 256)
        feat = self.ps2d_fc(h)                                  # (B, Z·16·5·5)
        feat = feat.view(B * self.n_redshifts, 16, 5, 5)       # (B·Z, 16, 5, 5)
        ps2d = self.ps2d_cnn(feat).squeeze(1)                   # (B·Z, 10, 10)
        ps2d = ps2d.view(B, self.n_redshifts, 10, 10)          # (B, Z, 10, 10)
        xhi = self.xhi_head(h)                                  # (B, 3)

        return ps2d, xhi


def mse_loss(ps2d_pred, ps2d_target, xhi_pred, xhi_target,
             w_ps: float = 1.0, w_xhi: float = 1.0) -> torch.Tensor:
    loss_ps = F.mse_loss(ps2d_pred, ps2d_target)
    loss_xhi = F.mse_loss(xhi_pred,  xhi_target)
    return w_ps * loss_ps + w_xhi * loss_xhi

def mse_loss_variance(ps2d_pred, ps2d_target, xhi_pred, xhi_target,
             w_ps: float = 1.0, w_xhi: float = 1.0, sigma: np.ndarray = None, scalers =None) -> torch.Tensor:
    if sigma is None:
        sigma = np.concatenate([
            np.loadtxt(f).flatten()
            for f in sorted(glob.glob("PS1_PS2_Data/err_Pk_PS1_*.txt"))
        ])
    if scalers is not None:
        ps_mean  = torch.tensor(scalers["ps_mean"])
        ps_std   = torch.tensor(scalers["ps_std"])
    pred_phys   = 10**((ps2d_pred * ps_std) + ps_mean)
    target_phys = 10**((ps2d_target * ps_std) + ps_mean)
    sigma_tensor = torch.tensor(sigma, dtype=torch.float32).view(1, 3, 10, 10)
    loss_ps = torch.mean(((pred_phys - target_phys) / (sigma_tensor )) ** 2)    
    #loss_xhi = F.mse_loss(xhi_pred, xhi_target)
    return w_ps * loss_ps #+ w_xhi * loss_xhi

def compute_scalers(
    ps2d: torch.Tensor,   # (N, 3, 10, 10)
):
    log_ps = torch.log10(ps2d + 1e-30)                   # (N, 3, 10, 10)
    ps_mean = log_ps.mean(dim=0, keepdim=True)           # (1, 3, 10, 10)
    ps_std  = log_ps.std(dim=0, keepdim=True).clamp(min=1e-8)

    return ps_mean, ps_std


def scale_ps(ps2d, ps_mean, ps_std):
    return (torch.log10(ps2d + 1e-30) - ps_mean) / ps_std # maybe transform it to 0,1

def train(
    train_thetas : torch.Tensor,
    train_ps2d   : torch.Tensor,
    train_xhi    : torch.Tensor,
    val_thetas   : torch.Tensor,        # ← NEW
    val_ps2d     : torch.Tensor,        # ← NEW
    val_xhi      : torch.Tensor,        # ← NEW
    epochs       : int   = 300,
    batch_size   : int   = 256,
    lr           : float = 1e-3,
    w_ps         : float = 1.0,
    w_xhi        : float = 1.0,
    checkpoint_dir: str  = "emulator/checkpoints",
) -> tuple[Emulator21cm, dict]:         # ← returns history too

    os.makedirs(checkpoint_dir, exist_ok=True)

    ps_mean, ps_std = compute_scalers(train_ps2d)

    train_ps2d_scaled = scale_ps(train_ps2d, ps_mean, ps_std)
    # Scale val set with TRAIN scalers 
    val_ps2d_scaled = scale_ps(val_ps2d, ps_mean, ps_std)

    scaler_path = f"{checkpoint_dir}/scalers.npz"
    np.savez(
        scaler_path,
        ps_mean=ps_mean.numpy(), ps_std=ps_std.numpy(),
    )
    print(f"Scalers saved → {scaler_path}")

    model     = Emulator21cm(n_params=6, n_redshifts=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loader = DataLoader(
        TensorDataset(train_thetas, train_ps2d_scaled, train_xhi),
        batch_size=batch_size, shuffle=True,
    )

    history = {"train_loss": [], "val_loss": []}  
    sigma = np.concatenate([
                np.loadtxt(f).flatten()
                for f in sorted(glob.glob("PS1_PS2_Data/err_Pk_PS1_*.txt"))
            ])
    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        for theta_b, ps_b, xhi_b in loader:
            ps_pred, xhi_pred = model(theta_b)
            loss = mse_loss(ps_pred, ps_b, xhi_pred, xhi_b, w_ps, w_xhi)
            #loss = mse_loss_variance(ps_pred, ps_b, xhi_pred, xhi_b, w_ps, w_xhi, sigma=sigma, scalers={"ps_mean": ps_mean, "ps_std": ps_std})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(loader)

        # ── Validation ──
        model.eval()
        with torch.no_grad():
            ps_val_pred, xhi_val_pred = model(val_thetas)
            val_loss = mse_loss(
                ps_val_pred, val_ps2d_scaled,
                xhi_val_pred, val_xhi,
                w_ps, w_xhi,
            ).item()
            #val_loss = mse_loss_variance(
            #    ps_val_pred, val_ps2d_scaled,
            #    xhi_val_pred, val_xhi,
            #    w_ps, w_xhi, sigma=sigma, scalers={"ps_mean": ps_mean, "ps_std": ps_std}
            #).item()

        history["train_loss"].append(avg_train)     
        history["val_loss"].append(val_loss)        

        scheduler.step()

        if epoch % 50 == 0:
            print(f"epoch {epoch:>4}/{epochs}  train={avg_train:.6f}  val={val_loss:.6f}")

    torch.save(model.state_dict(), f"{checkpoint_dir}/emulator.pt")
    print(f"Model saved → {checkpoint_dir}/emulator.pt")
    return model, history    #

def run_inference(
    model: Emulator21cm,
    theta: torch.Tensor,        # (N, 6) or (6,)
    checkpoint_dir: str = "emulator/basic_model/checkpoints",
    scalers = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scalers is None:
        scalers  = np.load(f"{checkpoint_dir}/scalers.npz")

    if isinstance(theta, np.ndarray):
        theta = torch.tensor(theta, dtype=torch.float32)
    if theta.dim() == 1:
        theta = theta.unsqueeze(0)   
    
    ps_mean  = torch.tensor(scalers["ps_mean"])  
    ps_std   = torch.tensor(scalers["ps_std"])    
    model.eval()
    with torch.no_grad():
        ps_pred_scaled, xhi_pred = model(theta)
        ps_pred  = 10 ** (ps_pred_scaled * ps_std + ps_mean)   
    return ps_pred.numpy(), xhi_pred.numpy()