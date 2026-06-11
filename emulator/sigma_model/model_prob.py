import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import glob

class Emulator21cm(nn.Module):
    """
    Probabilistic emulator for 21cm observables.
    Predicts mean and standard deviation for each output.

    Input  : theta  (B, 6)       — 6 astrophysical parameters
    Outputs:
        ps2d_mu    : (B, 3, 10, 10)  — mean 2-D power spectra at 3 redshifts
        ps2d_sigma : (B, 3, 10, 10)  — std  2-D power spectra at 3 redshifts
        xhi_mu     : (B, 3)          — mean neutral fractions at 3 redshifts
        xhi_sigma  : (B, 3)          — std  neutral fractions at 3 redshifts
    """

    N_REDSHIFTS = 3

    def __init__(self, n_params: int = 6, n_redshifts: int = N_REDSHIFTS):
        super().__init__()
        self.n_redshifts = n_redshifts

        # ── Shared encoder ───────────────────────────────────────────────────
        self.shared = nn.Sequential(
            nn.Linear(n_params, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 512),      nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256),      nn.LayerNorm(256), nn.GELU(),
        )

        # ── PS2D backbone (shared spatial features) ──────────────────────────
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
        )

        # ── PS2D mean / log-sigma heads ───────────────────────────────────────
        self.ps2d_mu_head    = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.ps2d_lsig_head  = nn.Conv2d(16, 1, kernel_size=3, padding=1)

        # ── xHI mean / log-sigma heads ────────────────────────────────────────
        self.xhi_mu_head = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64,  32), nn.GELU(),
            nn.Linear(32, n_redshifts),
            nn.Sigmoid(),           # mu in (0, 1)
        )
        self.xhi_lsig_head = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64,  32), nn.GELU(),
            nn.Linear(32, n_redshifts),
        )

    def forward(self, theta: torch.Tensor):
        """
        theta : (B, 6)
        returns
            ps2d_mu    : (B, 3, 10, 10)
            ps2d_sigma : (B, 3, 10, 10)  — always positive
            xhi_mu     : (B, 3)           — in (0, 1)
            xhi_sigma  : (B, 3)           — always positive
        """
        B = theta.size(0)

        h = self.shared(theta)                                      # (B, 256)

        # ── PS2D ─────────────────────────────────────────────────────────────
        feat = self.ps2d_fc(h)                                      # (B, Z·16·5·5)
        feat = feat.view(B * self.n_redshifts, 16, 5, 5)           # (B·Z, 16, 5, 5)
        feat = self.ps2d_cnn(feat)                                  # (B·Z, 16, 10, 10)

        ps2d_mu   = self.ps2d_mu_head(feat).squeeze(1)             # (B·Z, 10, 10)
        ps2d_lsig = self.ps2d_lsig_head(feat).squeeze(1)           # (B·Z, 10, 10)

        ps2d_mu    = ps2d_mu.view(B, self.n_redshifts, 10, 10)
        ps2d_sigma = F.softplus(ps2d_lsig).view(B, self.n_redshifts, 10, 10)

        # ── xHI ──────────────────────────────────────────────────────────────
        xhi_mu    = self.xhi_mu_head(h)                            # (B, 3)
        #xhi_sigma = F.softplus(self.xhi_lsig_head(h))             # (B, 3)

        return ps2d_mu, ps2d_sigma, xhi_mu

def gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
    """
    Mean Gaussian negative log-likelihood over all elements.
    NLL = 0.5 * [ log(2π) + 2·log(σ) + ((y - μ)/σ)² ]
    """
    return F.gaussian_nll_loss(mu, target, sigma ** 2, full=True, reduction="mean")

def probabilistic_loss(
    ps2d_mu:    torch.Tensor,
    ps2d_sigma: torch.Tensor,
    ps2d_target: torch.Tensor,
    xhi_mu:     torch.Tensor,
    xhi_target:  torch.Tensor,
    w_ps:  float = 1.0,
    w_xhi: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Combined Gaussian NLL for ps2d and xhi.

    Returns
    -------
    total_loss : scalar tensor
    metrics    : dict with individual loss components (for logging)
    """
    loss_ps  = gaussian_nll(ps2d_mu, ps2d_sigma, ps2d_target)
    loss_xhi = F.mse_loss(xhi_mu,  xhi_target)
    total    = w_ps * loss_ps + w_xhi * loss_xhi

    metrics = {
        "loss_total": total.item(),
        "loss_ps2d":  loss_ps.item(),
        "loss_xhi":   loss_xhi.item(),
    }
    return total, metrics

def compute_scalers(
    ps2d: torch.Tensor,   # (N, 3, 10, 10)
):
    log_ps = torch.log10(ps2d + 1e-30)                   # (N, 3, 10, 10)
    ps_mean = log_ps.mean(dim=0, keepdim=True)           # (1, 3, 10, 10)
    ps_std  = log_ps.std(dim=0, keepdim=True).clamp(min=1e-8)

    return ps_mean, ps_std


def scale_ps(ps2d, ps_mean, ps_std):
    return (torch.log10(ps2d + 1e-30) - ps_mean) / ps_std

def train(
    train_thetas : torch.Tensor,
    train_ps2d   : torch.Tensor,
    train_xhi    : torch.Tensor,
    val_thetas   : torch.Tensor,        
    val_ps2d     : torch.Tensor,        
    val_xhi      : torch.Tensor,        
    epochs       : int   = 1000,
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

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        for theta_b, ps_b, xhi_b in loader:
            ps2d_mu, ps2d_sigma, xhi_mu = model(theta_b)
            loss, _ = probabilistic_loss(
                ps2d_mu, ps2d_sigma, ps_b,
                xhi_mu, xhi_b,
                w_ps, w_xhi,
            )   
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(loader)

        # ── Validation ──
        model.eval()
        with torch.no_grad():
            ps2d_mu_val, ps2d_sigma_val, xhi_mu_val = model(val_thetas)
            val_loss = probabilistic_loss(
                ps2d_mu_val, ps2d_sigma_val, val_ps2d_scaled,
                xhi_mu_val,  val_xhi,
                w_ps, w_xhi,
            )[0].item()

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
    checkpoint_dir: str = "emulator/sigma_model/checkpoints",
    scalers = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    if scalers is None:
        scalers  = np.load(f"{checkpoint_dir}/scalers.npz")
    if isinstance(theta, np.ndarray):
        theta = torch.tensor(theta, dtype=torch.float32)
    if theta.dim() == 1:
        theta = theta.unsqueeze(0)   
    
    ps_mean  = torch.tensor(scalers["ps_mean"])  
    ps_std   = torch.tensor(scalers["ps_std"])    
    LN10 = torch.log(torch.tensor(10.0, dtype=theta.dtype, device=theta.device))

    model.eval()
    with torch.no_grad():
        ps2d_mu, ps2d_sigma, xhi_mu = model(theta)

        # Unscale: back to log10(PS) space
        ps_mu_log10    = ps2d_mu    * ps_std + ps_mean
        ps_sigma_log10 = ps2d_sigma * ps_std

        # Convert log10-normal params to linear-space mean and std
        # If X = log10(PS) ~ N(mu, sigma²), then PS is log-normal in base 10:
        #   E[PS]   = 10^(mu + 0.5 * sigma² * ln(10))
        #   Std[PS] = E[PS] * sqrt(10^(sigma² * ln(10)) - 1)
        ps2d_pred       = 10 ** (ps_mu_log10 + 0.5 * ps_sigma_log10 ** 2 * LN10)
        ps2d_sigma_pred = ps2d_pred * torch.sqrt(10 ** (ps_sigma_log10 ** 2 * LN10) - 1)

    return ps2d_pred, xhi_mu, ps2d_sigma_pred, ps_mu_log10, ps_sigma_log10