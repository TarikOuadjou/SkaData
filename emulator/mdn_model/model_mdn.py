import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

K_COMPONENTS = 3  


class Emulator21cm(nn.Module):
    """
    MDN emulator for 21cm observables.
    Models p(log10(PS2D) | theta) as a mixture of K log-normals,
    and p(xHI | theta) as a single Gaussian (xHI stays as-is).
 
    Input  : theta  (B, 6)
 
    PS2D outputs (mixture of K log-normals per pixel):
        ps2d_pi    : (B, 3, K, 10, 10)  — mixture weights (sum to 1 over K)
        ps2d_mu    : (B, 3, K, 10, 10)  — means in log10 space
        ps2d_sigma : (B, 3, K, 10, 10)  — stds in log10 space (positive)
 
    xHI outputs:
        xhi_mu     : (B, 3)
    """
 
    N_REDSHIFTS = 3
 
    def __init__(self, n_params: int = 6, n_redshifts: int = N_REDSHIFTS, K: int = K_COMPONENTS):
        super().__init__()
        self.n_redshifts = n_redshifts
        self.K = K
 
        # ── Shared encoder ───────────────────────────────────────────────────
        self.shared = nn.Sequential(
            nn.Linear(n_params, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 512),      nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256),      nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 256),      nn.LayerNorm(256), nn.GELU(),  # added layer
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
            nn.Conv2d(16, 16, kernel_size=3, padding=1),  # added layer
            nn.GELU(),
        )
 
        # ── PS2D MDN heads: pi, mu, log-sigma — each outputs K channels ──────
        # Each head now has one extra 16->16 conv before the final K-channel conv
        self.ps2d_pi_head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(16, K, kernel_size=3, padding=1),  # raw logits → softmax
        )
        self.ps2d_mu_head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(16, K, kernel_size=3, padding=1),  # means in log10 space
        )
        self.ps2d_lsig_head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(16, K, kernel_size=3, padding=1),  # log-sigma → softplus
        )
 
        # ── xHI head (unchanged) ─────────────────────────────────────────────
        self.xhi_mu_head = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64,  32), nn.GELU(),
            nn.Linear(32, n_redshifts),
            nn.Sigmoid(),
        )
 
    def forward(self, theta: torch.Tensor):
        """
        theta : (B, 6)
        returns
            ps2d_pi    : (B, Z, K, 10, 10)  — mixture weights, sum to 1 over K dim
            ps2d_mu    : (B, Z, K, 10, 10)  — component means in log10 space
            ps2d_sigma : (B, Z, K, 10, 10)  — component stds (positive)
            xhi_mu     : (B, Z)
        """
        B = theta.size(0)
        Z = self.n_redshifts
        K = self.K
 
        h = self.shared(theta)                                      # (B, 256)
 
        # ── PS2D ─────────────────────────────────────────────────────────────
        feat = self.ps2d_fc(h)                                      # (B, Z·16·5·5)
        feat = feat.view(B * Z, 16, 5, 5)                          # (B·Z, 16, 5, 5)
        feat = self.ps2d_cnn(feat)                                  # (B·Z, 16, 10, 10)
 
        # Each head: (B·Z, K, 10, 10)
        ps2d_pi_logit = self.ps2d_pi_head(feat)
        ps2d_mu_raw   = self.ps2d_mu_head(feat)
        ps2d_lsig     = self.ps2d_lsig_head(feat)
 
        # Reshape to (B, Z, K, 10, 10)
        ps2d_pi = F.softmax(ps2d_pi_logit.view(B, Z, K, 10, 10), dim=2)
        ps2d_mu    = ps2d_mu_raw.view(B, Z, K, 10, 10)
        ps2d_sigma = F.softplus(ps2d_lsig).view(B, Z, K, 10, 10) + 1e-6
    
        # ── xHI ──────────────────────────────────────────────────────────────
        xhi_mu = self.xhi_mu_head(h)                               # (B, Z)
 
        return ps2d_pi, ps2d_mu, ps2d_sigma, xhi_mu


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def mdn_chi2_loss(
    pi: torch.Tensor,     # (B, Z, K, H, W)  mixture weights
    mu: torch.Tensor,     # (B, Z, K, H, W)  component means
    sigma: torch.Tensor,  # (B, Z, K, H, W)  component stds
    target: torch.Tensor, # (B, Z, H, W)     ground truth in log10 space
) -> torch.Tensor:
    """
    MDN loss where each component uses a chi-squared (Gaussian NLL) metric.
    Still uses log-sum-exp over K components for numerical stability.

        log p(y) = logsumexp_k [ log π_k - 0.5*χ²_k - log σ_k - 0.5*log(2π) ]

    where χ²_k = ((y - μ_k) / σ_k)²
    """
    y = target.unsqueeze(2).expand_as(mu)           # (B, Z, K, H, W)

    chi2 = ((y - mu) / sigma) ** 2                  # (B, Z, K, H, W)

    log_gauss = (
        -0.5 * chi2
        - torch.log(sigma)
        - 0.5 * math.log(2 * math.pi)
    )                                                # (B, Z, K, H, W)

    log_pi = torch.log(pi.clamp(min=1e-8))          # (B, Z, K, H, W)
    log_weighted = log_pi + log_gauss               # (B, Z, K, H, W)

    log_prob = torch.logsumexp(log_weighted, dim=2) # (B, Z, H, W)

    return -log_prob.mean()

def probabilistic_loss(
    ps2d_pi:     torch.Tensor,
    ps2d_mu:     torch.Tensor,
    ps2d_sigma:  torch.Tensor,
    ps2d_target: torch.Tensor,   # in log10 space (scaled)
    xhi_mu:      torch.Tensor,
    xhi_target:  torch.Tensor,
    w_ps:  float = 1.0,
    w_xhi: float = 1.0,
) -> tuple[torch.Tensor, dict]:

    loss_ps  = mdn_chi2_loss(ps2d_pi, ps2d_mu, ps2d_sigma, ps2d_target)
    loss_xhi = F.mse_loss(xhi_mu, xhi_target)
    total    = w_ps * loss_ps + w_xhi * loss_xhi

    metrics = {
        "loss_total": total.item(),
        "loss_ps2d":  loss_ps.item(),
        "loss_xhi":   loss_xhi.item(),
    }
    return total, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Scalers (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def compute_scalers(ps2d: torch.Tensor):
    ps_mean = ps2d.mean(dim=0, keepdim=True)
    ps_std  = ps2d.std(dim=0, keepdim=True).clamp(min=1e-8)
    return ps_mean, ps_std


def scale_ps(ps2d, ps_mean, ps_std):
    return (ps2d - ps_mean) / ps_std


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(
    train_thetas : torch.Tensor,
    train_ps2d   : torch.Tensor,
    train_xhi    : torch.Tensor,
    val_thetas   : torch.Tensor,
    val_ps2d     : torch.Tensor,
    val_xhi      : torch.Tensor,
    epochs       : int   = 2000,
    batch_size   : int   = 256,
    lr           : float = 1e-3,
    w_ps         : float = 1.0,
    w_xhi        : float = 1.0,
    K            : int   = K_COMPONENTS,
    checkpoint_dir: str  = "emulator/checkpoints",
) -> tuple[Emulator21cm, dict]:

    os.makedirs(checkpoint_dir, exist_ok=True)

    ps_mean, ps_std = compute_scalers(train_ps2d)
    train_ps2d_scaled = scale_ps(train_ps2d, ps_mean, ps_std)
    val_ps2d_scaled   = scale_ps(val_ps2d,   ps_mean, ps_std)

    scaler_path = f"{checkpoint_dir}/scalers.npz"
    np.savez(scaler_path, ps_mean=ps_mean.numpy(), ps_std=ps_std.numpy())
    print(f"Scalers saved → {scaler_path}")

    model     = Emulator21cm(n_params=6, n_redshifts=3, K=K)
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
            ps2d_pi, ps2d_mu, ps2d_sigma, xhi_mu = model(theta_b)
            loss, _ = probabilistic_loss(
                ps2d_pi, ps2d_mu, ps2d_sigma, ps_b,
                xhi_mu, xhi_b, w_ps, w_xhi,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(loader)

        # ── Validation ──
        model.eval()
        with torch.no_grad():
            ps2d_pi_v, ps2d_mu_v, ps2d_sigma_v, xhi_mu_v = model(val_thetas)
            val_loss = probabilistic_loss(
                ps2d_pi_v, ps2d_mu_v, ps2d_sigma_v, val_ps2d_scaled,
                xhi_mu_v, val_xhi, w_ps, w_xhi,
            )[0].item()

        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_loss)
        scheduler.step()

        if epoch % 50 == 0:
            print(f"epoch {epoch:>4}/{epochs}  train={avg_train:.6f}  val={val_loss:.6f}")

    torch.save(model.state_dict(), f"{checkpoint_dir}/emulator.pt")
    print(f"Model saved → {checkpoint_dir}/emulator.pt")
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    model: Emulator21cm,
    theta: torch.Tensor,
    checkpoint_dir: str = "emulator/mdn_model/checkpoints",
    scalers = None,
) -> dict:
    if isinstance(theta, np.ndarray):
        theta = torch.tensor(theta, dtype=torch.float32)
    if theta.dim() == 1:
        theta = theta.unsqueeze(0)
    if scalers is None:
        scalers = np.load(f"{checkpoint_dir}/scalers.npz")

    ps_mean = torch.tensor(scalers["ps_mean"])   # (1, 3, 10, 10)
    ps_std  = torch.tensor(scalers["ps_std"])     # (1, 3, 10, 10)
    
    model.eval()
    with torch.no_grad():
        ps2d_pi, ps2d_mu_s, ps2d_sigma_s, xhi_mu = model(theta)

        # Unscale to log10 space
        ps2d_mu    = ps2d_mu_s    * ps_std.unsqueeze(2) + ps_mean.unsqueeze(2)
        ps2d_sigma = ps2d_sigma_s * ps_std.unsqueeze(2)

        # Mixture mean in linear space
        ps2d_mean = (ps2d_pi * ps2d_mu).sum(dim=2)

    return {
        "ps2d_mean":     ps2d_mean,
        "ps2d_mu":       ps2d_mu,
        "ps2d_std":    ps2d_sigma,
        "ps2d_pi":       ps2d_pi,
        "xhi_mu":        xhi_mu,
    }

