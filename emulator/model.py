import torch
import torch.nn as nn
import torch.nn.functional as F


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
                               padding=1, output_padding=1),  # 5→10
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
            nn.Sigmoid(),   # x_HI ∈ [0, 1]
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

        # ── PS2D ─────────────────────────────────────────────────────────────
        # Decode all redshifts at once, then apply the shared CNN per slice
        feat = self.ps2d_fc(h)                                  # (B, Z·16·5·5)
        feat = feat.view(B * self.n_redshifts, 16, 5, 5)       # (B·Z, 16, 5, 5)
        ps2d = self.ps2d_cnn(feat).squeeze(1)                   # (B·Z, 10, 10)
        ps2d = ps2d.view(B, self.n_redshifts, 10, 10)          # (B, Z, 10, 10)

        # ── xHI ──────────────────────────────────────────────────────────────
        xhi = self.xhi_head(h)                                  # (B, 3)

        return ps2d, xhi


# ── Loss ─────────────────────────────────────────────────────────────────────

def mse_loss(ps2d_pred, ps2d_target, xhi_pred, xhi_target,
             w_ps: float = 1.0, w_xhi: float = 1.0) -> torch.Tensor:
    """
    Weighted MSE over both outputs.

    Args:
        w_ps  : weight for the PS2D term  (tune if scales differ)
        w_xhi : weight for the xHI term
    """
    loss_ps  = F.mse_loss(ps2d_pred, ps2d_target)
    loss_xhi = F.mse_loss(xhi_pred,  xhi_target)
    return w_ps * loss_ps + w_xhi * loss_xhi