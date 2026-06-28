"""Energy head and contrastive objectives (CWG-E block B, Gate-4 stubs).

Trained from the same per-cluster coupling Γ used by ``drift_loss_ot``
in cluster-mode. The energy push term coincides with W-Flow's self-transport
T_{q,q}, so the two halves of the CWG-E loss share machinery rather than
compose orthogonally.

Interface placeholder — fill in once Gate 3 (CIFAR-10) passes.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class EnergyHead(nn.Module):
    """Scalar potential head V_ψ(x) sharing the generator backbone.

    The backbone is expected to expose pooled features; this head projects
    them to a scalar potential.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features).squeeze(-1)


def langevin_sample(
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    x_init: torch.Tensor,
    num_steps: int = 8,
    step_size: float = 0.01,
    noise_std: float = 0.005,
) -> torch.Tensor:
    """Short Langevin chain initialised from generator samples.

    TODO(gate-4): annealed schedule + couple with the OT residual.
    """
    x = x_init.detach().clone().requires_grad_(True)
    for _ in range(int(num_steps)):
        e = energy_fn(x).sum()
        grad = torch.autograd.grad(e, x)[0]
        with torch.no_grad():
            x = x - step_size * grad + noise_std * torch.randn_like(x)
        x.requires_grad_(True)
    return x.detach()


def energy_loss_ot(
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    transport_target: torch.Tensor,
) -> torch.Tensor:
    """L_OT — align V_ψ's gradient with the cluster-wise OT residual at z.

    TODO(gate-4): use the T̃_{q,p} returned by ``_barycentric_map_clustered``
    so the same Γ drives both objectives.
    """
    raise NotImplementedError("energy_loss_ot — Gate 4")


def energy_loss_cd(
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    positives: torch.Tensor,
    negatives: torch.Tensor,
) -> torch.Tensor:
    """L_CD — contrastive divergence with generator-initialised negatives."""
    raise NotImplementedError("energy_loss_cd — Gate 4")
