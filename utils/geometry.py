# -*- coding: utf-8 -*-
"""Geometry utilities.

Includes standard Kabsch rigid alignment and a weighted variant used for
electron-cloud-weighted anchoring during diffusion sampling.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def kabsch(P: torch.Tensor, Q: torch.Tensor):
    """Rigid alignment that maps P -> Q.

    Args:
        P: (M,3)
        Q: (M,3)

    Returns:
        R: (3,3) rotation
        t: (3,) translation
    """
    if P.ndim != 2 or Q.ndim != 2 or P.shape != Q.shape or P.shape[1] != 3:
        raise ValueError(f"kabsch expects P,Q of shape (M,3); got {tuple(P.shape)} and {tuple(Q.shape)}")
    Pc = P.mean(dim=0)
    Qc = Q.mean(dim=0)
    P0 = P - Pc
    Q0 = Q - Qc
    H = P0.t() @ Q0  # (3,3)
    U, S, Vh = torch.linalg.svd(H)
    R = Vh.t() @ U.t()
    if torch.linalg.det(R) < 0:
        Vh[-1, :] *= -1
        R = Vh.t() @ U.t()
    t = Qc - (R @ Pc)
    return R, t


@torch.no_grad()
def weighted_kabsch(P: torch.Tensor, Q: torch.Tensor, w: torch.Tensor, eps: float = 1e-8):
    """Weighted rigid alignment mapping P -> Q.

    Args:
        P: (M,3)
        Q: (M,3)
        w: (M,) non-negative weights

    Returns:
        R: (3,3)
        t: (3,)
    """
    if P.ndim != 2 or Q.ndim != 2 or P.shape != Q.shape or P.shape[1] != 3:
        raise ValueError(f"weighted_kabsch expects P,Q of shape (M,3); got {tuple(P.shape)} and {tuple(Q.shape)}")
    if w.ndim != 1 or w.shape[0] != P.shape[0]:
        raise ValueError(f"weighted_kabsch expects w of shape (M,); got {tuple(w.shape)}")
    w = w.to(P.dtype)
    wsum = w.sum().clamp_min(eps)
    Pc = (P * w[:, None]).sum(dim=0) / wsum
    Qc = (Q * w[:, None]).sum(dim=0) / wsum
    P0 = P - Pc
    Q0 = Q - Qc
    H = (P0 * w[:, None]).t() @ Q0
    U, S, Vh = torch.linalg.svd(H)
    R = Vh.t() @ U.t()
    if torch.linalg.det(R) < 0:
        Vh[-1, :] *= -1
        R = Vh.t() @ U.t()
    t = Qc - (R @ Pc)
    return R, t


@torch.no_grad()
def apply_rt(X: torch.Tensor, R: torch.Tensor, t: torch.Tensor):
    """Apply rigid transform to coordinates.

    Args:
        X: (...,3)
        R: (3,3)
        t: (3,)
    """
    return (X @ R.t()) + t
