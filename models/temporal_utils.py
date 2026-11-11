import math
import os
import time

import torch
import torch.nn.functional as F

from config import (
    COORD_SCALE,
    PROTEIN_BACKBONE_PROJECT,
    PROTEIN_BACKBONE_FORCE_SEQUENTIAL_CHAIN,
    PROTEIN_BACKBONE_PROJECT_MODE,
    PROTEIN_BACKBONE_PROJECT_ITERS,
    PROTEIN_BACKBONE_PROJECT_STRENGTH,
    TEMPORAL_DEBUG_ALL_RANKS,
    TEMPORAL_DEBUG_EVERY_FRAMES,
    TEMPORAL_INTERNAL_DEBUG,
)


AA_VOCAB = 21


def _debug_rank():
    try:
        return int(os.environ.get("RANK", "0"))
    except ValueError:
        return 0


def _debug_enabled():
    return bool(TEMPORAL_INTERNAL_DEBUG) and (bool(TEMPORAL_DEBUG_ALL_RANKS) or _debug_rank() == 0)


def _debug_every_frames():
    return max(1, int(TEMPORAL_DEBUG_EVERY_FRAMES))


def _debug_sync(device=None):
    if not _debug_enabled() or not torch.cuda.is_available():
        return
    if device is not None and getattr(device, "type", None) != "cuda":
        return
    torch.cuda.synchronize(device=device if getattr(device, "type", None) == "cuda" else None)


def _debug_now(device=None):
    _debug_sync(device)
    return time.perf_counter()


def _debug_log(message):
    if not _debug_enabled():
        return
    print(f"[temporal_debug rank={_debug_rank()}] {message}", flush=True)

def _masked_mean(x, mask, dim, keepdim=False):
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0e4, neginf=-1.0e4).clamp(-1.0e4, 1.0e4)
    mask = mask.to(device=x.device, dtype=x.dtype)
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    return (x * mask).sum(dim=dim, keepdim=keepdim) / denom


def _masked_smooth_l1(pred, target, mask):
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0e4, neginf=-1.0e4).clamp(-1.0e4, 1.0e4)
    target = torch.nan_to_num(target, nan=0.0, posinf=1.0e4, neginf=-1.0e4).clamp(-1.0e4, 1.0e4)
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.expand_as(loss).sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


def _safe_normalize(x, fallback, eps=1e-8):
    norm = torch.linalg.norm(x, dim=-1, keepdim=True)
    fb = fallback.to(device=x.device, dtype=x.dtype)
    while fb.ndim < x.ndim:
        fb = fb.unsqueeze(0)
    fb = fb.expand_as(x)
    return torch.where(norm > eps, x / norm.clamp_min(eps), fb)


def _backbone_frames(backbone):
    n = backbone[..., 0, :]
    ca = backbone[..., 1, :]
    c = backbone[..., 2, :]
    x_axis = _safe_normalize(c - ca, backbone.new_tensor([1.0, 0.0, 0.0]))
    n_vec = n - ca
    y_axis = n_vec - (n_vec * x_axis).sum(dim=-1, keepdim=True) * x_axis
    y_axis = _safe_normalize(y_axis, backbone.new_tensor([0.0, 1.0, 0.0]))
    z_axis = _safe_normalize(torch.cross(x_axis, y_axis, dim=-1), backbone.new_tensor([0.0, 0.0, 1.0]))
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    rot = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    return rot, ca


def _chain_pair_mask_from_residue_mask(protein_mask, chain_pair_mask=None):
    if protein_mask.shape[-1] < 2:
        return protein_mask.new_zeros(protein_mask.shape[:-1] + (0,), dtype=torch.bool)
    pair_mask = protein_mask[..., :-1].bool() & protein_mask[..., 1:].bool()
    if chain_pair_mask is not None:
        chain_pair_mask = chain_pair_mask.to(device=protein_mask.device, dtype=torch.bool)
        if chain_pair_mask.ndim == protein_mask.ndim - 1:
            chain_pair_mask = chain_pair_mask.unsqueeze(1)
        pair_mask = pair_mask & chain_pair_mask[..., : pair_mask.shape[-1]]
    elif not bool(PROTEIN_BACKBONE_FORCE_SEQUENTIAL_CHAIN):
        return protein_mask.new_zeros(protein_mask.shape[:-1] + (protein_mask.shape[-1] - 1,), dtype=torch.bool)
    return pair_mask


def _idealize_backbone_local_geometry(backbone, residue_mask=None):
    n_ca = 1.46 / float(COORD_SCALE)
    ca_c = 1.53 / float(COORD_SCALE)
    n = backbone[..., 0, :]
    ca = backbone[..., 1, :]
    c = backbone[..., 2, :]
    n_dir = _safe_normalize(n - ca, backbone.new_tensor([-1.0, 0.0, 0.0]))
    c_dir = _safe_normalize(c - ca, backbone.new_tensor([1.0, 0.0, 0.0]))
    ideal = torch.stack([ca + n_dir * n_ca, ca, ca + c_dir * ca_c], dim=-2)
    if residue_mask is None:
        return ideal
    mask = residue_mask.to(device=backbone.device, dtype=torch.bool)
    while mask.ndim < backbone.ndim - 2:
        mask = mask.unsqueeze(1)
    return torch.where(mask.unsqueeze(-1).unsqueeze(-1), ideal, backbone)


def _continuous_ca_trace(ca, residue_mask, pair_mask):
    if ca.shape[-2] < 2:
        return ca
    caca = 3.80 / float(COORD_SCALE)
    out = ca.clone()
    for i in range(1, int(ca.shape[-2])):
        pair_i = pair_mask[..., i - 1]
        valid_i = residue_mask[..., i]
        if not bool((pair_i & valid_i).any().item()):
            continue
        step = ca[..., i, :] - out[..., i - 1, :]
        fallback = ca[..., min(i + 1, ca.shape[-2] - 1), :] - ca[..., max(i - 1, 0), :]
        direction = _safe_normalize(step, fallback)
        placed = out[..., i - 1, :] + direction * caca
        out[..., i, :] = torch.where((pair_i & valid_i).unsqueeze(-1), placed, out[..., i, :])
    return out


def _ideal_chain_from_ca_trace(backbone, residue_mask, pair_mask):
    ca = backbone[..., 1, :]
    ca = _continuous_ca_trace(ca, residue_mask, pair_mask)
    rot, _origin = _backbone_frames(backbone)
    y_axis = rot[..., 1]

    if ca.shape[-2] > 1:
        forward = ca[..., 1:, :] - ca[..., :-1, :]
        fallback_forward = backbone[..., 2, :] - backbone[..., 1, :]
        forward = _safe_normalize(forward, fallback_forward[..., :-1, :])
        first = forward[..., :1, :]
        last = forward[..., -1:, :]
        prev_axis = torch.cat([first, forward], dim=-2)
        next_axis = torch.cat([forward, last], dim=-2)
        prev_valid = torch.cat([pair_mask.new_zeros(pair_mask.shape[:-1] + (1,)), pair_mask], dim=-1)
        next_valid = torch.cat([pair_mask, pair_mask.new_zeros(pair_mask.shape[:-1] + (1,))], dim=-1)
        pred_x = _safe_normalize(backbone[..., 2, :] - backbone[..., 1, :], backbone.new_tensor([1.0, 0.0, 0.0]))
        prev_axis = torch.where(prev_valid.unsqueeze(-1), prev_axis, pred_x)
        next_axis = torch.where(next_valid.unsqueeze(-1), next_axis, pred_x)
    else:
        pred_x = _safe_normalize(backbone[..., 2, :] - backbone[..., 1, :], backbone.new_tensor([1.0, 0.0, 0.0]))
        prev_axis = pred_x
        next_axis = pred_x

    y_axis = y_axis - (y_axis * next_axis).sum(dim=-1, keepdim=True) * next_axis
    y_axis = _safe_normalize(y_axis, backbone.new_tensor([0.0, 1.0, 0.0]))

    ca_c = 1.53 / float(COORD_SCALE)
    n_ca = 1.46 / float(COORD_SCALE)
    # Rosetta/AlphaFold-style idealized placement around the CA trace. The axial
    # components leave roughly a peptide-length gap between C_i and N_{i+1};
    # perpendicular components preserve a backbone plane without using sidechains.
    c_x = 1.23 / float(COORD_SCALE)
    n_x = 1.24 / float(COORD_SCALE)
    c_y = math.sqrt(max(1.53 * 1.53 - 1.23 * 1.23, 1.0e-6)) / float(COORD_SCALE)
    n_y = math.sqrt(max(1.46 * 1.46 - 1.24 * 1.24, 1.0e-6)) / float(COORD_SCALE)
    c_scale = ca_c / math.sqrt(c_x * c_x + c_y * c_y)
    n_scale = n_ca / math.sqrt(n_x * n_x + n_y * n_y)
    c_x, c_y = c_x * c_scale, c_y * c_scale
    n_x, n_y = n_x * n_scale, n_y * n_scale

    n = ca - prev_axis * n_x + y_axis * n_y
    c = ca + next_axis * c_x + y_axis * c_y
    ideal = torch.stack([n, ca, c], dim=-2)
    return torch.where(residue_mask.unsqueeze(-1).unsqueeze(-1), ideal, backbone)


def _project_backbone_chain_geometry(backbone, protein_mask=None, chain_pair_mask=None, iters=None, strength=None):
    """
    Differentiable constrained backbone projection inspired by RFdiffusion/Rosetta-style
    ideal geometry: keep predicted residue placement as the prior, but project N/CA/C
    toward ideal local bonds and adjacent-residue C-N / CA-CA continuity.
    Coordinates are normalized by COORD_SCALE.
    """
    if not bool(PROTEIN_BACKBONE_PROJECT):
        return backbone
    if backbone.shape[-3] < 2:
        return backbone
    num_iters = int(PROTEIN_BACKBONE_PROJECT_ITERS if iters is None else iters)
    if num_iters <= 0:
        return backbone
    blend = float(PROTEIN_BACKBONE_PROJECT_STRENGTH if strength is None else strength)
    if blend <= 0.0:
        return backbone
    blend = min(blend, 1.0)

    out = torch.nan_to_num(backbone, nan=0.0, posinf=1.0e4, neginf=-1.0e4).clamp(-1.0e4, 1.0e4)
    residue_mask = protein_mask
    if residue_mask is None:
        residue_mask = torch.ones(out.shape[:-2], device=out.device, dtype=torch.bool)
    else:
        residue_mask = residue_mask.to(device=out.device, dtype=torch.bool)
    pair_mask = _chain_pair_mask_from_residue_mask(residue_mask, chain_pair_mask)
    if pair_mask.numel() == 0:
        return _idealize_backbone_local_geometry(out, residue_mask)

    project_mode = str(PROTEIN_BACKBONE_PROJECT_MODE).lower()
    use_ideal_chain = project_mode in ("ideal_chain", "chain", "ca_trace", "rosetta", "alphafold", "strong")
    if use_ideal_chain:
        out = _ideal_chain_from_ca_trace(out, residue_mask, pair_mask)

    cn = 1.33 / float(COORD_SCALE)
    caca = 3.80 / float(COORD_SCALE)
    eps = 1.0e-8
    pair_f = pair_mask.to(dtype=out.dtype).unsqueeze(-1)
    for _ in range(num_iters):
        out = _idealize_backbone_local_geometry(out, residue_mask)

        ca = out[..., 1, :]
        ca_vec = ca[..., 1:, :] - ca[..., :-1, :]
        ca_dist = torch.linalg.norm(ca_vec, dim=-1, keepdim=True).clamp_min(eps)
        ca_corr = 0.5 * (ca_dist - caca) * (ca_vec / ca_dist) * pair_f
        left = F.pad(ca_corr, (0, 0, 0, 1))
        right = F.pad(-ca_corr, (0, 0, 1, 0))
        count = F.pad(pair_f, (0, 0, 0, 1)) + F.pad(pair_f, (0, 0, 1, 0))
        out = out + ((left + right) / count.clamp_min(1.0)).unsqueeze(-2)

        out = _idealize_backbone_local_geometry(out, residue_mask)
        c = out[..., :-1, 2, :]
        n_next = out[..., 1:, 0, :]
        cn_vec = n_next - c
        cn_dist = torch.linalg.norm(cn_vec, dim=-1, keepdim=True).clamp_min(eps)
        cn_corr = 0.5 * (cn_dist - cn) * (cn_vec / cn_dist) * pair_f
        left = F.pad(cn_corr, (0, 0, 0, 1))
        right = F.pad(-cn_corr, (0, 0, 1, 0))
        count = F.pad(pair_f, (0, 0, 0, 1)) + F.pad(pair_f, (0, 0, 1, 0))
        out = out + ((left + right) / count.clamp_min(1.0)).unsqueeze(-2)

        if use_ideal_chain:
            out = _ideal_chain_from_ca_trace(out, residue_mask, pair_mask)

    out = _ideal_chain_from_ca_trace(out, residue_mask, pair_mask) if use_ideal_chain else _idealize_backbone_local_geometry(out, residue_mask)
    if blend >= 1.0:
        return out
    mask = residue_mask
    while mask.ndim < backbone.ndim - 2:
        mask = mask.unsqueeze(1)
    projected = backbone + float(blend) * (out - backbone)
    return torch.where(mask.unsqueeze(-1).unsqueeze(-1), projected, backbone)
