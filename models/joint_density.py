"""Positive continuous joint density, in electrons/Angstrom^3.

This is a learned approximation, not a quantum solver. QM reference samples
are accepted by supervision_loss only; no labels/grid metadata enter predict.
The pre-existing Gaussian voxel channel is independent and unchanged.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from config import COORD_SCALE, LIGAND_NEAR_RES_CUTOFF, MD_USE_PBC, NODE_FEAT_DIM
from utils.md import minimum_image_displacement
from utils.pocket import dynamic_residue_pocket_mask


class JointElectronDensity(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.components = 4
        self.parameter_head = nn.Sequential(
            nn.LayerNorm(NODE_FEAT_DIM + 1 + 2 * d_model),
            nn.Linear(NODE_FEAT_DIM + 1 + 2 * d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, self.components * 5),
        )
        nn.init.zeros_(self.parameter_head[-1].weight)
        nn.init.zeros_(self.parameter_head[-1].bias)
        self.register_buffer('base_width_A', torch.tensor([0.10, 0.25, 0.6, 1.2]))
        probes = [torch.zeros(3)]
        for radius in (0.5, 1.0, 2.0):
            for axis in torch.eye(3):
                probes.extend([radius * axis, -radius * axis])
        self.register_buffer('probes_A', torch.stack(probes))
        self.field_encoder = nn.Sequential(nn.LayerNorm(2 * len(probes)),
                                           nn.Linear(2 * len(probes), d_model), nn.SiLU(),
                                           nn.Linear(d_model, d_model))

    def predict(self, coords, node_feat, node_mask, node_is_ligand,
                node_residue_index, node_rot, mapped_context, pocket_context, boxes=None):
        """Only generated protein + coarse ligand and chemical identity enter."""
        B, T, N = coords.shape[:3]
        boxes = boxes if MD_USE_PBC else None
        active = dynamic_residue_pocket_mask(
            coords, coords, node_mask, node_is_ligand, node_residue_index, boxes,
            cutoff=float(LIGAND_NEAR_RES_CUTOFF) / float(COORD_SCALE),
        )
        lig = node_mask.bool() & node_is_ligand.bool()
        active = active | lig[:, None]
        features = torch.cat([
            node_feat[:, None].expand(-1, T, -1, -1),
            node_is_ligand[:, None, :, None].expand(-1, T, -1, -1).to(coords),
            mapped_context, pocket_context[:, :, None].expand(-1, -1, N, -1),
        ], dim=-1)
        raw = self.parameter_head(features).float().reshape(B, T, N, self.components, 5)
        z = (node_feat[..., 0].float() * 53.0).clamp_min(1.0)[:, None, :, None]
        amplitude = z * F.softplus(raw[..., 0]) / (self.components * math.log(2.0))
        width = self.base_width_A * torch.exp(0.7 * torch.tanh(raw[..., 1]))
        local_offset_A = torch.tanh(raw[..., 2:5])
        world_offset_A = torch.einsum('btnki,btnji->btnkj', local_offset_A, node_rot.float())
        centers_A = coords.float()[..., None, :] * float(COORD_SCALE) + world_offset_A
        return dict(centers_A=centers_A, amplitude=amplitude, width_A=width,
                    atom_mask=active, ligand_coords_A=coords.float() * float(COORD_SCALE),
                    ligand_mask=lig, boxes_A=None if boxes is None else boxes.float() * float(COORD_SCALE))

    @staticmethod
    def evaluate(field, points_A):
        """Evaluate only the 10-A ligand neighborhood; points are (B,T,S,3).

        Chunking bounds temporary memory. No maximum-density normalization or
        density-based top-k selection is used in this new branch.
        """
        rows = []
        for b in range(points_A.shape[0]):
            frames = []
            for t in range(points_A.shape[1]):
                active = field['atom_mask'][b, t]
                centers = field['centers_A'][b, t, active].reshape(-1, 3)
                amp = field['amplitude'][b, t, active].reshape(-1)
                width = field['width_A'][b, t, active].reshape(-1)
                box = None if field['boxes_A'] is None else field['boxes_A'][b, t]
                lig = field['ligand_coords_A'][b, t, field['ligand_mask'][b]]
                chunks = []
                for q0 in range(0, points_A.shape[2], 128):
                    points = points_A[b, t, q0:q0 + 128].float()
                    values = points.new_zeros(points.shape[0]) + amp.sum() * 0.0
                    for a0 in range(0, centers.shape[0], 256):
                        delta = points[:, None] - centers[None, a0:a0 + 256]
                        if box is not None:
                            delta = minimum_image_displacement(delta, box)
                        sigma = width[a0:a0 + 256]
                        basis = torch.exp(-0.5 * delta.square().sum(-1) / sigma.square()[None])
                        norm = (2.0 * math.pi) ** 1.5 * sigma.pow(3)
                        values = values + (basis * (amp[a0:a0 + 256] / norm)[None]).sum(-1)
                    if lig.numel():
                        delta_lig = points[:, None] - lig[None]
                        if box is not None:
                            delta_lig = minimum_image_displacement(delta_lig, box)
                        roi = delta_lig.square().sum(-1).amin(-1) <= float(LIGAND_NEAR_RES_CUTOFF) ** 2
                        values = values * roi
                    else:
                        values = values * 0.0
                    chunks.append(values)
                frames.append(torch.cat(chunks))
            rows.append(torch.stack(frames))
        return torch.stack(rows)

    def ligand_context(self, field, query_coords, node_rot):
        """Encode predicted field samples, never QM labels, around ligand queries."""
        B, T, N = query_coords.shape[:3]
        result = query_coords.new_zeros((B, T, N, self.field_encoder[-1].out_features))
        # Query ligand atoms only; protein nodes keep a zero context.
        for b in range(B):
            ids = field['ligand_mask'][b].nonzero(as_tuple=False).flatten()
            if not ids.numel():
                continue
            offsets = torch.einsum('pi,tnji->tnpj', self.probes_A, node_rot[b, :, ids].float())
            query_points = query_coords[b, :, ids, None].float() * float(COORD_SCALE) + offsets
            # At high noise, ligand queries can lie outside the 10-A field.
            # Always include probes at the predicted coarse ligand as well,
            # so density conditioning does not disappear early in sampling.
            anchor_points = field['ligand_coords_A'][b, :, ids, None] + offsets
            points = torch.cat([anchor_points, query_points], dim=2)
            subfield = {k: (None if v is None else v[b:b + 1]) for k, v in field.items()}
            rho = self.evaluate(subfield, points.reshape(1, T, -1, 3))
            features = torch.log1p(rho).reshape(T, ids.numel(), -1)
            result[b, :, ids] = self.field_encoder(features).to(result.dtype)
        return result

    def supervision_loss(self, field, points, density, mask):
        """QM labels in normalized coordinates and e/A^3; missing != zero density."""
        zero = field['amplitude'].sum() * 0.0
        if points is None or density is None or mask is None or not bool(mask.any()):
            return zero, zero.detach()
        if points.shape[:-1] != density.shape or density.shape != mask.shape:
            raise ValueError('QM sample point/value/mask shapes disagree')
        if not bool(torch.isfinite(points[mask]).all() & torch.isfinite(density[mask]).all() & (density[mask] >= 0).all()):
            raise ValueError('QM targets must be finite nonnegative electron-number density')
        safe_points = torch.where(mask[..., None], points, torch.zeros_like(points)).detach()
        predicted = self.evaluate(field, safe_points * float(COORD_SCALE))
        target = torch.where(mask, density, torch.zeros_like(density)).detach()
        errors = F.smooth_l1_loss(torch.log1p(predicted), torch.log1p(target), reduction='none')
        loss = (errors * mask).sum() / mask.sum().clamp_min(1)
        return loss, mask.any(-1).float().mean()
