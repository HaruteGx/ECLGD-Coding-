import math

import torch
import torch.nn as nn

from config import (
    COORD_SCALE,
    DROPOUT,
    PROTEIN_FF_DIM,
    PROTEIN_HEADS,
    PROTEIN_LAYERS,
    PROTEIN_SE3_KNN,
    PROTEIN_SE3_LAYERS,
    PROTEIN_SE3_RBF_DIM,
    PROTEIN_SE3_RADIUS_A,
    PROTEIN_SE3_VECTOR_CHANNELS,
)
from .temporal_utils import AA_VOCAB


def _finite(x, limit=1.0e4):
    return torch.nan_to_num(x, nan=0.0, posinf=limit, neginf=-limit).clamp(-limit, limit)


def _safe_unit(x, eps=1.0e-8):
    return x / torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(eps)


def _gather_residues(x, index):
    """Gather the residue dimension of (B,R,...) tensors with (B,R,K) indices."""
    batch = torch.arange(x.shape[0], device=x.device)[:, None, None]
    return x[batch, index]


def _sinusoidal_positions(length, dim, device, dtype):
    if dim < 2:
        return torch.zeros((length, dim), device=device, dtype=dtype)
    half = dim // 2
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    scale = math.log(10000.0) / max(half - 1, 1)
    frequency = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -scale)[None]
    emb = torch.cat([torch.sin(position * frequency), torch.cos(position * frequency)], dim=-1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb.to(dtype=dtype)


class SE3ResidueBlock(nn.Module):
    """Scalar/vector message passing with SE(3)-equivariant vector updates."""

    def __init__(self, d_scalar, d_vector, rbf_dim, dropout):
        super().__init__()
        edge_in = 2 * int(d_scalar) + int(rbf_dim) + 2
        self.edge_mlp = nn.Sequential(
            nn.LayerNorm(edge_in),
            nn.Linear(edge_in, d_scalar),
            nn.SiLU(),
            nn.Linear(d_scalar, d_scalar),
            nn.SiLU(),
        )
        self.edge_weight = nn.Linear(d_scalar, 1)
        self.direction_gate = nn.Linear(d_scalar, d_vector)
        self.vector_gate = nn.Linear(d_scalar, d_vector)
        self.vector_mix = nn.Parameter(torch.eye(d_vector))
        self.scalar_update = nn.Sequential(
            nn.LayerNorm(2 * d_scalar + d_vector),
            nn.Linear(2 * d_scalar + d_vector, d_scalar),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_scalar, d_scalar),
        )
        self.vector_update = nn.Parameter(torch.eye(d_vector))
        self.output_vector_gate = nn.Linear(d_scalar, d_vector)
        self.scalar_norm = nn.LayerNorm(d_scalar)

    def forward(self, scalar, vector, ca, neighbor_idx, neighbor_mask, edge_features):
        scalar_j = _gather_residues(scalar, neighbor_idx)
        scalar_i = scalar[:, :, None].expand_as(scalar_j)
        edge = self.edge_mlp(torch.cat([scalar_i, scalar_j, edge_features], dim=-1))

        mask = neighbor_mask.to(dtype=edge.dtype).unsqueeze(-1)
        weight = torch.sigmoid(self.edge_weight(edge)) * mask
        denom = weight.sum(dim=2).clamp_min(1.0)
        scalar_message = (edge * weight).sum(dim=2) / denom

        ca_j = _gather_residues(ca, neighbor_idx)
        direction = _safe_unit(ca_j - ca[:, :, None])
        vector_j = _gather_residues(vector, neighbor_idx)
        mixed_j = torch.einsum("brkvc,uv->brkuc", vector_j, self.vector_mix)
        vector_message = (
            mixed_j * torch.sigmoid(self.vector_gate(edge)).unsqueeze(-1)
            + direction.unsqueeze(-2) * torch.tanh(self.direction_gate(edge)).unsqueeze(-1)
        )
        vector_message = (vector_message * weight.unsqueeze(-1)).sum(dim=2) / denom.unsqueeze(-1)
        vector_norm = torch.linalg.norm(vector_message.float(), dim=-1).to(dtype=scalar.dtype)

        scalar_delta = self.scalar_update(torch.cat([scalar, scalar_message, vector_norm], dim=-1))
        scalar = self.scalar_norm(scalar + scalar_delta)
        mixed_message = torch.einsum("brvc,uv->bruc", vector_message, self.vector_update)
        vector = vector + mixed_message * torch.sigmoid(self.output_vector_gate(scalar)).unsqueeze(-1)
        return _finite(scalar), _finite(vector)


class ProteinSE3TemporalTransformer(nn.Module):
    """
    Hybrid protein encoder.

    SE(3)-equivariant residue graph blocks encode backbone geometry, a scalar
    Transformer handles long-range sequence interactions, and a residue-wise
    GRU carries temporal state between observed and generated frames.
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = int(d_model)
        self.vector_channels = int(PROTEIN_SE3_VECTOR_CHANNELS)
        self.rbf_dim = int(PROTEIN_SE3_RBF_DIM)
        self.knn = int(PROTEIN_SE3_KNN)
        radius = float(PROTEIN_SE3_RADIUS_A) / float(COORD_SCALE)
        self.radius = radius
        self.register_buffer("rbf_centers", torch.linspace(0.0, radius, self.rbf_dim), persistent=False)
        self.rbf_width = max(radius / max(1, self.rbf_dim - 1), 1.0e-3)

        self.seq_emb = nn.Embedding(int(AA_VOCAB), self.d_model)
        self.chain_embedding = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, self.d_model),
        )
        self.geometry_in = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.vector_seed = nn.Parameter(torch.empty(self.vector_channels, 3))
        nn.init.orthogonal_(self.vector_seed)
        self.se3_blocks = nn.ModuleList(
            [
                SE3ResidueBlock(
                    self.d_model,
                    self.vector_channels,
                    self.rbf_dim,
                    float(DROPOUT),
                )
                for _ in range(int(PROTEIN_SE3_LAYERS))
            ]
        )

        n_heads = max(1, min(int(PROTEIN_HEADS), self.d_model))
        while self.d_model % n_heads != 0:
            n_heads -= 1
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=int(PROTEIN_FF_DIM),
            dropout=float(DROPOUT),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.sequence_transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=int(PROTEIN_LAYERS),
            enable_nested_tensor=False,
        )
        self.temporal_cell = nn.GRUCell(self.d_model, self.d_model)
        self.temporal_norm = nn.LayerNorm(self.d_model)
        self.temporal_dropout = nn.Dropout(float(DROPOUT))
        self.vector_output_gate = nn.Linear(self.d_model, self.vector_channels)

    def _chain_ids(self, mask, chain_pair_mask):
        B, R = mask.shape
        if R == 0:
            return torch.zeros((B, 0), device=mask.device, dtype=torch.long)
        if chain_pair_mask is None:
            pair = mask[:, :-1] & mask[:, 1:]
        else:
            pair = chain_pair_mask.to(device=mask.device, dtype=torch.bool)[:, : max(0, R - 1)]
            pair = pair & mask[:, :-1] & mask[:, 1:]
        starts = torch.ones((B, R), device=mask.device, dtype=torch.long)
        if R > 1:
            starts[:, 1:] = (~pair).long()
        return torch.cumsum(starts, dim=1) - 1

    def _neighbors(self, ca, mask, chain_pair_mask):
        B, R = mask.shape
        if R == 0:
            empty_idx = torch.zeros((B, 0, 1), device=ca.device, dtype=torch.long)
            return empty_idx, mask[:, :, None], ca.new_zeros((B, 0, 1, self.rbf_dim + 2))

        distance = torch.cdist(ca.float(), ca.float()).to(dtype=ca.dtype)
        valid_pair = mask[:, :, None] & mask[:, None, :]
        eye = torch.eye(R, device=ca.device, dtype=torch.bool)[None]
        distance = distance.masked_fill(~valid_pair | eye, float("inf"))
        k = max(1, min(self.knn, max(1, R - 1)))
        knn_distance, knn_idx = torch.topk(distance, k=k, dim=-1, largest=False, sorted=False)
        knn_mask = torch.isfinite(knn_distance) & (knn_distance <= float(self.radius))

        residue_idx = torch.arange(R, device=ca.device)
        prev_idx = (residue_idx - 1).clamp_min(0)[None, :, None].expand(B, -1, -1)
        next_idx = (residue_idx + 1).clamp_max(R - 1)[None, :, None].expand(B, -1, -1)
        if chain_pair_mask is None:
            pair = mask[:, :-1] & mask[:, 1:]
        else:
            pair = chain_pair_mask.to(device=ca.device, dtype=torch.bool)[:, : max(0, R - 1)]
            pair = pair & mask[:, :-1] & mask[:, 1:]
        prev_mask = torch.zeros((B, R, 1), device=ca.device, dtype=torch.bool)
        next_mask = torch.zeros_like(prev_mask)
        if R > 1:
            prev_mask[:, 1:, 0] = pair
            next_mask[:, :-1, 0] = pair

        neighbor_idx = torch.cat([knn_idx, prev_idx, next_idx], dim=-1)
        neighbor_mask = torch.cat([knn_mask, prev_mask, next_mask], dim=-1)
        ca_j = _gather_residues(ca, neighbor_idx)
        dist = torch.linalg.norm(ca_j - ca[:, :, None], dim=-1)
        rbf = torch.exp(-((dist[..., None] - self.rbf_centers) / self.rbf_width) ** 2)

        idx_i = residue_idx[None, :, None].expand_as(neighbor_idx)
        seq_sep = torch.log1p((neighbor_idx - idx_i).abs().to(dtype=ca.dtype))
        seq_sep = seq_sep / math.log(max(2, R))
        chain_ids = self._chain_ids(mask, chain_pair_mask)
        same_chain = (_gather_residues(chain_ids, neighbor_idx) == chain_ids[:, :, None]).to(dtype=ca.dtype)
        edge_features = torch.cat([rbf, seq_sep[..., None], same_chain[..., None]], dim=-1)
        return neighbor_idx, neighbor_mask, edge_features

    def forward(self, seq_ids, seq_mask, protein_bb, temporal_state=None, chain_pair_mask=None):
        mask = seq_mask.to(device=protein_bb.device, dtype=torch.bool)
        n_vec = protein_bb[..., 0, :] - protein_bb[..., 1, :]
        c_vec = protein_bb[..., 2, :] - protein_bb[..., 1, :]
        n_len = torch.linalg.norm(n_vec, dim=-1)
        c_len = torch.linalg.norm(c_vec, dim=-1)
        n_unit = _safe_unit(n_vec)
        c_unit = _safe_unit(c_vec)
        normal = _safe_unit(torch.cross(n_unit, c_unit, dim=-1))
        geometry = torch.stack(
            [n_len, c_len, (n_unit * c_unit).sum(dim=-1), n_len * c_len, (n_len - c_len).abs()],
            dim=-1,
        )

        scalar_length = protein_bb.shape[1]
        position = _sinusoidal_positions(
            scalar_length,
            self.d_model,
            protein_bb.device,
            protein_bb.dtype,
        )
        chain_ids = self._chain_ids(mask, chain_pair_mask).to(dtype=protein_bb.dtype)
        chain_context = torch.stack(
            [
                torch.sin(chain_ids),
                torch.cos(chain_ids),
                torch.sin(0.1 * chain_ids),
                torch.cos(0.1 * chain_ids),
            ],
            dim=-1,
        )
        scalar = self.seq_emb(seq_ids) + self.geometry_in(geometry)
        scalar = scalar + position[None] + self.chain_embedding(chain_context)
        raw_vectors = torch.stack([n_unit, c_unit, normal], dim=-2)
        vector = torch.einsum("brkc,vk->brvc", raw_vectors, self.vector_seed)
        ca = protein_bb[..., 1, :]
        neighbor_idx, neighbor_mask, edge_features = self._neighbors(ca, mask, chain_pair_mask)
        for block in self.se3_blocks:
            scalar, vector = block(
                scalar,
                vector,
                ca,
                neighbor_idx,
                neighbor_mask,
                edge_features,
            )

        scalar = self.sequence_transformer(scalar, src_key_padding_mask=~mask)
        if temporal_state is None or temporal_state.shape != scalar.shape:
            temporal_state = torch.zeros_like(scalar)
        next_temporal = self.temporal_cell(
            scalar.reshape(-1, self.d_model),
            temporal_state.reshape(-1, self.d_model),
        ).reshape_as(scalar)
        next_temporal = next_temporal * mask.unsqueeze(-1).to(dtype=next_temporal.dtype)
        residue_h = self.temporal_norm(scalar + self.temporal_dropout(next_temporal))
        residue_h = residue_h * mask.unsqueeze(-1).to(dtype=residue_h.dtype)
        vector = vector * torch.sigmoid(self.vector_output_gate(residue_h)).unsqueeze(-1)
        vector = vector * mask[:, :, None, None].to(dtype=vector.dtype)
        denom = mask.to(dtype=residue_h.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
        protein_h = (residue_h * mask.unsqueeze(-1).to(dtype=residue_h.dtype)).sum(dim=1) / denom
        return _finite(residue_h), _finite(protein_h), _finite(vector), _finite(next_temporal)


class EquivariantBackboneUpdate(nn.Module):
    """Generate N/CA/C displacements as invariant coefficients over equivariant vectors."""

    def __init__(self, scalar_dim, vector_channels, hidden_dim, step_scale=0.1):
        super().__init__()
        self.vector_channels = int(vector_channels)
        self.basis_channels = self.vector_channels + 3
        self.step_scale = float(step_scale)
        self.coefficients = nn.Sequential(
            nn.LayerNorm(int(scalar_dim)),
            nn.Linear(int(scalar_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 3 * self.basis_channels),
            nn.Tanh(),
        )

    def forward(self, scalar_context, residue_vectors, source_bb, residue_mask):
        n_axis = _safe_unit(source_bb[..., 0, :] - source_bb[..., 1, :])
        c_axis = _safe_unit(source_bb[..., 2, :] - source_bb[..., 1, :])
        normal = _safe_unit(torch.cross(n_axis, c_axis, dim=-1))
        local_basis = torch.stack([n_axis, c_axis, normal], dim=-2)
        basis = torch.cat([residue_vectors, local_basis], dim=-2)
        coeff = self.coefficients(scalar_context).reshape(
            scalar_context.shape[0], scalar_context.shape[1], 3, self.basis_channels
        )
        delta = self.step_scale * torch.einsum("brak,brkc->brac", coeff, basis)
        delta = delta * residue_mask[:, :, None, None].to(dtype=delta.dtype)
        return _finite(delta)
