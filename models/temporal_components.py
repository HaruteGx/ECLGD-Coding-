import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    CLOUD_CHANNELS, CLOUD_EMB_DIM, CLOUD_LIGAND_DROPOUT,
    GNN_LAYERS, LSTM_PROTEIN_CONTEXT_WEIGHT, MD_INTEGRATOR_ACCEL_SCALE, NODE_FEAT_DIM,
)
from utils.md import minimum_image_displacement_batched
from .cloud_encoder import EquivariantCloudEncoder
from .gnn import EdgeGNN, edge_attr_from_coords
from .protein_se3 import EquivariantBackboneUpdate, ProteinSE3TemporalTransformer
from .temporal_utils import _masked_mean, _project_backbone_chain_geometry


def _finite_tensor(x, limit=1.0e4):
    limit = float(limit)
    return torch.nan_to_num(x, nan=0.0, posinf=limit, neginf=-limit).clamp(-limit, limit)


def _finite_loss(loss):
    return torch.nan_to_num(loss, nan=0.0, posinf=1.0e4, neginf=0.0)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t):
        half = self.dim // 2
        if half < 1:
            raise ValueError("time embedding dim must be at least 2")
        factor = math.log(10000.0) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -factor)
        emb = t.float().reshape(-1, 1) * emb.reshape(1, -1)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class GatedMultimodalLSTMCell(nn.Module):
    """
    LSTM cell whose internal memory gates encode graph-cache tensors directly.

    Protein backbone/sequence, pocket graph, and cloud grid are encoded inside
    the cell, decoded into three x0 memory states, then concatenated into the
    frame-diffusion condition used by downstream protein and pocket diffusion.
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = int(d_model)
        self.cloud = EquivariantCloudEncoder(d_out=int(CLOUD_EMB_DIM), in_channels=int(CLOUD_CHANNELS), d_hid=d_model)
        self.cloud_proj = nn.Linear(int(CLOUD_EMB_DIM), d_model)
        self.protein = ProteinSE3TemporalTransformer(d_model)
        self.gnn = EdgeGNN(
            d_node_in=int(NODE_FEAT_DIM) + 8,
            d_edge=4,
            d_hid=d_model,
            n_layers=int(GNN_LAYERS),
            d_cloud=int(CLOUD_EMB_DIM),
        )
        self.node_out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.state_mixer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        memory_in_dim = 3 * d_model
        self.protein_memory_write = nn.Sequential(
            nn.LayerNorm(memory_in_dim),
            nn.Linear(memory_in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3 * d_model),
        )
        self.graph_memory_write = nn.Sequential(
            nn.LayerNorm(memory_in_dim),
            nn.Linear(memory_in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3 * d_model),
        )
        self.cloud_memory_write = nn.Sequential(
            nn.LayerNorm(memory_in_dim),
            nn.Linear(memory_in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3 * d_model),
        )
        self.protein_x0_decode = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.graph_x0_decode = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.cloud_x0_decode = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.frame_diffusion_input = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.mold_from_x0 = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.protein_frame_update = EquivariantBackboneUpdate(
            scalar_dim=4 * d_model,
            vector_channels=self.protein.vector_channels,
            hidden_dim=d_model,
            step_scale=0.1,
        )
        self.backbone_cloud_proj = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.rebuilt_cloud_fuse = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.coord_frame_diffusion = nn.Sequential(
            nn.LayerNorm(6 * d_model),
            nn.Linear(6 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )
        self.coord_y_proj = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.y0_fuse = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.gates = nn.Linear(3 * d_model, 4 * d_model)
        self.mold_gate = nn.Linear(3 * d_model, d_model)
        self.norm_h = nn.LayerNorm(d_model)
        self.norm_c = nn.LayerNorm(d_model)
        self.norm_x0 = nn.LayerNorm(d_model)
        self.norm_y0 = nn.LayerNorm(d_model)

    def _cloud_for_encoding(self, cloud_grid):
        if (not self.training) or float(CLOUD_LIGAND_DROPOUT) <= 0.0:
            return cloud_grid
        drop = (
            torch.rand((cloud_grid.shape[0], 1, 1, 1, 1), device=cloud_grid.device)
            < float(CLOUD_LIGAND_DROPOUT)
        ).to(dtype=cloud_grid.dtype)
        protein = cloud_grid[:, 0:1]
        ligand = cloud_grid[:, 1:2] * (1.0 - drop)
        return torch.cat([protein, ligand], dim=1)

    def _encode_protein_memory(
        self,
        protein_bb,
        protein_bb_mask,
        protein_seq_ids,
        protein_full_mask,
        protein_chain_pair_mask,
        protein_temporal_state,
        h_prev,
        c_prev,
    ):
        protein_mask = protein_full_mask * protein_bb_mask
        res_h, protein_z, res_v, protein_temporal_state = self.protein(
            protein_seq_ids,
            protein_mask,
            protein_bb,
            temporal_state=protein_temporal_state,
            chain_pair_mask=protein_chain_pair_mask,
        )
        res_h = _finite_tensor(res_h)
        protein_z = _finite_tensor(protein_z)
        res_v = _finite_tensor(res_v)
        protein_temporal_state = _finite_tensor(protein_temporal_state)
        memory, x0 = self._modality_memory_x0(
            self.protein_memory_write,
            self.protein_x0_decode,
            protein_z,
            h_prev,
            c_prev,
        )
        return protein_z, res_h, res_v, protein_mask, protein_temporal_state, memory, x0

    def _encode_cloud_memory(self, cloud_grid, h_prev, c_prev):
        if cloud_grid.shape[1] != int(CLOUD_CHANNELS):
            raise ValueError(f"expected {int(CLOUD_CHANNELS)} cloud channels, got {cloud_grid.shape[1]}")
        cloud_grid = self._cloud_for_encoding(cloud_grid)
        cloud_raw = _finite_tensor(self.cloud(cloud_grid, return_tokens=False))
        cloud_z = _finite_tensor(self.cloud_proj(cloud_raw))
        memory, x0 = self._modality_memory_x0(
            self.cloud_memory_write,
            self.cloud_x0_decode,
            cloud_z,
            h_prev,
            c_prev,
        )
        return cloud_z, cloud_raw, memory, x0

    def _encode_graph_memory(
        self,
        coords,
        node_feat,
        node_mask,
        node_is_ligand,
        edge_index,
        k_idx,
        l_idx,
        node_batch,
        cloud_raw,
        h_prev,
        c_prev,
        velocity=None,
        frame_dt=None,
        boxes=None,
    ):
        B, N = coords.shape[:2]
        flat_coords = coords.reshape(B * N, 3)
        flat_feat = node_feat.reshape(B * N, node_feat.shape[-1])
        flat_lig = node_is_ligand.reshape(B * N, 1).to(dtype=coords.dtype)
        anchor_mask = node_mask.bool() & ~node_is_ligand.bool()
        anchor_mask = torch.where(
            anchor_mask.any(dim=1, keepdim=True),
            anchor_mask,
            node_mask.bool(),
        )
        anchor_idx = anchor_mask.to(dtype=torch.int64).argmax(dim=1)
        anchor = torch.gather(coords, 1, anchor_idx[:, None, None].expand(-1, 1, 3))
        relative_coords = coords - anchor
        if boxes is not None:
            relative_coords = minimum_image_displacement_batched(relative_coords, boxes)
        flat_relative_coords = relative_coords.reshape(B * N, 3)
        if velocity is None:
            velocity = torch.zeros_like(coords)
        flat_velocity = velocity.reshape(B * N, 3).to(dtype=coords.dtype)
        if frame_dt is None:
            frame_dt = coords.new_ones((B,))
        flat_dt = frame_dt.reshape(B, 1, 1).expand(-1, N, -1).reshape(B * N, 1).to(dtype=coords.dtype)
        graph_in = _finite_tensor(
            torch.cat([flat_feat, flat_relative_coords, flat_velocity, flat_lig, flat_dt], dim=-1)
        )
        edge_attr = _finite_tensor(
            edge_attr_from_coords(
                flat_coords,
                edge_index,
                k_idx,
                l_idx,
                node_batch=node_batch,
                boxes=boxes,
            )
        )
        node_h = self.gnn(
            graph_in,
            edge_index,
            edge_attr,
            node_batch=node_batch,
            cloud_feat=cloud_raw,
        ).reshape(B, N, -1)
        node_h = _finite_tensor(self.node_out(_finite_tensor(node_h))) * node_mask.to(dtype=node_h.dtype).unsqueeze(-1)
        graph_z = _finite_tensor(_masked_mean(node_h, node_mask, dim=1))
        memory, x0 = self._modality_memory_x0(
            self.graph_memory_write,
            self.graph_x0_decode,
            graph_z,
            h_prev,
            c_prev,
        )
        return graph_z, node_h, memory, x0

    def encode_frame(
        self,
        coords,
        protein_bb,
        protein_bb_mask,
        cloud_grid,
        node_feat,
        node_mask,
        node_is_ligand,
        edge_index,
        k_idx,
        l_idx,
        node_batch,
        protein_seq_ids,
        protein_full_mask,
        protein_chain_pair_mask=None,
        protein_temporal_state=None,
        h_prev=None,
        c_prev=None,
        velocity=None,
        protein_velocity=None,
        frame_dt=None,
        boxes=None,
    ):
        B = coords.shape[0]
        if h_prev is None:
            h_prev = coords.new_zeros((B, self.d_model))
        if c_prev is None:
            c_prev = torch.zeros_like(h_prev)
        h_prev = _finite_tensor(h_prev)
        c_prev = _finite_tensor(c_prev)

        (
            protein_z,
            res_h,
            res_v,
            protein_mask,
            protein_temporal_state,
            protein_memory,
            protein_x0,
        ) = self._encode_protein_memory(
            protein_bb,
            protein_bb_mask,
            protein_seq_ids,
            protein_full_mask,
            protein_chain_pair_mask,
            protein_temporal_state,
            h_prev,
            c_prev,
        )
        cloud_z, cloud_raw, cloud_memory, cloud_x0 = self._encode_cloud_memory(cloud_grid, h_prev, c_prev)
        graph_z, node_h, graph_memory, graph_x0 = self._encode_graph_memory(
            coords,
            node_feat,
            node_mask,
            node_is_ligand,
            edge_index,
            k_idx,
            l_idx,
            node_batch,
            cloud_raw,
            h_prev,
            c_prev,
            velocity=velocity,
            frame_dt=frame_dt,
            boxes=boxes,
        )

        x0_matrix = torch.stack([protein_x0, graph_x0, cloud_x0], dim=1)
        x0_concat = _finite_tensor(torch.cat([protein_x0, graph_x0, cloud_x0], dim=-1))
        frame_diffusion_input = _finite_tensor(self.frame_diffusion_input(x0_concat))
        mold_z = _finite_tensor(self.mold_from_x0(x0_concat))
        gate_latent_loss = _finite_loss(
            (
                F.smooth_l1_loss(protein_x0, protein_z.detach())
                + F.smooth_l1_loss(graph_x0, graph_z.detach())
                + F.smooth_l1_loss(cloud_x0, cloud_z.detach())
            )
            / 3.0
        )
        encoder_aux_loss = _finite_loss((
            frame_diffusion_input.float().pow(2).mean()
            + mold_z.float().pow(2).mean()
            + node_h.float().pow(2).mean()
            + res_h.float().pow(2).mean()
        ).to(dtype=gate_latent_loss.dtype) * 1.0e-4)
        enc = {
            "frame_z": frame_diffusion_input,
            "mold_z": mold_z,
            "node_h": node_h,
            "res_h": res_h,
            "res_v": res_v,
            "cloud_z": cloud_z,
            "protein_z": protein_z,
            "graph_z": graph_z,
            "protein_mask": protein_mask,
            "x0": frame_diffusion_input,
            "x0_matrix": x0_matrix,
            "x0_concat": x0_concat,
            "frame_diffusion_input": frame_diffusion_input,
            "input_coords": coords,
            "input_velocity": torch.zeros_like(coords) if velocity is None else velocity,
            "input_protein_bb": protein_bb,
            "input_protein_velocity": torch.zeros_like(protein_bb) if protein_velocity is None else protein_velocity,
            "frame_dt": coords.new_ones((B,)) if frame_dt is None else frame_dt,
            "boxes": boxes,
            "input_protein_mask": protein_bb_mask,
            "protein_chain_pair_mask": protein_chain_pair_mask,
            "protein_temporal_state": protein_temporal_state,
            "node_mask": node_mask,
            "protein_memory": protein_memory,
            "graph_memory": graph_memory,
            "cloud_memory": cloud_memory,
            "gate_latent_loss": gate_latent_loss,
            "encoder_aux_loss": encoder_aux_loss,
        }
        return enc

    def _modality_memory_x0(self, writer, decoder, modality_z, h_prev, c_prev):
        write, forget, candidate = writer(torch.cat([modality_z, h_prev, c_prev], dim=-1)).chunk(3, dim=-1)
        write = torch.sigmoid(write)
        forget = torch.sigmoid(forget)
        candidate = torch.tanh(candidate)
        memory = _finite_tensor(forget * c_prev + write * candidate)
        x0 = _finite_tensor(self.norm_x0(modality_z + decoder(memory)))
        return memory, x0

    def _frame_diffusion_y0(self, enc, h_prev):
        frame_x0 = _finite_tensor(enc["frame_diffusion_input"])
        protein_x0 = _finite_tensor(enc["x0_matrix"][:, 0])
        graph_x0 = _finite_tensor(enc["x0_matrix"][:, 1])
        cloud_x0 = _finite_tensor(enc["x0_matrix"][:, 2])
        source_bb = _finite_tensor(enc["input_protein_bb"])
        source_coords = _finite_tensor(enc["input_coords"])
        source_velocity = _finite_tensor(enc["input_velocity"])
        source_bb_velocity = _finite_tensor(enc["input_protein_velocity"])
        frame_dt = _finite_tensor(enc["frame_dt"]).reshape(-1)
        protein_mask = enc["input_protein_mask"].to(device=source_bb.device, dtype=torch.bool)
        node_mask = enc["node_mask"].to(device=source_coords.device, dtype=torch.bool)

        B, R = source_bb.shape[:2]
        protein_ctx = torch.cat(
            [
                enc["res_h"],
                protein_x0[:, None].expand(-1, R, -1),
                frame_x0[:, None].expand(-1, R, -1),
                h_prev[:, None].expand(-1, R, -1),
            ],
            dim=-1,
        )
        bb_delta = self.protein_frame_update(
            protein_ctx,
            enc["res_v"],
            source_bb,
            protein_mask,
        )
        bb_dt = frame_dt[:, None, None, None]
        pred_bb = _finite_tensor(
            source_bb + source_bb_velocity * bb_dt + 0.5 * bb_delta * bb_dt.square()
        )
        pred_bb = torch.where(protein_mask[:, :, None, None], pred_bb, source_bb)
        pred_bb = _project_backbone_chain_geometry(
            pred_bb,
            protein_mask,
            chain_pair_mask=enc.get("protein_chain_pair_mask"),
        )
        pred_bb = torch.where(protein_mask[:, :, None, None], pred_bb, source_bb)

        n_vec = pred_bb[..., 0, :] - pred_bb[..., 1, :]
        c_vec = pred_bb[..., 2, :] - pred_bb[..., 1, :]
        n_len = torch.linalg.norm(n_vec, dim=-1)
        c_len = torch.linalg.norm(c_vec, dim=-1)
        n_unit = n_vec / n_len.unsqueeze(-1).clamp_min(1.0e-8)
        c_unit = c_vec / c_len.unsqueeze(-1).clamp_min(1.0e-8)
        bb_invariants = torch.stack(
            [n_len, c_len, (n_unit * c_unit).sum(dim=-1), n_len * c_len, (n_len - c_len).abs()],
            dim=-1,
        )
        bb_tokens = self.backbone_cloud_proj(bb_invariants)
        protein_y = _finite_tensor(_masked_mean(bb_tokens, protein_mask, dim=1))
        rebuilt_cloud_z = _finite_tensor(self.rebuilt_cloud_fuse(torch.cat([protein_y, cloud_x0], dim=-1)))

        N = source_coords.shape[1]
        coord_ctx = torch.cat(
            [
                enc["node_h"],
                graph_x0[:, None].expand(-1, N, -1),
                cloud_x0[:, None].expand(-1, N, -1),
                rebuilt_cloud_z[:, None].expand(-1, N, -1),
                frame_x0[:, None].expand(-1, N, -1),
                h_prev[:, None].expand(-1, N, -1),
            ],
            dim=-1,
        )
        coord_acceleration = float(MD_INTEGRATOR_ACCEL_SCALE) * self.coord_frame_diffusion(coord_ctx)
        coord_dt = frame_dt[:, None, None]
        pred_coords = _finite_tensor(
            source_coords + source_velocity * coord_dt + 0.5 * coord_acceleration * coord_dt.square()
        )
        pred_coords = torch.where(node_mask[:, :, None], pred_coords, source_coords)
        pred_motion = pred_coords - source_coords
        if enc.get("boxes") is not None:
            pred_motion = minimum_image_displacement_batched(pred_motion, enc["boxes"])
        graph_y = _finite_tensor(_masked_mean(self.coord_y_proj(pred_motion), node_mask, dim=1))

        y0 = self.norm_y0(
            _finite_tensor(
                frame_x0
                + self.y0_fuse(torch.cat([protein_y, graph_y, rebuilt_cloud_z, cloud_x0], dim=-1))
            )
        )
        enc["pred_protein_bb"] = pred_bb
        enc["pred_coords"] = pred_coords
        enc["pred_coord_acceleration"] = coord_acceleration
        enc["rebuilt_cloud_z"] = rebuilt_cloud_z
        enc["protein_y"] = protein_y
        enc["graph_y"] = graph_y
        enc["y0"] = y0
        return y0

    def forward(self, enc, h_prev, c_prev):
        h_prev = _finite_tensor(h_prev)
        c_prev = _finite_tensor(c_prev)
        enc["frame_z"] = _finite_tensor(enc["frame_z"])
        enc["mold_z"] = _finite_tensor(enc["mold_z"])
        enc["node_h"] = _finite_tensor(enc["node_h"])
        enc["res_h"] = _finite_tensor(enc["res_h"])
        frame_diffusion_input = _finite_tensor(enc["frame_diffusion_input"])
        protein_memory = _finite_tensor(enc["protein_memory"])
        graph_memory = _finite_tensor(enc["graph_memory"])
        cloud_memory = _finite_tensor(enc["cloud_memory"])
        y0 = self._frame_diffusion_y0(enc, h_prev)
        gate_z = _finite_tensor(y0 + float(LSTM_PROTEIN_CONTEXT_WEIGHT) * protein_memory)
        h_context = _finite_tensor(h_prev + self.state_mixer(h_prev))
        x = _finite_tensor(torch.cat([gate_z, enc["mold_z"], h_context], dim=-1))
        i, f, o, g = self.gates(x).chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        mold_update = torch.sigmoid(self.mold_gate(x)) * torch.tanh(enc["mold_z"])
        c = _finite_tensor(self.norm_c(f * c_prev + i * g + mold_update))
        h = _finite_tensor(self.norm_h(o * torch.tanh(c)))
        h = _finite_tensor(
            self.norm_h(
                h
                + float(LSTM_PROTEIN_CONTEXT_WEIGHT) * protein_memory
                + 0.5 * float(LSTM_PROTEIN_CONTEXT_WEIGHT) * (graph_memory + cloud_memory)
            )
        )
        enc["gate_z"] = gate_z
        return h, c, enc
