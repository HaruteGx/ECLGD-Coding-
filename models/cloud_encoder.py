import torch
import torch.nn as nn

from config import (
    CLOUD_EGNN_LAYERS, CLOUD_GRID_SIZE, CLOUD_KNN, CLOUD_POINT_TOPK,
)


class E3CloudBlock(nn.Module):
    def __init__(self, d_hid):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.LayerNorm(2 * d_hid + 1),
            nn.Linear(2 * d_hid + 1, d_hid),
            nn.SiLU(),
            nn.Linear(d_hid, d_hid),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.LayerNorm(2 * d_hid),
            nn.Linear(2 * d_hid, d_hid),
            nn.SiLU(),
            nn.Linear(d_hid, d_hid),
        )
        self.coord_mlp = nn.Sequential(
            nn.LayerNorm(d_hid),
            nn.Linear(d_hid, d_hid),
            nn.SiLU(),
            nn.Linear(d_hid, 1),
            nn.Tanh(),
        )

    def forward(self, h, pos, density, knn):
        B, K, D = h.shape
        gather_idx = knn.unsqueeze(-1).expand(-1, -1, -1, D)
        h_j = torch.gather(h.unsqueeze(1).expand(-1, K, -1, -1), 2, gather_idx)
        pos_j = torch.gather(
            pos.unsqueeze(1).expand(-1, K, -1, -1),
            2,
            knn.unsqueeze(-1).expand(-1, -1, -1, 3),
        )
        h_i = h.unsqueeze(2).expand_as(h_j)
        diff = pos.unsqueeze(2) - pos_j
        dist2 = (diff ** 2).sum(dim=-1, keepdim=True)
        msg = self.edge_mlp(torch.cat([h_i, h_j, dist2], dim=-1))
        density_j = torch.gather(
            density.unsqueeze(1).expand(-1, K, -1),
            2,
            knn,
        )
        weights = density_j.to(dtype=msg.dtype).clamp_min(0.0).unsqueeze(-1)
        weight_sum = weights.sum(dim=2, keepdim=True)
        uniform = torch.full_like(weights, 1.0 / float(max(1, weights.shape[2])))
        weights = torch.where(weight_sum > 1e-6, weights / weight_sum.clamp_min(1e-6), uniform)
        agg = (msg * weights).sum(dim=2)
        h = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        coord_gate = self.coord_mlp(msg) * weights
        pos = pos + 0.05 * (diff * coord_gate).sum(dim=2)
        return h, pos


class EquivariantCloudEncoder(nn.Module):
    """
    E(3)-style point cloud encoder for electron-density templates.

    The voxel grid is sparsified to its densest voxels, then processed as a
    local point cloud with distance-based messages and coordinate updates.
    """

    def __init__(self, d_out=128, in_channels=1, d_hid=128, topk=None, knn=None, n_layers=None):
        super().__init__()
        self.topk = int(CLOUD_POINT_TOPK if topk is None else topk)
        self.knn = int(CLOUD_KNN if knn is None else knn)
        self.n_layers = int(CLOUD_EGNN_LAYERS if n_layers is None else n_layers)
        self.in_channels = int(in_channels)
        self.feat_in = nn.Sequential(
            nn.LayerNorm(self.in_channels + 1),
            nn.Linear(self.in_channels + 1, d_hid),
            nn.SiLU(),
            nn.Linear(d_hid, d_hid),
        )
        self.blocks = nn.ModuleList([E3CloudBlock(d_hid) for _ in range(self.n_layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(d_hid),
            nn.Linear(d_hid, d_out),
            nn.SiLU(),
        )
        coords_1d = torch.linspace(-1.0, 1.0, int(CLOUD_GRID_SIZE), dtype=torch.float32)
        zz, yy, xx = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
        coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        self.register_buffer("voxel_coords", coords, persistent=False)

    def forward(self, grid, return_tokens=False):
        B = grid.shape[0]
        flat = grid.reshape(B, grid.shape[1], -1).transpose(1, 2)
        density = flat.sum(dim=-1).clamp_min(0.0)
        K = max(1, min(int(self.topk), flat.shape[1]))
        values, idx = torch.topk(density, k=K, dim=1, largest=True, sorted=False)
        feat = torch.gather(flat, 1, idx.unsqueeze(-1).expand(-1, -1, flat.shape[-1]))
        pos = self.voxel_coords.to(device=grid.device, dtype=grid.dtype)[idx]
        radius = torch.linalg.norm(pos, dim=-1, keepdim=True)
        h = self.feat_in(torch.cat([feat, radius], dim=-1))

        if K > 1 and self.blocks:
            k_neighbors = max(1, min(int(self.knn), K - 1))
            for block in self.blocks:
                d2 = torch.cdist(pos.float(), pos.float(), p=2).to(dtype=pos.dtype) ** 2
                eye = torch.eye(K, device=grid.device, dtype=torch.bool).unsqueeze(0)
                d2 = d2.masked_fill(eye, float("inf"))
                knn = torch.topk(d2, k=k_neighbors, dim=-1, largest=False, sorted=False).indices
                h, pos = block(h, pos, values, knn)

        weights = values.to(dtype=h.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = (h * weights.unsqueeze(-1)).sum(dim=1)
        out = self.head(pooled)
        if return_tokens:
            return out, h, pos, values
        return out
