import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.md import edge_displacements


def _safe_angle(u, w, eps=1e-7):
    cross = torch.cross(u, w, dim=-1)
    cross_norm = torch.sqrt((cross * cross).sum(dim=-1).clamp_min(float(eps) ** 2))
    dot = (u * w).sum(dim=-1)
    return torch.atan2(cross_norm, dot)


def _safe_dihedral(p0, p1, p2, p3, eps=1e-7):
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = torch.sqrt((b1 * b1).sum(dim=-1, keepdim=True).clamp_min(float(eps) ** 2))
    b1n = b1 / b1_norm

    v = b0 - (b0 * b1n).sum(dim=-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(dim=-1, keepdim=True) * b1n

    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1n, v, dim=-1) * w).sum(dim=-1)
    tiny = (x * x + y * y) < float(eps) ** 2
    x = torch.where(tiny, torch.full_like(x, float(eps)), x)
    y = torch.where(tiny, torch.zeros_like(y), y)
    return torch.atan2(y, x)


def edge_attr_from_coords(coords, edge_index, k_idx, l_idx, node_batch=None, boxes=None):
    """
    coords: (N,3) normalized
    edge_index: (2,E)
    k_idx/l_idx: (E,)  -1表示不存在
    return edge_attr: (E,4) = [bond_len, angle_at_src, angle_at_dst, dihedral]
    """
    src = edge_index[0]
    dst = edge_index[1]
    pi = coords[src]  # (E,3)
    pj = coords[dst]
    v = (
        edge_displacements(coords, edge_index, node_batch, boxes)
        if node_batch is not None and boxes is not None
        else pj - pi
    )
    bond = torch.linalg.norm(v, dim=-1).clamp_min(1e-9)  # (E,)

    # angle at src: k - i  with  j - i
    angle_i = torch.zeros_like(bond)
    mask_k = k_idx >= 0
    if mask_k.any():
        local_src = src[mask_k]
        local_k = k_idx[mask_k]
        u = (
            edge_displacements(coords, torch.stack([local_src, local_k]), node_batch, boxes)
            if node_batch is not None and boxes is not None
            else coords[local_k] - coords[local_src]
        )
        w = v[mask_k]
        angle_i[mask_k] = _safe_angle(u, w)

    # angle at dst: i - j with l - j
    angle_j = torch.zeros_like(bond)
    mask_l = l_idx >= 0
    if mask_l.any():
        local_dst = dst[mask_l]
        local_l = l_idx[mask_l]
        u = -v[mask_l]
        w = (
            edge_displacements(coords, torch.stack([local_dst, local_l]), node_batch, boxes)
            if node_batch is not None and boxes is not None
            else coords[local_l] - coords[local_dst]
        )
        angle_j[mask_l] = _safe_angle(u, w)

    # dihedral: k-i-j-l (only when both exist)
    dih = torch.zeros_like(bond)
    mask_kl = (k_idx >= 0) & (l_idx >= 0)
    if mask_kl.any():
        local_src = src[mask_kl]
        local_dst = dst[mask_kl]
        local_k = k_idx[mask_kl]
        local_l = l_idx[mask_kl]
        k_from_i = (
            edge_displacements(coords, torch.stack([local_src, local_k]), node_batch, boxes)
            if node_batch is not None and boxes is not None
            else coords[local_k] - coords[local_src]
        )
        j_from_i = v[mask_kl]
        l_from_j = (
            edge_displacements(coords, torch.stack([local_dst, local_l]), node_batch, boxes)
            if node_batch is not None and boxes is not None
            else coords[local_l] - coords[local_dst]
        )
        p1 = torch.zeros_like(j_from_i)
        p0 = k_from_i
        p2 = j_from_i
        p3 = j_from_i + l_from_j
        dih[mask_kl] = _safe_dihedral(p0, p1, p2, p3)

    return torch.nan_to_num(torch.stack([bond, angle_i, angle_j, dih], dim=-1), nan=0.0, posinf=0.0, neginf=0.0)  # (E,4)

class EdgeGNN(nn.Module):
    """
    Edge-based message passing GNN.
    扩展：支持每个图一个全局 cloud embedding（“电子云罩子”），通过 node_batch/edge_batch 广播进入消息。
    """
    def __init__(self, d_node_in=4, d_edge=4, d_hid=128, n_layers=3, d_cloud=0):
        super().__init__()
        self.d_cloud = int(d_cloud)

        node_in_dim = int(d_node_in) + (self.d_cloud if self.d_cloud > 0 else 0)
        self.node_in = nn.Linear(node_in_dim, d_hid)
        self.edge_in = nn.Linear(d_edge, d_hid)

        msg_in_dim = d_hid * 3 + (self.d_cloud if self.d_cloud > 0 else 0)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(msg_in_dim, d_hid),
                nn.ReLU(),
                nn.Linear(d_hid, d_hid),
                nn.ReLU(),
                nn.Linear(d_hid, d_hid),
            ) for _ in range(n_layers)
        ])
        self.out = nn.Linear(d_hid, d_hid)

    def forward(self, node_feat, edge_index, edge_attr, node_batch=None, cloud_feat=None):
        """
        node_feat: (N, d_node_in)
        edge_attr: (E, d_edge)
        node_batch: (N,)  每个node属于哪个图（用于广播cloud）
        cloud_feat: (B, d_cloud)  每个图一个全局条件向量
        """
        if (self.d_cloud > 0) and (cloud_feat is not None) and (node_batch is not None):
            cloud_node = cloud_feat[node_batch]  # (N, d_cloud)
            node_in = torch.cat([node_feat, cloud_node], dim=-1)
        else:
            cloud_node = None
            node_in = node_feat

        h = F.relu(self.node_in(node_in))         # (N,d)
        e = F.relu(self.edge_in(edge_attr))       # (E,d)
        src, dst = edge_index[0], edge_index[1]

        # edge-level cloud broadcast
        if (self.d_cloud > 0) and (cloud_feat is not None) and (node_batch is not None):
            edge_batch = node_batch[dst]          # (E,)
            cloud_edge = cloud_feat[edge_batch]   # (E, d_cloud)
        else:
            cloud_edge = None

        for mlp in self.layers:
            h_src = h[src]
            h_dst = h[dst]
            if cloud_edge is None:
                m_in = torch.cat([h_dst, h_src, e], dim=-1)
            else:
                m_in = torch.cat([h_dst, h_src, e, cloud_edge], dim=-1)
            m = mlp(m_in)
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, m)
            deg = torch.zeros((h.shape[0], 1), device=h.device, dtype=h.dtype)
            deg.index_add_(0, dst, torch.ones((dst.shape[0], 1), device=h.device, dtype=h.dtype))
            h = h + agg / deg.clamp_min(1.0)

        return F.relu(self.out(h))
