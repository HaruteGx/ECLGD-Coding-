import torch
import torch.nn.functional as F


def weighted_point_loss(pred, target, mask=None, weight=None):
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
        if weight is not None:
            weight = weight[mask]

    if pred.numel() == 0:
        return torch.zeros((), device=target.device if torch.is_tensor(target) else pred.device)

    sq = ((pred - target) ** 2).sum(dim=-1)
    if weight is None:
        return sq.mean()

    w = weight.to(device=sq.device, dtype=sq.dtype)
    return (sq * w).sum() / w.sum().clamp_min(1e-6)


def build_local_edge_mask(coords, edge_index, max_len):
    if edge_index.numel() == 0:
        return torch.zeros((0,), device=coords.device, dtype=torch.bool)
    src, dst = edge_index[0], edge_index[1]
    dist = torch.linalg.norm(coords[src] - coords[dst], dim=-1)
    return (src < dst) & (dist <= float(max_len))


def edge_length_loss(pred_coords, ref_coords, edge_index, edge_mask):
    if edge_mask is None or edge_mask.numel() == 0 or not bool(edge_mask.any()):
        return pred_coords.new_zeros(())

    src = edge_index[0][edge_mask]
    dst = edge_index[1][edge_mask]
    pred_len = torch.linalg.norm(pred_coords[src] - pred_coords[dst], dim=-1)
    ref_len = torch.linalg.norm(ref_coords[src] - ref_coords[dst], dim=-1)
    return F.smooth_l1_loss(pred_len, ref_len)


def ligand_interface_clash_loss(coords, node_batch, node_is_ligand, min_dist):
    if coords.numel() == 0 or node_batch.numel() == 0:
        return coords.new_zeros(())

    total = coords.new_zeros(())
    count = 0
    n_graphs = int(node_batch.max().item()) + 1
    for graph_id in range(n_graphs):
        idx = torch.nonzero(node_batch == graph_id, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue

        lig_idx = idx[node_is_ligand[idx]]
        prot_idx = idx[~node_is_ligand[idx]]
        if lig_idx.numel() == 0 or prot_idx.numel() == 0:
            continue

        dist = torch.cdist(coords[lig_idx], coords[prot_idx])
        penalty = F.relu(float(min_dist) - dist)
        total = total + (penalty ** 2).mean()
        count += 1

    if count == 0:
        return total
    return total / float(count)


def environment_clash_loss(coords, node_batch, env_coords=None, env_mask=None, min_dist=0.16):
    if env_coords is None or coords.numel() == 0 or node_batch.numel() == 0:
        return coords.new_zeros(())

    total = coords.new_zeros(())
    count = 0
    n_graphs = int(node_batch.max().item()) + 1
    for graph_id in range(n_graphs):
        idx = torch.nonzero(node_batch == graph_id, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue

        graph_env = env_coords[graph_id]
        if env_mask is not None:
            graph_env = graph_env[env_mask[graph_id]]
        if graph_env.numel() == 0:
            continue

        dist = torch.cdist(coords[idx], graph_env)
        penalty = F.relu(float(min_dist) - dist)
        total = total + (penalty ** 2).mean()
        count += 1

    if count == 0:
        return total
    return total / float(count)


def refine_pocket_coords(
    initial_coords,
    source_coords,
    edge_index,
    node_batch,
    node_is_ligand,
    anchor_target=None,
    anchor_mask=None,
    anchor_weight=None,
    env_coords=None,
    env_mask=None,
    bond_max_len=0.22,
    clash_min_dist=0.16,
    num_steps=0,
    lr=0.02,
    stay_weight=0.01,
    bond_weight=0.10,
    anchor_weight_scale=0.15,
    clash_weight=0.05,
):
    if int(num_steps) <= 0 or initial_coords.numel() == 0:
        return initial_coords

    with torch.enable_grad():
        x = initial_coords.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=float(lr))
        local_edge_mask = build_local_edge_mask(source_coords.detach(), edge_index, max_len=bond_max_len)
        x_ref = initial_coords.detach()

        for _ in range(int(num_steps)):
            optimizer.zero_grad(set_to_none=True)
            loss = x.new_zeros(())

            if local_edge_mask.numel() > 0 and bool(local_edge_mask.any()):
                loss = loss + float(bond_weight) * edge_length_loss(x, source_coords, edge_index, local_edge_mask)

            if anchor_target is not None and anchor_mask is not None:
                loss = loss + float(anchor_weight_scale) * weighted_point_loss(
                    x, anchor_target, mask=anchor_mask, weight=anchor_weight
                )

            loss = loss + float(clash_weight) * ligand_interface_clash_loss(
                x, node_batch, node_is_ligand, min_dist=clash_min_dist
            )

            if env_coords is not None:
                loss = loss + float(clash_weight) * environment_clash_loss(
                    x, node_batch, env_coords=env_coords, env_mask=env_mask, min_dist=clash_min_dist
                )

            loss = loss + float(stay_weight) * F.mse_loss(x, x_ref)

            if not torch.isfinite(loss):
                break

            loss.backward()
            optimizer.step()

        return x.detach()
