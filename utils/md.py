import math

import torch


def box_to_matrix(box, coord_scale=1.0):
    """Convert [lx,ly,lz], [lx,ly,lz,alpha,beta,gamma], or 3x3 boxes."""
    if box is None:
        return None
    box = torch.as_tensor(box, dtype=torch.float32)
    if box.shape[-2:] == (3, 3):
        return box / float(coord_scale)
    if box.shape[-1] == 3:
        return torch.diag_embed(box / float(coord_scale))
    if box.shape[-1] != 6:
        raise ValueError(f"unsupported periodic box shape: {tuple(box.shape)}")

    lengths = box[..., :3] / float(coord_scale)
    angles = torch.deg2rad(box[..., 3:6])
    alpha, beta, gamma = angles.unbind(dim=-1)
    a, b, c = lengths.unbind(dim=-1)
    sin_gamma = torch.sin(gamma).clamp_min(1.0e-7)
    ax = a
    bx = b * torch.cos(gamma)
    by = b * sin_gamma
    cx = c * torch.cos(beta)
    cy = c * (torch.cos(alpha) - torch.cos(beta) * torch.cos(gamma)) / sin_gamma
    cz = torch.sqrt((c.square() - cx.square() - cy.square()).clamp_min(1.0e-12))
    zero = torch.zeros_like(ax)
    row_a = torch.stack([ax, zero, zero], dim=-1)
    row_b = torch.stack([bx, by, zero], dim=-1)
    row_c = torch.stack([cx, cy, cz], dim=-1)
    return torch.stack([row_a, row_b, row_c], dim=-2)


def valid_box(box):
    if box is None or box.shape[-2:] != (3, 3):
        return False
    if not bool(torch.isfinite(box).all()):
        return False
    return bool(torch.linalg.det(box.float()).abs().item() > 1.0e-8)


def minimum_image_displacement(delta, box):
    """Apply the triclinic minimum-image convention to displacement vectors."""
    if not valid_box(box):
        return delta
    work = delta.float()
    matrix = box.to(device=delta.device, dtype=torch.float32)
    frac = torch.matmul(work, torch.linalg.inv(matrix))
    frac = frac - torch.round(frac)
    return torch.matmul(frac, matrix).to(dtype=delta.dtype)


def minimum_image_displacement_batched(delta, boxes):
    """Minimum-image displacement where the leading axis indexes boxes."""
    if boxes is None:
        return delta
    matrices = boxes.to(device=delta.device, dtype=torch.float32)
    if matrices.ndim != 3 or matrices.shape[-2:] != (3, 3):
        return delta
    det = torch.linalg.det(matrices)
    valid = torch.isfinite(matrices).all(dim=(-2, -1)) & torch.isfinite(det) & (det.abs() > 1.0e-8)
    eye = torch.eye(3, device=delta.device, dtype=torch.float32).expand_as(matrices)
    safe = torch.where(valid[:, None, None], matrices, eye)
    inverse = torch.linalg.inv(safe)
    work = delta.float()
    frac = torch.einsum("t...i,tij->t...j", work, inverse)
    frac = frac - torch.round(frac)
    wrapped = torch.einsum("t...i,tij->t...j", frac, safe)
    shape = (valid.shape[0],) + (1,) * (delta.ndim - 1)
    return torch.where(valid.reshape(shape), wrapped, work).to(dtype=delta.dtype)


def _make_bonded_components_whole(coords, box, bonds):
    if bonds is None or not valid_box(box):
        return coords.clone()
    bonds = torch.as_tensor(bonds, device=coords.device, dtype=torch.long)
    if bonds.ndim != 2 or bonds.shape[-1] < 2:
        return coords.clone()
    atom_count = int(coords.shape[0])
    adjacency = [[] for _ in range(atom_count)]
    for a, b in bonds[:, :2].detach().cpu().tolist():
        if 0 <= int(a) < atom_count and 0 <= int(b) < atom_count and int(a) != int(b):
            adjacency[int(a)].append(int(b))
            adjacency[int(b)].append(int(a))
    out = coords.clone()
    visited = [False] * atom_count
    for seed in range(atom_count):
        if visited[seed]:
            continue
        visited[seed] = True
        stack = [seed]
        while stack:
            atom = stack.pop()
            for neighbor in adjacency[atom]:
                if visited[neighbor]:
                    continue
                step = minimum_image_displacement(coords[neighbor] - coords[atom], box)
                out[neighbor] = out[atom] + step
                visited[neighbor] = True
                stack.append(neighbor)
    return out


def unwrap_trajectory(coords, boxes, bonds=None):
    """Unwrap atom trajectories by accumulating minimum-image frame increments.

    ``coords`` is ``(T,N,3)`` and ``boxes`` may be any representation accepted
    by :func:`box_to_matrix`.  The operation is deliberately per atom: bonded
    geometry is retained while atoms crossing a periodic face no longer create
    artificial one-box jumps in velocity or diffusion targets.
    """
    coords = torch.as_tensor(coords)
    if coords.ndim != 3 or coords.shape[-1] != 3 or coords.shape[0] < 2 or boxes is None:
        return coords.clone()
    matrices = box_to_matrix(boxes, coord_scale=1.0).to(device=coords.device, dtype=coords.dtype)
    if matrices.ndim == 2:
        matrices = matrices.unsqueeze(0).expand(coords.shape[0], -1, -1)
    if matrices.ndim != 3 or matrices.shape[0] < coords.shape[0]:
        return coords.clone()
    out = torch.empty_like(coords)
    out[0] = _make_bonded_components_whole(coords[0], matrices[0], bonds)
    for frame_idx in range(1, int(coords.shape[0])):
        step = minimum_image_displacement(coords[frame_idx] - coords[frame_idx - 1], matrices[frame_idx - 1])
        out[frame_idx] = out[frame_idx - 1] + step
    return torch.nan_to_num(out)


def infer_time_scale(times):
    times = torch.as_tensor(times, dtype=torch.float32).reshape(-1)
    if times.numel() < 2:
        return times.new_tensor(1.0)
    dt = times[1:] - times[:-1]
    dt = dt[torch.isfinite(dt) & (dt > 0)]
    if dt.numel() == 0:
        return times.new_tensor(1.0)
    return dt.median().clamp_min(1.0e-8)


def sequence_velocities(coords, times, boxes=None):
    """Backward finite-difference velocities; the first frame uses forward difference."""
    coords = torch.as_tensor(coords)
    times = torch.as_tensor(times, device=coords.device, dtype=coords.dtype).reshape(-1)
    if coords.shape[0] < 2:
        return torch.zeros_like(coords)
    velocities = torch.zeros_like(coords)
    for t in range(1, coords.shape[0]):
        delta = coords[t] - coords[t - 1]
        if boxes is not None:
            delta = minimum_image_displacement(delta, boxes[t - 1])
        dt = (times[t] - times[t - 1]).abs().clamp_min(1.0e-8)
        velocities[t] = delta / dt
    velocities[0] = velocities[1]
    return torch.nan_to_num(velocities)


def _radius_edges_one(coords, valid_nodes, cutoff, max_neighbors, chunk_size, box=None):
    valid_idx = valid_nodes.nonzero(as_tuple=False).flatten()
    if valid_idx.numel() <= 1:
        return torch.zeros((2, 0), device=coords.device, dtype=torch.long)
    dst_coords = coords.index_select(0, valid_idx)
    src_parts = []
    dst_parts = []
    cutoff2 = float(cutoff) ** 2
    k = min(max(1, int(max_neighbors)), max(1, int(valid_idx.numel()) - 1))
    for start in range(0, int(valid_idx.numel()), max(1, int(chunk_size))):
        src_idx = valid_idx[start:start + max(1, int(chunk_size))]
        delta = dst_coords[None, :, :] - coords.index_select(0, src_idx)[:, None, :]
        if box is not None:
            delta = minimum_image_displacement(delta, box)
        dist2 = delta.square().sum(dim=-1)
        same = src_idx[:, None] == valid_idx[None, :]
        dist2 = dist2.masked_fill(same | (dist2 <= 1.0e-12) | (dist2 >= cutoff2), float("inf"))
        values, local = torch.topk(dist2, k=k, dim=1, largest=False)
        keep = torch.isfinite(values)
        if bool(keep.any()):
            src_parts.append(src_idx[:, None].expand_as(local)[keep])
            dst_parts.append(valid_idx[local[keep]])
    if not src_parts:
        return torch.zeros((2, 0), device=coords.device, dtype=torch.long)
    return torch.stack([torch.cat(src_parts), torch.cat(dst_parts)], dim=0)


def neighbor_reference_indices(edge_index, num_nodes):
    """Choose a deterministic alternate neighbor for edge angles/dihedrals."""
    if edge_index.numel() == 0:
        empty = torch.zeros((0,), device=edge_index.device, dtype=torch.long)
        return empty, empty
    src, dst = edge_index
    sentinel = int(num_nodes)
    first = torch.full((num_nodes,), sentinel, device=edge_index.device, dtype=torch.long)
    first.scatter_reduce_(0, src, dst, reduce="amin", include_self=True)
    second_values = torch.where(dst == first[src], torch.full_like(dst, sentinel), dst)
    second = torch.full_like(first, sentinel)
    second.scatter_reduce_(0, src, second_values, reduce="amin", include_self=True)
    k_idx = torch.where(first[src] != dst, first[src], second[src])
    l_idx = torch.where(first[dst] != src, first[dst], second[dst])
    k_idx = torch.where(k_idx < sentinel, k_idx, torch.full_like(k_idx, -1))
    l_idx = torch.where(l_idx < sentinel, l_idx, torch.full_like(l_idx, -1))
    return k_idx, l_idx


def dynamic_batched_graph(
    coords,
    node_mask,
    bond_edge_index=None,
    boxes=None,
    cutoff=0.45,
    max_neighbors=32,
    chunk_size=256,
):
    """Rebuild non-bonded neighbors for each frame and retain covalent edges."""
    batch_size, num_nodes = coords.shape[:2]
    edge_parts = []
    for b in range(batch_size):
        box = None if boxes is None else boxes[b]
        local = _radius_edges_one(
            coords[b],
            node_mask[b].bool(),
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            chunk_size=chunk_size,
            box=box,
        )
        if local.numel() > 0:
            edge_parts.append(local + b * num_nodes)
    if bond_edge_index is not None and bond_edge_index.numel() > 0:
        bond = bond_edge_index.to(device=coords.device, dtype=torch.long)
        edge_parts.append(torch.cat([bond, bond.flip(0)], dim=1))
    if not edge_parts:
        edge_index = torch.zeros((2, 0), device=coords.device, dtype=torch.long)
    else:
        edge_index = torch.cat(edge_parts, dim=1)
        keys = edge_index[0] * (batch_size * num_nodes) + edge_index[1]
        unique = torch.unique(keys, sorted=False)
        edge_index = torch.stack(
            [torch.div(unique, batch_size * num_nodes, rounding_mode="floor"), unique % (batch_size * num_nodes)],
            dim=0,
        )
    k_idx, l_idx = neighbor_reference_indices(edge_index, batch_size * num_nodes)
    return edge_index, k_idx, l_idx


def edge_displacements(coords_flat, edge_index, node_batch, boxes=None):
    src, dst = edge_index
    delta = coords_flat[dst] - coords_flat[src]
    if boxes is None or edge_index.numel() == 0:
        return delta
    out = delta.clone()
    edge_batch = node_batch[src]
    for b in range(int(boxes.shape[0])):
        mask = edge_batch == b
        if bool(mask.any()):
            out[mask] = minimum_image_displacement(delta[mask], boxes[b])
    return out


def _canonical_pairs(edge_index, total_nodes):
    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros((2, 0), device=edge_index.device if torch.is_tensor(edge_index) else "cpu", dtype=torch.long)
    lo = torch.minimum(edge_index[0], edge_index[1])
    hi = torch.maximum(edge_index[0], edge_index[1])
    keep = lo != hi
    keys = lo[keep] * int(total_nodes) + hi[keep]
    keys = torch.unique(keys)
    return torch.stack([torch.div(keys, int(total_nodes), rounding_mode="floor"), keys % int(total_nodes)], dim=0)


@torch.no_grad()
def _energy_pairs(coords, node_mask, boxes, cutoff, chunk_size=128):
    """All unordered valid pairs within cutoff, including coincident atoms.

    Unlike a top-k message graph this does not change discontinuously when two
    neighbors exchange rank. Pair membership changes only where the switched
    potential and its first two derivatives vanish.
    """
    batch_size, num_nodes = coords.shape[:2]
    parts = []
    for b in range(batch_size):
        ids = node_mask[b].bool().nonzero(as_tuple=False).flatten()
        for start in range(0, ids.numel(), chunk_size):
            src = ids[start:start + chunk_size]
            delta = coords[b, ids][None] - coords[b, src][:, None]
            if boxes is not None:
                delta = minimum_image_displacement(delta, boxes[b])
            keep = (delta.square().sum(-1) < float(cutoff) ** 2) & (src[:, None] < ids[None])
            left, right = keep.nonzero(as_tuple=True)
            if left.numel():
                parts.append(torch.stack([src[left], ids[right]]) + b * num_nodes)
    return torch.cat(parts, dim=1) if parts else torch.empty((2, 0), device=coords.device, dtype=torch.long)


def conservative_acceleration(
    coords,
    reference_coords,
    node_mask,
    node_masses,
    node_feat,
    node_vdw_radius_A,
    bond_edge_index,
    boxes=None,
    reference_boxes=None,
    coord_scale=10.0,
    cutoff_A=10.0,
    bond_k=300.0,
    lj_epsilon=0.10,
    coulomb_constant=332.06371,
    dielectric=4.0,
    force_to_accel_A_ps2=418.4,
    switch_A=8.0,
    softcore_ratio=0.5,
    return_energy=False,
    node_is_ligand=None,
    return_interface_energy=False,
):
    """Reduced conservative acceleration and optional potential energies.

    Forces are the analytic ``-grad(U)`` of the returned smooth reduced
    potential, including its soft-core radius and cutoff switch derivatives.
    There is no independent distance, LJ-ratio, force or acceleration clipping.
    This is NOT a parameterized all-atom force field. ``interface_energy`` contains
    only non-bonded protein--ligand pairs, so it is not dominated by internal
    pocket or ligand energy.  Coordinates are normalized by ``coord_scale``;
    returned acceleration is in normalized coordinates / ps^2 and energies are
    in kcal/mol.
    """
    if not (0.0 <= float(switch_A) < float(cutoff_A)):
        raise ValueError("require 0 <= switch_A < cutoff_A")
    if float(softcore_ratio) <= 0 or float(coord_scale) <= 0 or float(dielectric) <= 0:
        raise ValueError("softcore_ratio, coord_scale and dielectric must be positive")
    # Evaluate physics outside half precision; retain float64 for gradient tests.
    coords = coords.to(dtype=torch.float64 if coords.dtype == torch.float64 else torch.float32)
    batch_size, num_nodes = coords.shape[:2]
    total_nodes = batch_size * num_nodes
    flat = coords.reshape(total_nodes, 3)
    ref = reference_coords.to(coords).reshape(total_nodes, 3).detach()
    masses = node_masses.reshape(total_nodes).clamp_min(1.0)
    charges = node_feat[..., -1].reshape(total_nodes)
    radii = node_vdw_radius_A.reshape(total_nodes).clamp_min(0.25)
    node_batch = torch.arange(batch_size, device=coords.device).repeat_interleave(num_nodes)
    forces = torch.zeros_like(flat)
    energy = coords.new_zeros((batch_size,))
    interface_energy = coords.new_zeros((batch_size,))
    ligand_flags = None
    if node_is_ligand is not None:
        ligand_flags = node_is_ligand.to(device=coords.device, dtype=torch.bool).reshape(total_nodes)

    bond_pairs = _canonical_pairs(bond_edge_index, total_nodes).to(coords.device)
    if bond_pairs.numel():
        valid = node_mask.reshape(-1).bool()
        keep = valid[bond_pairs[0]] & valid[bond_pairs[1]]
        keep &= node_batch[bond_pairs[0]] == node_batch[bond_pairs[1]]
        bond_pairs = bond_pairs[:, keep]
    if bond_pairs.numel() > 0:
        src, dst = bond_pairs
        delta = edge_displacements(flat, bond_pairs, node_batch, boxes)
        ref_delta = edge_displacements(
            ref,
            bond_pairs,
            node_batch,
            boxes if reference_boxes is None else reference_boxes,
        )
        delta_A = delta * float(coord_scale)
        r_A = torch.sqrt(delta_A.square().sum(-1) + 1.0e-12)
        r0_A = torch.sqrt((ref_delta * float(coord_scale)).square().sum(-1) + 1.0e-12)
        unit = delta_A / r_A[:, None]
        scalar = float(bond_k) * (r_A - r0_A)
        f_src = scalar[:, None] * unit
        forces.index_add_(0, src, f_src)
        forces.index_add_(0, dst, -f_src)
        energy.index_add_(0, node_batch[src], 0.5 * float(bond_k) * (r_A - r0_A).square())

    nonbond_pairs = _energy_pairs(coords, node_mask, boxes, float(cutoff_A) / float(coord_scale))
    if nonbond_pairs.numel() > 0:
        excluded_nonbond_keys = set()
        if bond_pairs.numel() > 0:
            bond_cpu = bond_pairs.detach().cpu().T.tolist()
            adjacency = {}
            for a, b in bond_cpu:
                a, b = int(a), int(b)
                excluded_nonbond_keys.add(min(a, b) * total_nodes + max(a, b))
                adjacency.setdefault(a, set()).add(b)
                adjacency.setdefault(b, set()).add(a)
            # Classical force fields exclude 1-3 pairs from LJ/Coulomb.  The
            # previous prior applied a strong non-bonded force across bond
            # angles and therefore fought the bonded geometry it was meant to
            # preserve.
            for center, neighbors in adjacency.items():
                ordered = sorted(neighbors)
                for i, left in enumerate(ordered):
                    for right in ordered[i + 1:]:
                        excluded_nonbond_keys.add(min(left, right) * total_nodes + max(left, right))
        pair_keys = nonbond_pairs[0] * total_nodes + nonbond_pairs[1]
        if excluded_nonbond_keys:
            keep = torch.tensor(
                [int(key) not in excluded_nonbond_keys for key in pair_keys.detach().cpu().tolist()],
                device=coords.device,
                dtype=torch.bool,
            )
            nonbond_pairs = nonbond_pairs[:, keep]
        if nonbond_pairs.numel() > 0:
            src, dst = nonbond_pairs
            delta = edge_displacements(flat, nonbond_pairs, node_batch, boxes)
            delta_A = delta * float(coord_scale)
            r_A = torch.sqrt(delta_A.square().sum(-1) + 1.0e-12)
            unit = delta_A / r_A[:, None]
            contact = (radii[src] + radii[dst]).clamp_min(0.5)
            sigma = contact / (2.0 ** (1.0 / 6.0))
            # rho=sqrt(r^2+(alpha*sigma)^2); differentiate the SAME potential.
            rho = torch.sqrt(r_A.square() + (float(softcore_ratio) * sigma).square())
            sr6 = (sigma / rho).pow(6)
            coul = float(coulomb_constant) * charges[src] * charges[dst] / float(dielectric)
            raw_energy = 4.0 * float(lj_epsilon) * (sr6.square() - sr6) + coul / rho
            d_raw_dr = (24.0 * float(lj_epsilon) * (sr6 - 2.0 * sr6.square()) / rho
                        - coul / rho.square()) * (r_A / rho)
            width = float(cutoff_A) - float(switch_A)
            x = ((r_A - float(switch_A)) / width).clamp(0.0, 1.0)
            switch = 1.0 - 10.0 * x.pow(3) + 15.0 * x.pow(4) - 6.0 * x.pow(5)
            d_switch = (-30.0 * x.square() + 60.0 * x.pow(3) - 30.0 * x.pow(4)) / width
            f_src = (switch * d_raw_dr + d_switch * raw_energy)[:, None] * unit
            forces.index_add_(0, src, f_src)
            forces.index_add_(0, dst, -f_src)
            pair_energy = switch * raw_energy
            energy.index_add_(0, node_batch[src], pair_energy.to(energy.dtype))
            if ligand_flags is not None:
                is_interface = ligand_flags[src] != ligand_flags[dst]
                if bool(is_interface.any()):
                    interface_energy.index_add_(
                        0,
                        node_batch[src[is_interface]],
                        pair_energy[is_interface].to(energy.dtype),
                    )

    acceleration = forces * (float(force_to_accel_A_ps2) / float(coord_scale)) / masses[:, None]
    acceleration = acceleration.reshape(batch_size, num_nodes, 3)
    # Do not silently turn an invalid potential into a different clipped force.
    if not bool(torch.isfinite(acceleration).all() & torch.isfinite(energy).all()):
        raise FloatingPointError("non-finite reduced energy/force; check coordinates and force-field inputs")
    if return_energy and return_interface_energy:
        return acceleration, energy, interface_energy
    if return_interface_energy:
        return acceleration, interface_energy
    if return_energy:
        return acceleration, energy
    return acceleration


def md_validation_metrics(
    pred,
    target,
    source,
    source_velocity,
    node_mask,
    future_times,
    source_time,
    coord_scale=10.0,
    node_masses=None,
    node_feat=None,
    node_vdw_radius_A=None,
    node_is_ligand=None,
    bond_edge_index=None,
    boxes=None,
    source_box=None,
    rdf_bins=32,
    rdf_max_A=10.0,
    time_to_ps=1.0,
    force_kwargs=None,
):
    """Cheap trajectory statistics: speed, MSD, and velocity autocorrelation."""
    dtype = pred.dtype
    dt = future_times - torch.cat([source_time[:, None], future_times[:, :-1]], dim=1)
    dt = dt.abs().clamp_min(1.0e-8) * max(float(time_to_ps), 1.0e-8)
    pred_delta = pred - torch.cat([source[:, None], pred[:, :-1]], dim=1)
    target_delta = target - torch.cat([source[:, None], target[:, :-1]], dim=1)
    if boxes is not None:
        if source_box is None:
            source_box = boxes[:, 0]
        transition_boxes = torch.cat([source_box[:, None], boxes[:, :-1]], dim=1)
        pred_delta_pbc = pred_delta.clone()
        target_delta_pbc = target_delta.clone()
        for b in range(pred.shape[0]):
            pred_delta_pbc[b] = minimum_image_displacement_batched(pred_delta[b], transition_boxes[b])
            target_delta_pbc[b] = minimum_image_displacement_batched(target_delta[b], transition_boxes[b])
        pred_delta = pred_delta_pbc
        target_delta = target_delta_pbc
    pred_v = pred_delta / dt[:, :, None, None]
    target_v = target_delta / dt[:, :, None, None]
    mask = node_mask[:, None, :, None].to(dtype=dtype)
    denom = mask.expand_as(pred_v).sum().clamp_min(1.0)
    speed_rmse = torch.sqrt((((pred_v - target_v).square() * mask).sum() / denom).clamp_min(0.0)) * float(coord_scale)

    # Cumulative minimum-image increments unwrap boundary crossings and retain
    # displacements larger than half a periodic box.
    pred_disp = torch.cumsum(pred_delta, dim=1)
    target_disp = torch.cumsum(target_delta, dim=1)
    node_denom = node_mask.to(dtype=dtype).sum(dim=1).clamp_min(1.0)
    pred_msd = (pred_disp.square().sum(dim=-1) * node_mask[:, None].to(dtype)).sum(dim=2) / node_denom[:, None]
    target_msd = (target_disp.square().sum(dim=-1) * node_mask[:, None].to(dtype)).sum(dim=2) / node_denom[:, None]
    msd_rmse = torch.sqrt((pred_msd - target_msd).square().mean().clamp_min(0.0)) * float(coord_scale) ** 2

    v0 = source_velocity[:, None] / max(float(time_to_ps), 1.0e-8)
    pred_vacf = (pred_v * v0).sum(dim=-1)
    target_vacf = (target_v * v0).sum(dim=-1)
    pred_vacf = (pred_vacf * node_mask[:, None].to(dtype)).sum(dim=2) / node_denom[:, None]
    target_vacf = (target_vacf * node_mask[:, None].to(dtype)).sum(dim=2) / node_denom[:, None]
    vacf_mae = (pred_vacf - target_vacf).abs().mean() * float(coord_scale) ** 2
    metrics = {
        "md_speed_rmse_A_per_ps": torch.nan_to_num(speed_rmse),
        "md_msd_rmse_A2": torch.nan_to_num(msd_rmse),
        "md_vacf_mae_A2_per_ps2": torch.nan_to_num(vacf_mae),
    }
    if node_masses is not None:
        masses = node_masses[:, None, :, None].to(dtype=dtype)
        # amu*(A/ps)^2 -> kcal/mol uses the reciprocal of the standard
        # 418.4 force-to-acceleration conversion.
        kinetic_pred = 0.5 * (masses * (pred_v * float(coord_scale)).square() * mask).sum(dim=(2, 3)) / 418.4
        kinetic_target = 0.5 * (masses * (target_v * float(coord_scale)).square() * mask).sum(dim=(2, 3)) / 418.4
        denom_ke = kinetic_target.abs().mean().clamp_min(1.0e-8)
        metrics["md_kinetic_relative_mae"] = torch.nan_to_num(
            (kinetic_pred - kinetic_target).abs().mean() / denom_ke
        )

    if node_masses is not None and node_feat is not None and node_vdw_radius_A is not None and bond_edge_index is not None:
        energies_pred = []
        energies_target = []
        for t in sorted(set([0, max(0, pred.shape[1] - 1)])):
            box_t = boxes[:, t] if boxes is not None else None
            _, potential_pred = conservative_acceleration(
                pred[:, t], source, node_mask, node_masses, node_feat, node_vdw_radius_A,
                bond_edge_index, boxes=box_t, reference_boxes=source_box,
                coord_scale=float(coord_scale), return_energy=True, **(force_kwargs or {}),
            )
            _, potential_target = conservative_acceleration(
                target[:, t], source, node_mask, node_masses, node_feat, node_vdw_radius_A,
                bond_edge_index, boxes=box_t, reference_boxes=source_box,
                coord_scale=float(coord_scale), return_energy=True, **(force_kwargs or {}),
            )
            kinetic_pred_t = kinetic_pred[:, t] if 'kinetic_pred' in locals() else potential_pred.new_zeros(potential_pred.shape)
            kinetic_target_t = kinetic_target[:, t] if 'kinetic_target' in locals() else potential_target.new_zeros(potential_target.shape)
            energies_pred.append(potential_pred + kinetic_pred_t)
            energies_target.append(potential_target + kinetic_target_t)
        if len(energies_pred) > 1:
            pred_drift = energies_pred[-1] - energies_pred[0]
            target_drift = energies_target[-1] - energies_target[0]
            scale = energies_target[0].abs().mean().clamp_min(1.0e-6)
            metrics["md_energy_drift_error_relative"] = torch.nan_to_num(
                (pred_drift - target_drift).abs().mean() / scale
            )

    if bond_edge_index is not None and bond_edge_index.numel() > 0:
        total_nodes = pred.shape[0] * pred.shape[2]
        pairs = _canonical_pairs(bond_edge_index.to(pred.device), total_nodes)
        bond_errors = []
        num_nodes = pred.shape[2]
        for b in range(pred.shape[0]):
            lo, hi = b * num_nodes, (b + 1) * num_nodes
            keep = (
                (pairs[0] >= lo) & (pairs[0] < hi)
                & (pairs[1] >= lo) & (pairs[1] < hi)
            )
            if not bool(keep.any().item()):
                continue
            src = pairs[0, keep] - lo
            dst = pairs[1, keep] - lo
            pred_bond_delta = pred[b, :, dst] - pred[b, :, src]
            target_bond_delta = target[b, :, dst] - target[b, :, src]
            if boxes is not None:
                pred_bond_delta = minimum_image_displacement_batched(pred_bond_delta, boxes[b])
                target_bond_delta = minimum_image_displacement_batched(target_bond_delta, boxes[b])
            pred_bond = torch.linalg.norm(pred_bond_delta, dim=-1)
            target_bond = torch.linalg.norm(target_bond_delta, dim=-1)
            bond_errors.append((pred_bond - target_bond).abs().mean())
        if bond_errors:
            metrics["md_bond_length_mae_A"] = torch.nan_to_num(
                torch.stack(bond_errors).mean() * float(coord_scale)
            )

    if node_is_ligand is not None:
        pred_hist = pred.new_zeros((int(rdf_bins),), dtype=torch.float32)
        target_hist = pred.new_zeros((int(rdf_bins),), dtype=torch.float32)
        for b in range(pred.shape[0]):
            ligand = (node_mask[b].bool() & node_is_ligand[b].bool()).nonzero(as_tuple=False).flatten()
            protein = (node_mask[b].bool() & ~node_is_ligand[b].bool()).nonzero(as_tuple=False).flatten()
            if ligand.numel() == 0 or protein.numel() == 0:
                continue
            for t in range(pred.shape[1]):
                for start in range(0, protein.numel(), 128):
                    ids = protein[start:start + 128]
                    pred_delta = pred[b, t, ids][None] - pred[b, t, ligand][:, None]
                    target_delta = target[b, t, ids][None] - target[b, t, ligand][:, None]
                    if boxes is not None:
                        pred_delta = minimum_image_displacement(pred_delta, boxes[b, t])
                        target_delta = minimum_image_displacement(target_delta, boxes[b, t])
                    pred_d = torch.linalg.norm(pred_delta, dim=-1) * float(coord_scale)
                    target_d = torch.linalg.norm(target_delta, dim=-1) * float(coord_scale)
                    pred_hist += torch.histc(pred_d.float(), bins=int(rdf_bins), min=0.0, max=float(rdf_max_A))
                    target_hist += torch.histc(target_d.float(), bins=int(rdf_bins), min=0.0, max=float(rdf_max_A))
        if float(pred_hist.sum().item()) > 0.0 and float(target_hist.sum().item()) > 0.0:
            pred_hist = pred_hist / pred_hist.sum()
            target_hist = target_hist / target_hist.sum()
            metrics["md_interface_rdf_l1"] = torch.nan_to_num((pred_hist - target_hist).abs().mean())
    return metrics
