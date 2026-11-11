import torch
import torch.nn.functional as F

from config import (
    CLASH_DISTANCE_A,
    COORD_SCALE,
    DIFFUSION_ABSOLUTE_ANCHOR_LOSS_WEIGHT,
    DIFFUSION_INCREMENT_NOISE_LOSS_WEIGHT,
    DIFFUSION_INCREMENT_X0_LOSS_WEIGHT,
    DISTOGRAM_LOSS_WEIGHT,
    LATENT_LOSS_WEIGHT,
    QM_DENSITY_LOSS_WEIGHT,
    LIGAND_NEAR_RES_CUTOFF,
    LSTM_KEYFRAME_COUNT,
    MD_BOND_FORCE_CONSTANT,
    MD_CONSERVATIVE_LOSS_WEIGHT,
    MD_COULOMB_CONSTANT,
    MD_DIELECTRIC,
    MD_ENERGY_CONSISTENCY_LOSS_WEIGHT,
    MD_ENERGY_NORM_KCAL_PER_MOL_ATOM,
    MD_FORCE_ACCEL_NORM_A_PS2,
    MD_FORCE_SWITCH_A,
    MD_NONBONDED_SOFTCORE_RATIO,
    MD_FORCE_CONSISTENCY_LOSS_WEIGHT,
    MD_FORCE_CUTOFF_A,
    MD_FORCE_MAX_FRAMES,
    MD_FORCE_TO_ACCEL_A_PS2,
    MD_HIERARCHY_CLOUD_LOSS_WEIGHT,
    MD_INTERFACE_ENERGY_LOSS_WEIGHT,
    MD_INTERFACE_ENERGY_NORM_KCAL_PER_MOL_LIGAND_ATOM,
    MD_LJ_EPSILON_KCAL,
    MD_RDF_BINS,
    MD_RDF_MAX_A,
    MD_STORED_TIME_TO_PS,
    MD_VALIDATION_METRICS,
    OBSERVED_GRAD_FRAMES,
    PHYSICS_LOSS_WEIGHT,
    TRAJECTORY_BACKBONE_DIRECTION_WEIGHT,
    TRAJECTORY_BACKBONE_GEOMETRY_WEIGHT,
    TRAJECTORY_BACKBONE_SMOOTH_WEIGHT,
    TRAJECTORY_ACCELERATION_WEIGHT,
    TRAJECTORY_BOND_LENGTH_WEIGHT,
    TRAJECTORY_CA_CONTINUITY_WEIGHT,
    TRAJECTORY_CLASH_WEIGHT,
    TRAJECTORY_CONTACT_MAX_PAIRS,
    TRAJECTORY_HBOND_DISTANCE_A,
    TRAJECTORY_HBOND_WEIGHT,
    TRAJECTORY_HYDROPHOBIC_DISTANCE_A,
    TRAJECTORY_HYDROPHOBIC_WEIGHT,
    TRAJECTORY_PEPTIDE_BOND_WEIGHT,
    TRAJECTORY_VDW_REPULSION_WEIGHT,
    TRAJECTORY_VDW_SCALE,
    TRAJECTORY_VELOCITY_WEIGHT,
    TRAJ_STATIC_CONTEXT_PROB,
)
from utils.md import (
    conservative_acceleration,
    md_validation_metrics,
    minimum_image_displacement_batched,
)
from utils.pocket import dynamic_residue_pocket_mask
from .temporal_utils import _debug_enabled, _debug_every_frames, _debug_log, _debug_now, _masked_smooth_l1


def _finite_loss(loss):
    return torch.nan_to_num(loss, nan=0.0, posinf=1.0e4, neginf=0.0)


def _ddp_parameter_anchor(module, ref):
    anchor = ref.new_zeros(())
    for param in module.parameters():
        if param.requires_grad and param.numel() > 0:
            value = torch.nan_to_num(
                param.reshape(-1)[0].to(dtype=torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).to(dtype=ref.dtype)
            anchor = anchor + value * 0.0
    return anchor


def _pbc_displacement_bt(delta, boxes=None):
    """Apply a [B,T,3,3] box sequence to [B,T,...,3] displacements."""
    if boxes is None:
        return delta
    out = delta.clone()
    for b in range(delta.shape[0]):
        out[b] = minimum_image_displacement_batched(delta[b], boxes[b])
    return out


def _pbc_pairwise_distance(coords_a, coords_b, boxes=None):
    delta = coords_a[:, :, :, None, :] - coords_b[:, :, None, :, :]
    return torch.linalg.norm(_pbc_displacement_bt(delta, boxes), dim=-1)


def _distogram_loss(pred_coords, target_coords, node_mask, boxes=None, max_tokens=256):
    terms = []
    for b in range(pred_coords.shape[0]):
        idx = node_mask[b].nonzero(as_tuple=False).flatten()[: int(max_tokens)]
        if idx.numel() < 2:
            continue
        pred_sel = pred_coords[b:b + 1, :1].index_select(2, idx)
        target_sel = target_coords[b:b + 1, :1].index_select(2, idx)
        box = None if boxes is None else boxes[b:b + 1, :1]
        pred_d = _pbc_pairwise_distance(pred_sel, pred_sel, box)
        tgt_d = _pbc_pairwise_distance(target_sel, target_sel, box)
        terms.append(F.smooth_l1_loss(torch.nan_to_num(pred_d), torch.nan_to_num(tgt_d)))
    return _finite_loss(torch.stack(terms).mean()) if terms else pred_coords.new_zeros(())


def _chain_pair_mask(protein_mask, chain_pair_mask=None):
    if protein_mask.shape[2] < 2:
        return protein_mask.new_zeros(protein_mask.shape[:2] + (0,), dtype=torch.bool)
    pair_mask = protein_mask[:, :, :-1] & protein_mask[:, :, 1:]
    if chain_pair_mask is not None:
        chain_pair_mask = chain_pair_mask.to(device=protein_mask.device, dtype=torch.bool)
        if chain_pair_mask.ndim == 2:
            chain_pair_mask = chain_pair_mask[:, None, :]
        pair_mask = pair_mask & chain_pair_mask[:, :, : pair_mask.shape[-1]]
    return pair_mask


def _time_sequence(source_time, future_times, steps, device, dtype):
    if torch.is_tensor(source_time) and torch.is_tensor(future_times):
        src = source_time.to(device=device, dtype=dtype).reshape(-1, 1)
        fut = future_times.to(device=device, dtype=dtype)
        if fut.ndim == 1:
            fut = fut.reshape(1, -1).expand(src.shape[0], -1)
        if fut.shape[1] >= int(steps):
            return torch.cat([src, fut[:, : int(steps)]], dim=1)
    base = torch.arange(int(steps) + 1, device=device, dtype=dtype)
    batch = int(source_time.shape[0]) if torch.is_tensor(source_time) and source_time.ndim > 0 else 1
    return base.reshape(1, -1).expand(batch, -1)


def _transition_boxes(boxes, source_box, steps):
    if boxes is None and source_box is None:
        return None
    if source_box is None:
        source_box = boxes[:, 0]
    source_box = source_box[:, None]
    if int(steps) <= 1:
        return source_box
    if boxes is None:
        return source_box.expand(-1, int(steps), -1, -1)
    return torch.cat([source_box, boxes[:, : int(steps) - 1]], dim=1)


def _kinematic_loss(
    source_coords,
    pred_coords,
    target_coords,
    node_mask,
    source_time=None,
    future_times=None,
    boxes=None,
    source_box=None,
):
    if pred_coords.shape[1] < 1:
        zero = pred_coords.new_zeros(())
        return zero, zero
    pred_seq = torch.cat([source_coords[:, None], pred_coords], dim=1)
    target_seq = torch.cat([source_coords[:, None], target_coords[:, : pred_coords.shape[1]]], dim=1)
    times = _time_sequence(
        source_time,
        future_times,
        pred_coords.shape[1],
        pred_coords.device,
        pred_coords.dtype,
    )
    if times.shape[0] != pred_coords.shape[0]:
        times = times[:1].expand(pred_coords.shape[0], -1)
    dt = (
        (times[:, 1:] - times[:, :-1]).abs().clamp_min(1.0e-8)
        * max(float(MD_STORED_TIME_TO_PS), 1.0e-8)
    )
    transition_boxes = _transition_boxes(boxes, source_box, pred_coords.shape[1])
    pred_delta = _pbc_displacement_bt(pred_seq[:, 1:] - pred_seq[:, :-1], transition_boxes)
    target_delta = _pbc_displacement_bt(target_seq[:, 1:] - target_seq[:, :-1], transition_boxes)
    pred_vel = pred_delta / dt[:, :, None, None]
    target_vel = target_delta / dt[:, :, None, None]
    velocity = _masked_smooth_l1(pred_vel, target_vel.detach(), node_mask[:, None])
    if pred_vel.shape[1] < 2:
        return _finite_loss(velocity), pred_coords.new_zeros(())
    mid_dt = (0.5 * (dt[:, 1:] + dt[:, :-1])).clamp_min(1.0e-8)
    pred_acc = (pred_vel[:, 1:] - pred_vel[:, :-1]) / mid_dt[:, :, None, None]
    target_acc = (target_vel[:, 1:] - target_vel[:, :-1]) / mid_dt[:, :, None, None]
    acceleration = _masked_smooth_l1(pred_acc, target_acc.detach(), node_mask[:, None])
    return _finite_loss(velocity), _finite_loss(acceleration)


def _bond_length_loss(pred_coords, source_coords, edge_index, boxes=None, source_box=None):
    if edge_index is None or edge_index.numel() == 0:
        return pred_coords.new_zeros(())
    B, T, N, _ = pred_coords.shape
    src_all = edge_index[0].long()
    dst_all = edge_index[1].long()
    terms = []
    for b in range(B):
        lo, hi = b * N, (b + 1) * N
        keep = (src_all >= lo) & (src_all < hi) & (dst_all >= lo) & (dst_all < hi) & (src_all != dst_all)
        if not bool(keep.any().item()):
            continue
        src = src_all[keep] - lo
        dst = dst_all[keep] - lo
        pair_keys = torch.unique(torch.minimum(src, dst) * N + torch.maximum(src, dst), sorted=False)
        src = torch.div(pair_keys, N, rounding_mode="floor")
        dst = pair_keys % N
        ref_delta = source_coords[b, dst] - source_coords[b, src]
        if source_box is not None:
            ref_delta = minimum_image_displacement_batched(ref_delta[None], source_box[b:b + 1])[0]
        pred_delta = pred_coords[b, :, dst] - pred_coords[b, :, src]
        if boxes is not None:
            pred_delta = minimum_image_displacement_batched(pred_delta, boxes[b])
        ref_len = torch.linalg.norm(ref_delta, dim=-1).detach()
        pred_len = torch.linalg.norm(pred_delta, dim=-1)
        terms.append(F.smooth_l1_loss(torch.nan_to_num(pred_len), ref_len[None].expand_as(pred_len)))
    return _finite_loss(torch.stack(terms).mean()) if terms else pred_coords.new_zeros(())


def _contact_match_loss(pred_d, target_d, candidate_mask, cutoff_norm, max_pairs):
    candidates = candidate_mask if candidate_mask.ndim == target_d.ndim else candidate_mask[None]
    contact = candidates & (target_d.detach() <= float(cutoff_norm))
    if not bool(contact.any().item()):
        return pred_d.new_zeros(())
    flat = contact.reshape(-1).nonzero(as_tuple=False).flatten()
    if int(max_pairs) > 0 and flat.numel() > int(max_pairs):
        flat = flat[: int(max_pairs)]
    pred_flat = pred_d.reshape(-1).index_select(0, flat)
    target_flat = target_d.detach().reshape(-1).index_select(0, flat)
    return _finite_loss(F.smooth_l1_loss(torch.nan_to_num(pred_flat), torch.nan_to_num(target_flat)))


def _interface_potential_losses(
    pred_coords,
    target_coords,
    node_mask,
    node_is_ligand,
    node_vdw_radius_A=None,
    hbond_donor_mask=None,
    hbond_acceptor_mask=None,
    hydrophobic_mask=None,
    boxes=None,
    node_residue_index=None,
):
    zero = pred_coords.new_zeros(())
    vdw_terms = []
    hbond_terms = []
    hydrophobic_terms = []
    max_pairs = int(TRAJECTORY_CONTACT_MAX_PAIRS)
    if node_residue_index is not None:
        pocket_args = dict(node_mask=node_mask, node_is_ligand=node_is_ligand,
                           node_residue_index=node_residue_index, boxes=boxes,
                           cutoff=float(LIGAND_NEAR_RES_CUTOFF) / float(COORD_SCALE))
        active = dynamic_residue_pocket_mask(pred_coords, pred_coords, **pocket_args)
        # Labels may choose pairs for supervision, never for generation.
        active |= dynamic_residue_pocket_mask(target_coords, target_coords, **pocket_args)
    else:
        active = (node_mask & ~node_is_ligand)[:, None].expand(-1, pred_coords.shape[1], -1)
    for b in range(pred_coords.shape[0]):
        lig = (node_mask[b] & node_is_ligand[b]).nonzero(as_tuple=False).flatten()
        prot = active[b].any(0).nonzero(as_tuple=False).flatten()
        if lig.numel() == 0 or prot.numel() == 0:
            continue
        box = None if boxes is None else boxes[b:b + 1, : pred_coords.shape[1]]
        pred_d = _pbc_pairwise_distance(
            pred_coords[b:b + 1, :, lig], pred_coords[b:b + 1, :, prot], box
        )[0]
        target_d = _pbc_pairwise_distance(
            target_coords[b:b + 1, : pred_coords.shape[1], lig],
            target_coords[b:b + 1, : pred_coords.shape[1], prot],
            box,
        )[0]

        if node_vdw_radius_A is not None:
            radii = node_vdw_radius_A.to(device=pred_coords.device, dtype=pred_coords.dtype)
            threshold = (
                float(TRAJECTORY_VDW_SCALE)
                * (radii[b, lig][:, None] + radii[b, prot][None, :])
                / float(COORD_SCALE)
            ).clamp_min(0.05)
            pair_mask = active[b, :, prot][:, None].expand_as(pred_d)
            vdw_terms.append((F.relu(threshold[None] - pred_d).pow(2) * pair_mask).sum() / pair_mask.sum().clamp_min(1))

        if hbond_donor_mask is not None and hbond_acceptor_mask is not None:
            donor = hbond_donor_mask.to(device=pred_coords.device, dtype=torch.bool)
            acceptor = hbond_acceptor_mask.to(device=pred_coords.device, dtype=torch.bool)
            candidates = (donor[b, lig][:, None] & acceptor[b, prot][None, :]) | (
                acceptor[b, lig][:, None] & donor[b, prot][None, :]
            )
            hbond_terms.append(
                _contact_match_loss(
                    pred_d,
                    target_d,
                    candidates[None] & active[b, :, prot][:, None],
                    float(TRAJECTORY_HBOND_DISTANCE_A) / float(COORD_SCALE),
                    max_pairs,
                )
            )

        if hydrophobic_mask is not None:
            hydro = hydrophobic_mask.to(device=pred_coords.device, dtype=torch.bool)
            candidates = hydro[b, lig][:, None] & hydro[b, prot][None, :]
            hydrophobic_terms.append(
                _contact_match_loss(
                    pred_d,
                    target_d,
                    candidates[None] & active[b, :, prot][:, None],
                    float(TRAJECTORY_HYDROPHOBIC_DISTANCE_A) / float(COORD_SCALE),
                    max_pairs,
                )
            )

    vdw = _finite_loss(torch.stack(vdw_terms).mean()) if vdw_terms else zero
    hbond = _finite_loss(torch.stack(hbond_terms).mean()) if hbond_terms else zero
    hydrophobic = _finite_loss(torch.stack(hydrophobic_terms).mean()) if hydrophobic_terms else zero
    return vdw, hbond, hydrophobic


def _force_field_kwargs():
    """One parameter contract for both training and validation energies."""
    return dict(
        cutoff_A=float(MD_FORCE_CUTOFF_A), switch_A=float(MD_FORCE_SWITCH_A),
        softcore_ratio=float(MD_NONBONDED_SOFTCORE_RATIO),
        bond_k=float(MD_BOND_FORCE_CONSTANT), lj_epsilon=float(MD_LJ_EPSILON_KCAL),
        coulomb_constant=float(MD_COULOMB_CONSTANT), dielectric=float(MD_DIELECTRIC),
        force_to_accel_A_ps2=float(MD_FORCE_TO_ACCEL_A_PS2),
    )


def _conservative_dynamics_losses(
    source_coords,
    source_velocity,
    pred_coords,
    target_coords,
    node_mask,
    node_is_ligand,
    node_masses,
    node_feat,
    node_vdw_radius_A,
    bond_edge_index,
    source_time,
    future_times,
    boxes=None,
    source_box=None,
):
    """Auxiliary force, Verlet, and relative-energy constraints.

    Coordinates remain supervised by the MD trajectory.  These losses are
    evaluated only after the hierarchy has produced final atomic coordinates:

    * ``a_kin = 2 (dx - v_prev dt) / dt^2`` must agree with ``-grad(U) / m``;
    * the same force supplies the one-step velocity-Verlet position prior;
    * predicted total and protein--ligand interaction energy *changes* must
      match the corresponding changes along the reference MD trajectory.

    Comparing energy differences instead of minimizing absolute energy avoids
    collapsing every predicted frame toward a static local minimum.
    """
    zero = pred_coords.new_zeros(())
    total_steps = int(pred_coords.shape[1])
    if total_steps < 1:
        return zero, zero, zero, zero
    max_frames = max(1, int(MD_FORCE_MAX_FRAMES))
    frame_idx = torch.linspace(
        0,
        total_steps - 1,
        min(total_steps, max_frames),
        device=pred_coords.device,
    ).round().long().unique_consecutive()
    reference_box = source_box if source_box is not None else (boxes[:, 0] if boxes is not None else None)
    common_force_kwargs = dict(
        reference_coords=source_coords,
        node_mask=node_mask,
        node_masses=node_masses,
        node_feat=node_feat,
        node_vdw_radius_A=node_vdw_radius_A,
        bond_edge_index=bond_edge_index,
        reference_boxes=reference_box,
        coord_scale=float(COORD_SCALE),
        **_force_field_kwargs(),
        return_energy=True,
        node_is_ligand=node_is_ligand,
        return_interface_energy=True,
    )
    source_acceleration, source_energy, source_interface_energy = conservative_acceleration(
        source_coords,
        boxes=reference_box,
        **common_force_kwargs,
    )
    source_energy = source_energy.detach()
    source_interface_energy = source_interface_energy.detach()
    atom_count = node_mask.to(dtype=pred_coords.dtype).sum(dim=-1).clamp_min(1.0)
    ligand_count = (node_mask & node_is_ligand.bool()).to(dtype=pred_coords.dtype).sum(dim=-1).clamp_min(1.0)
    energy_norm = max(float(MD_ENERGY_NORM_KCAL_PER_MOL_ATOM), 1.0e-8)
    interface_norm = max(float(MD_INTERFACE_ENERGY_NORM_KCAL_PER_MOL_LIGAND_ATOM), 1.0e-8)
    accel_norm = max(float(MD_FORCE_ACCEL_NORM_A_PS2), 1.0e-8)
    verlet_terms = []
    force_terms = []
    energy_terms = []
    interface_energy_terms = []
    dynamics_enabled = source_velocity is not None and source_time is not None and future_times is not None
    predicted_states = {}

    def _predicted_state(frame_index):
        if frame_index not in predicted_states:
            frame_box = None if boxes is None else boxes[:, frame_index]
            predicted_states[frame_index] = conservative_acceleration(
                pred_coords[:, frame_index],
                boxes=frame_box,
                **common_force_kwargs,
            )
        return predicted_states[frame_index]

    for idx in frame_idx.tolist():
        current = pred_coords[:, idx]
        target_current = target_coords[:, idx]
        current_box = None if boxes is None else boxes[:, idx]
        _, pred_energy, pred_interface_energy = _predicted_state(idx)
        with torch.no_grad():
            _, target_energy, target_interface_energy = conservative_acceleration(
                target_current,
                boxes=current_box,
                **common_force_kwargs,
            )
        pred_delta_energy = (pred_energy - source_energy) / atom_count
        target_delta_energy = (target_energy - source_energy) / atom_count
        energy_terms.append(F.smooth_l1_loss(
            pred_delta_energy / energy_norm,
            target_delta_energy / energy_norm,
        ))
        pred_delta_interface = (pred_interface_energy - source_interface_energy) / ligand_count
        target_delta_interface = (target_interface_energy - source_interface_energy) / ligand_count
        interface_energy_terms.append(F.smooth_l1_loss(
            pred_delta_interface / interface_norm,
            target_delta_interface / interface_norm,
        ))

        if not dynamics_enabled:
            continue
        prev_coords = source_coords if idx == 0 else pred_coords[:, idx - 1]
        prev_time = source_time if idx == 0 else future_times[:, idx - 1]
        prev_box = reference_box if idx == 0 else (None if boxes is None else boxes[:, idx - 1])
        next_time = future_times[:, idx]
        dt = (next_time - prev_time).abs().clamp_min(1.0e-8) * float(MD_STORED_TIME_TO_PS)
        if idx == 0:
            prev_velocity = source_velocity / max(float(MD_STORED_TIME_TO_PS), 1.0e-8)
        else:
            prev_prev_coords = source_coords if idx == 1 else pred_coords[:, idx - 2]
            prev_prev_time = source_time if idx == 1 else future_times[:, idx - 2]
            prev_prev_box = reference_box if idx == 1 else (None if boxes is None else boxes[:, idx - 2])
            velocity_dt = (prev_time - prev_prev_time).abs().clamp_min(1.0e-8) * float(MD_STORED_TIME_TO_PS)
            previous_step = prev_coords - prev_prev_coords
            if prev_prev_box is not None:
                previous_step = minimum_image_displacement_batched(previous_step, prev_prev_box)
            prev_velocity = previous_step / velocity_dt[:, None, None]
        force_acceleration = source_acceleration if idx == 0 else _predicted_state(idx - 1)[0]
        expected = (
            prev_coords
            + prev_velocity * dt[:, None, None]
            + 0.5 * force_acceleration * dt[:, None, None].square()
        )
        verlet_delta = current - expected
        step_delta = current - prev_coords
        if prev_box is not None:
            verlet_delta = minimum_image_displacement_batched(verlet_delta, prev_box)
            step_delta = minimum_image_displacement_batched(step_delta, prev_box)
        verlet_terms.append(_masked_smooth_l1(
            verlet_delta,
            torch.zeros_like(verlet_delta),
            node_mask,
        ))
        kinematic_acceleration = 2.0 * (
            step_delta - prev_velocity * dt[:, None, None]
        ) / dt[:, None, None].square()
        acceleration_delta_A_ps2 = (
            kinematic_acceleration - force_acceleration
        ) * float(COORD_SCALE)
        force_terms.append(_masked_smooth_l1(
            acceleration_delta_A_ps2 / accel_norm,
            torch.zeros_like(acceleration_delta_A_ps2),
            node_mask,
        ))

    def _mean_or_zero(values):
        return _finite_loss(torch.stack(values).mean()) if values else zero

    return (
        _mean_or_zero(verlet_terms),
        _mean_or_zero(force_terms),
        _mean_or_zero(energy_terms),
        _mean_or_zero(interface_energy_terms),
    )


def _trajectory_physics_loss(
    pred_coords,
    pred_bb,
    target_coords,
    target_bb,
    source_coords,
    node_mask,
    protein_mask,
    node_is_ligand,
    bond_edge_index=None,
    node_masses=None,
    node_feat=None,
    node_vdw_radius_A=None,
    hbond_donor_mask=None,
    hbond_acceptor_mask=None,
    hydrophobic_mask=None,
    source_time=None,
    future_times=None,
    source_velocity=None,
    boxes=None,
    source_box=None,
    chain_pair_mask=None,
    node_residue_index=None,
):
    ca = pred_bb[..., 1, :]
    n_ca = torch.linalg.norm(_pbc_displacement_bt(pred_bb[..., 0, :] - ca, boxes), dim=-1)
    ca_c = torch.linalg.norm(_pbc_displacement_bt(ca - pred_bb[..., 2, :], boxes), dim=-1)
    geom_values = (n_ca - 1.46 / float(COORD_SCALE)).abs() + (ca_c - 1.53 / float(COORD_SCALE)).abs()
    geom = (geom_values * protein_mask.to(dtype=geom_values.dtype)).sum() / protein_mask.to(dtype=geom_values.dtype).sum().clamp_min(1.0)

    if pred_bb.shape[1] > 1:
        bb_step = pred_bb[:, 1:] - pred_bb[:, :-1]
        bb_step = _pbc_displacement_bt(bb_step, None if boxes is None else boxes[:, :-1])
        smooth = _masked_smooth_l1(
            bb_step,
            torch.zeros_like(bb_step),
            protein_mask[:, 1:] & protein_mask[:, :-1],
        )
    else:
        smooth = pred_bb.new_zeros(())

    pair_mask = _chain_pair_mask(protein_mask, chain_pair_mask)
    if pair_mask.numel() > 0 and bool(pair_mask.any().item()):
        pred_c = pred_bb[:, :, :-1, 2]
        pred_n_next = pred_bb[:, :, 1:, 0]
        tgt_c = target_bb[:, :, :-1, 2]
        tgt_n_next = target_bb[:, :, 1:, 0]
        pred_peptide = _pbc_displacement_bt(pred_n_next - pred_c, boxes)
        tgt_peptide = _pbc_displacement_bt(tgt_n_next - tgt_c, boxes)
        peptide = _masked_smooth_l1(pred_peptide, tgt_peptide, pair_mask)
        pred_ca = pred_bb[:, :, :, 1]
        tgt_ca = target_bb[:, :, :, 1]
        pred_ca_step = _pbc_displacement_bt(pred_ca[:, :, 1:] - pred_ca[:, :, :-1], boxes)
        tgt_ca_step = _pbc_displacement_bt(tgt_ca[:, :, 1:] - tgt_ca[:, :, :-1], boxes)
        ca_cont = _masked_smooth_l1(pred_ca_step, tgt_ca_step, pair_mask)
        pred_dir = F.normalize(pred_peptide, dim=-1)
        tgt_dir = F.normalize(tgt_peptide, dim=-1)
        direction = _masked_smooth_l1(torch.nan_to_num(pred_dir), torch.nan_to_num(tgt_dir), pair_mask)
    else:
        peptide = ca_cont = direction = pred_bb.new_zeros(())

    clash_terms = []
    if node_residue_index is not None:
        active = dynamic_residue_pocket_mask(
            pred_coords, pred_coords, node_mask, node_is_ligand, node_residue_index,
            boxes=boxes, cutoff=float(LIGAND_NEAR_RES_CUTOFF) / float(COORD_SCALE),
        )
    else:
        active = (node_mask & ~node_is_ligand)[:, None].expand(-1, pred_coords.shape[1], -1)
    for b in range(pred_coords.shape[0]):
        lig = (node_mask[b] & node_is_ligand[b]).nonzero(as_tuple=False).flatten()
        prot = active[b].any(0).nonzero(as_tuple=False).flatten()
        if lig.numel() == 0 or prot.numel() == 0:
            continue
        box = None if boxes is None else boxes[b:b + 1]
        d = _pbc_pairwise_distance(
            pred_coords[b:b + 1, :, lig], pred_coords[b:b + 1, :, prot], box
        )[0]
        pair_mask = active[b, :, prot][:, None].expand_as(d)
        clash_terms.append((F.relu(float(CLASH_DISTANCE_A) / float(COORD_SCALE) - d).pow(2) * pair_mask).sum() / pair_mask.sum().clamp_min(1))
    clash = torch.stack(clash_terms).mean() if clash_terms else pred_coords.new_zeros(())
    velocity, acceleration = _kinematic_loss(
        source_coords,
        pred_coords,
        target_coords,
        node_mask,
        source_time=source_time,
        future_times=future_times,
        boxes=boxes,
        source_box=source_box,
    )
    # Never reinterpret radius/contact edges as covalent bonds.  If topology
    # metadata is absent, disable bonded priors while retaining dynamic
    # non-bonded neighbors in the graph encoder and conservative potential.
    bond_edges = (
        bond_edge_index
        if bond_edge_index is not None
        else torch.zeros((2, 0), device=pred_coords.device, dtype=torch.long)
    )
    bond_length = _bond_length_loss(
        pred_coords,
        source_coords,
        bond_edges,
        boxes=boxes,
        source_box=source_box,
    )
    vdw, hbond, hydrophobic = _interface_potential_losses(
        pred_coords,
        target_coords,
        node_mask,
        node_is_ligand,
        node_vdw_radius_A=node_vdw_radius_A,
        hbond_donor_mask=hbond_donor_mask,
        hbond_acceptor_mask=hbond_acceptor_mask,
        hydrophobic_mask=hydrophobic_mask,
        boxes=boxes,
        node_residue_index=node_residue_index,
    )
    if node_masses is None:
        node_masses = pred_coords.new_ones(pred_coords.shape[0], pred_coords.shape[2])
    if node_feat is None:
        node_feat = pred_coords.new_zeros(pred_coords.shape[0], pred_coords.shape[2], 5)
    if node_vdw_radius_A is None:
        node_vdw_radius_A = pred_coords.new_full(pred_coords.shape[:1] + pred_coords.shape[2:3], 1.7)
    conservative, force_consistency, energy_consistency, interface_energy = _conservative_dynamics_losses(
        source_coords,
        source_velocity,
        pred_coords,
        target_coords,
        node_mask,
        node_is_ligand,
        node_masses,
        node_feat,
        node_vdw_radius_A,
        bond_edges,
        source_time,
        future_times,
        boxes=boxes,
        source_box=source_box,
    )

    total = (
        float(TRAJECTORY_BACKBONE_GEOMETRY_WEIGHT) * geom
        + float(TRAJECTORY_BACKBONE_SMOOTH_WEIGHT) * smooth
        + float(TRAJECTORY_PEPTIDE_BOND_WEIGHT) * peptide
        + float(TRAJECTORY_CA_CONTINUITY_WEIGHT) * ca_cont
        + float(TRAJECTORY_BACKBONE_DIRECTION_WEIGHT) * direction
        + float(TRAJECTORY_CLASH_WEIGHT) * clash
        + float(TRAJECTORY_VELOCITY_WEIGHT) * velocity
        + float(TRAJECTORY_ACCELERATION_WEIGHT) * acceleration
        + float(TRAJECTORY_BOND_LENGTH_WEIGHT) * bond_length
        + float(TRAJECTORY_VDW_REPULSION_WEIGHT) * vdw
        + float(TRAJECTORY_HBOND_WEIGHT) * hbond
        + float(TRAJECTORY_HYDROPHOBIC_WEIGHT) * hydrophobic
        + float(MD_CONSERVATIVE_LOSS_WEIGHT) * conservative
        + float(MD_FORCE_CONSISTENCY_LOSS_WEIGHT) * force_consistency
        + float(MD_ENERGY_CONSISTENCY_LOSS_WEIGHT) * energy_consistency
        + float(MD_INTERFACE_ENERGY_LOSS_WEIGHT) * interface_energy
    )
    return (
        _finite_loss(total),
        _finite_loss(geom),
        _finite_loss(smooth),
        _finite_loss(peptide),
        _finite_loss(ca_cont),
        _finite_loss(direction),
        _finite_loss(clash),
        _finite_loss(velocity),
        _finite_loss(acceleration),
        _finite_loss(bond_length),
        _finite_loss(vdw),
        _finite_loss(hbond),
        _finite_loss(hydrophobic),
        _finite_loss(conservative),
        _finite_loss(force_consistency),
        _finite_loss(energy_consistency),
        _finite_loss(interface_energy),
    )


class TemporalTrainingMixin:
    def _observed_rollout(self, batch):
        start_t = _debug_now(batch["obs_coords"].device)
        B = batch["obs_coords"].shape[0]
        h, c = self._zero_state(B, batch["obs_coords"].device, batch["obs_coords"].dtype)
        protein_temporal_state = None
        last_enc = None
        gate_latent_loss = batch["obs_coords"].new_zeros(())
        encoder_aux_loss = batch["obs_coords"].new_zeros(())
        obs_len = batch["obs_coords"].shape[1]
        # The physical source must always be the last observed frame.  Reusing
        # frame zero here creates a coordinate/time/graph mismatch downstream.
        use_static_context = False
        self._observed_source_idx = obs_len - 1
        obs_times = batch["obs_times"]
        time_scale = batch.get("time_scale", torch.ones_like(obs_times[:, 0])).clamp_min(1.0e-8)
        if obs_len > 1:
            obs_dt = (obs_times[:, 1:] - obs_times[:, :-1]).abs()
            obs_dt = torch.cat([obs_dt, obs_dt[:, -1:]], dim=1) / time_scale[:, None]
        else:
            obs_dt = torch.ones_like(obs_times)
        observed_grad_frames = min(obs_len, max(0, int(OBSERVED_GRAD_FRAMES)))
        first_grad_frame = obs_len - observed_grad_frames
        _debug_log(
            f"observed_rollout_start B={B} obs_len={obs_len} nodes={batch['obs_coords'].shape[2]} "
            f"res={batch['obs_protein_bb'].shape[2]} source_idx={self._observed_source_idx} "
            f"static_context={int(use_static_context)} grad_frames={observed_grad_frames}"
        )
        for t in range(obs_len):
            frame_t = _debug_now(batch["obs_coords"].device)
            frame_idx = 0 if use_static_context else t
            frame_velocity = batch.get("obs_velocity")
            frame_velocity = None if frame_velocity is None else frame_velocity[:, frame_idx] * time_scale[:, None, None]
            frame_protein_velocity = batch.get("obs_protein_velocity")
            frame_protein_velocity = (
                None
                if frame_protein_velocity is None
                else frame_protein_velocity[:, frame_idx] * time_scale[:, None, None, None]
            )
            frame_box = batch.get("obs_box")
            frame_box = None if frame_box is None else frame_box[:, frame_idx]
            track_observed_grad = bool(self.training) and t >= first_grad_frame
            if track_observed_grad:
                last_enc, h, c, protein_temporal_state = self._step_frame(
                    batch,
                    batch["obs_coords"][:, frame_idx],
                    batch["obs_protein_bb"][:, frame_idx],
                    batch["obs_protein_bb_mask"][:, frame_idx],
                    batch["obs_cloud_grid_seq"][:, frame_idx],
                    h,
                    c,
                    protein_temporal_state,
                    velocity=frame_velocity,
                    protein_velocity=frame_protein_velocity,
                    frame_dt=obs_dt[:, frame_idx],
                    box=frame_box,
                )
            else:
                # Warm up the recurrent state without retaining the large graph,
                # cloud and SE(3) activation graphs for early observed frames.
                with torch.no_grad():
                    last_enc, h, c, protein_temporal_state = self._step_frame(
                        batch,
                        batch["obs_coords"][:, frame_idx],
                        batch["obs_protein_bb"][:, frame_idx],
                        batch["obs_protein_bb_mask"][:, frame_idx],
                        batch["obs_cloud_grid_seq"][:, frame_idx],
                        h,
                        c,
                        protein_temporal_state,
                        velocity=frame_velocity,
                        protein_velocity=frame_protein_velocity,
                        frame_dt=obs_dt[:, frame_idx],
                        box=frame_box,
                    )
                h = h.detach()
                c = c.detach()
                protein_temporal_state = protein_temporal_state.detach()
            gate_latent_loss = gate_latent_loss + last_enc.get("gate_latent_loss", gate_latent_loss.new_zeros(()))
            encoder_aux_loss = encoder_aux_loss + last_enc.get("encoder_aux_loss", encoder_aux_loss.new_zeros(()))
            if t == 0 or t == obs_len - 1 or (t + 1) % _debug_every_frames() == 0:
                _debug_log(
                    f"observed_rollout_frame frame={t + 1}/{obs_len} input_idx={frame_idx} "
                    f"pred_coords={tuple(last_enc['pred_coords'].shape)} pred_bb={tuple(last_enc['pred_protein_bb'].shape)} "
                    f"seconds={_debug_now(batch['obs_coords'].device) - frame_t:.3f}"
                )
        gate_latent_loss = _finite_loss(gate_latent_loss / float(max(1, obs_len)))
        encoder_aux_loss = _finite_loss(encoder_aux_loss / float(max(1, obs_len)))
        _debug_log(
            f"observed_rollout_done seconds={_debug_now(batch['obs_coords'].device) - start_t:.3f} "
            f"h={tuple(h.shape)} c={tuple(c.shape)}"
        )
        return last_enc, h, c, protein_temporal_state, gate_latent_loss, encoder_aux_loss

    def forward(self, batch):
        forward_t = _debug_now(batch["obs_coords"].device)
        target_steps = int(batch["future_coords"].shape[1])
        keyframe_idx = self._future_keyframe_indices(target_steps, batch["future_coords"].device)
        key_protein_mask = self._select_future_keyframes(batch["future_protein_bb_mask"].bool(), keyframe_idx)
        _debug_log(
            f"forward_start obs={tuple(batch['obs_coords'].shape)} future={tuple(batch['future_coords'].shape)} "
            f"protein_future={tuple(batch['future_protein_bb'].shape)} node_mask={tuple(batch['node_mask'].shape)} "
            f"lstm_keyframes={int(keyframe_idx.numel())}/{target_steps} keyframe_idx={keyframe_idx.detach().cpu().tolist()}"
        )
        _last_enc, h, c, protein_temporal_state, gate_latent_loss, encoder_aux_loss = self._observed_rollout(batch)
        _debug_log(f"forward_after_observed elapsed={_debug_now(batch['obs_coords'].device) - forward_t:.3f}")
        source_coords = batch["obs_coords"][:, self._observed_source_idx]
        source_bb = batch["obs_protein_bb"][:, self._observed_source_idx]
        source_mask = batch["obs_protein_bb_mask"][:, self._observed_source_idx]
        source_cloud = batch["obs_cloud_grid_seq"][:, self._observed_source_idx]
        key_times = self._select_future_keyframes(batch["future_times"], keyframe_idx)
        key_prev_times = torch.cat([batch["source_time"][:, None], key_times[:, :-1]], dim=1)
        key_dts = (key_times - key_prev_times).abs() / batch["time_scale"][:, None].clamp_min(1.0e-8)
        key_boxes = self._select_future_keyframes(batch["future_box"], keyframe_idx)
        rollout_t = _debug_now(batch["obs_coords"].device)
        rollout = self._rollout_internal_frames(
            batch,
            h,
            c,
            source_coords,
            source_bb,
            source_mask,
            source_cloud,
            int(keyframe_idx.numel()),
            protein_mask_seq=key_protein_mask,
            protein_temporal_state=protein_temporal_state,
            source_velocity=batch.get("source_velocity_scaled"),
            source_bb_velocity=batch.get("source_protein_velocity_scaled"),
            step_dts=key_dts,
            step_boxes=key_boxes,
        )
        _debug_log(f"forward_after_internal_rollout seconds={_debug_now(batch['obs_coords'].device) - rollout_t:.3f}")
        rough_coords = rollout["pred_coords"]
        rough_bb = rollout["pred_protein_bb"]
        _debug_log(f"refine_start rough_coords={tuple(rough_coords.shape)} rough_bb={tuple(rough_bb.shape)}")
        refine_t = _debug_now(batch["obs_coords"].device)
        refine_out = self.trajectory_refinement.train_refine(
            rough_coords=rough_coords,
            rough_bb=rough_bb,
            source_coords=source_coords,
            source_bb=source_bb,
            node_feat=batch["node_feat"],
            node_mask=batch["node_mask"],
            node_is_ligand=batch["node_is_ligand"],
            protein_mask=batch["future_protein_bb_mask"].bool(),
            target_coords=batch["future_coords"],
            target_bb=batch["future_protein_bb"],
            protein_chain_pair_mask=batch.get("protein_chain_pair_mask"),
            keyframe_indices=keyframe_idx,
            target_steps=target_steps,
            node_residue_index=batch.get("node_residue_index"),
            node_backbone_slot=batch.get("node_backbone_slot"),
            source_velocity=batch.get("source_velocity_scaled"),
            source_bb_velocity=batch.get("source_protein_velocity_scaled"),
            source_box=batch.get("source_box"),
            source_time=batch.get("source_time"),
            future_times=batch.get("future_times"),
            future_dt_scaled=batch.get("future_dt_scaled"),
            boxes=batch.get("future_box"),
            qm_points=batch.get("future_qm_points"),
            qm_density=batch.get("future_qm_density"),
            qm_mask=batch.get("future_qm_mask"),
        )
        _debug_log(
            f"refine_done seconds={_debug_now(batch['obs_coords'].device) - refine_t:.3f} "
            f"refined_coords={tuple(refine_out['refined_coords'].shape)} "
            f"refined_bb={tuple(refine_out['refined_protein_bb'].shape)}"
        )
        pred_coords = refine_out["refined_coords"]
        pred_bb = refine_out["refined_protein_bb"]
        loss_t = _debug_now(batch["obs_coords"].device)
        future_boxes = batch.get("future_box")
        coord_alignment_delta = _pbc_displacement_bt(pred_coords - batch["future_coords"], future_boxes)
        protein_alignment_delta = _pbc_displacement_bt(
            pred_bb - batch["future_protein_bb"], future_boxes
        )
        coord_alignment = _finite_loss(
            _masked_smooth_l1(
                coord_alignment_delta,
                torch.zeros_like(coord_alignment_delta),
                batch["node_mask"][:, None],
            )
        )
        protein_alignment = _finite_loss(
            _masked_smooth_l1(
                protein_alignment_delta,
                torch.zeros_like(protein_alignment_delta),
                batch["future_protein_bb_mask"].bool(),
            )
        )
        noise_diffusion_loss = _finite_loss(0.5 * (
            refine_out["coord_diffusion"] + refine_out["protein_diffusion"]
        ))
        absolute_anchor_loss = _finite_loss(0.5 * (
            refine_out["coord_x0"] + refine_out["protein_x0"]
        ))
        increment_x0_loss = _finite_loss(0.5 * (
            refine_out["coord_increment_x0"] + refine_out["protein_increment_x0"]
        ))
        diffusion_loss = _finite_loss(
            float(DIFFUSION_INCREMENT_NOISE_LOSS_WEIGHT) * noise_diffusion_loss
            + float(DIFFUSION_INCREMENT_X0_LOSS_WEIGHT) * increment_x0_loss
            + float(DIFFUSION_ABSOLUTE_ANCHOR_LOSS_WEIGHT) * absolute_anchor_loss
            + float(MD_HIERARCHY_CLOUD_LOSS_WEIGHT) * refine_out.get(
                "cloud_hierarchy",
                absolute_anchor_loss.new_zeros(()),
            )
        )
        reconstruction_weight = _finite_loss(
            refine_out.get("diffusion_x0_weight_mean", coord_alignment.new_ones(()))
        )
        trajectory_loss_raw = _finite_loss(0.5 * (coord_alignment + protein_alignment))
        trajectory_loss = _finite_loss(reconstruction_weight * trajectory_loss_raw)
        (
            physics_loss,
            geom,
            smooth,
            peptide,
            ca_cont,
            direction,
            clash,
            velocity,
            acceleration,
            bond_length,
            vdw,
            hbond,
            hydrophobic,
            conservative,
            force_consistency,
            energy_consistency,
            interface_energy,
        ) = _trajectory_physics_loss(
            pred_coords,
            pred_bb,
            batch["future_coords"],
            batch["future_protein_bb"],
            source_coords,
            batch["node_mask"],
            batch["future_protein_bb_mask"].bool(),
            batch["node_is_ligand"],
            bond_edge_index=batch.get("bond_edge_index"),
            node_masses=batch.get("node_masses"),
            node_feat=batch.get("node_feat"),
            node_vdw_radius_A=batch.get("node_vdw_radius_A"),
            hbond_donor_mask=batch.get("hbond_donor_mask"),
            hbond_acceptor_mask=batch.get("hbond_acceptor_mask"),
            hydrophobic_mask=batch.get("hydrophobic_mask"),
            source_time=batch.get("source_time"),
            future_times=batch.get("future_times"),
            source_velocity=batch.get("source_velocity"),
            boxes=batch.get("future_box"),
            source_box=batch.get("source_box"),
            chain_pair_mask=batch.get("protein_chain_pair_mask"),
            node_residue_index=batch.get("node_residue_index"),
        )
        physics_loss_raw = _finite_loss(physics_loss)
        physics_loss = _finite_loss(reconstruction_weight * physics_loss_raw)
        distogram_loss_raw = _finite_loss(
            _distogram_loss(
                pred_coords,
                batch["future_coords"],
                batch["node_mask"],
                boxes=batch.get("future_box"),
            )
        )
        distogram_loss = _finite_loss(reconstruction_weight * distogram_loss_raw)
        _debug_log(f"loss_terms_done seconds={_debug_now(batch['obs_coords'].device) - loss_t:.3f}")
        target_key_coords = self._select_future_keyframes(batch["future_coords"], keyframe_idx)
        target_key_bb = self._select_future_keyframes(batch["future_protein_bb"], keyframe_idx)
        target_key_mask = self._select_future_keyframes(batch["future_protein_bb_mask"].bool(), keyframe_idx)
        key_boxes = (
            self._select_future_keyframes(future_boxes, keyframe_idx)
            if future_boxes is not None
            else None
        )
        key_coord_delta = _pbc_displacement_bt(rough_coords - target_key_coords, key_boxes)
        key_protein_delta = _pbc_displacement_bt(rough_bb - target_key_bb, key_boxes)
        key_coord_alignment = _finite_loss(
            _masked_smooth_l1(key_coord_delta, torch.zeros_like(key_coord_delta), batch["node_mask"][:, None])
        )
        key_protein_alignment = _finite_loss(
            _masked_smooth_l1(key_protein_delta, torch.zeros_like(key_protein_delta), target_key_mask)
        )
        keyframe_loss = _finite_loss(0.5 * (key_coord_alignment + key_protein_alignment))
        latent_loss = _finite_loss(gate_latent_loss + encoder_aux_loss + keyframe_loss)
        alignment_loss = trajectory_loss
        total = _finite_loss(
            diffusion_loss
            + trajectory_loss
            + float(DISTOGRAM_LOSS_WEIGHT) * distogram_loss
            + float(LATENT_LOSS_WEIGHT) * latent_loss
            + float(PHYSICS_LOSS_WEIGHT) * physics_loss
            + float(QM_DENSITY_LOSS_WEIGHT) * refine_out["qm_density_loss"]
        )
        total_without_anchor = total
        total = _finite_loss(total + _ddp_parameter_anchor(self, total))
        md_metrics = {}
        if bool(MD_VALIDATION_METRICS) and not self.training:
            with torch.no_grad():
                md_metrics = md_validation_metrics(
                    pred_coords,
                    batch["future_coords"],
                    source_coords,
                    batch.get("source_velocity", torch.zeros_like(source_coords)),
                    batch["node_mask"],
                    batch["future_times"],
                    batch["source_time"],
                    coord_scale=float(COORD_SCALE),
                    node_masses=batch.get("node_masses"),
                    node_feat=batch.get("node_feat"),
                    node_vdw_radius_A=batch.get("node_vdw_radius_A"),
                    node_is_ligand=batch.get("node_is_ligand"),
                    bond_edge_index=batch.get("bond_edge_index"),
                    boxes=batch.get("future_box"),
                    source_box=batch.get("source_box"),
                    rdf_bins=int(MD_RDF_BINS),
                    rdf_max_A=float(MD_RDF_MAX_A),
                    time_to_ps=float(MD_STORED_TIME_TO_PS),
                    force_kwargs=_force_field_kwargs(),
                )
        if _debug_enabled():
            _debug_log(
                f"forward_done seconds={_debug_now(batch['obs_coords'].device) - forward_t:.3f} "
                f"total={float(total.detach().item()):.6f} diffusion={float(diffusion_loss.detach().item()):.6f} "
                f"latent={float(latent_loss.detach().item()):.6f}"
            )
        self.last_loss_dict = {
            "total": total.detach(),
            "total_without_anchor": total_without_anchor.detach(),
            "diffusion": diffusion_loss.detach(),
            "increment_x0": increment_x0_loss.detach(),
            "absolute_anchor": absolute_anchor_loss.detach(),
            "diffusion_t_mean": refine_out.get("diffusion_t_mean", total.new_zeros(())).detach(),
            "diffusion_x0_weight_mean": refine_out.get("diffusion_x0_weight_mean", total.new_zeros(())).detach(),
            "coord_increment_target_rms_A_per_tau": refine_out.get(
                "coord_increment_target_rms_A_per_tau",
                total.new_zeros(()),
            ).detach(),
            "protein_increment_target_rms_A_per_tau": refine_out.get(
                "protein_increment_target_rms_A_per_tau",
                total.new_zeros(()),
            ).detach(),
            "coord_noise_rms": refine_out.get("coord_noise_rms", total.new_zeros(())).detach(),
            "protein_noise_rms": refine_out.get("protein_noise_rms", total.new_zeros(())).detach(),
            "coord_eps_rms": refine_out.get("coord_eps_rms", total.new_zeros(())).detach(),
            "protein_eps_rms": refine_out.get("protein_eps_rms", total.new_zeros(())).detach(),
            "coord_x0_rms_A": refine_out.get("coord_x0_rms_A", total.new_zeros(())).detach(),
            "protein_x0_rms_A": refine_out.get("protein_x0_rms_A", total.new_zeros(())).detach(),
            "trajectory": trajectory_loss.detach(),
            "trajectory_unweighted": trajectory_loss_raw.detach(),
            "distogram": distogram_loss.detach(),
            "distogram_unweighted": distogram_loss_raw.detach(),
            "keyframe": keyframe_loss.detach(),
            "alignment": alignment_loss.detach(),
            "coord_alignment": coord_alignment.detach(),
            "protein_alignment": protein_alignment.detach(),
            "rough_coord_alignment": key_coord_alignment.detach(),
            "rough_protein_alignment": key_protein_alignment.detach(),
            "lstm_keyframes": int(keyframe_idx.numel()),
            "target_frames": int(target_steps),
            "backbone_geometry": geom.detach(),
            "backbone_smooth": smooth.detach(),
            "peptide_bond": peptide.detach(),
            "ca_continuity": ca_cont.detach(),
            "backbone_direction": direction.detach(),
            "clash": clash.detach(),
            "velocity": velocity.detach(),
            "acceleration": acceleration.detach(),
            "bond_length": bond_length.detach(),
            "vdw_repulsion": vdw.detach(),
            "hbond": hbond.detach(),
            "hydrophobic": hydrophobic.detach(),
            "conservative_verlet": conservative.detach(),
            "force_consistency": force_consistency.detach(),
            "relative_energy": energy_consistency.detach(),
            "interface_relative_energy": interface_energy.detach(),
            "cloud_hierarchy": refine_out.get("cloud_hierarchy", total.new_zeros(())).detach(),
            "qm_density_loss": refine_out["qm_density_loss"].detach(),
            "qm_density_coverage": refine_out["qm_density_coverage"].detach(),
            "pocket_rigid": refine_out.get("pocket_rigid", total.new_zeros(())).detach(),
            "pocket_diffusion": refine_out.get("pocket_diffusion", total.new_zeros(())).detach(),
            "ligand_diffusion": refine_out.get("ligand_diffusion", total.new_zeros(())).detach(),
            "keyframe_physics": 0.0,
            "gate_latent": gate_latent_loss.detach(),
            "encoder_aux": encoder_aux_loss.detach(),
            "latent": latent_loss.detach(),
            "physics": physics_loss.detach(),
            "physics_unweighted": physics_loss_raw.detach(),
            "noise": noise_diffusion_loss.detach(),
            "recon": trajectory_loss.detach(),
            "plan": keyframe_loss.detach(),
            "key": keyframe_loss.detach(),
            **{key: value.detach() for key, value in md_metrics.items()},
        }
        return total, diffusion_loss, latent_loss
