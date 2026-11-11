import torch
import torch.nn as nn
import warnings

from config import (
    COORD_SCALE,
    DIFFUSION_ABSOLUTE_ANCHOR_HORIZON_GAMMA,
    DIFFUSION_HIDDEN_DIM,
    DIFFUSION_STEPS,
    DIFFUSION_TIME_DIM,
    NODE_FEAT_DIM,
    LIGAND_NEAR_RES_CUTOFF,
    MD_ELECTRON_CUTOFF_A,
    MD_ELECTRON_TIME_CHUNK,
    MD_ELECTRON_SHELLS,
    MD_ELECTRON_SIGMA_A,
    MD_HIERARCHY_RIGID_POCKET_WEIGHT,
    MD_PHYSICAL_TIME_DIM,
    MD_STORED_TIME_TO_PS,
    MD_USE_PBC,
    PROTEIN_BACKBONE_PROJECT_EACH_DIFFUSION_STEP,
    SAMPLE_DIFFUSION_STEPS,
    QM_DENSITY_REQUIRE_LABELS,
)
from .joint_density import JointElectronDensity
from utils.md import minimum_image_displacement_batched
from utils.pocket import dynamic_residue_pocket_mask
from .temporal_components import SinusoidalTimeEmbedding
from .temporal_utils import (
    _backbone_frames,
    _debug_log,
    _debug_now,
    _masked_mean,
    _masked_smooth_l1,
    _project_backbone_chain_geometry,
)


def _finite_values(x, clamp=1.0e4):
    return torch.nan_to_num(x, nan=0.0, posinf=clamp, neginf=-clamp).clamp(-clamp, clamp)


def _finite_loss(x):
    return torch.nan_to_num(x, nan=0.0, posinf=1.0e4, neginf=0.0)


def _pbc_displacement_bt(delta, boxes=None):
    if not bool(MD_USE_PBC) or boxes is None:
        return delta
    out = delta.clone()
    for b in range(delta.shape[0]):
        out[b] = minimum_image_displacement_batched(delta[b], boxes[b])
    return out


def _previous_frames(sequence, source):
    """Return the causal predecessor of every future frame."""
    if sequence.shape[1] == 1:
        return source[:, None]
    return torch.cat([source[:, None], sequence[:, :-1]], dim=1)


def _frame_increments(sequence, source, boxes=None, source_box=None):
    """Minimum-image increments [source->frame0, frame0->frame1, ...]."""
    transition_boxes = (
        None
        if boxes is None
        else boxes.to(device=sequence.device, dtype=sequence.dtype)
    )
    if source_box is not None:
        source_box = source_box.to(device=sequence.device, dtype=sequence.dtype)
        if boxes is None:
            transition_boxes = source_box[:, None].expand(-1, sequence.shape[1], -1, -1)
        elif sequence.shape[1] == 1:
            transition_boxes = source_box[:, None]
        else:
            transition_boxes = torch.cat([source_box[:, None], transition_boxes[:, :-1]], dim=1)
    return _pbc_displacement_bt(sequence - _previous_frames(sequence, source), transition_boxes)


def _dt_broadcast(frame_dt, values):
    return frame_dt.reshape(frame_dt.shape + (1,) * (values.ndim - 2)).to(
        device=values.device,
        dtype=values.dtype,
    )


def _local_increment_residual_velocity(
    target,
    base,
    source,
    frame_dt,
    rotations,
    boxes=None,
    source_box=None,
):
    """u* = R_prev^T (Delta target - Delta base) / Delta tau."""
    target_increment = _frame_increments(target, source, boxes=boxes, source_box=source_box)
    base_increment = _frame_increments(base, source, boxes=boxes, source_box=source_box)
    residual_velocity = (target_increment - base_increment) / _dt_broadcast(frame_dt, target_increment)
    return _finite_values(_to_local(residual_velocity, rotations))


def _integrate_local_increment_residual_velocity(
    base,
    source,
    local_residual_velocity,
    rotations,
    frame_dt,
    boxes=None,
    source_box=None,
):
    """Integrate Delta X = Delta Q + Delta tau R_prev u from the observed anchor."""
    base_increment = _frame_increments(base, source, boxes=boxes, source_box=source_box)
    correction = _to_global(local_residual_velocity, rotations) * _dt_broadcast(
        frame_dt,
        local_residual_velocity,
    )
    predicted_increment = _finite_values(base_increment + correction)
    predicted = source[:, None] + torch.cumsum(predicted_increment, dim=1)
    return _finite_values(predicted), predicted_increment


def _to_local(vectors, rotations):
    """Project global row vectors onto local frame columns."""
    return torch.matmul(vectors.unsqueeze(-2), rotations).squeeze(-2)


def _to_global(vectors, rotations):
    """Rotate local row-vector components back to the global frame."""
    return torch.matmul(vectors.unsqueeze(-2), rotations.transpose(-1, -2)).squeeze(-2)


def _transport_previous_local(vectors, rotations):
    """Express the preceding local vector in each current local frame."""
    previous_vectors = torch.cat([torch.zeros_like(vectors[:, :1]), vectors[:, :-1]], dim=1)
    previous_rotations = torch.cat([rotations[:, :1], rotations[:, :-1]], dim=1)
    previous_global = _to_global(previous_vectors, previous_rotations)
    return _to_local(previous_global, rotations)


def restore_pocket_from_backbone(
    source_coords,
    source_bb,
    target_bb,
    node_residue_index,
    protein_node_mask,
    source_box=None,
):
    """Rigidly carry each source pocket residue into its predicted backbone frame.

    Side-chain torsions are deliberately left to the following pocket diffusion;
    this function supplies the deterministic main-chain -> residue-atom prior.
    """
    B, T, R = target_bb.shape[:3]
    N = source_coords.shape[1]
    index = node_residue_index.to(device=target_bb.device, dtype=torch.long)
    valid = protein_node_mask.bool() & (index >= 0) & (index < R)
    safe = index.clamp(0, max(0, R - 1))
    batch_index = torch.arange(B, device=target_bb.device)[:, None]
    source_residue_bb = source_bb[batch_index, safe]
    source_rot, source_origin = _backbone_frames(source_residue_bb)
    target_residue_bb = torch.gather(
        target_bb,
        2,
        safe[:, None, :, None, None].expand(-1, T, -1, 3, 3),
    )
    target_rot, target_origin = _backbone_frames(target_residue_bb)

    source_delta = source_coords - source_origin
    if bool(MD_USE_PBC) and source_box is not None:
        source_delta = minimum_image_displacement_batched(source_delta, source_box)
    local_template = _to_local(source_delta, source_rot)
    restored = target_origin + _to_global(
        local_template[:, None].expand(-1, T, -1, -1),
        target_rot,
    )
    return torch.where(valid[:, None, :, None], restored, source_coords[:, None].expand(-1, T, -1, -1))


def enforce_backbone_nodes(coords, backbone, node_residue_index, node_backbone_slot):
    """Make selected N/CA/C atom coordinates exactly match the backbone branch."""
    if node_backbone_slot is None:
        return coords
    B, T, R = backbone.shape[:3]
    index = node_residue_index.to(device=coords.device, dtype=torch.long)
    slot = node_backbone_slot.to(device=coords.device, dtype=torch.long)
    valid = (index >= 0) & (index < R) & (slot >= 0) & (slot < 3)
    combined = (index.clamp(0, max(0, R - 1)) * 3 + slot.clamp(0, 2)).long()
    flat_bb = backbone.reshape(B, T, R * 3, 3)
    exact = torch.gather(flat_bb, 2, combined[:, None, :, None].expand(-1, T, -1, 3))
    return torch.where(valid[:, None, :, None], exact, coords)


def _masked_rms(x, mask, scale=1.0):
    x = _finite_values(x).float()
    mask = mask.to(device=x.device, dtype=x.dtype)
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.expand_as(x).sum().clamp_min(1.0)
    value = ((x * x) * mask).sum() / denom
    return _finite_loss(torch.sqrt(value.clamp_min(0.0)) * float(scale))


def _masked_weighted_smooth_l1(pred, target, mask, sample_weight, horizon_gamma=0.0):
    """Masked Smooth-L1 averaged per sample, then weighted by diffusion SNR."""
    pred = _finite_values(pred)
    target = _finite_values(target)
    loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
    mask = mask.to(device=loss.device, dtype=loss.dtype)
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(loss)
    if loss.ndim >= 3 and loss.shape[1] > 0 and float(horizon_gamma) != 0.0:
        horizon = torch.linspace(0.0, 1.0, loss.shape[1], device=loss.device, dtype=loss.dtype)
        horizon = 1.0 + float(horizon_gamma) * horizon
        horizon = horizon.reshape((1, loss.shape[1]) + (1,) * (loss.ndim - 2))
        mask = mask * horizon
    reduce_dims = tuple(range(1, loss.ndim))
    per_sample = (loss * mask).sum(dim=reduce_dims) / mask.sum(dim=reduce_dims).clamp_min(1.0)
    weight = sample_weight.to(device=loss.device, dtype=loss.dtype).reshape(-1)
    return _finite_loss((per_sample * weight).mean())


class TrajectoryRefinementDiffusion(nn.Module):
    """
    Joint future-increment residual diffusion after the LSTM rollout.

    The diffusion variable is a time-normalized correction to each coarse
    inter-frame increment, represented in the preceding coarse backbone
    frame.  Denoised increments are cumulatively integrated from the last
    observed coordinates, which makes every future frame depend explicitly on
    its predecessor.  Absolute-coordinate losses anchor the accumulated path.
    The backbone -> rigid pocket -> pocket cloud/electron membrane -> ligand
    hierarchy is preserved.
    """

    def __init__(self, d_model, diffusion_steps=None):
        super().__init__()
        self.d_model = int(d_model)
        self.num_steps = int(diffusion_steps or DIFFUSION_STEPS)
        self.time_embed = SinusoidalTimeEmbedding(int(DIFFUSION_TIME_DIM))

        self.physical_time_embed = SinusoidalTimeEmbedding(int(MD_PHYSICAL_TIME_DIM))
        self.protein_increment_context = nn.Sequential(
            nn.LayerNorm(27),
            nn.Linear(27, 18),
            nn.SiLU(),
            nn.Linear(18, 9),
        )
        self.node_increment_context = nn.Sequential(
            nn.LayerNorm(9),
            nn.Linear(9, 12),
            nn.SiLU(),
            nn.Linear(12, 3),
        )
        protein_in = (
            9 + 9 + 9 + 9 + self.d_model + int(DIFFUSION_TIME_DIM)
            + int(MD_PHYSICAL_TIME_DIM) + 1
        )
        pocket_in = (
            3 + 3 + 3 + 3 + int(NODE_FEAT_DIM) + 1 + 2 * self.d_model
            + int(DIFFUSION_TIME_DIM) + int(MD_PHYSICAL_TIME_DIM) + 1
        )
        ligand_in = pocket_in + 3 * self.d_model
        self.joint_density = JointElectronDensity(self.d_model)
        self._warned_missing_qm = False
        self.rough_backbone_proj = nn.Sequential(
            nn.LayerNorm(9),
            nn.Linear(9, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.protein_temporal = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
        )
        self.protein_eps = nn.Sequential(
            nn.LayerNorm(protein_in),
            nn.Linear(protein_in, int(DIFFUSION_HIDDEN_DIM)),
            nn.SiLU(),
            nn.Linear(int(DIFFUSION_HIDDEN_DIM), int(DIFFUSION_HIDDEN_DIM)),
            nn.SiLU(),
            nn.Linear(int(DIFFUSION_HIDDEN_DIM), 9),
        )
        self.backbone_cloud_proj = nn.Sequential(
            nn.LayerNorm(9),
            nn.Linear(9, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.cloud_temporal = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
        )
        self.mapped_backbone_proj = nn.Sequential(
            nn.LayerNorm(9),
            nn.Linear(9, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.pocket_eps = nn.Sequential(
            nn.LayerNorm(pocket_in),
            nn.Linear(pocket_in, int(DIFFUSION_HIDDEN_DIM)),
            nn.SiLU(),
            nn.Linear(int(DIFFUSION_HIDDEN_DIM), int(DIFFUSION_HIDDEN_DIM)),
            nn.SiLU(),
            nn.Linear(int(DIFFUSION_HIDDEN_DIM), 3),
        )
        self.pocket_cloud_proj = nn.Sequential(
            nn.LayerNorm(3 + int(NODE_FEAT_DIM) + self.d_model),
            nn.Linear(3 + int(NODE_FEAT_DIM) + self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.pocket_cloud_temporal = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
        )
        self.electron_shell_proj = nn.Sequential(
            nn.LayerNorm(4 * int(MD_ELECTRON_SHELLS)),
            nn.Linear(4 * int(MD_ELECTRON_SHELLS), self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.ligand_eps = nn.Sequential(
            nn.LayerNorm(ligand_in),
            nn.Linear(ligand_in, int(DIFFUSION_HIDDEN_DIM)),
            nn.SiLU(),
            nn.Linear(int(DIFFUSION_HIDDEN_DIM), int(DIFFUSION_HIDDEN_DIM)),
            nn.SiLU(),
            nn.Linear(int(DIFFUSION_HIDDEN_DIM), 3),
        )

        # A cosine schedule makes q(r_T | r_0) effectively standard Gaussian
        # even for the default 100 steps.  The previous short linear schedule
        # retained too much x0 signal at T to support Gaussian initialization.
        schedule_t = torch.arange(self.num_steps + 1, dtype=torch.float32)
        schedule_t = schedule_t / float(max(1, self.num_steps))
        alpha_bar = torch.cos(((schedule_t + 0.008) / 1.008) * (torch.pi / 2.0)).square()
        alpha_bar = alpha_bar / alpha_bar[0].clamp_min(1.0e-12)
        betas = (1.0 - alpha_bar[1:] / alpha_bar[:-1].clamp_min(1.0e-12)).clamp(1.0e-6, 0.999)
        alpha_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))

    def _q_sample(self, x0, t, noise):
        shape = (t.shape[0],) + (1,) * (x0.ndim - 1)
        a = self.sqrt_alpha_cumprod[t].reshape(shape).to(device=x0.device, dtype=x0.dtype)
        b = self.sqrt_one_minus_alpha_cumprod[t].reshape(shape).to(device=x0.device, dtype=x0.dtype)
        return a * x0 + b * noise

    def _predict_x0(self, xt, t, eps):
        shape = (t.shape[0],) + (1,) * (xt.ndim - 1)
        a = self.sqrt_alpha_cumprod[t].reshape(shape).to(device=xt.device, dtype=xt.dtype).clamp_min(1.0e-4)
        b = self.sqrt_one_minus_alpha_cumprod[t].reshape(shape).to(device=xt.device, dtype=xt.dtype)
        return _finite_values((xt - b * eps) / a)

    def _ddim_step(self, xt, t, next_t, eps, x0=None):
        """Deterministic DDIM transition from timestep ``t`` to ``next_t``."""
        if x0 is None:
            x0 = self._predict_x0(xt, t, eps)
        if int(next_t) < 0:
            return _finite_values(x0)
        shape = (1,) + (1,) * (xt.ndim - 1)
        a_next = self.sqrt_alpha_cumprod[int(next_t)].reshape(shape).to(device=xt.device, dtype=xt.dtype)
        b_next = self.sqrt_one_minus_alpha_cumprod[int(next_t)].reshape(shape).to(
            device=xt.device,
            dtype=xt.dtype,
        )
        return _finite_values(a_next * x0 + b_next * eps)

    def _keyframe_indices(self, keyframe_indices, keyframes, target_steps, device):
        keyframes = max(1, int(keyframes))
        target_steps = max(1, int(target_steps))
        if keyframe_indices is None:
            if keyframes >= target_steps:
                return torch.arange(target_steps, device=device, dtype=torch.long)
            idx = torch.linspace(0, target_steps - 1, keyframes, device=device).round().long()
        else:
            idx = keyframe_indices.to(device=device, dtype=torch.long)
            if idx.ndim > 1:
                idx = idx[0]
        if idx.numel() == 0:
            idx = torch.zeros((1,), device=device, dtype=torch.long)
        idx = idx[:keyframes].clamp(0, target_steps - 1)
        idx[0] = 0
        idx[-1] = target_steps - 1
        return idx

    def _expand_keyframes_to_steps(self, rough, target_steps, keyframe_indices=None, target_times=None):
        B, K = rough.shape[:2]
        target_steps = max(1, int(target_steps))
        if K == target_steps:
            return rough
        idx = self._keyframe_indices(keyframe_indices, K, target_steps, rough.device)
        if K == 1:
            return rough[:, :1].expand((B, target_steps) + rough.shape[2:]).contiguous()
        out = rough.new_empty((B, target_steps) + rough.shape[2:])
        first = int(idx[0].item())
        if first > 0:
            out[:, : first + 1] = rough[:, :1]
        for k in range(K - 1):
            start = int(idx[k].item())
            end = int(idx[k + 1].item())
            if end < start:
                continue
            span = max(1, end - start)
            if target_times is not None:
                segment_times = target_times[:, start:end + 1].to(device=rough.device, dtype=rough.dtype)
                left_time = target_times[:, start:start + 1].to(device=rough.device, dtype=rough.dtype)
                right_time = target_times[:, end:end + 1].to(device=rough.device, dtype=rough.dtype)
                alpha = (segment_times - left_time) / (right_time - left_time).clamp_min(1.0e-8)
                alpha = alpha.reshape((B, -1) + (1,) * (rough.ndim - 2))
            else:
                alpha = torch.linspace(0.0, 1.0, end - start + 1, device=rough.device, dtype=rough.dtype)
                alpha = alpha.reshape((1, -1) + (1,) * (rough.ndim - 2))
            left = rough[:, k:k + 1]
            right = rough[:, k + 1:k + 2]
            out[:, start:end + 1] = left + alpha * (right - left)
        last = int(idx[-1].item())
        if last < target_steps - 1:
            out[:, last:] = rough[:, -1:]
        return _finite_values(out)

    def _expand_rough_condition(self, rough_coords, rough_bb, target_steps, keyframe_indices=None, target_times=None):
        target_steps = max(1, int(target_steps))
        rough_coords_full = self._expand_keyframes_to_steps(
            rough_coords, target_steps, keyframe_indices, target_times=target_times
        )
        rough_bb_full = self._expand_keyframes_to_steps(
            rough_bb, target_steps, keyframe_indices, target_times=target_times
        )
        return rough_coords_full, rough_bb_full

    def _backbone_anchor_relative(self, backbone, protein_mask, boxes=None):
        B, T, R = backbone.shape[:3]
        first = protein_mask.bool().to(dtype=torch.int64).argmax(dim=2)
        anchor = torch.gather(
            backbone[..., 1, :],
            2,
            first[:, :, None, None].expand(-1, -1, 1, 3),
        ).squeeze(2)
        relative = _pbc_displacement_bt(backbone - anchor[:, :, None, None, :], boxes)
        residue_rot, _ = _backbone_frames(backbone)
        anchor_rot = torch.gather(
            residue_rot,
            2,
            first[:, :, None, None, None].expand(-1, -1, 1, 3, 3),
        ).squeeze(2)
        return _to_local(relative, anchor_rot[:, :, None, None])

    def _rough_backbone_context(self, rough_bb, protein_mask, boxes=None):
        B, T, R = rough_bb.shape[:3]
        relative = self._backbone_anchor_relative(rough_bb, protein_mask, boxes=boxes)
        tokens = self.rough_backbone_proj(relative.reshape(B, T, R, 9))
        pooled = _masked_mean(tokens, protein_mask.bool(), dim=2)
        temporal = self.protein_temporal(pooled.transpose(1, 2)).transpose(1, 2)
        return _finite_values(pooled + temporal)

    def _physical_time_context(
        self,
        future_dt_scaled,
        batch_size,
        target_steps,
        device,
        dtype,
        source_time=None,
        future_times=None,
    ):
        if future_dt_scaled is None:
            future_dt_scaled = torch.ones((batch_size, target_steps), device=device, dtype=dtype)
        future_dt_scaled = future_dt_scaled.to(device=device, dtype=dtype).clamp_min(1.0e-8)
        if source_time is not None and future_times is not None:
            source_time = source_time.to(device=device, dtype=dtype).reshape(batch_size, 1)
            future_times = future_times.to(device=device, dtype=dtype)[:, :target_steps]
            elapsed = (future_times - source_time).clamp_min(0.0) * float(MD_STORED_TIME_TO_PS)
        else:
            elapsed = torch.cumsum(future_dt_scaled, dim=1)
        emb = self.physical_time_embed(elapsed.reshape(-1)).to(dtype=dtype)
        return emb.reshape(batch_size, target_steps, -1), future_dt_scaled

    def _protein_eps(
        self,
        noisy_increment_velocity,
        rough_bb,
        source_bb,
        source_bb_velocity,
        protein_mask,
        t_emb,
        physical_t_emb,
        frame_dt,
        boxes=None,
        source_box=None,
    ):
        B, T, R = noisy_increment_velocity.shape[:3]
        previous_rough_bb = _previous_frames(rough_bb, source_bb)
        previous_rot, _previous_origin = _backbone_frames(previous_rough_bb)
        frame_rot = previous_rot.unsqueeze(-3)
        previous_noisy = _transport_previous_local(noisy_increment_velocity, frame_rot).reshape(B, T, R, 9)
        noisy_local = noisy_increment_velocity.reshape(B, T, R, 9)
        noisy_local = noisy_local + self.protein_increment_context(
            torch.cat([noisy_local, previous_noisy, noisy_local - previous_noisy], dim=-1)
        )
        base_velocity = _frame_increments(
            rough_bb,
            source_bb,
            boxes=boxes,
            source_box=source_box,
        ) / _dt_broadcast(
            frame_dt,
            rough_bb,
        )
        base_velocity_local = _to_local(base_velocity, frame_rot).reshape(B, T, R, 9)
        if source_bb_velocity is None:
            source_bb_velocity = torch.zeros_like(source_bb)
        source_velocity = _to_local(
            source_bb_velocity[:, None].expand(-1, T, -1, -1, -1),
            frame_rot,
        ).reshape(B, T, R, 9)
        previous_base_global = torch.cat(
            [source_bb_velocity[:, None], base_velocity[:, :-1]],
            dim=1,
        )
        previous_base_velocity = _to_local(previous_base_global, frame_rot).reshape(B, T, R, 9)
        temporal_ctx = self._rough_backbone_context(rough_bb, protein_mask, boxes=boxes)
        feat = torch.cat(
            [
                noisy_local,
                base_velocity_local,
                previous_base_velocity,
                source_velocity,
                temporal_ctx[:, :, None].expand(-1, -1, R, -1),
                t_emb[:, :, None].expand(-1, -1, R, -1),
                physical_t_emb[:, :, None].expand(-1, -1, R, -1),
                frame_dt[:, :, None, None].expand(-1, -1, R, -1),
            ],
            dim=-1,
        )
        # Epsilon stays in the fixed previous-coarse-frame basis.  Keeping the
        # diffusion state in one basis is required for a valid DDIM trajectory.
        return self.protein_eps(feat).reshape(B, T, R, 3, 3)

    def _cloud_from_backbone(self, refined_bb, protein_mask, boxes=None):
        B, T, R = refined_bb.shape[:3]
        relative = self._backbone_anchor_relative(refined_bb, protein_mask, boxes=boxes)
        tokens = self.backbone_cloud_proj(relative.reshape(B, T, R, 9))
        cloud = _masked_mean(tokens, protein_mask.bool(), dim=2)
        temporal = self.cloud_temporal(cloud.transpose(1, 2)).transpose(1, 2)
        return _finite_values(cloud + temporal)

    def _mapped_backbone_context(self, refined_bb, node_residue_index):
        B, T, R = refined_bb.shape[:3]
        if node_residue_index is None:
            return refined_bb.new_zeros((B, T, 1, self.d_model))
        index = node_residue_index.to(device=refined_bb.device, dtype=torch.long)
        N = index.shape[1]
        valid = (index >= 0) & (index < R)
        safe = index.clamp(0, max(0, R - 1))
        flat_bb = refined_bb.reshape(B, T, R, 9)
        gathered = torch.gather(flat_bb, 2, safe[:, None, :, None].expand(-1, T, -1, 9))
        local = gathered.reshape(B, T, N, 3, 3)
        rotations, origins = _backbone_frames(local)
        local = _to_local(local - origins.unsqueeze(-2), rotations.unsqueeze(-3)).reshape(B, T, N, 9)
        context = self.mapped_backbone_proj(local)
        return context * valid[:, None, :, None].to(dtype=context.dtype)

    def _node_frames(self, refined_bb, protein_mask, node_residue_index):
        """Residue-local frames for protein atoms and a stable protein frame for ligand atoms."""
        B, T, R = refined_bb.shape[:3]
        rotations, origins = _backbone_frames(refined_bb)
        index = node_residue_index.to(device=refined_bb.device, dtype=torch.long)
        valid = (index >= 0) & (index < R)
        safe = index.clamp(0, max(0, R - 1))
        mapped_rot = torch.gather(
            rotations,
            2,
            safe[:, None, :, None, None].expand(-1, T, -1, 3, 3),
        )
        mapped_origin = torch.gather(
            origins,
            2,
            safe[:, None, :, None].expand(-1, T, -1, 3),
        )
        first = protein_mask.bool().to(dtype=torch.int64).argmax(dim=2)
        first_rot = torch.gather(
            rotations,
            2,
            first[:, :, None, None, None].expand(-1, -1, 1, 3, 3),
        ).squeeze(2)
        first_origin = torch.gather(
            origins,
            2,
            first[:, :, None, None].expand(-1, -1, 1, 3),
        ).squeeze(2)
        node_rot = torch.where(valid[:, None, :, None, None], mapped_rot, first_rot[:, :, None])
        node_origin = torch.where(valid[:, None, :, None], mapped_origin, first_origin[:, :, None])
        return node_rot, node_origin

    def _node_diffusion_features(
        self,
        noisy_increment_velocity,
        base_coords,
        source_coords,
        source_velocity,
        node_feat,
        node_is_ligand,
        backbone_cloud_z,
        mapped_backbone_z,
        t_emb,
        physical_t_emb,
        frame_dt,
        node_rot,
        boxes=None,
        source_box=None,
    ):
        B, T, N = noisy_increment_velocity.shape[:3]
        if source_velocity is None:
            source_velocity = torch.zeros_like(source_coords)
        noisy_local = noisy_increment_velocity
        previous_noisy = _transport_previous_local(noisy_local, node_rot)
        noisy_local = noisy_local + self.node_increment_context(
            torch.cat([noisy_local, previous_noisy, noisy_local - previous_noisy], dim=-1)
        )
        base_velocity = _frame_increments(
            base_coords,
            source_coords,
            boxes=boxes,
            source_box=source_box,
        ) / _dt_broadcast(
            frame_dt,
            base_coords,
        )
        base_velocity_local = _to_local(base_velocity, node_rot)
        velocity_local = _to_local(source_velocity[:, None].expand(-1, T, -1, -1), node_rot)
        previous_base_global = torch.cat(
            [source_velocity[:, None], base_velocity[:, :-1]],
            dim=1,
        )
        previous_base_velocity = _to_local(previous_base_global, node_rot)
        return torch.cat(
            [
                noisy_local,
                base_velocity_local,
                previous_base_velocity,
                velocity_local,
                node_feat[:, None].expand(-1, T, -1, -1),
                node_is_ligand[:, None, :, None].to(dtype=noisy_increment_velocity.dtype).expand(-1, T, -1, -1),
                backbone_cloud_z[:, :, None].expand(-1, -1, N, -1),
                mapped_backbone_z,
                t_emb[:, :, None].expand(-1, -1, N, -1),
                physical_t_emb[:, :, None].expand(-1, -1, N, -1),
                frame_dt[:, :, None, None].expand(-1, -1, N, -1),
            ],
            dim=-1,
        )

    def _pocket_cloud_from_coords(
        self,
        protein_coords,
        node_feat,
        protein_node_mask,
        mapped_backbone_z,
        node_rot,
        node_origin,
        node_mask,
        node_is_ligand,
        node_residue_index,
        boxes=None,
    ):
        B, T, N = protein_coords.shape[:3]
        relative_coords = _pbc_displacement_bt(protein_coords - node_origin, boxes)
        relative_coords = _to_local(relative_coords, node_rot)
        feat = torch.cat(
            [
                relative_coords,
                node_feat[:, None].expand(-1, T, -1, -1),
                mapped_backbone_z,
            ],
            dim=-1,
        )
        tokens = self.pocket_cloud_proj(feat)
        mask = dynamic_residue_pocket_mask(
            protein_coords, protein_coords, node_mask, node_is_ligand,
            node_residue_index, boxes=boxes if bool(MD_USE_PBC) else None,
            cutoff=float(LIGAND_NEAR_RES_CUTOFF) / float(COORD_SCALE),
        )
        pooled = _masked_mean(tokens, mask, dim=2)
        temporal = self.pocket_cloud_temporal(pooled.transpose(1, 2)).transpose(1, 2)
        return _finite_values(pooled + temporal)

    def _electron_membrane(
        self,
        protein_coords,
        query_coords,
        refined_bb,
        protein_mask,
        node_feat,
        node_mask,
        node_is_ligand,
        node_residue_index,
        boxes=None,
        return_descriptor=False,
    ):
        """Protein electron-density shells plus directional first moments.

        Density alone is radial and cannot tell a ligand denoiser which way the
        pocket surface lies.  Directional moments are expressed in a predicted
        backbone frame, so they retain shape while remaining invariant to a
        global rigid rotation/translation of the complex.
        """
        B, T, N = protein_coords.shape[:3]
        shells = int(MD_ELECTRON_SHELLS)
        cutoff_A = float(MD_ELECTRON_CUTOFF_A)
        sigma_A = max(float(MD_ELECTRON_SIGMA_A), 1.0e-3)
        centers = torch.linspace(0.0, cutoff_A, shells, device=protein_coords.device, dtype=protein_coords.dtype)
        shell_values = protein_coords.new_zeros((B, T, N, 4 * shells))
        # The pocket is selected from this step's recovered protein and coarse
        # ligand, not from noisy ligand queries or ground-truth future frames.
        active_pocket = dynamic_residue_pocket_mask(
            protein_coords, protein_coords, node_mask, node_is_ligand,
            node_residue_index, boxes=boxes if bool(MD_USE_PBC) else None,
            cutoff=float(LIGAND_NEAR_RES_CUTOFF) / float(COORD_SCALE),
        )
        residue_rot, _residue_origin = _backbone_frames(refined_bb)
        first = protein_mask.bool().to(dtype=torch.int64).argmax(dim=2)
        anchor_rot = torch.gather(
            residue_rot,
            2,
            first[:, :, None, None, None].expand(-1, -1, 1, 3, 3),
        ).squeeze(2)
        for b in range(B):
            protein_idx = active_pocket[b].any(dim=0).nonzero(as_tuple=False).flatten()
            ligand_idx = (node_mask[b].bool() & node_is_ligand[b].bool()).nonzero(as_tuple=False).flatten()
            if protein_idx.numel() == 0 or ligand_idx.numel() == 0:
                continue
            weights = (node_feat[b, protein_idx, 0] * 53.0).clamp_min(1.0)
            for start in range(0, T, max(1, int(MD_ELECTRON_TIME_CHUNK))):
                end = min(T, start + max(1, int(MD_ELECTRON_TIME_CHUNK)))
                protein_xyz = protein_coords[b, start:end].index_select(1, protein_idx)
                ligand_xyz = query_coords[b, start:end].index_select(1, ligand_idx)
                delta = protein_xyz[:, None] - ligand_xyz[:, :, None]
                if bool(MD_USE_PBC) and boxes is not None:
                    delta = minimum_image_displacement_batched(delta, boxes[b, start:end])
                chunk_size = end - start
                chunk_weights = weights[None] * active_pocket[b, start:end, protein_idx].to(weights.dtype)
                weight_norm = chunk_weights.sum(dim=1).clamp_min(1.0)
                distance_A = torch.linalg.norm(delta, dim=-1) * float(COORD_SCALE)
                radial = torch.exp(-0.5 * ((distance_A[:, :, :, None] - centers) / sigma_A).square())
                density = (
                    radial * chunk_weights[:, None, :, None]
                ).sum(dim=2) / weight_norm[:, None, None]
                local_delta = _to_local(delta, anchor_rot[b, start:end, None, None])
                unit_local = local_delta / torch.linalg.norm(local_delta, dim=-1, keepdim=True).clamp_min(1.0e-8)
                moment = (
                    radial[..., None]
                    * chunk_weights[:, None, :, None, None]
                    * unit_local[:, :, :, None, :]
                ).sum(dim=2) / weight_norm[:, None, None, None]
                descriptor = torch.cat([density, moment.reshape(end - start, ligand_idx.numel(), 3 * shells)], dim=-1)
                shell_values[b, start:end, ligand_idx] = descriptor
        embedding = _finite_values(self.electron_shell_proj(shell_values))
        if return_descriptor:
            return embedding, _finite_values(shell_values)
        return embedding

    def train_refine(
        self,
        rough_coords,
        rough_bb,
        source_coords,
        source_bb,
        node_feat,
        node_mask,
        node_is_ligand,
        protein_mask,
        target_coords,
        target_bb,
        protein_chain_pair_mask=None,
        keyframe_indices=None,
        target_steps=None,
        node_residue_index=None,
        node_backbone_slot=None,
        source_velocity=None,
        source_bb_velocity=None,
        source_box=None,
        source_time=None,
        future_times=None,
        future_dt_scaled=None,
        boxes=None,
        qm_points=None,
        qm_density=None,
        qm_mask=None,
    ):
        start_t = _debug_now(rough_coords.device)
        target_steps = int(target_coords.shape[1] if target_steps is None else target_steps)
        keyframes = int(rough_coords.shape[1])
        rough_coords, rough_bb = self._expand_rough_condition(
            rough_coords,
            rough_bb,
            target_steps=target_steps,
            keyframe_indices=keyframe_indices,
            target_times=future_times,
        )
        B, T = rough_coords.shape[:2]
        _debug_log(
            f"trajectory_refine_train_start B={B} T={T} keyframes={keyframes} N={rough_coords.shape[2]} R={rough_bb.shape[2]} "
            f"rough_coords={tuple(rough_coords.shape)} rough_bb={tuple(rough_bb.shape)}"
        )
        t = torch.randint(0, self.num_steps, (B,), device=rough_coords.device)
        t_emb = self.time_embed(t).to(dtype=rough_coords.dtype)[:, None].expand(-1, T, -1)
        physical_t_emb, frame_dt = self._physical_time_context(
            future_dt_scaled,
            B,
            T,
            rough_coords.device,
            rough_coords.dtype,
            source_time=source_time,
            future_times=future_times,
        )
        if node_residue_index is None:
            raise ValueError("dynamic pockets require residue mapping from schema-4 caches")
        if node_backbone_slot is None:
            node_backbone_slot = torch.full_like(node_residue_index, -1)

        protein_valid = protein_mask.bool()[:, :, :, None, None]
        node_valid = node_mask.bool()[:, None, :, None]
        protein_node_mask = node_mask.bool() & ~node_is_ligand.bool()
        ligand_node_mask = node_mask.bool() & node_is_ligand.bool()
        pocket_increment_mask = protein_node_mask & (node_backbone_slot < 0)
        coord_increment_mask = pocket_increment_mask | ligand_node_mask
        coord_increment_valid = coord_increment_mask[:, None, :, None]

        # The local basis is conditioned only on the deterministic coarse path.
        # Consequently q(u_t | u_0) and every DDIM state share one coordinate
        # system even while the reconstructed backbone changes during denoising.
        previous_rough_bb = _previous_frames(rough_bb, source_bb)
        previous_protein_mask = torch.cat([protein_mask[:, :1], protein_mask[:, :-1]], dim=1)
        previous_bb_rot, _previous_bb_origin = _backbone_frames(previous_rough_bb)
        bb_increment_rot = previous_bb_rot.unsqueeze(-3)
        bb_increment_target = _local_increment_residual_velocity(
            target_bb,
            rough_bb,
            source_bb,
            frame_dt,
            bb_increment_rot,
            boxes=boxes,
            source_box=source_box,
        )
        bb_increment_target = torch.where(
            protein_valid,
            bb_increment_target,
            torch.zeros_like(bb_increment_target),
        ).detach()
        bb_noise = torch.where(protein_valid, torch.randn_like(target_bb), torch.zeros_like(target_bb))
        bb_increment_noisy = self._q_sample(bb_increment_target, t, bb_noise)
        protein_t = _debug_now(rough_coords.device)
        bb_eps = self._protein_eps(
            bb_increment_noisy,
            rough_bb,
            source_bb,
            source_bb_velocity,
            protein_mask,
            t_emb,
            physical_t_emb,
            frame_dt,
            boxes=boxes,
            source_box=source_box,
        )
        bb_increment_x0 = self._predict_x0(bb_increment_noisy, t, bb_eps)
        bb_increment_x0 = torch.where(protein_valid, bb_increment_x0, torch.zeros_like(bb_increment_x0))
        refined_bb, _refined_bb_increment = _integrate_local_increment_residual_velocity(
            rough_bb,
            source_bb,
            bb_increment_x0,
            bb_increment_rot,
            frame_dt,
            boxes=boxes,
            source_box=source_box,
        )
        refined_bb = torch.where(protein_valid, refined_bb, rough_bb)
        _debug_log(
            f"trajectory_refine_protein_eps_done seconds={_debug_now(rough_coords.device) - protein_t:.3f} "
            f"bb_eps={tuple(bb_eps.shape)}"
        )
        project_t = _debug_now(rough_coords.device)
        refined_bb = _project_backbone_chain_geometry(
            refined_bb,
            protein_mask,
            chain_pair_mask=protein_chain_pair_mask,
        )
        refined_bb = torch.where(protein_valid, refined_bb, rough_bb)
        _debug_log(f"trajectory_refine_project_done seconds={_debug_now(rough_coords.device) - project_t:.3f}")

        cloud_t = _debug_now(rough_coords.device)
        cloud_z = self._cloud_from_backbone(refined_bb, protein_mask, boxes=boxes)
        _debug_log(
            f"trajectory_refine_cloud_done seconds={_debug_now(rough_coords.device) - cloud_t:.3f} "
            f"cloud_z={tuple(cloud_z.shape)}"
        )

        pocket_rigid_prior = restore_pocket_from_backbone(
            source_coords,
            source_bb,
            refined_bb,
            node_residue_index,
            protein_node_mask,
            source_box=source_box,
        )
        pocket_rigid_prior = enforce_backbone_nodes(
            pocket_rigid_prior,
            refined_bb,
            node_residue_index,
            node_backbone_slot,
        )
        hierarchy_coords = torch.where(
            protein_node_mask[:, None, :, None],
            pocket_rigid_prior,
            rough_coords,
        )
        node_rot, node_origin = self._node_frames(
            previous_rough_bb,
            previous_protein_mask,
            node_residue_index,
        )
        coord_increment_target = _local_increment_residual_velocity(
            target_coords,
            hierarchy_coords,
            source_coords,
            frame_dt,
            node_rot,
            boxes=boxes,
            source_box=source_box,
        )
        coord_increment_target = torch.where(
            coord_increment_valid,
            coord_increment_target,
            torch.zeros_like(coord_increment_target),
        ).detach()
        coord_noise = torch.where(
            coord_increment_valid,
            torch.randn_like(target_coords),
            torch.zeros_like(target_coords),
        )
        coord_increment_noisy = self._q_sample(coord_increment_target, t, coord_noise)
        coord_t = _debug_now(rough_coords.device)
        mapped_backbone_z = self._mapped_backbone_context(refined_bb, node_residue_index)
        node_diffusion_feat = self._node_diffusion_features(
            coord_increment_noisy,
            hierarchy_coords,
            source_coords,
            source_velocity,
            node_feat,
            node_is_ligand,
            cloud_z,
            mapped_backbone_z,
            t_emb,
            physical_t_emb,
            frame_dt,
            node_rot,
            boxes=boxes,
            source_box=source_box,
        )
        pocket_eps = self.pocket_eps(node_diffusion_feat)
        pocket_increment_x0 = self._predict_x0(coord_increment_noisy, t, pocket_eps)
        pocket_increment_x0 = torch.where(
            coord_increment_valid,
            pocket_increment_x0,
            torch.zeros_like(pocket_increment_x0),
        )
        pocket_coords_x0, _pocket_predicted_increment = _integrate_local_increment_residual_velocity(
            hierarchy_coords,
            source_coords,
            pocket_increment_x0,
            node_rot,
            frame_dt,
            boxes=boxes,
            source_box=source_box,
        )
        refined_protein_coords = torch.where(
            protein_node_mask[:, None, :, None],
            pocket_coords_x0,
            hierarchy_coords,
        )
        refined_protein_coords = enforce_backbone_nodes(
            refined_protein_coords,
            refined_bb,
            node_residue_index,
            node_backbone_slot,
        )
        pocket_cloud_z = self._pocket_cloud_from_coords(
            refined_protein_coords,
            node_feat,
            protein_node_mask,
            mapped_backbone_z,
            node_rot,
            node_origin,
            node_mask,
            node_is_ligand,
            node_residue_index,
            boxes=boxes,
        )
        noisy_query_coords, _noisy_query_increment = _integrate_local_increment_residual_velocity(
            hierarchy_coords,
            source_coords,
            coord_increment_noisy,
            node_rot,
            frame_dt,
            boxes=boxes,
            source_box=source_box,
        )
        electron_membrane_z = self._electron_membrane(
            refined_protein_coords,
            noisy_query_coords,
            refined_bb,
            protein_mask,
            node_feat,
            node_mask,
            node_is_ligand,
            node_residue_index,
            boxes=boxes,
        )
        joint_field = self.joint_density.predict(
            refined_protein_coords, node_feat, node_mask, node_is_ligand,
            node_residue_index, node_rot, mapped_backbone_z, pocket_cloud_z, boxes,
        )
        joint_density_z = self.joint_density.ligand_context(joint_field, noisy_query_coords, node_rot)
        # The ligand may use its own chemical identity/motion prior, but all
        # protein structural information must pass through the reconstructed
        # pocket cloud rather than shortcutting via backbone tokens.
        ligand_node_diffusion_feat = self._node_diffusion_features(
            coord_increment_noisy,
            hierarchy_coords,
            source_coords,
            source_velocity,
            node_feat,
            node_is_ligand,
            torch.zeros_like(cloud_z),
            torch.zeros_like(mapped_backbone_z),
            t_emb,
            physical_t_emb,
            frame_dt,
            node_rot,
            boxes=boxes,
            source_box=source_box,
        )
        ligand_feat = torch.cat(
            [
                ligand_node_diffusion_feat,
                pocket_cloud_z[:, :, None].expand(-1, -1, rough_coords.shape[2], -1),
                electron_membrane_z,
                joint_density_z,
            ],
            dim=-1,
        )
        ligand_eps = self.ligand_eps(ligand_feat)
        coord_eps = torch.where(ligand_node_mask[:, None, :, None], ligand_eps, pocket_eps)
        coord_increment_x0 = self._predict_x0(coord_increment_noisy, t, coord_eps)
        coord_increment_x0 = torch.where(
            coord_increment_valid,
            coord_increment_x0,
            torch.zeros_like(coord_increment_x0),
        )
        refined_coords, _refined_coord_increment = _integrate_local_increment_residual_velocity(
            hierarchy_coords,
            source_coords,
            coord_increment_x0,
            node_rot,
            frame_dt,
            boxes=boxes,
            source_box=source_box,
        )
        refined_coords = torch.where(node_valid, refined_coords, hierarchy_coords)
        refined_coords = enforce_backbone_nodes(
            refined_coords,
            refined_bb,
            node_residue_index,
            node_backbone_slot,
        )
        _debug_log(
            f"trajectory_refine_coord_done seconds={_debug_now(rough_coords.device) - coord_t:.3f} "
            f"coord_eps={tuple(coord_eps.shape)} refined_coords={tuple(refined_coords.shape)}"
        )

        loss_t = _debug_now(rough_coords.device)
        protein_diffusion = _masked_smooth_l1(bb_eps, bb_noise, protein_mask.bool())
        # Both the increment reconstruction and cumulative absolute-coordinate
        # anchor are auxiliary to epsilon prediction.  Weight them by alpha_bar
        # so near-pure-noise timesteps do not amplify x0 errors.
        x0_weight = self.alpha_cumprod[t].to(device=rough_coords.device, dtype=rough_coords.dtype)
        protein_increment_x0 = _masked_weighted_smooth_l1(
            bb_increment_x0,
            bb_increment_target,
            protein_mask.bool(),
            x0_weight,
        )
        protein_x0_delta = _pbc_displacement_bt(refined_bb - target_bb, boxes)
        protein_x0 = _masked_weighted_smooth_l1(
            protein_x0_delta,
            torch.zeros_like(protein_x0_delta),
            protein_mask.bool(),
            x0_weight,
            horizon_gamma=float(DIFFUSION_ABSOLUTE_ANCHOR_HORIZON_GAMMA),
        )
        coord_diffusion = _masked_smooth_l1(coord_eps, coord_noise, coord_increment_mask[:, None])
        pocket_diffusion = _masked_smooth_l1(pocket_eps, coord_noise, pocket_increment_mask[:, None])
        ligand_diffusion = _masked_smooth_l1(ligand_eps, coord_noise, ligand_node_mask[:, None])
        coord_increment_x0_loss = _masked_weighted_smooth_l1(
            coord_increment_x0,
            coord_increment_target,
            coord_increment_mask[:, None],
            x0_weight,
        )
        coord_x0_delta = _pbc_displacement_bt(refined_coords - target_coords, boxes)
        coord_x0 = _masked_weighted_smooth_l1(
            coord_x0_delta,
            torch.zeros_like(coord_x0_delta),
            node_mask[:, None].bool(),
            x0_weight,
            horizon_gamma=float(DIFFUSION_ABSOLUTE_ANCHOR_HORIZON_GAMMA),
        )
        target_mapped_z = self._mapped_backbone_context(target_bb, node_residue_index)
        # Ground-truth density points and their grid geometry enter ONLY here,
        # after the coordinate prediction has already been constructed.
        if qm_mask is None or not bool(qm_mask.any()):
            if bool(QM_DENSITY_REQUIRE_LABELS):
                raise ValueError('No QM density labels; import aligned joint-density CUBE files before training')
            if not self._warned_missing_qm:
                warnings.warn('No QM density labels: joint field receives coordinate supervision only; it is NOT QM-supervised.', RuntimeWarning)
                self._warned_missing_qm = True
        elif bool(QM_DENSITY_REQUIRE_LABELS) and not bool(qm_mask.any(-1).all()):
            raise ValueError('QM_DENSITY_REQUIRE_LABELS requires labels for every future frame in the batch')
        qm_density_loss, qm_density_coverage = self.joint_density.supervision_loss(
            joint_field, qm_points, qm_density, qm_mask,
        )
        target_node_rot, target_node_origin = self._node_frames(target_bb, protein_mask, node_residue_index)
        target_pocket_cloud_z = self._pocket_cloud_from_coords(
            target_coords,
            node_feat,
            protein_node_mask,
            target_mapped_z,
            target_node_rot,
            target_node_origin,
            node_mask,
            node_is_ligand,
            node_residue_index,
            boxes=boxes,
        )
        _predicted_target_query_membrane_z, predicted_target_query_descriptor = self._electron_membrane(
            refined_protein_coords,
            target_coords,
            refined_bb,
            protein_mask,
            node_feat,
            node_mask,
            node_is_ligand,
            node_residue_index,
            boxes=boxes,
            return_descriptor=True,
        )
        _target_electron_membrane_z, target_electron_descriptor = self._electron_membrane(
            target_coords,
            target_coords,
            target_bb,
            protein_mask,
            node_feat,
            node_mask,
            node_is_ligand,
            node_residue_index,
            boxes=boxes,
            return_descriptor=True,
        )
        pocket_rigid = _masked_smooth_l1(
            _pbc_displacement_bt(pocket_rigid_prior - target_coords, boxes),
            torch.zeros_like(target_coords),
            protein_node_mask[:, None],
        )
        cloud_hierarchy = (
            torch.nn.functional.smooth_l1_loss(pocket_cloud_z, target_pocket_cloud_z.detach())
            + _masked_smooth_l1(
                predicted_target_query_descriptor,
                target_electron_descriptor.detach(),
                ligand_node_mask[:, None],
            )
            + float(MD_HIERARCHY_RIGID_POCKET_WEIGHT) * pocket_rigid
        ) / (2.0 + float(MD_HIERARCHY_RIGID_POCKET_WEIGHT))
        _debug_log(
            f"trajectory_refine_losses_done seconds={_debug_now(rough_coords.device) - loss_t:.3f} "
            f"total_refine_s={_debug_now(rough_coords.device) - start_t:.3f}"
        )

        return {
            "refined_coords": _finite_values(refined_coords),
            "refined_protein_bb": _finite_values(refined_bb),
            "rough_coords_full": _finite_values(rough_coords),
            "hierarchy_coords_full": _finite_values(hierarchy_coords),
            "rough_protein_bb_full": _finite_values(rough_bb),
            "cloud_z": _finite_values(cloud_z),
            "pocket_cloud_z": _finite_values(pocket_cloud_z),
            "electron_membrane_z": _finite_values(electron_membrane_z),
            "protein_diffusion": _finite_loss(protein_diffusion),
            "protein_increment_x0": _finite_loss(protein_increment_x0),
            "protein_x0": _finite_loss(protein_x0),
            "coord_diffusion": _finite_loss(coord_diffusion),
            "pocket_diffusion": _finite_loss(pocket_diffusion),
            "ligand_diffusion": _finite_loss(ligand_diffusion),
            "coord_increment_x0": _finite_loss(coord_increment_x0_loss),
            "coord_x0": _finite_loss(coord_x0),
            "cloud_hierarchy": _finite_loss(cloud_hierarchy),
            "qm_density_loss": qm_density_loss,
            "qm_density_coverage": qm_density_coverage,
            "pocket_rigid": _finite_loss(pocket_rigid),
            "diffusion_t_mean": _finite_loss(t.float().mean() / float(max(1, self.num_steps - 1))),
            "diffusion_x0_weight_mean": _finite_loss(x0_weight.float().mean()),
            "coord_increment_target_rms_A_per_tau": _masked_rms(
                coord_increment_target,
                coord_increment_mask[:, None],
                COORD_SCALE,
            ),
            "protein_increment_target_rms_A_per_tau": _masked_rms(
                bb_increment_target,
                protein_mask.bool(),
                COORD_SCALE,
            ),
            "coord_noise_rms": _masked_rms(coord_noise, coord_increment_mask[:, None], 1.0),
            "protein_noise_rms": _masked_rms(bb_noise, protein_mask.bool(), 1.0),
            "coord_eps_rms": _masked_rms(coord_eps, coord_increment_mask[:, None], 1.0),
            "protein_eps_rms": _masked_rms(bb_eps, protein_mask.bool(), 1.0),
            "coord_x0_rms_A": _masked_rms(coord_x0_delta, node_mask[:, None], COORD_SCALE),
            "protein_x0_rms_A": _masked_rms(protein_x0_delta, protein_mask.bool(), COORD_SCALE),
        }

    @torch.no_grad()
    def sample_refine(
        self,
        rough_coords,
        rough_bb,
        source_coords,
        source_bb,
        node_feat,
        node_mask,
        node_is_ligand,
        protein_mask,
        sample_steps=None,
        protein_chain_pair_mask=None,
        keyframe_indices=None,
        target_steps=None,
        node_residue_index=None,
        node_backbone_slot=None,
        source_velocity=None,
        source_bb_velocity=None,
        source_box=None,
        source_time=None,
        future_times=None,
        future_dt_scaled=None,
        boxes=None,
    ):
        target_steps = int(protein_mask.shape[1] if target_steps is None else target_steps)
        rough_coords, rough_bb = self._expand_rough_condition(
            rough_coords,
            rough_bb,
            target_steps=target_steps,
            keyframe_indices=keyframe_indices,
            target_times=future_times,
        )
        B, T = rough_coords.shape[:2]
        sample_steps = int(sample_steps or SAMPLE_DIFFUSION_STEPS)
        sample_steps = max(1, min(sample_steps, self.num_steps))
        schedule = torch.linspace(self.num_steps - 1, 0, sample_steps, device=rough_coords.device).long().unique_consecutive()
        physical_t_emb, frame_dt = self._physical_time_context(
            future_dt_scaled,
            B,
            T,
            rough_coords.device,
            rough_coords.dtype,
            source_time=source_time,
            future_times=future_times,
        )
        if node_residue_index is None:
            raise ValueError("dynamic pockets require residue mapping from schema-4 caches")
        if node_backbone_slot is None:
            node_backbone_slot = torch.full_like(node_residue_index, -1)

        protein_valid = protein_mask.bool()[:, :, :, None, None]
        node_valid = node_mask.bool()[:, None, :, None]
        protein_node_mask = node_mask.bool() & ~node_is_ligand.bool()
        ligand_node_mask = node_mask.bool() & node_is_ligand.bool()
        pocket_increment_mask = protein_node_mask & (node_backbone_slot < 0)
        coord_increment_mask = pocket_increment_mask | ligand_node_mask
        coord_increment_valid = coord_increment_mask[:, None, :, None]

        previous_rough_bb = _previous_frames(rough_bb, source_bb)
        previous_protein_mask = torch.cat([protein_mask[:, :1], protein_mask[:, :-1]], dim=1)
        previous_bb_rot, _previous_bb_origin = _backbone_frames(previous_rough_bb)
        bb_increment_rot = previous_bb_rot.unsqueeze(-3)
        node_rot, node_origin = self._node_frames(
            previous_rough_bb,
            previous_protein_mask,
            node_residue_index,
        )
        bb_increment = torch.where(protein_valid, torch.randn_like(rough_bb), torch.zeros_like(rough_bb))
        coord_increment = torch.where(
            coord_increment_valid,
            torch.randn_like(rough_coords),
            torch.zeros_like(rough_coords),
        )
        bb_x0 = rough_bb
        coord_x0 = rough_coords
        bb_increment_x0 = torch.zeros_like(rough_bb)
        coord_increment_x0 = torch.zeros_like(rough_coords)
        pocket_cloud_z = rough_coords.new_zeros((B, T, self.d_model))
        electron_membrane_z = rough_coords.new_zeros((B, T, rough_coords.shape[2], self.d_model))
        hierarchy_coords = rough_coords
        for step_idx, t_scalar in enumerate(schedule):
            t = t_scalar.expand(B)
            next_t = int(schedule[step_idx + 1].item()) if step_idx + 1 < schedule.numel() else -1
            t_emb = self.time_embed(t).to(dtype=rough_coords.dtype)[:, None].expand(-1, T, -1)
            bb_eps = self._protein_eps(
                bb_increment,
                rough_bb,
                source_bb,
                source_bb_velocity,
                protein_mask,
                t_emb,
                physical_t_emb,
                frame_dt,
                boxes=boxes,
                source_box=source_box,
            )
            bb_increment_x0 = self._predict_x0(bb_increment, t, bb_eps)
            bb_increment_x0 = torch.where(protein_valid, bb_increment_x0, torch.zeros_like(bb_increment_x0))
            bb_x0, _bb_predicted_increment = _integrate_local_increment_residual_velocity(
                rough_bb,
                source_bb,
                bb_increment_x0,
                bb_increment_rot,
                frame_dt,
                boxes=boxes,
                source_box=source_box,
            )
            bb_x0 = torch.where(protein_valid, bb_x0, rough_bb)
            if bool(PROTEIN_BACKBONE_PROJECT_EACH_DIFFUSION_STEP):
                bb_x0 = _project_backbone_chain_geometry(
                    bb_x0,
                    protein_mask,
                    chain_pair_mask=protein_chain_pair_mask,
                )
                bb_x0 = torch.where(protein_valid, bb_x0, rough_bb)
                bb_increment_x0 = _local_increment_residual_velocity(
                    bb_x0,
                    rough_bb,
                    source_bb,
                    frame_dt,
                    bb_increment_rot,
                    boxes=boxes,
                    source_box=source_box,
                )
                bb_increment_x0 = torch.where(
                    protein_valid,
                    bb_increment_x0,
                    torch.zeros_like(bb_increment_x0),
                )
            bb_increment = self._ddim_step(
                bb_increment,
                t,
                next_t,
                bb_eps,
                x0=bb_increment_x0,
            )
            bb_increment = torch.where(protein_valid, bb_increment, torch.zeros_like(bb_increment))

            cloud_z = self._cloud_from_backbone(bb_x0, protein_mask, boxes=boxes)
            mapped_backbone_z = self._mapped_backbone_context(bb_x0, node_residue_index)
            pocket_rigid_prior = restore_pocket_from_backbone(
                source_coords,
                source_bb,
                bb_x0,
                node_residue_index,
                protein_node_mask,
                source_box=source_box,
            )
            pocket_rigid_prior = enforce_backbone_nodes(
                pocket_rigid_prior,
                bb_x0,
                node_residue_index,
                node_backbone_slot,
            )
            hierarchy_coords = torch.where(
                protein_node_mask[:, None, :, None],
                pocket_rigid_prior,
                rough_coords,
            )
            node_diffusion_feat = self._node_diffusion_features(
                coord_increment,
                hierarchy_coords,
                source_coords,
                source_velocity,
                node_feat,
                node_is_ligand,
                cloud_z,
                mapped_backbone_z,
                t_emb,
                physical_t_emb,
                frame_dt,
                node_rot,
                boxes=boxes,
                source_box=source_box,
            )
            pocket_eps = self.pocket_eps(node_diffusion_feat)
            pocket_increment_x0 = self._predict_x0(coord_increment, t, pocket_eps)
            pocket_increment_x0 = torch.where(
                coord_increment_valid,
                pocket_increment_x0,
                torch.zeros_like(pocket_increment_x0),
            )
            pocket_coords_x0, _pocket_predicted_increment = _integrate_local_increment_residual_velocity(
                hierarchy_coords,
                source_coords,
                pocket_increment_x0,
                node_rot,
                frame_dt,
                boxes=boxes,
                source_box=source_box,
            )
            refined_protein_coords = torch.where(
                protein_node_mask[:, None, :, None],
                pocket_coords_x0,
                hierarchy_coords,
            )
            refined_protein_coords = enforce_backbone_nodes(
                refined_protein_coords,
                bb_x0,
                node_residue_index,
                node_backbone_slot,
            )
            pocket_cloud_z = self._pocket_cloud_from_coords(
                refined_protein_coords,
                node_feat,
                protein_node_mask,
                mapped_backbone_z,
                node_rot,
                node_origin,
                node_mask,
                node_is_ligand,
                node_residue_index,
                boxes=boxes,
            )
            noisy_query_coords, _noisy_query_increment = _integrate_local_increment_residual_velocity(
                hierarchy_coords,
                source_coords,
                coord_increment,
                node_rot,
                frame_dt,
                boxes=boxes,
                source_box=source_box,
            )
            electron_membrane_z = self._electron_membrane(
                refined_protein_coords,
                noisy_query_coords,
                bb_x0,
                protein_mask,
                node_feat,
                node_mask,
                node_is_ligand,
                node_residue_index,
                boxes=boxes,
            )
            joint_field = self.joint_density.predict(
                refined_protein_coords, node_feat, node_mask, node_is_ligand,
                node_residue_index, node_rot, mapped_backbone_z, pocket_cloud_z, boxes,
            )
            joint_density_z = self.joint_density.ligand_context(joint_field, noisy_query_coords, node_rot)
            ligand_node_diffusion_feat = self._node_diffusion_features(
                coord_increment,
                hierarchy_coords,
                source_coords,
                source_velocity,
                node_feat,
                node_is_ligand,
                torch.zeros_like(cloud_z),
                torch.zeros_like(mapped_backbone_z),
                t_emb,
                physical_t_emb,
                frame_dt,
                node_rot,
                boxes=boxes,
                source_box=source_box,
            )
            ligand_feat = torch.cat(
                [
                    ligand_node_diffusion_feat,
                    pocket_cloud_z[:, :, None].expand(-1, -1, rough_coords.shape[2], -1),
                    electron_membrane_z,
                    joint_density_z,
                ],
                dim=-1,
            )
            ligand_eps = self.ligand_eps(ligand_feat)
            coord_eps = torch.where(ligand_node_mask[:, None, :, None], ligand_eps, pocket_eps)
            coord_increment_x0 = self._predict_x0(coord_increment, t, coord_eps)
            coord_increment_x0 = torch.where(
                coord_increment_valid,
                coord_increment_x0,
                torch.zeros_like(coord_increment_x0),
            )
            coord_x0, _coord_predicted_increment = _integrate_local_increment_residual_velocity(
                hierarchy_coords,
                source_coords,
                coord_increment_x0,
                node_rot,
                frame_dt,
                boxes=boxes,
                source_box=source_box,
            )
            coord_x0 = torch.where(node_valid, coord_x0, hierarchy_coords)
            coord_x0 = enforce_backbone_nodes(
                coord_x0,
                bb_x0,
                node_residue_index,
                node_backbone_slot,
            )
            coord_increment = self._ddim_step(
                coord_increment,
                t,
                next_t,
                coord_eps,
                x0=coord_increment_x0,
            )
            coord_increment = torch.where(
                coord_increment_valid,
                coord_increment,
                torch.zeros_like(coord_increment),
            )

        final_bb = _project_backbone_chain_geometry(
            bb_x0,
            protein_mask,
            chain_pair_mask=protein_chain_pair_mask,
        )
        final_bb = torch.where(protein_valid, final_bb, rough_bb)
        final_pocket_rigid = restore_pocket_from_backbone(
            source_coords,
            source_bb,
            final_bb,
            node_residue_index,
            protein_node_mask,
            source_box=source_box,
        )
        final_hierarchy_coords = torch.where(
            protein_node_mask[:, None, :, None],
            final_pocket_rigid,
            rough_coords,
        )
        coord_x0, _final_coord_increment = _integrate_local_increment_residual_velocity(
            final_hierarchy_coords,
            source_coords,
            coord_increment_x0,
            node_rot,
            frame_dt,
            boxes=boxes,
            source_box=source_box,
        )
        coord_x0 = torch.where(node_valid, coord_x0, final_hierarchy_coords)
        coord_x0 = enforce_backbone_nodes(
            coord_x0,
            final_bb,
            node_residue_index,
            node_backbone_slot,
        )
        final_cloud_z = self._cloud_from_backbone(final_bb, protein_mask, boxes=boxes)
        final_mapped_backbone_z = self._mapped_backbone_context(final_bb, node_residue_index)
        final_pocket_cloud_z = self._pocket_cloud_from_coords(
            coord_x0,
            node_feat,
            protein_node_mask,
            final_mapped_backbone_z,
            node_rot,
            node_origin,
            node_mask,
            node_is_ligand,
            node_residue_index,
            boxes=boxes,
        )
        final_electron_membrane_z = self._electron_membrane(
            coord_x0,
            coord_x0,
            final_bb,
            protein_mask,
            node_feat,
            node_mask,
            node_is_ligand,
            node_residue_index,
            boxes=boxes,
        )
        return {
            "refined_coords": _finite_values(coord_x0),
            "refined_protein_bb": _finite_values(final_bb),
            "rough_coords_full": _finite_values(rough_coords),
            "hierarchy_coords_full": _finite_values(final_hierarchy_coords),
            "rough_protein_bb_full": _finite_values(rough_bb),
            "cloud_z": _finite_values(final_cloud_z),
            "pocket_cloud_z": _finite_values(final_pocket_cloud_z),
            "electron_membrane_z": _finite_values(final_electron_membrane_z),
            # The last field actually consumed by the ligand denoiser. Recomputing
            # after final coordinates would describe a different conditioning field.
            "predicted_joint_density_field": joint_field,
            "active_pocket_mask": dynamic_residue_pocket_mask(
                coord_x0, coord_x0, node_mask, node_is_ligand, node_residue_index,
                boxes=boxes if bool(MD_USE_PBC) else None,
                cutoff=float(LIGAND_NEAR_RES_CUTOFF) / float(COORD_SCALE),
            ),
        }
