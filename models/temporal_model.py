import torch
import torch.nn as nn

from config import (
    CLOUD_BOX_PADDING_A, CLOUD_GAUSS_SIGMA_A, CLOUD_GRID_SIZE,
    CLOUD_MAX_VOXEL_A, CLOUD_MIN_VOXEL_A, CONDITION_PROTEIN_ONLY_CLOUD, COORD_SCALE, DIFFUSION_STEPS,
    FUTURE_FRAMES, LSTM_KEYFRAME_COUNT, MODEL_DIM, OBSERVED_FRAMES, ROLLOUT_FAST_PROTEIN_CLOUD,
    DYNAMIC_GRAPH_CHUNK_SIZE, DYNAMIC_GRAPH_ENABLED, DYNAMIC_GRAPH_MAX_NEIGHBORS,
    DYNAMIC_GRAPH_RADIUS_A, MD_USE_PBC, LIGAND_NEAR_RES_CUTOFF,
)
from utils.ecloud import cloud_frame_from_atoms, protein_ligand_cloud_grid
from utils.md import dynamic_batched_graph, minimum_image_displacement_batched
from utils.pocket import dynamic_residue_pocket_mask
from .temporal_components import GatedMultimodalLSTMCell
from .temporal_inference import TemporalInferenceMixin
from .temporal_training import TemporalTrainingMixin
from .temporal_utils import _debug_every_frames, _debug_log, _debug_now
from .trajectory_refinement import (
    TrajectoryRefinementDiffusion,
    enforce_backbone_nodes,
    restore_pocket_from_backbone,
)


class TemporalCloudDiffusionModel(TemporalTrainingMixin, TemporalInferenceMixin, nn.Module):
    """
    LSTM-centered trajectory model.

    Per frame:
      graph-cache tensors -> three internal memory gates -> X0 matrix
      X0 -> internal frame diffusion -> y0
      y0 -> LSTM state update

    Future prediction first produces a coarse autoregressive trajectory.  A
    joint diffusion then denoises adjacent-frame increment corrections over the
    full horizon and cumulatively integrates them from the last observed frame,
    with an absolute-coordinate anchor and the backbone -> pocket -> electron
    membrane -> ligand reconstruction hierarchy.
    """

    def __init__(self, observed_frames=None, future_frames=None, diffusion_steps=None):
        super().__init__()
        self.observed_frames = int(OBSERVED_FRAMES if observed_frames is None else observed_frames)
        self.future_frames = int(FUTURE_FRAMES if future_frames is None else future_frames)
        self.diffusion_steps = int(DIFFUSION_STEPS if diffusion_steps is None else diffusion_steps)
        self.d_model = int(MODEL_DIM)
        self.cell = GatedMultimodalLSTMCell(self.d_model)
        self.trajectory_refinement = TrajectoryRefinementDiffusion(
            self.d_model,
            diffusion_steps=self.diffusion_steps,
        )
        self.last_loss_dict = {}

    def _future_keyframe_indices(self, total_steps, device):
        total_steps = max(1, int(total_steps))
        keyframes = max(1, min(int(LSTM_KEYFRAME_COUNT), total_steps))
        if keyframes >= total_steps:
            return torch.arange(total_steps, device=device, dtype=torch.long)
        idx = torch.linspace(0, total_steps - 1, keyframes, device=device).round().long()
        idx[0] = 0
        idx[-1] = total_steps - 1
        return idx

    def _select_future_keyframes(self, x, keyframe_idx):
        return x.index_select(1, keyframe_idx.to(device=x.device, dtype=torch.long))

    def _zero_state(self, batch_size, device, dtype):
        h = torch.zeros((batch_size, self.d_model), device=device, dtype=dtype)
        c = torch.zeros_like(h)
        return h, c

    def _condition_cloud(self, cloud_grid):
        if not bool(CONDITION_PROTEIN_ONLY_CLOUD):
            return cloud_grid
        if cloud_grid.shape[1] < 2:
            return cloud_grid
        cloud_grid = cloud_grid.clone()
        cloud_grid[:, 1:] = 0.0
        return cloud_grid

    def _encode_frame(
        self,
        batch,
        coords,
        protein_bb,
        protein_mask,
        cloud_grid,
        h_prev=None,
        c_prev=None,
        protein_temporal_state=None,
        velocity=None,
        protein_velocity=None,
        frame_dt=None,
        box=None,
    ):
        cloud_grid = self._condition_cloud(cloud_grid)
        edge_index = batch["edge_index"]
        k_idx = batch["k_idx"]
        l_idx = batch["l_idx"]
        boxes = box if bool(MD_USE_PBC) else None
        if bool(DYNAMIC_GRAPH_ENABLED):
            edge_index, k_idx, l_idx = dynamic_batched_graph(
                coords,
                batch["node_mask"],
                bond_edge_index=batch.get("bond_edge_index"),
                boxes=boxes,
                cutoff=float(DYNAMIC_GRAPH_RADIUS_A) / float(COORD_SCALE),
                max_neighbors=int(DYNAMIC_GRAPH_MAX_NEIGHBORS),
                chunk_size=int(DYNAMIC_GRAPH_CHUNK_SIZE),
            )
        return self.cell.encode_frame(
            coords=coords,
            protein_bb=protein_bb,
            protein_bb_mask=protein_mask,
            cloud_grid=cloud_grid,
            node_feat=batch["node_feat"],
            node_mask=batch["node_mask"],
            node_is_ligand=batch["node_is_ligand"],
            edge_index=edge_index,
            k_idx=k_idx,
            l_idx=l_idx,
            node_batch=batch["node_batch"],
            protein_seq_ids=batch["protein_full_seq_ids"],
            protein_full_mask=batch["protein_full_mask"],
            protein_chain_pair_mask=batch.get("protein_chain_pair_mask"),
            protein_temporal_state=protein_temporal_state,
            h_prev=h_prev,
            c_prev=c_prev,
            velocity=velocity,
            protein_velocity=protein_velocity,
            frame_dt=frame_dt,
            boxes=boxes,
        )

    def _step_frame(
        self,
        batch,
        coords,
        protein_bb,
        protein_mask,
        cloud_grid,
        h,
        c,
        protein_temporal_state=None,
        velocity=None,
        protein_velocity=None,
        frame_dt=None,
        box=None,
    ):
        enc = self._encode_frame(
            batch,
            coords,
            protein_bb,
            protein_mask,
            cloud_grid,
            h_prev=h,
            c_prev=c,
            protein_temporal_state=protein_temporal_state,
            velocity=velocity,
            protein_velocity=protein_velocity,
            frame_dt=frame_dt,
            box=box,
        )
        h, c, enc = self.cell(enc, h, c)
        return enc, h, c, enc["protein_temporal_state"]

    def _rollout_internal_frames(
        self,
        batch,
        h,
        c,
        source_coords,
        source_bb,
        source_protein_mask,
        source_cloud_grid,
        steps,
        protein_mask_seq=None,
        protein_temporal_state=None,
        source_velocity=None,
        source_bb_velocity=None,
        step_dts=None,
        step_boxes=None,
    ):
        pred_coords = []
        pred_bb = []
        cur_coords = source_coords
        cur_bb = source_bb
        cur_mask = source_protein_mask
        cur_cloud = source_cloud_grid
        cur_box = batch.get("source_box")
        cur_velocity = torch.zeros_like(source_coords) if source_velocity is None else source_velocity
        cur_bb_velocity = torch.zeros_like(source_bb) if source_bb_velocity is None else source_bb_velocity
        rollout_t = _debug_now(source_coords.device)
        _debug_log(
            f"internal_rollout_start steps={int(steps)} coords={tuple(source_coords.shape)} "
            f"bb={tuple(source_bb.shape)} cloud={tuple(source_cloud_grid.shape)}"
        )
        for frame_idx in range(int(steps)):
            frame_dt = (
                source_coords.new_ones((source_coords.shape[0],))
                if step_dts is None
                else step_dts[:, min(frame_idx, step_dts.shape[1] - 1)].to(dtype=source_coords.dtype)
            )
            next_box = None if step_boxes is None else step_boxes[:, min(frame_idx, step_boxes.shape[1] - 1)]
            frame_t = _debug_now(source_coords.device)
            _debug_log(
                f"internal_rollout_frame_start frame={frame_idx + 1}/{int(steps)} "
                f"coords={tuple(cur_coords.shape)} bb={tuple(cur_bb.shape)} cloud={tuple(cur_cloud.shape)}"
            )
            step_t = _debug_now(source_coords.device)
            enc, h, c, protein_temporal_state = self._step_frame(
                batch,
                cur_coords,
                cur_bb,
                cur_mask,
                cur_cloud,
                h,
                c,
                protein_temporal_state,
                velocity=cur_velocity,
                protein_velocity=cur_bb_velocity,
                frame_dt=frame_dt,
                box=cur_box,
            )
            step_s = _debug_now(source_coords.device) - step_t
            next_coords = enc["pred_coords"]
            next_bb = enc["pred_protein_bb"]
            protein_node_mask = batch["node_mask"].bool() & ~batch["node_is_ligand"].bool()
            rigid_pocket = restore_pocket_from_backbone(
                cur_coords,
                cur_bb,
                next_bb[:, None],
                batch["node_residue_index"],
                protein_node_mask,
                source_box=cur_box,
            )[:, 0]
            next_coords = torch.where(protein_node_mask[:, :, None], rigid_pocket, next_coords)
            next_coords = enforce_backbone_nodes(
                next_coords[:, None],
                next_bb[:, None],
                batch["node_residue_index"],
                batch.get("node_backbone_slot"),
            )[:, 0]
            enc["pred_coords"] = next_coords
            pred_coords.append(next_coords)
            pred_bb.append(next_bb)

            dt_coords = frame_dt[:, None, None].clamp_min(1.0e-8)
            dt_bb = frame_dt[:, None, None, None].clamp_min(1.0e-8)
            coord_step = next_coords - cur_coords
            bb_step = next_bb - cur_bb
            if bool(MD_USE_PBC) and cur_box is not None:
                coord_step = minimum_image_displacement_batched(coord_step, cur_box)
                bb_step = minimum_image_displacement_batched(bb_step, cur_box)
            next_velocity = coord_step / dt_coords
            next_bb_velocity = bb_step / dt_bb
            cur_coords = next_coords
            cur_bb = next_bb
            cur_velocity = next_velocity
            cur_bb_velocity = next_bb_velocity
            cur_box = next_box
            if protein_mask_seq is not None:
                cur_mask = protein_mask_seq[:, min(frame_idx, protein_mask_seq.shape[1] - 1)]
            cloud_t = _debug_now(source_coords.device)
            cloud_grid, _cloud_center, _cloud_voxel = self._rebuild_cloud_from_protein_coords(batch, next_coords, boxes=next_box)
            cloud_s = _debug_now(source_coords.device) - cloud_t
            cur_cloud = cloud_grid
            if frame_idx == 0 or frame_idx == int(steps) - 1 or (frame_idx + 1) % _debug_every_frames() == 0:
                _debug_log(
                    f"internal_rollout_frame_done frame={frame_idx + 1}/{int(steps)} "
                    f"step_s={step_s:.3f} cloud_s={cloud_s:.3f} total_s={_debug_now(source_coords.device) - frame_t:.3f} "
                    f"next_coords={tuple(next_coords.shape)} next_bb={tuple(next_bb.shape)}"
                )
        stack_t = _debug_now(source_coords.device)
        pred_coords = torch.stack(pred_coords, dim=1)
        pred_bb = torch.stack(pred_bb, dim=1)
        _debug_log(
            f"internal_rollout_done seconds={_debug_now(source_coords.device) - rollout_t:.3f} "
            f"stack_s={_debug_now(source_coords.device) - stack_t:.3f} pred_coords={tuple(pred_coords.shape)} "
            f"pred_bb={tuple(pred_bb.shape)}"
        )
        return {
            "pred_coords": pred_coords,
            "pred_protein_bb": pred_bb,
            "h": h,
            "c": c,
            "protein_temporal_state": protein_temporal_state,
            "pred_velocity": cur_velocity,
            "pred_protein_velocity": cur_bb_velocity,
        }

    def _rebuild_cloud_from_protein_bb(self, batch, protein_bb):
        if bool(ROLLOUT_FAST_PROTEIN_CLOUD):
            return self._fast_rebuild_cloud_from_protein_bb(batch, protein_bb)
        clouds = []
        centers = []
        voxels = []
        bb_cpu = protein_bb.detach().float().cpu()
        pocket_cpu = batch["pocket_residue_indices"].detach().long().cpu()
        pocket_mask_cpu = batch["pocket_residue_mask"].detach().bool().cpu()
        protein_mask_cpu = batch["protein_full_mask"].detach().bool().cpu()
        cloud_device = str(protein_bb.device)
        if cloud_device == "cpu":
            cloud_device = "cpu"
        atom_weights = torch.tensor([7.0, 6.0, 6.0], dtype=torch.float32)
        for b in range(protein_bb.shape[0]):
            valid_idx = pocket_cpu[b, pocket_mask_cpu[b]]
            valid_idx = valid_idx[(valid_idx >= 0) & (valid_idx < bb_cpu.shape[1])]
            if valid_idx.numel() == 0:
                valid_idx = protein_mask_cpu[b].nonzero(as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                valid_idx = torch.arange(min(1, bb_cpu.shape[1]), dtype=torch.long)
            coords_A = bb_cpu[b, valid_idx].reshape(-1, 3).numpy() * float(COORD_SCALE)
            weights = atom_weights.repeat(max(1, int(valid_idx.numel()))).numpy()
            is_ligand = torch.zeros((coords_A.shape[0],), dtype=torch.bool).numpy()
            center_A, voxel_A, _half_width_A = cloud_frame_from_atoms(
                coords_A,
                grid_size=int(CLOUD_GRID_SIZE),
                padding_A=float(CLOUD_BOX_PADDING_A),
                min_voxel_A=float(CLOUD_MIN_VOXEL_A),
                max_voxel_A=float(CLOUD_MAX_VOXEL_A),
            )
            clouds.append(
                protein_ligand_cloud_grid(
                    coords_A,
                    weights,
                    is_ligand,
                    center_A,
                    grid_size=int(CLOUD_GRID_SIZE),
                    voxel_A=float(voxel_A),
                    sigma_A=float(CLOUD_GAUSS_SIGMA_A),
                    device=cloud_device,
                )
            )
            centers.append(torch.as_tensor(center_A / float(COORD_SCALE), dtype=protein_bb.dtype))
            voxels.append(torch.as_tensor(float(voxel_A) / float(COORD_SCALE), dtype=protein_bb.dtype))
        cloud_grid = torch.stack(clouds, dim=0).to(device=protein_bb.device, dtype=protein_bb.dtype)
        cloud_center = torch.stack(centers, dim=0).to(device=protein_bb.device, dtype=protein_bb.dtype)
        cloud_voxel = torch.stack(voxels, dim=0).to(device=protein_bb.device, dtype=protein_bb.dtype)
        return cloud_grid, cloud_center, cloud_voxel

    def _current_pocket_cloud_inputs(self, batch, coords, boxes=None):
        active = dynamic_residue_pocket_mask(
            coords[:, None], coords[:, None], batch["node_mask"], batch["node_is_ligand"],
            batch["node_residue_index"],
            boxes=boxes[:, None] if bool(MD_USE_PBC) and boxes is not None else None,
            cutoff=float(LIGAND_NEAR_RES_CUTOFF) / float(COORD_SCALE),
        )[:, 0]
        lig = batch["node_mask"].bool() & batch["node_is_ligand"].bool()
        if bool(MD_USE_PBC) and boxes is not None:
            center = (coords * lig[..., None]).sum(1) / lig.sum(1).clamp_min(1)[:, None]
            coords = center[:, None] + minimum_image_displacement_batched(coords - center[:, None], boxes)
        return coords, active | lig

    def _rebuild_cloud_from_protein_coords(self, batch, protein_coords, boxes=None):
        """Rebuild from current predicted 10-Angstrom whole-residue membership."""
        if bool(ROLLOUT_FAST_PROTEIN_CLOUD):
            return self._fast_rebuild_cloud_from_protein_coords(batch, protein_coords, boxes=boxes)
        clouds = []
        centers = []
        voxels = []
        cloud_coords, cloud_mask = self._current_pocket_cloud_inputs(batch, protein_coords, boxes)
        coords_cpu = cloud_coords.detach().float().cpu()
        mask_cpu = cloud_mask.cpu()
        weights_cpu = (batch["node_feat"][..., 0].detach().float().cpu() * 53.0).clamp_min(1.0)
        cloud_device = "cpu" if protein_coords.device.type == "cpu" else str(protein_coords.device)
        for b in range(protein_coords.shape[0]):
            valid = mask_cpu[b].nonzero(as_tuple=False).flatten()
            if valid.numel() == 0:
                raise ValueError("pocket cloud requires at least one valid ligand atom")
            coords_A = coords_cpu[b, valid].numpy() * float(COORD_SCALE)
            weights = weights_cpu[b, valid].numpy()
            is_ligand = batch["node_is_ligand"][b].detach().bool().cpu()[valid].numpy()
            center_A, voxel_A, _half_width_A = cloud_frame_from_atoms(
                coords_A,
                grid_size=int(CLOUD_GRID_SIZE),
                padding_A=float(CLOUD_BOX_PADDING_A),
                min_voxel_A=float(CLOUD_MIN_VOXEL_A),
                max_voxel_A=float(CLOUD_MAX_VOXEL_A),
            )
            clouds.append(
                protein_ligand_cloud_grid(
                    coords_A,
                    weights,
                    is_ligand,
                    center_A,
                    grid_size=int(CLOUD_GRID_SIZE),
                    voxel_A=float(voxel_A),
                    sigma_A=float(CLOUD_GAUSS_SIGMA_A),
                    device=cloud_device,
                )
            )
            centers.append(torch.as_tensor(center_A / float(COORD_SCALE), dtype=protein_coords.dtype))
            voxels.append(torch.as_tensor(float(voxel_A) / float(COORD_SCALE), dtype=protein_coords.dtype))
        return (
            torch.stack(clouds, dim=0).to(device=protein_coords.device, dtype=protein_coords.dtype),
            torch.stack(centers, dim=0).to(device=protein_coords.device, dtype=protein_coords.dtype),
            torch.stack(voxels, dim=0).to(device=protein_coords.device, dtype=protein_coords.dtype),
        )

    def _fast_rebuild_cloud_from_protein_coords(self, batch, protein_coords, boxes=None):
        B = int(protein_coords.shape[0])
        D = int(CLOUD_GRID_SIZE)
        cloud_grid = protein_coords.new_zeros((B, 2, D, D, D))
        centers = []
        voxels = []
        protein_coords, cloud_mask = self._current_pocket_cloud_inputs(batch, protein_coords, boxes)
        weights_all = (batch["node_feat"][..., 0].to(dtype=protein_coords.dtype) * 53.0).clamp_min(1.0)
        min_voxel = float(CLOUD_MIN_VOXEL_A) / float(COORD_SCALE)
        max_voxel = float(CLOUD_MAX_VOXEL_A) / float(COORD_SCALE)
        padding = float(CLOUD_BOX_PADDING_A) / float(COORD_SCALE)
        for b in range(B):
            valid = cloud_mask[b].nonzero(as_tuple=False).flatten()
            if valid.numel() == 0:
                raise ValueError("pocket cloud requires at least one valid ligand atom")
            coords = protein_coords[b, valid]
            lo = coords.min(dim=0).values
            hi = coords.max(dim=0).values
            center = 0.5 * (lo + hi)
            half_extent = 0.5 * (hi - lo).max() + coords.new_tensor(padding)
            voxel = (2.0 * half_extent / max(D - 1, 1)).clamp(min=min_voxel, max=max_voxel)
            grid_xyz = torch.round((coords - center) / voxel + (D - 1) * 0.5).long().clamp(0, D - 1)
            flat_idx = grid_xyz[:, 0] * (D * D) + grid_xyz[:, 1] * D + grid_xyz[:, 2]
            for channel in (0, 1):
                use = batch["node_is_ligand"][b, valid].bool() == bool(channel)
                flat_cloud = cloud_grid[b, channel].reshape(-1)
                flat_cloud.scatter_add_(0, flat_idx[use], weights_all[b, valid[use]])
                flat_cloud.div_(flat_cloud.max().clamp_min(torch.finfo(protein_coords.dtype).eps))
            centers.append(center)
            voxels.append(voxel)
        return cloud_grid, torch.stack(centers, dim=0), torch.stack(voxels, dim=0)

    def _fast_rebuild_cloud_from_protein_bb(self, batch, protein_bb):
        B = int(protein_bb.shape[0])
        D = int(CLOUD_GRID_SIZE)
        device = protein_bb.device
        dtype = protein_bb.dtype
        cloud_grid = protein_bb.new_zeros((B, 2, D, D, D))
        centers = []
        voxels = []
        pocket_idx = batch["pocket_residue_indices"].to(device=device, dtype=torch.long)
        pocket_mask = batch["pocket_residue_mask"].to(device=device, dtype=torch.bool)
        protein_mask = batch["protein_full_mask"].to(device=device, dtype=torch.bool)
        atom_weights = protein_bb.new_tensor([7.0, 6.0, 6.0])
        min_voxel = float(CLOUD_MIN_VOXEL_A) / float(COORD_SCALE)
        max_voxel = float(CLOUD_MAX_VOXEL_A) / float(COORD_SCALE)
        padding = float(CLOUD_BOX_PADDING_A) / float(COORD_SCALE)
        for b in range(B):
            valid_idx = pocket_idx[b, pocket_mask[b]]
            valid_idx = valid_idx[(valid_idx >= 0) & (valid_idx < protein_bb.shape[1])]
            if valid_idx.numel() == 0:
                valid_idx = protein_mask[b].nonzero(as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                valid_idx = torch.arange(min(1, protein_bb.shape[1]), device=device, dtype=torch.long)

            coords = protein_bb[b, valid_idx].reshape(-1, 3)
            lo = coords.min(dim=0).values
            hi = coords.max(dim=0).values
            center = 0.5 * (lo + hi)
            half_extent = 0.5 * (hi - lo).max() + coords.new_tensor(padding)
            voxel = (2.0 * half_extent / max(D - 1, 1)).clamp(min=min_voxel, max=max_voxel)
            grid_xyz = torch.round((coords - center) / voxel + (D - 1) * 0.5).long()
            grid_xyz = grid_xyz.clamp(0, D - 1)
            flat_idx = grid_xyz[:, 0] * (D * D) + grid_xyz[:, 1] * D + grid_xyz[:, 2]
            weights = atom_weights.repeat(max(1, int(valid_idx.numel()))).to(dtype=dtype)
            flat_cloud = cloud_grid[b, 0].reshape(-1)
            flat_cloud.scatter_add_(0, flat_idx, weights)
            # Avoid a per-sample GPU-to-CPU synchronization in every rollout.
            flat_cloud.div_(flat_cloud.max().clamp_min(torch.finfo(dtype).eps))
            centers.append(center)
            voxels.append(voxel)
        return cloud_grid, torch.stack(centers, dim=0), torch.stack(voxels, dim=0).to(dtype=dtype)

