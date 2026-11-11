import torch

from config import MD_USE_PBC
from utils.md import minimum_image_displacement_batched


class TemporalInferenceMixin:
    def _future_time_context(self, batch, steps):
        source_time = batch.get("source_time", batch["obs_times"][:, -1]).reshape(-1)
        time_scale = batch.get("time_scale", torch.ones_like(source_time)).reshape(-1).clamp_min(1.0e-8)
        existing = batch.get("future_times")
        use_existing = (
            torch.is_tensor(existing)
            and existing.shape[1] >= int(steps)
            and bool((existing[:, 0] > source_time).all())
        )
        if use_existing:
            future_times = existing[:, : int(steps)]
        else:
            offsets = torch.arange(1, int(steps) + 1, device=source_time.device, dtype=source_time.dtype)
            future_times = source_time[:, None] + time_scale[:, None] * offsets[None]
        prev = torch.cat([source_time[:, None], future_times[:, :-1]], dim=1)
        dt_scaled = (future_times - prev).abs() / time_scale[:, None]

        source_box = batch.get("source_box")
        future_box = batch.get("future_box")
        if torch.is_tensor(future_box) and future_box.shape[1] >= int(steps):
            boxes = future_box[:, : int(steps)]
        elif torch.is_tensor(source_box):
            boxes = source_box[:, None].expand(-1, int(steps), -1, -1)
        else:
            boxes = batch["obs_coords"].new_zeros((batch["obs_coords"].shape[0], int(steps), 3, 3))
        return future_times, dt_scaled, boxes

    @torch.no_grad()
    def _predict_future_once(self, batch, steps=None, sample_steps=None):
        _last_enc, h, c, protein_temporal_state, _gate_loss, _encoder_aux_loss = self._observed_rollout(batch)
        steps = int(self.future_frames if steps is None else steps)
        steps = max(1, steps)
        protein_mask = batch["future_protein_bb_mask"].bool()
        if protein_mask.shape[1] < steps:
            pad = protein_mask[:, -1:].expand(-1, steps - protein_mask.shape[1], -1)
            protein_mask = torch.cat([protein_mask, pad], dim=1)
        protein_mask = protein_mask[:, :steps]
        keyframe_idx = self._future_keyframe_indices(steps, protein_mask.device)
        key_protein_mask = self._select_future_keyframes(protein_mask, keyframe_idx)
        future_times, future_dt_scaled, future_boxes = self._future_time_context(batch, steps)
        key_times = self._select_future_keyframes(future_times, keyframe_idx)
        key_prev = torch.cat([batch["source_time"][:, None], key_times[:, :-1]], dim=1)
        key_dt_scaled = (key_times - key_prev).abs() / batch["time_scale"][:, None].clamp_min(1.0e-8)
        key_boxes = self._select_future_keyframes(future_boxes, keyframe_idx)
        rollout = self._rollout_internal_frames(
            batch,
            h,
            c,
            batch["obs_coords"][:, -1],
            batch["obs_protein_bb"][:, -1],
            batch["obs_protein_bb_mask"][:, -1],
            batch["obs_cloud_grid_seq"][:, -1],
            int(keyframe_idx.numel()),
            protein_mask_seq=key_protein_mask,
            protein_temporal_state=protein_temporal_state,
            source_velocity=batch.get("source_velocity_scaled"),
            source_bb_velocity=batch.get("source_protein_velocity_scaled"),
            step_dts=key_dt_scaled,
            step_boxes=key_boxes,
        )
        refine_out = self.trajectory_refinement.sample_refine(
            rough_coords=rollout["pred_coords"],
            rough_bb=rollout["pred_protein_bb"],
            source_coords=batch["obs_coords"][:, -1],
            source_bb=batch["obs_protein_bb"][:, -1],
            node_feat=batch["node_feat"],
            node_mask=batch["node_mask"],
            node_is_ligand=batch["node_is_ligand"],
            protein_mask=protein_mask,
            sample_steps=sample_steps,
            protein_chain_pair_mask=batch.get("protein_chain_pair_mask"),
            keyframe_indices=keyframe_idx,
            target_steps=steps,
            node_residue_index=batch.get("node_residue_index"),
            node_backbone_slot=batch.get("node_backbone_slot"),
            source_velocity=batch.get("source_velocity_scaled"),
            source_bb_velocity=batch.get("source_protein_velocity_scaled"),
            source_box=batch.get("source_box"),
            source_time=batch.get("source_time"),
            future_times=future_times,
            future_dt_scaled=future_dt_scaled,
            boxes=future_boxes,
        )
        return {
            "pred_coords": refine_out["refined_coords"],
            "pred_protein_bb": refine_out["refined_protein_bb"],
            "rough_coords": refine_out.get("rough_coords_full", rollout["pred_coords"]),
            "rough_protein_bb": refine_out.get("rough_protein_bb_full", rollout["pred_protein_bb"]),
            "rough_keyframe_coords": rollout["pred_coords"],
            "rough_keyframe_protein_bb": rollout["pred_protein_bb"],
            "keyframe_indices": keyframe_idx,
            "keyframe_times": key_times,
            "future_times": future_times,
            "future_boxes": future_boxes,
            "active_pocket_mask": refine_out["active_pocket_mask"],
            "predicted_joint_density_field": refine_out["predicted_joint_density_field"],
        }

    @torch.no_grad()
    def _predict_future_autoregressive(self, batch, steps, sample_steps=None):
        step_batch = dict(batch)
        obs_coords = batch["obs_coords"]
        obs_bb = batch["obs_protein_bb"]
        obs_bb_mask = batch["obs_protein_bb_mask"]
        obs_cloud = batch["obs_cloud_grid_seq"]
        obs_center = batch["obs_cloud_center_seq"]
        obs_voxel = batch["obs_cloud_voxel_seq"]
        obs_velocity = batch.get("obs_velocity")
        obs_bb_velocity = batch.get("obs_protein_velocity")
        obs_times = batch.get("obs_times")
        obs_boxes = batch.get("obs_box")
        requested_future_times = batch.get("future_times")
        requested_future_boxes = batch.get("future_box")
        pred_coords_all = []
        pred_bb_all = []
        pred_times_all = []
        pred_pocket_all = []
        pred_density_all = []
        for step_idx in range(int(steps)):
            step_batch["obs_coords"] = obs_coords
            step_batch["obs_protein_bb"] = obs_bb
            step_batch["obs_protein_bb_mask"] = obs_bb_mask
            step_batch["obs_cloud_grid_seq"] = obs_cloud
            step_batch["obs_cloud_center_seq"] = obs_center
            step_batch["obs_cloud_voxel_seq"] = obs_voxel
            if obs_velocity is not None:
                step_batch["obs_velocity"] = obs_velocity
                step_batch["source_velocity"] = obs_velocity[:, -1]
                step_batch["source_velocity_scaled"] = obs_velocity[:, -1] * step_batch["time_scale"][:, None, None]
            if obs_bb_velocity is not None:
                step_batch["obs_protein_velocity"] = obs_bb_velocity
                step_batch["source_protein_velocity"] = obs_bb_velocity[:, -1]
                step_batch["source_protein_velocity_scaled"] = (
                    obs_bb_velocity[:, -1] * step_batch["time_scale"][:, None, None, None]
                )
            if obs_times is not None:
                step_batch["obs_times"] = obs_times
                step_batch["source_time"] = obs_times[:, -1]
            if obs_boxes is not None:
                step_batch["obs_box"] = obs_boxes
                step_batch["source_box"] = obs_boxes[:, -1]
            if torch.is_tensor(requested_future_times) and step_idx < requested_future_times.shape[1]:
                step_batch["future_times"] = requested_future_times[:, step_idx:step_idx + 1]
            else:
                step_batch.pop("future_times", None)
            if torch.is_tensor(requested_future_boxes) and step_idx < requested_future_boxes.shape[1]:
                step_batch["future_box"] = requested_future_boxes[:, step_idx:step_idx + 1]
            else:
                step_batch.pop("future_box", None)
            step_batch["future_protein_bb_mask"] = obs_bb_mask[:, -1:].clone()
            prev_coords = obs_coords[:, -1]
            prev_bb = obs_bb[:, -1]
            prev_box = None if obs_boxes is None else obs_boxes[:, -1]
            out_i = self._predict_future_once(step_batch, steps=1, sample_steps=sample_steps)
            next_coords = out_i["pred_coords"][:, -1]
            next_bb = out_i["pred_protein_bb"][:, -1]
            pred_coords_all.append(next_coords)
            pred_bb_all.append(next_bb)
            pred_pocket_all.append(out_i["active_pocket_mask"][:, -1])
            pred_density_all.append(out_i["predicted_joint_density_field"])

            next_time = out_i["future_times"][:, -1]
            pred_times_all.append(next_time)
            previous_time = step_batch["source_time"]
            dt = (next_time - previous_time).abs().clamp_min(1.0e-8)
            coord_step = next_coords - prev_coords
            bb_step = next_bb - prev_bb
            if bool(MD_USE_PBC) and prev_box is not None:
                coord_step = minimum_image_displacement_batched(coord_step, prev_box)
                bb_step = minimum_image_displacement_batched(bb_step, prev_box)
            next_velocity = coord_step / dt[:, None, None]
            next_bb_velocity = bb_step / dt[:, None, None, None]
            obs_coords = torch.cat([obs_coords[:, 1:], next_coords[:, None]], dim=1)
            obs_bb = torch.cat([obs_bb[:, 1:], next_bb[:, None]], dim=1)
            obs_bb_mask = torch.cat([obs_bb_mask[:, 1:], obs_bb_mask[:, -1:]], dim=1)
            if obs_velocity is not None:
                obs_velocity = torch.cat([obs_velocity[:, 1:], next_velocity[:, None]], dim=1)
            if obs_bb_velocity is not None:
                obs_bb_velocity = torch.cat([obs_bb_velocity[:, 1:], next_bb_velocity[:, None]], dim=1)
            if obs_times is not None:
                obs_times = torch.cat([obs_times[:, 1:], next_time[:, None]], dim=1)
            next_box = (
                requested_future_boxes[:, step_idx]
                if torch.is_tensor(requested_future_boxes) and step_idx < requested_future_boxes.shape[1]
                else (obs_boxes[:, -1] if obs_boxes is not None else None)
            )
            cloud_grid, cloud_center, cloud_voxel = self._rebuild_cloud_from_protein_coords(step_batch, next_coords, boxes=next_box)
            obs_cloud = torch.cat([obs_cloud[:, 1:], cloud_grid[:, None]], dim=1)
            obs_center = torch.cat([obs_center[:, 1:], cloud_center[:, None]], dim=1)
            obs_voxel = torch.cat([obs_voxel[:, 1:], cloud_voxel[:, None]], dim=1)
            if obs_boxes is not None:
                next_box = (
                    requested_future_boxes[:, step_idx]
                    if torch.is_tensor(requested_future_boxes) and step_idx < requested_future_boxes.shape[1]
                    else obs_boxes[:, -1]
                )
                obs_boxes = torch.cat([obs_boxes[:, 1:], next_box[:, None]], dim=1)

        return {
            "pred_coords": torch.stack(pred_coords_all, dim=1),
            "pred_protein_bb": torch.stack(pred_bb_all, dim=1),
            "future_times": torch.stack(pred_times_all, dim=1),
            "active_pocket_mask": torch.stack(pred_pocket_all, dim=1),
            "predicted_joint_density_field": {
                key: (None if pred_density_all[0][key] is None else
                      pred_density_all[0][key] if key == "ligand_mask" else
                      torch.cat([field[key] for field in pred_density_all], dim=1))
                for key in pred_density_all[0]
            },
        }

    @torch.no_grad()
    def predict_future(self, batch, steps=None, sample_steps=None):
        was_training = self.training
        self.eval()
        requested_steps = None if steps is None else max(1, int(steps))
        if requested_steps is not None and requested_steps > int(self.future_frames) and int(self.future_frames) == 1:
            out = self._predict_future_autoregressive(
                batch,
                steps=requested_steps,
                sample_steps=sample_steps,
            )
        else:
            out = self._predict_future_once(batch, steps=requested_steps, sample_steps=sample_steps)
        future_frame_ids = batch.get("future_frame_ids")
        if future_frame_ids is not None:
            future_ids_tensor = None
            if torch.is_tensor(future_frame_ids):
                future_ids_tensor = future_frame_ids.to(device=out["pred_coords"].device, dtype=torch.long)
            else:
                try:
                    future_ids_tensor = torch.as_tensor(
                        future_frame_ids,
                        device=out["pred_coords"].device,
                        dtype=torch.long,
                    )
                except (TypeError, ValueError):
                    out["future_frame_ids"] = future_frame_ids
            if future_ids_tensor is not None:
                if future_ids_tensor.ndim == 1:
                    future_ids_tensor = future_ids_tensor.unsqueeze(0).expand(out["pred_coords"].shape[0], -1)
                if future_ids_tensor.shape[1] < out["pred_coords"].shape[1]:
                    start = future_ids_tensor[:, -1:] if future_ids_tensor.shape[1] > 0 else None
                    if start is None:
                        frame_id = batch.get("frame_id")
                        if torch.is_tensor(frame_id):
                            start = frame_id.to(device=out["pred_coords"].device, dtype=torch.long).reshape(-1, 1)
                        else:
                            start = torch.zeros((out["pred_coords"].shape[0], 1), device=out["pred_coords"].device, dtype=torch.long)
                    extra = torch.arange(
                        1,
                        out["pred_coords"].shape[1] - future_ids_tensor.shape[1] + 1,
                        device=out["pred_coords"].device,
                        dtype=torch.long,
                    ).reshape(1, -1)
                    future_ids_tensor = torch.cat([future_ids_tensor, start + extra], dim=1)
                out["future_frame_ids"] = future_ids_tensor[:, : out["pred_coords"].shape[1]]
        if was_training:
            self.train()
        return out
