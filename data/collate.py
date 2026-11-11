import torch


def _collate_qm_targets(batch, future_len):
    count = max([item['future_qm_density'].shape[-1] for item in batch if 'future_qm_density' in item] or [1])
    points = torch.zeros((len(batch), future_len, count, 3))
    density = torch.zeros((len(batch), future_len, count))
    mask = torch.zeros((len(batch), future_len, count), dtype=torch.bool)
    for b, item in enumerate(batch):
        if 'future_qm_density' not in item:
            continue
        s = item['future_qm_density'].shape[-1]
        points[b, :, :s] = item['future_qm_points']
        density[b, :, :s] = item['future_qm_density']
        mask[b, :, :s] = item['future_qm_mask']
    return dict(future_qm_points=points, future_qm_density=density, future_qm_mask=mask)


def collate_trajectory_batch(batch):
    B = len(batch)
    obs_len = batch[0]["obs_coords"].shape[0]
    fut_len = batch[0]["future_coords"].shape[0]
    max_nodes = max(item["source_coords"].shape[0] for item in batch)
    node_feat_dim = batch[0]["node_feat"].shape[-1]

    obs_coords = torch.zeros((B, obs_len, max_nodes, 3), dtype=torch.float32)
    obs_velocity = torch.zeros((B, obs_len, max_nodes, 3), dtype=torch.float32)
    future_coords = torch.zeros((B, fut_len, max_nodes, 3), dtype=torch.float32)
    source_coords = torch.zeros((B, max_nodes, 3), dtype=torch.float32)
    source_velocity = torch.zeros((B, max_nodes, 3), dtype=torch.float32)
    source_velocity_scaled = torch.zeros((B, max_nodes, 3), dtype=torch.float32)
    node_feat = torch.zeros((B, max_nodes, node_feat_dim), dtype=torch.float32)
    node_mask = torch.zeros((B, max_nodes), dtype=torch.bool)
    node_is_ligand = torch.zeros((B, max_nodes), dtype=torch.bool)
    node_atomic_numbers = torch.zeros((B, max_nodes), dtype=torch.long)
    node_masses = torch.zeros((B, max_nodes), dtype=torch.float32)
    node_vdw_radius_A = torch.zeros((B, max_nodes), dtype=torch.float32)
    hbond_donor_mask = torch.zeros((B, max_nodes), dtype=torch.bool)
    hbond_acceptor_mask = torch.zeros((B, max_nodes), dtype=torch.bool)
    hydrophobic_mask = torch.zeros((B, max_nodes), dtype=torch.bool)
    node_residue_index = torch.full((B, max_nodes), -1, dtype=torch.long)
    node_backbone_slot = torch.full((B, max_nodes), -1, dtype=torch.long)

    edge_index_list = []
    bond_edge_index_list = []
    k_list = []
    l_list = []
    for i, item in enumerate(batch):
        n = item["source_coords"].shape[0]
        obs_coords[i, :, :n] = item["obs_coords"]
        obs_velocity[i, :, :n] = item["obs_velocity"]
        future_coords[i, :, :n] = item["future_coords"]
        source_coords[i, :n] = item["source_coords"]
        source_velocity[i, :n] = item["source_velocity"]
        source_velocity_scaled[i, :n] = item["source_velocity_scaled"]
        node_feat[i, :n] = item["node_feat"]
        node_mask[i, :n] = True
        node_is_ligand[i, :n] = item["node_is_ligand"].bool()
        node_atomic_numbers[i, :n] = item["node_atomic_numbers"].long()
        node_masses[i, :n] = item["node_masses"].float()
        node_vdw_radius_A[i, :n] = item["node_vdw_radius_A"].float()
        hbond_donor_mask[i, :n] = item["hbond_donor_mask"].bool()
        hbond_acceptor_mask[i, :n] = item["hbond_acceptor_mask"].bool()
        hydrophobic_mask[i, :n] = item["hydrophobic_mask"].bool()
        node_residue_index[i, :n] = item["node_residue_index"].long()
        node_backbone_slot[i, :n] = item["node_backbone_slot"].long()

        off = i * max_nodes
        edge_index_list.append(item["edge_index"].clone() + off)
        bond_edge_index_list.append(item["bond_edge_index"].clone() + off)
        k_list.append(torch.where(item["k_idx"] >= 0, item["k_idx"] + off, item["k_idx"]))
        l_list.append(torch.where(item["l_idx"] >= 0, item["l_idx"] + off, item["l_idx"]))

    node_batch = torch.arange(B, dtype=torch.long).repeat_interleave(max_nodes)
    edge_index = torch.cat(edge_index_list, dim=1)
    bond_edge_index = torch.cat(bond_edge_index_list, dim=1)
    k_idx = torch.cat(k_list, dim=0)
    l_idx = torch.cat(l_list, dim=0)

    max_protein_len = max(item["source_protein_bb"].shape[0] for item in batch)
    obs_protein_bb = torch.zeros((B, obs_len, max_protein_len, 3, 3), dtype=torch.float32)
    obs_protein_velocity = torch.zeros((B, obs_len, max_protein_len, 3, 3), dtype=torch.float32)
    obs_protein_bb_mask = torch.zeros((B, obs_len, max_protein_len), dtype=torch.float32)
    source_protein_bb = torch.zeros((B, max_protein_len, 3, 3), dtype=torch.float32)
    source_protein_bb_mask = torch.zeros((B, max_protein_len), dtype=torch.float32)
    source_protein_velocity = torch.zeros((B, max_protein_len, 3, 3), dtype=torch.float32)
    source_protein_velocity_scaled = torch.zeros((B, max_protein_len, 3, 3), dtype=torch.float32)
    future_protein_bb = torch.zeros((B, fut_len, max_protein_len, 3, 3), dtype=torch.float32)
    future_protein_bb_mask = torch.zeros((B, fut_len, max_protein_len), dtype=torch.float32)

    protein_full_seq_ids = torch.zeros((B, max_protein_len), dtype=torch.long)
    protein_full_mask = torch.zeros((B, max_protein_len), dtype=torch.float32)
    protein_chain_pair_mask = torch.zeros((B, max(0, max_protein_len - 1)), dtype=torch.bool)
    max_pocket_res = max(item["pocket_residue_indices"].numel() for item in batch)
    pocket_residue_indices = torch.full((B, max_pocket_res), -1, dtype=torch.long)
    pocket_residue_mask = torch.zeros((B, max_pocket_res), dtype=torch.float32)
    for i, item in enumerate(batch):
        l = item["source_protein_bb"].shape[0]
        obs_protein_bb[i, :, :l] = item["obs_protein_bb"]
        obs_protein_velocity[i, :, :l] = item["obs_protein_velocity"]
        obs_protein_bb_mask[i, :, :l] = item["obs_protein_bb_mask"]
        source_protein_bb[i, :l] = item["source_protein_bb"]
        source_protein_bb_mask[i, :l] = item["source_protein_bb_mask"]
        source_protein_velocity[i, :l] = item["source_protein_velocity"]
        source_protein_velocity_scaled[i, :l] = item["source_protein_velocity_scaled"]
        future_protein_bb[i, :, :l] = item["future_protein_bb"]
        future_protein_bb_mask[i, :, :l] = item["future_protein_bb_mask"]
        protein_full_seq_ids[i, : item["protein_full_seq_ids"].numel()] = item["protein_full_seq_ids"]
        protein_full_mask[i, : item["protein_full_mask"].numel()] = item["protein_full_mask"]
        pair_mask = item.get("protein_chain_pair_mask")
        if pair_mask is not None and pair_mask.numel() > 0:
            protein_chain_pair_mask[i, : pair_mask.numel()] = pair_mask.bool()
        p = item["pocket_residue_indices"].numel()
        pocket_residue_indices[i, :p] = item["pocket_residue_indices"].long()
        pocket_residue_mask[i, :p] = 1.0

    return {
        "obs_coords": obs_coords,
        "obs_velocity": obs_velocity,
        "future_coords": future_coords,
        **_collate_qm_targets(batch, fut_len),
        "source_coords": source_coords,
        "source_velocity": source_velocity,
        "source_velocity_scaled": source_velocity_scaled,
        "node_feat": node_feat,
        "node_mask": node_mask,
        "node_is_ligand": node_is_ligand,
        "node_atomic_numbers": node_atomic_numbers,
        "node_masses": node_masses,
        "node_vdw_radius_A": node_vdw_radius_A,
        "hbond_donor_mask": hbond_donor_mask,
        "hbond_acceptor_mask": hbond_acceptor_mask,
        "hydrophobic_mask": hydrophobic_mask,
        "node_residue_index": node_residue_index,
        "node_backbone_slot": node_backbone_slot,
        "pocket_residue_indices": pocket_residue_indices,
        "pocket_residue_mask": pocket_residue_mask,
        "edge_index": edge_index,
        "bond_edge_index": bond_edge_index,
        "k_idx": k_idx,
        "l_idx": l_idx,
        "node_batch": node_batch,
        "obs_cloud_grid_seq": torch.stack([item["obs_cloud_grid_seq"] for item in batch], dim=0),
        "obs_cloud_center_seq": torch.stack([item["obs_cloud_center_seq"] for item in batch], dim=0),
        "source_cloud_center": torch.stack([item["source_cloud_center"] for item in batch], dim=0),
        "obs_cloud_voxel_seq": torch.stack([item["obs_cloud_voxel_seq"] for item in batch], dim=0),
        "source_cloud_voxel": torch.stack([item["source_cloud_voxel"] for item in batch], dim=0),
        "obs_protein_bb": obs_protein_bb,
        "obs_protein_velocity": obs_protein_velocity,
        "obs_protein_bb_mask": obs_protein_bb_mask,
        "source_protein_bb": source_protein_bb,
        "source_protein_bb_mask": source_protein_bb_mask,
        "source_protein_velocity": source_protein_velocity,
        "source_protein_velocity_scaled": source_protein_velocity_scaled,
        "future_protein_bb": future_protein_bb,
        "future_protein_bb_mask": future_protein_bb_mask,
        "obs_times": torch.stack([item["obs_times"] for item in batch], dim=0),
        "future_times": torch.stack([item["future_times"] for item in batch], dim=0),
        "source_time": torch.stack([item["source_time"] for item in batch], dim=0),
        "time_scale": torch.stack([item["time_scale"] for item in batch], dim=0),
        "future_dt": torch.stack([item["future_dt"] for item in batch], dim=0),
        "future_dt_scaled": torch.stack([item["future_dt_scaled"] for item in batch], dim=0),
        "obs_box": torch.stack([item["obs_box"] for item in batch], dim=0),
        "obs_box_mask": torch.stack([item["obs_box_mask"] for item in batch], dim=0),
        "source_box": torch.stack([item["source_box"] for item in batch], dim=0),
        "source_box_mask": torch.stack([item["source_box_mask"] for item in batch], dim=0),
        "future_box": torch.stack([item["future_box"] for item in batch], dim=0),
        "future_box_mask": torch.stack([item["future_box_mask"] for item in batch], dim=0),
        "protein_full_seq_ids": protein_full_seq_ids,
        "protein_full_mask": protein_full_mask,
        "protein_chain_pair_mask": protein_chain_pair_mask,
        "traj_dir": [item["traj_dir"] for item in batch],
        "observed_frame_ids": [item["observed_frame_ids"] for item in batch],
        "future_frame_ids": [item["future_frame_ids"] for item in batch],
        "frame_id": [item["frame_id"] for item in batch],
        "target_frame_id": [item["target_frame_id"] for item in batch],
        "horizon": [item["horizon"] for item in batch],
        "target_frame_stride": [item["target_frame_stride"] for item in batch],
        "observed_frame_stride": [item["observed_frame_stride"] for item in batch],
    }
