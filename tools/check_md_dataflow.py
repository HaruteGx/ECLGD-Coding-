import argparse
from pathlib import Path

import torch

from data.dataset import load_trajectory_sample


def main():
    parser = argparse.ArgumentParser(description="Validate MD-aware temporal source/time/velocity/PBC alignment.")
    parser.add_argument("traj_dir", type=Path)
    parser.add_argument("--observed", type=int, default=10)
    parser.add_argument("--future", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    sample = load_trajectory_sample(
        args.traj_dir,
        observed_frames=args.observed,
        future_frames=args.future,
        start_frame=args.start,
        source_frame_id=None,
        target_start_frame=None,
    )
    expected_source = sample["observed_frame_ids"][-1]
    if int(sample["frame_id"]) != int(expected_source):
        raise AssertionError(f"source frame mismatch: frame_id={sample['frame_id']} expected={expected_source}")
    if not torch.equal(sample["source_coords"], sample["obs_coords"][-1]):
        raise AssertionError("source_coords is not the last observed coordinate frame")
    if not torch.equal(sample["source_protein_bb"], sample["obs_protein_bb"][-1]):
        raise AssertionError("source_protein_bb is not the last observed backbone frame")
    if not torch.allclose(sample["source_time"], sample["obs_times"][-1]):
        raise AssertionError("source_time is not the last observed time")
    if not bool((sample["future_dt"] > 0).all()):
        raise AssertionError(f"non-positive future dt: {sample['future_dt'].tolist()}")
    if not bool(torch.isfinite(sample["source_velocity"]).all()):
        raise AssertionError("source velocity contains NaN/Inf")
    residue_index = sample["node_residue_index"].long()
    backbone_slot = sample["node_backbone_slot"].long()
    mapped = (residue_index >= 0) & (backbone_slot >= 0)
    if bool(mapped.any()):
        expected_backbone_atoms = sample["source_protein_bb"][residue_index[mapped], backbone_slot[mapped]]
        if not torch.allclose(sample["source_coords"][mapped], expected_backbone_atoms, atol=1.0e-5, rtol=1.0e-5):
            raise AssertionError("selected N/CA/C atoms disagree with source_protein_bb")

    print(
        "md_dataflow_ok",
        f"source_frame={sample['frame_id']}",
        f"target_frames={sample['future_frame_ids'][0]}..{sample['future_frame_ids'][-1]}",
        f"source_time={float(sample['source_time']):.8g}",
        f"time_scale={float(sample['time_scale']):.8g}",
        f"velocity_rms={float(sample['source_velocity'].square().mean().sqrt()):.8g}",
        f"pbc={int(bool(sample['source_box_mask']))}",
        f"mapped_backbone_atoms={int(mapped.sum())}",
    )


if __name__ == "__main__":
    main()
