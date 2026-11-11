"""Small deterministic checks for the MD hierarchy geometry primitives."""

import torch

from models.trajectory_refinement import enforce_backbone_nodes, restore_pocket_from_backbone
from utils.md import unwrap_trajectory


def main():
    dtype = torch.float32
    source_bb = torch.tensor(
        [[[[ -0.10, 0.10, 0.00], [0.00, 0.00, 0.00], [0.10, 0.00, 0.00]]]],
        dtype=dtype,
    )
    source_coords = torch.tensor(
        [[[ -0.10, 0.10, 0.00], [0.00, 0.00, 0.00], [0.10, 0.00, 0.00], [0.00, 0.10, 0.10], [0.50, 0.50, 0.50]]],
        dtype=dtype,
    )
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)
    translation = torch.tensor([0.30, -0.20, 0.10], dtype=dtype)
    target_bb = (torch.matmul(source_bb, rotation.T) + translation)[:, None]
    expected = torch.matmul(source_coords[:, :4], rotation.T) + translation
    residue_index = torch.tensor([[0, 0, 0, 0, -1]])
    protein_mask = torch.tensor([[True, True, True, True, False]])
    restored = restore_pocket_from_backbone(
        source_coords,
        source_bb,
        target_bb,
        residue_index,
        protein_mask,
    )
    if not torch.allclose(restored[0, 0, :4], expected[0], atol=1.0e-5, rtol=1.0e-5):
        raise AssertionError("backbone-frame rigid pocket restoration is not SE(3)-consistent")

    slots = torch.tensor([[0, 1, 2, -1, -1]])
    corrupted = restored + 1.0
    enforced = enforce_backbone_nodes(corrupted, target_bb, residue_index, slots)
    if not torch.allclose(enforced[0, 0, :3], target_bb[0, 0, 0], atol=1.0e-6, rtol=1.0e-6):
        raise AssertionError("selected backbone atoms were not synchronized with the backbone branch")

    wrapped = torch.tensor([[[4.9, 0.0, 0.0]], [[-4.8, 0.0, 0.0]]], dtype=dtype)
    boxes = torch.tensor([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]], dtype=dtype)
    unwrapped = unwrap_trajectory(wrapped, boxes)
    if not torch.allclose(unwrapped[1, 0, 0], torch.tensor(5.2), atol=1.0e-5):
        raise AssertionError("PBC unwrapping retained an artificial box jump")

    print("md_hierarchy_ok rigid_restore=1 backbone_sync=1 pbc_unwrap=1")


if __name__ == "__main__":
    main()
