"""Numerical checks for the adjacent-frame increment diffusion data flow."""

import torch

from models.trajectory_refinement import (
    _frame_increments,
    _integrate_local_increment_residual_velocity,
    _local_increment_residual_velocity,
)


def _rotation_z(angle, *, dtype=torch.float64):
    c = torch.cos(torch.tensor(angle, dtype=dtype))
    s = torch.sin(torch.tensor(angle, dtype=dtype))
    return torch.stack(
        [
            torch.stack([c, -s, torch.zeros_like(c)]),
            torch.stack([s, c, torch.zeros_like(c)]),
            torch.tensor([0.0, 0.0, 1.0], dtype=dtype),
        ]
    )


def main():
    torch.manual_seed(7)
    dtype = torch.float64
    batch, steps, atoms = 2, 5, 4
    source = torch.randn(batch, atoms, 3, dtype=dtype)
    base_increment = 0.05 * torch.randn(batch, steps, atoms, 3, dtype=dtype)
    base = source[:, None] + torch.cumsum(base_increment, dim=1)
    frame_dt = torch.tensor(
        [[0.5, 1.0, 1.5, 0.75, 2.0], [1.25, 0.6, 1.8, 1.1, 0.9]],
        dtype=dtype,
    )
    frame_rotation = _rotation_z(0.37).reshape(1, 1, 1, 3, 3).expand(
        batch,
        steps,
        atoms,
        -1,
        -1,
    )
    true_local_velocity = 0.02 * torch.randn(batch, steps, atoms, 3, dtype=dtype)
    target, expected_increment = _integrate_local_increment_residual_velocity(
        base,
        source,
        true_local_velocity,
        frame_rotation,
        frame_dt,
    )
    recovered_velocity = _local_increment_residual_velocity(
        target,
        base,
        source,
        frame_dt,
        frame_rotation,
    )
    torch.testing.assert_close(recovered_velocity, true_local_velocity, rtol=1.0e-9, atol=1.0e-9)

    reconstructed, reconstructed_increment = _integrate_local_increment_residual_velocity(
        base,
        source,
        recovered_velocity,
        frame_rotation,
        frame_dt,
    )
    torch.testing.assert_close(reconstructed, target, rtol=1.0e-9, atol=1.0e-9)
    torch.testing.assert_close(reconstructed_increment, expected_increment, rtol=1.0e-9, atol=1.0e-9)

    # A final-frame absolute anchor must backpropagate through every preceding
    # increment because reconstruction is a cumulative sum from the source.
    differentiable_velocity = true_local_velocity.clone().requires_grad_(True)
    cumulative, _ = _integrate_local_increment_residual_velocity(
        base,
        source,
        differentiable_velocity,
        frame_rotation,
        frame_dt,
    )
    cumulative[:, -1].square().sum().backward()
    assert differentiable_velocity.grad is not None
    assert bool((differentiable_velocity.grad.abs().sum(dim=(-1, -2)) > 0).all())

    # A global rigid rotation changes global coordinates and frame columns in
    # tandem but leaves the local increment target invariant.
    global_rotation = _rotation_z(-0.61)
    rotated_source = source @ global_rotation.T
    rotated_base = base @ global_rotation.T
    rotated_target = target @ global_rotation.T
    rotated_frames = torch.matmul(global_rotation, frame_rotation)
    rotated_velocity = _local_increment_residual_velocity(
        rotated_target,
        rotated_base,
        rotated_source,
        frame_dt,
        rotated_frames,
    )
    torch.testing.assert_close(rotated_velocity, true_local_velocity, rtol=1.0e-9, atol=1.0e-9)

    # The first transition uses the source-frame box; later transitions use
    # the preceding future-frame box, matching trajectory unwrapping.
    periodic_source = torch.tensor([[[4.9, 0.0, 0.0]]], dtype=dtype)
    periodic_path = torch.tensor([[[[-4.9, 0.0, 0.0]], [[-4.7, 0.0, 0.0]]]], dtype=dtype)
    periodic_box = 10.0 * torch.eye(3, dtype=dtype)
    periodic_increment = _frame_increments(
        periodic_path,
        periodic_source,
        boxes=periodic_box.reshape(1, 1, 3, 3).expand(-1, 2, -1, -1),
        source_box=periodic_box.reshape(1, 3, 3),
    )
    torch.testing.assert_close(
        periodic_increment[..., 0],
        torch.tensor([[[0.2], [0.2]]], dtype=dtype),
        rtol=1.0e-9,
        atol=1.0e-9,
    )
    print("increment diffusion checks passed")


if __name__ == "__main__":
    main()
