"""CPU regression checks: python -m tools.check_conservative_pocket."""
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch

from utils.md import conservative_acceleration, _energy_pairs
from utils.pocket import dynamic_residue_pocket_mask
from preprocess.build_graph_cache import select_protein_ligand_candidate_atoms


class ConservativePocketTests(unittest.TestCase):
    def physics(self, coords, bonded=False, reference=None, boxes=None):
        b, n = coords.shape[:2]
        feat = coords.new_zeros((b, n, 5))
        feat[..., -1] = torch.linspace(-0.4, 0.3, n).to(coords)
        masses = coords.new_full((b, n), 12.0)
        bonds = torch.tensor([[0], [1]]) if bonded else torch.empty((2, 0), dtype=torch.long)
        result = conservative_acceleration(
            coords, coords.detach() if reference is None else reference,
            torch.ones((b, n), dtype=torch.bool), masses, feat,
            coords.new_full((b, n), 1.7), bonds, boxes=boxes,
            return_energy=True, return_interface_energy=True,
            node_is_ligand=torch.arange(n)[None].expand(b, -1) % 2 == 1,
        )
        return result, masses

    def test_force_equals_energy_gradient_and_backward(self):
        # Includes old distance/LJ/force/acceleration clipping regimes, exact
        # overlap, normal separation, and both sides of the switching region.
        for distance in [0.0, 0.001, 0.1, 0.49, 1.0, 3.4, 7.99, 8.0, 8.1, 9.5, 9.999]:
            for bonded in [False, True]:
                x = torch.tensor([[[0., 0., 0.], [distance / 10., 0., 0.]]], dtype=torch.float64, requires_grad=True)
                ref = torch.tensor([[[0., 0., 0.], [0.15, 0., 0.]]], dtype=torch.float64)
                (acc, energy, interface), mass = self.physics(x, bonded, ref)
                grad, = torch.autograd.grad(energy.sum(), x, create_graph=True)
                expected = -grad * 418.4 / (100.0 * mass[..., None])
                torch.testing.assert_close(acc, expected, atol=2e-8, rtol=2e-9)
                torch.testing.assert_close((acc * mass[..., None]).sum(1), torch.zeros((1, 3), dtype=x.dtype))
                loss = acc.square().mean() + energy.square().mean() + interface.square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_cutoff_is_smooth_and_neighbors_not_topk(self):
        x = torch.randn((1, 40, 3), dtype=torch.float64) * 0.01
        pairs = _energy_pairs(x, torch.ones((1, 40), dtype=torch.bool), None, 1.0)
        self.assertEqual(pairs.shape[1], 40 * 39 // 2)
        for distance in [9.99999, 10.0, 10.00001]:
            x = torch.tensor([[[0., 0., 0.], [distance / 10., 0., 0.]]], dtype=torch.float64)
            (acc, energy, _), _ = self.physics(x)
            self.assertLess(float(energy.abs().max()), 1e-8)
            self.assertLess(float(acc.abs().max()), 1e-7)

    def test_pbc_gradient(self):
        x = torch.tensor([[[0.99, 0., 0.], [0.01, 0., 0.]]], requires_grad=True)
        box = torch.eye(3)[None]
        (acc, energy, _), mass = self.physics(x, boxes=box)
        grad, = torch.autograd.grad(energy.sum(), x)
        torch.testing.assert_close(acc, -grad * 418.4 / (100. * mass[..., None]), rtol=2e-5, atol=0.01)

    def test_dynamic_entry_exit_whole_residues_and_padding(self):
        x = torch.zeros((1, 3, 6, 3))
        x[0, :, :, 0] = torch.tensor([[0., 0.5, 1.5, 2., 0., 0.],
                                                    [0., 2., 3., 0.5, 0., 0.],
                                                    [0., 2., 3., 2., 0., 0.]])
        mask = torch.tensor([[1, 1, 1, 1, 0, 1]], dtype=torch.bool)
        lig = torch.tensor([[1, 0, 0, 0, 0, 0]], dtype=torch.bool)
        residues = torch.tensor([[-1, 0, 0, 1, 1, -1]])
        active = dynamic_residue_pocket_mask(x, x, mask, lig, residues)
        self.assertEqual(active[0].tolist(), [[False, True, True, False, False, False],
                                             [False, False, False, True, False, False],
                                             [False] * 6])
        # Prefix decisions cannot depend on later frames.
        prefix = dynamic_residue_pocket_mask(x[:, :1], x[:, :1], mask, lig, residues)
        self.assertTrue(torch.equal(prefix, active[:, :1]))
        wrapped = x[:, :1].clone()
        wrapped[0, 0, 3, 0] = 9.8
        active = dynamic_residue_pocket_mask(wrapped, wrapped, mask, lig, residues,
                                             boxes=(torch.eye(3) * 10)[None, None])
        self.assertTrue(active[0, 0, 3])
        empty = dynamic_residue_pocket_mask(x, x, mask, torch.zeros_like(lig), residues)
        self.assertFalse(empty.any())

    def test_candidate_selection_uses_only_topology(self):
        names = np.array(['N', 'CA', 'C', 'CB', 'N', 'CA', 'C', 'O', 'C1'])
        residues = np.array(['ALA'] * 4 + ['GLY'] * 3 + ['HOH', 'LIG'])
        ids = np.array([1] * 4 + [2] * 3 + [3, 4])
        selected = select_protein_ligand_candidate_atoms(names, residues, ids, np.array(['A'] * 9), {'LIG'})
        np.testing.assert_array_equal(selected, [0, 1, 2, 3, 4, 5, 6, 8])

    def test_refinement_train_backward_and_sample(self):
        from models.trajectory_refinement import TrajectoryRefinementDiffusion
        torch.manual_seed(12)
        model = TrajectoryRefinementDiffusion(d_model=16, diffusion_steps=4)
        bb = torch.tensor([[[[-.1, .1, 0.], [0., 0., 0.], [.1, 0., 0.]],
                             [[2.9, .1, 0.], [3., 0., 0.], [3.1, 0., 0.]]]])
        coords = torch.cat([bb.flatten(1, 2), torch.tensor([[[0., .2, 0.], [3., .2, 0.], [.5, 0., 0.]]])], 1)
        mapping = torch.tensor([[0, 0, 0, 1, 1, 1, 0, 1, -1]])
        slots = torch.tensor([[0, 1, 2, 0, 1, 2, -1, -1, -1]])
        rough_bb = bb[:, None].repeat(1, 2, 1, 1, 1)
        rough_coords = coords[:, None].repeat(1, 2, 1, 1)
        features = torch.zeros((1, 9, 5)); features[..., 0] = 6 / 53.
        lig = mapping == -1
        args = dict(rough_coords=rough_coords, rough_bb=rough_bb,
                    source_coords=coords, source_bb=bb, node_feat=features,
                    node_mask=torch.ones_like(lig), node_is_ligand=lig,
                    protein_mask=torch.ones((1, 2, 2), dtype=torch.bool),
                    node_residue_index=mapping, node_backbone_slot=slots,
                    source_time=torch.tensor([0.]), future_times=torch.tensor([[1., 2.]]),
                    future_dt_scaled=torch.ones((1, 2)),
                    source_velocity=torch.zeros_like(coords), source_bb_velocity=torch.zeros_like(bb))
        out = model.train_refine(**args, target_coords=rough_coords, target_bb=rough_bb)
        scalar_losses = [v for v in out.values() if torch.is_tensor(v) and v.ndim == 0 and v.requires_grad]
        self.assertTrue(scalar_losses)
        sum(scalar_losses).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        sampled = model.sample_refine(**args, sample_steps=2)
        self.assertEqual(sampled['active_pocket_mask'].shape, (1, 2, 9))
        self.assertTrue(torch.isfinite(sampled['refined_coords']).all())

    def test_rollout_cloud_membership(self):
        from models.temporal_model import TemporalCloudDiffusionModel
        # Exercise cloud methods without constructing the expensive encoder.
        obj = TemporalCloudDiffusionModel.__new__(TemporalCloudDiffusionModel)
        torch.nn.Module.__init__(obj)
        x = torch.tensor([[[0., 0., 0.], [.4, 0., 0.], [2., 0., 0.]]])
        batch = dict(node_mask=torch.ones((1, 3), dtype=torch.bool),
                     node_is_ligand=torch.tensor([[1, 0, 0]], dtype=torch.bool),
                     node_residue_index=torch.tensor([[-1, 0, 1]]),
                     node_feat=torch.ones((1, 3, 5)))
        _, active = obj._current_pocket_cloud_inputs(batch, x)
        self.assertEqual(active.tolist(), [[True, True, False]])
        grids, _, _ = obj._fast_rebuild_cloud_from_protein_coords(batch, x)
        self.assertGreater(float(grids[:, 0].sum()), 0.)
        far = x.clone(); far[0, 1, 0] = 2.
        grids, _, _ = obj._fast_rebuild_cloud_from_protein_coords(batch, far)
        self.assertEqual(float(grids[:, 0].sum()), 0.)
        self.assertGreater(float(grids[:, 1].sum()), 0.)

    def test_cache_build_and_dataset_roundtrip(self):
        import preprocess.build_graph_cache as cache
        from data.dataset import load_trajectory_sample
        with TemporaryDirectory(prefix='ecmd_pocket_test_') as tmp:
            path = Path(tmp)
            xyz = np.zeros((3, 9, 3), dtype=np.float32)
            xyz[:, :, 0] = np.array([0., 1., 2., 3., 30., 31., 32., 33., 5.])
            xyz[1, 4:8, 0] -= 25.
            np.savez(path / 'input.npz', coords=xyz,
                     atom_names=np.array(['N', 'CA', 'C', 'CB'] * 2 + ['C1']),
                     resnames=np.array(['ALA'] * 8 + ['LIG']),
                     resids=np.array([1] * 4 + [2] * 4 + [3]), segids=np.array(['A'] * 9),
                     ligand_resnames=np.array(['LIG']), times=np.arange(3, dtype=np.float32))
            with patch.object(cache, 'CLOUD_GRID_SIZE', 8), patch.object(cache, 'GRAPH_CACHE_CLOUD_DEVICE', 'cpu'):
                cache.process_one(path / 'input.npz', path / 'cache')
            meta = cache._torch_load_cache(path / 'cache/meta.pt')
            packed = cache._torch_load_cache(path / 'cache' / cache.PACKED_FRAMES_NAME)
            self.assertEqual(meta['cache_schema_version'], 4)
            self.assertEqual(len(meta['sel_global_indices']), 9)
            self.assertEqual(int(packed['frames'][0]['active_pocket_mask'].sum()), 4)
            self.assertEqual(int(packed['frames'][1]['active_pocket_mask'].sum()), 8)
            self.assertEqual(int(packed['frames'][2]['active_pocket_mask'].sum()), 4)
            sample = load_trajectory_sample(path / 'cache', observed_frames=2, future_frames=1, start_frame=0)
            self.assertEqual(sample['source_coords'].shape, (9, 3))

    def test_interface_loss_includes_late_candidate_atoms(self):
        from models.temporal_training import _interface_potential_losses
        x = torch.zeros((1, 2, 302, 3))
        x[:, :, 1:, 0] = 5.
        x[:, 0, -1, 0] = .05
        mask = torch.ones((1, 302), dtype=torch.bool)
        ligand = torch.zeros_like(mask); ligand[:, 0] = True
        residues = torch.arange(302)[None]; residues[:, 0] = -1
        vdw, _, _ = _interface_potential_losses(
            x, x, mask, ligand, node_vdw_radius_A=torch.full((1, 302), 1.7),
            node_residue_index=residues,
        )
        self.assertGreater(float(vdw), 0.)

    def test_physics_loss_backward(self):
        from models.temporal_training import _trajectory_physics_loss
        bb = torch.tensor([[[[[-.1, .1, 0.], [0., 0., 0.], [.1, 0., 0.]]]]]).repeat(1, 2, 1, 1, 1)
        x = torch.tensor([[[[0., 0., 0.], [.3, .1, 0.]]]]).repeat(1, 2, 1, 1).requires_grad_()
        mask = torch.ones((1, 2), dtype=torch.bool)
        result = _trajectory_physics_loss(
            x, bb, x.detach(), bb, x[:, 0].detach(), mask,
            torch.ones((1, 2, 1), dtype=torch.bool), torch.tensor([[False, True]]),
            node_residue_index=torch.tensor([[0, -1]]),
            source_time=torch.tensor([0.]), future_times=torch.tensor([[1., 2.]]),
            source_velocity=torch.zeros((1, 2, 3)),
        )
        result[0].backward()
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
