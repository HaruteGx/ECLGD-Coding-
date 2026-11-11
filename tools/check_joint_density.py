"""Synthetic plumbing/regression tests, NOT evidence of QM density accuracy."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import torch

from models.joint_density import JointElectronDensity
from models.trajectory_refinement import TrajectoryRefinementDiffusion
from utils.qm_density import read_density_cube, load_qm_targets


def refinement_inputs():
    bb = torch.tensor([[[[-.1, .1, 0.], [0., 0., 0.], [.1, 0., 0.]]]])
    coords = torch.cat([bb.flatten(1, 2), torch.tensor([[[0., .2, 0.], [.4, 0., 0.]]])], 1)
    mapping = torch.tensor([[0, 0, 0, 0, -1]])
    features = torch.zeros((1, 5, 5)); features[..., 0] = 6 / 53.
    return dict(rough_coords=coords[:, None].repeat(1, 2, 1, 1),
                rough_bb=bb[:, None].repeat(1, 2, 1, 1, 1), source_coords=coords, source_bb=bb,
                node_feat=features, node_mask=torch.ones((1, 5), dtype=torch.bool),
                node_is_ligand=mapping == -1, protein_mask=torch.ones((1, 2, 1), dtype=torch.bool),
                node_residue_index=mapping, node_backbone_slot=torch.tensor([[0, 1, 2, -1, -1]]),
                source_time=torch.tensor([0.]), future_times=torch.tensor([[1., 2.]]),
                future_dt_scaled=torch.ones((1, 2)), source_velocity=torch.zeros_like(coords),
                source_bb_velocity=torch.zeros_like(bb))


def synthetic_cube(path, atom_coords, atom_numbers, values=None):
    # Intentionally synthetic scalar field for I/O tests, not a quantum label.
    lines = ['SYNTHETIC UNIT TEST FIELD', 'Not a QM calculation', f'{len(atom_numbers)} -1 -1 -1',
             '2 1 0 0', '2 0 1 0', '2 0 0 1']
    for z, xyz in zip(atom_numbers, atom_coords):
        lines.append(f'{z} 0 {xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}')
    lines.append(' '.join(str(v) for v in ([1.] * 8 if values is None else values)))
    path.write_text('\n'.join(lines), encoding='utf-8')


class JointDensityTests(unittest.TestCase):
    def test_labels_and_target_grid_do_not_change_predictions(self):
        torch.manual_seed(2)
        model = TrajectoryRefinementDiffusion(d_model=16, diffusion_steps=4)
        args = refinement_inputs()
        points = torch.rand((1, 2, 8, 3)) * .2
        mask = torch.ones((1, 2, 8), dtype=torch.bool)
        def run(values, locations):
            torch.manual_seed(41)
            return model.train_refine(**args, target_coords=args['rough_coords'], target_bb=args['rough_bb'],
                                      qm_points=locations, qm_density=values, qm_mask=mask)
        with patch('models.trajectory_refinement.QM_DENSITY_REQUIRE_LABELS', True):
            first = run(torch.ones((1, 2, 8)), points)
            second = run(torch.full((1, 2, 8), 20.), points + .1)
        torch.testing.assert_close(first['refined_coords'], second['refined_coords'], atol=0, rtol=0)
        self.assertNotEqual(float(first['qm_density_loss']), float(second['qm_density_loss']))
        self.assertEqual(float(first['qm_density_coverage']), 1.)
        first['qm_density_loss'].backward()
        self.assertTrue(any(p.grad is not None and bool((p.grad != 0).any()) for p in model.joint_density.parameters()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_ligand_output_really_depends_on_predicted_density(self):
        model = TrajectoryRefinementDiffusion(d_model=16, diffusion_steps=4).eval()
        args = refinement_inputs()
        torch.manual_seed(42)
        first = model.sample_refine(**args, sample_steps=2)
        with torch.no_grad():
            model.joint_density.parameter_head[-1].bias.reshape(4, 5)[:, 1] += 1.
        torch.manual_seed(42)
        second = model.sample_refine(**args, sample_steps=2)
        self.assertFalse(torch.equal(first['refined_coords'][..., -1, :], second['refined_coords'][..., -1, :]))
        field = second['predicted_joint_density_field']
        points = torch.tensor([[[[0., 0., 0.], [200., 0., 0.]], [[0., 0., 0.], [200., 0., 0.]]]])
        density = JointElectronDensity.evaluate(field, points)
        self.assertTrue(torch.isfinite(density).all() and (density >= 0).all())
        self.assertTrue((density[..., 1] == 0).all())

    def test_strict_missing_labels(self):
        model = TrajectoryRefinementDiffusion(d_model=16, diffusion_steps=4)
        args = refinement_inputs()
        with patch('models.trajectory_refinement.QM_DENSITY_REQUIRE_LABELS', True), self.assertRaisesRegex(ValueError, 'No QM density'):
            model.train_refine(**args, target_coords=args['rough_coords'], target_bb=args['rough_bb'])
        # Inference needs no label, even if strict supervised training is configured.
        with patch('models.trajectory_refinement.QM_DENSITY_REQUIRE_LABELS', True):
            self.assertIn('predicted_joint_density_field', model.sample_refine(**args, sample_steps=1))

    def test_cube_units_and_reject_negative_fields(self):
        with TemporaryDirectory(prefix='ecmd_qm_test_') as tmp:
            path = Path(tmp) / 'test.cube'
            synthetic_cube(path, [[0., 0., 0.]], [1])
            cube = read_density_cube(path, 'bohr', 'e/bohr^3')
            np.testing.assert_allclose(cube['axes_A'], np.eye(3) * .529177210903)
            np.testing.assert_allclose(cube['density_e_A3'], 1 / .529177210903 ** 3)
            synthetic_cube(path, [[0., 0., 0.]], [1], [-1.] * 8)
            with self.assertRaises(ValueError):
                read_density_cube(path, 'angstrom', 'e/angstrom^3')

    def test_import_dataset_collate_and_alignment_guards(self):
        import preprocess.build_graph_cache as cache
        from preprocess.import_qm_density import import_manifest
        from data.dataset import load_trajectory_sample
        from data.collate import collate_trajectory_batch
        with TemporaryDirectory(prefix='ecmd_qm_test_') as tmp:
            root = Path(tmp)
            coords = np.array([[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [1., 1., 0.], [4., 0., 0.]]] * 3, dtype=np.float32)
            numbers = np.array([7, 6, 6, 6, 6])
            np.savez(root / 'source.npz', coords=coords, atomic_numbers=numbers,
                     atom_names=np.array(['N', 'CA', 'C', 'CB', 'C1']),
                     resnames=np.array(['ALA'] * 4 + ['LIG']), resids=np.array([1] * 4 + [2]),
                     segids=np.array(['A'] * 5), ligand_resnames=np.array(['LIG']), times=np.arange(3, dtype=np.float32))
            with patch.object(cache, 'CLOUD_GRID_SIZE', 8), patch.object(cache, 'GRAPH_CACHE_CLOUD_DEVICE', 'cpu'):
                cache.process_one(root / 'source.npz', root / 'cache')
            synthetic_cube(root / 'test.cube', coords[2], numbers)
            entry = dict(frame_id=2, cube='test.cube', charge=0, spin=0,
                         cube_length_unit='angstrom', cube_density_unit='e/angstrom^3',
                         cube_atom_global_indices=list(range(5)))
            manifest = dict(kind='joint_electron_number_density', electron_convention='all_electron',
                            method='SYNTHETIC I/O TEST ONLY', basis='not applicable test fixture', frames=[entry])
            manifest_path = root / 'manifest.json'
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            output = import_manifest(root / 'cache', manifest_path, samples=8)
            sample = load_trajectory_sample(root / 'cache', 2, 1, start_frame=0)
            self.assertEqual(int(sample['future_qm_mask'].sum()), 8)
            self.assertEqual(sample['obs_cloud_grid_seq'].shape[1], 2)
            batch = collate_trajectory_batch([sample])
            self.assertEqual(batch['future_qm_points'].shape[:2], (1, 1))
            self.assertEqual(float(batch['future_qm_density'][batch['future_qm_mask']].mean()), 1.)
            # End-to-end entrypoint smoke, not just the refinement submodule.
            from models.temporal_model import TemporalCloudDiffusionModel
            with patch('models.cloud_encoder.CLOUD_GRID_SIZE', 8), patch('models.temporal_model.CLOUD_GRID_SIZE', 8), patch('models.temporal_model.MODEL_DIM', 16):
                model = TemporalCloudDiffusionModel(observed_frames=2, future_frames=1, diffusion_steps=4)
                total, _, _ = model(batch)
                self.assertTrue(torch.isfinite(total))
                total.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                prediction = model.predict_future(batch, steps=1, sample_steps=2)
                self.assertIn('predicted_joint_density_field', prediction)
                # Multi-chunk rollout must also expose the field and not read targets.
                batch['future_times'] = torch.tensor([[2., 3.]])
                batch['future_box'] = batch['future_box'].repeat(1, 2, 1, 1)
                extended = model.predict_future(batch, steps=2, sample_steps=2)
                self.assertEqual(extended['predicted_joint_density_field']['centers_A'].shape[1], 2)
            with self.assertRaises(FileExistsError):
                import_manifest(root / 'cache', manifest_path)
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                load_qm_targets(root / 'cache', [2], [{'coords': torch.ones((5, 3))}])
            entry['translation_A'] = [1., 0., 0.]
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'coordinate mismatch'):
                import_manifest(root / 'cache', manifest_path, overwrite=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
