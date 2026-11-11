"""Import aligned QM CUBEs as supervision ONLY; existing voxels stay unchanged.

python -m preprocess.import_qm_density --traj-dir graph_cache/ID --manifest qm.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from config import COORD_SCALE, LIGAND_NEAR_RES_CUTOFF, QM_DENSITY_SIDECAR
from data.dataset import _load_frames, _load_meta_features
from utils.md import box_to_matrix, minimum_image_displacement
from utils.pocket import dynamic_residue_pocket_mask
from utils.qm_density import coordinate_fingerprint, read_density_cube


def import_manifest(traj_dir, manifest_path, samples=4096, overwrite=False, seed=0):
    traj_dir, manifest_path = Path(traj_dir).resolve(), Path(manifest_path).resolve()
    destination = traj_dir / QM_DENSITY_SIDECAR
    if destination.exists() and not overwrite:
        raise FileExistsError(f'{destination} exists; use --overwrite for intentional replacement')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('kind') != 'joint_electron_number_density':
        raise ValueError('Declare TOTAL residue+ligand electron-number density, not ESP/orbitals')
    if manifest.get('electron_convention') != 'all_electron':
        raise ValueError('Require all_electron density; valence/ECP labels must not be mixed silently')
    if not manifest.get('method') or not manifest.get('basis'):
        raise ValueError('Record QM method and basis; Gaussian proxies are not QM labels')
    entries = manifest['frames']
    ids = [int(e['frame_id']) for e in entries]
    if not ids or len(ids) != len(set(ids)) or samples < 1:
        raise ValueError('Require unique frame ids and positive sample count')
    meta = _load_meta_features(traj_dir)['meta']
    frames = _load_frames(traj_dir, ids)
    global_ids = meta['node_global_indices'].tolist()
    lookup = {int(g): i for i, g in enumerate(global_ids)}
    n = len(global_ids)
    ligand = meta['node_is_ligand'].bool()
    all_points, all_density, all_masks, fingerprints, provenance = [], [], [], [], []
    rng = np.random.default_rng(seed)
    for entry, frame in zip(entries, frames):
        for key in ('charge', 'spin', 'cube_length_unit', 'cube_density_unit', 'cube_atom_global_indices'):
            if key not in entry:
                raise ValueError(f'Missing frame metadata: {key}')
        cube_path = (manifest_path.parent / entry['cube']).resolve()
        cube = read_density_cube(cube_path, entry['cube_length_unit'], entry['cube_density_unit'])
        rotation = np.asarray(entry.get('rotation', np.eye(3)), dtype=float)
        translation = np.asarray(entry.get('translation_A', [0., 0., 0.]), dtype=float)
        if rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError('Invalid alignment transform')
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
            raise ValueError('Alignment must be a proper rotation')
        mapping = [int(g) for g in entry['cube_atom_global_indices']]
        if len(mapping) != len(cube['atom_numbers']):
            raise ValueError('Map every CUBE atom to a global atom id; -1 denotes link/cap atoms')
        mapped = [g for g in mapping if g >= 0]
        if len(mapped) != len(set(mapped)) or any(g not in lookup for g in mapped):
            raise ValueError('Ambiguous/noncandidate atom mapping')
        coords_A = frame['coords'].float() * float(COORD_SCALE)
        box_A = box_to_matrix(frame['box']) if 'box' in frame else None
        active = dynamic_residue_pocket_mask(
            coords_A[None, None], coords_A[None, None], torch.ones((1, n), dtype=torch.bool),
            ligand[None], meta['node_residue_index'][None],
            boxes=None if box_A is None else box_A[None, None], cutoff=float(LIGAND_NEAR_RES_CUTOFF),
        )[0, 0] | ligand
        required_ids = {global_ids[i] for i in active.nonzero(as_tuple=False).flatten().tolist()}
        if not required_ids.issubset(set(mapped)):
            raise ValueError('QM CUBE must contain all ligand atoms and complete active residues')
        aligned_atoms = cube['atom_coords_A'] @ rotation.T + translation
        for c, g in enumerate(mapping):
            if g < 0:
                continue
            local = lookup[g]
            if int(cube['atom_numbers'][c]) != int(meta['node_atomic_numbers'][local]):
                raise ValueError('CUBE/cache atomic number mismatch')
            delta = torch.tensor(aligned_atoms[c], dtype=torch.float32) - coords_A[local]
            if box_A is not None:
                delta = minimum_image_displacement(delta, box_A)
            if float(delta.norm()) > 0.02:
                raise ValueError(f'CUBE/cache coordinate mismatch at global atom {g}; supply correct alignment')
        selected, priorities = np.empty(0, dtype=np.int64), np.empty(0)
        total_valid = 0
        # Uniform reservoir sample, no density-based selection and no full XYZ mesh.
        for start in range(0, cube['density_e_A3'].size, 8192):
            flat = np.arange(start, min(start + 8192, cube['density_e_A3'].size))
            ijk = np.stack(np.unravel_index(flat, cube['shape']), axis=-1)
            points = (cube['origin_A'] + ijk @ cube['axes_A']) @ rotation.T + translation
            delta = torch.from_numpy(points).float()[:, None] - coords_A[ligand][None]
            if box_A is not None:
                delta = minimum_image_displacement(delta, box_A)
            keep = (delta.square().sum(-1).amin(-1) <= float(LIGAND_NEAR_RES_CUTOFF) ** 2).numpy()
            valid = flat[keep]
            total_valid += len(valid)
            selected = np.concatenate([selected, valid])
            priorities = np.concatenate([priorities, rng.random(len(valid))])
            if len(selected) > samples:
                chosen = np.argpartition(priorities, samples - 1)[:samples]
                selected, priorities = selected[chosen], priorities[chosen]
        if not len(selected):
            raise ValueError('CUBE has no support inside the 10-A ligand neighborhood')
        ijk = np.stack(np.unravel_index(selected, cube['shape']), axis=-1)
        p = (cube['origin_A'] + ijk @ cube['axes_A']) @ rotation.T + translation
        count = len(selected)
        padded_p = np.zeros((samples, 3), dtype=np.float32)
        padded_d = np.zeros(samples, dtype=np.float32)
        padded_m = np.zeros(samples, dtype=bool)
        padded_p[:count], padded_d[:count], padded_m[:count] = p, cube['density_e_A3'][selected], True
        all_points.append(padded_p); all_density.append(padded_d); all_masks.append(padded_m)
        fingerprints.append(coordinate_fingerprint(frame['coords']))
        provenance.append(dict(frame_id=int(entry['frame_id']), cube=str(cube_path),
                               cube_sha256=hashlib.sha256(cube_path.read_bytes()).hexdigest(),
                               charge=entry['charge'], spin=entry['spin'], sampled_voxels=count,
                               eligible_cube_voxels=total_valid, alignment=entry))
    metadata = dict(schema=1, kind='joint_electron_number_density', coordinate_unit='angstrom',
                    density_unit='e/angstrom^3', electron_convention='all_electron',
                    method=manifest['method'], basis=manifest['basis'], provenance=provenance,
                    support='CUBE grid intersected with true 10-A ROI; outside support is unlabeled')
    np.savez_compressed(destination, frame_ids=np.asarray(ids), points_A=np.stack(all_points),
                        density_e_A3=np.stack(all_density), mask=np.stack(all_masks),
                        coordinate_sha256=np.asarray(fingerprints), metadata_json=json.dumps(metadata))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traj-dir', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--samples', default=4096, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    print(import_manifest(args.traj_dir, args.manifest, args.samples, args.overwrite, args.seed))


if __name__ == '__main__':
    main()
