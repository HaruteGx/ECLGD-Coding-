"""Strict optional reference-density samples, separate from model conditions."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from config import COORD_SCALE, QM_DENSITY_SAMPLES, QM_DENSITY_SIDECAR


def coordinate_fingerprint(coords):
    array = torch.as_tensor(coords).detach().cpu().float().contiguous().numpy()
    return hashlib.sha256(array.astype('<f4', copy=False).tobytes()).hexdigest()


def load_qm_targets(traj_dir, frame_ids, frames):
    """Missing frames produce masked entries, never zero-valued QM labels."""
    T, S = len(frame_ids), max(1, int(QM_DENSITY_SAMPLES))
    points = torch.zeros((T, S, 3))
    values = torch.zeros((T, S))
    mask = torch.zeros((T, S), dtype=torch.bool)
    path = Path(traj_dir) / QM_DENSITY_SIDECAR
    if not path.exists():
        return dict(future_qm_points=points, future_qm_density=values, future_qm_mask=mask)
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata_json'].item()))
        if meta.get('kind') != 'joint_electron_number_density' or meta.get('density_unit') != 'e/angstrom^3':
            raise ValueError('QM sidecar is not joint electron-number density in e/angstrom^3')
        if meta.get('coordinate_unit') != 'angstrom' or meta.get('schema') != 1:
            raise ValueError('Unsupported QM density sidecar coordinate units/schema')
        ids = data['frame_ids'].tolist()
        arrays = {key: data[key] for key in ('coordinate_sha256', 'mask', 'points_A', 'density_e_A3')}
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate QM frame ids')
        for t, (fid, frame) in enumerate(zip(frame_ids, frames)):
            if int(fid) not in ids:
                continue
            index = ids.index(int(fid))
            if str(arrays['coordinate_sha256'][index]) != coordinate_fingerprint(frame['coords']):
                raise ValueError(f'QM density frame {fid} coordinate fingerprint mismatch; reimport after cache rebuild')
            valid = np.flatnonzero(arrays['mask'][index])
            if len(valid) > S:
                # Uniform subset, not densest-voxel selection. No metadata from
                # these target points is used to place the predicted field.
                valid = np.random.choice(valid, S, replace=False)
            p = arrays['points_A'][index, valid]
            v = arrays['density_e_A3'][index, valid]
            if not np.isfinite(p).all() or not np.isfinite(v).all() or (v < 0).any():
                raise ValueError(f'Invalid QM reference samples in frame {fid}')
            n = len(valid)
            points[t, :n] = torch.from_numpy(p.copy()).float() / float(COORD_SCALE)
            values[t, :n] = torch.from_numpy(v.copy()).float()
            mask[t, :n] = True
    return dict(future_qm_points=points, future_qm_density=values, future_qm_mask=mask)


def read_density_cube(path, length_unit, density_unit):
    """Read scalar CUBE, x outer / z inner. Reject orbital/multi-field files.

    Caller must declare units and attest density kind: a positive ESP field
    cannot be identified as density from values alone.
    """
    length_factor = {'bohr': 0.529177210903, 'angstrom': 1.0}[length_unit]
    density_factor = {'e/bohr^3': 1.0 / 0.529177210903 ** 3, 'e/angstrom^3': 1.0}[density_unit]
    with Path(path).open(encoding='utf-8') as f:
        comments = [f.readline().rstrip(), f.readline().rstrip()]
        header = f.readline().split()
        natoms = int(header[0])
        if natoms <= 0 or (len(header) > 4 and int(header[4]) != 1):
            raise ValueError('Only scalar density CUBE with positive atom count is supported')
        origin = np.asarray(header[1:4], dtype=float)
        axes, shape = [], []
        for _ in range(3):
            row = f.readline().split()
            shape.append(abs(int(row[0])))
            axes.append([float(v) for v in row[1:4]])
        atoms = np.asarray([[float(v) for v in f.readline().split()[:5]] for _ in range(natoms)])
        values = np.fromstring(f.read().replace('D', 'E').replace('d', 'e'), sep=' ')
    axes = np.asarray(axes) * length_factor
    if min(shape) < 1 or abs(np.linalg.det(axes)) < 1e-15:
        raise ValueError('Invalid CUBE grid axes')
    if values.size != int(np.prod(shape)):
        raise ValueError('CUBE voxel count mismatch or multi-field/orbital data')
    values *= density_factor
    if not np.isfinite(values).all() or (values < -1e-8).any():
        raise ValueError('Not a nonnegative total electron-number-density field')
    return dict(origin_A=origin * length_factor, axes_A=axes, shape=tuple(shape),
                atom_numbers=atoms[:, 0].astype(int), atom_coords_A=atoms[:, 2:5] * length_factor,
                density_e_A3=np.maximum(values, 0), comments=comments)
