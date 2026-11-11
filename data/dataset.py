from pathlib import Path
from contextlib import contextmanager
import os
import random
import signal
import time

import torch
from utils.qm_density import load_qm_targets
from torch.utils.data import Dataset, get_worker_info

from config import (
    BAD_GRAPH_LOG, CACHED_FRAME_COUNT, CACHED_FRAME_START, CLOUD_CHANNELS, CLOUD_GRID_SIZE,
    COORD_SCALE,
    DATASET_DEBUG_LOADS, DATASET_SAMPLE_TIMEOUT_SECONDS, DATASET_VALIDATE_GRAPH,
    DATASET_META_CACHE, FUTURE_FRAMES, GRAPH_CACHE_SCHEMA_VERSION, GRAPH_ROOT,
    MAX_GRAPH_EDGES, MAX_GRAPH_NODES, MAX_PROTEIN_RESIDUES,
    OBSERVED_FRAMES, PACKED_FRAMES_NAME, TEST_SPLIT_FRAC, TRAJ_SOURCE_FRAME,
    SKIP_BAD_GRAPH_CACHE,
    TRAJ_TARGET_FRAME_STRIDE, TRAJ_TARGET_START_FRAME, TRAJ_FORCE_FINAL_FRAME,
    TRAJ_OBSERVED_FRAME_STRIDE, TRAJ_RANDOM_STRIDE,
    TRAJ_MIN_FRAME_STRIDE, TRAJ_MAX_FRAME_STRIDE, TRAJ_RANDOM_WINDOW,
    TRAJ_RANDOM_WINDOWS_PER_TRAJ, TRAJ_WINDOW_STRIDE, VAL_SPLIT_FRAC,
    STRICT_TRAJECTORY_ALIGNMENT,
)
from utils.md import box_to_matrix, infer_time_scale, sequence_velocities


RES3_TO_RES1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_IDX = {a: i for i, a in enumerate(AA_ALPHABET)}
HYDROPHOBIC_RESNAMES = {
    "ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "TYR", "PRO", "CYS", "CYX",
}


_META_FEATURE_CACHE = {}
_PACKED_CACHE_KEY = "__packed_frame_map__"


@contextmanager
def _sample_timeout(seconds, traj_dir):
    seconds = int(seconds)
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise_timeout(_signum, _frame):
        raise TimeoutError(f"loading sample timed out after {seconds}s: {Path(traj_dir)}")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)[0]
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def _debug_load(message):
    if not bool(DATASET_DEBUG_LOADS):
        return
    worker = get_worker_info()
    worker_id = -1 if worker is None else int(worker.id)
    rank = os.environ.get("RANK", "0")
    print(f"[dataset rank={rank} worker={worker_id}] {message}", flush=True)


def _append_bad_graph_log(traj_dir, exc):
    try:
        BAD_GRAPH_LOG.parent.mkdir(parents=True, exist_ok=True)
        marker_dir = BAD_GRAPH_LOG.parent / f".{BAD_GRAPH_LOG.name}.seen"
        marker_dir.mkdir(parents=True, exist_ok=True)
        traj_name = Path(traj_dir).name
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in traj_name)
        marker_path = marker_dir / f"{safe_name}.seen"
        try:
            with marker_path.open("x", encoding="utf-8") as marker:
                marker.write(str(Path(traj_dir)) + "\n")
        except FileExistsError:
            return
        with BAD_GRAPH_LOG.open("a", encoding="utf-8") as f:
            f.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t"
                f"traj={traj_name}\tpath={Path(traj_dir)}\t"
                f"error={type(exc).__name__}: {exc}\n"
            )
    except Exception:
        pass


def res3_to_idx(res3):
    return AA_TO_IDX[RES3_TO_RES1.get(str(res3).upper(), "X")]


def _zero_cloud_grid():
    return torch.zeros((CLOUD_CHANNELS, CLOUD_GRID_SIZE, CLOUD_GRID_SIZE, CLOUD_GRID_SIZE), dtype=torch.float32)


def _safe_torch_load(path, mmap=False):
    if mmap:
        try:
            return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except TypeError:
            try:
                return torch.load(path, map_location="cpu", mmap=True)
            except (TypeError, RuntimeError):
                pass
        except RuntimeError:
            # Legacy non-ZIP torch.save files cannot be memory mapped.
            pass
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _frame_path(traj_dir, frame_id):
    return Path(traj_dir) / f"frame_{int(frame_id):04d}.pt"


def _packed_frames_path(traj_dir):
    return Path(traj_dir) / PACKED_FRAMES_NAME


def _has_packed_frames(traj_dir):
    return _packed_frames_path(traj_dir).exists()


def _has_any_frame(traj_dir):
    return next(Path(traj_dir).glob("frame_*.pt"), None) is not None


def _frame_paths(traj_dir):
    return sorted(Path(traj_dir).glob("frame_*.pt"))


def _frame_ids_from_paths(frame_paths):
    return [int(fp.stem.split("_")[1]) for fp in frame_paths]


def _future_frame_ids(
    future_frames=FUTURE_FRAMES,
    source_frame_id=TRAJ_SOURCE_FRAME,
    target_start_frame=TRAJ_TARGET_START_FRAME,
    target_frame_stride=TRAJ_TARGET_FRAME_STRIDE,
    force_final_frame=TRAJ_FORCE_FINAL_FRAME,
):
    future_frames = int(future_frames)
    source_frame_id = int(source_frame_id)
    target_start_frame = int(target_start_frame)
    target_frame_stride = max(1, int(target_frame_stride))
    future_ids = [target_start_frame + i * target_frame_stride for i in range(future_frames)]
    if future_ids and bool(force_final_frame):
        cache_last = int(CACHED_FRAME_START) + int(CACHED_FRAME_COUNT) - 1
        final_id = min(cache_last, source_frame_id + future_frames * target_frame_stride)
        if final_id not in future_ids:
            future_ids[-1] = final_id
    return future_ids


def _load_packed_frame_map(traj_dir):
    # Packed trajectory files contain all cached clouds. mmap keeps untouched
    # future-cloud storage off the process heap and lets the OS read only pages
    # backing tensors that this sample actually consumes.
    packed = _safe_torch_load(_packed_frames_path(traj_dir), mmap=True)
    if isinstance(packed, dict) and "frames" in packed:
        frames = packed["frames"]
        frame_ids = packed.get("frame_ids")
        if isinstance(frames, dict):
            return {int(k): v for k, v in frames.items()}
        if frame_ids is None:
            frame_ids = [
                int(frame.get("t", i)) if isinstance(frame, dict) else i
                for i, frame in enumerate(frames)
            ]
        return {int(fid): frame for fid, frame in zip(frame_ids, frames)}
    if isinstance(packed, list):
        return {
            int(frame.get("t", i)) if isinstance(frame, dict) else i: frame
            for i, frame in enumerate(packed)
        }
    raise ValueError(f"unsupported packed frame file format: {_packed_frames_path(traj_dir)}")


def _load_frame(traj_dir, frame_id, frame_cache=None):
    frame_id = int(frame_id)
    if frame_cache is not None and frame_id in frame_cache:
        return frame_cache[frame_id]

    path = _frame_path(traj_dir, frame_id)
    if path.exists():
        frame = _safe_torch_load(path)
    else:
        if frame_cache is not None and _PACKED_CACHE_KEY in frame_cache:
            frame_map = frame_cache[_PACKED_CACHE_KEY]
        else:
            frame_map = _load_packed_frame_map(traj_dir)
            if frame_cache is not None:
                frame_cache[_PACKED_CACHE_KEY] = frame_map
        if frame_id not in frame_map:
            raise KeyError(f"missing frame id {frame_id} in {_packed_frames_path(traj_dir)}")
        frame = frame_map[frame_id]

    if frame_cache is not None:
        frame_cache[frame_id] = frame
    return frame


def _load_frames(traj_dir, frame_ids):
    frame_ids = [int(fid) for fid in frame_ids]
    if _has_packed_frames(traj_dir):
        frame_map = _load_packed_frame_map(traj_dir)
        missing = [fid for fid in frame_ids if fid not in frame_map]
        if missing:
            raise KeyError(f"missing frame ids in {_packed_frames_path(traj_dir)}: {missing[:5]}")
        return [frame_map[fid] for fid in frame_ids]
    cache = {}
    return [_load_frame(traj_dir, fid, frame_cache=cache) for fid in frame_ids]


def _load_meta_features(traj_dir):
    traj_dir = Path(traj_dir)
    cache_key = str(traj_dir.resolve())
    if DATASET_META_CACHE and cache_key in _META_FEATURE_CACHE:
        return _META_FEATURE_CACHE[cache_key]

    meta = _safe_torch_load(traj_dir / "meta.pt")
    schema = int(meta.get("cache_schema_version", 0)) if isinstance(meta, dict) else 0
    if schema < int(GRAPH_CACHE_SCHEMA_VERSION):
        raise ValueError(
            f"graph cache schema {schema} is older than required schema {GRAPH_CACHE_SCHEMA_VERSION}; "
            "rebuild graph_cache with 0runbuild.sh before training or sampling"
        )
    if meta.get("pocket_selection_mode") != "full_protein_dynamic_residue":
        raise ValueError("cache lacks a topology-only full protein reservoir; rebuild graph_cache")
    full_seq = torch.tensor([res3_to_idx(k[2]) for k in meta["protein_res_keys"]], dtype=torch.long)
    features = {
        "meta": meta,
        "full_seq": full_seq,
        "full_mask": torch.ones_like(full_seq, dtype=torch.float32),
    }
    features.update(_load_node_physics(meta))
    if DATASET_META_CACHE:
        _META_FEATURE_CACHE[cache_key] = features
    return features


def _protein_chain_pair_mask(protein_res_keys):
    mask = []
    for left, right in zip(protein_res_keys[:-1], protein_res_keys[1:]):
        try:
            same_segment = str(left[0]) == str(right[0])
            consecutive_resid = int(right[1]) == int(left[1]) + 1
        except (TypeError, ValueError, IndexError):
            same_segment = False
            consecutive_resid = False
        mask.append(bool(same_segment and consecutive_resid))
    return torch.tensor(mask, dtype=torch.bool)


def _valid_1d_tensor(value, n, dtype):
    if torch.is_tensor(value) and int(value.ndim) == 1 and int(value.shape[0]) == int(n):
        return value.to(dtype=dtype)
    return None


def _infer_interaction_masks(meta, atomic_numbers, n):
    donor = _valid_1d_tensor(meta.get("hbond_donor_mask"), n, torch.bool)
    acceptor = _valid_1d_tensor(meta.get("hbond_acceptor_mask"), n, torch.bool)
    hydrophobic = _valid_1d_tensor(meta.get("hydrophobic_mask"), n, torch.bool)
    if donor is not None and acceptor is not None and hydrophobic is not None:
        return donor, acceptor, hydrophobic

    atom_names = list(meta.get("atom_names", []))
    resnames = list(meta.get("resnames", []))
    if len(atom_names) != int(n):
        atom_names = [""] * int(n)
    if len(resnames) != int(n):
        resnames = [""] * int(n)
    ligand_mask = _valid_1d_tensor(meta.get("node_is_ligand"), n, torch.bool)
    if ligand_mask is None:
        ligand_mask = torch.zeros((int(n),), dtype=torch.bool)

    z = atomic_numbers.long().reshape(-1)
    donor_values = torch.zeros((int(n),), dtype=torch.bool)
    acceptor_values = torch.zeros((int(n),), dtype=torch.bool)
    hydrophobic_values = torch.zeros((int(n),), dtype=torch.bool)
    for i in range(int(n)):
        atom = str(atom_names[i]).strip().upper()
        res = str(resnames[i]).strip().upper()
        zi = int(z[i].item()) if i < z.numel() else 6
        polar = zi in (7, 8, 16)
        donor_values[i] = bool(polar and atom not in {"O", "OXT"})
        acceptor_values[i] = bool(polar and not atom.startswith("H"))
        hydrophobic_values[i] = bool(
            zi in (6, 16)
            and (bool(ligand_mask[i].item()) or res in HYDROPHOBIC_RESNAMES)
            and atom not in {"C", "O", "OXT"}
        )
    return donor_values, acceptor_values, hydrophobic_values


def _load_node_physics(meta):
    node_feat = meta.get("node_feat")
    n = int(node_feat.shape[0]) if torch.is_tensor(node_feat) and int(node_feat.ndim) >= 1 else 0
    if n <= 0:
        return {
            "node_atomic_numbers": torch.zeros((0,), dtype=torch.long),
            "node_masses": torch.zeros((0,), dtype=torch.float32),
            "node_vdw_radius_A": torch.zeros((0,), dtype=torch.float32),
            "hbond_donor_mask": torch.zeros((0,), dtype=torch.bool),
            "hbond_acceptor_mask": torch.zeros((0,), dtype=torch.bool),
            "hydrophobic_mask": torch.zeros((0,), dtype=torch.bool),
            "node_backbone_slot": torch.zeros((0,), dtype=torch.long),
            "bond_edge_index": torch.zeros((2, 0), dtype=torch.long),
        }

    atomic_numbers = _valid_1d_tensor(meta.get("node_atomic_numbers"), n, torch.long)
    masses = _valid_1d_tensor(meta.get("node_masses"), n, torch.float32)
    vdw_radius = _valid_1d_tensor(meta.get("node_vdw_radius_A"), n, torch.float32)
    if torch.is_tensor(node_feat) and int(node_feat.ndim) == 2:
        feat = node_feat.float()
        if atomic_numbers is None and feat.shape[1] >= 1:
            atomic_numbers = torch.round(feat[:, 0] * 53.0).long().clamp_min(1)
        if masses is None and feat.shape[1] >= 2:
            masses = (feat[:, 1] * 126.90).float().clamp_min(1.0)
        if vdw_radius is None and feat.shape[1] >= 3:
            vdw_radius = (feat[:, 2] * 2.75).float().clamp_min(0.5)
    if atomic_numbers is None:
        atomic_numbers = torch.full((n,), 6, dtype=torch.long)
    if masses is None:
        masses = torch.full((n,), 12.011, dtype=torch.float32)
    if vdw_radius is None:
        vdw_radius = torch.full((n,), 1.70, dtype=torch.float32)

    donor, acceptor, hydrophobic = _infer_interaction_masks(meta, atomic_numbers, n)
    backbone_slot = _valid_1d_tensor(meta.get("node_backbone_slot"), n, torch.long)
    if backbone_slot is None:
        atom_names = list(meta.get("atom_names", []))
        node_is_ligand = _valid_1d_tensor(meta.get("node_is_ligand"), n, torch.bool)
        if len(atom_names) != n:
            atom_names = [""] * n
        if node_is_ligand is None:
            node_is_ligand = torch.zeros((n,), dtype=torch.bool)
        backbone_slot = torch.tensor(
            [
                -1 if bool(node_is_ligand[i]) else {"N": 0, "CA": 1, "C": 2}.get(str(atom_names[i]).strip().upper(), -1)
                for i in range(n)
            ],
            dtype=torch.long,
        )
    bond_edge_index = meta.get("bond_edge_index")
    if not torch.is_tensor(bond_edge_index) or int(bond_edge_index.ndim) != 2 or int(bond_edge_index.shape[0]) != 2:
        bond_edge_index = torch.zeros((2, 0), dtype=torch.long)
    return {
        "node_atomic_numbers": atomic_numbers.long(),
        "node_masses": masses.float(),
        "node_vdw_radius_A": vdw_radius.float(),
        "hbond_donor_mask": donor.bool(),
        "hbond_acceptor_mask": acceptor.bool(),
        "hydrophobic_mask": hydrophobic.bool(),
        "node_backbone_slot": backbone_slot.long(),
        "bond_edge_index": bond_edge_index.long(),
    }


def _frame_time(frame, fallback_id):
    value = frame.get("time", None) if isinstance(frame, dict) else None
    if torch.is_tensor(value) and value.numel() > 0:
        return value.reshape(-1)[0].float()
    if value is not None:
        try:
            return torch.tensor(float(value), dtype=torch.float32)
        except (TypeError, ValueError):
            pass
    return torch.tensor(float(frame.get("t", fallback_id)), dtype=torch.float32)


def _frame_box(frame):
    value = frame.get("box") if isinstance(frame, dict) else None
    if value is None:
        return torch.zeros((3, 3), dtype=torch.float32), torch.tensor(False)
    try:
        matrix = box_to_matrix(value, coord_scale=float(COORD_SCALE)).float()
    except (TypeError, ValueError, RuntimeError):
        return torch.zeros((3, 3), dtype=torch.float32), torch.tensor(False)
    valid = bool(torch.isfinite(matrix).all()) and bool(torch.linalg.det(matrix).abs() > 1.0e-8)
    if not valid:
        return torch.zeros((3, 3), dtype=torch.float32), torch.tensor(False)
    return matrix, torch.tensor(True)


def _assert_tensor(name, value, ndim=None, tail_shape=None, finite=False):
    if not torch.is_tensor(value):
        raise ValueError(f"{name} must be a tensor, got {type(value).__name__}")
    if ndim is not None and int(value.ndim) != int(ndim):
        raise ValueError(f"{name} must have ndim={ndim}, got shape={tuple(value.shape)}")
    if tail_shape is not None and tuple(value.shape[-len(tail_shape):]) != tuple(tail_shape):
        raise ValueError(f"{name} must end with shape={tuple(tail_shape)}, got shape={tuple(value.shape)}")
    if finite and value.numel() > 0 and not bool(torch.isfinite(value.float()).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    return value


def _check_limit(name, value, limit):
    limit = int(limit)
    if limit > 0 and int(value) > limit:
        raise ValueError(f"{name}={int(value)} exceeds configured limit {limit}")


def _enforce_lightweight_limits(frame, meta=None, frame_id=None):
    label = "frame" if frame_id is None else f"frame {int(frame_id)}"
    if not isinstance(frame, dict):
        raise ValueError(f"{label} must be a dict, got {type(frame).__name__}")
    coords = frame.get("coords")
    if not torch.is_tensor(coords) or int(coords.ndim) != 2:
        raise ValueError(f"{label} coords must be a 2D tensor")
    n_nodes = int(coords.shape[0])
    if n_nodes <= 0:
        raise ValueError(f"{label} has no nodes")
    _check_limit("graph_nodes", n_nodes, MAX_GRAPH_NODES)

    protein_bb = frame.get("protein_bb")
    if not torch.is_tensor(protein_bb) or int(protein_bb.ndim) != 3:
        raise ValueError(f"{label} protein_bb must be a 3D tensor")
    n_res = int(protein_bb.shape[0])
    if n_res <= 0:
        raise ValueError(f"{label} has no protein residues")
    _check_limit("protein_residues", n_res, MAX_PROTEIN_RESIDUES)

    edge_index = frame.get("edge_index")
    if torch.is_tensor(edge_index):
        if int(edge_index.ndim) != 2 or int(edge_index.shape[0]) != 2:
            raise ValueError(f"{label} edge_index must have shape (2,E), got {tuple(edge_index.shape)}")
        _check_limit("graph_edges", int(edge_index.shape[1]), MAX_GRAPH_EDGES)

    if meta is not None:
        node_feat = meta.get("node_feat")
        if torch.is_tensor(node_feat):
            _check_limit("meta_node_features", int(node_feat.shape[0]), MAX_GRAPH_NODES)


def _validate_frame(frame, frame_id, expected_nodes=None, expected_residues=None, require_graph=False):
    if not bool(DATASET_VALIDATE_GRAPH):
        return
    if not isinstance(frame, dict):
        raise ValueError(f"frame {frame_id} must be a dict, got {type(frame).__name__}")

    coords = _assert_tensor(f"frame {frame_id} coords", frame.get("coords"), ndim=2, tail_shape=(3,), finite=True)
    n_nodes = int(coords.shape[0])
    if n_nodes <= 0:
        raise ValueError(f"frame {frame_id} has no nodes")
    _check_limit("graph_nodes", n_nodes, MAX_GRAPH_NODES)
    if expected_nodes is not None and n_nodes != int(expected_nodes):
        raise ValueError(f"frame {frame_id} node count mismatch: got {n_nodes}, expected {int(expected_nodes)}")

    cloud = _assert_tensor(f"frame {frame_id} cloud_grid", frame.get("cloud_grid"), ndim=4)
    expected_cloud_shape = (int(CLOUD_CHANNELS), int(CLOUD_GRID_SIZE), int(CLOUD_GRID_SIZE), int(CLOUD_GRID_SIZE))
    if tuple(cloud.shape) != expected_cloud_shape:
        raise ValueError(f"frame {frame_id} cloud_grid shape={tuple(cloud.shape)} expected={expected_cloud_shape}")
    _assert_tensor(f"frame {frame_id} cloud_center", frame.get("cloud_center"), ndim=1, tail_shape=(3,), finite=True)
    voxel = _assert_tensor(f"frame {frame_id} cloud_voxel", frame.get("cloud_voxel"), finite=True)
    if voxel.numel() != 1 or float(voxel.reshape(-1)[0].item()) <= 0.0:
        raise ValueError(f"frame {frame_id} cloud_voxel must be a positive scalar")

    protein_bb = _assert_tensor(
        f"frame {frame_id} protein_bb",
        frame.get("protein_bb"),
        ndim=3,
        tail_shape=(3, 3),
        finite=True,
    )
    n_res = int(protein_bb.shape[0])
    if n_res <= 0:
        raise ValueError(f"frame {frame_id} has no protein residues")
    _check_limit("protein_residues", n_res, MAX_PROTEIN_RESIDUES)
    if expected_residues is not None and n_res != int(expected_residues):
        raise ValueError(f"frame {frame_id} residue count mismatch: got {n_res}, expected {int(expected_residues)}")
    protein_mask = _assert_tensor(f"frame {frame_id} protein_bb_mask", frame.get("protein_bb_mask"), ndim=1)
    if int(protein_mask.shape[0]) != n_res:
        raise ValueError(f"frame {frame_id} protein_bb_mask length mismatch")

    if require_graph:
        edge_index = _assert_tensor(f"frame {frame_id} edge_index", frame.get("edge_index"), ndim=2)
        if int(edge_index.shape[0]) != 2:
            raise ValueError(f"frame {frame_id} edge_index must have shape (2,E), got {tuple(edge_index.shape)}")
        n_edges = int(edge_index.shape[1])
        _check_limit("graph_edges", n_edges, MAX_GRAPH_EDGES)
        if n_edges > 0:
            edge_long = edge_index.long()
            if int(edge_long.min().item()) < 0 or int(edge_long.max().item()) >= n_nodes:
                raise ValueError(f"frame {frame_id} edge_index contains out-of-range node ids")
        for key in ("k_idx", "l_idx"):
            idx = _assert_tensor(f"frame {frame_id} {key}", frame.get(key), ndim=1)
            if int(idx.shape[0]) != n_edges:
                raise ValueError(f"frame {frame_id} {key} length mismatch")
            if idx.numel() > 0:
                idx_long = idx.long()
                valid = idx_long >= 0
                if bool(valid.any()) and int(idx_long[valid].max().item()) >= n_nodes:
                    raise ValueError(f"frame {frame_id} {key} contains out-of-range node ids")


def _validate_meta(meta, n_nodes, n_residues):
    if not bool(DATASET_VALIDATE_GRAPH):
        return
    node_feat = _assert_tensor("meta node_feat", meta.get("node_feat"), ndim=2, finite=True)
    if int(node_feat.shape[0]) != int(n_nodes):
        raise ValueError(f"meta node_feat length mismatch: got {node_feat.shape[0]}, expected {int(n_nodes)}")
    node_residue_index = _assert_tensor("meta node_residue_index", meta.get("node_residue_index"), ndim=1)
    if int(node_residue_index.shape[0]) != int(n_nodes):
        raise ValueError("meta node_residue_index length mismatch")
    if node_residue_index.numel() > 0:
        node_residue_long = node_residue_index.long()
        valid_residue = node_residue_long >= 0
        if bool(valid_residue.any()) and int(node_residue_long[valid_residue].max().item()) >= int(n_residues):
            raise ValueError("meta node_residue_index contains out-of-range residue ids")
    if "node_is_ligand" in meta:
        node_is_ligand = _assert_tensor("meta node_is_ligand", meta.get("node_is_ligand"), ndim=1)
        if int(node_is_ligand.shape[0]) != int(n_nodes):
            raise ValueError("meta node_is_ligand length mismatch")
    for key in (
        "node_atomic_numbers",
        "node_masses",
        "node_vdw_radius_A",
        "node_charge",
        "hbond_donor_mask",
        "hbond_acceptor_mask",
        "hydrophobic_mask",
        "node_backbone_slot",
    ):
        if key in meta:
            value = _assert_tensor(f"meta {key}", meta.get(key), ndim=1)
            if int(value.shape[0]) != int(n_nodes):
                raise ValueError(f"meta {key} length mismatch")
    if "bond_edge_index" in meta:
        bond_edge_index = _assert_tensor("meta bond_edge_index", meta.get("bond_edge_index"), ndim=2)
        if int(bond_edge_index.shape[0]) != 2:
            raise ValueError(f"meta bond_edge_index must have shape (2,E), got {tuple(bond_edge_index.shape)}")
        if bond_edge_index.numel() > 0:
            edge_long = bond_edge_index.long()
            if int(edge_long.min().item()) < 0 or int(edge_long.max().item()) >= int(n_nodes):
                raise ValueError("meta bond_edge_index contains out-of-range node ids")
    pocket_residue_indices = _assert_tensor("meta pocket_residue_indices", meta.get("pocket_residue_indices"), ndim=1)
    if pocket_residue_indices.numel() > 0:
        pocket_long = pocket_residue_indices.long()
        valid_pocket = pocket_long >= 0
        if bool(valid_pocket.any()) and int(pocket_long[valid_pocket].max().item()) >= int(n_residues):
            raise ValueError("meta pocket_residue_indices contains out-of-range residue ids")
    protein_res_keys = meta.get("protein_res_keys")
    if protein_res_keys is None or len(protein_res_keys) != int(n_residues):
        raise ValueError(
            f"meta protein_res_keys length mismatch: got {0 if protein_res_keys is None else len(protein_res_keys)}, "
            f"expected {int(n_residues)}"
        )


def split_traj_dirs(graph_root=GRAPH_ROOT, test_frac=TEST_SPLIT_FRAC, val_frac=VAL_SPLIT_FRAC, seed=42, max_dirs=0):
    graph_root = Path(graph_root)
    traj_dirs = sorted([p for p in graph_root.iterdir() if p.is_dir()])
    if not traj_dirs:
        raise RuntimeError(f"no cached graphs under {graph_root}")

    rng = random.Random(int(seed))
    rng.shuffle(traj_dirs)
    if int(max_dirs) > 0:
        traj_dirs = traj_dirs[: min(int(max_dirs), len(traj_dirs))]

    test_frac = max(0.0, float(test_frac))
    val_frac = max(0.0, float(val_frac))
    if test_frac <= 0 and val_frac <= 0:
        return traj_dirs, [], []

    n_total = len(traj_dirs)
    n_test = max(1, int(round(n_total * test_frac))) if test_frac > 0 else 0
    n_val = max(1, int(round(n_total * val_frac))) if val_frac > 0 else 0
    if n_test + n_val >= n_total:
        if n_total <= 2:
            return traj_dirs, [], []
        overflow = n_test + n_val - (n_total - 1)
        while overflow > 0 and n_val > 0:
            n_val -= 1
            overflow -= 1
        while overflow > 0 and n_test > 0:
            n_test -= 1
            overflow -= 1
    test_dirs = traj_dirs[:n_test]
    val_dirs = traj_dirs[n_test:n_test + n_val]
    train_dirs = traj_dirs[n_test + n_val:]
    return train_dirs, val_dirs, test_dirs


def _node_is_ligand(meta, n_nodes):
    value = meta.get("node_is_ligand", torch.zeros((n_nodes,), dtype=torch.bool)).bool()
    if value.numel() != n_nodes:
        return torch.zeros((n_nodes,), dtype=torch.bool)
    return value


def _cloud_center(coords, node_is_ligand):
    if bool(node_is_ligand.any()):
        return coords[node_is_ligand].mean(dim=0)
    return coords.mean(dim=0)


def load_trajectory_sample(
    traj_dir,
    observed_frames=OBSERVED_FRAMES,
    future_frames=FUTURE_FRAMES,
    start_frame=CACHED_FRAME_START,
    source_frame_id=TRAJ_SOURCE_FRAME,
    target_start_frame=TRAJ_TARGET_START_FRAME,
    target_frame_stride=TRAJ_TARGET_FRAME_STRIDE,
    observed_frame_stride=TRAJ_OBSERVED_FRAME_STRIDE,
    force_final_frame=TRAJ_FORCE_FINAL_FRAME,
):
    traj_dir = Path(traj_dir)
    observed_frames = int(observed_frames)
    future_frames = int(future_frames)
    start_frame = int(start_frame)
    observed_frame_stride = max(1, int(observed_frame_stride))
    target_frame_stride = max(1, int(target_frame_stride))
    observed_ids = [start_frame + i * observed_frame_stride for i in range(observed_frames)]
    expected_source_id = int(observed_ids[-1])
    source_frame_id = expected_source_id if source_frame_id is None else int(source_frame_id)
    if bool(STRICT_TRAJECTORY_ALIGNMENT) and source_frame_id != expected_source_id:
        raise ValueError(
            f"source frame must equal the last observed frame: source={source_frame_id}, "
            f"last_observed={expected_source_id}. Clear TRAJ_SOURCE_FRAME or set it to {expected_source_id}."
        )
    target_start_frame = source_frame_id + target_frame_stride if target_start_frame is None else int(target_start_frame)
    if bool(STRICT_TRAJECTORY_ALIGNMENT) and target_start_frame <= source_frame_id:
        raise ValueError(
            f"target_start_frame={target_start_frame} must be later than source_frame={source_frame_id}"
        )

    future_ids = _future_frame_ids(
        future_frames=future_frames,
        source_frame_id=source_frame_id,
        target_start_frame=target_start_frame,
        target_frame_stride=target_frame_stride,
        force_final_frame=force_final_frame,
    )
    load_ids = sorted(set(observed_ids + future_ids + [source_frame_id]))
    loaded = {fid: frame for fid, frame in zip(load_ids, _load_frames(traj_dir, load_ids))}
    obs_frames = [loaded[fid] for fid in observed_ids]
    future_frames_list = [loaded[fid] for fid in future_ids]
    source_frame = loaded[source_frame_id]

    meta_features = _load_meta_features(traj_dir)
    meta = meta_features["meta"]
    for fid in load_ids:
        _enforce_lightweight_limits(loaded[fid], meta=meta if fid == source_frame_id else None, frame_id=fid)
    if bool(DATASET_VALIDATE_GRAPH):
        _validate_frame(source_frame, source_frame_id, require_graph=True)
        expected_nodes = int(source_frame["coords"].shape[0])
        expected_residues = int(source_frame["protein_bb"].shape[0])
        _validate_meta(meta, expected_nodes, expected_residues)
        for fid in observed_ids:
            _validate_frame(loaded[fid], fid, expected_nodes=expected_nodes, expected_residues=expected_residues)
        for fid in future_ids:
            _validate_frame(loaded[fid], fid, expected_nodes=expected_nodes, expected_residues=expected_residues)
    node_is_ligand = _node_is_ligand(meta, obs_frames[0]["coords"].shape[0])

    obs_coords = torch.stack([f["coords"].float() for f in obs_frames], dim=0)
    future_coords = torch.stack([f["coords"].float() for f in future_frames_list], dim=0)
    obs_cloud = torch.stack([f["cloud_grid"].float() for f in obs_frames], dim=0)
    obs_center = torch.stack([f["cloud_center"].float() for f in obs_frames], dim=0)
    obs_voxel = torch.stack([f["cloud_voxel"].float() for f in obs_frames], dim=0)

    obs_protein = torch.stack([f["protein_bb"].float() for f in obs_frames], dim=0)
    obs_protein_mask = torch.stack([f["protein_bb_mask"].float() for f in obs_frames], dim=0)
    future_protein = torch.stack([f["protein_bb"].float() for f in future_frames_list], dim=0)
    future_protein_mask = torch.stack([f["protein_bb_mask"].float() for f in future_frames_list], dim=0)
    obs_time = torch.stack([_frame_time(f, fid) for f, fid in zip(obs_frames, observed_ids)], dim=0)
    future_time = torch.stack([_frame_time(f, fid) for f, fid in zip(future_frames_list, future_ids)], dim=0)
    source_time = _frame_time(source_frame, source_frame_id)
    if bool(STRICT_TRAJECTORY_ALIGNMENT) and obs_time.numel() > 1 and bool((obs_time[1:] <= obs_time[:-1]).any()):
        raise ValueError(f"observed times must be strictly increasing, got {obs_time.tolist()}")
    obs_box_data = [_frame_box(f) for f in obs_frames]
    future_box_data = [_frame_box(f) for f in future_frames_list]
    source_box, source_box_mask = _frame_box(source_frame)
    obs_box = torch.stack([item[0] for item in obs_box_data], dim=0)
    obs_box_mask = torch.stack([item[1] for item in obs_box_data], dim=0).bool()
    future_box = torch.stack([item[0] for item in future_box_data], dim=0)
    future_box_mask = torch.stack([item[1] for item in future_box_data], dim=0).bool()
    time_scale = infer_time_scale(obs_time)
    obs_velocity = sequence_velocities(obs_coords, obs_time, boxes=obs_box)
    obs_protein_velocity = sequence_velocities(obs_protein, obs_time, boxes=obs_box)
    source_velocity = obs_velocity[-1]
    source_protein_velocity = obs_protein_velocity[-1]
    future_dt = future_time - torch.cat([source_time.reshape(1), future_time[:-1]], dim=0)
    if bool(STRICT_TRAJECTORY_ALIGNMENT) and bool((future_dt <= 0).any()):
        raise ValueError(
            f"future times must be strictly increasing after source time; source={float(source_time)}, "
            f"future={future_time.tolist()}"
        )
    bond_edge_index = meta_features["bond_edge_index"]

    return {
        "obs_coords": obs_coords,
        "future_coords": future_coords,
        **load_qm_targets(traj_dir, future_ids, future_frames_list),
        "source_coords": source_frame["coords"].float(),
        "obs_velocity": obs_velocity.float(),
        "source_velocity": source_velocity.float(),
        "source_velocity_scaled": (source_velocity * time_scale).float(),
        "edge_index": source_frame["edge_index"].long(),
        "k_idx": source_frame["k_idx"].long(),
        "l_idx": source_frame["l_idx"].long(),
        "obs_cloud_grid_seq": obs_cloud,
        "obs_cloud_center_seq": obs_center.float(),
        "source_cloud_center": source_frame["cloud_center"].float(),
        "obs_cloud_voxel_seq": obs_voxel.float(),
        "source_cloud_voxel": source_frame["cloud_voxel"].float(),
        "node_feat": meta["node_feat"].float(),
        "node_is_ligand": node_is_ligand,
        "node_atomic_numbers": meta_features["node_atomic_numbers"].long(),
        "node_masses": meta_features["node_masses"].float(),
        "node_vdw_radius_A": meta_features["node_vdw_radius_A"].float(),
        "hbond_donor_mask": meta_features["hbond_donor_mask"].bool(),
        "hbond_acceptor_mask": meta_features["hbond_acceptor_mask"].bool(),
        "hydrophobic_mask": meta_features["hydrophobic_mask"].bool(),
        "node_backbone_slot": meta_features["node_backbone_slot"].long(),
        "node_residue_index": meta["node_residue_index"].long(),
        "pocket_residue_indices": meta["pocket_residue_indices"].long(),
        "bond_edge_index": bond_edge_index.long(),
        "protein_full_seq_ids": meta_features["full_seq"],
        "protein_full_mask": meta_features["full_mask"],
        "protein_chain_pair_mask": _protein_chain_pair_mask(meta["protein_res_keys"]),
        "obs_protein_bb": obs_protein,
        "obs_protein_bb_mask": obs_protein_mask,
        "source_protein_bb": source_frame["protein_bb"].float(),
        "source_protein_bb_mask": source_frame["protein_bb_mask"].float(),
        "obs_protein_velocity": obs_protein_velocity.float(),
        "source_protein_velocity": source_protein_velocity.float(),
        "source_protein_velocity_scaled": (source_protein_velocity * time_scale).float(),
        "future_protein_bb": future_protein,
        "future_protein_bb_mask": future_protein_mask,
        "obs_times": obs_time.float(),
        "future_times": future_time.float(),
        "source_time": source_time.float(),
        "time_scale": time_scale.float(),
        "future_dt": future_dt.float(),
        "future_dt_scaled": (future_dt / time_scale.clamp_min(1.0e-8)).float(),
        "obs_box": obs_box.float(),
        "obs_box_mask": obs_box_mask,
        "source_box": source_box.float(),
        "source_box_mask": source_box_mask.bool(),
        "future_box": future_box.float(),
        "future_box_mask": future_box_mask,
        "traj_dir": str(traj_dir),
        "observed_frame_ids": observed_ids,
        "future_frame_ids": future_ids,
        "frame_id": int(source_frame.get("t", source_frame_id)),
        "target_frame_id": int(future_frames_list[-1].get("t", future_ids[-1])),
        "horizon": int(future_ids[-1]) - int(source_frame_id),
        "target_frame_stride": int(target_frame_stride),
        "observed_frame_stride": int(observed_frame_stride),
    }


def load_observed_trajectory_sample(traj_dir, observed_frames=OBSERVED_FRAMES, start_frame=CACHED_FRAME_START):
    traj_dir = Path(traj_dir)
    observed_frames = int(observed_frames)
    frame_ids = list(range(int(start_frame), int(start_frame) + observed_frames))
    obs_frames = _load_frames(traj_dir, frame_ids)

    meta_features = _load_meta_features(traj_dir)
    meta = meta_features["meta"]
    source_frame = obs_frames[-1]
    for fid, frame in zip(frame_ids, obs_frames):
        _enforce_lightweight_limits(frame, meta=meta if fid == frame_ids[-1] else None, frame_id=fid)
    if bool(DATASET_VALIDATE_GRAPH):
        _validate_frame(source_frame, frame_ids[-1], require_graph=True)
        expected_nodes = int(source_frame["coords"].shape[0])
        expected_residues = int(source_frame["protein_bb"].shape[0])
        _validate_meta(meta, expected_nodes, expected_residues)
        for fid, frame in zip(frame_ids, obs_frames):
            _validate_frame(frame, fid, expected_nodes=expected_nodes, expected_residues=expected_residues)
    node_is_ligand = _node_is_ligand(meta, obs_frames[0]["coords"].shape[0])

    obs_coords = torch.stack([f["coords"].float() for f in obs_frames], dim=0)
    obs_cloud = torch.stack([f["cloud_grid"].float() for f in obs_frames], dim=0)
    obs_center = torch.stack([f["cloud_center"].float() for f in obs_frames], dim=0)
    obs_voxel = torch.stack([f["cloud_voxel"].float() for f in obs_frames], dim=0)
    obs_protein = torch.stack([f["protein_bb"].float() for f in obs_frames], dim=0)
    obs_protein_mask = torch.stack([f["protein_bb_mask"].float() for f in obs_frames], dim=0)
    obs_time = torch.stack([_frame_time(f, fid) for f, fid in zip(obs_frames, frame_ids)], dim=0)
    source_time = _frame_time(source_frame, frame_ids[-1])
    if bool(STRICT_TRAJECTORY_ALIGNMENT) and obs_time.numel() > 1 and bool((obs_time[1:] <= obs_time[:-1]).any()):
        raise ValueError(f"observed times must be strictly increasing, got {obs_time.tolist()}")
    obs_box_data = [_frame_box(f) for f in obs_frames]
    obs_box = torch.stack([item[0] for item in obs_box_data], dim=0)
    obs_box_mask = torch.stack([item[1] for item in obs_box_data], dim=0).bool()
    source_box, source_box_mask = _frame_box(source_frame)
    time_scale = infer_time_scale(obs_time)
    obs_velocity = sequence_velocities(obs_coords, obs_time, boxes=obs_box)
    obs_protein_velocity = sequence_velocities(obs_protein, obs_time, boxes=obs_box)
    bond_edge_index = meta_features["bond_edge_index"]

    return {
        "obs_coords": obs_coords,
        "future_coords": source_frame["coords"].float().unsqueeze(0),
        "source_coords": source_frame["coords"].float(),
        "obs_velocity": obs_velocity.float(),
        "source_velocity": obs_velocity[-1].float(),
        "source_velocity_scaled": (obs_velocity[-1] * time_scale).float(),
        "edge_index": source_frame["edge_index"].long(),
        "k_idx": source_frame["k_idx"].long(),
        "l_idx": source_frame["l_idx"].long(),
        "obs_cloud_grid_seq": obs_cloud,
        "obs_cloud_center_seq": obs_center.float(),
        "source_cloud_center": obs_center[-1].float(),
        "obs_cloud_voxel_seq": obs_voxel.float(),
        "source_cloud_voxel": obs_voxel[-1].float(),
        "node_feat": meta["node_feat"].float(),
        "node_is_ligand": node_is_ligand,
        "node_atomic_numbers": meta_features["node_atomic_numbers"].long(),
        "node_masses": meta_features["node_masses"].float(),
        "node_vdw_radius_A": meta_features["node_vdw_radius_A"].float(),
        "hbond_donor_mask": meta_features["hbond_donor_mask"].bool(),
        "hbond_acceptor_mask": meta_features["hbond_acceptor_mask"].bool(),
        "hydrophobic_mask": meta_features["hydrophobic_mask"].bool(),
        "node_backbone_slot": meta_features["node_backbone_slot"].long(),
        "node_residue_index": meta["node_residue_index"].long(),
        "pocket_residue_indices": meta["pocket_residue_indices"].long(),
        "bond_edge_index": bond_edge_index.long(),
        "protein_full_seq_ids": meta_features["full_seq"],
        "protein_full_mask": meta_features["full_mask"],
        "protein_chain_pair_mask": _protein_chain_pair_mask(meta["protein_res_keys"]),
        "obs_protein_bb": obs_protein,
        "obs_protein_bb_mask": obs_protein_mask,
        "source_protein_bb": source_frame["protein_bb"].float(),
        "source_protein_bb_mask": source_frame["protein_bb_mask"].float(),
        "obs_protein_velocity": obs_protein_velocity.float(),
        "source_protein_velocity": obs_protein_velocity[-1].float(),
        "source_protein_velocity_scaled": (obs_protein_velocity[-1] * time_scale).float(),
        "future_protein_bb": source_frame["protein_bb"].float().unsqueeze(0),
        "future_protein_bb_mask": source_frame["protein_bb_mask"].float().unsqueeze(0),
        "obs_times": obs_time.float(),
        "future_times": obs_time[-1:].float(),
        "source_time": source_time.float(),
        "time_scale": time_scale.float(),
        "future_dt": time_scale.reshape(1).float(),
        "future_dt_scaled": torch.ones((1,), dtype=torch.float32),
        "obs_box": obs_box.float(),
        "obs_box_mask": obs_box_mask,
        "source_box": source_box.float(),
        "source_box_mask": source_box_mask.bool(),
        "future_box": source_box.float().unsqueeze(0),
        "future_box_mask": source_box_mask.bool().reshape(1),
        "traj_dir": str(traj_dir),
        "observed_frame_ids": frame_ids,
        "future_frame_ids": [],
        "frame_id": int(source_frame.get("t", frame_ids[-1])),
        "target_frame_id": int(source_frame.get("t", frame_ids[-1])),
        "horizon": 0,
        "target_frame_stride": 1,
        "observed_frame_stride": 1,
    }


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        graph_root=GRAPH_ROOT,
        observed_frames=OBSERVED_FRAMES,
        future_frames=FUTURE_FRAMES,
        start_frame=CACHED_FRAME_START,
        traj_dirs=None,
        source_frame_id=TRAJ_SOURCE_FRAME,
        target_start_frame=TRAJ_TARGET_START_FRAME,
        target_frame_stride=TRAJ_TARGET_FRAME_STRIDE,
        observed_frame_stride=TRAJ_OBSERVED_FRAME_STRIDE,
        force_final_frame=TRAJ_FORCE_FINAL_FRAME,
        window_stride=TRAJ_WINDOW_STRIDE,
        random_window=TRAJ_RANDOM_WINDOW,
        random_stride=TRAJ_RANDOM_STRIDE,
        min_frame_stride=TRAJ_MIN_FRAME_STRIDE,
        max_frame_stride=TRAJ_MAX_FRAME_STRIDE,
        random_windows_per_traj=TRAJ_RANDOM_WINDOWS_PER_TRAJ,
    ):
        self.observed_frames = int(observed_frames)
        self.future_frames = int(future_frames)
        self.start_frame = int(start_frame)
        self.source_frame_id = source_frame_id
        self.target_start_frame = target_start_frame
        self.target_frame_stride = max(1, int(target_frame_stride))
        self.observed_frame_stride = max(1, int(observed_frame_stride))
        self.force_final_frame = bool(force_final_frame)
        self.window_stride = max(0, int(window_stride))
        self.random_window = bool(random_window)
        self.random_stride = bool(random_stride)
        self.min_frame_stride = max(1, int(min_frame_stride))
        self.max_frame_stride = max(self.min_frame_stride, int(max_frame_stride))
        self.random_windows_per_traj = max(1, int(random_windows_per_traj))

        if traj_dirs is None:
            candidates = sorted([p for p in Path(graph_root).iterdir() if p.is_dir()])
        else:
            candidates = [Path(p) for p in traj_dirs]
        self.bad_dirs = set()
        self.traj_dirs = candidates
        if not self.traj_dirs:
            raise RuntimeError(f"no cached graphs under {graph_root}")

        self.source_offset = (
            (self.observed_frames - 1) * self.observed_frame_stride
            if self.source_frame_id is None
            else int(self.source_frame_id) - int(CACHED_FRAME_START)
        )
        self.target_offset = (
            self.source_offset + self.target_frame_stride
            if self.target_start_frame is None
            else int(self.target_start_frame) - int(CACHED_FRAME_START)
        )

        cache_last = int(CACHED_FRAME_START) + int(CACHED_FRAME_COUNT) - 1
        regular_future = self.future_frames - 1
        if self.force_final_frame and self.future_frames > 1:
            regular_future = self.future_frames - 2
        required_end = max(
            self.start_frame + (self.observed_frames - 1) * self.observed_frame_stride,
            self.start_frame + self.source_offset,
            self.start_frame + self.target_offset + max(0, regular_future) * self.target_frame_stride,
        )

        if self.random_window:
            self.samples = [
                (traj_dir, None)
                for _ in range(self.random_windows_per_traj)
                for traj_dir in self.traj_dirs
            ]
        elif self.window_stride > 0:
            max_start = min(
                cache_last - (self.observed_frames - 1) * self.observed_frame_stride,
                cache_last - self.source_offset,
                cache_last - self.target_offset - max(0, regular_future) * self.target_frame_stride,
            )
            if max_start < self.start_frame:
                raise RuntimeError(
                    f"trajectory window does not fit cached frames: observed={self.observed_frames} "
                    f"future={self.future_frames} target_start_offset={self.target_offset} "
                    f"target_stride={self.target_frame_stride} cached_range=[{CACHED_FRAME_START},{cache_last}]"
                )
            window_starts = list(range(self.start_frame, max_start + 1, self.window_stride))
            self.samples = [(traj_dir, win_start) for traj_dir in self.traj_dirs for win_start in window_starts]
        else:
            if required_end > cache_last:
                raise RuntimeError(
                    f"trajectory window does not fit cached frames: observed={self.observed_frames} "
                    f"future={self.future_frames} start={self.start_frame} source_offset={self.source_offset} "
                    f"target_start_offset={self.target_offset} target_stride={self.target_frame_stride} "
                    f"requires_frame={required_end} cached_range=[{CACHED_FRAME_START},{cache_last}]"
                )
            self.samples = [(traj_dir, self.start_frame) for traj_dir in self.traj_dirs]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        first_idx = int(idx)
        attempts = len(self.samples) if bool(SKIP_BAD_GRAPH_CACHE) else 1
        last_exc = None
        for attempt in range(attempts):
            cur_idx = (first_idx + attempt) % len(self.samples)
            traj_dir, window_start = self.samples[cur_idx]
            bad_key = str(traj_dir.resolve())
            if bad_key in self.bad_dirs:
                _debug_load(f"skip_known_bad idx={cur_idx} traj={traj_dir.name}")
                continue

            if self.random_window:
                target_stride = random.randint(self.min_frame_stride, self.max_frame_stride) if self.random_stride else self.target_frame_stride
                observed_stride = target_stride if self.random_stride else self.observed_frame_stride
                observed_span = (self.observed_frames - 1) * observed_stride
                regular_future = self.future_frames - 1
                if self.force_final_frame and self.future_frames > 1:
                    regular_future = self.future_frames - 2
                cache_last = int(CACHED_FRAME_START) + int(CACHED_FRAME_COUNT) - 1
                required_span = observed_span + target_stride + max(0, regular_future) * target_stride
                max_start = cache_last - required_span
                if max_start < self.start_frame:
                    raise ValueError(
                        f"trajectory window does not fit cached frames: observed={self.observed_frames} "
                        f"future={self.future_frames} stride={target_stride} cached={CACHED_FRAME_COUNT}"
                    )
                window_start = random.randint(self.start_frame, max_start)
                source_frame_id = int(window_start) + observed_span
                target_start_frame = source_frame_id + target_stride
            else:
                target_stride = self.target_frame_stride
                observed_stride = self.observed_frame_stride
                source_frame_id = int(window_start) + int(self.source_offset)
                target_start_frame = int(window_start) + int(self.target_offset)
            start = time.perf_counter()
            _debug_load(
                f"load_start idx={cur_idx} requested_idx={first_idx} traj={traj_dir.name} "
                f"window_start={window_start} source={source_frame_id} target_start={target_start_frame} "
                f"obs_stride={observed_stride} target_stride={target_stride}"
            )
            try:
                with _sample_timeout(DATASET_SAMPLE_TIMEOUT_SECONDS, traj_dir):
                    sample = load_trajectory_sample(
                        traj_dir,
                        observed_frames=self.observed_frames,
                        future_frames=self.future_frames,
                        start_frame=window_start,
                        source_frame_id=source_frame_id,
                        target_start_frame=target_start_frame,
                        target_frame_stride=target_stride,
                        observed_frame_stride=observed_stride,
                        force_final_frame=self.force_final_frame,
                    )
            except Exception as exc:
                last_exc = exc
                self.bad_dirs.add(bad_key)
                _append_bad_graph_log(traj_dir, exc)
                _debug_load(
                    f"load_error idx={cur_idx} requested_idx={first_idx} traj={traj_dir.name} "
                    f"error={type(exc).__name__}: {exc}"
                )
                if not bool(SKIP_BAD_GRAPH_CACHE):
                    raise
                continue
            _debug_load(
                f"load_done idx={cur_idx} requested_idx={first_idx} traj={traj_dir.name} "
                f"seconds={time.perf_counter() - start:.3f}"
            )
            return sample

        raise RuntimeError(
            f"all graph cache samples failed while trying to replace idx={first_idx}; "
            f"last error={type(last_exc).__name__ if last_exc is not None else 'unknown'}: {last_exc}"
        )
