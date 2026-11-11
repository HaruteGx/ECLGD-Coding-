from pathlib import Path
import os
import time

import torch

from config import (
    CACHED_FRAME_COUNT,
    CLOUD_CHANNELS,
    CLOUD_CHANNEL_NAMES,
    CLOUD_GRID_SIZE,
    DATA_ROOT,
    GRAPH_ROOT,
    PACKED_FRAMES_NAME,
)
from preprocess.build_graph_cache import (
    _cache_dir_is_complete,
    _elapsed_s,
    _iter_cache_dirs,
    _iter_npz_files,
    _stem_shard,
    _torch_load_cache,
)


REQUIRED_META_KEYS = (
    "cache_schema_version",
    "source_npz",
    "source_npz_name",
    "source_npz_stem",
    "source_npz_size",
    "source_npz_mtime_ns",
    "sel_global_indices",
    "node_global_indices",
    "anchor_global_indices",
    "node_feat",
    "node_atomic_numbers",
    "node_masses",
    "node_vdw_radius_A",
    "node_charge",
    "hbond_donor_mask",
    "hbond_acceptor_mask",
    "hydrophobic_mask",
    "node_is_ligand",
    "node_residue_index",
    "node_backbone_slot",
    "pocket_residue_indices",
    "bond_edge_index",
    "segids",
    "resids",
    "resnames",
    "atom_names",
    "ligand_resnames",
    "protein_res_keys",
    "pocket_res_keys",
    "cloud_channels",
    "cloud_cutoff_A",
    "cloud_grid_size",
    "cloud_box_padding_A",
    "cloud_min_voxel_A",
    "cloud_max_voxel_A",
    "cloud_sigma_A",
    "pocket_selection_mode",
    "pbc_unwrap_requested",
    "coordinates_unwrapped",
)

REQUIRED_FRAME_KEYS = (
    "t",
    "time",
    "coords",
    "active_pocket_mask",
    "edge_index",
    "k_idx",
    "l_idx",
    "cloud_grid",
    "cloud_center",
    "cloud_voxel",
    "cloud_half_width",
    "protein_bb",
    "protein_bb_mask",
)


def _env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name, default=0):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _build_npz_index():
    index = {}
    duplicates = set()
    for path in _iter_npz_files(DATA_ROOT):
        if path.stem in index:
            duplicates.add(path.stem)
        else:
            index[path.stem] = path
    return index, duplicates


def _record_bad(bad_items, reason, path, detail=""):
    message = f"cache_check_bad reason={reason} path={path}"
    if detail:
        message += f" {detail}"
    print(message, flush=True)
    bad_items.append((str(path), reason, detail))


def _is_tensor_shape(x, shape=None, ndim=None):
    if not torch.is_tensor(x):
        return False
    if ndim is not None and x.ndim != int(ndim):
        return False
    if shape is not None and tuple(x.shape) != tuple(shape):
        return False
    return True


def _check_meta_shapes(meta, cache_dir, bad_items):
    ok = True
    node_feat = meta.get("node_feat")
    if not _is_tensor_shape(node_feat, ndim=2) or int(node_feat.shape[-1]) < 5:
        _record_bad(bad_items, "bad_node_feat_shape", cache_dir, f"shape={getattr(node_feat, 'shape', None)}")
        return False
    n_nodes = int(node_feat.shape[0])
    one_dim_node_keys = (
        "node_atomic_numbers",
        "node_masses",
        "node_vdw_radius_A",
        "node_charge",
        "hbond_donor_mask",
        "hbond_acceptor_mask",
        "hydrophobic_mask",
        "node_is_ligand",
        "node_residue_index",
        "node_backbone_slot",
    )
    for key in one_dim_node_keys:
        value = meta.get(key)
        if not torch.is_tensor(value) or value.ndim != 1 or int(value.shape[0]) != n_nodes:
            _record_bad(bad_items, "bad_node_field_shape", cache_dir, f"key={key} shape={getattr(value, 'shape', None)} expected=({n_nodes},)")
            ok = False
    bond_edge_index = meta.get("bond_edge_index")
    if not torch.is_tensor(bond_edge_index) or bond_edge_index.ndim != 2 or int(bond_edge_index.shape[0]) != 2:
        _record_bad(bad_items, "bad_bond_edge_index_shape", cache_dir, f"shape={getattr(bond_edge_index, 'shape', None)} expected=(2,E)")
        ok = False
    return ok


def _frame_list(packed):
    frame_ids = packed.get("frame_ids")
    frames = packed.get("frames")
    if isinstance(frames, dict):
        out = []
        for frame_id in frame_ids:
            out.append(frames.get(frame_id, frames.get(str(frame_id))))
        return out
    if isinstance(frames, (list, tuple)):
        return list(frames)
    return None


def _check_frame_shapes(frames, cache_dir, bad_items):
    ok = True
    expected_cloud_shape = (
        int(CLOUD_CHANNELS),
        int(CLOUD_GRID_SIZE),
        int(CLOUD_GRID_SIZE),
        int(CLOUD_GRID_SIZE),
    )
    for i, frame in enumerate(frames[: int(CACHED_FRAME_COUNT)]):
        if not isinstance(frame, dict):
            _record_bad(bad_items, "frame_not_dict", cache_dir, f"frame_index={i}")
            return False
        missing = [key for key in REQUIRED_FRAME_KEYS if key not in frame]
        if missing:
            _record_bad(bad_items, "missing_frame_keys", cache_dir, f"frame_index={i} keys={missing}")
            ok = False
            continue
        if not _is_tensor_shape(frame.get("cloud_grid"), shape=expected_cloud_shape):
            _record_bad(
                bad_items,
                "bad_cloud_grid_shape",
                cache_dir,
                f"frame_index={i} shape={getattr(frame.get('cloud_grid'), 'shape', None)} expected={expected_cloud_shape}",
            )
            ok = False
        coords = frame.get("coords")
        if not torch.is_tensor(coords) or coords.ndim != 2 or int(coords.shape[-1]) != 3:
            _record_bad(bad_items, "bad_coords_shape", cache_dir, f"frame_index={i} shape={getattr(coords, 'shape', None)} expected=(N,3)")
            ok = False
        edge_index = frame.get("edge_index")
        if not torch.is_tensor(edge_index) or edge_index.ndim != 2 or int(edge_index.shape[0]) != 2:
            _record_bad(bad_items, "bad_edge_index_shape", cache_dir, f"frame_index={i} shape={getattr(edge_index, 'shape', None)} expected=(2,E)")
            ok = False
        if not _is_tensor_shape(frame.get("cloud_center"), shape=(3,)):
            _record_bad(bad_items, "bad_cloud_center_shape", cache_dir, f"frame_index={i} shape={getattr(frame.get('cloud_center'), 'shape', None)} expected=(3,)")
            ok = False
        protein_bb = frame.get("protein_bb")
        if not torch.is_tensor(protein_bb) or protein_bb.ndim != 3 or tuple(protein_bb.shape[-2:]) != (3, 3):
            _record_bad(bad_items, "bad_protein_bb_shape", cache_dir, f"frame_index={i} shape={getattr(protein_bb, 'shape', None)} expected=(L,3,3)")
            ok = False
    return ok


def _check_cache_dir(cache_dir, npz_path, bad_items, validate_frames=True):
    start_time = time.monotonic()
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "meta.pt"
    packed_path = cache_dir / PACKED_FRAMES_NAME
    if npz_path is None:
        _record_bad(bad_items, "missing_matching_npz", cache_dir)
        return False
    if not meta_path.exists():
        _record_bad(bad_items, "missing_meta_pt", cache_dir)
        return False
    if not packed_path.exists():
        _record_bad(bad_items, "missing_frames_pt", cache_dir)
        return False
    try:
        meta = _torch_load_cache(meta_path)
    except Exception as exc:
        _record_bad(bad_items, "meta_load_error", cache_dir, f"error={type(exc).__name__}: {exc}")
        return False
    if not isinstance(meta, dict):
        _record_bad(bad_items, "meta_not_dict", cache_dir)
        return False
    missing = [key for key in REQUIRED_META_KEYS if key not in meta]
    if missing:
        _record_bad(bad_items, "missing_meta_keys", cache_dir, f"keys={missing}")
        return False
    if not _check_meta_shapes(meta, cache_dir, bad_items):
        return False
    if not _cache_dir_is_complete(cache_dir, npz_path=npz_path, validate_packed_load=False):
        _record_bad(bad_items, "not_current_build_format", cache_dir)
        return False
    if validate_frames:
        try:
            packed = _torch_load_cache(packed_path)
        except Exception as exc:
            _record_bad(bad_items, "frames_load_error", cache_dir, f"error={type(exc).__name__}: {exc}")
            return False
        if not isinstance(packed, dict):
            _record_bad(bad_items, "packed_not_dict", cache_dir)
            return False
        if int(packed.get("version", 0)) < 2:
            _record_bad(bad_items, "old_packed_version", cache_dir, f"version={packed.get('version', None)}")
            return False
        frame_ids = packed.get("frame_ids")
        if frame_ids is None or len(frame_ids) < int(CACHED_FRAME_COUNT):
            _record_bad(bad_items, "bad_frame_ids", cache_dir, f"frame_ids={0 if frame_ids is None else len(frame_ids)} expected={int(CACHED_FRAME_COUNT)}")
            return False
        expected_ids = list(range(int(CACHED_FRAME_COUNT)))
        if [int(x) for x in frame_ids[: int(CACHED_FRAME_COUNT)]] != expected_ids:
            _record_bad(bad_items, "unexpected_frame_ids", cache_dir)
            return False
        frames = _frame_list(packed)
        if frames is None:
            _record_bad(bad_items, "frames_not_list_or_dict", cache_dir)
            return False
        if len(frames) < int(CACHED_FRAME_COUNT):
            _record_bad(bad_items, "too_few_frames", cache_dir, f"frames={len(frames)} expected={int(CACHED_FRAME_COUNT)}")
            return False
        if not _check_frame_shapes(frames, cache_dir, bad_items):
            return False
    print(f"cache_check_ok elapsed_s={_elapsed_s(start_time)} path={cache_dir}", flush=True)
    return True


def main():
    start_time = time.monotonic()
    validate_frames = _env_bool("CHECK_CACHE_VALIDATE_FRAMES", True)
    require_all_npz = _env_bool("CHECK_CACHE_REQUIRE_ALL_NPZ", True)
    max_bad_print = max(1, _env_int("CHECK_CACHE_MAX_BAD_PRINT", 200))
    shard_count = max(1, _env_int("CHECK_CACHE_SHARD_COUNT", 1))
    shard_index = _env_int("CHECK_CACHE_SHARD_INDEX", 0)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"CHECK_CACHE_SHARD_INDEX={shard_index} must be in [0, {shard_count})")

    print(
        "cache_check_start:",
        f"graph_root={GRAPH_ROOT}",
        f"data_root={DATA_ROOT}",
        f"grid={CLOUD_GRID_SIZE}",
        f"channels={','.join(CLOUD_CHANNEL_NAMES)}",
        f"validate_frames={int(validate_frames)}",
        f"require_all_npz={int(require_all_npz)}",
        f"shard={shard_index}/{shard_count}",
        flush=True,
    )

    npz_index, duplicate_stems = _build_npz_index()
    bad_items = []
    if duplicate_stems:
        for stem in sorted(duplicate_stems)[:max_bad_print]:
            _record_bad(bad_items, "duplicate_npz_stem", DATA_ROOT, f"stem={stem}")

    cache_seen = 0
    cache_checked = 0
    cache_ok = 0
    cache_stems = set()
    for cache_dir in _iter_cache_dirs(GRAPH_ROOT):
        cache_seen += 1
        stem = cache_dir.name
        cache_stems.add(stem)
        if _stem_shard(stem, shard_count) != shard_index:
            continue
        cache_checked += 1
        if len(bad_items) >= max_bad_print:
            print("cache_check_bad_limit_reached", flush=True)
            break
        if _check_cache_dir(cache_dir, npz_index.get(stem), bad_items, validate_frames=validate_frames):
            cache_ok += 1

    missing_cache = 0
    if require_all_npz:
        for stem, npz_path in sorted(npz_index.items()):
            if _stem_shard(stem, shard_count) != shard_index:
                continue
            if stem not in cache_stems:
                missing_cache += 1
                if len(bad_items) < max_bad_print:
                    _record_bad(bad_items, "missing_graph_cache_for_npz", npz_path)

    bad_count = len(bad_items)
    print(
        "cache_check_done:",
        f"cache_seen={cache_seen}",
        f"cache_checked={cache_checked}",
        f"cache_ok={cache_ok}",
        f"npz_seen={len(npz_index)}",
        f"missing_cache={missing_cache}",
        f"bad={bad_count}",
        f"elapsed_s={_elapsed_s(start_time)}",
        flush=True,
    )
    if bad_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
