from pathlib import Path
import hashlib
import os
import shutil
import time
import numpy as np
import torch

from config import (DATA_ROOT, GRAPH_ROOT, LIGAND_NEAR_RES_CUTOFF, COORD_SCALE, EDGE_RADIUS_A,
                    CACHED_FRAME_COUNT,
                    CLOUD_BOX_PADDING_A, CLOUD_GRID_SIZE, CLOUD_GAUSS_SIGMA_A,
                    CLOUD_MAX_VOXEL_A, CLOUD_MIN_VOXEL_A, USE_NPZ_CHARGES_IF_AVAILABLE,
                    PACKED_FRAMES_NAME, CLOUD_CHANNEL_NAMES,
                    GRAPH_CACHE_ATOM_CHUNK, GRAPH_CACHE_CLOUD_DEVICE,
                    GRAPH_CACHE_SAVE_FRAME_FILES, GRAPH_CACHE_SHARD_COUNT,
                    GRAPH_CACHE_SKIP_EXISTING,
                    GRAPH_CACHE_SCHEMA_VERSION,
                    GRAPH_CACHE_SHARD_INDEX, GRAPH_CACHE_UNWRAP_PBC,
                    GRAPH_CACHE_VOXEL_CHUNK)
from chemistry import atom_features_from_names, interaction_flags
from utils.ecloud import cloud_frame_from_atoms, protein_ligand_cloud_grid
from utils.md import box_to_matrix, minimum_image_displacement, unwrap_trajectory
from utils.pocket import dynamic_residue_pocket_mask

RES3_TO_RES1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V"
}


def _torch_load_cache(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _torch_save_cache(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    retries = max(0, _env_int_local("GRAPH_CACHE_SAVE_RETRIES", 2))
    retry_sleep = float(os.environ.get("GRAPH_CACHE_SAVE_RETRY_SLEEP", "5.0"))
    fsync_enabled = _env_bool_local("GRAPH_CACHE_SAVE_FSYNC", False)
    last_exc = None
    for attempt in range(retries + 1):
        tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}.{attempt}")
        try:
            with open(tmp_path, "wb") as f:
                torch.save(obj, f)
                f.flush()
                if fsync_enabled:
                    os.fsync(f.fileno())
            os.replace(tmp_path, path)
            if fsync_enabled:
                try:
                    dir_fd = os.open(str(path.parent), getattr(os, "O_DIRECTORY", 0))
                    try:
                        os.fsync(dir_fd)
                    finally:
                        os.close(dir_fd)
                except OSError:
                    pass
            return
        except Exception as exc:
            last_exc = exc
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as cleanup_exc:
                print(
                    f"cache_save_tmp_cleanup_failed path={tmp_path} "
                    f"error={type(cleanup_exc).__name__}: {cleanup_exc}",
                    flush=True,
                )
            if attempt < retries:
                print(
                    f"cache_save_retry path={path} attempt={attempt + 1}/{retries} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                time.sleep(retry_sleep)
                continue
            print(
                f"cache_save_failed path={path} attempts={retries + 1} "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
    if last_exc is not None:
        raise last_exc


def _env_bool_local(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int_local(name, default=0):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _elapsed_s(start_time):
    return f"{time.monotonic() - float(start_time):.2f}"


def _ligand_image_shift(coords0, is_ligand, protein_mask, box0):
    """Return one lattice shift that places the ligand beside its nearest protein image."""
    if box0 is None or not bool(np.asarray(is_ligand, dtype=bool).any()):
        return np.zeros((3,), dtype=np.float32)
    ligand_idx = np.flatnonzero(np.asarray(is_ligand, dtype=bool))
    protein_idx = np.flatnonzero(np.asarray(protein_mask, dtype=bool))
    if ligand_idx.size == 0 or protein_idx.size == 0:
        return np.zeros((3,), dtype=np.float32)
    matrix = box_to_matrix(torch.as_tensor(box0), coord_scale=1.0)
    ligand = torch.from_numpy(np.asarray(coords0[ligand_idx], dtype=np.float32))
    protein = torch.from_numpy(np.asarray(coords0[protein_idx], dtype=np.float32))
    raw = ligand[:, None] - protein[None]
    wrapped = minimum_image_displacement(raw, matrix)
    pair = int(torch.linalg.norm(wrapped, dim=-1).argmin().item())
    p = pair // int(protein.shape[0])
    q = pair % int(protein.shape[0])
    return (wrapped[p, q] - raw[p, q]).cpu().numpy().astype(np.float32)

def select_atoms_by_residue_around_ligand(coords0, resnames, resids, segids, ligand_resnames, cutoff_A):
    is_lig = np.array([r in ligand_resnames for r in resnames], dtype=bool)
    lig_xyz = coords0[is_lig]
    if lig_xyz.shape[0] == 0:
        raise RuntimeError("未找到配体原子：ligand_resnames 可能不匹配")

    res2idx = {}
    for i in range(len(resids)):
        key = (str(segids[i]), int(resids[i]), str(resnames[i]))
        res2idx.setdefault(key, []).append(i)

    sel_res = []
    for rkey, idxs in res2idx.items():
        xyz = coords0[np.array(idxs, dtype=int)]
        d = np.linalg.norm(xyz[:, None, :] - lig_xyz[None, :, :], axis=-1).min()
        if d <= cutoff_A:
            sel_res.append(rkey)

    sel = set(np.where(is_lig)[0].tolist())
    for rkey in sel_res:
        for i in res2idx[rkey]:
            sel.add(int(i))
    return np.array(sorted(sel), dtype=int)

def filter_bonds_to_selected(bonds, sel_global_to_local):
    keep = []
    for a, b in bonds:
        a = int(a); b = int(b)
        if a in sel_global_to_local and b in sel_global_to_local:
            keep.append([sel_global_to_local[a], sel_global_to_local[b]])
    if not keep:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(keep, dtype=np.int64)

def build_radius_edges(xyz_A, radius_A):
    diff = xyz_A[:, None, :] - xyz_A[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    adj = (dist < radius_A) & (dist > 1e-6)
    src, dst = np.where(adj)
    return np.stack([src, dst], axis=0).astype(np.int64)

def build_neighbors(edge_index, M):
    neigh = [[] for _ in range(M)]
    for a, b in edge_index.T:
        neigh[int(a)].append(int(b))
    return neigh

def pick_k_l_for_edges(edge_index, neighbors):
    src = edge_index[0]
    dst = edge_index[1]
    E = src.shape[0]
    k_idx = np.full((E,), -1, dtype=np.int64)
    l_idx = np.full((E,), -1, dtype=np.int64)
    for e in range(E):
        i = int(src[e]); j = int(dst[e])
        ni = [x for x in neighbors[i] if x != j]
        nj = [x for x in neighbors[j] if x != i]
        if len(ni) > 0: k_idx[e] = ni[0]
        if len(nj) > 0: l_idx[e] = nj[0]
    return k_idx, l_idx

def build_protein_res_keys(resnames, resids, segids, ligand_resnames):
    is_lig = np.array([r in ligand_resnames for r in resnames], dtype=bool)
    is_protein_res = np.array([r in RES3_TO_RES1 for r in resnames], dtype=bool)
    is_protein = is_protein_res & (~is_lig)
    order = np.lexsort((resids.astype(int), np.array(segids, dtype=str)))

    keys = []
    seen = set()
    for i in order:
        if not is_protein[i]:
            continue
        key = (str(segids[i]), int(resids[i]), str(resnames[i]))
        if key in seen: 
            continue
        seen.add(key)
        keys.append(key)
    if not keys:
        raise RuntimeError("未识别到蛋白残基（resnames 非标准三字母？）")
    return keys


def build_residue_to_indices(resnames, resids, segids):
    res2idx = {}
    for i in range(len(resids)):
        key = (str(segids[i]), int(resids[i]), str(resnames[i]))
        res2idx.setdefault(key, []).append(i)
    return res2idx


def residue_has_backbone(atom_names, indices):
    names = {str(atom_names[i]).strip().upper() for i in indices}
    return {"N", "CA", "C"}.issubset(names)


def select_protein_ligand_candidate_atoms(atom_names, resnames, resids, segids, ligand_resnames):
    """Topology-only full protein reservoir; no trajectory coordinates accepted."""
    is_lig = np.array([r in ligand_resnames for r in resnames], dtype=bool)
    if not is_lig.any():
        raise RuntimeError("no ligand atoms found for ligand_resnames")

    res2idx = build_residue_to_indices(resnames, resids, segids)
    sel = set(np.where(is_lig)[0].tolist())
    for rkey, idxs in res2idx.items():
        if str(rkey[2]) in ligand_resnames:
            continue
        if not residue_has_backbone(atom_names, idxs):
            continue
        sel.update(int(i) for i in idxs)
    return np.array(sorted(sel), dtype=int)


def build_all_backbone_res_keys(atom_names, resnames, resids, segids, ligand_resnames):
    is_lig = np.array([r in ligand_resnames for r in resnames], dtype=bool)
    res2idx = build_residue_to_indices(resnames, resids, segids)
    order = np.lexsort((resids.astype(int), np.array(segids, dtype=str)))
    keys = []
    seen = set()
    for i in order:
        if is_lig[i]:
            continue
        key = (str(segids[i]), int(resids[i]), str(resnames[i]))
        if key in seen:
            continue
        if not residue_has_backbone(atom_names, res2idx.get(key, [])):
            continue
        seen.add(key)
        keys.append(key)
    if not keys:
        raise RuntimeError("no protein-like residues with N/CA/C backbone were found")
    return keys

def build_res2atommap(atom_names, resnames, resids, segids):
    # reskey -> { "N": idx, "CA": idx, "C": idx }
    mp = {}
    for i in range(len(resids)):
        key = (str(segids[i]), int(resids[i]), str(resnames[i]))
        name = str(atom_names[i]).strip().upper()
        if key not in mp:
            mp[key] = {}
        if name in ("N", "CA", "C"):
            mp[key][name] = i
    return mp

def extract_protein_backbone(coords_A, protein_res_keys, res2bb):
    """
    coords_A:(N,3) Å
    return protein_bb:(L,3,3) Å, protein_mask:(L,)
    """
    L = len(protein_res_keys)
    bb = np.zeros((L, 3, 3), dtype=np.float32)
    mask = np.zeros((L,), dtype=np.float32)
    for i, k in enumerate(protein_res_keys):
        d = res2bb.get(k, {})
        if "N" in d and "CA" in d and "C" in d:
            bb[i, 0] = coords_A[d["N"]]
            bb[i, 1] = coords_A[d["CA"]]
            bb[i, 2] = coords_A[d["C"]]
            mask[i] = 1.0
    return bb, mask

def process_one(npz_path: Path, out_dir: Path = None):
    npz_path = Path(npz_path)
    npz_stat = npz_path.stat()
    data = np.load(npz_path, allow_pickle=True)
    coords = data["coords"].astype(np.float32)  # (T,N,3) Å
    charges = None
    if USE_NPZ_CHARGES_IF_AVAILABLE and ("charges" in data.files):
        charges = data["charges"].astype(np.float32)  # (N,)
    resnames = data["resnames"]
    atom_names = data["atom_names"]
    resids = data["resids"]
    segids = data["segids"]
    ligand_resnames = set(data["ligand_resnames"].tolist())

    T, N, _ = coords.shape
    times = data["times"].astype(np.float32) if "times" in data.files else np.arange(T, dtype=np.float32)
    box = data["box"].astype(np.float32) if "box" in data.files else None
    temporal_box = box is not None and box.ndim >= 2 and int(box.shape[0]) == int(T) and tuple(box.shape) != (3, 3)
    bonds = data["bonds"] if "bonds" in data.files else None
    elements = data["elements"] if "elements" in data.files else None
    atomic_numbers = data["atomic_numbers"].astype(np.int64) if "atomic_numbers" in data.files else None
    masses = data["masses"].astype(np.float32) if "masses" in data.files else None
    protein_res_keys = build_all_backbone_res_keys(atom_names, resnames, resids, segids, ligand_resnames)
    protein_key_set = set(protein_res_keys)
    protein_atom_mask_global = np.array(
        [
            (str(segids[i]), int(resids[i]), str(resnames[i])) in protein_key_set
            for i in range(len(resnames))
        ],
        dtype=bool,
    )
    coordinates_unwrapped = bool(GRAPH_CACHE_UNWRAP_PBC) and box is not None
    if coordinates_unwrapped:
        coords = unwrap_trajectory(
            torch.from_numpy(coords),
            torch.from_numpy(box),
            bonds=None if bonds is None else torch.as_tensor(bonds),
        ).cpu().numpy().astype(np.float32)
        ligand_mask_global = np.array([r in ligand_resnames for r in resnames], dtype=bool)
        initial_box = box[0] if temporal_box else box
        image_shift = _ligand_image_shift(coords[0], ligand_mask_global, protein_atom_mask_global, initial_box)
        coords[:, ligand_mask_global] += image_shift[None, None, :]
    coords0 = coords[0]

    sel_atoms = select_protein_ligand_candidate_atoms(
        atom_names,
        resnames,
        resids,
        segids,
        ligand_resnames,
    )
    sel_global_to_local = {int(g): int(i) for i, g in enumerate(sel_atoms)}
    M = len(sel_atoms)

    node_feat3, _node_elements = atom_features_from_names(atom_names[sel_atoms], None if elements is None else elements[sel_atoms])
    is_lig_global = np.array([r in ligand_resnames for r in resnames], dtype=bool)
    is_lig_local = np.array([bool(is_lig_global[g]) for g in sel_atoms], dtype=np.float32)[:, None]
    q = np.zeros((coords.shape[1],), dtype=np.float32) if charges is None else charges
    q_local = q[sel_atoms].astype(np.float32)[:, None]
    node_feat = np.concatenate([node_feat3, is_lig_local, q_local], axis=-1).astype(np.float32)  # (M,5)
    node_atomic_numbers = (
        atomic_numbers[sel_atoms].astype(np.int64)
        if atomic_numbers is not None
        else np.rint(node_feat3[:, 0] * 53.0).astype(np.int64)
    )
    node_masses = (
        masses[sel_atoms].astype(np.float32)
        if masses is not None
        else (node_feat3[:, 1] * 126.90).astype(np.float32)
    )
    node_vdw_radius_A = (node_feat3[:, 2] * 2.75).astype(np.float32)
    hbond_donor, hbond_acceptor, hydrophobic = interaction_flags(
        atom_names[sel_atoms],
        resnames[sel_atoms],
        atomic_numbers=node_atomic_numbers,
        ligand_mask=is_lig_local.squeeze(-1).astype(np.bool_),
    )
    selected_bonds_und = (
        filter_bonds_to_selected(bonds, sel_global_to_local)
        if bonds is not None
        else np.zeros((0, 2), dtype=np.int64)
    )

    protein_res_index = {tuple(k): int(i) for i, k in enumerate(protein_res_keys)}

    pocket_res_keys = []
    seen2 = set()
    node_residue_index = np.full((M,), -1, dtype=np.int64)
    node_backbone_slot = np.full((M,), -1, dtype=np.int64)
    for local_i, g in enumerate(sel_atoms):
        key = (str(segids[g]), int(resids[g]), str(resnames[g]))
        if not bool(is_lig_global[g]):
            node_residue_index[local_i] = protein_res_index.get(key, -1)
            atom_name = str(atom_names[g]).strip().upper()
            node_backbone_slot[local_i] = {"N": 0, "CA": 1, "C": 2}.get(atom_name, -1)
        if (not bool(is_lig_global[g])) and key not in seen2:
            seen2.add(key)
            pocket_res_keys.append(key)
    pocket_residue_indices = np.array([protein_res_index[k] for k in pocket_res_keys], dtype=np.int64)

    # Stable protein reservoir; active pocket membership is frame-local.
    pocket_set = set(pocket_res_keys)
    is_lig_global = np.array([r in ligand_resnames for r in resnames], dtype=bool)
    atom_res_keys = [(str(segids[i]), int(resids[i]), str(resnames[i])) for i in range(len(resnames))]
    anchor_global = np.array([ (atom_res_keys[i] in pocket_set) and (not bool(is_lig_global[i])) for i in range(len(atom_res_keys)) ], dtype=bool)
    anchor_global_indices = np.where(anchor_global)[0].astype(np.int64)

    out_dir = Path(out_dir) if out_dir is not None else GRAPH_ROOT / npz_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in out_dir.glob("frame_*.pt"):
        old_frame.unlink()
    packed_path = out_dir / PACKED_FRAMES_NAME
    if packed_path.exists():
        packed_path.unlink()

    meta = {
        "cache_schema_version": int(GRAPH_CACHE_SCHEMA_VERSION),
        "source_npz": str(npz_path),
        "source_npz_name": npz_path.name,
        "source_npz_stem": npz_path.stem,
        "source_npz_size": int(npz_stat.st_size),
        "source_npz_mtime_ns": int(npz_stat.st_mtime_ns),
        "sel_global_indices": torch.as_tensor(sel_atoms, dtype=torch.long),
        "node_global_indices": torch.as_tensor(sel_atoms, dtype=torch.long),
        "anchor_global_indices": torch.as_tensor(anchor_global_indices, dtype=torch.long),
        "node_feat": torch.from_numpy(node_feat),
        "node_atomic_numbers": torch.from_numpy(node_atomic_numbers),
        "node_masses": torch.from_numpy(node_masses),
        "node_vdw_radius_A": torch.from_numpy(node_vdw_radius_A),
        "node_charge": torch.from_numpy(q_local.squeeze(-1).astype(np.float32)),
        "hbond_donor_mask": torch.as_tensor(hbond_donor, dtype=torch.bool),
        "hbond_acceptor_mask": torch.as_tensor(hbond_acceptor, dtype=torch.bool),
        "hydrophobic_mask": torch.as_tensor(hydrophobic, dtype=torch.bool),
        "node_is_ligand": torch.as_tensor(is_lig_local.squeeze(-1).astype(np.bool_), dtype=torch.bool),
        "node_residue_index": torch.from_numpy(node_residue_index),
        "node_backbone_slot": torch.from_numpy(node_backbone_slot),
        "pocket_residue_indices": torch.from_numpy(pocket_residue_indices),
        "bond_edge_index": torch.from_numpy(selected_bonds_und.T.astype(np.int64)),
        "segids": [str(segids[g]) for g in sel_atoms],
        "resids": [int(resids[g]) for g in sel_atoms],
        "resnames": [str(resnames[g]) for g in sel_atoms],
        "atom_names": [str(atom_names[g]) for g in sel_atoms],
        "ligand_resnames": [str(x) for x in ligand_resnames],
        "protein_res_keys": protein_res_keys,
        "pocket_res_keys": pocket_res_keys,
        "cloud_channels": list(CLOUD_CHANNEL_NAMES),
        "cloud_cutoff_A": float(LIGAND_NEAR_RES_CUTOFF),
        "cloud_grid_size": int(CLOUD_GRID_SIZE),
        "cloud_box_padding_A": float(CLOUD_BOX_PADDING_A),
        "cloud_min_voxel_A": float(CLOUD_MIN_VOXEL_A),
        "cloud_max_voxel_A": float(CLOUD_MAX_VOXEL_A),
        "cloud_sigma_A": float(CLOUD_GAUSS_SIGMA_A),
        "pocket_selection_mode": "full_protein_dynamic_residue",
        "pbc_unwrap_requested": bool(GRAPH_CACHE_UNWRAP_PBC),
        "coordinates_unwrapped": bool(coordinates_unwrapped),
    }
    _torch_save_cache(meta, out_dir / "meta.pt")

    # 用于抽 backbone
    res2bb = build_res2atommap(atom_names, resnames, resids, segids)

    packed_frames = []
    packed_frame_ids = []
    for t in range(T):
        coords_A = coords[t]  # (N,3) Å

        # 全蛋白 backbone 真值（按 protein_res_keys 对齐）
        bb_A, bb_mask = extract_protein_backbone(coords_A, protein_res_keys, res2bb)
        bb_norm = (bb_A / COORD_SCALE).astype(np.float32)

        # 口袋图
        xyz_A = coords_A[sel_atoms]
        frame_box = None if box is None else (box[t] if temporal_box else box)
        matrix_A = None if frame_box is None else box_to_matrix(frame_box)
        local_coords = torch.from_numpy(xyz_A)[None, None]
        active_pocket = dynamic_residue_pocket_mask(
            local_coords, local_coords, torch.ones((1, M), dtype=torch.bool),
            torch.from_numpy(is_lig_local[:, 0].astype(bool))[None],
            torch.from_numpy(node_residue_index)[None],
            boxes=None if matrix_A is None else matrix_A[None, None],
            cutoff=float(LIGAND_NEAR_RES_CUTOFF),
        )[0, 0].numpy()
        cloud_mask = active_pocket | is_lig_local[:, 0].astype(bool)
        cloud_xyz = xyz_A[cloud_mask].copy()
        if matrix_A is not None:
            ligand_center = xyz_A[is_lig_local[:, 0].astype(bool)].mean(0)
            cloud_xyz = (minimum_image_displacement(
                torch.from_numpy(cloud_xyz - ligand_center), matrix_A
            ).numpy() + ligand_center)
        center_A, cloud_voxel_A, cloud_half_width_A = cloud_frame_from_atoms(
            cloud_xyz,
            grid_size=CLOUD_GRID_SIZE,
            padding_A=CLOUD_BOX_PADDING_A,
            min_voxel_A=CLOUD_MIN_VOXEL_A,
            max_voxel_A=CLOUD_MAX_VOXEL_A,
        )
        cloud_weights = node_feat[:, 0] * 53.0
        cloud_grid = protein_ligand_cloud_grid(
            cloud_xyz,
            cloud_weights[cloud_mask],
            is_lig_local.squeeze(-1).astype(np.bool_)[cloud_mask],
            center_A,
            grid_size=CLOUD_GRID_SIZE,
            voxel_A=cloud_voxel_A,
            sigma_A=CLOUD_GAUSS_SIGMA_A,
            device=GRAPH_CACHE_CLOUD_DEVICE,
            voxel_chunk=GRAPH_CACHE_VOXEL_CHUNK,
            atom_chunk=GRAPH_CACHE_ATOM_CHUNK,
        )

        if bonds is not None:
            und = selected_bonds_und
            if und.shape[0] == 0:
                edge_index = build_radius_edges(xyz_A, EDGE_RADIUS_A)
            else:
                edge_index = np.concatenate([und, und[:, ::-1]], axis=0).T.astype(np.int64)
        else:
            edge_index = build_radius_edges(xyz_A, EDGE_RADIUS_A)

        neighbors = build_neighbors(edge_index, M)
        k_idx, l_idx = pick_k_l_for_edges(edge_index, neighbors)

        frame = {
            "t": int(t),
            "time": torch.tensor(float(times[t]), dtype=torch.float32),
            "coords": torch.from_numpy((xyz_A / COORD_SCALE).astype(np.float32)),  # pocket coords (M,3) norm
            "active_pocket_mask": torch.from_numpy(active_pocket),
            "edge_index": torch.from_numpy(edge_index.astype(np.int64)),
            "k_idx": torch.from_numpy(k_idx),
            "l_idx": torch.from_numpy(l_idx),
            "cloud_grid": cloud_grid,  # channels follow meta["cloud_channels"]
            "cloud_center": torch.from_numpy((center_A / COORD_SCALE).astype(np.float32)),
            "cloud_voxel": torch.tensor(float(cloud_voxel_A / COORD_SCALE), dtype=torch.float32),
            "cloud_half_width": torch.tensor(float(cloud_half_width_A / COORD_SCALE), dtype=torch.float32),
            "protein_bb": torch.from_numpy(bb_norm),          # (L,3,3) norm
            "protein_bb_mask": torch.from_numpy(bb_mask),     # (L,)
        }
        if box is not None:
            frame_box = box[t] if temporal_box else box
            frame["box"] = torch.from_numpy(np.asarray(frame_box, dtype=np.float32))
        if bool(GRAPH_CACHE_SAVE_FRAME_FILES):
            _torch_save_cache(frame, out_dir / f"frame_{t:04d}.pt")
        packed_frames.append(frame)
        packed_frame_ids.append(int(t))

    _torch_save_cache(
        {
            "version": int(GRAPH_CACHE_SCHEMA_VERSION),
            "frame_ids": packed_frame_ids,
            "frames": packed_frames,
        },
        packed_path,
    )


def _remove_tree(path: Path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)


def _cleanup_cache_artifacts(stem: str, include_final: bool = False):
    final_dir = GRAPH_ROOT / stem
    patterns = (
        f".{stem}.tmp.*",
        f".{stem}.replace_backup.*",
    )
    for pattern in patterns:
        for path in GRAPH_ROOT.glob(pattern):
            if path.is_dir():
                print(f"cleanup_stale_cache_artifact: {path}", flush=True)
                _remove_tree(path)
    if include_final and final_dir.exists():
        print(f"delete_old_cache_before_rebuild: {final_dir}", flush=True)
        _remove_tree(final_dir)


def _replace_cache_dir(tmp_dir: Path, final_dir: Path):
    tmp_dir = Path(tmp_dir)
    final_dir = Path(final_dir)
    backup_dir = final_dir.with_name(f".{final_dir.name}.replace_backup.{os.getpid()}")
    _remove_tree(backup_dir)
    try:
        if final_dir.exists():
            final_dir.rename(backup_dir)
        tmp_dir.rename(final_dir)
    except Exception:
        if not final_dir.exists() and backup_dir.exists():
            backup_dir.rename(final_dir)
        raise
    finally:
        _remove_tree(backup_dir)


def build_or_replace_one(npz_path: Path):
    start_time = time.monotonic()
    npz_path = Path(npz_path)
    final_dir = GRAPH_ROOT / npz_path.stem
    tmp_dir = GRAPH_ROOT / f".{npz_path.stem}.tmp.{os.getpid()}"
    delete_old_first = _env_bool_local("GRAPH_CACHE_DELETE_OLD_BEFORE_REBUILD", False)
    _cleanup_cache_artifacts(npz_path.stem, include_final=delete_old_first)
    _remove_tree(tmp_dir)
    try:
        process_one(npz_path, out_dir=tmp_dir)
        validate_tmp_packed = _env_bool_local("GRAPH_CACHE_VALIDATE_TMP_PACKED_LOAD", False)
        if not _cache_dir_is_complete(tmp_dir, npz_path=npz_path, validate_packed_load=validate_tmp_packed):
            raise RuntimeError(f"new graph cache failed schema validation: {tmp_dir}")
        _replace_cache_dir(tmp_dir, final_dir)
        print(f"build_replace_done elapsed_s={_elapsed_s(start_time)} path={npz_path}", flush=True)
    finally:
        _remove_tree(tmp_dir)


def _meta_matches_current_build(meta, out_dir: Path, npz_path: Path = None):
    if int(meta.get("cache_schema_version", 0)) < int(GRAPH_CACHE_SCHEMA_VERSION):
        print(
            f"cache_check_incomplete path={out_dir} reason=old_md_hierarchy_schema "
            f"cached={meta.get('cache_schema_version', None)} expected={int(GRAPH_CACHE_SCHEMA_VERSION)}",
            flush=True,
        )
        return False
    if str(meta.get("pocket_selection_mode", "")) != "full_protein_dynamic_residue":
        print(f"cache_check_incomplete path={out_dir} reason=pocket_selection_mode_mismatch", flush=True)
        return False
    expected_unwrapped = bool(GRAPH_CACHE_UNWRAP_PBC)
    if bool(meta.get("pbc_unwrap_requested", False)) != expected_unwrapped:
        print(
            f"cache_check_incomplete path={out_dir} reason=pbc_unwrap_mismatch "
            f"cached={meta.get('pbc_unwrap_requested', None)} expected={expected_unwrapped}",
            flush=True,
        )
        return False
    if int(meta.get("cloud_grid_size", -1)) != int(CLOUD_GRID_SIZE):
        print(
            f"cache_check_incomplete path={out_dir} reason=cloud_grid_mismatch "
            f"cached={meta.get('cloud_grid_size', None)} expected={int(CLOUD_GRID_SIZE)}",
            flush=True,
        )
        return False
    cached_channels = tuple(str(x) for x in meta.get("cloud_channels", ()))
    expected_channels = tuple(str(x) for x in CLOUD_CHANNEL_NAMES)
    if cached_channels != expected_channels:
        print(
            f"cache_check_incomplete path={out_dir} reason=cloud_channels_mismatch "
            f"cached={cached_channels} expected={expected_channels}",
            flush=True,
        )
        return False
    cloud_params = (
        ("cloud_box_padding_A", CLOUD_BOX_PADDING_A),
        ("cloud_min_voxel_A", CLOUD_MIN_VOXEL_A),
        ("cloud_max_voxel_A", CLOUD_MAX_VOXEL_A),
        ("cloud_sigma_A", CLOUD_GAUSS_SIGMA_A),
        ("cloud_cutoff_A", LIGAND_NEAR_RES_CUTOFF),
    )
    for key, expected in cloud_params:
        try:
            cached = float(meta.get(key))
        except (TypeError, ValueError):
            print(f"cache_check_incomplete path={out_dir} reason=missing_cloud_param key={key}", flush=True)
            return False
        if abs(cached - float(expected)) > 1e-6:
            print(
                f"cache_check_incomplete path={out_dir} reason=cloud_param_mismatch "
                f"key={key} cached={cached} expected={float(expected)}",
                flush=True,
            )
            return False
    if npz_path is not None:
        npz_path = Path(npz_path)
        is_tmp_cache = out_dir.name.startswith(f".{npz_path.stem}.tmp.")
        if not is_tmp_cache and out_dir.name != npz_path.stem:
            print(
                f"cache_check_incomplete path={out_dir} reason=cache_npz_stem_mismatch "
                f"cache_stem={out_dir.name} npz_stem={npz_path.stem}",
                flush=True,
            )
            return False
        source_name = str(meta.get("source_npz_name") or Path(str(meta.get("source_npz", ""))).name)
        if source_name != npz_path.name:
            print(
                f"cache_check_incomplete path={out_dir} reason=source_npz_name_mismatch "
                f"cached={source_name} expected={npz_path.name}",
                flush=True,
            )
            return False
        try:
            stat = npz_path.stat()
        except FileNotFoundError:
            print(f"cache_check_incomplete path={out_dir} reason=source_npz_missing npz={npz_path}", flush=True)
            return False
        if int(meta.get("source_npz_size", -1)) != int(stat.st_size):
            print(
                f"cache_check_incomplete path={out_dir} reason=source_npz_size_mismatch "
                f"cached={meta.get('source_npz_size', None)} expected={int(stat.st_size)}",
                flush=True,
            )
            return False
        if int(meta.get("source_npz_mtime_ns", -1)) != int(stat.st_mtime_ns):
            print(
                f"cache_check_incomplete path={out_dir} reason=source_npz_mtime_mismatch",
                flush=True,
            )
            return False
    return True


def _cache_dir_is_complete(out_dir: Path, npz_path: Path = None, validate_packed_load: bool = True):
    out_dir = Path(out_dir)
    meta_path = out_dir / "meta.pt"
    packed_path = out_dir / PACKED_FRAMES_NAME
    if not meta_path.exists() or not packed_path.exists():
        return False
    try:
        meta = _torch_load_cache(meta_path)
    except Exception as exc:
        print(f"cache_check_incomplete path={out_dir} reason=meta_load_error error={type(exc).__name__}: {exc}", flush=True)
        return False
    required_meta = (
        "node_atomic_numbers",
        "node_masses",
        "node_vdw_radius_A",
        "node_charge",
        "hbond_donor_mask",
        "hbond_acceptor_mask",
        "hydrophobic_mask",
        "node_backbone_slot",
        "bond_edge_index",
    )
    if not isinstance(meta, dict) or int(meta.get("cache_schema_version", 0)) < int(GRAPH_CACHE_SCHEMA_VERSION):
        print(f"cache_check_incomplete path={out_dir} reason=old_or_missing_schema", flush=True)
        return False
    if any(key not in meta for key in required_meta):
        missing = [key for key in required_meta if key not in meta]
        print(f"cache_check_incomplete path={out_dir} reason=missing_meta keys={missing}", flush=True)
        return False
    if not _meta_matches_current_build(meta, out_dir, npz_path=npz_path):
        return False
    if not validate_packed_load:
        packed_size = packed_path.stat().st_size if packed_path.exists() else 0
        if packed_size <= 0:
            print(f"cache_check_incomplete path={out_dir} reason=empty_packed_file", flush=True)
            return False
        return True
    try:
        retries = max(0, _env_int_local("GRAPH_CACHE_VALIDATE_LOAD_RETRIES", 3))
        last_exc = None
        for attempt in range(retries + 1):
            try:
                packed = _torch_load_cache(packed_path)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if attempt < retries:
                    time.sleep(float(os.environ.get("GRAPH_CACHE_VALIDATE_LOAD_SLEEP", "2.0")))
        if last_exc is not None:
            raise last_exc
    except Exception as exc:
        packed_size = packed_path.stat().st_size if packed_path.exists() else 0
        print(
            f"cache_check_incomplete path={out_dir} reason=frames_load_error "
            f"size={packed_size} error={type(exc).__name__}: {exc}",
            flush=True,
        )
        return False
    if not isinstance(packed, dict):
        print(f"cache_check_incomplete path={out_dir} reason=packed_not_dict", flush=True)
        return False
    if int(packed.get("version", 0)) < int(GRAPH_CACHE_SCHEMA_VERSION):
        print(f"cache_check_incomplete path={out_dir} reason=old_packed_version version={packed.get('version', None)}", flush=True)
        return False
    frame_ids = packed.get("frame_ids")
    frames = packed.get("frames")
    if frame_ids is None or frames is None:
        print(f"cache_check_incomplete path={out_dir} reason=missing_frames_or_ids", flush=True)
        return False
    if len(frame_ids) < int(CACHED_FRAME_COUNT) or len(frames) < int(CACHED_FRAME_COUNT):
        print(
            f"cache_check_incomplete path={out_dir} reason=too_few_frames "
            f"frame_ids={0 if frame_ids is None else len(frame_ids)} frames={0 if frames is None else len(frames)}",
            flush=True,
        )
        return False
    expected_ids = list(range(int(CACHED_FRAME_COUNT)))
    if [int(x) for x in frame_ids[: int(CACHED_FRAME_COUNT)]] != expected_ids:
        print(f"cache_check_incomplete path={out_dir} reason=unexpected_frame_ids", flush=True)
        return False
    if isinstance(frames, dict):
        first_frame = frames.get(frame_ids[0], frames.get(str(frame_ids[0])))
    elif isinstance(frames, (list, tuple)):
        first_frame = frames[0] if frames else None
    else:
        first_frame = None
    if not isinstance(first_frame, dict) or "time" not in first_frame:
        print(f"cache_check_incomplete path={out_dir} reason=missing_frame_time", flush=True)
        return False
    return True


def _cache_is_complete(npz_path: Path):
    return _cache_dir_is_complete(GRAPH_ROOT / npz_path.stem, npz_path=npz_path)


def _iter_npz_files(root: Path):
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if name.lower().endswith(".npz"):
                yield Path(dirpath) / name


def _iter_cache_dirs(root: Path):
    root = Path(root)
    if not root.exists():
        return
    for path in sorted(root.iterdir(), key=lambda p: p.name):
        if not path.is_dir() or path.name.startswith("."):
            continue
        yield path


def _stem_shard(stem: str, shard_count: int):
    if int(shard_count) <= 1:
        return 0
    digest = hashlib.blake2b(str(stem).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % int(shard_count)


def _find_npz_for_stem(stem: str):
    direct = Path(DATA_ROOT) / f"{stem}.npz"
    if direct.exists():
        return direct
    target = f"{stem}.npz"
    for path in _iter_npz_files(DATA_ROOT):
        if path.name == target:
            return path
    return None


def _record_build_failure(npz_path: Path, phase: str, exc: Exception, elapsed_s: str):
    message = (
        f"build_failed phase={phase} elapsed_s={elapsed_s} path={npz_path} "
        f"error={type(exc).__name__}: {exc}"
    )
    print(message, flush=True)
    failure_log = os.environ.get("GRAPH_CACHE_FAILURE_LOG", "").strip()
    if not failure_log:
        return
    try:
        log_path = Path(failure_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{phase}\t{elapsed_s}\t"
                f"{npz_path}\t{type(exc).__name__}\t{exc}\n"
            )
    except Exception as log_exc:
        print(
            f"build_failure_log_write_failed path={failure_log} "
            f"error={type(log_exc).__name__}: {log_exc}",
            flush=True,
        )


def _maybe_build_npz(npz_path: Path, counters: dict, phase: str):
    start_time = time.monotonic()
    if bool(GRAPH_CACHE_SKIP_EXISTING) and _cache_is_complete(npz_path):
        print(f"skip_current_v2 phase={phase} elapsed_s={_elapsed_s(start_time)} path={npz_path}", flush=True)
        counters["skipped"] += 1
        return
    try:
        build_or_replace_one(npz_path)
    except Exception as exc:
        counters["failed"] += 1
        _record_build_failure(npz_path, phase, exc, _elapsed_s(start_time))
        if _env_bool_local("GRAPH_CACHE_FAIL_FAST", False):
            raise
        return
    print(f"replaced phase={phase} elapsed_s={_elapsed_s(start_time)} path={npz_path}", flush=True)
    counters["replaced"] += 1


def main():
    start_time = time.monotonic()
    if not str(GRAPH_CACHE_CLOUD_DEVICE).startswith("cuda"):
        raise RuntimeError("graph cache builds are configured for GPU only; set GRAPH_CACHE_CLOUD_DEVICE=cuda or cuda:<id>")
    if not torch.cuda.is_available():
        raise RuntimeError("graph cache builds require CUDA, but torch.cuda.is_available() is false")
    GRAPH_ROOT.mkdir(parents=True, exist_ok=True)
    shard_count = max(1, int(GRAPH_CACHE_SHARD_COUNT))
    shard_index = int(GRAPH_CACHE_SHARD_INDEX)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"GRAPH_CACHE_SHARD_INDEX={shard_index} must be in [0, {shard_count})")
    print(
        "build_cache:",
        "streaming=1",
        "order=graph_cache_first",
        f"shard={shard_index}/{shard_count}",
        f"cloud_channels={','.join(CLOUD_CHANNEL_NAMES)}",
        f"cloud_device={GRAPH_CACHE_CLOUD_DEVICE}",
        f"grid={CLOUD_GRID_SIZE}",
        f"padding_A={CLOUD_BOX_PADDING_A}",
        f"voxel_A_range=[{CLOUD_MIN_VOXEL_A},{CLOUD_MAX_VOXEL_A}]",
        f"sigma_A={CLOUD_GAUSS_SIGMA_A}",
        f"skip_existing={int(bool(GRAPH_CACHE_SKIP_EXISTING))}",
    )
    counters = {
        "cache_seen": 0,
        "cache_shard_seen": 0,
        "npz_seen": 0,
        "npz_shard_seen": 0,
        "orphan_cache": 0,
        "skipped": 0,
        "replaced": 0,
        "failed": 0,
    }
    processed_stems = set()
    delete_orphans = _env_bool_local("GRAPH_CACHE_DELETE_ORPHAN_CACHE", False)

    for cache_dir in _iter_cache_dirs(GRAPH_ROOT):
        stem = cache_dir.name
        counters["cache_seen"] += 1
        if _stem_shard(stem, shard_count) != shard_index:
            continue
        counters["cache_shard_seen"] += 1
        npz_path = _find_npz_for_stem(stem)
        if npz_path is None:
            counters["orphan_cache"] += 1
            print(f"orphan_graph_cache_no_npz: {cache_dir}", flush=True)
            if delete_orphans:
                print(f"delete_orphan_graph_cache: {cache_dir}", flush=True)
                _remove_tree(cache_dir)
            continue
        processed_stems.add(stem)
        _maybe_build_npz(npz_path, counters, phase="existing_graph_cache")

    for npz_path in _iter_npz_files(DATA_ROOT):
        counters["npz_seen"] += 1
        stem = npz_path.stem
        if _stem_shard(stem, shard_count) != shard_index:
            continue
        counters["npz_shard_seen"] += 1
        if stem in processed_stems:
            continue
        _maybe_build_npz(npz_path, counters, phase="missing_graph_cache")

    if counters["npz_seen"] <= 0:
        raise RuntimeError(f"no npz found under {DATA_ROOT}")
    print(
        "build_cache_done:",
        f"cache_seen={counters['cache_seen']}",
        f"cache_shard_seen={counters['cache_shard_seen']}",
        f"npz_seen={counters['npz_seen']}",
        f"npz_shard_seen={counters['npz_shard_seen']}",
        f"orphan_cache={counters['orphan_cache']}",
        f"skipped={counters['skipped']}",
        f"replaced={counters['replaced']}",
        f"failed={counters['failed']}",
        f"elapsed_s={_elapsed_s(start_time)}",
        flush=True,
    )
    if counters["failed"] > 0 and _env_bool_local("GRAPH_CACHE_EXIT_NONZERO_ON_FAILED", True):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
