#!/usr/bin/env python
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sample import postprocess_prediction_arrays, resolve_npz_path


def _as_str(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _chain_pair_mask_from_keys(protein_res_keys):
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


def _write_model_records(handle, coords_A, atom_names, resnames, resids, segids, ligand_resnames, model_id):
    handle.write(f"MODEL     {int(model_id):4d}\n")
    ligset = {_as_str(x) for x in ligand_resnames}
    serial = 1
    for idx in range(len(atom_names)):
        atom_name = _as_str(atom_names[idx])[:4].rjust(4)
        resname = _as_str(resnames[idx])[:3].rjust(3)
        chain = (_as_str(segids[idx])[:1] if _as_str(segids[idx]) else "A")
        resid = int(resids[idx])
        x, y, z = coords_A[idx]
        record = "HETATM" if _as_str(resnames[idx]) in ligset else "ATOM  "
        handle.write(
            f"{record}{serial:5d} {atom_name} {resname} {chain}"
            f"{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
            "  1.00  0.00           \n"
        )
        serial += 1
    handle.write("ENDMDL\n")


def main():
    parser = argparse.ArgumentParser(description="Convert sampled trajectory .pt to a multi-model PDB.")
    parser.add_argument("--pt", required=True, help="Sampled trajectory .pt file from 0runsample.sh")
    parser.add_argument("--graph_root", default="graph_cache", help="Directory containing graph_cache trajectory folders")
    parser.add_argument("--data_root", default="npz dataset", help="Directory containing source npz files")
    parser.add_argument("--out", default="", help="Output PDB path; defaults to <pt>.multimodel.pdb")
    args = parser.parse_args()

    pt_path = Path(args.pt)
    obj = torch.load(pt_path, map_location="cpu")
    traj_name = str(obj["traj_name"])
    src_frame = int(obj["src_frame"])
    pred_coords = obj["pred_coords"]
    pred_protein_bb = obj["pred_protein_bb"]
    if pred_coords.ndim != 3 or pred_protein_bb.ndim != 4:
        raise ValueError("Expected pred_coords=(T,N,3) and pred_protein_bb=(T,R,3,3)")
    if int(pred_coords.shape[0]) != int(pred_protein_bb.shape[0]):
        raise ValueError("pred_coords and pred_protein_bb have different frame counts")

    traj_dir = Path(args.graph_root) / traj_name
    meta = torch.load(traj_dir / "meta.pt", map_location="cpu")
    npz_path = resolve_npz_path(meta["source_npz"], args.data_root)
    data = np.load(npz_path, allow_pickle=True)

    protein_res_keys = meta["protein_res_keys"]
    batch = {
        "source_protein_bb_mask": torch.ones((1, len(protein_res_keys)), dtype=torch.bool),
        "protein_chain_pair_mask": _chain_pair_mask_from_keys(protein_res_keys).unsqueeze(0),
    }
    ligand_resnames = set(data["ligand_resnames"].tolist())
    out_path = Path(args.out) if args.out else pt_path.with_suffix(".multimodel.pdb")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as handle:
        for frame_idx in range(int(pred_coords.shape[0])):
            post = postprocess_prediction_arrays(
                src_frame=src_frame,
                data=data,
                meta=meta,
                pred_coords_norm=pred_coords[frame_idx],
                pred_protein_bb_norm=pred_protein_bb[frame_idx],
                model=None,
                batch=batch,
                ligand_sample_steps=0,
            )
            _write_model_records(
                handle,
                post["coords_out_A"],
                data["atom_names"],
                data["resnames"],
                data["resids"],
                data["segids"],
                ligand_resnames,
                model_id=frame_idx + 1,
            )
        handle.write("END\n")
    print(f"wrote multi-model pdb: {out_path}")


if __name__ == "__main__":
    main()
