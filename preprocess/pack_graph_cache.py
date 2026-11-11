from pathlib import Path
import argparse

import torch

from config import GRAPH_ROOT, CACHED_FRAME_START, CACHED_FRAME_COUNT, PACKED_FRAMES_NAME


def _frame_path(traj_dir: Path, frame_id: int):
    return traj_dir / f"frame_{int(frame_id):04d}.pt"


def pack_traj_dir(traj_dir: Path, start: int, count: int, overwrite: bool = False):
    traj_dir = Path(traj_dir)
    out_path = traj_dir / PACKED_FRAMES_NAME
    if out_path.exists() and not overwrite:
        return "skipped"

    frame_ids = list(range(int(start), int(start) + int(count)))
    missing = [fid for fid in frame_ids if not _frame_path(traj_dir, fid).exists()]
    if missing:
        preview = ",".join(str(x) for x in missing[:5])
        suffix = "" if len(missing) <= 5 else f"...(+{len(missing) - 5})"
        raise FileNotFoundError(f"{traj_dir}: missing frame ids {preview}{suffix}")

    frames = [torch.load(_frame_path(traj_dir, fid), map_location="cpu") for fid in frame_ids]
    torch.save(
        {
            "version": 1,
            "frame_ids": frame_ids,
            "frames": frames,
        },
        out_path,
    )
    return "packed"


def main():
    parser = argparse.ArgumentParser(description="Pack per-frame graph cache files into one frames.pt per trajectory.")
    parser.add_argument("--graph-root", type=Path, default=GRAPH_ROOT)
    parser.add_argument("--start", type=int, default=CACHED_FRAME_START)
    parser.add_argument("--count", type=int, default=CACHED_FRAME_COUNT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    traj_dirs = sorted([p for p in Path(args.graph_root).iterdir() if p.is_dir()])
    if not traj_dirs:
        raise RuntimeError(f"no trajectory directories under {args.graph_root}")

    counts = {"packed": 0, "skipped": 0}
    for traj_dir in traj_dirs:
        status = pack_traj_dir(traj_dir, args.start, args.count, overwrite=args.overwrite)
        counts[status] += 1
        print(f"[{status}] {traj_dir}")

    print(f"done packed={counts['packed']} skipped={counts['skipped']}")


if __name__ == "__main__":
    main()
