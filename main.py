import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build_cache", action="store_true")
    ap.add_argument("--check_cache", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--traj", type=str, default=None, help="cache directory name under GRAPH_ROOT")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--resume_ckpt", type=str, default=None)
    ap.add_argument("--src_frame", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--target_frame", type=int, default=None)
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--graph_root", type=str, default=None)
    ap.add_argument("--data_root", type=str, default=None)
    args = ap.parse_args()

    import config as cfg

    if args.graph_root is not None:
        cfg.GRAPH_ROOT = Path(args.graph_root)
    if args.data_root is not None:
        cfg.DATA_ROOT = Path(args.data_root)

    if args.build_cache:
        from preprocess.build_graph_cache import main as build_cache_main

        build_cache_main()

    if args.check_cache:
        from preprocess.check_graph_cache import main as check_cache_main

        check_cache_main()

    if args.train:
        from train import train

        train(resume_ckpt=args.resume_ckpt)

    if args.sample:
        if args.traj is None or args.ckpt is None:
            raise ValueError("sampling requires --traj and --ckpt")
        from sample import sample_fused

        src_frame = int(cfg.OBSERVED_FRAMES - 1 if args.src_frame is None else args.src_frame)
        horizon = int(cfg.FUTURE_FRAMES if args.horizon is None else args.horizon)
        if args.target_frame is not None:
            horizon = int(args.target_frame) - src_frame
            if horizon < 1:
                raise ValueError("--target_frame must be greater than --src_frame")
        sample_fused(
            args.traj,
            args.ckpt,
            src_frame=src_frame,
            horizon=horizon,
            n_samples=args.n_samples,
        )


if __name__ == "__main__":
    main()
