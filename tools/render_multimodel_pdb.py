import argparse
from pathlib import Path

from pymol import cmd


def main():
    parser = argparse.ArgumentParser(description="Render a multi-model PDB trajectory to PNG frames with PyMOL.")
    parser.add_argument("--pdb", required=True, help="Input multi-model PDB")
    parser.add_argument("--frames_dir", required=True, help="Directory for rendered PNG frames")
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--ray", type=int, default=0, help="Use ray tracing: 0 is faster, 1 is prettier")
    parser.add_argument("--protein_color", default="cyan")
    parser.add_argument("--ligand_color", default="orange")
    args = parser.parse_args()

    pdb_path = Path(args.pdb)
    frames_dir = Path(args.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    cmd.reinitialize()
    cmd.load(str(pdb_path), "traj")
    n_states = int(cmd.count_states("traj"))
    if n_states < 1:
        raise RuntimeError(f"no states found in {pdb_path}")

    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("opaque_background", 1)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)
    cmd.set("stick_radius", 0.16)
    cmd.set("sphere_scale", 0.22)
    cmd.hide("everything", "traj")
    cmd.show("cartoon", "traj and polymer")
    cmd.show("sticks", "traj and organic")
    cmd.color(args.protein_color, "traj and polymer")
    cmd.color(args.ligand_color, "traj and organic")
    cmd.orient("traj", animate=0)
    view = cmd.get_view()

    for state in range(1, n_states + 1):
        cmd.delete("frame_obj")
        cmd.create("frame_obj", "traj", state, 1)
        cmd.hide("everything", "frame_obj")
        cmd.show("cartoon", "frame_obj and polymer")
        cmd.show("sticks", "frame_obj and organic")
        cmd.color(args.protein_color, "frame_obj and polymer")
        cmd.color(args.ligand_color, "frame_obj and organic")
        cmd.disable("traj")
        cmd.enable("frame_obj")
        cmd.set_view(view)
        out_png = frames_dir / f"frame_{state:04d}.png"
        cmd.png(str(out_png), width=int(args.width), height=int(args.height), ray=int(args.ray), quiet=1)

    print(f"rendered {n_states} frames to {frames_dir}")


if __name__ == "__main__":
    main()
