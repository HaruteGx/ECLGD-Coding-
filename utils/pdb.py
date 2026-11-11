# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 16:01:39 2025

@author: HaruteGx
"""

def write_pdb(path, coords_A, atom_names, resnames, resids, segids, ligand_resnames=None):
    """
    coords_A: (N,3) in Å
    atom_names/resnames/segids: list/array-like length N
    resids: list/array-like length N (int)
    ligand_resnames: set/list, 用于写 HETATM
    """
    ligset = set(ligand_resnames) if ligand_resnames is not None else set()

    with open(path, "w") as f:
        serial = 1
        for i in range(len(atom_names)):
            name = str(atom_names[i])[:4].rjust(4)
            resn = str(resnames[i])[:3].rjust(3)
            chain = (str(segids[i])[:1] if str(segids[i]) else "A")
            resid = int(resids[i])
            x, y, z = coords_A[i]
            rec = "HETATM" if str(resnames[i]) in ligset else "ATOM  "
            line = (
                f"{rec}{serial:5d} {name} {resn} {chain}"
                f"{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00           \n"
            )
            f.write(line)
            serial += 1
        f.write("END\n")
