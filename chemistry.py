import re
import numpy as np

PT = {
    "H":  (1,  1.008, 1.20),
    "C":  (6,  12.011, 1.70),
    "N":  (7,  14.007, 1.55),
    "O":  (8,  15.999, 1.52),
    "F":  (9,  18.998, 1.47),
    "P":  (15, 30.974, 1.80),
    "S":  (16, 32.06,  1.80),
    "CL": (17, 35.45,  1.75),
    "BR": (35, 79.904, 1.85),
    "I":  (53, 126.90, 1.98),
    "NA": (11, 22.99,  2.27),
    "K":  (19, 39.10,  2.75),
    "CA": (20, 40.08,  2.31),
    "MG": (12, 24.31,  1.73),
    "ZN": (30, 65.38,  1.39),
    "FE": (26, 55.85,  1.56),
}

ATOMIC_NUMBER_TO_ELEMENT = {v[0]: k for k, v in PT.items()}

HYDROPHOBIC_RESNAMES = {
    "ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "TYR", "PRO", "CYS", "CYX",
}

def infer_element(atom_name: str) -> str:
    s = str(atom_name).strip().upper()
    s = re.sub(r"[^A-Z]", "", s)
    if s == "CA":
        # In protein structures "CA" is almost always alpha-carbon, not calcium.
        return "C"
    if len(s) >= 2 and s[:2] in {"CL", "BR", "NA", "MG", "ZN", "FE"}:
        return s[:2]
    if len(s) >= 1 and s[0] in PT:
        return s[0]
    if len(s) >= 2 and s[:2] in PT:
        return s[:2]
    return "C"

def atom_features_from_names(atom_names, elements=None):
    atom_names = np.asarray(atom_names, dtype=object)
    if elements is None:
        elements = np.array([infer_element(a) for a in atom_names], dtype=object)
    else:
        elements = np.asarray(elements, dtype=object)
        elements = np.array([str(e).strip().upper() for e in elements], dtype=object)

    Z = np.zeros(len(elements), dtype=np.float32)
    mass = np.zeros(len(elements), dtype=np.float32)
    radius = np.zeros(len(elements), dtype=np.float32)
    for i, e in enumerate(elements):
        z, m, r = PT.get(e, PT["C"])
        Z[i], mass[i], radius[i] = z, m, r

    # 归一化
    Zn = Z / 53.0
    mn = mass / 126.90
    rn = radius / 2.75
    return np.stack([Zn, mn, rn], axis=-1).astype(np.float32), elements


def atomic_number_to_element(z):
    try:
        return ATOMIC_NUMBER_TO_ELEMENT.get(int(z), "C")
    except (TypeError, ValueError):
        return "C"


def interaction_flags(atom_names, resnames, atomic_numbers=None, ligand_mask=None):
    """Return rough donor, acceptor, and hydrophobic masks.

    The public npz files do not include protonation states, bond orders, or
    force-field atom types, so these masks are intentionally soft heuristics.
    """
    atom_names = np.asarray(atom_names, dtype=object)
    resnames = np.asarray(resnames, dtype=object)
    n = int(atom_names.shape[0])
    if atomic_numbers is None:
        elements = np.array([infer_element(a) for a in atom_names], dtype=object)
    else:
        z = np.asarray(atomic_numbers).reshape(-1)
        if z.shape[0] == n:
            elements = np.array([atomic_number_to_element(x) for x in z], dtype=object)
        else:
            elements = np.array([infer_element(a) for a in atom_names], dtype=object)
    if ligand_mask is None:
        ligand_mask = np.zeros((n,), dtype=bool)
    else:
        ligand_mask = np.asarray(ligand_mask, dtype=bool).reshape(-1)
        if ligand_mask.shape[0] != n:
            ligand_mask = np.zeros((n,), dtype=bool)

    donors = np.zeros((n,), dtype=bool)
    acceptors = np.zeros((n,), dtype=bool)
    hydrophobic = np.zeros((n,), dtype=bool)
    for i in range(n):
        atom = str(atom_names[i]).strip().upper()
        res = str(resnames[i]).strip().upper()
        elem = str(elements[i]).strip().upper()

        is_heavy_polar = elem in {"N", "O", "S"}
        donors[i] = is_heavy_polar and atom not in {"O", "OXT"}
        acceptors[i] = is_heavy_polar and not atom.startswith("H")

        carbon_or_sulfur = elem in {"C", "S"}
        backbone_carbonyl = atom in {"C", "O", "OXT"}
        hydrophobic[i] = carbon_or_sulfur and (
            bool(ligand_mask[i]) or res in HYDROPHOBIC_RESNAMES
        ) and not backbone_carbonyl

    return donors.astype(np.bool_), acceptors.astype(np.bool_), hydrophobic.astype(np.bool_)
