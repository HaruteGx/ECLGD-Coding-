"""Frame-local residue membership; candidate identities never use future labels."""
import torch

from utils.md import minimum_image_displacement


@torch.no_grad()
def dynamic_residue_pocket_mask(coords, query_coords, node_mask, node_is_ligand,
                                node_residue_index, boxes=None, cutoff=1.0,
                                chunk_size=128):
    """Return (B,T,N) protein mask for whole residues near any ligand atom.

    Coordinates, boxes and cutoff use the same units. Queries are the current
    predicted/coarse ligand, never future labels in the generation branch.
    Stable full-protein identities allow residues absent at the source to enter.
    Membership is discrete conditioning, not a differentiable energy cutoff.
    """
    if node_residue_index is None:
        raise ValueError("dynamic pockets require node_residue_index; rebuild schema-4 cache")
    result = torch.zeros(coords.shape[:-1], dtype=torch.bool, device=coords.device)
    if float(cutoff) <= 0:
        raise ValueError("pocket cutoff must be positive")
    for b in range(coords.shape[0]):
        protein = node_mask[b].bool() & ~node_is_ligand[b].bool() & (node_residue_index[b] >= 0)
        prot = protein.nonzero(as_tuple=False).flatten()
        lig = (node_mask[b].bool() & node_is_ligand[b].bool()).nonzero(as_tuple=False).flatten()
        if not prot.numel() or not lig.numel():
            continue
        for t in range(coords.shape[1]):
            near = torch.zeros(prot.numel(), dtype=torch.bool, device=coords.device)
            for start in range(0, prot.numel(), chunk_size):
                ids = prot[start:start + chunk_size]
                delta = coords[b, t, ids, None] - query_coords[b, t, lig][None]
                if boxes is not None:
                    delta = minimum_image_displacement(delta, boxes[b, t])
                near[start:start + ids.numel()] = (delta.square().sum(-1) <= float(cutoff) ** 2).any(-1)
            residues = torch.unique(node_residue_index[b, prot[near]])
            # Avoid an N by N residue-membership matrix on a full protein.
            for residue in residues:
                result[b, t] |= protein & (node_residue_index[b] == residue)
    return result
