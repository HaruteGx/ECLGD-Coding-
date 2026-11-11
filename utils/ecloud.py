import numpy as np
import torch

def gaussian_cloud_grid(
    coords_A: np.ndarray,
    weights: np.ndarray,
    center_A: np.ndarray,
    grid_size: int = 32,
    voxel_A: float = 0.5,
    sigma_A: float = 0.8,
) -> torch.Tensor:
    """
    将(原子坐标, 权重)近似成电子云体素：rho(r)=Σ w_i * exp(-||r-r_i||^2/(2σ^2))

    参数
    - coords_A: (M,3) Å
    - weights: (M,)  原子权重（可用原子序数Z、|q|、Z-|q|等）
    - center_A: (3,) 体素立方体中心（Å）
    - 返回: (1,D,D,D) float32
    """
    coords_A = np.asarray(coords_A, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    center_A = np.asarray(center_A, dtype=np.float32).reshape(3)

    D = int(grid_size)
    half = (D - 1) / 2.0
    # voxel centers in Å, shape (D,)
    ax = (np.arange(D, dtype=np.float32) - half) * float(voxel_A)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")  # (D,D,D)
    grid_xyz = np.stack([X, Y, Z], axis=-1) + center_A[None, None, None, :]  # (D,D,D,3)

    # 计算到每个原子的平方距离并累加（向量化）
    # (D,D,D,1,3) - (1,1,1,M,3) -> (D,D,D,M,3)
    sigma2 = float(sigma_A) ** 2
    grid_flat = grid_xyz.reshape(-1, 3).astype(np.float32)
    rho = np.zeros((grid_flat.shape[0],), dtype=np.float32)
    atom_chunk = 64
    for start in range(0, coords_A.shape[0], atom_chunk):
        end = min(start + atom_chunk, coords_A.shape[0])
        diff = grid_flat[:, None, :] - coords_A[None, start:end, :]
        d2 = np.sum(diff * diff, axis=-1)
        contrib = np.exp(-0.5 * d2 / max(sigma2, 1e-6)) * weights[None, start:end]
        rho += np.sum(contrib, axis=-1).astype(np.float32)
    rho = rho.reshape(D, D, D)

    # 归一化（可选）：避免数值跨度过大
    vmax = float(np.max(rho)) if rho.size else 0.0
    if vmax > 0:
        rho = rho / vmax

    return torch.from_numpy(rho[None, ...])  # (1,D,D,D)


def gaussian_cloud_grid_torch(
    coords_A,
    weights,
    center_A,
    grid_size: int = 64,
    voxel_A: float = 0.75,
    sigma_A: float = 1.0,
    device: str = "cuda",
    voxel_chunk: int = 65536,
    atom_chunk: int = 128,
) -> torch.Tensor:
    """
    Torch implementation for GPU graph-cache builds. Returns a CPU tensor with
    shape (1,D,D,D), normalized to max 1 like gaussian_cloud_grid.
    """
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("GRAPH_CACHE_CLOUD_DEVICE requires CUDA, but torch.cuda.is_available() is false")
    dev = torch.device(device)
    coords = torch.as_tensor(coords_A, dtype=torch.float32, device=dev)
    weights = torch.as_tensor(weights, dtype=torch.float32, device=dev).reshape(-1)
    center = torch.as_tensor(center_A, dtype=torch.float32, device=dev).reshape(3)
    D = int(grid_size)
    ax = (torch.arange(D, dtype=torch.float32, device=dev) - (D - 1) / 2.0) * float(voxel_A)
    xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing="ij")
    grid = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3) + center.reshape(1, 3)
    rho = torch.empty((grid.shape[0],), dtype=torch.float32, device=dev)
    sigma2 = max(float(sigma_A) ** 2, 1e-6)
    voxel_chunk = max(1, int(voxel_chunk))
    atom_chunk = max(1, int(atom_chunk))
    for v0 in range(0, grid.shape[0], voxel_chunk):
        v1 = min(v0 + voxel_chunk, grid.shape[0])
        g = grid[v0:v1]
        acc = torch.zeros((g.shape[0],), dtype=torch.float32, device=dev)
        for a0 in range(0, coords.shape[0], atom_chunk):
            a1 = min(a0 + atom_chunk, coords.shape[0])
            diff = g[:, None, :] - coords[None, a0:a1, :]
            d2 = (diff * diff).sum(dim=-1)
            acc = acc + (torch.exp(-0.5 * d2 / sigma2) * weights[None, a0:a1]).sum(dim=-1)
        rho[v0:v1] = acc
    vmax = rho.max() if rho.numel() else torch.zeros((), device=dev)
    if float(vmax.item()) > 0.0:
        rho = rho / vmax
    return rho.reshape(1, D, D, D).detach().cpu()


def protein_ligand_cloud_grid(
    coords_A: np.ndarray,
    weights: np.ndarray,
    is_ligand: np.ndarray,
    center_A: np.ndarray,
    grid_size: int = 64,
    voxel_A: float = 0.75,
    sigma_A: float = 1.0,
    device: str = "cuda",
    voxel_chunk: int = 65536,
    atom_chunk: int = 128,
) -> torch.Tensor:
    """
    Build a 2-channel electron cloud:
      channel 0: protein/pocket residue atoms
      channel 1: ligand atoms

    Inputs are in Angstrom. The result is (2,D,D,D).
    """
    coords_A = np.asarray(coords_A, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    is_ligand = np.asarray(is_ligand).astype(bool).reshape(-1)
    if coords_A.shape[0] != weights.shape[0] or coords_A.shape[0] != is_ligand.shape[0]:
        raise ValueError("coords_A, weights, and is_ligand must have the same first dimension")

    def _cloud(mask):
        if not bool(np.any(mask)):
            D = int(grid_size)
            return torch.zeros((1, D, D, D), dtype=torch.float32)
        if str(device).startswith("cuda"):
            return gaussian_cloud_grid_torch(
                coords_A[mask],
                weights[mask],
                center_A,
                grid_size=grid_size,
                voxel_A=voxel_A,
                sigma_A=sigma_A,
                device=device,
                voxel_chunk=voxel_chunk,
                atom_chunk=atom_chunk,
            )
        return gaussian_cloud_grid(
            coords_A[mask], weights[mask], center_A,
            grid_size=grid_size, voxel_A=voxel_A, sigma_A=sigma_A,
        )

    protein = _cloud(~is_ligand)
    ligand = _cloud(is_ligand)
    return torch.cat([protein, ligand], dim=0)


def cloud_frame_from_atoms(coords_A, grid_size=64, padding_A=2.0, min_voxel_A=0.25, max_voxel_A=1.0):
    """
    Return (center_A, voxel_A, half_width_A) for a fixed-size grid enclosing the
    selected complete residues/ligand atoms.
    """
    coords_A = np.asarray(coords_A, dtype=np.float32)
    if coords_A.ndim != 2 or coords_A.shape[1] != 3 or coords_A.shape[0] == 0:
        raise ValueError("coords_A must have shape (N,3) with N > 0")
    lo = coords_A.min(axis=0)
    hi = coords_A.max(axis=0)
    center = ((lo + hi) * 0.5).astype(np.float32)
    half_extent = float(np.max((hi - lo) * 0.5)) + float(padding_A)
    voxel = (2.0 * half_extent) / max(int(grid_size) - 1, 1)
    voxel = float(np.clip(voxel, float(min_voxel_A), float(max_voxel_A)))
    half_width = ((int(grid_size) - 1) * voxel) * 0.5
    return center, voxel, half_width


def cloud_center_from_ligand(coords_A: np.ndarray, is_ligand: np.ndarray) -> np.ndarray:
    """
    用配体原子重心作为云中心（Å）。若无配体则用全体原子重心。
    """
    coords_A = np.asarray(coords_A, dtype=np.float32)
    is_ligand = np.asarray(is_ligand).astype(bool)
    if is_ligand.any():
        return coords_A[is_ligand].mean(axis=0)
    return coords_A.mean(axis=0)
