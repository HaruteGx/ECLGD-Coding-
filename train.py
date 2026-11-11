import math
import os
import random
import shutil
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from config import (
    ALLOW_BATCH_SIZE_ROUND_UP, AMP_DTYPE, AMP_ENABLED, BATCH_SIZE, CKPT_ROOT,
    CACHED_FRAME_COUNT, CACHED_FRAME_START,
    CLOUD_BOX_PADDING_A, CLOUD_ENCODER, CLOUD_EGNN_LAYERS, CLOUD_GAUSS_SIGMA_A, CLOUD_GRID_SIZE,
    CLOUD_KNN, CLOUD_LIGAND_DROPOUT, CLOUD_MAX_VOXEL_A, CLOUD_MIN_VOXEL_A, CLOUD_POINT_TOPK,
    CLOUD_MOLD_CENTER_WEIGHT, CLOUD_MOLD_LIGAND_WEIGHT, CLOUD_MOLD_PROTEIN_WEIGHT,
    CONDITION_PROTEIN_ONLY_CLOUD, ROLLOUT_FAST_PROTEIN_CLOUD,
    COORD_SCALE, DATALOADER_PERSISTENT_WORKERS, DATALOADER_PIN_MEMORY,
    DATALOADER_PREFETCH_FACTOR, DATALOADER_TIMEOUT_SECONDS,
    DDP_FIND_UNUSED_PARAMETERS, DDP_RANK_LOG_DIR,
    DDP_RANK_LOG_EVERY_STEPS, DDP_TIMEOUT_MINUTES, DEVICE, DIFFUSION_STEPS, DISTOGRAM_LOSS_WEIGHT,
    EARLY_STOPPING_ENABLED, EARLY_STOPPING_METRIC, EARLY_STOPPING_MIN_DELTA, EARLY_STOPPING_PATIENCE,
    FUTURE_FRAMES, GRAPH_ROOT,
    DIFFUSION_ABSOLUTE_ANCHOR_HORIZON_GAMMA, DIFFUSION_ABSOLUTE_ANCHOR_LOSS_WEIGHT,
    DIFFUSION_INCREMENT_NOISE_LOSS_WEIGHT, DIFFUSION_INCREMENT_X0_LOSS_WEIGHT,
    GRAD_ACCUM_STEPS, LIGAND_RIGID_POSE, LR, LR_PLATEAU_COOLDOWN, LR_PLATEAU_FACTOR,
    LSTM_KEYFRAME_COUNT,
    LR_ANCHOR_EPOCH, LR_ANCHOR_VALUE, LR_FINAL_EPOCH, LR_FINAL_VALUE,
    LR_PLATEAU_PATIENCE, LR_PLATEAU_THRESHOLD, LR_SCHEDULER, MAX_NUM_WORKERS, MIN_LR,
    MODEL_ARCHITECTURE, NUM_EPOCHS, NUM_WORKERS, OUT_ROOT,
    DYNAMIC_GRAPH_ENABLED, DYNAMIC_GRAPH_MAX_NEIGHBORS, DYNAMIC_GRAPH_RADIUS_A,
    MD_CONSERVATIVE_LOSS_WEIGHT, MD_ELECTRON_CUTOFF_A, MD_ELECTRON_SHELLS,
    MD_ENERGY_CONSISTENCY_LOSS_WEIGHT, MD_ENERGY_NORM_KCAL_PER_MOL_ATOM,
    MD_FORCE_ACCEL_NORM_A_PS2,
    MD_FORCE_CONSISTENCY_LOSS_WEIGHT, MD_INTERFACE_ENERGY_LOSS_WEIGHT,
    MD_INTERFACE_ENERGY_NORM_KCAL_PER_MOL_LIGAND_ATOM,
    MD_ELECTRON_SIGMA_A, MD_FORCE_CUTOFF_A, MD_FORCE_SWITCH_A, MD_NONBONDED_SOFTCORE_RATIO,
    QM_DENSITY_LOSS_WEIGHT, QM_DENSITY_REQUIRE_LABELS,
    MD_HIERARCHY_CLOUD_LOSS_WEIGHT, MD_HIERARCHY_RIGID_POCKET_WEIGHT, MD_PHYSICAL_TIME_DIM,
    MD_RDF_BINS, MD_RDF_MAX_A, MD_STORED_TIME_TO_PS, MD_USE_PBC, MD_VALIDATION_METRICS,
    OBSERVED_FRAMES, OBSERVED_GRAD_FRAMES, SEED, SMOKE_TEST_GRAPHS, TEST_SPLIT_FRAC,
    LATENT_LOSS_WEIGHT, PHYSICS_GEOMETRY_WEIGHT, PHYSICS_LOSS_WEIGHT, PHYSICS_MOLD_WEIGHT,
    PHYSICS_VELOCITY_WEIGHT, PLANNER_DELTA_LOSS_WEIGHT, PLANNER_KEY_DELTA_LOSS_WEIGHT, PROTEIN_OUTPUT_MODE,
    ROLLOUT_CENTER_LOSS_WEIGHT, ROLLOUT_FINAL_LOSS_WEIGHT, ROLLOUT_LOSS_WEIGHT, ROLLOUT_VELOCITY_LOSS_WEIGHT,
    TRAJECTORY_BACKBONE_GEOMETRY_WEIGHT, TRAJECTORY_BACKBONE_SMOOTH_WEIGHT,
    TRAJECTORY_BACKBONE_DIRECTION_WEIGHT, TRAJECTORY_CA_CONTINUITY_WEIGHT,
    TRAJECTORY_CLASH_WEIGHT, TRAJECTORY_FRAME_ROT_SCALE_RAD, TRAJECTORY_PEPTIDE_BOND_WEIGHT,
    TRAJECTORY_ACCELERATION_WEIGHT, TRAJECTORY_BOND_LENGTH_WEIGHT,
    TRAJECTORY_HBOND_WEIGHT, TRAJECTORY_HYDROPHOBIC_WEIGHT,
    TRAJECTORY_VDW_REPULSION_WEIGHT,
    TRAJECTORY_VELOCITY_WEIGHT,
    PROTEIN_BACKBONE_PROJECT, PROTEIN_BACKBONE_PROJECT_MODE, PROTEIN_BACKBONE_PROJECT_ITERS,
    PROTEIN_BACKBONE_PROJECT_STRENGTH, PROTEIN_BACKBONE_FORCE_SEQUENTIAL_CHAIN,
    PROTEIN_BACKBONE_PROJECT_EACH_DIFFUSION_STEP,
    PROTEIN_SE3_KNN, PROTEIN_SE3_LAYERS, PROTEIN_SE3_RADIUS_A,
    PROTEIN_SE3_RBF_DIM, PROTEIN_SE3_VECTOR_CHANNELS,
    TRAIN_FINAL_SAMPLE_STEPS, TRAIN_SAVE_FINAL_SAMPLE,
    TRAIN_LOG_EVERY_STEPS, TRAIN_MODE, TRAJ_EVAL_EVERY_EPOCHS, TRAJ_EVAL_LOG_EVERY_SAMPLES,
    TRAJ_EVAL_MAX_SAMPLES, TRAJ_EVAL_POSTPROCESS_PDB,
    TRAJ_EVAL_SAMPLE_STEPS, TRAJ_EVAL_TEACHER_FORCED,
    TRAJ_EVAL_MATCH_TRAIN_WINDOWS, TRAJ_EVAL_SEED, TRAJ_EVAL_STRIDE_SWEEP,
    TRAJ_FORCE_FINAL_FRAME, TRAJ_LEVEL, TRAJ_SOURCE_FRAME, TRAJ_TARGET_FRAME_STRIDE,
    TRAJ_TARGET_START_FRAME, TRAJ_OBSERVED_FRAME_STRIDE, TRAJ_RANDOM_WINDOW,
    TRAJ_RANDOM_STRIDE, TRAJ_MIN_FRAME_STRIDE, TRAJ_MAX_FRAME_STRIDE,
    TRAJ_RANDOM_WINDOWS_PER_TRAJ, TRAJ_STATIC_CONTEXT_PROB,
    STRICT_TRAJECTORY_ALIGNMENT,
    TRAJ_WINDOW_STRIDE,
    TRAIN_STEP_DEBUG, VAL_SPLIT_FRAC,
)
from data.collate import collate_trajectory_batch
from data.dataset import TrajectoryDataset, split_traj_dirs
from models.temporal_cloud_diffusion import TemporalCloudDiffusionModel
from utils.md import md_validation_metrics, minimum_image_displacement_batched


_TRAIN_START_TIME = time.perf_counter()


def _dist_is_initialized():
    return dist.is_available() and dist.is_initialized()


def _rank():
    if _dist_is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def _world_size():
    if _dist_is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_main_process():
    return _rank() == 0


def _log(message):
    if not _is_main_process():
        return
    elapsed = time.perf_counter() - _TRAIN_START_TIME
    print(f"[train +{elapsed:8.1f}s] {message}", flush=True)


def _rank_log(message):
    if not _dist_is_initialized() or int(DDP_RANK_LOG_EVERY_STEPS) <= 0:
        return
    elapsed = time.perf_counter() - _TRAIN_START_TIME
    log_dir = Path(DDP_RANK_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"rank{_rank()}.log").open("a", encoding="utf-8") as f:
        f.write(f"[train +{elapsed:8.1f}s] {message}\n")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move_batch_to_device(batch, device):
    non_blocking = bool(DATALOADER_PIN_MEMORY) and torch.device(device).type == "cuda"
    return {
        k: (v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def _loss_dict_to_scalars(loss_dict):
    return {
        key: float(value.item()) if torch.is_tensor(value) else value
        for key, value in loss_dict.items()
    }


def _setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA; CPU training path has been removed")
    backend = "nccl"
    if not _dist_is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=float(DDP_TIMEOUT_MINUTES)),
        )
    return True, local_rank, rank, world_size


def _cleanup_distributed():
    if _dist_is_initialized():
        dist.destroy_process_group()


def _effective_num_workers(world_size=1):
    requested = max(0, int(NUM_WORKERS))
    cap = max(0, int(MAX_NUM_WORKERS))
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        cap = min(cap, max(0, int(slurm_cpus)))
    return max(0, min(requested, cap) // max(1, int(world_size)))


def _per_rank_batch_size(world_size=1):
    requested = int(BATCH_SIZE)
    world = max(1, int(world_size))
    if requested % world != 0 and not bool(ALLOW_BATCH_SIZE_ROUND_UP):
        raise ValueError(
            f"BATCH_SIZE={requested} is not divisible by world_size={world}. "
            "Set BATCH_SIZE to an exact global batch or export ALLOW_BATCH_SIZE_ROUND_UP=1."
        )
    return max(1, int(math.ceil(requested / world)))


def _grad_accum_steps():
    return max(1, int(GRAD_ACCUM_STEPS))


def _raw_model(model):
    return model.module if hasattr(model, "module") else model


def _build_lr_scheduler(opt):
    scheduler_name = str(LR_SCHEDULER).lower()
    if scheduler_name in ("", "none", "off", "false"):
        return None
    if scheduler_name in ("anchored_exp", "anchor_exp", "anchored_exponential", "piecewise_exp"):
        base_lr = float(LR)
        anchor_epoch = max(1, int(LR_ANCHOR_EPOCH))
        final_epoch = max(anchor_epoch + 1, int(LR_FINAL_EPOCH))
        anchor_lr = min(base_lr, float(LR_ANCHOR_VALUE))
        final_lr = min(anchor_lr, float(LR_FINAL_VALUE))

        def lr_lambda(epoch):
            epoch = max(0, int(epoch))
            if epoch <= anchor_epoch:
                ratio = epoch / float(anchor_epoch)
                target_lr = base_lr * ((anchor_lr / base_lr) ** ratio)
            elif epoch <= final_epoch:
                ratio = (epoch - anchor_epoch) / float(final_epoch - anchor_epoch)
                target_lr = anchor_lr * ((final_lr / anchor_lr) ** ratio)
            else:
                target_lr = final_lr
            return max(target_lr / base_lr, final_lr / base_lr)

        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=max(1, int(NUM_EPOCHS)),
            eta_min=float(MIN_LR),
        )
    if scheduler_name in ("plateau", "reduce_on_plateau", "reduce_lr_on_plateau"):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=float(LR_PLATEAU_FACTOR),
            patience=max(0, int(LR_PLATEAU_PATIENCE)),
            threshold=float(LR_PLATEAU_THRESHOLD),
            threshold_mode="rel",
            cooldown=max(0, int(LR_PLATEAU_COOLDOWN)),
            min_lr=float(MIN_LR),
            eps=1e-9,
        )
    raise ValueError(f"Unsupported LR_SCHEDULER={LR_SCHEDULER!r}")


def _current_lr(opt):
    return float(opt.param_groups[0]["lr"])


def _should_eval_epoch(epoch):
    eval_every = max(0, int(TRAJ_EVAL_EVERY_EPOCHS))
    return eval_every > 0 and int(epoch) % eval_every == 0


def _resolve_amp(device):
    if device.type != "cuda" or not bool(AMP_ENABLED):
        return False, None, "off"
    dtype_name = str(AMP_DTYPE).strip().lower()
    if dtype_name in ("bf16", "bfloat16"):
        if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return True, torch.bfloat16, "bf16"
        return True, torch.float16, "fp16"
    if dtype_name in ("fp16", "float16", "half"):
        return True, torch.float16, "fp16"
    if dtype_name in ("fp32", "float32", "off", "false", "0"):
        return False, None, "off"
    raise ValueError(f"Unsupported AMP_DTYPE={AMP_DTYPE!r}")


def _autocast_context(device, enabled, dtype):
    if not enabled or dtype is None:
        return nullcontext()
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type=device.type, dtype=dtype)
    return torch.cuda.amp.autocast(dtype=dtype)


def _make_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _load_training_checkpoint(path, model, opt, scheduler, device):
    checkpoint = torch.load(Path(path), map_location=device)
    ckpt_arch = str(checkpoint.get("model_architecture", "")).strip()
    if ckpt_arch != str(MODEL_ARCHITECTURE):
        raise RuntimeError(
            f"resume checkpoint architecture mismatch: checkpoint={ckpt_arch or 'legacy'} "
            f"current={MODEL_ARCHITECTURE}. Start a fresh training run or use a matching checkpoint."
        )
    expected_md = {
        "md_physical_time_dim": int(MD_PHYSICAL_TIME_DIM),
        "md_electron_shells": int(MD_ELECTRON_SHELLS),
        "md_use_pbc": bool(MD_USE_PBC),
        "dynamic_graph_enabled": bool(DYNAMIC_GRAPH_ENABLED),
    }
    mismatches = []
    for key, expected in expected_md.items():
        actual = checkpoint.get(key, "<missing>")
        if actual == "<missing>" or actual != expected:
            mismatches.append(f"{key}: checkpoint={actual} current={expected}")
    if mismatches:
        raise RuntimeError(
            "resume checkpoint MD configuration mismatch: " + "; ".join(mismatches)
        )
    _raw_model(model).load_state_dict(checkpoint["state"], strict=True)
    if checkpoint.get("optimizer") is not None:
        try:
            opt.load_state_dict(checkpoint["optimizer"])
        except ValueError as exc:
            _log(f"stage=resume_ckpt optimizer_skipped reason={type(exc).__name__}: {exc}")
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
        except ValueError as exc:
            _log(f"stage=resume_ckpt scheduler_skipped reason={type(exc).__name__}: {exc}")
    return int(checkpoint.get("epoch", 0)) + 1, checkpoint


def _masked_rmse_A(pred, target, mask, boxes=None):
    diff = pred - target
    if boxes is not None:
        if boxes.ndim == 4:
            wrapped = diff.clone()
            for b in range(diff.shape[0]):
                wrapped[b] = minimum_image_displacement_batched(diff[b], boxes[b])
            diff = wrapped
        elif boxes.ndim == 3:
            diff = minimum_image_displacement_batched(diff, boxes)
    diff2 = (diff * float(COORD_SCALE)) ** 2
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    while mask.ndim < diff2.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.expand_as(diff2).sum().clamp_min(1.0)
    return torch.sqrt((diff2 * mask).sum() / denom)


def _clone_batch_for_sampling(batch):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.detach().clone()
        elif isinstance(value, list):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def _make_eval_loader(dataset, rank, world_size):
    if dataset is None:
        return None
    world_size = max(1, int(world_size))
    rank = int(rank)
    max_samples = int(TRAJ_EVAL_MAX_SAMPLES)
    if max_samples > 0:
        dataset = Subset(dataset, list(range(min(max_samples, len(dataset)))))
    if world_size > 1:
        indices = list(range(rank, len(dataset), world_size))
        dataset = Subset(dataset, indices)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_trajectory_batch,
        num_workers=0,
        pin_memory=bool(DATALOADER_PIN_MEMORY),
    )


def _unique_positive_ints(values):
    out = []
    seen = set()
    for value in values:
        value = int(value)
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _max_future_frames_for_cache(
    observed_frames,
    requested_future_frames,
    source_frame_id,
    target_start_frame,
    target_frame_stride,
    observed_frame_stride,
):
    observed_frames = int(observed_frames)
    requested_future_frames = int(requested_future_frames)
    target_frame_stride = max(1, int(target_frame_stride))
    observed_frame_stride = max(1, int(observed_frame_stride))
    cache_last = int(CACHED_FRAME_START) + int(CACHED_FRAME_COUNT) - 1
    observed_last = int(CACHED_FRAME_START) + (observed_frames - 1) * observed_frame_stride
    if observed_last > cache_last:
        raise RuntimeError(
            f"observed window does not fit graph cache: observed_frames={observed_frames} "
            f"obs_stride={observed_frame_stride} requires frame {observed_last}, "
            f"but cached range is [{CACHED_FRAME_START}, {cache_last}]"
        )
    source_id = (
        int(source_frame_id)
        if source_frame_id is not None
        else observed_last
    )
    if bool(STRICT_TRAJECTORY_ALIGNMENT) and source_id != observed_last:
        raise RuntimeError(
            f"TRAJ_SOURCE_FRAME={source_id} conflicts with last observed frame {observed_last}; "
            "clear TRAJ_SOURCE_FRAME or align it with the observed window"
        )
    target_start = (
        int(target_start_frame)
        if target_start_frame is not None
        else source_id + target_frame_stride
    )
    if source_id > cache_last:
        raise RuntimeError(
            f"source frame does not fit graph cache: source_frame={source_id}, "
            f"cached range is [{CACHED_FRAME_START}, {cache_last}]"
        )
    if target_start > cache_last:
        raise RuntimeError(
            f"target_start_frame={target_start} is outside cached range "
            f"[{CACHED_FRAME_START}, {cache_last}]"
        )
    max_future = ((cache_last - target_start) // target_frame_stride) + 1
    return max(1, min(requested_future_frames, int(max_future)))


def _build_eval_datasets(split_name, traj_dirs, future_frames=None):
    if not traj_dirs:
        return []
    future_frames = int(FUTURE_FRAMES if future_frames is None else future_frames)
    stride_sweep = _unique_positive_ints(TRAJ_EVAL_STRIDE_SWEEP)
    match_train_windows = bool(TRAJ_EVAL_MATCH_TRAIN_WINDOWS)
    random_window = bool(TRAJ_RANDOM_WINDOW) if match_train_windows else False
    window_stride = int(TRAJ_WINDOW_STRIDE) if match_train_windows else 0
    windows_per_traj = int(TRAJ_RANDOM_WINDOWS_PER_TRAJ) if match_train_windows else 1

    if stride_sweep:
        datasets = []
        for stride in stride_sweep:
            observed_stride = int(stride) if bool(TRAJ_RANDOM_STRIDE) else int(TRAJ_OBSERVED_FRAME_STRIDE)
            source_id = (
                int(TRAJ_SOURCE_FRAME)
                if TRAJ_SOURCE_FRAME is not None
                else int(CACHED_FRAME_START) + (int(OBSERVED_FRAMES) - 1) * int(observed_stride)
            )
            target_start = int(TRAJ_TARGET_START_FRAME) if TRAJ_TARGET_START_FRAME is not None else source_id + int(stride)
            cache_last = int(CACHED_FRAME_START) + int(CACHED_FRAME_COUNT) - 1
            max_future = 0
            if target_start <= cache_last:
                max_future = ((cache_last - target_start) // int(stride)) + 1
            if max_future <= 0:
                continue
            eval_future_frames = min(int(future_frames), int(max_future))
            datasets.append(
                (
                    f"{split_name}_s{int(stride)}",
                    TrajectoryDataset(
                        traj_dirs=traj_dirs,
                        observed_frames=OBSERVED_FRAMES,
                        future_frames=eval_future_frames,
                        source_frame_id=TRAJ_SOURCE_FRAME,
                        target_start_frame=TRAJ_TARGET_START_FRAME,
                        target_frame_stride=int(stride),
                        observed_frame_stride=observed_stride,
                        force_final_frame=TRAJ_FORCE_FINAL_FRAME,
                        window_stride=window_stride,
                        random_window=random_window,
                        random_stride=False,
                        random_windows_per_traj=windows_per_traj,
                    ),
                )
            )
        return datasets

    return [
        (
            split_name,
            TrajectoryDataset(
                traj_dirs=traj_dirs,
                observed_frames=OBSERVED_FRAMES,
                future_frames=future_frames,
                source_frame_id=TRAJ_SOURCE_FRAME,
                target_start_frame=TRAJ_TARGET_START_FRAME,
                target_frame_stride=TRAJ_TARGET_FRAME_STRIDE,
                observed_frame_stride=TRAJ_OBSERVED_FRAME_STRIDE,
                force_final_frame=TRAJ_FORCE_FINAL_FRAME,
                window_stride=window_stride,
                random_window=random_window,
                random_stride=bool(TRAJ_RANDOM_STRIDE) if random_window else False,
                min_frame_stride=TRAJ_MIN_FRAME_STRIDE,
                max_frame_stride=TRAJ_MAX_FRAME_STRIDE,
                random_windows_per_traj=windows_per_traj,
            ),
        )
    ]


def _mean_metric_sums(metrics, keys, device):
    if metrics is None:
        metrics = {}
    values = [float(metrics.get("samples", 0.0))]
    values.extend(float(metrics.get(key, 0.0)) for key in keys)
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    if _dist_is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    samples = float(tensor[0].item())
    if samples <= 0.0:
        return None
    out = {"samples": int(round(samples))}
    for idx, key in enumerate(keys, start=1):
        out[key] = float(tensor[idx].item() / samples)
    return out


@torch.no_grad()
def save_final_train_sample(model, batch, epoch, device, log_fn=None):
    if batch is None:
        return None
    was_training = model.training
    model.eval()
    steps = int(batch["future_coords"].shape[1])
    sample_steps = int(TRAIN_FINAL_SAMPLE_STEPS)
    pred = model.predict_future(
        batch,
        steps=steps,
        sample_steps=sample_steps,
    )
    out_dir = OUT_ROOT / "train_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"final_train_batch_epoch{int(epoch)}.pt"
    torch.save(
        {
            "epoch": int(epoch),
            "traj_dir": batch.get("traj_dir"),
            "observed_frame_ids": batch.get("observed_frame_ids"),
            "future_frame_ids": batch.get("future_frame_ids"),
            "pred_coords": pred["pred_coords"].detach().cpu(),
            "pred_protein_bb": pred["pred_protein_bb"].detach().cpu(),
            "target_coords": batch["future_coords"].detach().cpu(),
            "target_protein_bb": batch["future_protein_bb"].detach().cpu(),
            "pred_future_frame_ids": (
                pred["future_frame_ids"].detach().cpu()
                if torch.is_tensor(pred.get("future_frame_ids"))
                else pred.get("future_frame_ids", batch.get("future_frame_ids"))
            ),
            "pred_coarse_frame_ids": (
                pred["coarse_frame_ids"].detach().cpu()
                if torch.is_tensor(pred.get("coarse_frame_ids"))
                else pred.get("coarse_frame_ids")
            ),
            "coarse_frame_offsets": (
                pred["coarse_frame_offsets"].detach().cpu()
                if torch.is_tensor(pred.get("coarse_frame_offsets"))
                else pred.get("coarse_frame_offsets")
            ),
            "node_mask": batch["node_mask"].detach().cpu(),
            "future_protein_bb_mask": batch["future_protein_bb_mask"].detach().cpu(),
            "coord_scale": float(COORD_SCALE),
            "sample_steps": sample_steps,
        },
        path,
    )
    if was_training:
        model.train()
    if log_fn is not None:
        log_fn(f"stage=save_final_train_sample done path={path}")
    return path


@torch.no_grad()
def evaluate_temporal_diffusion(model, dl, device, log_fn=None, eval_seed=None):
    was_training = model.training
    model.eval()
    py_random_state = random.getstate()
    np_random_state = np.random.get_state()
    if eval_seed is not None:
        random.seed(int(eval_seed))
        np.random.seed(int(eval_seed) % (2**32 - 1))
    pocket_final = []
    pocket_traj = []
    protein_final = []
    protein_traj = []
    hold_pocket_final = []
    hold_pocket_traj = []
    hold_protein_final = []
    hold_protein_traj = []
    horizon_frames = []
    target_strides = []
    post_pocket_final = []
    post_protein_final = []
    postprocess_cache = {}
    md_metric_sums = {}
    postprocess_helpers = None
    if bool(TRAJ_EVAL_POSTPROCESS_PDB):
        from sample import load_postprocess_context_from_traj_dir, postprocess_prediction_arrays

        postprocess_helpers = (load_postprocess_context_from_traj_dir, postprocess_prediction_arrays)
    total_batches = len(dl)
    log_every = max(0, int(TRAJ_EVAL_LOG_EVERY_SAMPLES))
    for idx, batch in enumerate(dl, start=1):
        batch = _move_batch_to_device(batch, device)
        rng_devices = []
        if device.type == "cuda" and torch.cuda.is_available():
            rng_devices = [device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=rng_devices, enabled=eval_seed is not None):
            if eval_seed is not None:
                sample_seed = int(eval_seed) + int(idx)
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sample_seed)
            pred = model.predict_future(
                batch,
                steps=batch["future_coords"].shape[1],
                sample_steps=int(TRAJ_EVAL_SAMPLE_STEPS),
            )
        node_mask = batch["node_mask"]
        protein_mask = batch["future_protein_bb_mask"]
        horizon_frames.append(float(np.mean([int(x) for x in batch.get("horizon", [0])])))
        target_strides.append(float(np.mean([int(x) for x in batch.get("target_frame_stride", [0])])))
        hold_coords = batch["obs_coords"][:, -1:].expand_as(batch["future_coords"])
        hold_protein = batch["obs_protein_bb"][:, -1:].expand_as(batch["future_protein_bb"])
        future_boxes = batch.get("future_box")
        final_box = None if future_boxes is None else future_boxes[:, -1]
        pocket_traj.append(_masked_rmse_A(pred["pred_coords"], batch["future_coords"], node_mask[:, None, :], future_boxes))
        pocket_final.append(_masked_rmse_A(pred["pred_coords"][:, -1], batch["future_coords"][:, -1], node_mask, final_box))
        protein_traj.append(_masked_rmse_A(pred["pred_protein_bb"], batch["future_protein_bb"], protein_mask, future_boxes))
        protein_final.append(
            _masked_rmse_A(pred["pred_protein_bb"][:, -1], batch["future_protein_bb"][:, -1], protein_mask[:, -1], final_box)
        )
        hold_pocket_traj.append(_masked_rmse_A(hold_coords, batch["future_coords"], node_mask[:, None, :], future_boxes))
        hold_pocket_final.append(_masked_rmse_A(hold_coords[:, -1], batch["future_coords"][:, -1], node_mask, final_box))
        hold_protein_traj.append(_masked_rmse_A(hold_protein, batch["future_protein_bb"], protein_mask, future_boxes))
        hold_protein_final.append(
            _masked_rmse_A(hold_protein[:, -1], batch["future_protein_bb"][:, -1], protein_mask[:, -1], final_box)
        )
        if bool(MD_VALIDATION_METRICS):
            md_values = md_validation_metrics(
                pred["pred_coords"],
                batch["future_coords"],
                batch["obs_coords"][:, -1],
                batch.get("source_velocity", torch.zeros_like(batch["obs_coords"][:, -1])),
                node_mask,
                batch["future_times"],
                batch["source_time"],
                coord_scale=float(COORD_SCALE),
                node_masses=batch.get("node_masses"),
                node_feat=batch.get("node_feat"),
                node_vdw_radius_A=batch.get("node_vdw_radius_A"),
                node_is_ligand=batch.get("node_is_ligand"),
                bond_edge_index=batch.get("bond_edge_index"),
                boxes=batch.get("future_box"),
                source_box=batch.get("source_box"),
                rdf_bins=int(MD_RDF_BINS),
                rdf_max_A=float(MD_RDF_MAX_A),
                time_to_ps=float(MD_STORED_TIME_TO_PS),
            )
            for key, value in md_values.items():
                md_metric_sums[key] = md_metric_sums.get(key, 0.0) + float(value.item())
        if postprocess_helpers is not None:
            load_postprocess_context_from_traj_dir, postprocess_prediction_arrays = postprocess_helpers
            traj_dir = batch["traj_dir"][0]
            try:
                if traj_dir not in postprocess_cache:
                    postprocess_cache[traj_dir] = load_postprocess_context_from_traj_dir(traj_dir)
                meta, data = postprocess_cache[traj_dir]
                src_frame = int(batch["observed_frame_ids"][0][-1])
                post = postprocess_prediction_arrays(
                    src_frame=src_frame,
                    data=data,
                    meta=meta,
                    pred_coords_norm=pred["pred_coords"][0, -1],
                    pred_protein_bb_norm=pred["pred_protein_bb"][0, -1],
                    model=model,
                    batch=batch,
                    ligand_sample_steps=int(TRAJ_EVAL_SAMPLE_STEPS),
                )
                post_coords = torch.from_numpy(post["selected_coords_norm"]).to(
                    device=batch["future_coords"].device,
                    dtype=batch["future_coords"].dtype,
                ).unsqueeze(0)
                post_bb = torch.from_numpy(post["protein_bb_norm"]).to(
                    device=batch["future_protein_bb"].device,
                    dtype=batch["future_protein_bb"].dtype,
                ).unsqueeze(0)
                post_pocket_final.append(_masked_rmse_A(post_coords, batch["future_coords"][:, -1], node_mask, final_box))
                post_protein_final.append(
                    _masked_rmse_A(post_bb, batch["future_protein_bb"][:, -1], protein_mask[:, -1], final_box)
                )
            except Exception as exc:
                if log_fn is not None:
                    log_fn(
                        f"stage=eval_postprocess_skip sample={idx}/{total_batches} "
                        f"traj={traj_dir} reason={type(exc).__name__}: {exc}"
                    )
        if log_fn is not None and log_every > 0 and (idx == 1 or idx == total_batches or idx % log_every == 0):
            log_fn(f"stage=eval_progress sample={idx}/{total_batches}")

    if was_training:
        model.train()
    random.setstate(py_random_state)
    np.random.set_state(np_random_state)
    if not pocket_final:
        return {"samples": 0}
    metrics = {
        "samples": len(pocket_final),
        "horizon_frames": float(sum(horizon_frames)),
        "target_frame_stride": float(sum(target_strides)),
        "pocket_final_rmse_A": torch.stack(pocket_final).sum().item(),
        "pocket_traj_rmse_A": torch.stack(pocket_traj).sum().item(),
        "protein_final_rmse_A": torch.stack(protein_final).sum().item(),
        "protein_traj_rmse_A": torch.stack(protein_traj).sum().item(),
        "hold_pocket_final_rmse_A": torch.stack(hold_pocket_final).sum().item(),
        "hold_pocket_traj_rmse_A": torch.stack(hold_pocket_traj).sum().item(),
        "hold_protein_final_rmse_A": torch.stack(hold_protein_final).sum().item(),
        "hold_protein_traj_rmse_A": torch.stack(hold_protein_traj).sum().item(),
    }
    if post_pocket_final:
        metrics["post_pocket_final_rmse_A"] = torch.stack(post_pocket_final).sum().item()
        metrics["post_protein_final_rmse_A"] = torch.stack(post_protein_final).sum().item()
    metrics.update(md_metric_sums)
    return metrics


@torch.no_grad()
def evaluate_teacher_forced_loss(model, dl, device, eval_seed=None):
    was_training = model.training
    model.eval()
    py_random_state = random.getstate()
    np_random_state = np.random.get_state()
    if eval_seed is not None:
        random.seed(int(eval_seed))
        np.random.seed(int(eval_seed) % (2**32 - 1))
    totals = {}
    n = 0
    for batch in dl:
        batch = _move_batch_to_device(batch, device)
        loss, diffusion_loss, latent_loss = model(batch)
        loss_dict = getattr(model, "last_loss_dict", {})
        values = {
            "loss": float(loss.item()),
            "diffusion": float(diffusion_loss.item()),
            "latent": float(latent_loss.item()),
            "physics": float(loss_dict.get("physics", 0.0)),
            "rollout": float(loss_dict.get("rollout", 0.0)),
            "noise": float(loss_dict.get("noise", 0.0)),
            "increment_x0": float(loss_dict.get("increment_x0", 0.0)),
            "absolute_anchor": float(loss_dict.get("absolute_anchor", 0.0)),
            "recon": float(loss_dict.get("recon", 0.0)),
            "final": float(loss_dict.get("final", 0.0)),
        }
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value
        n += 1
    if was_training:
        model.train()
    random.setstate(py_random_state)
    np.random.set_state(np_random_state)
    if n <= 0:
        return {"samples": 0}
    return totals | {"samples": n}


def train(resume_ckpt=None):
    distributed, local_rank, rank, world_size = _setup_distributed()
    _log("stage=start")
    set_seed(SEED + rank)
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA; CPU training path has been removed")
    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(DEVICE)

    _log(
        f"stage=init mode={TRAIN_MODE} seed={SEED} device={device} "
        f"cuda_available={torch.cuda.is_available()} distributed={distributed} world_size={world_size}"
    )

    train_dirs, val_dirs, test_dirs = split_traj_dirs(
        graph_root=GRAPH_ROOT,
        test_frac=TEST_SPLIT_FRAC,
        val_frac=VAL_SPLIT_FRAC,
        seed=SEED,
        max_dirs=SMOKE_TEST_GRAPHS,
    )
    _log(
        f"stage=split_dirs done train_dirs={len(train_dirs)} "
        f"val_dirs={len(val_dirs)} test_dirs={len(test_dirs)} "
        f"val_frac={VAL_SPLIT_FRAC} test_frac={TEST_SPLIT_FRAC}"
    )

    effective_future_frames = _max_future_frames_for_cache(
        observed_frames=OBSERVED_FRAMES,
        requested_future_frames=FUTURE_FRAMES,
        source_frame_id=TRAJ_SOURCE_FRAME,
        target_start_frame=TRAJ_TARGET_START_FRAME,
        target_frame_stride=TRAJ_TARGET_FRAME_STRIDE,
        observed_frame_stride=TRAJ_OBSERVED_FRAME_STRIDE,
    )
    if int(effective_future_frames) < int(FUTURE_FRAMES):
        cache_last = int(CACHED_FRAME_START) + int(CACHED_FRAME_COUNT) - 1
        _log(
            f"stage=clip_future_frames requested={FUTURE_FRAMES} effective={effective_future_frames} "
            f"reason=cached_frame_range_{CACHED_FRAME_START}_{cache_last} "
            f"observed={OBSERVED_FRAMES} target_start={TRAJ_TARGET_START_FRAME} "
            f"target_stride={TRAJ_TARGET_FRAME_STRIDE}"
        )

    ds = TrajectoryDataset(
        traj_dirs=train_dirs,
        observed_frames=OBSERVED_FRAMES,
        future_frames=effective_future_frames,
        source_frame_id=TRAJ_SOURCE_FRAME,
        target_start_frame=TRAJ_TARGET_START_FRAME,
        target_frame_stride=TRAJ_TARGET_FRAME_STRIDE,
        observed_frame_stride=TRAJ_OBSERVED_FRAME_STRIDE,
        force_final_frame=TRAJ_FORCE_FINAL_FRAME,
        window_stride=TRAJ_WINDOW_STRIDE,
        random_window=TRAJ_RANDOM_WINDOW,
        random_stride=TRAJ_RANDOM_STRIDE,
        min_frame_stride=TRAJ_MIN_FRAME_STRIDE,
        max_frame_stride=TRAJ_MAX_FRAME_STRIDE,
        random_windows_per_traj=TRAJ_RANDOM_WINDOWS_PER_TRAJ,
    )
    eval_datasets = []
    eval_datasets.extend(_build_eval_datasets("val", val_dirs, future_frames=effective_future_frames))
    eval_datasets.extend(_build_eval_datasets("test", test_dirs, future_frames=effective_future_frames))
    _log(
        f"stage=build_dataset done samples={len(ds)} observed_frames={OBSERVED_FRAMES} "
        f"observed_grad_frames={OBSERVED_GRAD_FRAMES} "
        f"future_frames={effective_future_frames} requested_future_frames={FUTURE_FRAMES} "
        f"traj_level={TRAJ_LEVEL} source_frame={TRAJ_SOURCE_FRAME} "
        f"target_start={TRAJ_TARGET_START_FRAME} obs_stride={TRAJ_OBSERVED_FRAME_STRIDE} "
        f"target_stride={TRAJ_TARGET_FRAME_STRIDE} random_window={int(bool(TRAJ_RANDOM_WINDOW))} "
        f"random_stride={int(bool(TRAJ_RANDOM_STRIDE))} stride_range=[{TRAJ_MIN_FRAME_STRIDE},{TRAJ_MAX_FRAME_STRIDE}] "
        f"static_context_prob={TRAJ_STATIC_CONTEXT_PROB} "
        f"state_path=lstm_coarse_then_joint_local_increment_diffusion "
        f"force_final={int(bool(TRAJ_FORCE_FINAL_FRAME))} window_stride={TRAJ_WINDOW_STRIDE} "
        f"increment_diffusion_steps={DIFFUSION_STEPS}"
    )

    train_num_workers = _effective_num_workers(world_size=world_size)
    per_rank_batch_size = _per_rank_batch_size(world_size=world_size)
    grad_accum_steps = _grad_accum_steps()
    micro_global_batch_size = per_rank_batch_size * world_size
    effective_global_batch_size = micro_global_batch_size * grad_accum_steps
    train_sampler = DistributedSampler(ds, shuffle=True, drop_last=False) if distributed else None
    train_loader_kwargs = {
        "batch_size": per_rank_batch_size,
        "shuffle": train_sampler is None,
        "sampler": train_sampler,
        "collate_fn": collate_trajectory_batch,
        "num_workers": train_num_workers,
        "pin_memory": bool(DATALOADER_PIN_MEMORY),
    }
    if train_num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = max(1, int(DATALOADER_PREFETCH_FACTOR))
        train_loader_kwargs["persistent_workers"] = bool(DATALOADER_PERSISTENT_WORKERS)
        if int(DATALOADER_TIMEOUT_SECONDS) > 0:
            train_loader_kwargs["timeout"] = int(DATALOADER_TIMEOUT_SECONDS)
    dl = DataLoader(ds, **train_loader_kwargs)
    eval_loaders = [
        (split_name, _make_eval_loader(eval_ds, rank=rank, world_size=world_size))
        for split_name, eval_ds in eval_datasets
    ]
    eval_desc = ",".join(
        f"{split_name}:local_batches={len(eval_dl)} global_samples={len(eval_ds)}"
        for (split_name, eval_ds), (_, eval_dl) in zip(eval_datasets, eval_loaders)
    ) or "none"
    _log(
        f"stage=build_dataloader done train_batches={len(dl)} "
        f"eval_targets={eval_desc} "
        f"global_micro_batch={micro_global_batch_size} per_rank_batch_size={per_rank_batch_size} "
        f"grad_accum_steps={grad_accum_steps} effective_global_batch={effective_global_batch_size} "
        f"workers={train_num_workers} pin_memory={int(bool(DATALOADER_PIN_MEMORY))} "
        f"persistent_workers={int(bool(DATALOADER_PERSISTENT_WORKERS) and train_num_workers > 0)} "
        f"prefetch={DATALOADER_PREFETCH_FACTOR if train_num_workers > 0 else 0} "
        f"timeout={DATALOADER_TIMEOUT_SECONDS if train_num_workers > 0 else 0}"
    )

    init_model_start = time.perf_counter()
    _log(
        f"stage=init_model_build start observed_frames={OBSERVED_FRAMES} "
        f"future_frames={effective_future_frames} device={device} distributed={distributed}"
    )
    model = TemporalCloudDiffusionModel(
        observed_frames=OBSERVED_FRAMES,
        future_frames=effective_future_frames,
    )
    _log(f"stage=init_model_build done seconds={time.perf_counter() - init_model_start:.2f}")

    to_device_start = time.perf_counter()
    _log(f"stage=init_model_to_device start device={device}")
    model = model.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    _log(f"stage=init_model_to_device done seconds={time.perf_counter() - to_device_start:.2f}")

    if distributed:
        ddp_start = time.perf_counter()
        _log(
            f"stage=init_model_ddp_wrap start local_rank={local_rank} "
            f"find_unused={int(bool(DDP_FIND_UNUSED_PARAMETERS))}"
        )
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=bool(DDP_FIND_UNUSED_PARAMETERS),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        _log(f"stage=init_model_ddp_wrap done seconds={time.perf_counter() - ddp_start:.2f}")

    optim_start = time.perf_counter()
    _log("stage=init_optimizer start")
    opt = torch.optim.Adam(model.parameters(), lr=float(LR))
    scheduler = _build_lr_scheduler(opt)
    amp_enabled, amp_dtype, amp_name = _resolve_amp(device)
    scaler = _make_grad_scaler(amp_enabled and amp_dtype == torch.float16)
    _log(f"stage=init_optimizer done seconds={time.perf_counter() - optim_start:.2f}")
    start_epoch = 1
    if resume_ckpt is not None:
        _log(f"stage=resume_ckpt start path={resume_ckpt}")
        start_epoch, checkpoint = _load_training_checkpoint(resume_ckpt, model, opt, scheduler, device)
        _log(f"stage=resume_ckpt done loaded_epoch={checkpoint.get('epoch', 'unknown')} start_epoch={start_epoch}")
    _log(
        f"stage=init_model done lr={_current_lr(opt):.8g} scheduler={LR_SCHEDULER} "
        f"min_lr={MIN_LR} lr_anchor_epoch={LR_ANCHOR_EPOCH} "
        f"lr_anchor_value={LR_ANCHOR_VALUE} lr_final_epoch={LR_FINAL_EPOCH} "
        f"lr_final_value={LR_FINAL_VALUE} plateau_factor={LR_PLATEAU_FACTOR} "
        f"plateau_patience={LR_PLATEAU_PATIENCE} plateau_threshold={LR_PLATEAU_THRESHOLD} "
        f"plateau_cooldown={LR_PLATEAU_COOLDOWN} epochs={NUM_EPOCHS} amp={amp_name} "
        f"latent_loss_weight={LATENT_LOSS_WEIGHT} physics_loss_weight={PHYSICS_LOSS_WEIGHT} "
        f"distogram_loss_weight={DISTOGRAM_LOSS_WEIGHT} "
        f"diffusion_absolute_anchor_weight={DIFFUSION_ABSOLUTE_ANCHOR_LOSS_WEIGHT} "
        f"diffusion_increment_x0_weight={DIFFUSION_INCREMENT_X0_LOSS_WEIGHT} "
        f"absolute_anchor_horizon_gamma={DIFFUSION_ABSOLUTE_ANCHOR_HORIZON_GAMMA} "
        f"increment_noise_weight={DIFFUSION_INCREMENT_NOISE_LOSS_WEIGHT} "
        f"rollout_weight={ROLLOUT_LOSS_WEIGHT} rollout_final={ROLLOUT_FINAL_LOSS_WEIGHT} "
        f"rollout_velocity={ROLLOUT_VELOCITY_LOSS_WEIGHT} rollout_center={ROLLOUT_CENTER_LOSS_WEIGHT} "
        f"planner_delta_weight={PLANNER_DELTA_LOSS_WEIGHT} "
        f"planner_key_delta_weight={PLANNER_KEY_DELTA_LOSS_WEIGHT} "
        f"physics_mix=[vel:{PHYSICS_VELOCITY_WEIGHT},mold:{PHYSICS_MOLD_WEIGHT},geom:{PHYSICS_GEOMETRY_WEIGHT}] "
        f"traj_physics=[bb_geom:{TRAJECTORY_BACKBONE_GEOMETRY_WEIGHT},bb_smooth:{TRAJECTORY_BACKBONE_SMOOTH_WEIGHT},"
        f"peptide:{TRAJECTORY_PEPTIDE_BOND_WEIGHT},ca_cont:{TRAJECTORY_CA_CONTINUITY_WEIGHT},"
        f"bb_dir:{TRAJECTORY_BACKBONE_DIRECTION_WEIGHT},clash:{TRAJECTORY_CLASH_WEIGHT},"
        f"vel:{TRAJECTORY_VELOCITY_WEIGHT},acc:{TRAJECTORY_ACCELERATION_WEIGHT},"
        f"bond:{TRAJECTORY_BOND_LENGTH_WEIGHT},vdw:{TRAJECTORY_VDW_REPULSION_WEIGHT},"
        f"hbond:{TRAJECTORY_HBOND_WEIGHT},hyd:{TRAJECTORY_HYDROPHOBIC_WEIGHT}] "
        f"md=[pbc:{int(bool(MD_USE_PBC))},dynamic_graph:{int(bool(DYNAMIC_GRAPH_ENABLED))},"
        f"radius_A:{DYNAMIC_GRAPH_RADIUS_A},neighbors:{DYNAMIC_GRAPH_MAX_NEIGHBORS},"
        f"time_to_ps:{MD_STORED_TIME_TO_PS},electron_shells:{MD_ELECTRON_SHELLS},"
        f"hierarchy_w:{MD_HIERARCHY_CLOUD_LOSS_WEIGHT},verlet_w:{MD_CONSERVATIVE_LOSS_WEIGHT},"
        f"force_w:{MD_FORCE_CONSISTENCY_LOSS_WEIGHT},energy_w:{MD_ENERGY_CONSISTENCY_LOSS_WEIGHT},"
        f"interface_energy_w:{MD_INTERFACE_ENERGY_LOSS_WEIGHT},"
        f"force_norm_A_ps2:{MD_FORCE_ACCEL_NORM_A_PS2},"
        f"energy_norm:{MD_ENERGY_NORM_KCAL_PER_MOL_ATOM},"
        f"interface_energy_norm:{MD_INTERFACE_ENERGY_NORM_KCAL_PER_MOL_LIGAND_ATOM}] "
        f"frame_rot_scale_rad={TRAJECTORY_FRAME_ROT_SCALE_RAD} "
        f"protein_backbone_project={int(bool(PROTEIN_BACKBONE_PROJECT))} "
        f"protein_backbone_project_mode={PROTEIN_BACKBONE_PROJECT_MODE} "
        f"protein_backbone_project_iters={PROTEIN_BACKBONE_PROJECT_ITERS} "
        f"protein_backbone_project_strength={PROTEIN_BACKBONE_PROJECT_STRENGTH} "
        f"protein_backbone_force_sequential_chain={int(bool(PROTEIN_BACKBONE_FORCE_SEQUENTIAL_CHAIN))} "
        f"protein_backbone_project_each_diffusion_step={int(bool(PROTEIN_BACKBONE_PROJECT_EACH_DIFFUSION_STEP))} "
        f"protein_encoder=se3_temporal_transformer "
        f"protein_se3_layers={PROTEIN_SE3_LAYERS} protein_se3_knn={PROTEIN_SE3_KNN} "
        f"protein_se3_vectors={PROTEIN_SE3_VECTOR_CHANNELS} protein_se3_rbf={PROTEIN_SE3_RBF_DIM} "
        f"protein_se3_radius_A={PROTEIN_SE3_RADIUS_A} "
        f"cloud_mold_mix=[protein:{CLOUD_MOLD_PROTEIN_WEIGHT},ligand:{CLOUD_MOLD_LIGAND_WEIGHT},center:{CLOUD_MOLD_CENTER_WEIGHT}] "
        f"ddp_find_unused={int(bool(DDP_FIND_UNUSED_PARAMETERS))} "
        f"protein_output_mode={PROTEIN_OUTPUT_MODE} "
        f"ligand_rigid_pose={int(bool(LIGAND_RIGID_POSE))} "
        f"cloud_encoder={CLOUD_ENCODER} cloud_topk={CLOUD_POINT_TOPK} "
        f"cloud_knn={CLOUD_KNN} cloud_egnn_layers={CLOUD_EGNN_LAYERS} "
        f"cloud_channels=protein,ligand cloud_grid={CLOUD_GRID_SIZE} "
        f"cloud_padding_A={CLOUD_BOX_PADDING_A} cloud_voxel_A_range=[{CLOUD_MIN_VOXEL_A},{CLOUD_MAX_VOXEL_A}] "
        f"cloud_sigma_A={CLOUD_GAUSS_SIGMA_A} cloud_ligand_dropout={CLOUD_LIGAND_DROPOUT} "
        f"condition_protein_only_cloud={int(bool(CONDITION_PROTEIN_ONLY_CLOUD))} "
        f"fast_rollout_protein_cloud={int(bool(ROLLOUT_FAST_PROTEIN_CLOUD))} "
        f"traj_level={TRAJ_LEVEL} target_stride={TRAJ_TARGET_FRAME_STRIDE} "
        f"lstm_keyframes={LSTM_KEYFRAME_COUNT} "
        f"obs_stride={TRAJ_OBSERVED_FRAME_STRIDE} random_window={int(bool(TRAJ_RANDOM_WINDOW))} "
        f"random_stride={int(bool(TRAJ_RANDOM_STRIDE))} stride_range=[{TRAJ_MIN_FRAME_STRIDE},{TRAJ_MAX_FRAME_STRIDE}] "
        f"static_context_prob={TRAJ_STATIC_CONTEXT_PROB} "
        f"state_path=lstm_coarse_then_joint_local_increment_diffusion "
        f"force_final={int(bool(TRAJ_FORCE_FINAL_FRAME))} window_stride={TRAJ_WINDOW_STRIDE} "
        f"eval_every_epochs={TRAJ_EVAL_EVERY_EPOCHS} "
        f"eval_max_samples={TRAJ_EVAL_MAX_SAMPLES} eval_seed={TRAJ_EVAL_SEED} "
        f"eval_log_every_samples={TRAJ_EVAL_LOG_EVERY_SAMPLES} "
        f"eval_postprocess_pdb={int(bool(TRAJ_EVAL_POSTPROCESS_PDB))} "
        f"eval_sample_steps={TRAJ_EVAL_SAMPLE_STEPS} "
        f"eval_match_train_windows={int(bool(TRAJ_EVAL_MATCH_TRAIN_WINDOWS))} "
        f"eval_stride_sweep={','.join(str(x) for x in TRAJ_EVAL_STRIDE_SWEEP) or 'none'} "
        f"eval_teacher_forced={int(bool(TRAJ_EVAL_TEACHER_FORCED))} "
        f"early_stopping={int(bool(EARLY_STOPPING_ENABLED))} "
        f"early_stopping_metric={EARLY_STOPPING_METRIC} "
        f"early_stopping_patience={EARLY_STOPPING_PATIENCE} "
        f"early_stopping_min_delta={EARLY_STOPPING_MIN_DELTA} "
        f"save_final_train_sample={int(bool(TRAIN_SAVE_FINAL_SAMPLE))}"
    )

    try:
        last_train_batch_for_sample = None
        early_stopping_best = float("inf")
        early_stopping_bad_evals = 0
        for ep in range(start_epoch, int(NUM_EPOCHS) + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(ep)
            model.train()
            epoch_start = time.perf_counter()
            tot = tot_diffusion = tot_latent = 0.0
            _log(f"stage=epoch_start epoch={ep}/{NUM_EPOCHS} lr={_current_lr(opt):.8g}")
            opt.zero_grad(set_to_none=True)
            prev_step_end = time.perf_counter()
            for step, batch in enumerate(dl, start=1):
                batch_load_done = time.perf_counter()
                if _is_main_process() and bool(TRAIN_STEP_DEBUG):
                    _log(
                        f"stage=batch_loaded epoch={ep}/{NUM_EPOCHS} step={step}/{len(dl)} "
                        f"load_wait_s={batch_load_done - prev_step_end:.2f}"
                    )
                if bool(TRAIN_STEP_DEBUG):
                    _rank_log(
                        f"rank={rank} stage=batch_loaded epoch={ep}/{NUM_EPOCHS} "
                        f"step={step}/{len(dl)} traj={batch.get('traj_dir', [])}"
                    )
                batch = _move_batch_to_device(batch, device)
                batch_device_done = time.perf_counter()
                if _is_main_process() and bool(TRAIN_STEP_DEBUG):
                    node_max = int(batch["node_mask"].sum(dim=1).max().detach().item()) if "node_mask" in batch else -1
                    res_max = (
                        int(batch["future_protein_bb_mask"].bool().any(dim=1).sum(dim=1).max().detach().item())
                        if "future_protein_bb_mask" in batch
                        else -1
                    )
                    _log(
                        f"stage=batch_to_device_done epoch={ep}/{NUM_EPOCHS} step={step}/{len(dl)} "
                        f"to_device_s={batch_device_done - batch_load_done:.2f} max_nodes={node_max} max_res_tokens={res_max}"
                    )
                if (
                    _is_main_process()
                    and bool(TRAIN_SAVE_FINAL_SAMPLE)
                    and int(ep) == int(NUM_EPOCHS)
                    and int(step) == len(dl)
                ):
                    last_train_batch_for_sample = _clone_batch_for_sampling(batch)
                if bool(TRAIN_STEP_DEBUG):
                    _rank_log(
                        f"rank={rank} stage=batch_to_device_done epoch={ep}/{NUM_EPOCHS} "
                        f"step={step}/{len(dl)}"
                    )
                should_step = (step % grad_accum_steps == 0) or (step == len(dl))
                sync_context = (
                    model.no_sync()
                    if distributed and hasattr(model, "no_sync") and not should_step
                    else nullcontext()
                )
                if bool(TRAIN_STEP_DEBUG):
                    _rank_log(
                        f"rank={rank} stage=forward_backward_start epoch={ep}/{NUM_EPOCHS} "
                        f"step={step}/{len(dl)} should_step={int(should_step)}"
                )
                with sync_context:
                    with _autocast_context(device, amp_enabled, amp_dtype):
                        forward_start = time.perf_counter()
                        loss, diffusion_loss, latent_loss = model(batch)
                        loss_for_backward = loss / float(grad_accum_steps)
                    forward_done = time.perf_counter()
                    if _is_main_process() and bool(TRAIN_STEP_DEBUG):
                        _log(
                            f"stage=forward_done epoch={ep}/{NUM_EPOCHS} step={step}/{len(dl)} "
                            f"forward_s={forward_done - forward_start:.2f}"
                        )
                    if scaler.is_enabled():
                        scaler.scale(loss_for_backward).backward()
                    else:
                        loss_for_backward.backward()
                    backward_done = time.perf_counter()
                    if _is_main_process() and bool(TRAIN_STEP_DEBUG):
                        _log(
                            f"stage=backward_done epoch={ep}/{NUM_EPOCHS} step={step}/{len(dl)} "
                            f"backward_s={backward_done - forward_done:.2f}"
                        )
                forward_backward_done = time.perf_counter()
                if _is_main_process() and bool(TRAIN_STEP_DEBUG):
                    _log(
                        f"stage=forward_backward_done epoch={ep}/{NUM_EPOCHS} step={step}/{len(dl)} "
                        f"forward_backward_s={forward_backward_done - batch_device_done:.2f}"
                    )
                if bool(TRAIN_STEP_DEBUG):
                    _rank_log(
                        f"rank={rank} stage=forward_backward_done epoch={ep}/{NUM_EPOCHS} "
                        f"step={step}/{len(dl)}"
                    )

                if should_step:
                    if scaler.is_enabled():
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)
                prev_step_end = time.perf_counter()

                tot += float(loss.item())
                tot_diffusion += float(diffusion_loss.item())
                tot_latent += float(latent_loss.item())

                if step == len(dl) or (
                    int(TRAIN_LOG_EVERY_STEPS) > 0 and step % int(TRAIN_LOG_EVERY_STEPS) == 0
                ):
                    loss_dict = _loss_dict_to_scalars(getattr(_raw_model(model), "last_loss_dict", {}))
                    detail = ""
                    if loss_dict:
                        detail = (
                            f" totalRaw={loss_dict.get('total_without_anchor', 0.0):.6f}"
                            f" diff={loss_dict.get('diffusion', 0.0):.6f}"
                            f" incrementX0={loss_dict.get('increment_x0', 0.0):.6f}"
                            f" absoluteAnchor={loss_dict.get('absolute_anchor', 0.0):.6f}"
                            f" tMean={loss_dict.get('diffusion_t_mean', 0.0):.2f}"
                            f" x0Weight={loss_dict.get('diffusion_x0_weight_mean', 0.0):.3f}"
                            f" incrementTgtRmsA/Tau={loss_dict.get('coord_increment_target_rms_A_per_tau', 0.0):.3f}/{loss_dict.get('protein_increment_target_rms_A_per_tau', 0.0):.3f}"
                            f" noiseRms={loss_dict.get('coord_noise_rms', 0.0):.3f}/{loss_dict.get('protein_noise_rms', 0.0):.3f}"
                            f" epsRms={loss_dict.get('coord_eps_rms', 0.0):.3f}/{loss_dict.get('protein_eps_rms', 0.0):.3f}"
                            f" x0RmsA={loss_dict.get('coord_x0_rms_A', 0.0):.3f}/{loss_dict.get('protein_x0_rms_A', 0.0):.3f}"
                            f" traj={loss_dict.get('trajectory', 0.0):.6f}"
                            f" phys={loss_dict.get('physics', 0.0):.6f}"
                            f" pep={loss_dict.get('peptide_bond', 0.0):.6f}"
                            f" caCont={loss_dict.get('ca_continuity', 0.0):.6f}"
                            f" bbDir={loss_dict.get('backbone_direction', 0.0):.6f}"
                            f" vel={loss_dict.get('velocity', 0.0):.6f}"
                            f" acc={loss_dict.get('acceleration', 0.0):.6f}"
                            f" bond={loss_dict.get('bond_length', 0.0):.6f}"
                            f" vdw={loss_dict.get('vdw_repulsion', 0.0):.6f}"
                            f" hb={loss_dict.get('hbond', 0.0):.6f}"
                            f" hyd={loss_dict.get('hydrophobic', 0.0):.6f}"
                            f" verlet={loss_dict.get('conservative_verlet', 0.0):.6f}"
                            f" forceCons={loss_dict.get('force_consistency', 0.0):.6f}"
                            f" relEnergy={loss_dict.get('relative_energy', 0.0):.6f}"
                            f" interfaceEnergy={loss_dict.get('interface_relative_energy', 0.0):.6f}"
                            f" cloudHierarchy={loss_dict.get('cloud_hierarchy', 0.0):.6f}"
                            f" qmDensity={loss_dict.get('qm_density_loss', 0.0):.6f}"
                            f" qmCoverage={loss_dict.get('qm_density_coverage', 0.0):.3f}"
                            f" pocketRigid={loss_dict.get('pocket_rigid', 0.0):.6f}"
                            f" pocketDiff={loss_dict.get('pocket_diffusion', 0.0):.6f}"
                            f" ligandDiff={loss_dict.get('ligand_diffusion', 0.0):.6f}"
                            f" mdMSD={loss_dict.get('md_msd_rmse_A2', 0.0):.4f}"
                            f" mdVACF={loss_dict.get('md_vacf_mae_A2_per_ps2', 0.0):.4f}"
                            f" disto={loss_dict.get('distogram', 0.0):.6f}"
                            f" latent={loss_dict.get('latent', 0.0):.6f}"
                            f" gateLat={loss_dict.get('gate_latent', 0.0):.6f}"
                            f" encAux={loss_dict.get('encoder_aux', 0.0):.6f}"
                            f" step={loss_dict.get('keyframe', 0.0):.6f}"
                        )
                    _log(
                        f"stage=batch_progress epoch={ep}/{NUM_EPOCHS} step={step}/{len(dl)} "
                        f"loss={loss.item():.6f} diffusion={diffusion_loss.item():.6f} "
                        f"latent={latent_loss.item():.6f} optimizer_step={int(should_step)}{detail}"
                    )
                if distributed and (
                    step == len(dl) or (
                        int(DDP_RANK_LOG_EVERY_STEPS) > 0 and step % int(DDP_RANK_LOG_EVERY_STEPS) == 0
                    )
                ):
                    _rank_log(f"rank={rank} epoch={ep}/{NUM_EPOCHS} step={step}/{len(dl)} loss={loss.item():.6f}")

            epoch_loss = tot / len(dl)
            epoch_diffusion = tot_diffusion / len(dl)
            epoch_latent = tot_latent / len(dl)
            _log(
                f"[epoch {ep}] loss={epoch_loss:.6f} diffusion={epoch_diffusion:.6f} "
                f"latent={epoch_latent:.6f} seconds={time.perf_counter() - epoch_start:.1f}"
            )
            if scheduler is not None:
                if str(LR_SCHEDULER).lower() in ("plateau", "reduce_on_plateau", "reduce_lr_on_plateau"):
                    scheduler.step(epoch_loss)
                    _log(
                        f"stage=lr_scheduler epoch={ep}/{NUM_EPOCHS} "
                        f"metric=train_loss value={epoch_loss:.6f} next_lr={_current_lr(opt):.8g}"
                    )
                else:
                    scheduler.step()
                    _log(f"stage=lr_scheduler epoch={ep}/{NUM_EPOCHS} next_lr={_current_lr(opt):.8g}")

            if _is_main_process():
                CKPT_ROOT.mkdir(parents=True, exist_ok=True)
                ckpt_path = CKPT_ROOT / f"ckpt_epoch{ep}.pt"
                _log(f"stage=save_ckpt start path={ckpt_path}")
                torch.save(
                    {
                        "epoch": ep,
                        "state": _raw_model(model).state_dict(),
                        "train_mode": TRAIN_MODE,
                        "model_architecture": MODEL_ARCHITECTURE,
                        "dynamic_graph_enabled": bool(DYNAMIC_GRAPH_ENABLED),
                        "dynamic_graph_radius_A": DYNAMIC_GRAPH_RADIUS_A,
                        "dynamic_graph_max_neighbors": DYNAMIC_GRAPH_MAX_NEIGHBORS,
                        "md_use_pbc": bool(MD_USE_PBC),
                        "md_stored_time_to_ps": MD_STORED_TIME_TO_PS,
                        "md_physical_time_dim": MD_PHYSICAL_TIME_DIM,
                        "md_electron_shells": MD_ELECTRON_SHELLS,
                        "md_electron_cutoff_A": MD_ELECTRON_CUTOFF_A,
                        "md_electron_sigma_A": MD_ELECTRON_SIGMA_A,
                        "md_hierarchy_cloud_loss_weight": MD_HIERARCHY_CLOUD_LOSS_WEIGHT,
                        "md_hierarchy_rigid_pocket_weight": MD_HIERARCHY_RIGID_POCKET_WEIGHT,
                        "md_conservative_loss_weight": MD_CONSERVATIVE_LOSS_WEIGHT,
                        "md_force_consistency_loss_weight": MD_FORCE_CONSISTENCY_LOSS_WEIGHT,
                        "md_energy_consistency_loss_weight": MD_ENERGY_CONSISTENCY_LOSS_WEIGHT,
                        "md_interface_energy_loss_weight": MD_INTERFACE_ENERGY_LOSS_WEIGHT,
                        "md_force_accel_norm_A_ps2": MD_FORCE_ACCEL_NORM_A_PS2,
                        "md_energy_norm_kcal_per_mol_atom": MD_ENERGY_NORM_KCAL_PER_MOL_ATOM,
                        "md_interface_energy_norm_kcal_per_mol_ligand_atom": (
                            MD_INTERFACE_ENERGY_NORM_KCAL_PER_MOL_LIGAND_ATOM
                        ),
                        "md_force_cutoff_A": MD_FORCE_CUTOFF_A,
                        "md_force_switch_A": MD_FORCE_SWITCH_A,
                        "md_nonbonded_softcore_ratio": MD_NONBONDED_SOFTCORE_RATIO,
                        "qm_density_loss_weight": QM_DENSITY_LOSS_WEIGHT,
                        "qm_density_require_labels": QM_DENSITY_REQUIRE_LABELS,
                        "observed_frames": OBSERVED_FRAMES,
                        "observed_grad_frames": OBSERVED_GRAD_FRAMES,
                        "future_frames": effective_future_frames,
                        "requested_future_frames": FUTURE_FRAMES,
                        "lstm_keyframe_count": LSTM_KEYFRAME_COUNT,
                        "diffusion_steps": DIFFUSION_STEPS,
                        "protein_output_mode": PROTEIN_OUTPUT_MODE,
                        "protein_encoder": "se3_temporal_transformer",
                        "protein_se3_layers": PROTEIN_SE3_LAYERS,
                        "protein_se3_knn": PROTEIN_SE3_KNN,
                        "protein_se3_vector_channels": PROTEIN_SE3_VECTOR_CHANNELS,
                        "protein_se3_rbf_dim": PROTEIN_SE3_RBF_DIM,
                        "protein_se3_radius_A": PROTEIN_SE3_RADIUS_A,
                        "ligand_rigid_pose": bool(LIGAND_RIGID_POSE),
                        "cloud_encoder": CLOUD_ENCODER,
                        "cloud_channels": ("protein", "ligand"),
                        "cloud_grid_size": CLOUD_GRID_SIZE,
                        "cloud_box_padding_A": CLOUD_BOX_PADDING_A,
                        "cloud_min_voxel_A": CLOUD_MIN_VOXEL_A,
                        "cloud_max_voxel_A": CLOUD_MAX_VOXEL_A,
                        "cloud_sigma_A": CLOUD_GAUSS_SIGMA_A,
                        "cloud_ligand_dropout": CLOUD_LIGAND_DROPOUT,
                        "condition_protein_only_cloud": bool(CONDITION_PROTEIN_ONLY_CLOUD),
                        "rollout_fast_protein_cloud": bool(ROLLOUT_FAST_PROTEIN_CLOUD),
                        "cloud_point_topk": CLOUD_POINT_TOPK,
                        "cloud_knn": CLOUD_KNN,
                        "cloud_egnn_layers": CLOUD_EGNN_LAYERS,
                        "planner_type": "none",
                        "trajectory_decoder": "joint_local_increment_residual_diffusion_ddim",
                        "trajectory_diffusion_target": "(delta_ground_truth-delta_hierarchical_coarse)/delta_tau",
                        "trajectory_diffusion_frame": "previous_coarse_backbone_local_frame",
                        "trajectory_reconstruction": "source_anchor_plus_cumulative_denoised_increments",
                        "trajectory_diffusion_schedule": "cosine",
                        "trajectory_diffusion_prediction": "epsilon",
                        "trajectory_diffusion_sampler": "ddim_eta0",
                        "cloud_token_cross_attention": False,
                        "latent_loss_weight": LATENT_LOSS_WEIGHT,
                        "physics_loss_weight": PHYSICS_LOSS_WEIGHT,
                        "distogram_loss_weight": DISTOGRAM_LOSS_WEIGHT,
                        "trajectory_backbone_geometry_weight": TRAJECTORY_BACKBONE_GEOMETRY_WEIGHT,
                        "trajectory_backbone_smooth_weight": TRAJECTORY_BACKBONE_SMOOTH_WEIGHT,
                        "trajectory_clash_weight": TRAJECTORY_CLASH_WEIGHT,
                        "trajectory_velocity_weight": TRAJECTORY_VELOCITY_WEIGHT,
                        "trajectory_acceleration_weight": TRAJECTORY_ACCELERATION_WEIGHT,
                        "trajectory_bond_length_weight": TRAJECTORY_BOND_LENGTH_WEIGHT,
                        "trajectory_vdw_repulsion_weight": TRAJECTORY_VDW_REPULSION_WEIGHT,
                        "trajectory_hbond_weight": TRAJECTORY_HBOND_WEIGHT,
                        "trajectory_hydrophobic_weight": TRAJECTORY_HYDROPHOBIC_WEIGHT,
                        "protein_backbone_project": bool(PROTEIN_BACKBONE_PROJECT),
                        "protein_backbone_project_mode": PROTEIN_BACKBONE_PROJECT_MODE,
                        "protein_backbone_project_iters": PROTEIN_BACKBONE_PROJECT_ITERS,
                        "protein_backbone_project_strength": PROTEIN_BACKBONE_PROJECT_STRENGTH,
                        "protein_backbone_force_sequential_chain": bool(PROTEIN_BACKBONE_FORCE_SEQUENTIAL_CHAIN),
                        "protein_backbone_project_each_diffusion_step": bool(PROTEIN_BACKBONE_PROJECT_EACH_DIFFUSION_STEP),
                        "diffusion_absolute_anchor_loss_weight": DIFFUSION_ABSOLUTE_ANCHOR_LOSS_WEIGHT,
                        "diffusion_increment_x0_loss_weight": DIFFUSION_INCREMENT_X0_LOSS_WEIGHT,
                        "diffusion_absolute_anchor_horizon_gamma": DIFFUSION_ABSOLUTE_ANCHOR_HORIZON_GAMMA,
                        "diffusion_increment_noise_loss_weight": DIFFUSION_INCREMENT_NOISE_LOSS_WEIGHT,
                        "rollout_loss_weight": ROLLOUT_LOSS_WEIGHT,
                        "rollout_final_loss_weight": ROLLOUT_FINAL_LOSS_WEIGHT,
                        "rollout_velocity_loss_weight": ROLLOUT_VELOCITY_LOSS_WEIGHT,
                        "rollout_center_loss_weight": ROLLOUT_CENTER_LOSS_WEIGHT,
                        "batch_size": BATCH_SIZE,
                        "per_rank_batch_size": per_rank_batch_size,
                        "grad_accum_steps": grad_accum_steps,
                        "effective_global_batch_size": effective_global_batch_size,
                        "ddp_find_unused_parameters": bool(DDP_FIND_UNUSED_PARAMETERS),
                        "traj_eval_every_epochs": TRAJ_EVAL_EVERY_EPOCHS,
                        "traj_eval_max_samples": TRAJ_EVAL_MAX_SAMPLES,
                        "traj_eval_seed": TRAJ_EVAL_SEED,
                        "traj_eval_postprocess_pdb": bool(TRAJ_EVAL_POSTPROCESS_PDB),
                        "early_stopping_enabled": bool(EARLY_STOPPING_ENABLED),
                        "early_stopping_metric": EARLY_STOPPING_METRIC,
                        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                        "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
                        "train_save_final_sample": bool(TRAIN_SAVE_FINAL_SAMPLE),
                        "traj_level": TRAJ_LEVEL,
                        "traj_source_frame": TRAJ_SOURCE_FRAME,
                        "target_start_frame": TRAJ_TARGET_START_FRAME,
                        "traj_target_frame_stride": TRAJ_TARGET_FRAME_STRIDE,
                        "traj_observed_frame_stride": TRAJ_OBSERVED_FRAME_STRIDE,
                        "traj_random_window": bool(TRAJ_RANDOM_WINDOW),
                        "traj_random_stride": bool(TRAJ_RANDOM_STRIDE),
                        "traj_min_frame_stride": TRAJ_MIN_FRAME_STRIDE,
                        "traj_max_frame_stride": TRAJ_MAX_FRAME_STRIDE,
                        "traj_random_windows_per_traj": TRAJ_RANDOM_WINDOWS_PER_TRAJ,
                        "traj_static_context_prob": TRAJ_STATIC_CONTEXT_PROB,
                        "traj_force_final_frame": bool(TRAJ_FORCE_FINAL_FRAME),
                        "traj_window_stride": TRAJ_WINDOW_STRIDE,
                        "lr": _current_lr(opt),
                        "lr_scheduler": LR_SCHEDULER,
                        "lr_plateau_factor": LR_PLATEAU_FACTOR,
                        "lr_plateau_patience": LR_PLATEAU_PATIENCE,
                        "lr_plateau_threshold": LR_PLATEAU_THRESHOLD,
                        "lr_plateau_cooldown": LR_PLATEAU_COOLDOWN,
                        "optimizer": opt.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    },
                    ckpt_path,
                )
                _log(f"stage=save_ckpt done path={ckpt_path}")
            if distributed:
                dist.barrier()

            early_stop_requested = False
            if _should_eval_epoch(ep):
                early_stopping_values = []
                for split_name, eval_dl in eval_loaders:
                    split_eval_seed = int(TRAJ_EVAL_SEED) + int(ep) * 100000 + (0 if split_name.startswith("val") else 50000)
                    if bool(TRAJ_EVAL_TEACHER_FORCED):
                        tf_local = evaluate_teacher_forced_loss(
                            _raw_model(model),
                            eval_dl,
                            device,
                            eval_seed=split_eval_seed,
                        )
                        tf_metrics = _mean_metric_sums(
                            tf_local,
                            [
                                "loss", "diffusion", "latent", "physics", "rollout", "noise",
                                "increment_x0", "absolute_anchor", "recon", "final",
                            ],
                            device,
                        )
                        if _is_main_process() and tf_metrics is not None:
                            _log(
                                f"[{split_name} teacher_forced] samples={tf_metrics['samples']} "
                                f"loss={tf_metrics['loss']:.6f} diffusion={tf_metrics['diffusion']:.6f} "
                                f"latent={tf_metrics['latent']:.6f} physics={tf_metrics['physics']:.6f} "
                                f"rollout={tf_metrics['rollout']:.6f} noise={tf_metrics['noise']:.6f} "
                                f"increment_x0={tf_metrics['increment_x0']:.6f} "
                                f"absolute_anchor={tf_metrics['absolute_anchor']:.6f} "
                                f"recon={tf_metrics['recon']:.6f} "
                                f"final={tf_metrics['final']:.6f}"
                            )
                    if _is_main_process():
                        _log(
                            f"stage={split_name}_eval_start epoch={ep}/{NUM_EPOCHS} "
                            f"eval_every_epochs={TRAJ_EVAL_EVERY_EPOCHS} sample_steps={TRAJ_EVAL_SAMPLE_STEPS} "
                            f"max_samples={TRAJ_EVAL_MAX_SAMPLES} seed={TRAJ_EVAL_SEED} "
                            f"postprocess_pdb={int(bool(TRAJ_EVAL_POSTPROCESS_PDB))} "
                            f"match_train_windows={int(bool(TRAJ_EVAL_MATCH_TRAIN_WINDOWS))} "
                            f"stride_sweep={','.join(str(x) for x in TRAJ_EVAL_STRIDE_SWEEP) or 'none'} "
                            f"distributed_eval={int(distributed)} local_batches={len(eval_dl)}"
                        )
                    local_metrics = evaluate_temporal_diffusion(
                        _raw_model(model),
                        eval_dl,
                        device,
                        log_fn=_log if _is_main_process() else None,
                        eval_seed=split_eval_seed,
                    )
                    metrics = _mean_metric_sums(
                        local_metrics,
                        [
                            "horizon_frames",
                            "target_frame_stride",
                            "pocket_final_rmse_A",
                            "pocket_traj_rmse_A",
                            "protein_final_rmse_A",
                            "protein_traj_rmse_A",
                            "hold_pocket_final_rmse_A",
                            "hold_pocket_traj_rmse_A",
                            "hold_protein_final_rmse_A",
                            "hold_protein_traj_rmse_A",
                            "post_pocket_final_rmse_A",
                            "post_protein_final_rmse_A",
                            "md_speed_rmse_A_per_ps",
                            "md_msd_rmse_A2",
                            "md_vacf_mae_A2_per_ps2",
                            "md_kinetic_relative_mae",
                            "md_energy_drift_error_relative",
                            "md_bond_length_mae_A",
                            "md_interface_rdf_l1",
                        ],
                        device,
                    )
                    if metrics is not None and split_name.startswith("val"):
                        if EARLY_STOPPING_METRIC not in metrics:
                            raise ValueError(
                                f"EARLY_STOPPING_METRIC={EARLY_STOPPING_METRIC!r} is not available; "
                                f"available metrics={sorted(metrics)}"
                            )
                        early_stopping_values.append(float(metrics[EARLY_STOPPING_METRIC]))
                    if _is_main_process() and metrics is not None:
                        post_detail = ""
                        if bool(TRAJ_EVAL_POSTPROCESS_PDB):
                            post_detail = (
                                f" post_pocket_final_rmse={metrics['post_pocket_final_rmse_A']:.4f}A"
                                f" post_protein_final_rmse={metrics['post_protein_final_rmse_A']:.4f}A"
                            )
                        _log(
                            f"[{split_name} temporal diffusion] samples={metrics['samples']} "
                            f"avg_horizon={metrics['horizon_frames']:.2f} "
                            f"avg_target_stride={metrics['target_frame_stride']:.2f} "
                            f"pocket_final_rmse={metrics['pocket_final_rmse_A']:.4f}A "
                            f"pocket_traj_rmse={metrics['pocket_traj_rmse_A']:.4f}A "
                            f"protein_final_rmse={metrics['protein_final_rmse_A']:.4f}A "
                            f"protein_traj_rmse={metrics['protein_traj_rmse_A']:.4f}A "
                            f"hold_pocket_final_rmse={metrics['hold_pocket_final_rmse_A']:.4f}A "
                            f"hold_pocket_traj_rmse={metrics['hold_pocket_traj_rmse_A']:.4f}A "
                            f"hold_protein_final_rmse={metrics['hold_protein_final_rmse_A']:.4f}A "
                            f"hold_protein_traj_rmse={metrics['hold_protein_traj_rmse_A']:.4f}A"
                            f" md_speed_rmse={metrics['md_speed_rmse_A_per_ps']:.4f}A/ps"
                            f" md_msd_rmse={metrics['md_msd_rmse_A2']:.4f}A2"
                            f" md_vacf_mae={metrics['md_vacf_mae_A2_per_ps2']:.4f}A2/ps2"
                            f" md_energy_drift={metrics['md_energy_drift_error_relative']:.4f}"
                            f" md_bond_mae={metrics['md_bond_length_mae_A']:.4f}A"
                            f" md_rdf_l1={metrics['md_interface_rdf_l1']:.4f}"
                            f"{post_detail}"
                        )
                    if _is_main_process():
                        _log(f"stage={split_name}_eval_done epoch={ep}/{NUM_EPOCHS}")

                if bool(EARLY_STOPPING_ENABLED):
                    if early_stopping_values:
                        current_metric = float(sum(early_stopping_values) / len(early_stopping_values))
                        improved = current_metric < (
                            early_stopping_best - max(0.0, float(EARLY_STOPPING_MIN_DELTA))
                        )
                        if improved:
                            early_stopping_best = current_metric
                            early_stopping_bad_evals = 0
                            if _is_main_process():
                                best_path = CKPT_ROOT / "ckpt_best.pt"
                                shutil.copy2(CKPT_ROOT / f"ckpt_epoch{ep}.pt", best_path)
                                _log(
                                    f"stage=early_stopping_improved epoch={ep}/{NUM_EPOCHS} "
                                    f"metric={EARLY_STOPPING_METRIC} value={current_metric:.6f} "
                                    f"best={early_stopping_best:.6f} checkpoint={best_path}"
                                )
                        else:
                            early_stopping_bad_evals += 1
                            if _is_main_process():
                                _log(
                                    f"stage=early_stopping_no_improvement epoch={ep}/{NUM_EPOCHS} "
                                    f"metric={EARLY_STOPPING_METRIC} value={current_metric:.6f} "
                                    f"best={early_stopping_best:.6f} "
                                    f"bad_evals={early_stopping_bad_evals}/{max(1, int(EARLY_STOPPING_PATIENCE))}"
                                )
                        early_stop_requested = (
                            early_stopping_bad_evals >= max(1, int(EARLY_STOPPING_PATIENCE))
                        )
                    elif _is_main_process():
                        _log("stage=early_stopping_skipped reason=no_validation_metrics")
            if distributed:
                stop_tensor = torch.tensor(
                    int(early_stop_requested),
                    device=device,
                    dtype=torch.int32,
                )
                dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
                early_stop_requested = bool(stop_tensor.item())
                dist.barrier()

            if early_stop_requested:
                _log(
                    f"stage=early_stopping_stop epoch={ep}/{NUM_EPOCHS} "
                    f"metric={EARLY_STOPPING_METRIC} best={early_stopping_best:.6f} "
                    f"patience_evals={max(1, int(EARLY_STOPPING_PATIENCE))}"
                )
                break

            if _is_main_process() and bool(TRAIN_SAVE_FINAL_SAMPLE) and int(ep) == int(NUM_EPOCHS):
                save_final_train_sample(
                    _raw_model(model),
                    last_train_batch_for_sample,
                    ep,
                    device,
                    log_fn=_log,
                )
    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    train()
