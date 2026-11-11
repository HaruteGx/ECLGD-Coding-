# Environment Install

本目录提供项目环境的最小安装说明。核心原则是：Python 和轻量依赖可以固定，PyTorch 需要按当前 GPU 驱动/CUDA 运行时安装，尤其是 `gpu_5090` 节点。

## 1. 创建 Conda 环境

```bash
conda env create -f environment/environment.yml
conda activate ecmdgnn
```

如果环境已存在：

```bash
conda env update -f environment/environment.yml --prune
conda activate ecmdgnn
```

## 2. 安装项目 Python 依赖

```bash
python -m pip install -r environment/requirements.txt
```

## 3. 安装 GPU PyTorch

本项目训练和采样需要 CUDA 版 PyTorch。由于不同集群驱动差异较大，推荐优先使用管理员提供的 PyTorch CUDA 12.x 环境。

如果需要自行安装，请根据节点驱动和 CUDA 版本安装对应的 PyTorch wheel/conda 包。安装后必须确认：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

期望结果：

- `torch.cuda.is_available()` 为 `True`。
- `torch.version.cuda` 与集群环境兼容。
- 能正确识别 GPU。

## 4. 运行 smoke test

在项目根目录执行：

```bash
SMOKE_TEST_GRAPHS=10 \
NUM_EPOCHS=1 \
BATCH_SIZE=2 \
GRAD_ACCUM_STEPS=1 \
NUM_WORKERS=0 \
DIFFUSION_STEPS=10 \
SAMPLE_DIFFUSION_STEPS=10 \
TRAJ_EVAL_SAMPLE_STEPS=3 \
sbatch -p gpu_5090 --gpus=2 0runsmoke.sh
```

## 5. 常见问题

- `ModuleNotFoundError: torch`：PyTorch 没装，或者当前 shell 没激活正确环境。
- `torch.cuda.is_available() == False`：安装了 CPU 版 PyTorch，或节点没有分配 GPU。
- `no npz found under ...`：`DATA_ROOT` 指向错误。
- `checkpoint not found`：采样时 `CKPT_PATH` 错误，或 `outputs/checkpoints` 下还没有训练结果。
- NCCL timeout：可能是某个 rank 卡住、坏 cache、文件系统慢或通信问题；先看 SLURM 日志和 `outputs/bad_graph_cache.txt`。
