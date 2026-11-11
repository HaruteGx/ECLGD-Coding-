# ECMDGNN

ECMDGNN 根据蛋白–配体复合物的历史分子动力学轨迹，预测未来的蛋白骨架和原子坐标。模型融合原子图、蛋白骨架和双通道体素云，用 LSTM 生成粗轨迹，再通过扩散模型修正相邻帧的运动增量。

## 环境准备

在项目根目录运行：

```sh
conda env create -f environment/environment.yml
conda activate ecmdgnn
python -m pip install -r environment/requirements.txt
```

另外安装与显卡驱动兼容的 CUDA 版 PyTorch。训练和采样需要 CUDA。安装后可运行以下命令检查，预期输出为 `True`：

```sh
python -c "import torch; print(torch.cuda.is_available())"
```

更多环境说明见 [environment/INSTALL.md](environment/INSTALL.md)。

## 运行流程

**1. 准备数据。** 将轨迹 NPZ 放入默认目录 `npz dataset/`。输入需要 `coords`（形状 `[帧数, 原子数, 3]`，单位 Å）、`atom_names`、`resnames`、`resids`、`segids` 和 `ligand_resnames`。可选字段包括 `times`、`box`、`bonds` 和 `charges`。缺少 `times` 时使用帧编号。

**2. 构建并检查缓存。**

```sh
python main.py --build_cache
python main.py --check_cache
```

默认写入 `graph_cache/<NPZ 文件名去掉扩展名>/`，其中 `meta.pt` 保存原子、残基及拓扑信息，`frames.pt` 保存逐帧坐标、图结构、骨架和体素云。

**3. 训练。**

```sh
python main.py --train
```

训练按轨迹目录划分数据集，检查点默认位于 `outputs/checkpoints/`。如需续训，把下面的路径替换为实际检查点：

```sh
python main.py --train --resume_ckpt outputs/checkpoints/ckpt_epoch1.pt
```

**4. 采样。** 将 `TRAJ_NAME` 替换为实际缓存子目录名，并确认检查点文件存在。

```sh
python main.py --sample --traj TRAJ_NAME --ckpt outputs/checkpoints/ckpt_epoch1.pt --src_frame 3 --horizon 1 --n_samples 1
```

帧号从 0 开始；上例使用第 3 帧作为起点，导出第 4 帧。也可用 `--target_frame 4` 指定目标帧。默认在 `outputs/` 保存原始预测轨迹 `.pt` 和经过结构后处理的目标帧 `.pdb`，二者的坐标可能不同。

## 配置与目录

- `config.py`：集中配置路径、训练窗口、模型和采样参数。可通过环境变量覆盖 `DATA_ROOT`、`GRAPH_ROOT`、`OUT_ROOT`、`BATCH_SIZE`、`OBSERVED_FRAMES`、`FUTURE_FRAMES` 等。
- `preprocess/`：构建、检查和打包轨迹缓存。
- `data/`：读取历史与未来帧，计算速度，整理成带掩码的 batch。
- `models/`：图、骨架和云编码，LSTM 粗预测，增量扩散与联合电子密度。
- `train.py`、`sample.py`：训练与评估、检查点加载与预测导出。
- `tools/`、`docs/`：诊断脚本、结构转换工具及模型设计说明。

默认训练使用 **4 帧历史预测 1 帧**。单帧模型预测更远时，会把生成坐标、速度和重建体素云反馈到历史窗口；单步指标不能代表长程生成质量。联合电子密度是模型预测的连续场，和输入体素云是不同表示；可选 QM 标签仅用于训练监督。
