# 使用 LeRobot Force VLA 数据训练 OpenVLA-OFT

本文档记录当前仓库中使用 LeRobot 格式数据集训练 / 微调 OpenVLA-OFT 的具体方案、代码修改和复现命令。

## 1. 任务目标

当前目标是使用 LeRobot 格式数据集对 OpenVLA-OFT 进行训练 / 微调。

- 项目仓库路径：`/root/autodl-tmp/openvla-oft`
- LeRobot 数据集根目录：`/root/autodl-tmp/force_vla_data/data_lerobot`
- 当前数据集名称：`flexiv_pump_1bottle_inputForce`
- 数据集实际目录：`/root/autodl-tmp/force_vla_data/data_lerobot/flexiv_pump_1bottle_inputForce`
- 预训练权重：`moojink/openvla-7b-oft-finetuned-libero-spatial`
- 训练时使用 `observation.state` 的前 7 维作为 proprio/state 输入，其余维度暂不使用。

## 2. 当前仓库结构说明

OpenVLA-OFT 相关代码主要位于以下目录：

- `vla-scripts/`
  - 训练、部署、权重转换等脚本。
  - 当前训练入口脚本是 `vla-scripts/finetune.py`。
- `prismatic/vla/`
  - VLA 训练常量、action tokenizer、数据集封装等。
  - `prismatic/vla/constants.py` 定义 `ACTION_DIM`、`PROPRIO_DIM`、`NUM_ACTIONS_CHUNK` 等常量。
  - `prismatic/vla/datasets/datasets.py` 定义 RLDS 数据集包装逻辑；当前已新增 `LeRobotDataset`。
- `prismatic/models/`
  - action head、projector、VLA/VLM 模型组件等。
  - `prismatic/models/action_heads.py` 定义 L1 regression / diffusion action head。
  - `prismatic/models/projectors.py` 定义 `ProprioProjector`、`NoisyActionProjector`。
- `prismatic/extern/hf/`
  - Hugging Face 兼容的 OpenVLA config / model / processor。
- `experiments/robot/`
  - LIBERO、ALOHA 等机器人评估和推理工具。
- `prismatic/conf/`
  - 原始 VLA 训练配置注册文件。
  - `prismatic/conf/vla.py` 包含一些 OpenVLA 预训练 / 微调配置，但当前 OFT 微调主要通过 `vla-scripts/finetune.py` 的 `FinetuneConfig` 命令行参数控制。

训练入口：

```bash
vla-scripts/finetune.py
```

核心配置类：

```python
# vla-scripts/finetune.py
@dataclass
class FinetuneConfig:
    ...
```

## 3. 数据集格式分析

### 3.1 当前 LeRobot 数据集字段

当前数据集为 LeRobot v2 风格目录：

```text
/root/autodl-tmp/force_vla_data/data_lerobot/flexiv_pump_1bottle_inputForce
├── data/chunk-000/episode_000000.parquet
├── data/chunk-000/episode_000001.parquet
├── ...
└── meta
    ├── info.json
    ├── tasks.jsonl
    ├── episodes.jsonl
    └── episodes_stats.jsonl
```

`meta/info.json` 中关键字段：

- `total_episodes`: 50
- `total_frames`: 17563
- `splits.train`: `0:40`
- `splits.test`: `40:50`
- `data_path`: `data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet`

已检查到的数据字段：

- `action`
  - dtype: `float64`
  - shape: `[7]`
  - 用作训练目标 action。
- `observation.state`
  - dtype: `float64`
  - shape: `[13]`
  - 训练时只取前 7 维：`observation.state[:7]`。
- `observation.image`
  - dtype: image
  - shape: `[480, 640, 3]`
  - 作为第三人称主视角图像。
- `observation.wrist_image`
  - dtype: image
  - shape: `[480, 640, 3]`
  - 当 `--num_images_in_input 2` 时作为 wrist 图像输入。
- `task_index`
  - 用于从 `meta/tasks.jsonl` 查找语言指令。

当前任务文本为：

```text
Press the pump dispenser on the bottle all the way down.
```

### 3.2 OpenVLA-OFT 原始训练代码期望格式

原始 `vla-scripts/finetune.py` 默认使用 RLDS / TFDS 数据集，数据由 `RLDSDataset` 和 `RLDSBatchTransform` 转换成训练 batch。

OpenVLA-OFT 训练过程期望每个样本最终包含：

- `pixel_values`
  - 主图像经 processor image transform 后的 tensor。
- `pixel_values_wrist`
  - 可选，wrist 图像经 image transform 后的 tensor。
- `input_ids`
  - 文本 prompt + action token 的 token ids。
- `labels`
  - 与 `input_ids` 对齐，只对 action token 计算 loss，其余位置为 `IGNORE_INDEX=-100`。
- `actions`
  - 连续 action chunk，shape 约为 `[NUM_ACTIONS_CHUNK, ACTION_DIM]`。
- `proprio`
  - 可选，proprio/state 输入。
- `dataset_name`
  - 用于保存 dataset statistics 和推理时 unnormalize。

原始 RLDS 格式大致是：

```python
rlds_batch = {
    "observation": {
        "image_primary": ...,
        "image_wrist": ...,
        "proprio": ...,
    },
    "task": {
        "language_instruction": ...,
    },
    "action": ...,
    "dataset_name": ...,
}
```

### 3.3 LeRobot 到 OpenVLA-OFT 的适配方式

本次没有将 LeRobot 数据离线转换为 RLDS，而是在仓库内新增了直接读取 LeRobot parquet 的 `LeRobotDataset`。

适配关系：

| LeRobot 字段 | OpenVLA-OFT 训练字段 | 说明 |
| --- | --- | --- |
| `observation.image` | `pixel_values` | PIL 解码后使用 `processor.image_processor.apply_transform` |
| `observation.wrist_image` | `pixel_values_wrist` | `--num_images_in_input 2` 时启用 |
| `action` | `actions` | 组成长度为 `NUM_ACTIONS_CHUNK` 的 action chunk |
| `observation.state[:7]` | `proprio` | 只取前 7 维 |
| `task_index` + `meta/tasks.jsonl` | language instruction | 组成 OpenVLA prompt |

action 和 proprio 会基于训练 split 自动计算 `q01/q99`，并归一化到 `[-1, 1]`，以匹配 OpenVLA action tokenizer 和 OFT 训练逻辑。

## 4. 代码修改记录

### 4.1 `prismatic/vla/datasets/datasets.py`

新增 `LeRobotDataset`，位置：

```python
class LeRobotDataset(IterableDataset):
    ...
```

主要作用：

- 读取 LeRobot `meta/info.json`、`meta/tasks.jsonl`。
- 根据 `splits.train` / `splits.test` 找到对应 episode parquet。
- 读取 image、wrist image、action、state、task_index。
- 解码图像 bytes。
- 构造 OpenVLA prompt。
- 构造 action chunk。
- 计算 action/proprio statistics。
- 将 action/proprio 按 q01/q99 归一化到 `[-1, 1]`。

关键代码片段：

```python
def _compute_dataset_statistics(self) -> Dict[str, Any]:
    import pyarrow.parquet as pq

    actions, proprio = [], []
    for episode_file in self.episode_files:
        table = pq.read_table(episode_file, columns=[self.action_key, self.state_key])
        rows = table.to_pylist()
        actions.extend(row[self.action_key] for row in rows)
        proprio.extend(row[self.state_key][: self.state_dim] for row in rows)
```

这里的 `row[self.state_key][: self.state_dim]` 就是 state 只取前 7 维的位置。默认 `self.state_dim=7`。

实际样本构造时也只取前 7 维：

```python
state = np.asarray(row[self.state_key][: self.state_dim], dtype=np.float32)
proprio = self._normalize_bounds_q99(state, stats["proprio"])
```

action chunk 构造：

```python
def _build_action_chunk(self, actions: np.ndarray, index: int) -> np.ndarray:
    end = index + NUM_ACTIONS_CHUNK
    chunk = actions[index:end]
    if chunk.shape[0] < NUM_ACTIONS_CHUNK:
        pad = np.repeat(actions[-1][None], NUM_ACTIONS_CHUNK - chunk.shape[0], axis=0)
        chunk = np.concatenate([chunk, pad], axis=0)
    return chunk
```

图像解码：

```python
@staticmethod
def _decode_image(value: Any) -> Image.Image:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        value = value.get("path")
    return Image.open(value).convert("RGB")
```

prompt 和 action token 构造：

```python
conversation = [
    {"from": "human", "value": f"What action should the robot take to {lang}?"},
    {"from": "gpt", "value": action_chunk_string},
]
```

### 4.2 `prismatic/vla/datasets/__init__.py`

导出新增的数据集类：

```python
from .datasets import DummyDataset, EpisodicRLDSDataset, LeRobotDataset, RLDSBatchTransform, RLDSDataset
```

### 4.3 `vla-scripts/finetune.py`

新增导入：

```python
from prismatic.vla.datasets import LeRobotDataset, RLDSBatchTransform, RLDSDataset
```

在 `FinetuneConfig` 中新增 LeRobot 相关参数：

```python
dataset_format: str = "rlds"
proprio_dim: Optional[int] = None
lerobot_state_dim: int = 7
lerobot_image_key: str = "observation.image"
lerobot_wrist_image_key: str = "observation.wrist_image"
lerobot_state_key: str = "observation.state"
lerobot_action_key: str = "action"
```

新增 proprio 维度选择逻辑：

```python
if cfg.dataset_format == "lerobot" and cfg.proprio_dim is None:
    effective_proprio_dim = cfg.lerobot_state_dim
else:
    effective_proprio_dim = cfg.proprio_dim if cfg.proprio_dim is not None else PROPRIO_DIM
```

创建 `ProprioProjector` 时使用 `effective_proprio_dim`：

```python
proprio_projector = init_module(
    ProprioProjector,
    "proprio_projector",
    cfg,
    device_id,
    {"llm_dim": vla.module.llm_dim, "proprio_dim": effective_proprio_dim},
)
```

新增 `--dataset_format lerobot` 分支：

```python
elif cfg.dataset_format == "lerobot":
    train_dataset = LeRobotDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        train=True,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
        state_dim=cfg.lerobot_state_dim,
        image_key=cfg.lerobot_image_key,
        wrist_image_key=cfg.lerobot_wrist_image_key,
        state_key=cfg.lerobot_state_key,
        action_key=cfg.lerobot_action_key,
    )
```

修复 proprio 没有搬到 GPU 的问题：

```python
proprio=batch["proprio"].to(torch.bfloat16).to(device_id) if use_proprio else None
```

该修改出现在训练 forward 和 diffusion sampling forward 两处。

### 4.4 `pyproject.toml`

新增依赖：

```toml
"pyarrow",
```

原因：LeRobot v2 数据以 parquet 存储，`LeRobotDataset` 使用 `pyarrow.parquet` 读取。

## 5. 训练配置说明

### 5.1 数据路径

使用如下参数指定：

```bash
--dataset_format lerobot
--data_root_dir /root/autodl-tmp/force_vla_data/data_lerobot
--dataset_name flexiv_pump_1bottle_inputForce
```

`LeRobotDataset` 会优先查找：

```text
/root/autodl-tmp/force_vla_data/data_lerobot/flexiv_pump_1bottle_inputForce
```

如果该目录不存在，则会把 `data_root_dir` 本身当作 LeRobot repo 目录。

### 5.2 预训练权重

使用：

```bash
--vla_path moojink/openvla-7b-oft-finetuned-libero-spatial
```

训练脚本会通过 Hugging Face Hub 下载该模型，然后从本地缓存路径加载。

注意：当前 `finetune.py` 会用该 HF repo 初始化 VLA 主体模型，然后重新初始化本次训练所需的 `action_head` 和 `proprio_projector`。由于当前 proprio 输入是 7 维，而 LIBERO checkpoint 的 proprio projector 通常是 8 维，因此不应直接复用 LIBERO 的 proprio projector。

### 5.3 state / proprio 维度

当前 LeRobot state 是 13 维，但只取前 7 维：

```bash
--lerobot_state_dim 7
--proprio_dim 7
--use_proprio True
```

如果后续想使用更多 state 维度，需要同步调整：

- `--lerobot_state_dim`
- `--proprio_dim`
- 训练 / 推理时的 proprio 输入维度
- 可能还要重新训练 `proprio_projector`

### 5.4 图像输入数量

当前建议使用主图像 + wrist 图像：

```bash
--num_images_in_input 2
```

这会读取：

- `observation.image`
- `observation.wrist_image`

如果显存不足，可以先改成只用主图：

```bash
--num_images_in_input 1
```

### 5.5 batch size、learning rate、训练步数、保存路径

关键参数均通过 `vla-scripts/finetune.py` 的命令行参数传入：

```bash
--batch_size 1
--grad_accumulation_steps 8
--learning_rate 5e-4
--num_steps_before_decay 10000
--max_steps 20000
--save_freq 1000
--run_root_dir /root/autodl-tmp/openvla-oft/runs_force_lerobot
```

说明：

- `batch_size` 是每张 GPU 的 batch size。
- `grad_accumulation_steps` 用于模拟更大的有效 batch。
- 有效 batch size 约为：

```text
batch_size * grad_accumulation_steps * GPU 数量
```

当前参数偏向先跑通流程，不一定是最终最优训练参数。后续可调整：

- 显存充足：提高 `--batch_size`。
- loss 不稳定：降低 `--learning_rate`，例如 `1e-4` 或 `2e-4`。
- 数据量较小：减少 `--max_steps`，或者更频繁保存 / 验证。
- 多卡训练：提高 `--nproc-per-node`，并相应关注有效 batch size。

## 6. 实际运行命令

进入项目目录：

```bash
cd /root/autodl-tmp/openvla-oft
```

激活训练环境：

```bash
conda activate openvla-oft
```

如果环境里没有 `pyarrow`：

```bash
pip install pyarrow
```

可选：关闭在线 wandb，先本地离线记录：

```bash
export WANDB_MODE=offline
```

单卡跑通版命令：~68GB

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path moojink/openvla-7b-oft-finetuned-libero-spatial \
  --dataset_format lerobot \
  --data_root_dir /root/autodl-tmp/force_vla_data/data_lerobot \
  --dataset_name flexiv_pump_1bottle_inputForce \
  --run_root_dir /root/autodl-tmp/openvla-oft/runs_force_lerobot \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 2 \
  --use_proprio True \
  --lerobot_state_dim 7 \
  --proprio_dim 7 \
  --batch_size 8 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 10000 \
  --max_steps 20000 \
  --save_freq 999999 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity 1559589961-northwestern-university \
  --wandb_project openvla-oft-force-vla \
  --run_id_note flexiv_lerobot_state7
```

如果想启用验证集：

```bash
  --use_val_set True \
  --val_freq 1000
```

如果显存不足，优先尝试：

```bash
--num_images_in_input 1
--batch_size 1
--grad_accumulation_steps 16
```

## 7. 常见问题与排查

### 7.1 数据路径找不到怎么办

确认目录存在：

```bash
ls -lah /root/autodl-tmp/force_vla_data/data_lerobot
ls -lah /root/autodl-tmp/force_vla_data/data_lerobot/flexiv_pump_1bottle_inputForce
ls -lah /root/autodl-tmp/force_vla_data/data_lerobot/flexiv_pump_1bottle_inputForce/meta
```

确认 parquet 文件存在：

```bash
find /root/autodl-tmp/force_vla_data/data_lerobot/flexiv_pump_1bottle_inputForce/data -name '*.parquet' | head
```

命令中的参数应匹配：

```bash
--data_root_dir /root/autodl-tmp/force_vla_data/data_lerobot
--dataset_name flexiv_pump_1bottle_inputForce
```

### 7.2 state 维度不匹配怎么办

当前数据：

- `observation.state`: 13 维
- 训练只取前 7 维

命令中必须保持：

```bash
--lerobot_state_dim 7
--proprio_dim 7
--use_proprio True
```

如果报 `ProprioProjector` 输入维度相关错误，优先检查这两个参数是否一致。

代码中也有保护逻辑：当 `dataset_format=lerobot` 时，`proprio_dim` 必须等于 `lerobot_state_dim`。

### 7.3 显存不足怎么办

如果报错类似：

```text
torch.cuda.OutOfMemoryError: CUDA out of memory
GPU 0 has a total capacity of 23.52 GiB
```

说明已经进入模型 forward，但单步显存不够。RTX 4090 24GB 上，`--batch_size 1` 仍可能因为 `num_images_in_input=2`、`lora_rank=32`、7B LLM 激活和 L1 action head 同时存在而 OOM。

优先降低：

```bash
--batch_size 1
```

如果仍然不够，减少图像输入：

```bash
--num_images_in_input 1
```

并用梯度累积维持有效 batch：

```bash
--grad_accumulation_steps 16
```

也可以降低保存时合并 LoRA 的开销：

```bash
--merge_lora_during_training False
```

注意：如果关闭训练中 LoRA merge，后续需要用 `vla-scripts/merge_lora_weights_and_save.py` 离线合并。

24GB 单卡更稳的启动参数建议：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path moojink/openvla-7b-oft-finetuned-libero-spatial \
  --dataset_format lerobot \
  --data_root_dir /root/autodl-fs/force_vla_data/data_lerobot \
  --dataset_name flexiv_pump_1bottle_inputForce \
  --run_root_dir /root/autodl-tmp/openvla-oft/runs_force_lerobot \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio True \
  --lerobot_state_dim 7 \
  --proprio_dim 7 \
  --batch_size 1 \
  --grad_accumulation_steps 16 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 10000 \
  --max_steps 20000 \
  --save_freq 999999999 \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training False \
  --image_aug True \
  --lora_rank 16 \
  --wandb_entity 1559589961 \
  --wandb_project openvla-oft-force-vla \
  --run_id_note flexiv_lerobot_state7_24gb_single_img_lora16
```

如果仍然 OOM，再继续降：

```bash
--lora_rank 8
```

`--grad_accumulation_steps` 只影响有效 batch size 和优化频率，不会显著降低单次 forward/backward 的显存；真正降显存的是减少图像数、降低 LoRA rank、减少可训练模块或换更大显存 GPU。

如果原命令里误用了：

```bash
--use_film True
```

也很容易触发 OOM。FiLM 会额外包装 vision backbone，把语言信息注入视觉特征，训练时显存会高于：

```bash
--use_film False
```

当前 24GB 单卡建议保持：

```bash
--use_film False
```

如果后续必须打开 FiLM，建议同时使用更保守配置：

```bash
--num_images_in_input 1
--lora_rank 8
--batch_size 1
--grad_accumulation_steps 16
```

并预期训练速度和显存压力都会更高。

### 7.4 Hugging Face 权重下载到哪里

`finetune.py` 对 HF Hub 模型调用：

```python
snapshot_download(repo_id=cfg.vla_path)
```

默认会下载到 Hugging Face cache，通常在：

```text
~/.cache/huggingface/hub
```

可以通过环境变量改变缓存位置：

```bash
export HF_HOME=/root/autodl-tmp/huggingface
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/huggingface/hub
```

### 7.5 如何确认训练真的读取到了目标 LeRobot 数据

先检查命令参数：

```bash
--dataset_format lerobot
--data_root_dir /root/autodl-tmp/force_vla_data/data_lerobot
--dataset_name flexiv_pump_1bottle_inputForce
```

训练启动后，`LeRobotDataset` 会读取：

```text
/root/autodl-tmp/force_vla_data/data_lerobot/flexiv_pump_1bottle_inputForce/meta/info.json
/root/autodl-tmp/force_vla_data/data_lerobot/flexiv_pump_1bottle_inputForce/meta/tasks.jsonl
/root/autodl-tmp/force_vla_data/data_lerobot/flexiv_pump_1bottle_inputForce/data/chunk-000/*.parquet
```

也可以用下面命令独立确认数据字段：

```bash
python - <<'PY'
import json
from pathlib import Path
import pyarrow.parquet as pq

root = Path("/root/autodl-tmp/force_vla_data/data_lerobot/flexiv_pump_1bottle_inputForce")
info = json.load(open(root / "meta/info.json"))
print("splits:", info["splits"])
print("features:", list(info["features"]))

p = root / "data/chunk-000/episode_000000.parquet"
t = pq.read_table(p, columns=["action", "observation.state", "observation.image", "observation.wrist_image", "task_index"])
row = t.slice(0, 1).to_pylist()[0]
print("action_dim:", len(row["action"]))
print("state_dim:", len(row["observation.state"]))
print("state_first7:", row["observation.state"][:7])
print("task_index:", row["task_index"])
PY
```

预期输出应包含：

```text
action_dim: 7
state_dim: 13
state_first7: [...]
```

### 7.6 `wandb_telemetry_pb2` 导入错误怎么办

如果启动训练时报错：

```text
ImportError: cannot import name 'Imports' from 'wandb.proto.wandb_telemetry_pb2'
```

原因通常是 `wandb` 与 `protobuf` 版本组合不兼容。当前环境中已验证可用的组合是：

```text
wandb==0.16.6
protobuf==3.20.3
```

修复命令：

```bash
conda activate openvla-oft
python -m pip install 'wandb==0.16.6' 'protobuf==3.20.3'
```

验证：

```bash
python - <<'PY'
import google.protobuf
import wandb
from wandb.proto.wandb_telemetry_pb2 import Imports
print("protobuf:", google.protobuf.__version__)
print("wandb:", wandb.__version__)
print("wandb telemetry Imports OK")
PY
```

预期输出：

```text
protobuf: 3.20.3
wandb: 0.16.6
wandb telemetry Imports OK
```

注意：TensorFlow 启动时可能打印 cuDNN/cuFFT/cuBLAS factory already registered 或 TensorRT warning，这些不是导致本次退出的根因。真正导致退出的是 `wandb` 的 `ImportError`。

### 7.7 W&B entity 404 怎么办

如果启动训练时报错：

```text
wandb: ERROR Error while calling W&B API: entity your-wandb-entity not found
wandb.errors.CommError: It appears that you do not have permission to access the requested resource.
```

说明命令里还保留了示例占位符：

```bash
--wandb_entity your-wandb-entity
```

当前服务器上 W&B 已登录账号显示为：

```text
1559589961
```

因此在线记录时应改成：

```bash
--wandb_entity 1559589961
```

如果暂时不需要上传到 W&B，推荐先离线跑：

```bash
export WANDB_MODE=offline
```

离线模式会把日志保存在本地 `wandb/` 目录，不会访问 W&B API。

## 8. 下次复现步骤

从重新打开服务器开始，可以按以下步骤操作。

### 8.1 进入项目目录

```bash
cd /root/autodl-tmp/openvla-oft
```

### 8.2 激活环境

```bash
conda activate openvla-oft
```

确认 Python 能导入 torch 和 pyarrow：

```bash
python - <<'PY'
import torch
import pyarrow
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("pyarrow ok")
PY
```

如果 `pyarrow` 缺失：

```bash
pip install pyarrow
```

### 8.3 确认数据存在

```bash
ls -lah /root/autodl-tmp/force_vla_data/data_lerobot/flexiv_pump_1bottle_inputForce/meta
find /root/autodl-tmp/force_vla_data/data_lerobot/flexiv_pump_1bottle_inputForce/data -name '*.parquet' | head
```

### 8.4 可选：设置缓存和 wandb

```bash
export HF_HOME=/root/autodl-tmp/huggingface
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/huggingface/hub
export WANDB_MODE=offline
```

### 8.5 启动训练

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path moojink/openvla-7b-oft-finetuned-libero-spatial \
  --dataset_format lerobot \
  --data_root_dir /root/autodl-tmp/force_vla_data/data_lerobot \
  --dataset_name flexiv_pump_1bottle_inputForce \
  --run_root_dir /root/autodl-tmp/openvla-oft/runs_force_lerobot \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --lerobot_state_dim 7 \
  --proprio_dim 7 \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 10000 \
  --max_steps 20000 \
  --save_freq 1000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity 1559589961 \
  --wandb_project openvla-oft-force-vla \
  --run_id_note flexiv_lerobot_state7
```

### 8.6 查看输出

训练输出目录位于：

```text
/root/autodl-tmp/openvla-oft/runs_force_lerobot
```

每次运行的具体目录由 `finetune.py` 的 `get_run_id()` 生成，通常包含：

- 基础模型名
- 数据集名
- batch size
- learning rate
- LoRA rank
- `run_id_note`

可以查看：

```bash
find /root/autodl-tmp/openvla-oft/runs_force_lerobot -maxdepth 2 -type f -name 'dataset_statistics.json' -print
find /root/autodl-tmp/openvla-oft/runs_force_lerobot -maxdepth 2 -type d -name 'lora_adapter' -print
```

## 9. 当前验证状态

已完成：

- 检查 LeRobot 数据实际字段、split 和维度。
- 确认训练 split 为 40 episodes / 13751 frames。
- 确认 test split 为 10 episodes / 3812 frames。
- 确认 `action` 为 7 维。
- 确认 `observation.state` 为 13 维。
- 确认图像字段为 `observation.image` 和 `observation.wrist_image`。
- 完成代码静态语法检查：

```bash
python -m py_compile prismatic/vla/datasets/datasets.py vla-scripts/finetune.py prismatic/vla/datasets/__init__.py
```

待确认：

- 在完整 OpenVLA-OFT 训练环境中执行一次真实 training smoke test。
- 最终适合该数据集的 learning rate、max steps、batch size 等超参数。
