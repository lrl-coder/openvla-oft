"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

import json
import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights

@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, current_action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"]

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")

        # Get future action chunk
        future_actions = rlds_batch["action"][1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))

        # Get action chunk string
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return_dict = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=actions)

        # Add additional inputs
        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            return_dict["proprio"] = proprio

        return return_dict


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        if "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=NUM_ACTIONS_CHUNK-1,      # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)


class LeRobotDataset(IterableDataset):
    """
    Iterable PyTorch dataset for LeRobot v2 parquet datasets.

    It emits the same fields as RLDSBatchTransform/RLDSDataset:
    pixel_values, optional pixel_values_wrist, input_ids, labels, dataset_name,
    actions, and optional proprio.
    """

    def __init__(
        self,
        data_root_dir: Path,
        dataset_name: str,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        train: bool = True,
        use_wrist_image: bool = False,
        use_proprio: bool = False,
        state_dim: int = 7,
        image_key: str = "observation.image",
        wrist_image_key: str = "observation.wrist_image",
        state_key: str = "observation.state",
        action_key: str = "action",
    ) -> None:
        super().__init__()
        try:
            import pyarrow.parquet as pq  # noqa: F401
        except ImportError as e:
            raise ImportError("LeRobotDataset requires `pyarrow`; install it with `pip install pyarrow`.") from e

        self.data_root_dir = Path(data_root_dir)
        self.dataset_name = dataset_name
        self.repo_dir = self.data_root_dir / dataset_name
        if not self.repo_dir.exists():
            self.repo_dir = self.data_root_dir
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.train = train
        self.use_wrist_image = use_wrist_image
        self.use_proprio = use_proprio
        self.state_dim = state_dim
        self.image_key = image_key
        self.wrist_image_key = wrist_image_key
        self.state_key = state_key
        self.action_key = action_key

        self.info = self._load_json(self.repo_dir / "meta" / "info.json")
        self.tasks = self._load_tasks()
        self.episode_files = self._resolve_episode_files(train=train)
        self.dataset_statistics = {self.dataset_name: self._compute_dataset_statistics()}
        self.dataset_length = self.dataset_statistics[self.dataset_name]["num_transitions"]

    @staticmethod
    def _load_json(path: Path) -> dict:
        with open(path, "r") as f:
            return json.load(f)

    def _load_tasks(self) -> Dict[int, str]:
        tasks = {}
        task_path = self.repo_dir / "meta" / "tasks.jsonl"
        with open(task_path, "r") as f:
            for line in f:
                item = json.loads(line)
                tasks[int(item["task_index"])] = item["task"]
        return tasks

    @staticmethod
    def _parse_episode_range(range_str: str) -> Tuple[int, int]:
        start, end = range_str.split(":")
        return int(start), int(end)

    def _resolve_episode_files(self, train: bool) -> List[Path]:
        split_name = "train" if train else "test"
        split_range = self.info.get("splits", {}).get(split_name)
        if split_range is None and not train:
            split_range = self.info.get("splits", {}).get("val")
        if split_range is None:
            start, end = 0, int(self.info["total_episodes"])
        else:
            start, end = self._parse_episode_range(split_range)

        data_path = self.info["data_path"]
        paths = []
        chunks_size = int(self.info.get("chunks_size", 1000))
        for episode_index in range(start, end):
            episode_chunk = episode_index // chunks_size
            paths.append(self.repo_dir / data_path.format(episode_chunk=episode_chunk, episode_index=episode_index))
        return paths

    @staticmethod
    def _normalize_bounds_q99(values: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        scale = np.maximum(q99 - q01, 1e-6)
        values = 2.0 * (values - q01) / scale - 1.0
        return np.clip(values, -1.0, 1.0).astype(np.float32)

    def _compute_dataset_statistics(self) -> Dict[str, Any]:
        import pyarrow.parquet as pq

        actions, proprio = [], []
        for episode_file in self.episode_files:
            table = pq.read_table(episode_file, columns=[self.action_key, self.state_key])
            rows = table.to_pylist()
            actions.extend(row[self.action_key] for row in rows)
            proprio.extend(row[self.state_key][: self.state_dim] for row in rows)

        actions = np.asarray(actions, dtype=np.float32)
        proprio = np.asarray(proprio, dtype=np.float32)
        return {
            "action": {
                "mean": actions.mean(axis=0),
                "std": actions.std(axis=0) + 1e-6,
                "min": actions.min(axis=0),
                "max": actions.max(axis=0),
                "q01": np.quantile(actions, 0.01, axis=0),
                "q99": np.quantile(actions, 0.99, axis=0),
            },
            "proprio": {
                "mean": proprio.mean(axis=0),
                "std": proprio.std(axis=0) + 1e-6,
                "min": proprio.min(axis=0),
                "max": proprio.max(axis=0),
                "q01": np.quantile(proprio, 0.01, axis=0),
                "q99": np.quantile(proprio, 0.99, axis=0),
            },
            "num_transitions": int(actions.shape[0]),
            "num_trajectories": len(self.episode_files),
        }

    @staticmethod
    def _decode_image(value: Any) -> Image.Image:
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                return Image.open(BytesIO(value["bytes"])).convert("RGB")
            value = value.get("path")
        return Image.open(value).convert("RGB")

    def _build_action_chunk(self, actions: np.ndarray, index: int) -> np.ndarray:
        end = index + NUM_ACTIONS_CHUNK
        chunk = actions[index:end]
        if chunk.shape[0] < NUM_ACTIONS_CHUNK:
            pad = np.repeat(actions[-1][None], NUM_ACTIONS_CHUNK - chunk.shape[0], axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        return chunk

    def _format_example(self, row: Dict[str, Any], actions: np.ndarray, row_idx: int) -> Dict[str, Any]:
        stats = self.dataset_statistics[self.dataset_name]
        action_chunk = self._normalize_bounds_q99(self._build_action_chunk(actions, row_idx), stats["action"])
        state = np.asarray(row[self.state_key][: self.state_dim], dtype=np.float32)
        proprio = self._normalize_bounds_q99(state, stats["proprio"])

        img = self._decode_image(row[self.image_key])
        lang = self.tasks[int(row["task_index"])].lower()

        prompt_builder = self.prompt_builder_fn("openvla")
        current_action_string = self.action_tokenizer(action_chunk[0])
        future_actions_string = "".join(self.action_tokenizer(action_chunk[1:]))
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX

        return_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "dataset_name": self.dataset_name,
            "actions": action_chunk,
        }
        if self.use_wrist_image and self.wrist_image_key in row:
            img_wrist = self._decode_image(row[self.wrist_image_key])
            return_dict["pixel_values_wrist"] = self.image_transform(img_wrist)
        if self.use_proprio:
            return_dict["proprio"] = proprio
        return return_dict

    def _iter_once(self):
        import pyarrow.parquet as pq

        episode_files = list(self.episode_files)
        if self.train:
            random.shuffle(episode_files)

        for episode_file in episode_files:
            columns = [self.action_key, self.image_key, "task_index"]
            if self.use_wrist_image:
                columns.append(self.wrist_image_key)
            if self.use_proprio:
                columns.append(self.state_key)
            elif self.state_key not in columns:
                columns.append(self.state_key)
            table = pq.read_table(episode_file, columns=columns)
            rows = table.to_pylist()
            actions = np.asarray([row[self.action_key] for row in rows], dtype=np.float32)
            frame_indices = list(range(len(rows)))
            if self.train:
                random.shuffle(frame_indices)
            for row_idx in frame_indices:
                yield self._format_example(rows[row_idx], actions, row_idx)

    def __iter__(self):
        if self.train:
            while True:
                yield from self._iter_once()
        else:
            yield from self._iter_once()

    def __len__(self) -> int:
        return self.dataset_length
