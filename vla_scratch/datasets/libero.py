from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from vla_scratch.datasets.data_types import Observation, ActionChunk, DataSample


@dataclass(frozen=True)
class _IndexEntry:
    file_idx: int
    episode_key: str
    step_index: int


@dataclass(frozen=True)
class _DatasetInfo:
    path: Path
    prompt: str
    tokenised_prompt: torch.Tensor
    tokenised_prompt_mask: torch.Tensor


def _quat_wxyz_to_euler(quaternions: np.ndarray) -> np.ndarray:
    """Convert quaternions expressed as (w, x, y, z) into roll, pitch, yaw."""
    if quaternions.shape[-1] != 4:
        raise ValueError(f"Quaternion input must have size 4 on the last dimension, got {quaternions.shape[-1]}")

    q = np.asarray(quaternions, dtype=np.float64)
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp_clipped = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp_clipped)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    euler = np.stack((roll, pitch, yaw), axis=-1)
    return euler.astype(np.float32, copy=False)


def _aggregate_state_action_stats(dataset: LiberoDataset) -> tuple[list[tuple[str, np.ndarray]], list[tuple[str, np.ndarray]]]:
    """Collect flattened arrays for the requested histogram statistics."""
    grippers: list[np.ndarray] = []
    eef_positions: list[np.ndarray] = []
    eef_quaternions: list[np.ndarray] = []
    actions: list[np.ndarray] = []

    for info in dataset._datasets:
        with h5py.File(info.path, "r") as handle:
            data_group = handle["data"]
            for episode_key in data_group.keys():
                episode_group = data_group[episode_key]
                state_arr = np.asarray(episode_group["robot_states"], dtype=np.float32)
                if state_arr.shape[-1] < 9:
                    raise ValueError(
                        f"Expected robot_states to have at least 9 dims (got {state_arr.shape[-1]}) in episode {episode_key}"
                    )
                action_arr = np.asarray(episode_group["actions"], dtype=np.float32)
                grippers.append(state_arr[:, 0:2])
                eef_positions.append(state_arr[:, 2:5])
                eef_quaternions.append(state_arr[:, 5:9])
                actions.append(action_arr)

    if not grippers:
        raise RuntimeError("No state/action data collected for histogram visualisation.")

    grippers_cat = np.concatenate(grippers, axis=0)
    eef_pos_cat = np.concatenate(eef_positions, axis=0)
    eef_euler_cat = _quat_wxyz_to_euler(np.concatenate(eef_quaternions, axis=0))
    actions_cat = np.concatenate(actions, axis=0)

    state_series: list[tuple[str, np.ndarray]] = [
        ("gripper_0", grippers_cat[:, 0]),
        ("gripper_1", grippers_cat[:, 1]),
        ("eef_x", eef_pos_cat[:, 0]),
        ("eef_y", eef_pos_cat[:, 1]),
        ("eef_z", eef_pos_cat[:, 2]),
        ("eef_roll_rad", eef_euler_cat[:, 0]),
        ("eef_pitch_rad", eef_euler_cat[:, 1]),
        ("eef_yaw_rad", eef_euler_cat[:, 2]),
    ]

    if actions_cat.shape[-1] < 7:
        raise ValueError(f"Expected actions to have at least 7 dims (got {actions_cat.shape[-1]})")

    action_series: list[tuple[str, np.ndarray]] = [
        ("delta_eef_x", actions_cat[:, 0]),
        ("delta_eef_y", actions_cat[:, 1]),
        ("delta_eef_z", actions_cat[:, 2]),
        ("delta_axis_x", actions_cat[:, 3]),
        ("delta_axis_y", actions_cat[:, 4]),
        ("delta_axis_z", actions_cat[:, 5]),
        ("gripper_action", actions_cat[:, 6]),
    ]

    return state_series, action_series



class LiberoDataset:
    """PyTorch dataset for LIBERO demonstrations."""

    def __init__(
        self,
        root: str | Path,
        tokenizer,
        cameras: Sequence[str] | None = None,
        *,
        action_horizon: int = 1,
        image_dtype: torch.dtype = torch.float32,
        max_tokens: int = 256,
        profile: bool = False,
    ) -> None:
        super().__init__()
        self._src_h5_paths = self._resolve_sources(root)
        self._cameras = tuple(cameras) if cameras is not None else None
        self._tokenizer = tokenizer
        self._image_dtype = image_dtype
        self._max_tokens = max_tokens
        self._action_horizon = int(action_horizon)
        self._target_image_size = (224, 224)

        self._datasets: list[_DatasetInfo] = []
        self._indices: list[_IndexEntry] = []
        self._action_dim: int | None = None
        self._state_dim: int | None = None
        self._processed_image_shape: tuple[int, int, int] | None = None
        self._num_cameras: int = 0

        env_profile = str(os.environ.get("LIBERO_PROFILE", "")).strip().lower()
        self._profile_enabled = bool(profile or env_profile in {"1", "true", "yes", "on"})
        self._profile_totals: dict[tuple[str, ...], float] = {}
        self._profile_counts: dict[tuple[str, ...], int] = {}
        self._profile_stack: list[tuple[tuple[str, ...], float]] = []

        self._build_index()
        if not self._indices:
            raise RuntimeError("No samples found in LIBERO dataset.")
        self._inspect_structure()
        self._emit_profile(len(self._indices))

    def _resolve_sources(self, root: str | Path) -> list[Path]:
        path = Path(root)
        if not path.exists():
            raise FileNotFoundError(f"LIBERO dataset path not found: {path}")

        if path.is_file():
            return [path]

        candidates = sorted(
            {
                *path.rglob("*.hdf5"),
                *path.rglob("*.h5"),
            }
        )
        if not candidates:
            raise FileNotFoundError(f"No HDF5 files discovered under {path}")
        return candidates

    def _build_index(self) -> None:
        self._profile_push_event("build_index")
        flattened: list[_IndexEntry] = []
        for file_idx, dataset_path in enumerate(self._src_h5_paths):
            info, entries = self._index_single_file(
                dataset_path,
                file_idx,
            )
            self._datasets.append(info)
            flattened.extend(entries)
        self._indices = flattened
        self._profile_pop_event()

    def _inspect_structure(self) -> None:
        reference_entry = self._indices[0]
        dataset_info = self._datasets[reference_entry.file_idx]

        with h5py.File(dataset_info.path, "r") as handle:
            data_group = handle["data"]
            episode_group = data_group[reference_entry.episode_key]
            obs_group = episode_group["obs"]

            if self._cameras is None:
                cameras = [key for key in obs_group.keys() if key.endswith("_rgb")]
                if not cameras:
                    cameras = list(obs_group.keys())
                cameras.sort()
                self._cameras = tuple(cameras)
            else:
                missing = [camera for camera in self._cameras if camera not in obs_group]
                if missing:
                    raise KeyError(
                        f"Requested cameras {missing} not found in episode {reference_entry.episode_key}"
                    )

            if not self._cameras:
                raise ValueError("No camera streams available in episode")

            sample_camera = obs_group[self._cameras[0]]
            if sample_camera.ndim != 4:
                raise ValueError("Expected camera dataset to have shape (T, H, W, C)")
            _, height, width, channels = sample_camera.shape
            self._processed_image_shape = (int(channels), *self._target_image_size)
            self._num_cameras = len(self._cameras)

            state_dataset = episode_group["robot_states"]
            if state_dataset.ndim < 2:
                raise ValueError("Robot states dataset must be at least 2D")
            self._state_dim = int(state_dataset.shape[-1])

            action_dataset = episode_group["actions"]
            if action_dataset.ndim < 2:
                raise ValueError("Actions dataset must be at least 2D")
            self._action_dim = int(action_dataset.shape[-1])

    def _tokenize_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        call_kwargs = {
            "padding": "max_length",
            "truncation": True,
            "max_length": self._max_tokens,
            "return_tensors": "pt",
        }

        forward = getattr(self._tokenizer, "forward", None)
        if callable(forward):
            encoded = forward(prompt, **call_kwargs)
        else:
            callable_tokenizer = getattr(self._tokenizer, "__call__", None)
            if callable(callable_tokenizer):
                encoded = callable_tokenizer(prompt, **call_kwargs)
            else:
                tokens = self._tokenizer.encode(prompt)
                if len(tokens) > self._max_tokens:
                    tokens = tokens[: self._max_tokens]
                pad_len = self._max_tokens - len(tokens)
                mask = [True] * len(tokens) + [False] * pad_len
                tokens = tokens + [0] * pad_len
                token_tensor = torch.tensor(tokens, dtype=torch.long)
                mask_tensor = torch.tensor(mask, dtype=torch.bool)
                return token_tensor, mask_tensor

        tokens = encoded["input_ids"][0].to(dtype=torch.long)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            mask = torch.ones_like(tokens, dtype=torch.bool)
        else:
            mask = attention_mask[0].to(dtype=torch.bool)
        return tokens, mask

    def _load_images(self, episode_group: h5py.Group, step_index: int) -> torch.Tensor:
        assert self._cameras is not None
        images: list[torch.Tensor] = []
        obs_group = episode_group["obs"]
        for camera in self._cameras:
            frame_np = np.asarray(obs_group[camera][step_index], dtype=np.uint8)
            frame = torch.from_numpy(frame_np).permute(2, 0, 1).contiguous().to(torch.float32)
            frame = F.interpolate(
                frame.unsqueeze(0),
                size=self._target_image_size,
                mode="bilinear",
                align_corners=False,
            )[0]
            frame = frame.div_(255.0)
            frame = frame.sub_(0.5).div_(0.5)
            frame = frame.flip(dims=[1]).contiguous()
            images.append(frame.to(dtype=self._image_dtype))
        return torch.stack(images, dim=0)

    def _load_state(self, episode_group: h5py.Group, step_index: int) -> torch.Tensor:
        state = np.asarray(episode_group["robot_states"][step_index], dtype=np.float32)
        return torch.from_numpy(state)

    def _load_actions(self, episode_group: h5py.Group, step_index: int) -> torch.Tensor:
        horizon_stop = step_index + self._action_horizon
        raw_actions = np.asarray(episode_group["actions"][step_index:horizon_stop], dtype=np.float32)
        actions = torch.from_numpy(raw_actions)
        if actions.shape[0] != self._action_horizon:
            padded = torch.zeros(self._action_horizon, self._action_dim, dtype=torch.float32)
            padded[: actions.shape[0]] = actions
            actions = padded
        return actions

    def __len__(self) -> int:  # noqa: D401 - inherited behaviour
        return len(self._indices)

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self._indices):
            raise IndexError(index)
        entry = self._indices[index]
        dataset_info = self._datasets[entry.file_idx]

        with h5py.File(dataset_info.path, "r") as handle:
            episode_group = handle["data"][entry.episode_key]
            images = self._load_images(episode_group, entry.step_index)
            image_masks = torch.ones(self._num_cameras, 1, dtype=torch.bool)
            state = self._load_state(episode_group, entry.step_index)
            actions = self._load_actions(episode_group, entry.step_index)

        observation = Observation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=dataset_info.tokenised_prompt,
            tokenized_prompt_mask=dataset_info.tokenised_prompt_mask,
        )
        action = ActionChunk(actions=actions)
        data_sample = DataSample(observation=observation, action=action)
        return data_sample

    @property
    def action_dim(self) -> int:
        if self._action_dim is None:
            raise RuntimeError("Action dimension metadata is unavailable")
        return self._action_dim

    @property
    def state_dim(self) -> int:
        if self._state_dim is None:
            raise RuntimeError("State dimension metadata is unavailable")
        return self._state_dim

    @property
    def image_shape(self) -> tuple[int, int, int]:
        if self._processed_image_shape is None:
            raise RuntimeError("Image shape metadata is unavailable")
        return self._processed_image_shape

    def episode_keys(self) -> Iterable[str]:
        """Return the episode identifiers present in the dataset."""
        return tuple(
            f"{self._datasets[index.file_idx].path.name}:{index.episode_key}" for index in self._indices
        )

    def _profile_push_event(self, name: str) -> None:
        if not self._profile_enabled:
            return
        parent_path = self._profile_stack[-1][0] if self._profile_stack else ()
        path = (*parent_path, name)
        self._profile_stack.append((path, time.perf_counter()))

    def _profile_pop_event(self) -> None:
        if not self._profile_enabled:
            return
        if not self._profile_stack:
            return
        path, start = self._profile_stack.pop()
        duration = time.perf_counter() - start
        self._profile_totals[path] = self._profile_totals.get(path, 0.0) + duration
        self._profile_counts[path] = self._profile_counts.get(path, 0) + 1

    def _emit_profile(self, num_samples: int) -> None:
        if not self._profile_enabled or not self._profile_totals:
            return
        if self._profile_stack:
            self._profile_stack.clear()
        print("[LiberoDataset] Profiling summary")
        print(f"  samples: {num_samples}")
        tree: dict[str, dict[str, Any]] = {}
        for path, total in self._profile_totals.items():
            count = self._profile_counts.get(path, 0)
            node = tree
            for idx, name in enumerate(path):
                entry = node.setdefault(name, {"stats": {"total": 0.0, "count": 0}, "children": {}})
                if idx == len(path) - 1:
                    entry["stats"]["total"] = total
                    entry["stats"]["count"] = count
                node = entry["children"]

        rows: list[tuple[int, tuple[str, ...], float, int]] = []

        def collect(node_dict: dict[str, dict[str, Any]], path_prefix: tuple[str, ...], depth: int) -> None:
            items = sorted(node_dict.items(), key=lambda item: item[1]["stats"]["total"], reverse=True)
            for name, content in items:
                stats: dict[str, Any] = content["stats"]
                children: dict[str, dict[str, Any]] = content["children"]
                total = stats["total"]
                count = stats["count"]
                label_path = path_prefix + (name,)
                rows.append((depth, label_path, total, count))
                collect(children, label_path, depth + 1)

        collect(tree, tuple(), 1)

        if not rows:
            return

        labels = [f"{'/'.join(path)}:" for depth, path, _, _ in rows]
        label_width = max(len(label) for label in labels)
        totals = [f"{total:.3f}s total" for _, _, total, _ in rows]
        total_width = max(len(total_str) for total_str in totals)
        ms_values = []
        for _, _, total, count in rows:
            per_call_ms = (total / count) * 1e3 if count else 0.0
            ms_values.append(f"{per_call_ms:.3f}ms/call")
        ms_width = max(len(ms) for ms in ms_values)

        for label, total_str, ms_str in zip(labels, totals, ms_values):
            print(f"{label:<{label_width}} {total_str:>{total_width}} {ms_str:>{ms_width}}")

    def _index_single_file(
        self,
        dataset_path: Path,
        file_idx: int,
    ) -> tuple[_DatasetInfo, list[_IndexEntry]]:
        with h5py.File(dataset_path, "r") as handle:
            if "data" not in handle:
                raise KeyError(f"Expected group 'data' in LIBERO dataset: {dataset_path}")
            data_group = handle["data"]

            try:
                problem_info = data_group.attrs.get("problem_info")
                info = json.loads(problem_info)
                prompt = info.get("language_instruction", "") or ""
                prompt = prompt.strip().replace("_", " ").replace("\n", " ")
                prompt = f"<bos>Task: {prompt};\n Action:"
            except Exception:  # noqa: BLE001 - best effort parsing
                prompt = ""

            tokens, mask = self._tokenize_prompt(prompt)
            dataset_info = _DatasetInfo(
                path=dataset_path,
                prompt=prompt,
                tokenised_prompt=tokens,
                tokenised_prompt_mask=mask,
            )

            entries: list[_IndexEntry] = []
            for episode_key in sorted(data_group.keys()):
                episode_group = data_group[episode_key]
                num_samples = episode_group.attrs.get("num_samples").item()

                max_start = num_samples - self._action_horizon
                if max_start < 0:
                    continue

                for step in range(max_start + 1):
                    entries.append(_IndexEntry(file_idx, episode_key, step))

            return dataset_info, entries

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    
    class _ScriptTokenizer:
        def encode(self, prompt: str):  # noqa: D401 - minimal stub
            encoded = np.frombuffer(prompt.encode("utf-8"), dtype=np.uint8)
            return encoded.tolist()

        def decode(self, ids):
            if isinstance(ids, (list, tuple)):
                buffer = bytes(int(x) for x in ids if int(x) != 0)
            elif isinstance(ids, np.ndarray):
                buffer = bytes(int(x) for x in ids.tolist() if int(x) != 0)
            else:
                buffer = bytes(ids)
            return buffer.decode("utf-8", errors="ignore")

    repo_root = Path(__file__).resolve().parents[2]
    dataset_dir = repo_root / "datasets" / "libero_spatial"
    tokenizer = _ScriptTokenizer()
    dataset = LiberoDataset(dataset_dir, tokenizer=tokenizer, action_horizon=30, profile=True)
    data_sample: DataSample = dataset[0]
    observation, actions = data_sample.observation, data_sample.action.actions

    print("Loaded sample from:", dataset.episode_keys()[0])
    for key, img in zip(dataset._cameras, observation.images):
        print(f"  {key}:", tuple(img.shape), img.dtype)
    print("State shape:", tuple(observation.state.shape))
    print("ids:", observation.tokenized_prompt.tolist())
    prompt = tokenizer.decode(observation.tokenized_prompt.tolist())
    print("Prompt:", prompt)
    if actions is not None:
        print("Action shape:", tuple(actions.shape))

    state_series, action_series = _aggregate_state_action_stats(dataset)

    # state: (batch, action_horizon, action_dim=7), action = delta_pos 3 + delta_axis_angle 3 + gripper 1
    # actions: (batch, state_dim=9), state = gripper 2 + eef_pos 3 + eef_quat 4

    # in state: gripper 0: [0, 0.04], gripper 1: [-0.04, 0]
    # in action: gripper: [-1, 1], 1 close, -1 open

    # plot histogram for various quantities
    # state: gripper qpos, eef pos x, y, z, eef euler roll, pitch, yaw
    # actions: delta eef pos x, y, z, delta eef axis angle x, y, z, gripper

    def _plot_histograms(series: list[tuple[str, np.ndarray]], title: str) -> None:
        num_plots = len(series)
        cols = min(4, num_plots)
        rows = (num_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 3.0))
        flat_axes = np.array(axes, ndmin=1).ravel()
        for ax, (label, values) in zip(flat_axes, series):
            ax.hist(values, bins=50, color="steelblue", alpha=0.75)
            ax.set_title(label)
            ax.set_ylabel("count")
        for ax in flat_axes[num_plots:]:
            ax.remove()
        fig.suptitle(title)
        fig.tight_layout()

    _plot_histograms(state_series, "LIBERO State Distribution")
    _plot_histograms(action_series, "LIBERO Action Distribution")
    plt.show()
