"""CALVIN benchmark dataset loader.

CALVIN stores data as continuous sequences of NumPy files with:
- rgb_static: Static camera RGB images (200x200)
- rgb_gripper: Gripper camera RGB images (84x84)
- robot_obs: Robot proprioception (15D)
- actions/rel_actions: Robot actions (7D)
- Language annotations in separate files
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from vla_scratch.transforms.data_keys import (
    PROCESSED_STATE_KEY,
    PROCESSED_ACTION_KEY,
    PROCESSED_IMAGE_KEY,
    PROCESSED_IMAGE_MASK_KEY,
    TASK_KEY,
)

if TYPE_CHECKING:
    from vla_scratch.datasets.calvin.config import CalvinConfig


class CalvinDataset(Dataset):
    """Dataset for CALVIN benchmark.

    Loads data from raw NumPy files in CALVIN format.
    """

    def __init__(self, config: "CalvinConfig"):
        self.config = config
        self.data_root = Path(config.data_root)
        self.split = config.split
        self.cameras = config.cameras
        self.action_type = config.action_type
        self.action_horizon = config.action_horizon
        self.state_history = config.state_history

        # Load episode boundaries and language annotations
        self._load_episode_info()

        # Build valid frame indices (accounting for action horizon and state history)
        self._build_frame_indices()

    def _load_episode_info(self):
        """Load language annotations (only annotated segments are used)."""
        split_path = self.data_root / self.split

        # Load language annotations
        # Format: {'language': {'ann': [...], 'task': [...]}, 'info': {'indx': [(start, end), ...]}}
        lang_ann_path = split_path / "lang_annotations" / "auto_lang_ann.npy"
        self.lang_annotations_map: Dict[Tuple[int, int], str] = {}
        if lang_ann_path.exists():
            lang_data = np.load(lang_ann_path, allow_pickle=True).item()
            annotations = lang_data.get("language", {}).get("ann", [])
            indices = lang_data.get("info", {}).get("indx", [])
            for ann, (start, end) in zip(annotations, indices):
                self.lang_annotations_map[(start, end)] = ann
        else:
            raise FileNotFoundError(
                f"Language annotations file not found: {lang_ann_path}. "
                "Make sure data_root points to a valid CALVIN dataset with annotations."
            )

    def _build_frame_indices(self):
        """Build list of valid frame indices for sampling.

        Only includes frames within annotated segments from auto_lang_ann.npy.
        Each index must have action_horizon frames after it within the segment.
        State history is clipped to the segment start.
        """
        self.frame_indices: List[Tuple[int, int, str]] = []

        for (seg_start, seg_end), annotation in self.lang_annotations_map.items():
            # Valid frames must have enough future actions within this segment
            max_start_idx = seg_end - self.action_horizon + 1

            for frame_idx in range(seg_start, max_start_idx + 1):
                # seg_start is used as boundary for state history clipping
                self.frame_indices.append((frame_idx, seg_start, annotation))

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        frame_idx, ep_start, task = self.frame_indices[idx]

        # Load images from all cameras
        images = self._load_images(frame_idx)
        img_mask = torch.ones((len(self.cameras), 1), dtype=torch.bool)

        # Load robot state with history
        state = self._load_state(frame_idx, ep_start)

        # Load action chunk
        actions = self._load_actions(frame_idx)

        return {
            PROCESSED_IMAGE_KEY: images,
            PROCESSED_IMAGE_MASK_KEY: img_mask,
            PROCESSED_STATE_KEY: state,
            PROCESSED_ACTION_KEY: actions,
            TASK_KEY: task,
        }

    def _load_frame_npz(self, frame_idx: int) -> Dict[str, np.ndarray]:
        """Load all data for a single frame from NPZ file."""
        split_path = self.data_root / self.split
        npz_path = split_path / f"episode_{frame_idx:07d}.npz"
        return dict(np.load(npz_path))

    def _load_images(self, frame_idx: int) -> torch.Tensor:
        """Load images from all configured cameras."""
        frame_data = self._load_frame_npz(frame_idx)
        images = []
        target_size = getattr(self.config, "image_size", 200)

        for cam in self.cameras:
            img = frame_data[cam]
            # CALVIN stores images as uint8 (H, W, C)
            img_tensor = torch.from_numpy(img)
            # Resize if needed: (H, W, C) -> (C, H, W) for interpolate, then back
            if img_tensor.shape[0] != target_size or img_tensor.shape[1] != target_size:
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).float()
                img_tensor = F.interpolate(
                    img_tensor, size=(target_size, target_size), mode="bilinear", align_corners=False
                )
                img_tensor = img_tensor.squeeze(0).permute(1, 2, 0).to(torch.uint8)
            images.append(img_tensor)

        # Stack cameras: (num_cameras, H, W, C)
        return torch.stack(images, dim=0)

    def _load_state(self, frame_idx: int, ep_start: int) -> torch.Tensor:
        """Load robot state with history.

        Returns (state_history + 1, state_dim) tensor.
        """
        states = []

        for t in range(-self.state_history, 1):
            # Clip to episode start (don't cross episode boundary)
            state_idx = max(ep_start, frame_idx + t)
            frame_data = self._load_frame_npz(state_idx)
            state = frame_data["robot_obs"]
            states.append(torch.from_numpy(state).float())

        return torch.stack(states, dim=0)

    def _load_actions(self, frame_idx: int) -> torch.Tensor:
        """Load action chunk starting at frame_idx.

        Returns (action_horizon, action_dim) tensor.
        """
        action_key = "rel_actions" if self.action_type == "rel" else "actions"
        actions = []

        for t in range(self.action_horizon):
            action_idx = frame_idx + t
            frame_data = self._load_frame_npz(action_idx)
            action = frame_data[action_key]
            actions.append(torch.from_numpy(action).float())

        return torch.stack(actions, dim=0)
