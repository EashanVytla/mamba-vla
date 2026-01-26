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
        """Load episode boundaries and language annotations."""
        split_path = self.data_root / self.split

        # Load episode start/end indices
        ep_ids_path = split_path / "ep_start_end_ids.npy"
        if ep_ids_path.exists():
            self.episode_ids = np.load(ep_ids_path)
        else:
            raise FileNotFoundError(
                f"Episode IDs file not found: {ep_ids_path}. "
                "Make sure data_root points to a valid CALVIN dataset."
            )

        # Load language annotations if available
        lang_ann_path = split_path / "lang_annotations" / "auto_lang_ann.npy"
        if lang_ann_path.exists():
            self.lang_annotations = np.load(lang_ann_path, allow_pickle=True).item()
        else:
            self.lang_annotations = None

    def _build_frame_indices(self):
        """Build list of valid frame indices for sampling.

        Each index must have:
        - state_history frames before it (or be clipped to episode start)
        - action_horizon frames after it (within the same episode)
        """
        self.frame_indices: List[Tuple[int, int, Optional[str]]] = []

        for ep_idx, (ep_start, ep_end) in enumerate(self.episode_ids):
            # Valid frames must have enough future actions
            max_start_idx = ep_end - self.action_horizon + 1

            for frame_idx in range(ep_start, max_start_idx + 1):
                # Get language annotation for this episode if available
                task = self._get_episode_task(ep_start, ep_end)
                self.frame_indices.append((frame_idx, ep_start, task))

    def _get_episode_task(self, ep_start: int, ep_end: int) -> Optional[str]:
        """Get language annotation for an episode."""
        if self.lang_annotations is None:
            return None

        # CALVIN lang_annotations is a dict with (start, end) tuples as keys
        # pointing to list of annotations
        key = (ep_start, ep_end)
        if key in self.lang_annotations:
            annotations = self.lang_annotations[key]
            if annotations:
                # Return first annotation (could randomize for data augmentation)
                return annotations[0]
        return None

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

        result = {
            PROCESSED_IMAGE_KEY: images,
            PROCESSED_IMAGE_MASK_KEY: img_mask,
            PROCESSED_STATE_KEY: state,
            PROCESSED_ACTION_KEY: actions,
        }

        if task is not None:
            result[TASK_KEY] = task

        return result

    def _load_images(self, frame_idx: int) -> torch.Tensor:
        """Load images from all configured cameras."""
        split_path = self.data_root / self.split
        images = []

        for cam in self.cameras:
            img_path = split_path / cam / f"frame_{frame_idx:07d}.npy"
            img = np.load(img_path)
            # CALVIN stores images as uint8 (H, W, C)
            images.append(torch.from_numpy(img))

        # Stack cameras: (num_cameras, H, W, C)
        return torch.stack(images, dim=0)

    def _load_state(self, frame_idx: int, ep_start: int) -> torch.Tensor:
        """Load robot state with history.

        Returns (state_history + 1, state_dim) tensor.
        """
        split_path = self.data_root / self.split
        states = []

        for t in range(-self.state_history, 1):
            # Clip to episode start (don't cross episode boundary)
            state_idx = max(ep_start, frame_idx + t)
            state_path = split_path / "robot_obs" / f"frame_{state_idx:07d}.npy"
            state = np.load(state_path)
            states.append(torch.from_numpy(state).float())

        return torch.stack(states, dim=0)

    def _load_actions(self, frame_idx: int) -> torch.Tensor:
        """Load action chunk starting at frame_idx.

        Returns (action_horizon, action_dim) tensor.
        """
        split_path = self.data_root / self.split
        action_key = "rel_actions" if self.action_type == "rel" else "actions"
        actions = []

        for t in range(self.action_horizon):
            action_idx = frame_idx + t
            action_path = split_path / action_key / f"frame_{action_idx:07d}.npy"
            action = np.load(action_path)
            actions.append(torch.from_numpy(action).float())

        return torch.stack(actions, dim=0)
