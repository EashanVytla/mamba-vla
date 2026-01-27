from dataclasses import dataclass, field
from typing import List

from hydra.core.config_store import ConfigStore

from vla_scratch.datasets.config import DataConfig


@dataclass
class CalvinConfig(DataConfig):
    """Configuration for CALVIN benchmark dataset."""

    _target_: str = "vla_scratch.datasets.calvin.dataset.CalvinDataset"

    # Path to CALVIN dataset root (e.g., /path/to/calvin/task_D_D)
    data_root: str = "/local/scratch/vytla.4/calvin/dataset/calvin_debug_dataset/"

    # Which split to use: "training" or "validation"
    split: str = "training"

    # Camera selection (rgb_static: 200x200, rgb_gripper: 84x84)
    cameras: List[str] = field(
        default_factory=lambda: ["rgb_static", "rgb_gripper"]
    )

    # Action type: "rel" for relative actions, "abs" for absolute
    action_type: str = "rel"

    # Target image size (all cameras resized to this)
    image_size: int = 200

    # Normalization stats path
    norm_stats_path: str = "normalization_stats/calvin/calvin-horizon_{data.action_horizon}-history_{data.state_history}.npz"


# Default CALVIN configuration
calvin_config = CalvinConfig(
    action_horizon=10,
    state_history=1,
)

cs = ConfigStore.instance()
cs.store(name="calvin", node=calvin_config, group="data")
