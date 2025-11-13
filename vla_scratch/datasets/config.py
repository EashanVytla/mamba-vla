from dataclasses import dataclass, field
from typing import Any, List, Optional



@dataclass
class DataConfig:
    _target_: str
    action_horizon: Optional[int] = None
    state_history: Optional[int] = None
    # Structured transform lists
    input_transforms: List[Any] = field(default_factory=list)
    output_transforms: List[Any] = field(default_factory=list)
    output_inv_transforms: List[Any] = field(default_factory=list)
    norm_stats_path: Optional[str] = None

