from dataclasses import dataclass
from typing import Any, List, Literal, Optional
from vla_scratch.utils.config import locate_class


@dataclass(kw_only=True)
class PolicyConfig:
    _target_: str
    transforms: List[Any]

    state_history: int
    action_horizon: int
    state_dim: Optional[int] = None
    action_dim: Optional[int] = None

    action_head_type: Literal["mlp", "flow_matching"] = "flow_matching"

    def instantiate(self) -> Any:
        policy_cls = locate_class(self._target_)
        return policy_cls(self)
