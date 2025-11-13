from dataclasses import dataclass, MISSING
from typing import Any, List
from vla_scratch.utils.config import locate_class


@dataclass
class PolicyConfig:
    _target_: str
    transforms: List[Any]

    state_history: int = MISSING
    action_horizon: int = MISSING

def create_policy(policy_config: PolicyConfig) -> Any:
    policy_cls = locate_class(policy_config._target_)
    return policy_cls(policy_config)
