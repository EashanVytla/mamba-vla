

from pathlib import Path
from typing import Any, Optional
import importlib
import re


def locate_class(target: str) -> type:
    """Import and return a class/function given a fully-qualified path string.

    Example: "vla_scratch.datasets.spirit.transforms.SpiritImages"
    """
    module_name, _, attr_name = target.rpartition(".")
    if not module_name:
        raise ValueError(f"Target '{target}' must be a fully-qualified path.")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ImportError(f"Cannot import '{attr_name}' from '{module_name}'.") from exc


def instantiate_transform(spec: Any) -> Any:
    """Instantiate a transform object from a spec.

    Accepts either an existing object with a `.compute()` method, or a dict with
    a `_target_` key and constructor kwargs. Returns an object exposing
    `.compute(sample) -> Any`.
    """
    # Already a transform-like object
    if hasattr(spec, "compute") and callable(getattr(spec, "compute")):
        return spec

    # Config mapping
    if isinstance(spec, dict):
        target = spec.get("_target_")
        if target is None:
            raise ValueError("Transform configuration must define '_target_'.")
        kwargs = {k: v for k, v in spec.items() if k != "_target_"}
        cls = locate_class(target)
        obj = cls(**kwargs)
        if hasattr(obj, "compute") and callable(getattr(obj, "compute")):
            return obj
        raise TypeError(f"Instance of '{target}' does not expose a 'compute' method.")

    raise TypeError(f"Unsupported transform specification: {spec!r}")


def resolve_config_placeholders(
    template: str | Path | None,
    *,
    data_cfg: Any,
    policy_cfg: Optional[Any] = None,
) -> Optional[str]:
    """Resolve placeholders like '{data.attr}' or '{policy.attr}' in a string/Path.

    Unknown placeholders are left untouched; only 'data.*' and 'policy.*' are resolved
    by simple attribute lookup.
    """
    if template is None:
        return None
    s = str(template)

    def _replace_data(match: re.Match[str]) -> str:
        attr = match.group(1)
        return str(getattr(data_cfg, attr, match.group(0)))

    def _replace_policy(match: re.Match[str]) -> str:
        if policy_cfg is None:
            return match.group(0)
        attr = match.group(1)
        return str(getattr(policy_cfg, attr, match.group(0)))

    s = re.sub(r"\{data\.([a-zA-Z0-9_]+)\}", _replace_data, s)
    s = re.sub(r"\{policy\.([a-zA-Z0-9_]+)\}", _replace_policy, s)
    return s


from typing import Any, Tuple, Optional
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

def load_saved_cfgs(
    run_dir: Path,
) -> Tuple[Optional[DictConfig], Optional[DictConfig]]:
    policy_cfg_path = run_dir / "policy-cfg.yaml"
    data_cfg_path = run_dir / "data-cfg.yaml"
    policy_cfg = OmegaConf.load(policy_cfg_path) if policy_cfg_path.exists() else None
    data_cfg = OmegaConf.load(data_cfg_path) if data_cfg_path.exists() else None
    return policy_cfg, data_cfg


