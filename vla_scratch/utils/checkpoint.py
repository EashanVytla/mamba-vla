import torch
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    StateDictOptions,
    set_model_state_dict,
    set_optimizer_state_dict,
)


def find_latest_checkpoint(path: Path | str, desired_iter: Optional[int] = None) -> Optional[Path]:
    """Return a concrete checkpoint file path from a file or directory.

    - If `path` is a file, returns it.
    - If `path` is a directory, find the newest `checkpoint_*.pth` (or the one
      matching `desired_iter` if provided).
    - If no candidate is found or path missing, returns None.
    """
    p = Path(path)
    if p.is_file():
        return p
    if not p.exists():
        return None
    candidates = sorted(p.glob("checkpoint_*.pth"))
    if not candidates:
        return None

    def _epoch_num(cp: Path) -> int:
        stem = cp.stem  # checkpoint_X
        try:
            epoch = int(stem.split("_")[-1])
            # If a specific iter is desired, bubble it to the end
            if desired_iter is not None and epoch == desired_iter:
                return 10**9
            return epoch
        except Exception:
            return -1

    return sorted(candidates, key=_epoch_num)[-1]


def load_model_from_checkpoint(
    model: torch.nn.Module,
    path: Path | str,
    device: torch.device | str = "cpu",
    *,
    strict: bool = False,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Load a checkpoint into `model`.

    Expects either a raw state_dict or a dict with key "model".
    Returns (missing_keys, unexpected_keys) from load_state_dict.
    """
    ckpt = torch.load(path, map_location=device)
    model_state: Dict[str, Any]
    if isinstance(ckpt, dict) and "model" in ckpt:
        model_state = ckpt["model"]
    else:
        model_state = ckpt  # assume raw state dict
    missing, unexpected = model.load_state_dict(model_state, strict=strict)
    return tuple(missing), tuple(unexpected)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint: str,
    global_rank: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    # Only rank 0 reads from disk; others pass empty dict and receive via broadcast
    if global_rank == 0:
        state_dict = torch.load(
            checkpoint,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        model_sd = state_dict.get("model", {})
        optim_sd = state_dict.get("optimizer", {})
    else:
        model_sd = {}
        optim_sd = {}

    options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
    # set_model_state_dict is collective-aware and will synchronize as needed
    missing, unexpected = set_model_state_dict(model=model, model_state_dict=model_sd, options=options)

    # Always call optimizer load on all ranks to keep collectives in sync
    if optimizer is not None:
        set_optimizer_state_dict(
            model=model,
            optimizers=optimizer,
            optim_state_dict=optim_sd,
            options=options,
        )
    return tuple(missing), tuple(unexpected)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_rank: int,
    filename: str,
):
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    model_state_dict, optim_state_dict = get_state_dict(
        model,
        optimizers=optimizer,
        options=options,
    )

    if global_rank == 0:
        full_state_dict = {
            "model": model_state_dict,
            "optimizer": optim_state_dict,
        }
        torch.save(full_state_dict, filename)
        print(f"Saved checkpoint to {filename}")


