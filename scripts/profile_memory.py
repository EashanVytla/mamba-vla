"""
Memory profiling script for VLA models.

Usage:
    python scripts/profile_memory.py policy=pi-mamba batch_size=16

This script loads the model, creates dummy data, runs a forward and backward pass,
and reports GPU memory usage.
"""

from dataclasses import dataclass, field
from typing import Any, List, TYPE_CHECKING
import torch

import hydra
from hydra.core.config_store import ConfigStore
from hydra.conf import HydraConf, JobConf, RunDir
from omegaconf import DictConfig, OmegaConf, MISSING

from vla_scratch.policies.config import PolicyConfig
from vla_scratch.transforms.data_types import DataSample, Observation, ActionChunk
from vla_scratch.helpers.data import make_transforms

import vla_scratch.configs  # noqa: F401

if TYPE_CHECKING:
    from vla_scratch.policies.base import BasePolicy
    from vla_scratch.transforms.base import TransformFn


def create_single_dummy_sample(
    num_cam: int,
    height: int,
    width: int,
    state_history: int,
    state_dim: int,
    action_horizon: int,
    action_dim: int,
) -> DataSample:
    """Create a single dummy data sample (unbatched) for transform processing."""
    # Images: [num_cam, 3, height, width] -> need [num_cam, height, width, 3] for processors
    images = torch.randint(0, 255, (num_cam, height, width, 3), dtype=torch.uint8)

    observation = Observation(
        images=images,
        image_masks=torch.ones(num_cam, 1, dtype=torch.bool),
        state=torch.randn(state_history, state_dim),
        task="Pick up the red block and place it on the blue target.",
        generation_prompt="",
        generation_answer="",
        batch_size=[],
    )
    action_chunk = ActionChunk(
        actions=torch.randn(action_horizon, action_dim),
        batch_size=[],
    )
    return DataSample(
        observation=observation,
        action_chunk=action_chunk,
        batch_size=[],
    )


def apply_transforms(
    sample: DataSample,
    transforms: List["TransformFn"],
) -> DataSample:
    """Apply a list of transforms to a sample."""
    for transform in transforms:
        sample = transform.compute(sample)
    return sample


def create_dummy_batch(
    batch_size: int,
    num_cam: int,
    height: int,
    width: int,
    state_history: int,
    state_dim: int,
    action_horizon: int,
    action_dim: int,
    transforms: List["TransformFn"],
    device: torch.device,
) -> DataSample:
    """Create a batch of dummy data with transforms applied."""
    samples = []
    for _ in range(batch_size):
        sample = create_single_dummy_sample(
            num_cam=num_cam,
            height=height,
            width=width,
            state_history=state_history,
            state_dim=state_dim,
            action_horizon=action_horizon,
            action_dim=action_dim,
        )
        sample = apply_transforms(sample, transforms)
        samples.append(sample)

    # Stack samples into a batch
    batch = torch.stack(samples, dim=0)
    return batch.to(device)


def format_memory(bytes_val: float) -> str:
    """Format bytes as human-readable string."""
    gb = bytes_val / (1024**3)
    return f"{gb:.2f} GB"


def print_memory_stats(stage: str):
    """Print current GPU memory statistics."""
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_allocated = torch.cuda.max_memory_allocated()

    print(f"\n{'='*60}")
    print(f" {stage}")
    print(f"{'='*60}")
    print(f"  Currently allocated: {format_memory(allocated)}")
    print(f"  Currently reserved:  {format_memory(reserved)}")
    print(f"  Peak allocated:      {format_memory(max_allocated)}")


@dataclass
class ProfileConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"policy": "pi-qwen"},
        ]
    )

    batch_size: int = 16
    low_mem: bool = False

    # Image dimensions (override if your dataset uses different sizes)
    num_cam: int = 2
    image_height: int = 224
    image_width: int = 224

    # These are typically set from data config or inferred
    state_dim: int = 8
    action_dim: int = 7

    # Model config
    policy: PolicyConfig = MISSING

    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            job=JobConf(chdir=False),
            output_subdir=None,
            run=RunDir(dir="."),
        )
    )


cs = ConfigStore.instance()
cs.store(name="profile", node=ProfileConfig())


@hydra.main(config_name="profile", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    profile_cfg = OmegaConf.to_object(cfg)

    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU.")
        return

    device = torch.device("cuda:0")

    print("\n" + "="*60)
    print(" VLA Memory Profiler")
    print("="*60)
    print(f"  Policy: {profile_cfg.policy._target_}")
    print(f"  Batch size: {profile_cfg.batch_size}")
    print(f"  Image size: {profile_cfg.num_cam} x {profile_cfg.image_height} x {profile_cfg.image_width}")
    print(f"  State dim: {profile_cfg.state_dim}")
    print(f"  Action dim: {profile_cfg.action_dim}")
    print(f"  Low mem mode: {profile_cfg.low_mem}")

    # Get GPU info
    gpu_props = torch.cuda.get_device_properties(device)
    total_memory = gpu_props.total_memory
    print(f"\n  GPU: {gpu_props.name}")
    print(f"  Total GPU memory: {format_memory(total_memory)}")

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    print_memory_stats("Initial (empty)")

    # Set action/state dims on policy config
    profile_cfg.policy.action_dim = profile_cfg.action_dim
    profile_cfg.policy.state_dim = profile_cfg.state_dim

    # Instantiate policy transforms (processors)
    print("\nInstantiating transforms...")
    transforms = make_transforms(profile_cfg.policy.transforms)
    print(f"  Loaded {len(transforms)} transform(s)")

    # Create dummy data with transforms applied
    print("\nCreating dummy data and applying transforms...")
    dummy_data = create_dummy_batch(
        batch_size=profile_cfg.batch_size,
        num_cam=profile_cfg.num_cam,
        height=profile_cfg.image_height,
        width=profile_cfg.image_width,
        state_history=profile_cfg.policy.state_history,
        state_dim=profile_cfg.state_dim,
        action_horizon=profile_cfg.policy.action_horizon,
        action_dim=profile_cfg.action_dim,
        transforms=transforms,
        device=device,
    )

    print_memory_stats("After data creation")

    # Create model
    print("\nLoading model...")
    with torch.device(device):
        policy: "BasePolicy" = profile_cfg.policy.instantiate()

    if profile_cfg.low_mem:
        policy = policy.bfloat16()

    print_memory_stats("After model load")

    # Forward pass
    print("\nRunning forward pass...")
    torch.cuda.reset_peak_memory_stats()

    loss, log_dict = policy.compute_loss(dummy_data)

    print_memory_stats("After forward pass")
    print(f"\n  Loss value: {loss.item():.4f}")

    # Backward pass
    print("\nRunning backward pass...")
    torch.cuda.reset_peak_memory_stats()

    loss.backward()

    print_memory_stats("After backward pass (peak training memory)")

    # Optimizer step simulation
    print("\nSimulating optimizer step...")
    torch.cuda.reset_peak_memory_stats()

    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    # Create optimizer and do a step
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    optimizer.step()
    optimizer.zero_grad()

    print_memory_stats("After optimizer step")

    # Final summary
    print("\n" + "="*60)
    print(" Summary")
    print("="*60)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Parameter memory:     {format_memory(total_params * 4)}")  # fp32
    print(f"  Peak training memory: {format_memory(torch.cuda.max_memory_allocated())}")
    print(f"  Available GPU memory: {format_memory(total_memory)}")

    utilization = torch.cuda.max_memory_allocated() / total_memory * 100
    print(f"\n  Memory utilization:   {utilization:.1f}%")

    if utilization > 90:
        print("  WARNING: Memory usage is very high. Consider reducing batch size.")
    elif utilization > 75:
        print("  NOTE: Memory usage is moderate. You may have room to increase batch size.")
    else:
        print("  OK: Memory usage is comfortable.")

    # Detailed memory breakdown
    print("\n" + "="*60)
    print(" Detailed Memory Breakdown")
    print("="*60)
    print(torch.cuda.memory_summary(device=device, abbreviated=True))


if __name__ == "__main__":
    main()
