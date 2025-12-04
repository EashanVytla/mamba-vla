from __future__ import annotations

import einops
import torch

from vla_scratch.policies.utils.transformers import sample_noise


def build_beta_time_dist(
    alpha: float,
    beta: float,
    device: torch.device | str,
) -> torch.distributions.Distribution:
    """Construct a Beta distribution on the training device."""
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    return torch.distributions.Beta(beta_t, alpha_t)


def sample_clamped_time(
    time_dist: torch.distributions.Distribution,
    shape: torch.Size,
) -> torch.Tensor:
    """Sample diffusion timesteps with a small clamp to avoid numerical issues."""
    return time_dist.sample(shape) * 0.999 + 0.001


from typing import TypeVar
T = TypeVar("T")
def repeat_batch(t: T, repeat_times: int) -> T:
    """Repeat a tensor along a new leading dimension then flatten."""
    return t.expand(repeat_times, *t.shape).reshape(-1, *t.shape[1:])


def sample_topk_noise(
    actions: torch.Tensor,
    num_noise_before_topk: int,
    num_noise_per_sample: int,
    device: torch.device,
) -> torch.Tensor:
    """Draw noisy action candidates and choose the closest ones to the target actions."""
    batch_size, action_horizon, action_dim = actions.shape
    candidate_shape = (
        batch_size,
        num_noise_before_topk,
        action_horizon,
        action_dim,
    )
    noise_candidates = sample_noise(
        candidate_shape,
        device,
        dtype=actions.dtype,
    )
    action_flat = einops.rearrange(
        actions, "b h d -> b 1 (h d)", h=action_horizon, d=action_dim
    )
    noise_flat = einops.rearrange(
        noise_candidates,
        "b k h d -> b k (h d)",
        h=action_horizon,
        d=action_dim,
    )
    if num_noise_before_topk == num_noise_per_sample:
        selected_noise_flat = noise_flat
    else:
        distances = torch.sum((noise_flat - action_flat) ** 2, dim=-1)
        topk_indices = torch.topk(
            distances,
            k=num_noise_per_sample,
            dim=1,
            largest=False,
        ).indices
        gather_idx = topk_indices.unsqueeze(-1).expand(
            -1,
            -1,
            noise_flat.shape[-1],
        )
        selected_noise_flat = torch.gather(
            noise_flat,
            dim=1,
            index=gather_idx,
        )

    selected_noise = einops.rearrange(
        selected_noise_flat,
        "b k (h d) -> (k b) h d",
        h=action_horizon,
        d=action_dim,
    )
    return selected_noise
