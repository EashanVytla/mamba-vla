import torch
import jaxtyping as at
from typing import Iterable
import torch.distributed as dist
from torch.distributed.distributed_c10d import ProcessGroup


def create_sinusoidal_pos_embedding(
    time: at.Float[torch.Tensor, "b"],
    dimension: int,
    min_period: float,
    max_period: float,
    device="cpu",
    dtype=torch.float32,
) -> at.Float[torch.Tensor, "b d"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * torch.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_att_2d_masks(
    pad_masks: at.Bool[torch.Tensor, "b n"],
    att_masks: at.Bool[torch.Tensor, "b n"],
) -> at.Bool[torch.Tensor, "b n n"]:
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def attention_fill_false_to_inf(
    att_mask: at.Bool[torch.Tensor, "b q n"],
) -> at.Float[torch.Tensor, "b q n"]:
    """Helper method to prepare 4D attention masks for transformer."""
    return torch.where(att_mask, 0.0, -2.3819763e38)


def get_beta_dist(
    alpha: float, beta: float, device
) -> torch.distributions.Distribution:
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(beta_t, alpha_t)
    return dist


def sample_time(
    time_dist: torch.distributions.Distribution, bsize: torch.Size
) -> at.Float[torch.Tensor, "b"]:
    return time_dist.sample(bsize) * 0.999 + 0.001


def sample_noise(shape, device, dtype):
    return torch.normal(
        mean=0.0,
        std=1.0,
        size=shape,
        dtype=dtype,
        device=device,
    )


@torch.compile
@torch.no_grad()
def clip_grad_norm_(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pg: ProcessGroup | None = None,
) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)

    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(
        grads, norm_type, error_if_nonfinite, foreach
    )
    if pg is not None:
        total_norm **= norm_type
        dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pg)
        total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm
