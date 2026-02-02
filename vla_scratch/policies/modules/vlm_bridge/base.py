from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
from tensordict import TensorClass
import jaxtyping as at

if TYPE_CHECKING:
    from vla_scratch.transforms.data_types import Observation

TARGET_IGNORE_ID = -100


# Type alias for VLM cache - actual cache type depends on the model
# For Mamba: transformers.models.mamba.modeling_mamba.MambaCache
# For Transformers: None (no temporal state caching needed)
VLMCache = Any


class VLMOutputsBase(TensorClass):
    """Base output class with fields common to all VLM types."""

    last_hidden_state: at.Float[torch.Tensor, " batch seq_len hidden"]  # noqa: F722
    prefix_pad_masks: at.Bool[torch.Tensor, " batch seq_len"]  # noqa: F722
    hidden_state_list: at.Float[torch.Tensor, " batch n_layer seq_len hidden"]  # noqa: F722


class TransformerVLMOutputs(VLMOutputsBase):
    """Outputs for transformer-based VLMs (PaliGemma, Qwen, SmolVLM)."""

    key_states: at.Float[torch.Tensor, " batch n_layer n_head seq_len head_dim"]  # noqa: F722
    value_states: at.Float[
        torch.Tensor, " batch n_layer n_head seq_len head_dim"  # noqa: F722
    ]


class MambaVLMOutputs(VLMOutputsBase):
    """Outputs for Mamba-based VLMs."""

    # Optional SSM cache states (shape varies by Mamba version)
    # conv_states: (batch, n_layer, d_conv, d_inner) or None
    # ssm_states: (batch, n_layer, d_state, d_inner) or None
    conv_states: Optional[torch.Tensor] = None
    ssm_states: Optional[torch.Tensor] = None
    # Indices of last token of proprio_obs_t in the flattened sequence (shape (T,)).
    # use_state/state_history only define *whether* and *how many* state tokens exist;
    # they do not give the *position* of those tokens in the interleaved sequence, which
    # depends on text length, vis_per_t, and the [vis_t, proprio_t, act_t?] ordering.
    # The policy uses these indices to slice hidden_state_list at the right position
    # when building SuffixInput for the action expert at each timestep t.
    proprio_t_last_indices: Optional[torch.Tensor] = None


# Backwards compatibility alias
VLMOutputs = TransformerVLMOutputs


class VLMBridge(nn.Module):
    causal_model: nn.Module

    def get_text_dims(self) -> Tuple[int, int, int]:
        raise NotImplementedError

    @property
    def hidden_size(self) -> int:
        raise NotImplementedError

    def encode(
        self,
        observation: "Observation",
        *,
        cache: Optional[VLMCache] = None,
        actions: Optional[torch.Tensor] = None,
        extra_embs: Optional[torch.Tensor] = None,
        extra_pad_masks: Optional[torch.Tensor] = None,
        extra_att_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, VLMOutputsBase, Optional[VLMCache], Dict]:
        """Encode an observation through the VLM.

        Args:
            observation: The observation to encode.
            cache: Optional cache from previous timestep (for stateful models like Mamba).
            actions: Optional actions tensor; only used by Mamba bridge to build
                interleaved sequence [text, vis_0, proprio_0, act_0, ...].
            extra_embs: Additional embeddings to append (e.g., observation registers).
            extra_pad_masks: Padding masks for extra embeddings.
            extra_att_masks: Attention masks for extra embeddings.

        Returns:
            ce_loss: Cross-entropy loss on language modeling task.
            vlm_outputs: Model outputs containing hidden states.
            cache: Updated cache for next timestep (None for transformer models).
            log_dict: Logging metrics.
        """
        raise NotImplementedError

    def apply_fsdp(self, mp_policy, mesh):
        raise NotImplementedError
