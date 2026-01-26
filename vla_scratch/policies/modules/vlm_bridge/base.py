from __future__ import annotations

from typing import Dict, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
from tensordict import TensorClass
import jaxtyping as at

if TYPE_CHECKING:
    from vla_scratch.transforms.data_types import Observation

TARGET_IGNORE_ID = -100


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
        extra_embs: Optional[torch.Tensor] = None,
        extra_pad_masks: Optional[torch.Tensor] = None,
        extra_att_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, VLMOutputsBase, Dict]:
        raise NotImplementedError

    def apply_fsdp(self, mp_policy, mesh):
        raise NotImplementedError
