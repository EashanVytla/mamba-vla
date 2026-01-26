from __future__ import annotations

import time
from typing import Tuple, Dict, TYPE_CHECKING

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp._fully_shard import (
    MixedPrecisionPolicy,
    fully_shard,
    register_fsdp_forward_method,
)
from tensordict import TensorClass
import jaxtyping as at

from vla_scratch.policies.base import BasePolicy
from vla_scratch.policies.modules.action_expert import MLPModel
from vla_scratch.policies.modules.vlm_bridge import (
    Qwen3VLBridge,
    PaligemmaBridge,
    SmolVLMBridge,
    MambaBridge,
)
from vla_scratch.policies.utils.training import fully_shard_layers

if TYPE_CHECKING:
    from vla_scratch.policies.modules.vlm_bridge.base import VLMOutputsBase
    from vla_scratch.policies.mlp.config import MLPPolicyConfig
    from vla_scratch.transforms.data_types import Observation, DataSample


class SuffixInput(TensorClass):
    """Input to the action expert extracted from VLM outputs."""

    prefix_pad_masks: at.Bool[torch.Tensor, " batch prefix_len"]  # noqa: F722
    hidden_state_list: at.Float[
        torch.Tensor, " batch n_layer prefix_len hidden"  # noqa: F722
    ]


class MLPPolicy(BasePolicy):
    """Policy with simple MLP action expert (no diffusion).

    Uses MSE loss for training instead of flow matching. Predicts actions
    in a single forward pass without iterative denoising.
    """

    def __init__(self, config: "MLPPolicyConfig"):
        super().__init__()
        self.config = config

        if config.action_dim is None or config.state_dim is None:
            raise ValueError(
                "MLPPolicyConfig.action_dim and MLPPolicyConfig.state_dim must be set "
                "before initializing MLPPolicy."
            )

        # Initialize VLM bridge
        start_time = time.time()
        self._init_vlm_bridge(config)
        end_time = time.time()
        print(
            f"VLM model initialized in {end_time - start_time:.2f} seconds: {config.vlm_type}"
        )

        # Get VLM hidden size
        _, _, _, vlm_hidden_size = self.vlm_bridge.get_text_dims()

        # Observation registers
        self.use_obs_register = config.num_obs_registers > 0
        if self.use_obs_register:
            self.obs_registers = nn.Parameter(
                torch.zeros(config.num_obs_registers, vlm_hidden_size)
            )
            self.obs_registers_pad_masks = torch.ones(
                config.num_obs_registers, dtype=torch.bool
            )
            self.obs_registers_att_masks = torch.zeros(
                config.num_obs_registers, dtype=torch.bool
            )
            self.obs_registers_att_masks[0] = 1
        else:
            assert not config.expert_only_use_register, (
                "expert_only_use_register must be False when num_obs_registers is 0."
            )

        # MLP action expert
        start_time = time.time()
        # Update action_horizon in config
        config.action_expert_cfg.action_horizon = config.action_horizon
        self.action_expert = MLPModel(config=config.action_expert_cfg)
        end_time = time.time()
        print(f"MLP action expert initialized in {end_time - start_time:.2f} seconds.")

    def _init_vlm_bridge(self, config: "MLPPolicyConfig"):
        """Initialize the appropriate VLM bridge based on config."""
        if config.vlm_type == "PaliGemmaForConditionalGeneration":
            self.vlm_bridge = PaligemmaBridge(
                model_id=config.model_id,
                vlm_type=config.vlm_type,
            )
        elif config.vlm_type == "Qwen3VLForConditionalGeneration":
            self.vlm_bridge = Qwen3VLBridge(
                model_id=config.model_id,
                vlm_type=config.vlm_type,
            )
        elif config.vlm_type == "SmolVLMForConditionalGeneration":
            self.vlm_bridge = SmolVLMBridge(
                model_id=config.model_id,
                vlm_type=config.vlm_type,
            )
        elif config.vlm_type == "MambaForCausalLM":
            self.vlm_bridge = MambaBridge(
                model_id=config.model_id,
                vlm_type=config.vlm_type,
                config=config,
            )
        else:
            raise NotImplementedError(
                f"Unsupported VLM type for MLPPolicy: {config.vlm_type}"
            )

    def initialize_weights(self):
        """Initialize learnable parameters."""
        if self.use_obs_register:
            nn.init.normal_(
                self.obs_registers,
                mean=0.0,
                std=self.config.obs_register_init_gain,
            )
        self.action_expert.initialize_weights()

    def apply_fsdp(self, param_type, reduce_type, output_dtype, mesh):
        """Apply FSDP sharding to the policy."""
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_type,
            reduce_dtype=reduce_type,
            cast_forward_inputs=True,
        )
        self.vlm_bridge.apply_fsdp(mp_policy, mesh)

        mp_policy_root = MixedPrecisionPolicy(
            param_dtype=param_type,
            reduce_dtype=reduce_type,
            output_dtype=output_dtype,
            cast_forward_inputs=True,
        )
        fully_shard(self, mesh=mesh, mp_policy=mp_policy_root)
        register_fsdp_forward_method(self, "compute_loss")
        register_fsdp_forward_method(self, "encode_prefix")
        register_fsdp_forward_method(self, "predict_suffix")
        register_fsdp_forward_method(self, "sample_actions")
        return self

    def encode_prefix(
        self, observation: "Observation"
    ) -> Tuple[torch.Tensor, "VLMOutputsBase", Dict]:
        """Encode observation through VLM."""
        extra_embs = None
        extra_pad = None
        extra_att = None

        if self.use_obs_register:
            bsize = observation.shape[0]
            extra_embs = einops.repeat(self.obs_registers, "s d -> b s d", b=bsize)
            extra_pad = einops.repeat(
                self.obs_registers_pad_masks, "s -> b s", b=bsize
            )
            extra_att = self.obs_registers_att_masks

        ce_loss, vlm_outputs, log_dict = self.vlm_bridge.encode(
            observation=observation,
            extra_embs=extra_embs,
            extra_pad_masks=extra_pad,
            extra_att_masks=extra_att,
            zero_pos_id_for_extra=self.config.zero_pos_id_for_obs_register,
            extra_attention_mask=self.config.causal_mask_obs_register,
        )
        return ce_loss, vlm_outputs, log_dict

    def construct_suffix_input(self, vlm_outputs: "VLMOutputsBase") -> SuffixInput:
        """Construct SuffixInput from VLMOutputs."""
        prefix_pad_masks = vlm_outputs.prefix_pad_masks
        hidden_state_list = vlm_outputs.hidden_state_list

        # Only use observation registers if configured
        if self.config.expert_only_use_register:
            torch.cuda.nvtx.range_push("select_obs_registers")
            num_registers = self.config.num_obs_registers
            prefix_pad_masks = prefix_pad_masks[:, -num_registers:]
            hidden_state_list = hidden_state_list[:, :, -num_registers:, :]
            torch.cuda.nvtx.range_pop()

        return SuffixInput(
            prefix_pad_masks=prefix_pad_masks,
            hidden_state_list=hidden_state_list,
            batch_size=vlm_outputs.shape,
        )

    def predict_suffix(
        self,
        state: torch.Tensor,
        suffix_input: SuffixInput,
        noisy_actions: torch.Tensor,  # Unused in MLP policy
        time: torch.Tensor,  # Unused in MLP policy
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Predict actions directly (single pass, no diffusion)."""
        encoder_hidden_states = suffix_input.hidden_state_list.unbind(dim=1)
        prefix_pad_masks = suffix_input.prefix_pad_masks

        # Flatten state if needed
        if state is not None and state.dim() == 3:
            batch_size = state.shape[0]
            state = state.reshape(batch_size, -1)

        _, actions, log_dict = self.action_expert.forward(
            encoder_hidden_states=encoder_hidden_states,
            prefix_pad_masks=prefix_pad_masks,
            state=state if self.config.action_expert_cfg.concat_state else None,
            action_dim=self.config.action_dim,
        )

        return 0.0, actions, log_dict

    def compute_loss(
        self, data_sample: "DataSample"
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute MSE loss between predicted and target actions."""
        torch.cuda.nvtx.range_push("Model Encode Prefix")
        ce_loss, vlm_outputs, log_dict_prefix = self.encode_prefix(
            observation=data_sample.observation
        )
        torch.cuda.nvtx.range_pop()

        if data_sample.action_chunk is not None:
            suffix_input = self.construct_suffix_input(vlm_outputs)

            if self.config.detach_encoder_output:
                torch.cuda.nvtx.range_push("Detach Suffix Input")
                suffix_input = suffix_input.detach()
                torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("Model Predict Suffix")
            _, pred_actions, log_dict_suffix = self.predict_suffix(
                state=data_sample.observation.state,
                suffix_input=suffix_input,
                noisy_actions=None,
                time=None,
            )
            torch.cuda.nvtx.range_pop()

            # MSE loss (not flow matching)
            target_actions = data_sample.action_chunk.actions
            action_mse = F.mse_loss(pred_actions, target_actions.to(pred_actions.dtype))
            log_dict_suffix["loss/action_mse"] = action_mse.detach()
        else:
            action_mse = torch.tensor(0.0, device=ce_loss.device)
            log_dict_suffix = {}

        loss = action_mse + self.config.ce_loss_weight * ce_loss
        log_dict = {**log_dict_prefix, **log_dict_suffix}

        return loss, log_dict

    @torch.inference_mode()
    def sample_actions(
        self, observation: "Observation", num_steps: int = 1  # num_steps unused
    ) -> torch.Tensor:
        """Sample actions in a single forward pass (no iteration)."""
        torch.cuda.nvtx.range_push("VLM Encode Prefix")
        _, vlm_output, _ = self.encode_prefix(observation)
        suffix_input = self.construct_suffix_input(vlm_output)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("MLP Predict Actions")
        _, actions, _ = self.predict_suffix(
            state=observation.state,
            suffix_input=suffix_input,
            noisy_actions=None,
            time=None,
        )
        torch.cuda.nvtx.range_pop()

        return actions
