"""Mamba policy with temporal processing, interleaved sequences, and inference cache."""

from __future__ import annotations

import time
from typing import Any, Tuple, Optional, TYPE_CHECKING, Dict
import einops
import jaxtyping as at

import torch
import torch.nn as nn
from torch.distributed.fsdp._fully_shard import (
    MixedPrecisionPolicy,
    fully_shard,
    register_fsdp_forward_method,
)

from tensordict import TensorClass

from vla_scratch.policies.base import BasePolicy
from vla_scratch.policies.modules.action_expert import DiTModel
from vla_scratch.policies.modules.vlm_bridge import MambaBridge
from vla_scratch.policies.utils.training import (
    apply_checkpoint_when_training,
    fully_shard_layers,
)
from vla_scratch.policies.utils.diffusion import (
    build_beta_time_dist,
    sample_clamped_time,
    repeat_batch,
    sample_noise,
)
from vla_scratch.policies.utils.transformers import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
)

if TYPE_CHECKING:
    from vla_scratch.policies.modules.vlm_bridge.base import (
        VLMOutputsBase,
        MambaVLMOutputs,
    )
    from vla_scratch.policies.mamba.config import MambaPolicyConfig
    from vla_scratch.transforms.data_types import Observation, DataSample


class SuffixInput(TensorClass):
    prefix_pad_masks: at.Bool[torch.Tensor, " batch prefix_len"]  # noqa: F722
    hidden_state_list: at.Float[  # noqa: F722
        torch.Tensor, " batch n_layer prefix_len hidden"
    ]


class MambaPolicy(BasePolicy):
    """Mamba-based policy with temporal interleaved sequences and SSM inference cache."""

    suffix_pad_mask: at.Bool[torch.Tensor, " action_horizon"]  # noqa: F722
    suffix_att_mask: at.Bool[torch.Tensor, " action_horizon"]  # noqa: F722

    def __init__(self, config: "MambaPolicyConfig"):
        super().__init__()
        self.config = config

        if config.action_dim is None or config.state_dim is None:
            raise ValueError(
                "MambaPolicyConfig.action_dim and state_dim must be set before "
                "initializing MambaPolicy."
            )

        start_time = time.time()
        self.vlm_bridge = MambaBridge(
            model_id=config.model_id,
            vlm_type=config.vlm_type,
            config=config,
        )
        end_time = time.time()
        print(
            f"VLM model initialized in {end_time - start_time:.2f} seconds: {config.vlm_type}"
        )

        text_layers, text_head_dim, text_num_kv_heads, vlm_hidden_size = (
            self.vlm_bridge.get_text_dims()
        )

        action_expert_config = config.action_expert_cfg
        start_time = time.time()
        self.action_expert = DiTModel(config=action_expert_config)
        end_time = time.time()
        print(
            "Action expert initialized in {end_time - start_time:.2f} seconds."
        )

        action_expert_width = action_expert_config.hidden_size
        self.action_in_proj = nn.Linear(config.action_dim, action_expert_width)
        self.action_out_proj = nn.Linear(action_expert_width, config.action_dim)
        self.state_in_proj = nn.Linear(config.state_dim, action_expert_width)

        self.time_mlp = nn.Sequential(
            nn.Linear(action_expert_width, action_expert_width),
            nn.SiLU(),
            nn.Linear(action_expert_width, action_expert_width),
            nn.SiLU(),
        )

        suffix_len = config.action_horizon + (
            config.state_history if config.use_state else 0
        )
        self.suffix_len = suffix_len
        suffix_pad_mask = torch.ones(suffix_len, dtype=torch.bool)
        suffix_att_mask = torch.zeros(suffix_len, dtype=torch.bool)
        suffix_att_mask[0] = 1
        self.register_buffer(
            "suffix_pad_mask", suffix_pad_mask, persistent=False
        )
        self.register_buffer(
            "suffix_att_mask", suffix_att_mask, persistent=False
        )

        if config.suffix_add_pos_emb:
            pos_emb_state = torch.zeros(
                config.state_history, action_expert_width
            )
            self.position_embedding_state = nn.Parameter(pos_emb_state)
            pos_emb_action = torch.zeros(
                config.action_horizon, action_expert_width
            )
            self.position_embedding_action = nn.Parameter(pos_emb_action)

        param_device = next(self.parameters()).device
        self.time_dist = build_beta_time_dist(
            config.time_dist_alpha,
            config.time_dist_beta,
            device=param_device,
        )

        self._inference_cache = None

    def initialize_weights(self):
        if self.config.suffix_add_pos_emb:
            nn.init.normal_(
                self.position_embedding_state,
                mean=0.0,
                std=self.config.suffix_pos_emb_init_gain,
            )
            nn.init.normal_(
                self.position_embedding_action,
                mean=0.0,
                std=self.config.suffix_pos_emb_init_gain,
            )
        self.action_expert.initialize_weights()

    def reset_cache(self) -> None:
        """Clear the SSM inference cache (e.g. at episode reset)."""
        self._inference_cache = None

    def apply_fsdp(self, param_type, reduce_type, output_dtype, mesh):
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_type,
            reduce_dtype=reduce_type,
            cast_forward_inputs=True,
        )
        self.vlm_bridge.apply_fsdp(mp_policy, mesh)
        fully_shard_layers(
            self.action_expert.blocks, mesh, mp_policy, num_to_prefetch=6
        )
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
        self,
        observation: "Observation",
        *,
        actions: Optional[torch.Tensor] = None,
        cache: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, "VLMOutputsBase", Dict]:
        """Encode observation through Mamba bridge.

        Args:
            observation: The observation to encode (6D pixel_values for Mamba).
            actions: Optional (B, T-1, action_horizon, action_dim) for training.
            cache: Optional SSM cache for inference; updated in-place.

        Returns:
            ce_loss: Cross-entropy loss from language modeling.
            vlm_outputs: MambaVLMOutputs with hidden states and proprio_t_last_indices.
            log_dict: Logging metrics.
        """
        ce_loss, vlm_outputs, out_cache, log_dict = self.vlm_bridge.encode(
            observation=observation,
            cache=cache,
            actions=actions,
        )
        if cache is not None or out_cache is not None:
            self._inference_cache = out_cache
        return ce_loss, vlm_outputs, log_dict

    def construct_suffix_input_from_t(
        self, vlm_outputs: "MambaVLMOutputs", t: int
    ) -> SuffixInput:
        """Build SuffixInput for timestep t using proprio_t_last_indices."""
        if vlm_outputs.proprio_t_last_indices is None:
            raise ValueError(
                "MambaVLMOutputs must have proprio_t_last_indices for MambaPolicy."
            )
        idx = vlm_outputs.proprio_t_last_indices[t]
        action_expert_layers = self.config.action_expert_cfg.num_hidden_layers
        hidden_state_list = vlm_outputs.hidden_state_list[
            :, -action_expert_layers:, idx : idx + 1, :
        ]
        bsz = hidden_state_list.shape[0]
        device = hidden_state_list.device
        prefix_pad_masks = torch.ones(
            bsz, 1, dtype=torch.bool, device=device
        )
        return SuffixInput(
            prefix_pad_masks=prefix_pad_masks,
            hidden_state_list=hidden_state_list,
            batch_size=[bsz],
        )

    def predict_suffix(
        self,
        state: at.Float[torch.Tensor, " batch horizon dim"],  # noqa: F722
        suffix_input: SuffixInput,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """One denoising step for the action expert."""
        prefix_pad_masks = suffix_input.prefix_pad_masks

        suffix_embs, suffix_pad_masks, suffix_att_masks, time_emb = (
            self._embed_suffix(state, noisy_actions, time)
        )

        suffix_att_2d_masks = make_att_2d_masks(
            suffix_pad_masks, suffix_att_masks
        )
        prefix_pad_mask = einops.repeat(
            prefix_pad_masks, "b p -> b s p", s=self.suffix_len
        )
        full_att_2d_mask = torch.cat(
            [prefix_pad_mask, suffix_att_2d_masks], dim=2
        )
        full_att_mask = einops.rearrange(full_att_2d_mask, "b i j -> b 1 i j")

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = (
            prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        )

        encoder_hidden_states = suffix_input.hidden_state_list.unbind(dim=1)
        _, suffix_out, log_dict = self.action_expert.forward(
            inputs_embeds=suffix_embs,
            position_ids=position_ids,
            adarms_cond=time_emb,
            attention_mask=full_att_mask,
            encoder_hidden_states=encoder_hidden_states,
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :, :]
        return 0.0, self.action_out_proj(suffix_out), log_dict

    def compute_loss(
        self,
        data_sample: "DataSample",
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute loss (Option B): predict action at every timestep t, average over T."""
        observation = data_sample.observation
        actions_full = data_sample.action_chunk.actions  # (B, T, action_horizon, action_dim)
        T = observation.policy_input.pixel_values.shape[1]
        actions_bridge = actions_full[:, :-1]  # (B, T-1, action_horizon, action_dim)

        torch.cuda.nvtx.range_push("Model Encode Prefix")
        ce_loss, vlm_outputs, log_dict_prefix = self.encode_prefix(
            observation=observation,
            actions=actions_bridge,
        )
        torch.cuda.nvtx.range_pop()

        if data_sample.action_chunk is None:
            loss = self.config.ce_loss_weight * ce_loss
            return loss, log_dict_prefix

        state_full = observation.state
        if state_full.ndim == 3:
            state_full = state_full.unsqueeze(1)  # (B, 1, state_history, state_dim)

        flow_mse_list = []
        log_dict_suffix = {}
        for t in range(T):
            torch.cuda.nvtx.range_push("Construct Suffix Input")
            suffix_input_t = self.construct_suffix_input_from_t(
                vlm_outputs, t
            )
            if self.config.detach_encoder_output:
                suffix_input_t = suffix_input_t.detach()
            torch.cuda.nvtx.range_pop()

            state_t = state_full[:, t]  # (B, state_history, state_dim)
            action_t = actions_full[:, t]  # (B, action_horizon, action_dim)

            torch.cuda.nvtx.range_push("Expand Data Sample")
            suffix_input_t = repeat_batch(
                suffix_input_t, self.config.num_noise_per_sample
            )
            state_t_rep = repeat_batch(
                state_t, self.config.num_noise_per_sample
            )
            action_t_rep = repeat_batch(
                action_t, self.config.num_noise_per_sample
            )
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("Apply Noise")
            selected_noise = sample_noise(
                action_t_rep.shape,
                action_t_rep.device,
                action_t_rep.dtype,
            )
            u_t = selected_noise - action_t_rep
            timestep = sample_clamped_time(
                self.time_dist, (action_t_rep.shape[0],)
            )
            noisy_actions_t = (
                action_t_rep + timestep[:, None, None] * u_t
            )
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("Model Predict Suffix")
            _, v_t, _ = self.predict_suffix(
                state=state_t_rep,
                suffix_input=suffix_input_t,
                noisy_actions=noisy_actions_t,
                time=timestep,
            )
            torch.cuda.nvtx.range_pop()

            flow_mse_t = torch.nn.functional.mse_loss(
                u_t.to(v_t.dtype), v_t, reduction="none"
            ).mean()
            flow_mse_list.append(flow_mse_t)

        flow_mse = torch.stack(flow_mse_list).mean()
        log_dict_suffix["loss/flow_mse"] = flow_mse.detach()

        loss = flow_mse + self.config.ce_loss_weight * ce_loss
        log_dict = {**log_dict_prefix, **log_dict_suffix}
        return loss, log_dict

    @torch.inference_mode()
    def sample_actions(
        self, observation: "Observation", num_steps: int = 10
    ) -> at.Float[torch.Tensor, " batch_size action_horizon action_dim"]:  # noqa: F722
        """Inference with SSM cache; uses last timestep for action prediction."""
        torch.cuda.nvtx.range_push("VLM Encode Prefix")
        _, vlm_output, _ = self.encode_prefix(
            observation,
            cache=self._inference_cache,
        )
        T = vlm_output.proprio_t_last_indices.shape[0]
        t_last = T - 1
        suffix_input = self.construct_suffix_input_from_t(
            vlm_output, t_last
        )
        torch.cuda.nvtx.range_pop()

        state = observation.state
        if state.ndim == 3:
            state = state[:, None, :, :]
        state = state[:, -1]  # (B, state_history, state_dim)

        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype
        actions_shape = (
            bsize,
            self.config.action_horizon,
            self.config.action_dim,
        )
        noise = sample_noise(actions_shape, device, dtype)

        dt_float = 1.0 / num_steps
        time_float = 1.0
        dt = torch.tensor(dt_float, dtype=dtype, device=device)
        time = torch.tensor(time_float, dtype=dtype, device=device)
        x_t = noise

        while time_float >= dt_float / 2:
            torch.cuda.nvtx.range_push("Predict Suffix in Sampling")
            _, v_t, _ = self.predict_suffix(
                state,
                suffix_input=suffix_input,
                noisy_actions=x_t,
                time=time.expand(bsize),
            )
            x_t = x_t - dt * v_t
            time -= dt
            time_float -= dt_float
            torch.cuda.nvtx.range_pop()

        return x_t

    def _embed_suffix(
        self,
        state: at.Float[torch.Tensor, " batch_size state_history state_dim"],  # noqa: F722
        noisy_actions: at.Float[  # noqa: F722
            torch.Tensor, " batch_size action_horizon action_dim"
        ],
        time: at.Float[torch.Tensor, " batch_size"],  # noqa: F722
    ) -> Tuple[
        at.Float[torch.Tensor, " batch_size action_horizon hidden_dim"],  # noqa: F722
        at.Bool[torch.Tensor, " batch_size action_horizon"],  # noqa: F722
        at.Bool[torch.Tensor, " batch_size action_horizon"],  # noqa: F722
        at.Float[torch.Tensor, " batch_size hidden_dim"],  # noqa: F722
    ]:
        """Embed state, noisy_actions, and timestep for the action expert."""
        time_fp32 = time.to(torch.float32)
        time_emb_fp32 = create_sinusoidal_pos_embedding(
            time_fp32,
            dimension=self.config.action_expert_cfg.hidden_size,
            min_period=4e-3,
            max_period=4.0,
            device=time.device,
            dtype=time_fp32.dtype,
        )
        time_emb = time_emb_fp32.to(time.dtype)
        time_emb = apply_checkpoint_when_training(self, self.time_mlp, time_emb)
        action_emb = apply_checkpoint_when_training(
            self, self.action_in_proj, noisy_actions
        )

        if self.config.use_state:
            state_emb = apply_checkpoint_when_training(
                self, self.state_in_proj, state
            )
            if self.config.suffix_add_pos_emb:
                state_emb = (
                    state_emb + self.position_embedding_state[None, :, :]
                )
                action_emb = (
                    action_emb + self.position_embedding_action[None, :, :]
                )
            suffix_emb = torch.cat([state_emb, action_emb], dim=1)
        else:
            if self.config.suffix_add_pos_emb:
                action_emb = (
                    action_emb + self.position_embedding_action[None, :, :]
                )
            suffix_emb = action_emb

        bsize = action_emb.shape[0]
        pad_mask = einops.repeat(
            self.suffix_pad_mask, "action_horizon -> b action_horizon", b=bsize
        )
        att_mask = einops.repeat(
            self.suffix_att_mask, "action_horizon -> b action_horizon", b=bsize
        )
        return suffix_emb, pad_mask, att_mask, time_emb
