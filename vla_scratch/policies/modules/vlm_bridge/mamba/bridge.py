from __future__ import annotations

import importlib
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

from vla_scratch.policies.utils.training import (
    apply_checkpoint_when_training,
    fully_shard_layers,
)
from vla_scratch.policies.modules.vlm_bridge.base import (
    VLMBridge,
    VLMCache,
    MambaVLMOutputs,
    TARGET_IGNORE_ID,
)
from vla_scratch.policies.modules.vlm_bridge.mamba.processor import (
    MambaPolicyInput,
)
from vla_scratch.policies.utils.transformers import make_att_2d_masks

if TYPE_CHECKING:
    from transformers import MambaForCausalLM, SiglipVisionModel
    from transformers.models.mamba.modeling_mamba import MambaCache
    from vla_scratch.transforms.data_types import Observation
    from vla_scratch.policies.pi.config import PiConfig


class MambaBridge(VLMBridge):
    """VLM Bridge for Mamba with SigLIP vision encoder.

    Combines a text-only Mamba model with a separate SigLIP vision encoder.
    Images are encoded via SigLIP, projected to Mamba's hidden dimension,
    and prepended to the text embeddings before being processed by Mamba.
    """

    def __init__(
        self,
        *,
        model_id: str,
        vlm_type: str,
        config: "PiConfig" = None,
    ):
        super().__init__()
        tfm = importlib.import_module("transformers")

        # Pull Mamba-specific settings from config (with defaults)
        vision_encoder_id = getattr(
            config, "vision_encoder_id", "google/siglip-base-patch16-224"
        )
        freeze_llm_backbone = getattr(config, "freeze_llm_backbone", False)

        # Load Mamba model
        try:
            vlm_cls = getattr(tfm, vlm_type)
        except AttributeError as e:
            raise ImportError(
                f"transformers has no class named '{vlm_type}'."
            ) from e

        self.causal_model: "MambaForCausalLM" = vlm_cls.from_pretrained(
            model_id,
            device_map=torch.cuda.current_device(),
            torch_dtype=torch.bfloat16,
        )

        # Load SigLIP vision encoder
        SiglipVisionModel = getattr(tfm, "SiglipVisionModel")
        self.vision_encoder: "SiglipVisionModel" = SiglipVisionModel.from_pretrained(
            vision_encoder_id,
            device_map=torch.cuda.current_device(),
            torch_dtype=torch.bfloat16,
        )

        # Create projection layer: SigLIP hidden -> Mamba hidden
        vision_hidden_size = self.vision_encoder.config.hidden_size
        mamba_hidden_size = self.causal_model.config.hidden_size
        self.vision_projector = nn.Linear(
            vision_hidden_size,
            mamba_hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )

        # Proprio (state) and action projections for interleaved sequence (optional)
        state_dim = getattr(config, "state_dim", None) if config else None
        state_history = getattr(config, "state_history", 1) if config else 1
        action_dim = getattr(config, "action_dim", None) if config else None
        action_horizon = getattr(config, "action_horizon", None) if config else None
        if state_dim is not None:
            self.state_proj = nn.Linear(
                state_history * state_dim,
                mamba_hidden_size,
                device=torch.cuda.current_device(),
                dtype=torch.bfloat16,
            )
        else:
            self.state_proj = None
        if action_dim is not None and action_horizon is not None:
            self.action_proj = nn.Linear(
                action_horizon * action_dim,
                mamba_hidden_size,
                device=torch.cuda.current_device(),
                dtype=torch.bfloat16,
            )
        else:
            self.action_proj = None

        # Freeze LLM backbone if requested (allows finetuning only vision encoder)
        if freeze_llm_backbone:
            for param in self.causal_model.parameters():
                param.requires_grad = False

    def apply_fsdp(self, mp_policy, mesh):
        """Apply FSDP sharding to Mamba and vision encoder layers."""
        # Shard Mamba backbone layers
        if hasattr(self.causal_model, "backbone"):
            fully_shard_layers(
                self.causal_model.backbone.layers,
                mesh,
                mp_policy,
                num_to_prefetch=6,
            )
        elif hasattr(self.causal_model, "model"):
            fully_shard_layers(
                self.causal_model.model.layers,
                mesh,
                mp_policy,
                num_to_prefetch=6,
            )

        # Shard SigLIP vision encoder layers
        fully_shard_layers(
            self.vision_encoder.vision_model.encoder.layers,
            mesh,
            mp_policy,
            num_to_prefetch=6,
        )

    def get_text_dims(self) -> Tuple[int, int, int, int]:
        """Return (num_layers, head_dim, num_kv_heads, hidden_size).

        For Mamba, head_dim and num_kv_heads are synthetic since there's no attention.
        We use hidden_size as a proxy for compatibility with action experts.
        """
        cfg = self.causal_model.config
        hidden_size = cfg.hidden_size
        num_layers = cfg.num_hidden_layers

        # Synthetic values for compatibility with action expert cross-attention
        # The action expert primarily uses hidden_state_list
        head_dim = hidden_size
        num_kv_heads = 1

        return num_layers, head_dim, num_kv_heads, hidden_size

    @property
    def hidden_size(self) -> int:
        return self.causal_model.config.hidden_size

    def _init_mamba_cache(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> "MambaCache":
        """Initialize a fresh Mamba SSM cache.

        Args:
            batch_size: Batch size for the cache.
            device: Device to create cache on.
            dtype: Data type for cache tensors.

        Returns:
            Initialized MambaCache ready for use.
        """
        tfm = importlib.import_module("transformers")
        MambaCacheClass = getattr(tfm, "MambaCache")
        return MambaCacheClass(
            config=self.causal_model.config,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def encode(
        self,
        observation: "Observation",
        *,
        cache: Optional[VLMCache] = None,
        actions: Optional[torch.Tensor] = None,
        extra_embs: Optional[torch.Tensor] = None,
        extra_pad_masks: Optional[torch.Tensor] = None,
        extra_att_masks: Optional[torch.Tensor] = None,
        zero_pos_id_for_extra: bool = False,
        extra_attention_mask: bool = False,
    ) -> Tuple[torch.Tensor, MambaVLMOutputs, Optional[VLMCache], Dict]:
        """Encode observation through Mamba with SigLIP vision features.

        Args:
            observation: Contains policy_input with input_ids, pixel_values, etc.
            cache: Optional MambaCache from previous timestep for temporal state.
                   If None and use_temporal_cache is enabled, a fresh cache is created.
            extra_embs: Additional embeddings to append (observation registers)
            extra_pad_masks: Padding masks for extra embeddings
            extra_att_masks: Attention masks for extra embeddings
            zero_pos_id_for_extra: Zero out position IDs for extra embeddings
            extra_attention_mask: Apply causal masking to extra embeddings

        Returns:
            ce_loss: Cross-entropy loss on language modeling task
            vlm_outputs: MambaVLMOutputs containing hidden states
            cache: Updated MambaCache for next timestep (or None if not using cache)
            log_dict: Logging metrics
        """
        policy_td: MambaPolicyInput = observation.policy_input
        if not isinstance(policy_td, MambaPolicyInput):
            raise TypeError("Observation policy_input must be MambaPolicyInput")

        input_ids = policy_td.input_ids
        input_pad_masks = policy_td.attention_mask
        target_ids = policy_td.target_ids
        pixel_values = policy_td.pixel_values

        # Mamba bridge requires 6D pixel_values: (batch, time, cameras, C, H, W)
        if pixel_values.ndim != 6:
            raise ValueError(
                f"MambaBridge.encode expects policy_input.pixel_values to be 6D "
                f"(batch, time, cameras, C, H, W), got ndim={pixel_values.ndim}"
            )
        bsz, T, num_cams = pixel_values.shape[0], pixel_values.shape[1], pixel_values.shape[2]
        device = input_ids.device
        pv_trailing = pixel_values.shape[3], pixel_values.shape[4], pixel_values.shape[5]

        # Fold: (B, T, Cameras, C, H, W) -> (B*T*Cameras, C, H, W)
        torch.cuda.nvtx.range_push("encode_vision")
        pixel_values_flat = pixel_values.view(bsz * T * num_cams, *pv_trailing)
        vision_outputs = self.vision_encoder(pixel_values=pixel_values_flat)
        image_embeds = vision_outputs.last_hidden_state
        image_embeds = self.vision_projector(image_embeds)
        num_patches = image_embeds.shape[1]
        hidden_size = image_embeds.shape[2]
        # Unfold: (B*T*Cameras, num_patches, hidden) -> (B, T, Cameras, num_patches, hidden)
        image_embeds = image_embeds.view(
            bsz, T, num_cams, num_patches, hidden_size
        )
        # Per-timestep vision: (B, T, Cameras, num_patches, hidden)
        vis_per_t = num_cams * num_patches
        torch.cuda.nvtx.range_pop()

        # Get text embeddings from Mamba (used only when actions is not None / training)
        torch.cuda.nvtx.range_push("embed_text")
        mamba_model = self._get_mamba_backbone()
        text_embeds = mamba_model.embeddings(input_ids)
        text_len = text_embeds.shape[1]
        torch.cuda.nvtx.range_pop()

        # Build interleaved sequence: [text?, vis_0, proprio_0, act_0, ..., vis_{T-1}, proprio_{T-1}]
        torch.cuda.nvtx.range_push("merge_inputs")
        state = observation.state
        if self.state_proj is not None and state is not None:
            # state (B, T, state_history, state_dim) or (B, state_history, state_dim) for T=1
            if state.ndim == 3:
                state = state.unsqueeze(1)
            state_flat = state.reshape(bsz, T, -1)
            proprio_embeds = self.state_proj(state_flat.to(self.state_proj.weight.dtype)).unsqueeze(2)  # (B, T, 1, hidden)
        else:
            proprio_embeds = None

        if actions is not None and self.action_proj is not None:
            # actions (B, T-1, action_horizon, action_dim) -> (B, T-1, hidden)
            act_flat = actions.reshape(bsz, T - 1, -1)
            act_embeds = self.action_proj(act_flat.to(self.action_proj.weight.dtype)).unsqueeze(2)  # (B, T-1, 1, hidden)
        else:
            act_embeds = None

        # Vectorized interleaving: [text?, body (t=0..T-2: vis, proprio, act), tail (t=T-1: vis, proprio)]
        image_embeds_bt = image_embeds.view(bsz, T, vis_per_t, hidden_size)
        # Body: t=0..T-2 (T-1 steps). If T=1, body is empty.
        imgs_body = image_embeds_bt[:, :-1]  # (B, T-1, vis_per_t, D)
        imgs_tail = image_embeds_bt[:, -1:]  # (B, 1, vis_per_t, D)
        body_components = [imgs_body]
        if proprio_embeds is not None:
            props_body = proprio_embeds[:, :-1]  # (B, T-1, 1, D)
            props_tail = proprio_embeds[:, -1:]  # (B, 1, 1, D)
            body_components.append(props_body)
        if act_embeds is not None:
            body_components.append(act_embeds)  # (B, T-1, 1, D)
        body_block = torch.cat(body_components, dim=2)  # (B, T-1, tokens_per_step, D)
        body_seq = body_block.flatten(1, 2)  # (B, (T-1)*tokens_per_step, D)
        tail_components = [imgs_tail]
        if proprio_embeds is not None:
            tail_components.append(props_tail)
        tail_block = torch.cat(tail_components, dim=2)  # (B, 1, tokens_per_tail, D)
        tail_seq = tail_block.flatten(1, 2)  # (B, tokens_per_tail, D)
        final_parts = []
        pad_parts = []
        has_text = actions is not None and text_len > 0
        if has_text:
            final_parts.append(text_embeds)
            pad_parts.append(input_pad_masks)
        body_seq_len = body_seq.shape[1]
        tail_seq_len = tail_seq.shape[1]
        final_parts.append(body_seq)
        pad_parts.append(torch.ones(bsz, body_seq_len, dtype=torch.bool, device=device))
        final_parts.append(tail_seq)
        pad_parts.append(torch.ones(bsz, tail_seq_len, dtype=torch.bool, device=device))
        merged_embeds = torch.cat(final_parts, dim=1)
        merged_pad_masks = torch.cat(pad_parts, dim=1)

        # Indices of last token of proprio_obs_t (or last vis_t if no proprio) in sequence
        base = text_len if has_text else 0
        tokens_per_step_body = vis_per_t + (1 if proprio_embeds is not None else 0) + (1 if act_embeds is not None else 0)
        if proprio_embeds is not None:
            # Last token of proprio_t at position vis_per_t within each step
            proprio_t_last_indices = (
                torch.arange(T, device=device, dtype=torch.long).mul(tokens_per_step_body).add(vis_per_t).add(base)
            )
        else:
            # No proprio: use last token of vis_t (position vis_per_t - 1 within step)
            proprio_t_last_indices = (
                torch.arange(T, device=device, dtype=torch.long).mul(tokens_per_step_body).add(vis_per_t - 1).add(base)
            )

        torch.cuda.nvtx.range_pop()

        # Handle extra embeddings (observation registers)
        embs = [merged_embeds]
        pad_masks = [merged_pad_masks]
        merged_seq_len = merged_embeds.shape[1]
        att_masks = [
            torch.ones(merged_seq_len, dtype=torch.bool, device=device),
        ]

        extra_len = 0
        if extra_embs is not None:
            embs.append(extra_embs)
            pad_masks.append(extra_pad_masks)
            att_masks.append(extra_att_masks)
            extra_len = extra_embs.shape[1]

        embs = torch.cat(embs, dim=1)
        prefix_pad_masks = torch.cat(pad_masks, dim=1)
        prefix_att_masks_1d = torch.cat(att_masks, dim=0).expand(bsz, -1)

        seq_len = embs.shape[1]

        # Initialize or validate cache for temporal state propagation.
        # If cache is provided, we're continuing from a previous timestep.
        # If cache is None and actions is None (inference), create a fresh cache so the caller can store it.
        use_cache = cache is not None
        if cache is None and actions is None:
            cache = self._init_mamba_cache(bsz, device, embs.dtype)
            use_cache = True
            cache_position = torch.arange(
                0, seq_len, device=device, dtype=torch.long
            )
        elif use_cache:
            prev_seq_len = cache.conv_states.shape[2] if hasattr(cache, "conv_states") and cache.conv_states is not None else 0
            cache_position = torch.arange(
                prev_seq_len, prev_seq_len + seq_len, device=device, dtype=torch.long
            )
        else:
            cache_position = None

        # Forward through Mamba layers, collecting hidden states
        torch.cuda.nvtx.range_push("mamba_forward")
        hidden_states = embs
        encoder_hidden_states_list = []

        for layer_idx, layer in enumerate(mamba_model.layers):
            torch.cuda.nvtx.range_push(f"mamba_layer_{layer_idx}")
            # Pass cache_params and cache_position to enable stateful processing
            if use_cache:
                hidden_states = apply_checkpoint_when_training(
                    self,
                    layer,
                    hidden_states,
                    cache_params=cache,
                    cache_position=cache_position,
                )
            else:
                hidden_states = apply_checkpoint_when_training(
                    self,
                    layer,
                    hidden_states,
                )
            torch.cuda.nvtx.range_pop()
            encoder_hidden_states_list.append(hidden_states)

        # Final normalization
        hidden_states = mamba_model.norm_f(hidden_states)
        torch.cuda.nvtx.range_pop()

        # Stack hidden states: (batch, num_layers, seq_len, hidden)
        hidden_state_list = torch.stack(encoder_hidden_states_list, dim=1)

        # Compute cross-entropy loss on text tokens only when text is in sequence (training)
        # At inference (actions is None) sequence is [vis_t, proprio_t] only, no text
        if actions is not None and text_len > 0:
            text_hidden = hidden_states[:, :text_len]
            text_hidden = text_hidden.to(self.causal_model.lm_head.weight.dtype)
            pred_logits = self.causal_model.lm_head(text_hidden)
            pred_logits = pred_logits[:, :-1]
            pred_logits = pred_logits.reshape(-1, pred_logits.shape[-1])
            target_flat = target_ids[:, 1:].reshape(-1)
            ce_loss_sum = torch.nn.functional.cross_entropy(
                pred_logits,
                target_flat,
                ignore_index=TARGET_IGNORE_ID,
                reduction="sum",
            )
            num_correct_tokens = (pred_logits.argmax(dim=-1) == target_flat).sum()
            total = (target_flat != TARGET_IGNORE_ID).sum().clamp(min=1)
            ce_loss = ce_loss_sum / total
            accuracy = num_correct_tokens.float() / total
        else:
            ce_loss = torch.tensor(0.0, device=device, dtype=hidden_states.dtype)
            accuracy = torch.tensor(0.0, device=device, dtype=hidden_states.dtype)

        vlm_outputs = MambaVLMOutputs(
            last_hidden_state=hidden_states,
            prefix_pad_masks=prefix_pad_masks,
            hidden_state_list=hidden_state_list,
            conv_states=None,  # Cache is returned separately
            ssm_states=None,
            batch_size=[bsz],
            proprio_t_last_indices=proprio_t_last_indices,
        )

        padding_ratio = policy_td.attention_mask.float().mean(dim=-1)
        log_dict = {
            "padding_ratio/mean": padding_ratio.mean(),
            "padding_ratio/std": padding_ratio.std() if padding_ratio.numel() > 1 else torch.tensor(0.0, device=padding_ratio.device),
            "padding_ratio/min": padding_ratio.min(),
            "padding_ratio/max": padding_ratio.max(),
            "loss/ce_loss": ce_loss.detach(),
            "loss/accuracy": accuracy.detach(),
        }

        # Return updated cache (same object, modified in-place by Mamba layers)
        return ce_loss, vlm_outputs, cache, log_dict

    def _get_mamba_backbone(self):
        """Get the Mamba backbone model (handles different HF model structures)."""
        if hasattr(self.causal_model, "backbone"):
            return self.causal_model.backbone
        elif hasattr(self.causal_model, "model"):
            return self.causal_model.model
        else:
            raise AttributeError(
                "Could not find Mamba backbone. Expected 'backbone' or 'model' attribute."
            )
