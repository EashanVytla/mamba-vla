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
    MambaVLMOutputs,
    TARGET_IGNORE_ID,
)
from vla_scratch.policies.modules.vlm_bridge.mamba.processor import (
    MambaPolicyInput,
)
from vla_scratch.policies.utils.transformers import make_att_2d_masks

if TYPE_CHECKING:
    from transformers import MambaForCausalLM, SiglipVisionModel
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
        )

        # Load SigLIP vision encoder
        SiglipVisionModel = getattr(tfm, "SiglipVisionModel")
        self.vision_encoder: "SiglipVisionModel" = SiglipVisionModel.from_pretrained(
            vision_encoder_id,
            device_map=torch.cuda.current_device(),
        )

        # Create projection layer: SigLIP hidden -> Mamba hidden
        vision_hidden_size = self.vision_encoder.config.hidden_size
        mamba_hidden_size = self.causal_model.config.hidden_size
        self.vision_projector = nn.Linear(vision_hidden_size, mamba_hidden_size)

        # Move projector to same device
        self.vision_projector = self.vision_projector.to(
            device=torch.cuda.current_device()
        )

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

    def encode(
        self,
        observation: "Observation",
        *,
        extra_embs: Optional[torch.Tensor] = None,
        extra_pad_masks: Optional[torch.Tensor] = None,
        extra_att_masks: Optional[torch.Tensor] = None,
        zero_pos_id_for_extra: bool = False,
        extra_attention_mask: bool = False,
    ) -> Tuple[torch.Tensor, MambaVLMOutputs, Dict]:
        """Encode observation through Mamba with SigLIP vision features.

        Args:
            observation: Contains policy_input with input_ids, pixel_values, etc.
            extra_embs: Additional embeddings to append (observation registers)
            extra_pad_masks: Padding masks for extra embeddings
            extra_att_masks: Attention masks for extra embeddings
            zero_pos_id_for_extra: Zero out position IDs for extra embeddings
            extra_attention_mask: Apply causal masking to extra embeddings

        Returns:
            ce_loss: Cross-entropy loss on language modeling task
            vlm_outputs: MambaVLMOutputs containing hidden states
            log_dict: Logging metrics
        """
        policy_td: MambaPolicyInput = observation.policy_input
        if not isinstance(policy_td, MambaPolicyInput):
            raise TypeError("Observation policy_input must be MambaPolicyInput")

        input_ids = policy_td.input_ids
        input_pad_masks = policy_td.attention_mask
        target_ids = policy_td.target_ids
        pixel_values = policy_td.pixel_values

        bsz = input_ids.shape[0]
        device = input_ids.device

        # Encode images with SigLIP
        torch.cuda.nvtx.range_push("encode_vision")
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        # vision_outputs.last_hidden_state: (batch, num_patches, vision_hidden)
        image_embeds = vision_outputs.last_hidden_state
        # Project to Mamba hidden dimension
        image_embeds = self.vision_projector(image_embeds)
        torch.cuda.nvtx.range_pop()

        # Get text embeddings from Mamba
        torch.cuda.nvtx.range_push("embed_text")
        mamba_model = self._get_mamba_backbone()
        text_embeds = mamba_model.embeddings(input_ids)
        torch.cuda.nvtx.range_pop()

        # Merge image and text embeddings (prepend image tokens)
        torch.cuda.nvtx.range_push("merge_inputs")
        merged_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        num_image_tokens = image_embeds.shape[1]

        # Extend attention mask to include image tokens
        image_pad_masks = torch.ones(
            bsz, num_image_tokens, dtype=torch.bool, device=device
        )
        merged_pad_masks = torch.cat([image_pad_masks, input_pad_masks], dim=1)
        torch.cuda.nvtx.range_pop()

        # Handle extra embeddings (observation registers)
        embs = [merged_embeds]
        pad_masks = [merged_pad_masks]
        att_masks = [
            torch.ones(
                merged_embeds.shape[1], dtype=torch.bool, device=device
            ),
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

        # Forward through Mamba layers, collecting hidden states
        torch.cuda.nvtx.range_push("mamba_forward")
        hidden_states = embs
        encoder_hidden_states_list = []

        for layer_idx, layer in enumerate(mamba_model.layers):
            torch.cuda.nvtx.range_push(f"mamba_layer_{layer_idx}")
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

        # Compute cross-entropy loss on text tokens only (excluding image tokens)
        # Target is shifted by 1 for next-token prediction
        text_hidden = hidden_states[:, num_image_tokens:-extra_len] if extra_len > 0 else hidden_states[:, num_image_tokens:]
        pred_logits = self.causal_model.lm_head(text_hidden)
        pred_logits = pred_logits[:, :-1]
        pred_logits = pred_logits.reshape(-1, pred_logits.shape[-1])

        # Shift targets for next-token prediction
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

        vlm_outputs = MambaVLMOutputs(
            last_hidden_state=hidden_states,
            prefix_pad_masks=prefix_pad_masks,
            hidden_state_list=hidden_state_list,
            conv_states=None,  # Not storing Mamba cache for training
            ssm_states=None,
            batch_size=[bsz],
        )

        padding_ratio = policy_td.attention_mask.float().mean(dim=-1)
        log_dict = {
            "padding_ratio/mean": padding_ratio.mean(),
            "padding_ratio/std": padding_ratio.std(),
            "padding_ratio/min": padding_ratio.min(),
            "padding_ratio/max": padding_ratio.max(),
            "loss/ce_loss": ce_loss.detach(),
            "loss/accuracy": accuracy.detach(),
        }
        return ce_loss, vlm_outputs, log_dict

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
