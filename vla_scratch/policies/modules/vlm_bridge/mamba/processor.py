import importlib
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
from tensordict import TensorClass

from vla_scratch.transforms.base import TransformFn
from vla_scratch.policies.modules.vlm_bridge.base import TARGET_IGNORE_ID

if TYPE_CHECKING:
    from vla_scratch.transforms.data_types import DataSample
    from transformers import AutoTokenizer, SiglipImageProcessor


class MambaPolicyInput(TensorClass):
    """Policy input for Mamba VLM bridge with SigLIP vision encoder."""

    input_ids: torch.LongTensor
    target_ids: torch.LongTensor
    attention_mask: torch.BoolTensor
    obs_register_att_mask: torch.BoolTensor
    pixel_values: torch.FloatTensor


class MambaProcessor(TransformFn):
    """Tokenize prompt for Mamba and process images with SigLIP."""

    def __init__(
        self,
        model_id: str,
        vision_encoder_id: str = "google/siglip-base-patch16-224",
        max_length: int = 256,
        padding: str | bool = "max_length",
        image_size: int = 224,
    ) -> None:
        transformers = importlib.import_module("transformers")

        # Load Mamba tokenizer
        self.tokenizer: "AutoTokenizer" = transformers.AutoTokenizer.from_pretrained(
            model_id
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load SigLIP image processor
        self.image_processor: "SiglipImageProcessor" = (
            transformers.SiglipImageProcessor.from_pretrained(vision_encoder_id)
        )
        if image_size is not None:
            self.image_processor.size = {"height": image_size, "width": image_size}

        self.max_length = max_length
        self.padding = padding

        # Markers for building masks
        self.prompt_sep_text = "<<<PROMPT_SEP>>>"
        self.prompt_sep_ids = self.tokenizer.encode(
            self.prompt_sep_text, add_special_tokens=False
        )
        self.assistant_marker = "\nAssistant:"
        self.assistant_marker_ids = self.tokenizer.encode(
            self.assistant_marker, add_special_tokens=False
        )

    def compute(self, sample: "DataSample") -> "DataSample":
        images = sample.observation.images
        # Mamba processor only supports temporal input: 6D (B, T, cameras, C, H, W) or 5D (T, cameras, C, H, W) per sample
        if images.ndim == 5:
            images = images.unsqueeze(0)
        if images.ndim != 6:
            raise ValueError(
                f"MambaProcessor expects observation.images to be 5D (T, cameras, C, H, W) or 6D (B, T, cameras, C, H, W), "
                f"got ndim={sample.observation.images.ndim}."
            )
        B, T, num_cams = images.shape[0], images.shape[1], images.shape[2]

        # Flatten (B, T, num_cams, C, H, W) -> (B*T*num_cams, C, H, W)
        flattened_images = images.view(-1, *images.shape[-3:])
        # Single sync: move whole batch to CPU and numpy once; then slice into list for processor
        cpu_batch = flattened_images.cpu().numpy()
        N = cpu_batch.shape[0]
        if flattened_images.dim() == 4 and flattened_images.shape[1] == 3:
            # (N, C, H, W) -> list of (H, W, C) for SigLIP
            img_list = [cpu_batch[i].transpose(1, 2, 0) for i in range(N)]
        else:
            img_list = [cpu_batch[i] for i in range(N)]
        processed = self.image_processor(
            images=img_list,
            return_tensors="pt",
        )["pixel_values"]
        # (B*T*num_cams, C, H, W) -> (B, T, num_cams, C, H, W)
        C, H, W = processed.shape[1], processed.shape[2], processed.shape[3]
        pixel_values = processed.view(B, T, num_cams, C, H, W)

        # Build text prompt
        # Format: <task> <<<PROMPT_SEP>>> <generation_prompt> \nAssistant: <answer>
        prompt_text = (
            f"{sample.observation.task} "
            f"{self.prompt_sep_text} "
            f"{sample.observation.generation_prompt}"
            f"{self.assistant_marker} "
            f"{sample.observation.generation_answer}"
        )

        # Tokenize
        encoded = self.tokenizer(
            prompt_text,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # Build target_ids: mask out everything except assistant response
        target_ids = input_ids.clone()
        assistant_mask = self._build_assistant_mask(input_ids)
        target_ids[~assistant_mask] = TARGET_IGNORE_ID

        # Build obs_register_att_mask: mask up to prompt separator
        obs_register_att_mask = self._build_obs_register_att_mask(
            input_ids, attention_mask
        )

        # Expand text to batch size (B,) for temporal 6D input
        input_ids = input_ids.squeeze(0).long().unsqueeze(0).expand(B, -1)
        target_ids = target_ids.squeeze(0).long().unsqueeze(0).expand(B, -1)
        attention_mask = attention_mask.squeeze(0).bool().unsqueeze(0).expand(B, -1)
        obs_register_att_mask = obs_register_att_mask.squeeze(0).bool().unsqueeze(0).expand(B, -1)

        policy_td = MambaPolicyInput(
            input_ids=input_ids,
            target_ids=target_ids,
            attention_mask=attention_mask,
            obs_register_att_mask=obs_register_att_mask,
            pixel_values=pixel_values,
        )
        sample.observation.policy_input = policy_td
        return sample

    def _build_assistant_mask(self, input_ids: torch.LongTensor) -> torch.BoolTensor:
        """Build mask for assistant response tokens (tokens to train on)."""
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        bsz, seqlen = input_ids.shape[0], input_ids.shape[1]
        start_indices, has_match = self._find_subsequence_batched(
            input_ids, self.assistant_marker_ids
        )
        marker_len = len(self.assistant_marker_ids)
        content_start = start_indices + marker_len
        arange = torch.arange(seqlen, device=input_ids.device, dtype=torch.long)
        arange = arange.unsqueeze(0).expand(bsz, -1)
        mask = (arange >= content_start.unsqueeze(1)) & has_match.unsqueeze(1)
        return mask

    def _build_obs_register_att_mask(
        self, input_ids: torch.LongTensor, attention_mask: torch.Tensor
    ) -> torch.BoolTensor:
        """Build mask for observation context (before prompt separator)."""
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)

        bsz, seqlen = input_ids.shape[0], input_ids.shape[1]
        sep_start_indices, has_sep = self._find_subsequence_batched(
            input_ids, self.prompt_sep_ids
        )
        arange = torch.arange(seqlen, device=input_ids.device, dtype=torch.long)
        arange = arange.unsqueeze(0).expand(bsz, -1)
        att_bool = attention_mask.bool()
        mask_before_sep = (arange < sep_start_indices.unsqueeze(1)) & att_bool
        mask = torch.where(has_sep.unsqueeze(1), mask_before_sep, att_bool)
        return mask

    @staticmethod
    def _find_subsequence_batched(
        input_ids: torch.LongTensor, subsequence: List[int]
    ) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        """Find start index of subsequence in each row; vectorized over batch.

        Returns:
            start_indices: (bsz,) start index per row (0 when no match).
            has_match: (bsz,) True where subsequence was found.
        """
        bsz, seqlen = input_ids.shape[0], input_ids.shape[1]
        L = len(subsequence)
        if L == 0 or seqlen < L:
            return (
                torch.zeros(bsz, dtype=torch.long, device=input_ids.device),
                torch.zeros(bsz, dtype=torch.bool, device=input_ids.device),
            )
        sub_t = torch.as_tensor(
            subsequence, device=input_ids.device, dtype=input_ids.dtype
        ).view(1, 1, L)
        windows = input_ids.unfold(1, L, 1)
        eq = (windows == sub_t).all(dim=2)
        has_match = eq.any(dim=1)
        first_idx = eq.long().argmax(dim=1)
        return first_idx, has_match
