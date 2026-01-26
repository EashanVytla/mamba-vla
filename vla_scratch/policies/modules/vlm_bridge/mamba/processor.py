import importlib
from typing import Dict, List, Optional, TYPE_CHECKING

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

        # Process images with SigLIP
        pixel_values = self.image_processor(
            images=images,
            return_tensors="pt",
        )["pixel_values"]

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

        policy_td = MambaPolicyInput(
            input_ids=input_ids.squeeze(0).long(),
            target_ids=target_ids.squeeze(0).long(),
            attention_mask=attention_mask.squeeze(0).bool(),
            obs_register_att_mask=obs_register_att_mask.squeeze(0).bool(),
            pixel_values=pixel_values,
        )
        sample.observation.policy_input = policy_td
        return sample

    def _build_assistant_mask(self, input_ids: torch.LongTensor) -> torch.BoolTensor:
        """Build mask for assistant response tokens (tokens to train on)."""
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        bsz, seqlen = input_ids.shape
        mask = torch.zeros((bsz, seqlen), dtype=torch.bool, device=input_ids.device)

        for b in range(bsz):
            ids_list = input_ids[b].tolist()
            marker_start = self._find_subsequence(ids_list, self.assistant_marker_ids)
            if marker_start is None:
                continue
            # Mask everything after the assistant marker
            content_start = marker_start + len(self.assistant_marker_ids)
            mask[b, content_start:] = True

        return mask

    def _build_obs_register_att_mask(
        self, input_ids: torch.LongTensor, attention_mask: torch.Tensor
    ) -> torch.BoolTensor:
        """Build mask for observation context (before prompt separator)."""
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)

        bsz, seqlen = input_ids.shape
        mask = torch.zeros((bsz, seqlen), dtype=torch.bool, device=input_ids.device)

        for b in range(bsz):
            sep_start = self._find_subsequence(
                input_ids[b].tolist(), self.prompt_sep_ids
            )
            if sep_start is None:
                mask[b] = attention_mask[b].bool()
            else:
                mask[b, :sep_start] = attention_mask[b, :sep_start].bool()

        return mask

    @staticmethod
    def _find_subsequence(
        sequence: list[int], subsequence: list[int]
    ) -> Optional[int]:
        """Find the starting index of a subsequence in a sequence."""
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.tolist()
        if not subsequence:
            return None
        max_start = len(sequence) - len(subsequence)
        for i in range(max_start + 1):
            if sequence[i : i + len(subsequence)] == subsequence:
                return i
        return None
