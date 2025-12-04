from tensordict import TensorClass
import torch
from typing import Optional, Tuple

class VLMOutputs(TensorClass):
    last_hidden_state: torch.Tensor
    prefix_pad_masks: torch.Tensor
    hidden_state_list: Optional[Tuple[torch.Tensor, ...]] = None
    kv_cache_list: Optional[Tuple[torch.Tensor, ...]] = None