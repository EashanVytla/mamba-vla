from tensordict import TensorClass
import torch
import jaxtyping as at

class VLMOutputs(TensorClass):
    last_hidden_state: at.Float[torch.Tensor, "*b seq_len hidden"]
    prefix_pad_masks: at.Bool[torch.Tensor, "*b seq_len"]
    key_states: at.Float[torch.Tensor, "*b n_layer n_head seq_len head_dim"]
    value_states: at.Float[torch.Tensor, "*b n_layer n_head seq_len head_dim"]
    hidden_state_list: at.Float[torch.Tensor, "*b n_layer seq_len hidden"]