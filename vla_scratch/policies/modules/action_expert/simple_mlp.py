from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class MLPConfig:
    """Configuration for the simple MLP action expert."""

    # MLP architecture
    hidden_size: int = 1024
    num_hidden_layers: int = 4
    intermediate_size: int = 4096
    activation: str = "gelu"
    dropout: float = 0.0

    # How to consume VLM hidden states
    use_last_layer_only: bool = True  # True: only final VLM layer, False: average all
    use_mean_pooling: bool = True  # True: pool sequence to single vector

    # Additional inputs
    concat_state: bool = True  # Concatenate proprioceptive state to input

    # Output
    action_horizon: int = 10

    # Normalization
    layer_norm_eps: float = 1e-6


class MLPBlock(nn.Module):
    """Single MLP block with residual connection."""

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

        if config.activation == "gelu":
            self.activation = nn.GELU()
        elif config.activation == "silu":
            self.activation = nn.SiLU()
        elif config.activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {config.activation}")

        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class MLPModel(nn.Module):
    """Simple MLP action expert without diffusion.

    Takes VLM hidden states and outputs action chunk directly in a single forward pass.

    Input flow:
    1. hidden_state_list: (batch, n_layer, seq_len, hidden) from VLM
    2. Select layer(s): last only OR average all layers → (batch, seq_len, hidden)
    3. Pool sequence: mean pooling → (batch, vlm_hidden)
    4. Optionally concatenate state → (batch, vlm_hidden + state_dim)
    5. Project to MLP hidden size → (batch, mlp_hidden)
    6. Pass through MLP blocks → (batch, mlp_hidden)
    7. Project to action output → (batch, action_horizon * action_dim)
    8. Reshape → (batch, action_horizon, action_dim)
    """

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config

        # MLP blocks
        self.blocks = nn.ModuleList(
            [MLPBlock(config) for _ in range(config.num_hidden_layers)]
        )

        # Input projection (lazily initialized based on actual input size)
        self.input_proj: Optional[nn.Linear] = None

        # Output projection (lazily initialized based on action_dim)
        self.output_proj: Optional[nn.Linear] = None

        # Final normalization
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def _ensure_projections(
        self, input_dim: int, action_dim: int, device: torch.device, dtype: torch.dtype
    ):
        """Initialize input/output projections based on actual dimensions."""
        if self.input_proj is None:
            self.input_proj = nn.Linear(input_dim, self.config.hidden_size)
            self.input_proj = self.input_proj.to(device=device, dtype=dtype)

        if self.output_proj is None:
            self.output_proj = nn.Linear(
                self.config.hidden_size, self.config.action_horizon * action_dim
            )
            self.output_proj = self.output_proj.to(device=device, dtype=dtype)

    def initialize_weights(self):
        """Initialize weights with Xavier uniform."""

        def init_fn(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(init_fn)

    def forward(
        self,
        encoder_hidden_states: List[torch.Tensor],
        prefix_pad_masks: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        action_dim: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Forward pass through MLP action expert.

        Args:
            encoder_hidden_states: List of hidden states from VLM layers,
                each of shape (batch, seq_len, hidden)
            prefix_pad_masks: Attention mask for valid positions (batch, seq_len)
            state: Optional proprioceptive state (batch, state_history, state_dim)
                or (batch, state_dim)
            action_dim: Action dimension for output projection

        Returns:
            loss: Placeholder (0.0, actual loss computed in policy)
            actions: Predicted action chunk (batch, action_horizon, action_dim)
            log_dict: Empty dict (logging done in policy)
        """
        if action_dim is None:
            raise ValueError("action_dim must be provided")

        batch_size = encoder_hidden_states[0].shape[0]
        device = encoder_hidden_states[0].device
        dtype = encoder_hidden_states[0].dtype

        # Step 1: Select which VLM layers to use
        if self.config.use_last_layer_only:
            hidden = encoder_hidden_states[-1]  # (batch, seq_len, hidden)
        else:
            # Average across all layers
            stacked = torch.stack(encoder_hidden_states, dim=0)  # (n_layer, batch, seq, hidden)
            hidden = stacked.mean(dim=0)  # (batch, seq_len, hidden)

        # Step 2: Pool across sequence dimension
        if self.config.use_mean_pooling:
            # Masked mean pooling
            mask = prefix_pad_masks.unsqueeze(-1).float()  # (batch, seq_len, 1)
            hidden = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            # hidden: (batch, vlm_hidden)
        else:
            # Use last valid token position
            seq_lengths = prefix_pad_masks.sum(dim=1) - 1  # (batch,)
            seq_lengths = seq_lengths.clamp(min=0).long()
            hidden = hidden[torch.arange(batch_size, device=device), seq_lengths]
            # hidden: (batch, vlm_hidden)

        vlm_hidden = hidden.shape[-1]

        # Step 3: Optionally concatenate proprioceptive state
        if self.config.concat_state and state is not None:
            if state.dim() == 3:  # (batch, history, dim)
                state = state.reshape(batch_size, -1)
            hidden = torch.cat([hidden, state], dim=-1)

        input_dim = hidden.shape[-1]

        # Ensure projections are initialized
        self._ensure_projections(input_dim, action_dim, device, dtype)

        # Step 4: Project to MLP hidden size
        hidden = self.input_proj(hidden)  # (batch, hidden_size)

        # Step 5: Pass through MLP blocks
        for block in self.blocks:
            hidden = block(hidden)

        # Step 6: Final norm and output projection
        hidden = self.final_norm(hidden)
        actions = self.output_proj(hidden)  # (batch, action_horizon * action_dim)

        # Step 7: Reshape to action chunk
        actions = actions.reshape(batch_size, self.config.action_horizon, action_dim)

        return 0.0, actions, {}
