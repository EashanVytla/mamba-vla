"""Config for MambaPolicy (temporal interleaved sequences, inference cache)."""

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

from vla_scratch.policies.config import PolicyConfig
from vla_scratch.policies.modules.action_expert.cross_attention_dit import (
    DiTConfig,
)


@dataclass
class MambaPolicyConfig(PolicyConfig):
    vlm_type: str = "MambaForCausalLM"
    model_id: str = "state-spaces/mamba-2.8b-hf"
    vision_encoder_id: str = "google/siglip-base-patch16-224"

    action_expert_cfg: DiTConfig = field(
        default_factory=lambda: DiTConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            head_dim=128,
            num_hidden_layers=12,
            cross_attention_every=2,
            only_attend_to_final_layer=True,
        )
    )

    use_state: bool = False
    suffix_add_pos_emb: bool = True

    num_noise_per_sample: int = 2
    time_dist_alpha: float = 1.0
    time_dist_beta: float = 1.5

    detach_encoder_output: bool = False
    ce_loss_weight: float = 0.1

    freeze_llm_backbone: bool = False
    suffix_pos_emb_init_gain: float = 0.02


mamba_policy_config = MambaPolicyConfig(
    _target_="vla_scratch.policies.mamba.policy.MambaPolicy",
    model_id="state-spaces/mamba-2.8b-hf",
    vlm_type="MambaForCausalLM",
    vision_encoder_id="google/siglip-base-patch16-224",
    use_state=False,
    action_horizon=10,
    state_history=1,
    transforms=[
        {
            "_target_": "vla_scratch.policies.modules.vlm_bridge.mamba.processor.MambaProcessor",
            "model_id": "state-spaces/mamba-2.8b-hf",
            "vision_encoder_id": "google/siglip-base-patch16-224",
            "max_length": 256,
            "padding": "max_length",
        }
    ],
)

cs = ConfigStore.instance()
cs.store(name="mamba-policy", node=mamba_policy_config, group="policy")
