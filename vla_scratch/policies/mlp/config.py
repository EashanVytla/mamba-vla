from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

from vla_scratch.policies.config import PolicyConfig
from vla_scratch.policies.modules.action_expert.simple_mlp import MLPConfig


@dataclass
class MLPPolicyConfig(PolicyConfig):
    """Configuration for MLPPolicy with simple MLP action expert."""

    vlm_type: str
    model_id: str

    # Vision encoder for Mamba (ignored for transformer VLMs)
    vision_encoder_id: str = "google/siglip-base-patch16-224"

    # MLP action expert configuration
    action_expert_cfg: MLPConfig = field(
        default_factory=lambda: MLPConfig(
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=4096,
            use_last_layer_only=True,
            use_mean_pooling=True,
            concat_state=True,
        )
    )

    # Architecture
    num_obs_registers: int = 4
    expert_only_use_register: bool = True

    # Training
    detach_encoder_output: bool = False
    ce_loss_weight: float = 0.1

    # Misc
    obs_register_init_gain: float = 0.02
    zero_pos_id_for_obs_register: bool = True
    causal_mask_obs_register: bool = True


# Example configuration for MLP policy with SmolVLM
mlp_smolvlm_config = MLPPolicyConfig(
    _target_="vla_scratch.policies.mlp.policy.MLPPolicy",
    state_history=1,
    action_horizon=10,
    model_id="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    vlm_type="SmolVLMForConditionalGeneration",
    transforms=[
        {
            "_target_": "vla_scratch.policies.modules.vlm_bridge.smolvlm.processor.SmolVLMProcessor",
            "processor_class": "SmolVLMProcessor",
            "model_id": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
            "max_length": 180,
            "padding": "max_length",
            "image_size_longest_edge": 512,
            "max_image_size_longest_edge": 512,
        }
    ],
)

# Example configuration for MLP policy with Mamba
mlp_mamba_config = MLPPolicyConfig(
    _target_="vla_scratch.policies.mlp.policy.MLPPolicy",
    state_history=1,
    action_horizon=10,
    model_id="state-spaces/mamba-2.8b-hf",
    vlm_type="MambaForCausalLM",
    vision_encoder_id="google/siglip-base-patch16-224",
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
cs.store(name="mlp-smol", node=mlp_smolvlm_config, group="policy")
cs.store(name="mlp-mamba", node=mlp_mamba_config, group="policy")
