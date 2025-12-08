from dataclasses import dataclass, field, replace
from hydra.core.config_store import ConfigStore
from vla_scratch.policies.config import PolicyConfig
from vla_scratch.policies.modules.action_expert.cross_attention_dit import DiTConfig


@dataclass
class PiConfig(PolicyConfig):
    vlm_type: str
    model_id: str

    action_expert_cfg: DiTConfig = field(default_factory=lambda: DiTConfig(
        # hidden size
        hidden_size=1024,
        intermediate_size=4096,
        # attention size
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        # layers
        num_hidden_layers=12,
        cross_attention_every=2,
    ))
    suffix_add_pos_emb: bool = True

    use_state: bool = True
    num_obs_registers: int = 0
    expert_only_use_register: bool = False

    num_noise_per_sample: int = 2
    num_noise_before_topk: int = 2
    detach_kv_cache: bool = False
    disp_loss_weight: float = 0.
    time_dist_alpha: float = 1.0
    time_dist_beta: float = 1.5


default_pi_config = PiConfig(
    _target_="vla_scratch.policies.pi.policy.PiPolicy",
    model_id="google/paligemma-3b-mix-224",
    vlm_type="PaliGemmaForConditionalGeneration",
    state_history=1,
    action_horizon=20,
    transforms=[
        {
            "_target_": "vla_scratch.policies.modules.vlm_bridge.paligemma.processor.PaligemmaProcessor",
            "processor_class": "PaliGemmaProcessor",
            "model_id": "google/paligemma-3b-mix-224",
            "max_length": 64,
            "target_size": (224, 224),
        }
    ],
)

Cs = ConfigStore.instance()
Cs.store(
    name="pi",
    node=default_pi_config,
    group="policy",
)
Cs.store(
    name="pi-qwen",
    node=replace(
        default_pi_config,
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        vlm_type="Qwen3VLForConditionalGeneration",
        transforms=[
            {
                "_target_": "vla_scratch.policies.modules.vlm_bridge.qwen.processor.QwenProcessor",
                "processor_class": "Qwen3VLProcessor",
                "model_id": "Qwen/Qwen3-VL-2B-Instruct",
                "max_length": 160, 
                # WARN: select this based on your image sizes and prompt lengths, try to make it minimum as possible because if impacts iteration time a lot!
                "add_generation_prompt": True,
                "padding": "max_length",
            }
        ],
    ),
    group="policy",
)
