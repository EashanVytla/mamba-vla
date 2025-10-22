import torch
from typing import Tuple

from vla_scratch.datasets.data_types import DataSample, Observation, ActionChunk
from vla_scratch.policies.pi.config import PiConfig
from vla_scratch.policies.pi.policy import PiPolicy
from vla_scratch.policies.pi.utils import *


def build_dummy_observation(
    batch_size: int,
    prompt_len: int,
    num_cameras: int,
    image_shape: Tuple[int, int, int],
    action_horizon: int,
    action_dim: int,
    device: torch.device,
) -> DataSample:
    """Create random tensors that match the shapes PI0Pytorch expects."""

    images = torch.randn(
        (batch_size, num_cameras, *image_shape),
        dtype=torch.float32,
        device=device,
    )
    image_masks = torch.ones(
        batch_size,
        num_cameras,
        1,
        dtype=torch.bool,
        device=device,
    )

    state = torch.randn(batch_size, action_dim, dtype=torch.float32, device=device)
    tokenized_prompt = torch.randint(
        0,
        1000,
        (batch_size, prompt_len),
        dtype=torch.long,
        device=device,
    )
    tokenized_prompt_mask = torch.ones(
        batch_size,
        prompt_len,
        dtype=torch.bool,
        device=device,
    )

    observation = Observation(
        images=images,
        image_masks=image_masks,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        state=state,
    )
    action = ActionChunk(
        actions=torch.zeros(batch_size, action_horizon, action_dim, dtype=torch.float32, device=device)
    )
    return DataSample(observation=observation, action=action, batch_size=batch_size)
    


def main() -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config = PiConfig()
    model = PiPolicy(config)
    model.to(device)
    model.eval()

    data_sample = build_dummy_observation(
        batch_size=32,
        prompt_len=8,
        num_cameras=3,
        image_shape=(3, 224, 224),
        action_dim=config.action_dim,
        action_horizon=config.action_horizon,
        device=device,
    )

    time_dist = get_beta_dist(1.0, 1.5, device)

    with torch.inference_mode():
        actions = data_sample.action.actions
        noise = sample_noise(actions.shape, dtype=actions.dtype, device=device)
        # shape: (B, action_horizon, action_dim)

        time = sample_time(time_dist, data_sample.batch_size)
        time_expanded = time[:, None, None]

        noisy_actions = time_expanded * noise + (1 - time_expanded) * actions

        _, prefix_pad_masks, prefix_key_values = model.encode_prefix(
            observation=data_sample.observation,
        )
        v_t = model.predict_suffix(
            state=data_sample.observation.state,
            prefix_pad_masks=prefix_pad_masks,
            prefix_key_values=prefix_key_values,
            noisy_actions=noisy_actions,
            time=time,
        )
        print("v_t shape:", v_t.shape)
        predicted_actions = model.sample_actions(observation=data_sample.observation, num_steps=10)
        print("predicted_actions shape:", predicted_actions.shape)


if __name__ == "__main__":
    main()
