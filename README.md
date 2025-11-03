## Quickstart

```bash
uv run \
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    train_policy.py \
    policy=pi \
    policy.state_history=10 \
    policy.action_horizon=30 \
    data=libero-ipec \
    num_noise_per_sample=16 \
    num_workers=16 \
    batch_size=4 \
    epochs=10 \
    wandb.mode=online
```

## Instructions

### Datasets


```bash
hf download IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot --repo-type dataset

export HF_CACHE_TEMP=/home/elijah/.cache/huggingface/
ln -s ${HF_CACHE_TEMP}/hub/datasets--IPEC-COMMUNITY--libero_spatial_no_noops_1.0.0_lerobot/snapshots/cb7508999d4a8caa65b6448d6d700e1f347b809e ${HF_CACHE_TEMP}/lerobot/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot
```
