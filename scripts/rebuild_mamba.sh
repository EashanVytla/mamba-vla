#!/bin/bash
# Rebuild mamba-ssm and causal-conv1d from source with correct CUDA/PyTorch versions
# Run this once after setting up your environment

set -e

echo "🔧 Rebuilding mamba-ssm and causal-conv1d from source..."

# Uninstall existing packages
echo "Uninstalling existing packages..."
uv pip uninstall -y mamba-ssm causal-conv1d || true

# Install causal-conv1d from source first (dependency)
echo "Installing causal-conv1d from source..."
uv pip install causal-conv1d --no-build-isolation --no-binary causal-conv1d

# Install mamba-ssm from source
echo "Installing mamba-ssm from source..."
uv pip install mamba-ssm --no-build-isolation --no-binary mamba-ssm

echo "✅ Rebuild complete! The fast CUDA kernels should now be available."
