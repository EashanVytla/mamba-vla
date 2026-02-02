#!/usr/bin/env python3
"""
Post-install script to patch mamba-ssm for transformers compatibility.

mamba-ssm 2.x doesn't export `selective_state_update` at the top level,
but transformers 5.x expects it. This script patches the mamba-ssm __init__.py
to add the missing exports.

Run this after installing/reinstalling mamba-ssm:
    uv run python scripts/patch_mamba_ssm.py

Or add to your job script:
    uv run python scripts/patch_mamba_ssm.py
"""

import sys
from pathlib import Path

PATCHED_INIT = '''\
__version__ = "2.3.0"

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# Export selective_state_update for transformers compatibility
# Patched by scripts/patch_mamba_ssm.py
try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

# Also try to import causal_conv1d functions if available
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None
'''

PATCH_MARKER = "# Patched by scripts/patch_mamba_ssm.py"


def find_mamba_ssm_init() -> Path:
    """Find the mamba_ssm __init__.py file."""
    try:
        import mamba_ssm
        init_path = Path(mamba_ssm.__file__)
        if init_path.name == "__init__.py":
            return init_path
        # If __file__ points to something else, try parent
        return init_path.parent / "__init__.py"
    except ImportError:
        print("❌ mamba-ssm is not installed. Install it first:")
        print("   uv pip install mamba-ssm --no-build-isolation --no-binary mamba-ssm")
        sys.exit(1)
    except Exception:
        # On login nodes without GPU, mamba_ssm import may fail due to triton
        # Try to find it manually
        pass
    
    # Fallback: search in site-packages
    import site
    for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
        init_path = Path(site_dir) / "mamba_ssm" / "__init__.py"
        if init_path.exists():
            return init_path
    
    # Try virtual environment
    venv_path = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "mamba_ssm" / "__init__.py"
    if venv_path.exists():
        return venv_path
    
    print("❌ Could not find mamba_ssm installation. Is it installed?")
    sys.exit(1)


def is_already_patched(init_path: Path) -> bool:
    """Check if the __init__.py is already patched."""
    content = init_path.read_text()
    return PATCH_MARKER in content


def patch_mamba_ssm():
    """Patch mamba-ssm __init__.py to export selective_state_update."""
    init_path = find_mamba_ssm_init()
    print(f"📁 Found mamba_ssm at: {init_path}")
    
    if is_already_patched(init_path):
        print("✅ mamba-ssm is already patched. Nothing to do.")
        return
    
    # Read current content for version check
    current_content = init_path.read_text()
    
    # Check version
    if '__version__ = "2.3.0"' not in current_content:
        print("⚠️  Warning: mamba-ssm version may differ from expected 2.3.0")
        print("   Proceeding with patch anyway...")
    
    # Write patched content
    init_path.write_text(PATCHED_INIT)
    print("✅ Successfully patched mamba-ssm __init__.py")
    print("   Added exports: selective_state_update, causal_conv1d_fn, causal_conv1d_update")


if __name__ == "__main__":
    patch_mamba_ssm()
