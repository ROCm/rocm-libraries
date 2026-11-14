#!/bin/bash
# AICK-3062 gfx1151 kit: one-time setup on a Strix Halo (gfx1151) machine.
# Creates $AICK_WORK/venv with torch 2.13.0+rocm7.2 (it bundles the ROCm 7.2 runtime and
# comgr the rocKE resource gate expects) and this checkout's rocKE packages, editable.
set -euo pipefail
KIT=$(cd "$(dirname "$0")" && pwd)
ROCKE=$(cd "$KIT/../dnn-providers/hip-kernel-provider/rocke" && pwd)
WORK=${AICK_WORK:-$HOME/aick3062_gfx1151_work}
mkdir -p "$WORK"

command -v rocminfo >/dev/null || { echo "rocminfo not found: install it (ROCm 'rocminfo' package)"; exit 1; }
rocminfo | grep -qE 'Name: +gfx1151' || { echo "no gfx1151 GPU visible to rocminfo"; exit 1; }

PYBIN=$(command -v python3.12 || command -v python3)
"$PYBIN" -m venv "$WORK/venv"
V="$WORK/venv/bin"
"$V/pip" install -q --upgrade pip "setuptools>=61" wheel
"$V/pip" install --index-url https://download.pytorch.org/whl/rocm7.2 \
  --extra-index-url https://pypi.org/simple "torch==2.13.0+rocm7.2"
"$V/pip" install -q --config-settings editable_mode=compat -e "$ROCKE/platform[dev]"
"$V/pip" install -q --config-settings editable_mode=compat --no-deps -e "$ROCKE/library"
"$V/python" - <<'P'
import torch
assert torch.cuda.is_available(), "torch sees no GPU"
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
P
echo "SETUP_OK work dir: $WORK"
