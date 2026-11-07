#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Apply hipdnn_vllm's local vLLM patches (patches/vllm/*.patch) to the vllm
# installed for PYTHON (default: python3). Idempotent: an already-applied patch
# is skipped. Refuses to patch any vllm other than the pinned version in
# BUILD_PINS.toml. Local until upstreamed (see README, "Local vLLM patches").
#
# usage: scripts/apply_vllm_patches.sh [python]
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=${1:-python3}
PIN=$(sed -n 's/^vllm *= *"\(.*\)"/\1/p' "$HERE/BUILD_PINS.toml" | head -n1)
read -r VER SITE < <("$PY" -c 'import importlib.metadata as m, os, vllm; print(m.version("vllm"), os.path.dirname(os.path.dirname(vllm.__file__)))')
if [ -z "$PIN" ] || [ "${VER%%+*}" != "${PIN%%+*}" ]; then
  echo "apply_vllm_patches: installed vllm $VER is not the pinned $PIN; not patching" >&2
  exit 1
fi
for P in "$HERE"/patches/vllm/*.patch; do
  NAME=$(basename "$P")
  if patch -p1 -R --dry-run -s -f -d "$SITE" < "$P" >/dev/null 2>&1; then
    echo "apply_vllm_patches: $NAME already applied"
  elif patch -p1 --dry-run -s -f -d "$SITE" < "$P" >/dev/null; then
    patch -p1 -s -f -d "$SITE" < "$P"
    echo "apply_vllm_patches: applied $NAME to $SITE"
  else
    echo "apply_vllm_patches: $NAME does not apply to $SITE" >&2
    exit 1
  fi
done
