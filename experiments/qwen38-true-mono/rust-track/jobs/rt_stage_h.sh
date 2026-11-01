#!/usr/bin/env bash
# rt_stage_h.sh <version>: the harness stage rt-h-<version>: hackathon-prep/qwen38/{common,harness,gdn-b3,moe-fused,
# moe,tests} + this rust-track directory (scripts and jobs).
set -euo pipefail
V=${1:?version}
D=/c/Users/vanantha/Documents/dev/rt-work/stage-rt-h-$V
Q=/c/Users/vanantha/Documents/dev/hackathon-prep/qwen38
RT=/c/Users/vanantha/Documents/dev/copy-rocm-lib/experiments/qwen38-true-mono/rust-track
mkdir -p "$D"
for d in common harness gdn-b3 moe-fused moe tests; do mkdir -p "$D/$d"; cp -r "$Q/$d/." "$D/$d/"; done
mkdir -p "$D/rust-track/jobs" "$D/rust-pilot"
cp "$RT"/*.py "$D/rust-track/"; cp "$RT"/jobs/*.sh "$D/rust-track/jobs/"
cp "$RT"/../rust-pilot/*.py "$D/rust-pilot/"
find "$D" -name __pycache__ -prune -exec rm -rf {} + 2> /dev/null || true
bash /c/Users/vanantha/Documents/dev/hackathon-prep/tools/q38/q38_sync.sh "$D" "rt-h-$V" | tail -1
