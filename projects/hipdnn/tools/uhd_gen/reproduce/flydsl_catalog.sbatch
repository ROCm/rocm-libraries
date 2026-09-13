#!/bin/bash
#SBATCH --job-name=uhd-fly-cat
#SBATCH --partition=defq
#SBATCH --constraint="GFX950"
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log
#
# Build the flyDSL variant catalog: every functional tuple the pack ships, crossed with the
# scheduling knobs `kernel_match` does not pin. That cross-product IS the catalog -- without
# it every graph has exactly one applicable kernel and L2 has nothing to rank.
#
# No hipDNN build here: the kernels come from ROCm/FlyDSL plus the flydsl wheel, so this is
# a plain interpreter and finishes in minutes. The heavy job that follows consumes what this
# one stages.
set -uo pipefail

SRC=/tmp/uhd/src
OUT=/tmp/uhd/catalog
BRANCH=${UHD_BRANCH:-users/jscampb/uhd-integration-test-branch}
KEEP=${UHD_KEEP:-/exchange/uhd-950-flydsl-catalog}
step() { echo; echo "===== $* ====="; date -Is; }

step "node"
hostname; nproc; rocminfo 2>/dev/null | grep -m1 "Name:.*gfx" || echo "NO GPU VISIBLE"

step "toolchain"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq 2>&1 | tail -1
apt-get install -y -qq git python3.12-venv python3-dev 2>&1 | tail -2
export PIP_CACHE_DIR=/tmp/pipcache
rm -rf /tmp/uhd; mkdir -p /tmp/uhd "$OUT"
python3 -m venv /tmp/uhd/venv
PY=/tmp/uhd/venv/bin/python
# numpy because flydsl.compiler imports it at module scope; the CPU torch wheel because the
# builder only allocates host tensors whose shapes trigger the COMPILE_ONLY build.
"$PY" -m pip install --quiet numpy "flydsl==${UHD_FLYDSL_VERSION:-0.3.2}" 2>&1 | tail -2
"$PY" -m pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch 2>&1 | tail -2 \
    || "$PY" -m pip install --quiet torch 2>&1 | tail -2
"$PY" -c "import flydsl, torch, numpy; print('flydsl', getattr(flydsl,'__version__','?'), '| torch', torch.__version__)"

step "source"
# --depth 200, not 1: a bundle carries commits relative to a base, and a one-commit
# clone has no ancestor for it to apply against ("Repository lacks these prerequisite
# commits", run 67933122, on a node whose mirror was current rather than stale).
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 200 --single-branch --branch "$BRANCH" \
    https://github.com/ROCm/rocm-libraries.git "$SRC" > /tmp/uhd/clone.log 2>&1 || exit 1
if [ -n "${UHD_BUNDLE:-}" ]; then
    git -C "$SRC" fetch "$UHD_BUNDLE" "${UHD_COMMIT:-HEAD}" >> /tmp/uhd/clone.log 2>&1 \
        && git -C "$SRC" checkout --detach FETCH_HEAD >> /tmp/uhd/clone.log 2>&1 \
        || { echo "FATAL: bundle did not apply"; tail -6 /tmp/uhd/clone.log; exit 1; }
fi
git -C "$SRC" log --oneline -1
git clone -q --depth 1 "${UHD_FLYDSL_REPO:-https://github.com/ROCm/FlyDSL.git}" /tmp/uhd/flydsl \
    > /tmp/uhd/flydsl-clone.log 2>&1 || { echo "FATAL: FlyDSL clone failed"; exit 1; }
echo "flydsl kernels: $(git -C /tmp/uhd/flydsl log --oneline -1)"
export PYTHONPATH=/tmp/uhd/flydsl

step "build the variant catalog"
BUILD_START=$(date +%s)
"$PY" "$SRC/flydsl_poc_scratch/flydsl_build/build_flydsl_variants.py" \
    --out "$OUT" \
    --waves ${UHD_WAVES:-1 2 4} \
    --stagger ${UHD_STAGGER:-1 0} \
    --lazy ${UHD_LAZY:-1 0} \
    --setprio ${UHD_SETPRIO:-1} \
    > "$OUT/build.log" 2>&1
echo "variant build rc=$? in $(( $(date +%s) - BUILD_START ))s"
grep -E "functional tuples|wrote|FAILED" "$OUT/build.log" | tail -12
echo "hsacos: $(ls "$OUT"/*.hsaco 2>/dev/null | wc -l)"

step "ABI gate"
# A knob that changes the kernarg layout or the entry point is not a knob, it is a different
# kernel -- and dispatch would launch it with garbage rather than fail. Every variant must
# report the 608-byte kernarg and the symbol FlydslAttentionNative.cpp:32-45 decodes.
BAD=0
for hsaco in "$OUT"/*.hsaco; do
    read -r size symbol <<< "$(/opt/rocm/llvm/bin/llvm-readelf --notes "$hsaco" 2>/dev/null \
        | awk '/kernarg_segment_size/ {s=$2} /\.symbol:/ {n=$2} END {print s, n}')"
    if [ "$size" != "608" ] || [ "$symbol" != "flash_attn_dualwave_swp_gfx950_kernel_0.kd" ]; then
        echo "  MISMATCH $(basename "$hsaco"): kernarg=$size symbol=$symbol"
        BAD=$((BAD + 1))
    fi
done
echo "ABI gate: $(ls "$OUT"/*.hsaco 2>/dev/null | wc -l) checked, $BAD mismatched"
[ "$BAD" -eq 0 ] || { echo "FATAL: a variant changed the ABI; refusing to stage it"; exit 1; }

step "KEEP"
mkdir -p "$KEEP"
cp "$OUT"/*.hsaco "$KEEP/" 2>/dev/null
cp "$OUT"/flydsl_attention.kdp.json "$OUT"/flydsl_attention.kmd.json "$OUT"/build.log "$KEEP/" 2>/dev/null
echo "kept $(ls "$KEEP"/*.hsaco 2>/dev/null | wc -l) kernels -> $KEEP"
du -sh "$KEEP"

step "done"
date -Is; echo "node $(hostname)"
