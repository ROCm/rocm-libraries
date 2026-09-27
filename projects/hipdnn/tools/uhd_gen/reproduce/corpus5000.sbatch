#!/bin/bash
#SBATCH --job-name=uhd-corpus5000
#SBATCH --partition=defq
#SBATCH --constraint="GFX942"
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log
#
# Build one 5000-graph corpus PER ENGINE, each generated FOR that engine: every candidate
# problem is offered to it, so a corpus holds what the engine serves and nothing it declines.
#
# Per engine, not one shared corpus, and the reason is measured: AITER's first gfx950 L1
# trained on the 68 graphs it could serve out of a 1000-graph corpus drawn from rocKE's
# geometries, and came out systematically under-predicting by -285 TFLOPS. Rebuilt against
# a corpus of what AITER actually serves, the same pipeline produced -80 TFLOPS bias and took
# engine selection from 78.9% to 94.2% correct.
#
# A corpus an engine mostly declines is not a smaller training set; it is a biased one --
# the declines are not random, they are one end of the shape distribution. That is why the
# engine is asked here rather than its facets guessed with --keep, which is how these corpora
# used to be narrowed.
#
# Runs for ONE arch, the node's: UHD_ARCH picks the rows (gfx942 by default; submit with
# --constraint=GFX950 and UHD_ARCH=gfx950 for the other). The gfx950 rocKE pack lives on
# users/jascampb/uhd-integration-test-branch, the default branch here.
#
# UHD_BUILD=<dir> with UHD_SRC=<dir> reuses an existing build instead of cloning and building,
# which is how a change is tested before it is pushed.
set -uo pipefail

SRC=${UHD_SRC:-/tmp/uhd/src}
BUILD=${UHD_BUILD:-/tmp/uhd/build}
OUT=${UHD_OUT:-/exchange/corpus-5000}
# sbatch's --output lands in the job's OWN view of the submit directory, which inside a
# --container-remap-root container is not the host home that /exchange maps to: the file
# is written and then thrown away with the container. Everything is teed into /exchange
# from the first line instead, because a job whose failure is invisible costs a whole
# run to diagnose (this one: 18 minutes to learn nothing).
mkdir -p "$OUT"
exec > >(tee "$OUT/job-${SLURM_JOB_ID:-manual}.log") 2>&1
BRANCH=${UHD_BRANCH:-users/jascampb/uhd-integration-test-branch}
ARCH=${UHD_ARCH:-gfx942}
COUNT=${UHD_COUNT:-5000}
SEED=${UHD_SEED:-0}
step() { echo; echo "===== $* ====="; date -Is; }

step "node"
hostname
rocminfo 2>/dev/null | grep -m1 "Name:.*gfx" || echo "NO GPU VISIBLE"

if [ -z "${UHD_BUILD:-}" ]; then
    step "toolchain"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq 2>&1 | tail -1
    apt-get install -y -qq git cmake ninja-build python3.12-venv python3-dev 2>&1 | tail -1
    # pytest is a configure-time dependency of the provider's packaging tests.
    python3 -m pip install --quiet --break-system-packages pytest 2>&1 | tail -1

    step "source: $BRANCH"
    rm -rf /tmp/uhd; mkdir -p /tmp/uhd "$OUT"
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 200 --single-branch --branch "$BRANCH" \
        https://github.com/ROCm/rocm-libraries.git "$SRC" > /tmp/uhd/clone.log 2>&1 \
        || { echo "FATAL: clone failed"; tail -5 /tmp/uhd/clone.log; exit 1; }
    for ref in ${UHD_PREREQ_REFS:-}; do
        git -C "$SRC" fetch --unshallow origin "$ref" >> /tmp/uhd/clone.log 2>&1 \
            || git -C "$SRC" fetch origin "$ref" >> /tmp/uhd/clone.log 2>&1 \
            || { echo "FATAL: prerequisite ref $ref did not fetch"; tail -5 /tmp/uhd/clone.log; exit 1; }
    done
    if [ -n "${UHD_BUNDLE:-}" ]; then
        git -C "$SRC" fetch "$UHD_BUNDLE" "${UHD_COMMIT:-HEAD}" >> /tmp/uhd/clone.log 2>&1 \
            && git -C "$SRC" checkout --detach FETCH_HEAD >> /tmp/uhd/clone.log 2>&1 \
            || { echo "FATAL: bundle did not apply"; tail -8 /tmp/uhd/clone.log; exit 1; }
    fi
    git -C "$SRC" log --oneline -1

    # The providers, not just the tool: an engine is asked, so it has to be built and its
    # descriptors staged. ALL rather than a target list, because descriptor staging rides on
    # `add_custom_target(... ALL)` rules that a --target build skips (run 67928117 registered
    # 2 engines instead of 10 that way).
    step "build (hipdnn + hip-kernel-provider, $ARCH)"
    cmake -S "$SRC" -B "$BUILD" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="${ROCM_PATH:-/opt/rocm}" -DROCM_PATH="${ROCM_PATH:-/opt/rocm}" \
        -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
        -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
        -DROCM_LIBS_ENABLE_COMPONENTS="hipdnn;hipdnn-integration-tests;hip-kernel-provider" \
        -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON -DHIPDNN_ENABLE_SDPA=ON \
        -DHIPKERNELPROVIDER_ENABLE_ROCKE=ON -DHIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE=ON \
        -DHIPKERNELPROVIDER_KPACK_ALLOW_FETCH=ON \
        -DHIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT="$SRC/dnn-providers/hip-kernel-provider/descriptor-packaging/examples/descriptors" \
        -DGPU_TARGETS="$ARCH" -DCMAKE_HIP_ARCHITECTURES="$ARCH" \
        -DENABLE_CLANG_FORMAT=OFF -DENABLE_CLANG_TIDY=OFF > /tmp/uhd/configure.log 2>&1
    echo "configure rc=$?"
    cmake --build "$BUILD" -j "$(nproc)" > /tmp/uhd/build.log 2>&1
    BLD=$?; echo "build rc=$BLD"
    if [ "$BLD" -ne 0 ]; then grep -n -B2 -A20 "^FAILED:" /tmp/uhd/build.log | head -50; exit 1; fi
fi

GEN=$(find "$BUILD" -name hipdnn_corpus_gen -type f -perm -u+x | head -1)
PLUGIN_DIR=$(dirname "$(find "$BUILD" -name "libhip_kernel_provider*.so" -path "*plugins*" | head -1)")
[ -n "$GEN" ] || { echo "FATAL: hipdnn_corpus_gen was not built"; exit 1; }
[ -d "$PLUGIN_DIR" ] || { echo "FATAL: no hip-kernel-provider plugin under $BUILD"; exit 1; }
echo "tool:    $GEN"
echo "plugins: $PLUGIN_DIR"

TOOLS="$SRC/projects/hipdnn/tools"
PACKS="$SRC/dnn-providers/hip-kernel-provider/descriptor-packaging/examples/descriptors/rocKE"
OPS="$TOOLS/corpus_gen/operations"

# name|engine|extra hipdnn_corpus_gen arguments, for this arch only.
#
# rocKE's pack is a source, not a filter: it proposes the geometries its kernels are compiled
# for and the engine admits them. Its kernels match exact geometries, so rocKE's corpus is
# capped at its pack's shapes (664 on gfx942) however large COUNT is. That is recorded in
# corpus_gen/operations/engines.json, so the tool takes the pack, runs no search, and exits 0.
# AITER has no pack; its corpus is found by the search alone.
#
# flyDSL (hipkernel:FlydslAttention, gfx950) is not a row: its kernels are not committed and
# its pack lives outside arch_content, so it is only registered from the composed tree that
# generate.sbatch builds under UHD_COMPOSE_FLYDSL=1.
case "$ARCH" in
    gfx942) CORPORA="rocke-gfx942|hipkernel:Gfx942AttentionDense|--kdp-root PACKS/gfx942_attention_dense
aiter-gfx942|ASM_SDPA_ENGINE|" ;;
    gfx950) CORPORA="rocke-gfx950|hipkernel:Gfx950AttentionDense|--kdp-root PACKS/gfx950_attention_dense
aiter-gfx950|ASM_SDPA_ENGINE|" ;;
    *) echo "FATAL: no corpus rows for UHD_ARCH=$ARCH"; exit 1 ;;
esac

rc_all=0
while IFS='|' read -r name engine args; do
    [ -n "$name" ] || continue
    step "corpus $name ($engine)"
    args=${args//PACKS/$PACKS}
    dest="$OUT/$name"
    rm -rf "$dest"
    # shellcheck disable=SC2086
    "$GEN" --operations "$OPS" --operation sdpa_fwd --engine-name "$engine" \
        --plugin-dir "$PLUGIN_DIR" --output "$dest" --count "$COUNT" --seed "$SEED" $args \
        > "$OUT/$name.log" 2>&1
    rc=$?
    echo "rc=$rc"
    grep -E "engine queries|^Generated|^SHORT|saturated|found no served|not registered" "$OUT/$name.log"
    # 0 is a full corpus or one the engine physically cannot extend; anything else is a
    # failure the log explains (1: bad arguments or an unregistered engine, 2: empty,
    # 3: short without the engine being shown to serve no more).
    if [ $rc -ne 0 ]; then rc_all=1; continue; fi
    python3 - "$dest/manifest.json" <<'PY'
import json, sys, collections
manifest = json.load(open(sys.argv[1]))
graphs = manifest["graphs"] if isinstance(manifest, dict) else manifest
print(f"  graphs: {len(graphs)}  engine_verified: {manifest.get('reports', {}).get('engine_verified')}")
for facet in ("q.dtype", "q.head_dim", "q.is_causal", "q.alignment"):
    counts = collections.Counter(str(g.get(facet)) for g in graphs)
    print(f"  {facet}: {dict(sorted(counts.items()))}")
# `phase` is the first field of the regime label, which is built from declared axes.
phases = collections.Counter(str(g.get("regime", "")).split("_")[0] for g in graphs)
print(f"  phase: {dict(sorted(phases.items()))}")
print(f"  sources: {manifest.get('allocation')}")
PY
done <<< "$CORPORA"

step "staged"
du -sh "$OUT"/*/ 2>/dev/null
exit $rc_all
