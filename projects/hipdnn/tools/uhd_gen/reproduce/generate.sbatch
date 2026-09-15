#!/bin/bash
#SBATCH --job-name=uhd-gen
#SBATCH --partition=defq
#SBATCH --constraint="GFX942"
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.log
#
# Generate layer-2 (sort_kernel_catalog) and layer-1 (predict_engine_tflops) heuristics
# for one engine against a STAGED corpus -- `corpus_build`'s output, not a corpus the
# generator proposes for itself.
#
# Ordering is load-bearing (uhd_gen README, "Engine-level immediate predictions"): L2 is
# trained and installed FIRST, because it decides which kernel an immediate run executes,
# and L1's labels are measurements of exactly that run. UHD_ROLES=l1 skips L2 for an
# opaque engine, which has no catalog to rank.
#
# Sharding: UHD_SHARDS/UHD_SHARD take an interleaved slice of the corpus (every Nth graph,
# offset by the shard index), so each shard keeps the whole shape distribution instead of
# one contiguous alphabetical block of it. Shards collect and train independently; their
# rows are merged by `results_import` afterwards when a single model is wanted.
set -uo pipefail

BRANCH=${UHD_BRANCH:-users/jscampb/uhd-integration-test-branch}
SRC=/tmp/uhd/src
BUILD=/tmp/uhd/build
OUT=/tmp/uhd/out
ENGINE=${UHD_ENGINE:-hipkernel:Gfx942AttentionDense}
GRAPHS=${UHD_GRAPHS:?set UHD_GRAPHS to a staged corpus directory}
# `+`-separated, not comma: sbatch's own --export parser splits its value on commas, so
# UHD_ROLES=l2,l1 arrives as UHD_ROLES=l2 plus a stray empty l1 -- which is how run 67929135
# collected and trained L2 over 909 graphs and then silently skipped L1 entirely.
ROLES=${UHD_ROLES:-l2+l1}
SHARDS=${UHD_SHARDS:-1}
SHARD=${UHD_SHARD:-0}
ARCH=${UHD_ARCH:-gfx942}
KEEP=${UHD_KEEP:-/exchange/uhd-gen-${SLURM_JOB_ID:-manual}}
step() { echo; echo "===== $* ====="; date -Is; }

step "node"
hostname; nproc
rocminfo 2>/dev/null | grep -m1 "Name:.*gfx" || echo "NO GPU VISIBLE"
df -h /tmp | tail -1

step "toolchain"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq 2>&1 | tail -1
apt-get install -y -qq git cmake ninja-build python3.12-venv python3-dev 2>&1 | tail -2
export PIP_CACHE_DIR=/tmp/pipcache
python3 -m pip install --quiet --break-system-packages "zstandard>=0.20.0" msgpack \
    lightgbm pandas "scikit-learn>=1.3.0" pyarrow flatbuffers numpy 2>&1 | tail -1
for c in git cmake ninja python3; do
    printf '%-8s ' "$c"; command -v "$c" >/dev/null && "$c" --version 2>&1 | head -1 || echo MISSING
done

step "source: $BRANCH"
rm -rf /tmp/uhd; mkdir -p /tmp/uhd "$OUT"
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --single-branch --branch "$BRANCH" \
    https://github.com/ROCm/rocm-libraries.git "$SRC" > /tmp/uhd/clone.log 2>&1 \
    || { echo "FATAL: clone failed"; tail -5 /tmp/uhd/clone.log; exit 1; }
git -C "$SRC" log --oneline -1

step "configure"
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
grep -B2 -A10 "CMake Error" /tmp/uhd/configure.log | head -30 || tail -3 /tmp/uhd/configure.log

step "build"
# ALL, not a target list: the descriptors every engine is registered from are staged by
# `add_custom_target(... ALL)` rules (hip_kernel_provider_descriptors, hkp_descriptor_staging,
# hkp_packaging_*), which a `--target <list>` build skips. Run 67928117 did exactly that and
# registered 2 engines instead of 10.
cmake --build "$BUILD" -j "$(nproc)" > /tmp/uhd/build.log 2>&1
BLD=$?; echo "build rc=$BLD"
if [ "$BLD" -ne 0 ]; then grep -n -B2 -A20 "^FAILED:" /tmp/uhd/build.log | head -50; exit 1; fi

step "runtime layout"
BENCH=$(find "$BUILD" -name hipdnn_bench -type f -perm -u+x | head -1)
LIST_ENGINES=$(find "$BUILD" -name hipdnn_list_engines -type f -perm -u+x | head -1)
FEATURE_EVAL=$(find "$BUILD" -name hipdnn_uhd_features -type f -perm -u+x | head -1)
PLUGIN_DIR=$(dirname "$(find "$BUILD" -name "libhip_kernel_provider*.so" -path "*plugins*" | head -1)")
# The generator benchmarks against a COPY of --descriptor-tree, and HIPDNN_DESCRIPTOR_DIR
# REPLACES the search roots rather than adding to them. A kpack archive lives once per arch
# at the arch root and every descriptor climbs out of its own folder to reach it, so the
# tree handed to the generator must be the root that owns kpack -- copying one engine's
# folder leaves its kernels unresolvable and every pinned configuration comes back
# config_not_applicable (measured, run 67926620).
TREE_ROOT=$(find "$BUILD" -type d -name kpack -path "*arch_content*" | head -1)
TREE_ROOT=${TREE_ROOT%/kpack}
echo "bench:        ${BENCH:-MISSING}"
echo "features:     ${FEATURE_EVAL:-MISSING}"
echo "plugin dir:   ${PLUGIN_DIR:-MISSING}"
echo "tree root:    ${TREE_ROOT:-MISSING}"
for path in "$BENCH" "$LIST_ENGINES" "$FEATURE_EVAL" "$PLUGIN_DIR" "$TREE_ROOT"; do
    [ -e "$path" ] || { echo "FATAL: a required runtime path is missing"; exit 1; }
done

if [ "${UHD_COMPOSE_FLYDSL:-0}" = "1" ]; then
    step "compose the flyDSL pack into the descriptor tree"
    # flyDSL's 21 HSACOs are not committed (REPLICATE_ATTENTION.md) and its pack lives in
    # flydsl_poc_scratch, outside arch_content. HIPDNN_DESCRIPTOR_DIR replaces the search
    # roots, so the only way to have rocKE and flyDSL loaded at once is one composed tree.
    COMPOSED=/tmp/uhd/descriptors
    mkdir -p "$COMPOSED"
    cp -r "$TREE_ROOT"/. "$COMPOSED"/
    cp -r "$SRC/flydsl_poc_scratch/flydsl_descriptors_attention/." "$COMPOSED"/
    HSACO_DIR="$SRC/flydsl_poc_scratch"
    if [ -n "${UHD_HSACO_DIR:-}" ] && [ -d "${UHD_HSACO_DIR}" ]; then
        # Reuse the enablement job's binaries: the flyDSL compiler is a pip wheel whose
        # output must match the kernarg ABI the pack decodes, so the artifacts that were
        # already proven to launch are preferred over a fresh rebuild of the same source.
        cp "$UHD_HSACO_DIR"/*.hsaco "$HSACO_DIR"/ 2>/dev/null
        echo "reused $(ls "$UHD_HSACO_DIR"/*.hsaco 2>/dev/null | wc -l) staged HSACOs"
    else
        python3 -m pip install --quiet --break-system-packages flydsl==0.3.2 2>&1 | tail -1
        python3 "$SRC/flydsl_poc_scratch/flydsl_build/build_all_flydsl_attention.py" \
            > "$OUT/flydsl-build.log" 2>&1
        echo "flydsl build rc=$? hsacos=$(ls "$HSACO_DIR"/*.hsaco 2>/dev/null | wc -l)"
    fi
    export FLYDSL_ATTENTION_HSACO_DIR="$HSACO_DIR"
    export HIPDNN_DESCRIPTOR_DIR="$COMPOSED"
    TREE_ROOT="$COMPOSED"
    echo "composed tree: $TREE_ROOT ($(find "$COMPOSED" -name '*.json' | wc -l) descriptors, $(ls "$HSACO_DIR"/*.hsaco 2>/dev/null | wc -l) hsacos)"
fi
# AITER resolves its .co catalog from HIPDNN_AITER_ASM_DIR, falling back to the AITER_ASM_DIR
# compile definition -- which is the INSTALL path (/opt/rocm/lib/...), empty in a build tree.
# Run 67929136 collected 843 admitted graphs and then failed every execution with "failed to
# load kernel module from /opt/rocm/.../fwd_hd128_bf16_rtne.co". The directory wanted is the
# one holding the arch subdirectories, so it is chosen by looking for the arch inside it
# rather than by name alone: the build carries more than one directory called asm_kernels.
AITER_ASM=""
while read -r candidate; do
    [ -d "$candidate/$ARCH/fmha_v3_fwd" ] && { AITER_ASM="$candidate"; break; }
done < <(find "$BUILD" -type d -name asm_kernels 2>/dev/null)
if [ -n "$AITER_ASM" ]; then
    export HIPDNN_AITER_ASM_DIR="$AITER_ASM"
    echo "aiter asm:    $AITER_ASM ($(ls "$AITER_ASM/$ARCH/fmha_v3_fwd"/*/*.co 2>/dev/null | wc -l) kernels)"
else
    echo "aiter asm:    NONE FOUND under $BUILD -- ASM_SDPA execution will fail"
fi

step "engines this build ships"
"$LIST_ENGINES" --plugin-dir "$PLUGIN_DIR" 2>&1 | tee "$OUT/engines.log" | grep -E "^  \S+ \(0x" 
ENGINE_HEX=$(grep -F "$ENGINE (" "$OUT/engines.log" | head -1 | sed -E 's/.*\((0x[0-9a-fA-F]+)\).*/\1/')
[ -n "$ENGINE_HEX" ] || { echo "FATAL: engine $ENGINE is not registered"; exit 1; }
ENGINE_ID=$(python3 -c "v=int('$ENGINE_HEX',16); print(v-(1<<64) if v>=(1<<63) else v)")
echo "engine: $ENGINE $ENGINE_HEX -> $ENGINE_ID"

step "corpus shard $SHARD of $SHARDS"
PROBLEMS="$OUT/problems"
mkdir -p "$PROBLEMS"
python3 - "$GRAPHS" "$PROBLEMS" "$SHARDS" "$SHARD" <<'PYEOF'
import pathlib, shutil, sys
source, target, shards, shard = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
graphs = sorted((source / "graphs").glob("*.json")) or sorted(source.glob("*.json"))
# Interleaved, not contiguous: the corpus is sorted by name, which sorts by regime, so a
# contiguous block would hand one shard every decode problem and another every prefill one.
mine = graphs[shard::shards]
for path in mine:
    shutil.copy2(path, target / path.name)
print(f"  corpus {len(graphs)} graphs -> shard {shard}/{shards} takes {len(mine)}")
PYEOF
COUNT=$(ls "$PROBLEMS" | wc -l)
[ "$COUNT" -gt 0 ] || { echo "FATAL: empty shard"; exit 1; }

step "what this engine admits"
# The census is the cost model for everything below: L2 times every candidate of every
# admitted graph, so the admitted count -- not the corpus size -- sets the wall clock.
ADMITTED=0
CENSUS_START=$(date +%s)
for problem in "$PROBLEMS"/*.json; do
    if "$BENCH" --graph "$problem" --plugin-dir "$PLUGIN_DIR" --engine-id "$ENGINE_ID" \
        --describe-engine-prediction > /dev/null 2>&1; then
        ADMITTED=$((ADMITTED + 1))
    else
        mv "$problem" "$OUT/declined-$(basename "$problem")" 2>/dev/null
    fi
done
mkdir -p "$OUT/declined"; mv "$OUT"/declined-*.json "$OUT/declined/" 2>/dev/null
echo "admitted $ADMITTED of $COUNT in $(( $(date +%s) - CENSUS_START ))s"
[ "$ADMITTED" -gt 0 ] || { echo "FATAL: the engine serves none of this shard"; exit 1; }

cd "$SRC/projects/hipdnn/tools" || exit 1
# A heartbeat, because L2 over thousands of graphs is hours of silence otherwise and the
# rate is the only thing that says whether the shard count was right.
( while sleep 600; do
      echo "  [heartbeat $(date -Is)] rows=$(wc -l < "$OUT/l2/rows.csv" 2>/dev/null || echo 0) $(tail -1 "$OUT/l2.stdout" 2>/dev/null | cut -c1-120)"
  done ) & HEARTBEAT=$!
trap 'kill $HEARTBEAT 2>/dev/null' EXIT

if [[ "+$ROLES+" == *"+l2+"* ]]; then
    step "LAYER 2 - sort_kernel_catalog over $ADMITTED graphs"
    L2_START=$(date +%s)
    python3 -m uhd_gen generate \
        --graphs "$PROBLEMS" \
        --descriptor-tree "$TREE_ROOT" \
        --engine "$ENGINE" --engine-id "$ENGINE_ID" \
        --bench "$BENCH" --plugin-dir "$PLUGIN_DIR" \
        --feature-evaluator "$FEATURE_EVAL" \
        --role sort_kernel_catalog --arch "$ARCH" \
        --name "$ENGINE catalog ranker ($ARCH, $ADMITTED graphs)" \
        --output-dir "$OUT/l2" > "$OUT/l2.stdout" 2> "$OUT/l2.stderr"
    echo "L2 generate rc=$? in $(( $(date +%s) - L2_START ))s"
    tail -20 "$OUT/l2.stdout"; echo "--- stderr ---"; tail -10 "$OUT/l2.stderr"
fi

if [[ "+$ROLES+" == *"+l1+"* ]]; then
    step "LAYER 1 - predict_engine_tflops over $ADMITTED graphs"
    # After L2 is installed: an immediate run executes whatever the installed catalog
    # ranker picked, so these labels describe the selector that will ship.
    L1_START=$(date +%s)
    python3 -m uhd_gen generate \
        --graphs "$PROBLEMS" \
        --descriptor-tree "$TREE_ROOT" \
        --engine "$ENGINE" --engine-id "$ENGINE_ID" \
        --bench "$BENCH" --plugin-dir "$PLUGIN_DIR" \
        --feature-evaluator "$FEATURE_EVAL" \
        --role predict_engine_tflops --arch "$ARCH" \
        --name "$ENGINE throughput ($ARCH, $ADMITTED graphs)" \
        --output-dir "$OUT/l1" > "$OUT/l1.stdout" 2> "$OUT/l1.stderr"
    echo "L1 generate rc=$? in $(( $(date +%s) - L1_START ))s"
    tail -20 "$OUT/l1.stdout"; echo "--- stderr ---"; tail -10 "$OUT/l1.stderr"
fi

step "KEEP"
mkdir -p "$KEEP"
for dir in l1 l2; do
    [ -d "$OUT/$dir" ] && cp -r "$OUT/$dir" "$KEEP/" 2>/dev/null
    cp "$OUT/$dir.stdout" "$OUT/$dir.stderr" "$KEEP/" 2>/dev/null
done
cp "$OUT/engines.log" "$KEEP/" 2>/dev/null
ls "$OUT/declined" 2>/dev/null | sed 's/\.json$//' > "$KEEP/declined.txt"
echo "  admitted=$ADMITTED declined=$(wc -l < "$KEEP/declined.txt") -> $KEEP"
du -sh "$KEEP" 2>/dev/null

step "done"
date -Is; echo "node $(hostname)"
