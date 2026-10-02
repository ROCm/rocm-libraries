#!/bin/bash
#SBATCH --job-name=uhd-gen
#SBATCH --partition=defq
#SBATCH --constraint="GFX942"
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.log
#
# Generate layer-2 (sort_kernel_catalog) and layer-1 (predict_engine) heuristics for one
# engine against a STAGED corpus (`hipdnn_corpus_gen`'s output).
#
# L2 is trained and installed FIRST: it picks the kernel an immediate run executes, and L1's
# labels measure that run. UHD_ROLES=l1 skips L2 for an opaque engine (AITER, MIOpen).
#
# MIOpen convolution: UHD_PROVIDERS=miopen-provider UHD_ENGINE=MIOPEN_ENGINE UHD_ROLES=l1
# UHD_METRICS=tflops+time UHD_GRAPHS=<conv corpus root>.
#
# UHD_SHARDS/UHD_SHARD take an interleaved slice (every Nth graph); merge shards afterwards
# with `uhd_gen.dataset`. #SBATCH lines are defaults; pass site settings at submission
# (reproduce/README.md).
set -uo pipefail

BRANCH=${UHD_BRANCH:?set UHD_BRANCH to the rocm-libraries branch to build}
SRC=/tmp/uhd/src
# Descriptor root packed as `product`, relative to the checkout; unset keeps the provider's
# shipped root.
PRODUCTION_ROOT_ARGS=()
[ -n "${UHD_PRODUCTION_ROOT:-}" ] && \
    PRODUCTION_ROOT_ARGS=(-DHIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT="$SRC/$UHD_PRODUCTION_ROOT")
BUILD=/tmp/uhd/build
OUT=/tmp/uhd/out
ENGINE=${UHD_ENGINE:?set UHD_ENGINE to a registered engine name, e.g. hipkernel:Gfx950AttentionDense}
GRAPHS=${UHD_GRAPHS:?set UHD_GRAPHS to a staged corpus directory}
# `+`-separated, not comma: sbatch's --export splits its value on commas.
ROLES=${UHD_ROLES:-l2+l1}
# Ranking metrics, one UHD per metric per role (RFC 0019 §4.4). Unset keeps generate's
# default and the `<role>/model/` layout. UHD_L2_METRICS / UHD_L1_METRICS override per role
# (catalog pages publish no graph.flops, so L2 needs `time`).
L2_METRICS=${UHD_L2_METRICS:-${UHD_METRICS:-}}
L1_METRICS=${UHD_L1_METRICS:-${UHD_METRICS:-}}
L2_METRIC_ARGS=()
[ -n "$L2_METRICS" ] && L2_METRIC_ARGS=(--metric ${L2_METRICS//+/ })
L1_METRIC_ARGS=()
[ -n "$L1_METRICS" ] && L1_METRIC_ARGS=(--metric ${L1_METRICS//+/ })
SHARDS=${UHD_SHARDS:-1}
SHARD=${UHD_SHARD:-0}
ARCH=${UHD_ARCH:-gfx942}
KEEP=${UHD_KEEP:-/exchange/uhd-gen-${SLURM_JOB_ID:-manual}}
# Providers to build. `miopen-provider` adds MIOPEN_ENGINE(_DETERMINISTIC);
# `hip-kernel-provider` is rocKE/AITER.
PROVIDERS=${UHD_PROVIDERS:-hip-kernel-provider}
# The L1 role-map arch. MIOpen declares its ids under `default`, so its models go there.
case "$ENGINE" in
    MIOPEN_ENGINE|MIOPEN_ENGINE_DETERMINISTIC) L1_ARCH=${UHD_L1_ARCH:-default} ;;
    *) L1_ARCH=${UHD_L1_ARCH:-$ARCH} ;;
esac
# Tee into $KEEP before anything can fail: under --container-remap-root, sbatch's --output
# file is discarded with the container.
mkdir -p "$KEEP"
exec > >(tee "$KEEP/job-${SLURM_JOB_ID:-manual}.log") 2>&1
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
# pytest is a configure-time dependency: hkp_register_tests refuses to configure without it.
python3 -m pip install --quiet --break-system-packages "zstandard>=0.20.0" msgpack \
    lightgbm pandas "scikit-learn>=1.3.0" pyarrow flatbuffers numpy pytest 2>&1 | tail -1
for c in git cmake ninja python3; do
    printf '%-8s ' "$c"; command -v "$c" >/dev/null && "$c" --version 2>&1 | head -1 || echo MISSING
done

step "source: $BRANCH"
rm -rf /tmp/uhd; mkdir -p /tmp/uhd "$OUT"
# --depth 200, not 1: a bundle needs ancestor commits to apply against.
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 200 --single-branch --branch "$BRANCH" \
    https://github.com/ROCm/rocm-libraries.git "$SRC" > /tmp/uhd/clone.log 2>&1 \
    || { echo "FATAL: clone failed"; tail -5 /tmp/uhd/clone.log; exit 1; }
# UHD_PREREQ_REFS: other branches (space separated) holding a bundle's prerequisite commits;
# fetched unshallowed, since a prerequisite can sit deeper than the tip.
for ref in ${UHD_PREREQ_REFS:-}; do
    git -C "$SRC" fetch --unshallow origin "$ref" >> /tmp/uhd/clone.log 2>&1 \
        || git -C "$SRC" fetch origin "$ref" >> /tmp/uhd/clone.log 2>&1 \
        || { echo "FATAL: prerequisite ref $ref did not fetch"; tail -5 /tmp/uhd/clone.log; exit 1; }
done
if [ -n "${UHD_BUNDLE:-}" ]; then
    # Mirrors can lag; a bundle pins the exact commits. Its tip is HEAD, not a branch ref.
    git -C "$SRC" fetch "$UHD_BUNDLE" "${UHD_COMMIT:-HEAD}" >> /tmp/uhd/clone.log 2>&1 \
        && git -C "$SRC" checkout --detach FETCH_HEAD >> /tmp/uhd/clone.log 2>&1 \
        || { echo "FATAL: bundle $UHD_BUNDLE did not apply"; tail -8 /tmp/uhd/clone.log; exit 1; }
fi
git -C "$SRC" log --oneline -1
# Pass the declared ids (declared_ids.sh) so a drifted engine declaration fails the run.
# UHD_IDS=tflops=<uuid>+time=<uuid> overrides the table.
source "$SRC/projects/hipdnn/tools/uhd_gen/reproduce/declared_ids.sh"
mapfile -t ID_ARGS < <(uhd_id_args "${UHD_IDS:-$(declared_uhd_ids "$ENGINE" "$ARCH")}" "${L1_METRICS:-tflops}")
echo "declared ids: ${ID_ARGS[*]:-none (a UED engine, or ids reported by the engine description)}"

step "configure: $PROVIDERS"
HKP_ARGS=()
if [[ "+$PROVIDERS+" == *"+hip-kernel-provider+"* ]]; then
    HKP_ARGS=(-DHIPKERNELPROVIDER_ENABLE_ROCKE=ON
              -DHIPKERNELPROVIDER_KPACK_ALLOW_FETCH=ON
              "${PRODUCTION_ROOT_ARGS[@]}")
fi
cmake -S "$SRC" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="${ROCM_PATH:-/opt/rocm}" -DROCM_PATH="${ROCM_PATH:-/opt/rocm}" \
    -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
    -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DROCM_LIBS_ENABLE_COMPONENTS="hipdnn;hipdnn-integration-tests;${PROVIDERS//+/;}" \
    -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON -DHIPDNN_ENABLE_SDPA=ON \
    "${HKP_ARGS[@]}" \
    -DGPU_TARGETS="$ARCH" -DCMAKE_HIP_ARCHITECTURES="$ARCH" \
    -DENABLE_CLANG_FORMAT=OFF -DENABLE_CLANG_TIDY=OFF > /tmp/uhd/configure.log 2>&1
echo "configure rc=$?"
grep -B2 -A10 "CMake Error" /tmp/uhd/configure.log | head -30 || tail -3 /tmp/uhd/configure.log

step "build"
# Build ALL: descriptor staging rides on `add_custom_target(... ALL)` rules that a
# `--target <list>` build skips.
cmake --build "$BUILD" -j "$(nproc)" > /tmp/uhd/build.log 2>&1
BLD=$?; echo "build rc=$BLD"
if [ "$BLD" -ne 0 ]; then grep -n -B2 -A20 "^FAILED:" /tmp/uhd/build.log | head -50; exit 1; fi

step "runtime layout"
BENCH=$(find "$BUILD" -name hipdnn_bench -type f -perm -u+x | head -1)
LIST_ENGINES=$(find "$BUILD" -name hipdnn_list_engines -type f -perm -u+x | head -1)
FEATURE_EVAL=$(find "$BUILD" -name hipdnn_uhd_features -type f -perm -u+x | head -1)
# Every provider built installs its plugin into the one engine-plugin directory.
PLUGIN_DIR=$(dirname "$(find "$BUILD" \( -name "libhip_kernel_provider*.so" -o -name "libmiopen_plugin*.so" \) \
    -path "*plugins*" | head -1)")
# HIPDNN_DESCRIPTOR_DIR replaces the search roots and descriptors reach the per-arch kpack by
# climbing, so the generator needs the root that owns kpack. Never test_arch_content, which
# carries no provider engines.
TREE_ROOT=$(find "$BUILD" -type d -name kpack -path "*/arch_content/*/$ARCH/kpack" \
    -not -path "*/test_arch_content/*" | head -1)
TREE_ROOT=${TREE_ROOT%/kpack}
if [ -z "$TREE_ROOT" ] && [ -n "${ID_ARGS[*]:-}" ]; then
    # Declared-id engine with no pack for this arch (ASM_SDPA_ENGINE on gfx942): stage its
    # models loose into the provider's production tree.
    TREE_ROOT=$(find "$BUILD" -type d -path "*/engines/arch_content/hip-kernel-provider" \
        -not -path "*/test_arch_content/*" | head -1)
fi
if [ -z "$TREE_ROOT" ] && [[ "+$PROVIDERS+" != *"+hip-kernel-provider+"* ]]; then
    # MIOpen alone ships no descriptors: its models get a root of their own.
    TREE_ROOT="$OUT/descriptors"
    mkdir -p "$TREE_ROOT"
fi
echo "bench:        ${BENCH:-MISSING}"
echo "features:     ${FEATURE_EVAL:-MISSING}"
echo "plugin dir:   ${PLUGIN_DIR:-MISSING}"
echo "tree root:    ${TREE_ROOT:-MISSING}"
for path in "$BENCH" "$LIST_ENGINES" "$FEATURE_EVAL" "$PLUGIN_DIR" "$TREE_ROOT"; do
    [ -e "$path" ] || { echo "FATAL: a required runtime path is missing"; exit 1; }
done

# AITER's compiled-in fallback is the install path, empty in a build tree. Pick the
# asm_kernels directory that holds this arch; the build has more than one.
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
import json, pathlib, shutil, sys
source, target, shards, shard = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

# Manifest row order (already regime-spread) first; filename sort only as a fallback.
manifest = source / "manifest.json"
graphs = []
if manifest.is_file():
    document = json.load(open(manifest))
    rows = document["graphs"] if isinstance(document, dict) else document
    for row in rows:
        named = pathlib.Path(row.get("file", ""))
        path = named if named.is_absolute() else source / named
        if path.is_file():
            graphs.append(path)
if not graphs:
    for pattern in ("*.fb", "*.json"):
        found = sorted((source / "graphs").glob(pattern)) or sorted(source.glob(pattern))
        # manifest.json sits beside the graphs and is not one of them.
        graphs = [p for p in found if not p.name.startswith("manifest")]
        if graphs:
            break

# Interleaved, not contiguous, so every shard gets every regime.
mine = graphs[shard::shards]
for path in mine:
    shutil.copy2(path, target / path.name)
print(f"  corpus {len(graphs)} graphs -> shard {shard}/{shards} takes {len(mine)}")
PYEOF
COUNT=$(ls "$PROBLEMS" | wc -l)
[ "$COUNT" -gt 0 ] || { echo "FATAL: empty shard"; exit 1; }

step "what this engine admits"
# L2 times every candidate of every admitted graph, so the admitted count sets the wall clock.
ADMITTED=0
CENSUS_START=$(date +%s)
for problem in "$PROBLEMS"/*.json "$PROBLEMS"/*.fb; do
    [ -e "$problem" ] || continue
    if "$BENCH" --graph "$problem" --plugin-dir "$PLUGIN_DIR" --engine-id "$ENGINE_ID" \
        --describe-engine-prediction > /dev/null 2>&1; then
        ADMITTED=$((ADMITTED + 1))
    else
        mv "$problem" "$OUT/declined-$(basename "$problem")" 2>/dev/null
    fi
done
mkdir -p "$OUT/declined"; mv "$OUT"/declined-* "$OUT/declined/" 2>/dev/null
echo "admitted $ADMITTED of $COUNT in $(( $(date +%s) - CENSUS_START ))s"
[ "$ADMITTED" -gt 0 ] || { echo "FATAL: the engine serves none of this shard"; exit 1; }

cd "$SRC/projects/hipdnn/tools" || exit 1
# Heartbeat: L2 is hours of silence otherwise.
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
        --role sort_kernel_catalog "${L2_METRIC_ARGS[@]}" --arch "$ARCH" \
        --name "$ENGINE catalog ranker ($ARCH, $ADMITTED graphs)" \
        --output-dir "$OUT/l2" > "$OUT/l2.stdout" 2> "$OUT/l2.stderr"
    echo "L2 generate rc=$? in $(( $(date +%s) - L2_START ))s"
    tail -20 "$OUT/l2.stdout"; echo "--- stderr ---"; tail -10 "$OUT/l2.stderr"
fi

if [[ "+$ROLES+" == *"+l1+"* ]]; then
    step "LAYER 1 - predict_engine over $ADMITTED graphs"
    # After L2 is installed, so these labels describe the selector that will ship.
    L1_START=$(date +%s)
    python3 -m uhd_gen generate \
        --graphs "$PROBLEMS" \
        --descriptor-tree "$TREE_ROOT" \
        --engine "$ENGINE" --engine-id "$ENGINE_ID" \
        --bench "$BENCH" --plugin-dir "$PLUGIN_DIR" \
        --feature-evaluator "$FEATURE_EVAL" \
        --role predict_engine "${L1_METRIC_ARGS[@]}" "${ID_ARGS[@]}" --arch "$L1_ARCH" \
        --name "$ENGINE estimate ($ARCH, $ADMITTED graphs)" \
        --output-dir "$OUT/l1" > "$OUT/l1.stdout" 2> "$OUT/l1.stderr"
    echo "L1 generate rc=$? in $(( $(date +%s) - L1_START ))s"
    tail -20 "$OUT/l1.stdout"; echo "--- stderr ---"; tail -10 "$OUT/l1.stderr"
fi

step "VERIFY - the runtime admits what was installed"
# Offline evaluation can score a model the runtime still refuses; ask the built loader and
# engine. The validator covers descriptor-backed engines; declared-id engines are checked by
# their prediction.
VALIDATE=$(find "$BUILD" -name hipdnn_validate_descriptors -type f -perm -u+x | head -1)
{
    if [ -n "$VALIDATE" ] && [ -z "${ID_ARGS[*]:-}" ] \
       && [ -n "$(find "$TREE_ROOT" -name '*.ued.json' -print -quit)" ]; then
        "$VALIDATE" "$TREE_ROOT" --expect-engine "$ENGINE"
        echo "validate_descriptors rc=$?"
    fi
    PROBE=$(ls "$PROBLEMS"/*.json "$PROBLEMS"/*.fb 2>/dev/null | head -1)
    for metric in ${L1_METRICS//+/ }; do
        [ -n "$PROBE" ] || break
        HIPDNN_LOG_LEVEL=info "$BENCH" --graph "$PROBE" --plugin-dir "$PLUGIN_DIR" \
            --engine-id "$ENGINE_ID" --predict-engine --ranking-metric "$metric" 2>&1
        echo "predict-engine $metric rc=$?"
    done
} > "$OUT/verify.log" 2>&1
grep -E "rc=|\"status\"|\"reason\"|VIOLATION|is not used|does not expose|refus|disabled|not deployed|UNAVAILABLE" "$OUT/verify.log" \
    | cut -c1-300 | head -40

step "KEEP"
mkdir -p "$KEEP"
for dir in l1 l2; do
    [ -d "$OUT/$dir" ] && cp -r "$OUT/$dir" "$KEEP/" 2>/dev/null
    cp "$OUT/$dir.stdout" "$OUT/$dir.stderr" "$KEEP/" 2>/dev/null
done
# Keep a refused generate's preserved corpus: the measurements are the expensive part.
for preserved in "$OUT"/.uhd-generate-*; do
    [ -d "$preserved" ] && cp -r "$preserved" "$KEEP/preserved-$(basename "$preserved")" 2>/dev/null
done
cp "$OUT/engines.log" "$OUT/verify.log" "$KEEP/" 2>/dev/null
ls "$OUT/declined" 2>/dev/null | sed -E 's/\.(json|fb)$//' > "$KEEP/declined.txt"
echo "  admitted=$ADMITTED declined=$(wc -l < "$KEEP/declined.txt") -> $KEEP"
du -sh "$KEEP" 2>/dev/null

step "done"
date -Is; echo "node $(hostname)"
