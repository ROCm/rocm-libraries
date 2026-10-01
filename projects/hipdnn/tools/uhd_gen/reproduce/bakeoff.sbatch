#!/bin/bash
#SBATCH --job-name=uhd-bakeoff
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log
#
# The bake-off: install every engine's trained L1 model into ONE runtime and ask each of
# them, for every graph in the corpus, what it predicts.
#
# The generation jobs each measured their own engine in isolation, which answers "how fast
# is this engine here". Cross-engine selection needs the other half: what each engine's
# model SAYS before anything runs, on a graph every engine can see. A model is only useful
# if the number it reports can be compared with another engine's number -- RFC 0019 §11.3's
# calibrated L1 -- and the only way to check that claim is to install both and ask.
#
# Asked once per ranking metric (UHD_METRICS, `+`-separated; default tflops): an engine
# answers each metric with that metric's own model, and the winner of `time` is the lowest
# prediction, not the highest. A MIOpen convolution bake-off:
#   UHD_PROVIDERS=miopen-provider+hip-kernel-provider UHD_METRICS=tflops+time
#   UHD_MODELS="miopen-tflops=<l1>/model_tflops:MIOPEN_ENGINE;miopen-time=<l1>/model_time:MIOPEN_ENGINE;..."
#
# The #SBATCH lines are defaults; the site's partition, account and GPU constraint are
# passed at submission (reproduce/README.md, "What every job needs").
set -uo pipefail

BRANCH=${UHD_BRANCH:?set UHD_BRANCH to the rocm-libraries branch to build}
SRC=/tmp/uhd/src
# The descriptor root the provider packs as `product`. Unset keeps the provider's own
# default, the shipped root (src/engines/kernel_ingestor_engine/descriptors) that carries
# hipkernel:Gfx950AttentionDense; UHD_PRODUCTION_ROOT=<path relative to the checkout>
# packs another tree instead, e.g. the descriptor-packaging examples fixture tree.
PRODUCTION_ROOT_ARGS=()
[ -n "${UHD_PRODUCTION_ROOT:-}" ] && \
    PRODUCTION_ROOT_ARGS=(-DHIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT="$SRC/$UHD_PRODUCTION_ROOT")
BUILD=/tmp/uhd/build
OUT=/tmp/uhd/out
CORPUS=${UHD_CORPUS:?set UHD_CORPUS}
ARCH=${UHD_ARCH:-gfx950}
# LABEL=/path/to/kept/model-dir:ENGINE_NAME, semicolon-separated.
MODELS=${UHD_MODELS:?set UHD_MODELS to LABEL=modeldir:engine;... }
# `+`-separated, not comma: sbatch's --export splits its value on commas.
METRICS=${UHD_METRICS:-tflops}
# Providers to build; `miopen-provider` adds MIOPEN_ENGINE and MIOPEN_ENGINE_DETERMINISTIC.
PROVIDERS=${UHD_PROVIDERS:-hip-kernel-provider}
KEEP=${UHD_KEEP:-/exchange/uhd-bakeoff-${SLURM_JOB_ID:-manual}}
step() { echo; echo "===== $* ====="; date -Is; }

step "node"
hostname; nproc; rocminfo 2>/dev/null | grep -m1 "Name:.*gfx" || echo "NO GPU"

step "toolchain"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq 2>&1 | tail -1
apt-get install -y -qq git cmake ninja-build python3.12-venv python3-dev 2>&1 | tail -2
export PIP_CACHE_DIR=/tmp/pipcache
python3 -m pip install --quiet --break-system-packages "zstandard>=0.20.0" msgpack \
    lightgbm pandas "scikit-learn>=1.3.0" pyarrow flatbuffers numpy 2>&1 | tail -1

step "source: $BRANCH${UHD_COMMIT:+ @ $UHD_COMMIT}"
rm -rf /tmp/uhd; mkdir -p /tmp/uhd "$OUT"
# --depth 200, not 1: a bundle carries commits relative to a base, and a one-commit
# clone has no ancestor for it to apply against ("Repository lacks these prerequisite
# commits"), even on a node whose mirror is current rather than stale.
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 200 --single-branch --branch "$BRANCH" \
    https://github.com/ROCm/rocm-libraries.git "$SRC" > /tmp/uhd/clone.log 2>&1 || exit 1
if [ -n "${UHD_COMMIT:-}" ]; then
    # An opaque engine's model records the provider build it was measured on, and the
    # loader REFUSES a model whose recorded selector_revision is not the one the provider
    # reports -- by design, because L1 is the score compared across engines. A bake-off
    # built from a commit whose provider reports another revision refuses every model
    # collected before it: the engine answers and none of its answers score.
    # Pinning the collection's commit makes the runtime the build the model describes.
    #
    # Deepened rather than fetched by SHA: GitHub refuses `fetch <sha>` unless the server
    # opts in ("couldn't find remote ref"), and the commit is an ancestor of this branch,
    # so history is the reliable way to reach it.
    git -C "$SRC" fetch --depth 200 origin "$BRANCH" >> /tmp/uhd/clone.log 2>&1
    git -C "$SRC" checkout --detach "$UHD_COMMIT" >> /tmp/uhd/clone.log 2>&1 \
        || { echo "FATAL: cannot check out $UHD_COMMIT"; tail -5 /tmp/uhd/clone.log; exit 1; }
fi
if [ -n "${UHD_BUNDLE:-}" ]; then
    # The compute sites do not all resolve github.com to the same mirror: a node can clone a
    # branch tip hours behind the one `git ls-remote` reports from the login node. A bundle
    # carries the exact commits from the machine that pushed them, so verification cannot
    # silently test a different tree than the one under review.
    # `git bundle create <file> <base>..HEAD` names its tip HEAD, not a branch ref, so the
    # fetch asks for exactly that and checks out FETCH_HEAD.
    git -C "$SRC" fetch "$UHD_BUNDLE" "${UHD_COMMIT:-HEAD}" >> /tmp/uhd/clone.log 2>&1 \
        && git -C "$SRC" checkout --detach FETCH_HEAD >> /tmp/uhd/clone.log 2>&1 \
        || { echo "FATAL: bundle $UHD_BUNDLE did not apply"; tail -8 /tmp/uhd/clone.log; exit 1; }
fi
git -C "$SRC" log --oneline -1

step "configure + build: $PROVIDERS"
HKP_ARGS=()
if [[ "+$PROVIDERS+" == *"+hip-kernel-provider+"* ]]; then
    HKP_ARGS=(-DHIPKERNELPROVIDER_ENABLE_ROCKE=ON
              -DHIPKERNELPROVIDER_KPACK_ALLOW_FETCH=ON
              "${PRODUCTION_ROOT_ARGS[@]}")
fi
cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="${ROCM_PATH:-/opt/rocm}" -DROCM_PATH="${ROCM_PATH:-/opt/rocm}" \
    -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DROCM_LIBS_ENABLE_COMPONENTS="hipdnn;hipdnn-integration-tests;${PROVIDERS//+/;}" \
    -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON -DHIPDNN_ENABLE_SDPA=ON \
    "${HKP_ARGS[@]}" \
    -DGPU_TARGETS="$ARCH" -DCMAKE_HIP_ARCHITECTURES="$ARCH" \
    -DENABLE_CLANG_FORMAT=OFF -DENABLE_CLANG_TIDY=OFF > /tmp/uhd/configure.log 2>&1
echo "configure rc=$?"
cmake --build "$BUILD" -j "$(nproc)" > /tmp/uhd/build.log 2>&1
BLD=$?; echo "build rc=$BLD"
[ "$BLD" -eq 0 ] || { grep -n -B2 -A20 "^FAILED:" /tmp/uhd/build.log | head -40; exit 1; }

step "runtime layout"
BENCH=$(find "$BUILD" -name hipdnn_bench -type f -perm -u+x | head -1)
LIST_ENGINES=$(find "$BUILD" -name hipdnn_list_engines -type f -perm -u+x | head -1)
# Every provider built installs its plugin into the one engine-plugin directory.
PLUGIN_DIR=$(dirname "$(find "$BUILD" \( -name "libhip_kernel_provider*.so" -o -name "libmiopen_plugin*.so" \) \
    -path "*plugins*" | head -1)")
# The shipped (product) tree for this arch, never test_arch_content: that tree also holds
# kpack archives, sorts first, and carries none of the provider's engines, so taking it
# finds no UED for the engine.
TREE_ROOT=$(find "$BUILD" -type d -name kpack -path "*/arch_content/*/$ARCH/kpack" \
    -not -path "*/test_arch_content/*" | head -1); TREE_ROOT=${TREE_ROOT%/kpack}
# MIOpen alone ships no descriptors: its models are installed into a root of their own.
[ -n "$TREE_ROOT" ] || { TREE_ROOT="$OUT/descriptors"; mkdir -p "$TREE_ROOT"; }
AITER_ASM=""
while read -r candidate; do
    [ -d "$candidate/$ARCH" ] && { AITER_ASM="$candidate"; break; }
done < <(find "$BUILD" -type d -name asm_kernels 2>/dev/null)
[ -n "$AITER_ASM" ] && export HIPDNN_AITER_ASM_DIR="$AITER_ASM"
echo "tree root: $TREE_ROOT"; echo "aiter asm: ${AITER_ASM:-NONE}"


step "install every trained model into the one tree"
cd "$SRC/projects/hipdnn/tools" || exit 1
# An engine with no UED reads only the ids its provider declares; promote installs a model
# only under the one declared for its metric (declared_ids.sh), so name them all.
source uhd_gen/reproduce/declared_ids.sh
IFS=';' read -ra ENTRIES <<< "$MODELS"
for entry in "${ENTRIES[@]}"; do
    label=${entry%%=*}; rest=${entry#*=}
    model_dir=${rest%%:*}; engine=${rest#*:}
    echo "-- $label: $model_dir -> $engine"
    mapfile -t ID_ARGS < <(uhd_id_args "$(declared_uhd_ids "$engine" "$ARCH")" "tflops+time")
    python3 -m uhd_gen promote --model-dir "$model_dir" --descriptor-tree "$TREE_ROOT" \
        --engine "$engine" --role predict_engine --arch "$ARCH" "${ID_ARGS[@]}" 2>&1 | tail -4
done
export HIPDNN_DESCRIPTOR_DIR="$TREE_ROOT"

step "engines this runtime loads"
"$LIST_ENGINES" --plugin-dir "$PLUGIN_DIR" 2>&1 | tee "$OUT/engines.log" | grep -E "^  \S+ \(0x"

step "predictions in ${METRICS//+/ and }, every engine, every graph"
python3 - "$BENCH" "$CORPUS" "$OUT" "$PLUGIN_DIR" "$METRICS" <<'PYEOF'
import json, pathlib, re, subprocess, sys
bench, corpus, out, plugin_dir = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), sys.argv[4]
metrics = sys.argv[5].split("+")
# Which prediction wins, per metric: the registry's direction, never an assumed `max`.
sys.path.insert(0, str(pathlib.Path.cwd()))
from uhd_gen.ranking_metrics import ranking_metric
engines = []
for line in (out / "engines.log").read_text(errors="replace").splitlines():
    match = re.match(r"^  (\S+) \((0x[0-9a-fA-F]+)\)", line)
    if match:
        value = int(match.group(2), 16)
        engines.append((match.group(1), value - (1 << 64) if value >= (1 << 63) else value))
# The corpus's own graph list when it has one (hipdnn_corpus_gen's manifest.json), so a
# stray file under graphs/ is never asked about and the stem is the manifest row's `name`.
manifest = corpus / "manifest.json"
if manifest.is_file():
    document = json.loads(manifest.read_text())
    graphs = [corpus / row["file"] for row in (document["graphs"] if isinstance(document, dict) else document)]
else:
    graphs = sorted([*(corpus / "graphs").glob("*.json"), *(corpus / "graphs").glob("*.fb")])
rows = []
for metric in metrics:
    for graph in graphs:
        for name, engine_id in engines:
            proc = subprocess.run([bench, "--graph", str(graph), "--plugin-dir", plugin_dir,
                                   "--engine-id", str(engine_id), "--predict-engine",
                                   "--ranking-metric", metric],
                                  capture_output=True, text=True, timeout=600)
            row = {"graph": graph.stem, "engine": name, "requested_metric": metric, "rc": proc.returncode}
            for line in reversed(proc.stdout.splitlines()):
                if line.startswith("{"):
                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    row.update({k: doc.get(k) for k in ("status", "prediction_kind", "metric", "value", "reason")})
                    break
            rows.append(row)
(out / "predictions.json").write_text(json.dumps(rows, indent=1))

for metric in metrics:
    better = max if ranking_metric(metric).objective == "max" else min
    asked = [r for r in rows if r["requested_metric"] == metric]
    scored = [r for r in asked if r.get("value") is not None and r.get("metric") == metric]
    print(f"  [{metric}] {len(graphs)} graphs x {len(engines)} engines; {len(scored)} scored predictions")
    for name, _ in engines:
        mine = [r for r in asked if r["engine"] == name]
        ok = [r for r in scored if r["engine"] == name]
        served = [r for r in mine if r["rc"] == 0]
        if served:
            print(f"    {name:34s} answers {len(served):4d}  scored {len(ok):4d}")
    by_graph = {}
    for row in scored:
        by_graph.setdefault(row["graph"], {})[row["engine"]] = row["value"]
    contested = {g: s for g, s in by_graph.items() if len(s) > 1}
    print(f"  [{metric}] predicted winner is decidable on {len(contested)} graphs")
    wins = {}
    for graph, scores in contested.items():
        winner = better(scores, key=scores.get)
        wins[winner] = wins.get(winner, 0) + 1
    for engine, count in sorted(wins.items(), key=lambda kv: -kv[1]):
        print(f"    predicted winner {engine:34s} {count:4d}")
PYEOF

step "KEEP"
mkdir -p "$KEEP"
cp "$OUT/engines.log" "$OUT/predictions.json" "$KEEP/" 2>/dev/null
echo "-> $KEEP"

step "done"
date -Is; echo "node $(hostname)"
