#!/bin/bash
#SBATCH --job-name=uhd-950-fly
#SBATCH --partition=defq
#SBATCH --constraint="GFX950"
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.log
#
# Stand up every gfx950 SDPA engine in ONE runtime, and prove each of them answers.
#
# Three engines need three different things, which is why this job exists separately from
# the generation that follows it:
#
#   rocKE Gfx950AttentionDense  - descriptors staged by the build's own ALL targets.
#   AITER ASM_SDPA              - its .co catalog, found via HIPDNN_AITER_ASM_DIR.
#   flyDSL FlydslAttention      - a THROWAWAY POC pack whose 21 HSACOs are not committed
#                                 (REPLICATE_ATTENTION.md): they are regenerated here from
#                                 the committed builder plus the flyDSL 0.3.2 wheel, and
#                                 raw-loaded at dispatch from $FLYDSL_ATTENTION_HSACO_DIR.
#
# HIPDNN_DESCRIPTOR_DIR REPLACES the descriptor search roots rather than adding to them, so
# the flyDSL pack and the built rocKE arch_content tree are composed into ONE tree here --
# pointing it at the flyDSL pack alone (as the POC's own run_py.sh does) hides every rocKE
# engine, and a bake-off cannot be run against engines that are not loaded.
set -uo pipefail

BRANCH=${UHD_BRANCH:-users/jscampb/uhd-integration-test-branch}
SRC=/tmp/uhd/src
BUILD=/tmp/uhd/build
OUT=/tmp/uhd/out
CORPUS=${UHD_CORPUS:-/exchange/corpus-950-census}
KEEP=${UHD_KEEP:-/exchange/uhd-950-flydsl-${SLURM_JOB_ID:-manual}}
ARCH=gfx950
step() { echo; echo "===== $* ====="; date -Is; }

step "node"
hostname; nproc
rocminfo 2>/dev/null | grep -m1 "Name:.*gfx" || echo "NO GPU VISIBLE"

step "toolchain"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq 2>&1 | tail -1
apt-get install -y -qq git cmake ninja-build python3.12-venv python3-dev 2>&1 | tail -2
export PIP_CACHE_DIR=/tmp/pipcache
python3 -m pip install --quiet --break-system-packages "zstandard>=0.20.0" msgpack \
    lightgbm pandas "scikit-learn>=1.3.0" pyarrow flatbuffers numpy 2>&1 | tail -1

step "source: $BRANCH"
rm -rf /tmp/uhd; mkdir -p /tmp/uhd "$OUT"
# --depth 200, not 1: a bundle carries commits relative to a base, and a one-commit
# clone has no ancestor for it to apply against ("Repository lacks these prerequisite
# commits", run 67933122, on a node whose mirror was current rather than stale).
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 200 --single-branch --branch "$BRANCH" \
    https://github.com/ROCm/rocm-libraries.git "$SRC" > /tmp/uhd/clone.log 2>&1 \
    || { echo "FATAL: clone failed"; tail -5 /tmp/uhd/clone.log; exit 1; }
git -C "$SRC" log --oneline -1

step "flyDSL toolchain (deferred: it needs the build's rocke interpreter)"
# build_flash_attn_real.py imports `kernels.attention.flash_attn_gfx950`, which ships in
# the rocke wheel -- not in the flydsl wheel and not on the system interpreter. The build
# provisions a hermetic venv with exactly that wheel installed (HkpPackaging.cmake:699-741,
# probed with `import rocke, kernels`), so the HSACOs are built with THAT python, after the
# build, rather than with the system one. Run 67929208 used the system interpreter and all
# 21 builds died on `ModuleNotFoundError: No module named 'torch'` -- the first of two
# missing imports, the second being `kernels` itself.

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

step "build (ALL -- the descriptor staging targets are ALL-only)"
cmake --build "$BUILD" -j "$(nproc)" > /tmp/uhd/build.log 2>&1
BLD=$?; echo "build rc=$BLD"
if [ "$BLD" -ne 0 ]; then grep -n -B2 -A20 "^FAILED:" /tmp/uhd/build.log | head -50; exit 1; fi

step "regenerate the flyDSL HSACOs"
# 21 instances, one HSACO per (num_query_heads, num_kv_heads, causal, dtype); seq_len and
# batch are runtime scalars, so one binary spans every length for its head config.
#
# The kernel source is `kernels.attention.flash_attn_gfx950` from ROCm/FlyDSL -- NOT from
# this repository and not from the rocke wheel, whose `kernels` package ships the
# PRODUCTIZED descendant of this experiment (kernels/gfx950/attention_dense.py) under a
# different ABI: q/k/v/o pointers plus a runtime f32 scale, against the 608-byte 12-slot
# kernarg with the scale baked in that this pack decodes. Runs 67929208 and 67929292 failed
# here for exactly that reason ("No module named 'kernels.attention'").
HSACO_DIR="$SRC/flydsl_poc_scratch"
FLYDSL_SRC=/tmp/uhd/flydsl
git clone -q --depth 1 ${UHD_FLYDSL_COMMIT:+--no-single-branch} \
    "${UHD_FLYDSL_REPO:-https://github.com/ROCm/FlyDSL.git}" "$FLYDSL_SRC" > /tmp/uhd/flydsl-clone.log 2>&1 \
    && { [ -z "${UHD_FLYDSL_COMMIT:-}" ] || git -C "$FLYDSL_SRC" checkout --detach "$UHD_FLYDSL_COMMIT" >> /tmp/uhd/flydsl-clone.log 2>&1; } \
    || { echo "FATAL: FlyDSL clone failed"; tail -5 /tmp/uhd/flydsl-clone.log; exit 1; }
echo "flydsl kernels: $(git -C "$FLYDSL_SRC" log --oneline -1)"
ROCKE_PY=$(find "$BUILD" -path "*hkp-rocke-venv/bin/python" | head -1)
echo "rocke interpreter: ${ROCKE_PY:-MISSING}"
if [ -n "$ROCKE_PY" ]; then
    # flydsl 0.3.2 is the version REPLICATE_ATTENTION.md names as the one the pack's ABI was
    # decoded from; a different compiler may lay the kernargs out differently, which the ABI
    # check below is here to catch. torch only allocates the CPU tensors whose shapes trigger
    # the COMPILE_ONLY build, so the CPU wheel is enough.
    # The rocke venv is hermetic (`python -m venv --copies`, no system site packages), so
    # flydsl's own dependencies have to be installed into it too -- flydsl.compiler imports
    # numpy at module scope, which is what run 67931725 died on after the kernel source
    # finally resolved.
    "$ROCKE_PY" -m pip install --quiet numpy "flydsl==${UHD_FLYDSL_VERSION:-0.3.2}" 2>&1 | tail -2
    "$ROCKE_PY" -m pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch 2>&1 | tail -2 \
        || "$ROCKE_PY" -m pip install --quiet torch 2>&1 | tail -2
    # FlyDSL's `kernels` package must win over the rocke wheel's same-named one.
    export PYTHONPATH="$FLYDSL_SRC:${PYTHONPATH:-}"
    "$ROCKE_PY" -c "import flydsl, torch; from kernels.attention.flash_attn_gfx950 import build_flash_attn_dualwave_swp_module as b; print('flydsl', getattr(flydsl,'__version__','?'), '| torch', torch.__version__, '| builder', b.__module__)" 2>&1 | tail -3
    FLY_START=$(date +%s)
    "$ROCKE_PY" "$SRC/flydsl_poc_scratch/flydsl_build/build_all_flydsl_attention.py" \
        > "$OUT/flydsl-build.log" 2>&1
    echo "flydsl build rc=$? in $(( $(date +%s) - FLY_START ))s"
    grep -E "wrote|FAILED|Error" "$OUT/flydsl-build.log" | tail -8
else
    echo "no rocke venv in this build; flyDSL cannot be built here"
fi
echo "hsacos: $(ls "$HSACO_DIR"/*.hsaco 2>/dev/null | wc -l) of 21"

step "the emitted kernels carry the ABI the pack decodes"
# The pack reads a 608-byte kernarg as 12 tensor slots of stride 48 plus 8 i32 scalars at
# 576..604, and launches by symbol (FlydslAttentionNative.cpp:32-45). Both are properties of
# the BINARY, and a compiler that lays them out differently produces a kernel this engine
# launches with garbage rather than one it fails to launch. Checked before anything runs.
for hsaco in $(ls "$HSACO_DIR"/*.hsaco 2>/dev/null | head -3); do
    /opt/rocm/llvm/bin/llvm-readelf --notes "$hsaco" 2>/dev/null \
        | grep -E "\.name:|kernarg_segment_size|\.symbol:" | head -4 | sed "s|^|  $(basename "$hsaco"): |"
done

step "runtime layout"
BENCH=$(find "$BUILD" -name hipdnn_bench -type f -perm -u+x | head -1)
LIST_ENGINES=$(find "$BUILD" -name hipdnn_list_engines -type f -perm -u+x | head -1)
FEATURE_EVAL=$(find "$BUILD" -name hipdnn_uhd_features -type f -perm -u+x | head -1)
PLUGIN_DIR=$(dirname "$(find "$BUILD" -name "libhip_kernel_provider*.so" -path "*plugins*" | head -1)")
ARCH_CONTENT=$(find "$BUILD" -type d -name arch_content | head -1)
AITER_ASM=$(find "$BUILD" -type d -name asm_kernels | head -1)
echo "bench:        ${BENCH:-MISSING}"
echo "plugin dir:   ${PLUGIN_DIR:-MISSING}"
echo "arch_content: ${ARCH_CONTENT:-MISSING}"
echo "aiter asm:    ${AITER_ASM:-MISSING}"

step "compose one descriptor tree"
# Everything the loader must see in a single root, because HIPDNN_DESCRIPTOR_DIR replaces
# the search roots. The rocKE half is copied whole, kpack included: a descriptor reaches
# its kernel archive by climbing out of its own folder to the arch root, so copying one
# engine's folder leaves those kernels unresolvable.
TREE=/tmp/uhd/descriptors
mkdir -p "$TREE"
cp -r "$ARCH_CONTENT"/. "$TREE"/ 2>/dev/null
cp -r "$SRC/flydsl_poc_scratch/flydsl_descriptors_attention/." "$TREE"/ 2>/dev/null
find "$TREE" -maxdepth 2 -type d | head -12
echo "descriptor files: $(find "$TREE" -name "*.json" | wc -l)"

export HIPDNN_DESCRIPTOR_DIR="$TREE"
export FLYDSL_ATTENTION_HSACO_DIR="$HSACO_DIR"
export HIPDNN_AITER_ASM_DIR="${AITER_ASM:-}"

step "engines this runtime loads"
"$LIST_ENGINES" --plugin-dir "$PLUGIN_DIR" 2>&1 | tee "$OUT/engines.log" | grep -E "^  \S+ \(0x"
echo "engine count: $(grep -cE '^  \S+ \(0x' "$OUT/engines.log")"

step "census: which engine answers which graph"
python3 - "$BENCH" "$CORPUS" "$OUT" "$PLUGIN_DIR" <<'PYEOF'
import json, os, pathlib, re, subprocess, sys
bench, corpus, out, plugin_dir = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), sys.argv[4]
engines = []
for line in (out / "engines.log").read_text(errors="replace").splitlines():
    match = re.match(r"^  (\S+) \((0x[0-9a-fA-F]+)\)", line)
    if match:
        value = int(match.group(2), 16)
        engines.append((match.group(1), value - (1 << 64) if value >= (1 << 63) else value))
graphs = sorted((corpus / "graphs").glob("*.json"))
rows = []
for graph in graphs:
    for name, engine_id in engines:
        proc = subprocess.run([bench, "--graph", str(graph), "--plugin-dir", plugin_dir,
                               "--engine-id", str(engine_id), "--predict-engine"],
                              capture_output=True, text=True, timeout=600)
        row = {"graph": graph.stem, "engine": name, "rc": proc.returncode}
        tail = [l for l in (proc.stdout + "\n" + proc.stderr).splitlines()
                if l.strip() and not l.startswith("{") and "HIP Library Path" not in l]
        row["output"] = tail[-1][:200] if tail else ""
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith("{"):
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row.update({k: doc.get(k) for k in ("status", "prediction_kind", "tflops", "reason")})
                break
        rows.append(row)
(out / "census.json").write_text(json.dumps(rows, indent=1))
print(f"  {len(graphs)} graphs x {len(engines)} engines")
served = {}
for row in rows:
    if row["rc"] == 0:
        served.setdefault(row["graph"], []).append(row["engine"])
print(f"  graphs served by >1 engine: {sum(1 for v in served.values() if len(v) > 1)} of {len(graphs)}")
for name, _ in engines:
    mine = [r for r in rows if r["engine"] == name]
    ok = [r for r in mine if r["rc"] == 0]
    print(f"    {name:34s} serves {len(ok):3d}/{len(mine):3d}")
    if not ok and mine:
        print(f"      declines: {mine[0].get('output', '')[:120]}")
PYEOF

step "flyDSL executes, not just answers"
# An engine that predicts but cannot launch is worse than one that declines: the bake-off
# would rank a kernel that never ran. One real execution, numerically checked by the bench.
FLY_HEX=$(grep -F "hipkernel:FlydslAttention (" "$OUT/engines.log" | head -1 | sed -E 's/.*\((0x[0-9a-fA-F]+)\).*/\1/')
if [ -n "$FLY_HEX" ]; then
    FLY_ID=$(python3 -c "v=int('$FLY_HEX',16); print(v-(1<<64) if v>=(1<<63) else v)")
    FLY_GRAPH=$(python3 - "$OUT/census.json" "$CORPUS" <<'PYEOF'
import json, sys
rows = json.load(open(sys.argv[1]))
mine = [r for r in rows if r["engine"] == "hipkernel:FlydslAttention" and r["rc"] == 0]
print(f"{sys.argv[2]}/graphs/{mine[0]['graph']}.json" if mine else "")
PYEOF
)
    if [ -n "$FLY_GRAPH" ]; then
        echo "executing $FLY_GRAPH"
        "$BENCH" --graph "$FLY_GRAPH" --plugin-dir "$PLUGIN_DIR" --engine-id "$FLY_ID" \
            --verify --json 2>&1 | tail -5
    else
        echo "flyDSL answered no census graph; nothing to execute"
    fi
else
    echo "flyDSL is NOT registered -- see the engine list above"
fi

step "KEEP"
mkdir -p "$KEEP"
cp "$OUT/engines.log" "$OUT/census.json" "$OUT/flydsl-build.log" "$KEEP/" 2>/dev/null
mkdir -p "$KEEP/hsaco" && cp "$HSACO_DIR"/*.hsaco "$KEEP/hsaco/" 2>/dev/null
echo "kept hsacos: $(ls "$KEEP/hsaco" 2>/dev/null | wc -l)"
du -sh "$KEEP" 2>/dev/null

step "done"
date -Is; echo "node $(hostname)"
