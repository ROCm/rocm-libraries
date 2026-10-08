#!/bin/bash
#SBATCH --job-name=uhd-l1-validate
#SBATCH --output=%x-%j.log
#
# Validation, not generation: does every engine that serves a graph report an L1 number on
# it? Installs no model, so UNAVAILABLE with a reason is the expected answer.
#
# Site settings go on submission (reproduce/README.md, "What every job needs").
set -uo pipefail

BRANCH=${UHD_BRANCH:?set UHD_BRANCH to the rocm-libraries branch to build}
CORPUS=${UHD_CORPUS:-/exchange/corpus-small}
SRC=/tmp/uhd/src
# Descriptor root packed as `product`, relative to the checkout; unset keeps the provider's
# shipped root.
PRODUCTION_ROOT_ARGS=()
[ -n "${UHD_PRODUCTION_ROOT:-}" ] && \
    PRODUCTION_ROOT_ARGS=(-DHIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT="$SRC/$UHD_PRODUCTION_ROOT")
BUILD=/tmp/uhd/build
OUT=/tmp/uhd/out
step() { echo; echo "===== $* ====="; date -Is; }

step "node"
hostname; nproc
rocminfo 2>/dev/null | grep -m1 "Name:.*gfx" || echo "NO GPU VISIBLE"

step "toolchain"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq 2>&1 | tail -1
apt-get install -y -qq git cmake ninja-build python3.12-venv python3-dev 2>&1 | tail -2
if ! command -v cmake >/dev/null || ! command -v ninja >/dev/null; then
    export PIP_CACHE_DIR=/tmp/pipcache
    python3 -m pip install --quiet --break-system-packages cmake ninja 2>&1 | tail -1
    export PATH="$PATH:$(python3 -c "import sysconfig;print(sysconfig.get_path('scripts'))")"
    export PATH="$PATH:$(python3 -m site --user-base)/bin"
fi
python3 -m pip install --quiet --break-system-packages "zstandard>=0.20.0" msgpack \
    lightgbm pandas "scikit-learn>=1.3.0" flatbuffers numpy pyarrow pytest 2>&1 | tail -1
for c in git cmake ninja hipcc python3; do
    printf '%-8s ' "$c"; command -v "$c" >/dev/null && "$c" --version 2>&1 | head -1 || echo MISSING
done

step "source: $BRANCH"
rm -rf /tmp/uhd; mkdir -p /tmp/uhd
# --depth 200, not 1: a bundle needs ancestor commits to apply against.
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 200 --single-branch --branch "$BRANCH" \
    https://github.com/ROCm/rocm-libraries.git "$SRC" > /tmp/uhd/clone.log 2>&1 \
    || { echo "FATAL: clone failed"; tail -5 /tmp/uhd/clone.log; exit 1; }
if [ -n "${UHD_BUNDLE:-}" ]; then
    # Mirrors can lag; a bundle pins the exact commits. Its tip is HEAD, not a branch ref.
    git -C "$SRC" fetch "$UHD_BUNDLE" "${UHD_COMMIT:-HEAD}" >> /tmp/uhd/clone.log 2>&1 \
        && git -C "$SRC" checkout --detach FETCH_HEAD >> /tmp/uhd/clone.log 2>&1 \
        || { echo "FATAL: bundle $UHD_BUNDLE did not apply"; tail -8 /tmp/uhd/clone.log; exit 1; }
fi
git -C "$SRC" log --oneline -1

step "configure"
cmake -S "$SRC" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="${ROCM_PATH:-/opt/rocm}" -DROCM_PATH="${ROCM_PATH:-/opt/rocm}" \
    -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
    -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DROCM_LIBS_ENABLE_COMPONENTS="hipdnn;hipdnn-integration-tests;hip-kernel-provider" \
    -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON -DHIPDNN_ENABLE_SDPA=ON \
    -DHIPKERNELPROVIDER_ENABLE_ROCKE=ON \
    -DHIPKERNELPROVIDER_KPACK_ALLOW_FETCH=ON \
    "${PRODUCTION_ROOT_ARGS[@]}" \
    -DGPU_TARGETS="${UHD_GPU_TARGETS:-gfx942}" -DCMAKE_HIP_ARCHITECTURES="${UHD_GPU_TARGETS:-gfx942}" \
    -DENABLE_CLANG_FORMAT=OFF -DENABLE_CLANG_TIDY=OFF > /tmp/uhd/configure.log 2>&1
echo "configure rc=$?"
grep -B2 -A10 "CMake Error" /tmp/uhd/configure.log | head -30 || tail -3 /tmp/uhd/configure.log

step "build"
# Build ALL: descriptor staging rides on `add_custom_target(... ALL)` rules that a target
# list skips.
cmake --build "$BUILD" -j "$(nproc)" > /tmp/uhd/build.log 2>&1
BLD=$?; echo "build rc=$BLD"
if [ "$BLD" -ne 0 ]; then
    grep -n -B2 -A20 "^FAILED:" /tmp/uhd/build.log | head -50; exit 1
fi

step "layout"
BENCH=$(find "$BUILD" -name hipdnn_bench -type f -perm -u+x | head -1)
LIST=$(find "$BUILD" -name hipdnn_list_engines -type f -perm -u+x | head -1)
PLUGIN_DIR=$(dirname "$(find "$BUILD" -name "libhip_kernel_provider*.so" -path "*plugins*" | head -1)")
MIOPEN_DIR=$(dirname "$(find "$BUILD" -name "libmiopen_plugin*.so" -path "*plugins*" | head -1)" 2>/dev/null)
echo "bench=$BENCH"; echo "plugins=$PLUGIN_DIR"; echo "miopen=${MIOPEN_DIR:-none}"
mkdir -p "$OUT"
PLUGIN_ARGS=(--plugin-dir "$PLUGIN_DIR")
[ -n "${MIOPEN_DIR:-}" ] && [ "$MIOPEN_DIR" != "$PLUGIN_DIR" ] && PLUGIN_ARGS+=(--plugin-dir "$MIOPEN_DIR")

step "engines this build ships"
"$LIST" "${PLUGIN_ARGS[@]}" 2>&1 | tee "$OUT/engines.log" | grep -E "^  \S" | head -40

step "L1 matrix over $(ls "$CORPUS/graphs" | wc -l) graphs"
# One row per (graph, engine): does the engine serve it, and what does its L1 query say?
python3 - "$BENCH" "$CORPUS" "$OUT" "${PLUGIN_ARGS[@]}" <<'PYEOF'
import json, os, re, subprocess, sys, pathlib
bench, corpus, out = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
plugin_args = sys.argv[4:]
engines = []
listing = (out / "engines.log").read_text(errors="replace")
for line in listing.splitlines():
    match = re.match(r"^  (\S+) \((0x[0-9a-fA-F]+)\)", line)
    if match:
        value = int(match.group(2), 16)
        engines.append((match.group(1), value - (1 << 64) if value >= (1 << 63) else value))
print(f"  engines: {len(engines)}")
rows, graphs = [], sorted([*(corpus / "graphs").glob("*.json"), *(corpus / "graphs").glob("*.fb")])
for graph in graphs:
    for name, engine_id in engines:
        command = [bench, "--graph", str(graph), *plugin_args,
                   "--engine-id", str(engine_id), "--predict-engine"]
        proc = subprocess.run(command, capture_output=True, text=True, timeout=600)
        row = {"graph": graph.stem, "engine": name, "rc": proc.returncode}
        # A declining engine says why on stderr; keep it, not just the JSON line.
        tail = [line for line in (proc.stdout + "\n" + proc.stderr).splitlines()
                if line.strip() and not line.startswith("{") and "HIP Library Path" not in line]
        row["output"] = tail[-1][:200] if tail else ""
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith("{"):
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row.update({k: doc.get(k) for k in
                            ("status", "prediction_kind", "metric", "value", "model", "reason", "is_valid")})
                break
        rows.append(row)
(out / "l1-matrix.json").write_text(json.dumps(rows, indent=1))

served = {}
for row in rows:
    if row["rc"] == 0:
        served.setdefault(row["graph"], []).append(row["engine"])
print("\n  --- engines answering per graph ---")
for graph in sorted(served):
    print(f"    {graph[:62]:64s} {len(served[graph])}: {', '.join(sorted(served[graph]))}")
contested = {g: e for g, e in served.items() if len(e) > 1}
print(f"\n  graphs served by >1 engine: {len(contested)} of {len(graphs)}")
print("\n  --- per engine ---")
for name, _ in engines:
    mine = [r for r in rows if r["engine"] == name]
    ok = [r for r in mine if r["rc"] == 0]
    statuses = {}
    for r in ok:
        statuses[str(r.get("status"))] = statuses.get(str(r.get("status")), 0) + 1
    print(f"    {name:34s} serves {len(ok):3d}/{len(mine):3d}  {statuses}")
    reason = next((r.get("reason") for r in ok if r.get("reason")), None)
    if reason:
        print(f"      answers with: {str(reason)[:110]}")
    declined = [r for r in mine if r["rc"] != 0]
    if declined:
        seen = {}
        for r in declined:
            seen[r.get("output", "")] = seen.get(r.get("output", ""), 0) + 1
        for text, count in sorted(seen.items(), key=lambda kv: -kv[1])[:2]:
            print(f"      declines {count:3d}x: {text[:110]}")

# The generic decline text is identical whether the engine lacks the op or rejected this
# shape; the provider says which at INFO, so ask once per silent engine.
print("\n  --- why the silent engines decline (HIPDNN_LOG_LEVEL=info, first graph) ---")
verbose = dict(os.environ, HIPDNN_LOG_LEVEL="info")
for name, engine_id in engines:
    if any(r["rc"] == 0 for r in rows if r["engine"] == name):
        continue
    proc = subprocess.run([bench, "--graph", str(graphs[0]), *plugin_args,
                           "--engine-id", str(engine_id), "--predict-engine"],
                          capture_output=True, text=True, timeout=600, env=verbose)
    # Keep the whole transcript: early gates log without the `isApplicable` marker.
    transcript = proc.stdout + "\n" + proc.stderr
    (out / f"decline-{name.replace(':', '_')}.log").write_text(transcript)
    said = [line.strip() for line in transcript.splitlines()
            if line.strip() and not line.startswith("{") and "HIP Library Path" not in line]
    print(f"    {name}")
    for line in said[-4:]:
        print(f"      {line[:150]}")
PYEOF

step "KEEP"
KEEP="${UHD_KEEP:-/exchange/uhd-l1-validate-${SLURM_JOB_ID:-manual}}"
mkdir -p "$KEEP" && cp "$OUT"/engines.log "$OUT"/l1-matrix.json "$OUT"/decline-*.log "$KEEP/" 2>/dev/null
cp /tmp/uhd/build.log "$KEEP/" 2>/dev/null
echo "  -> $KEEP"

step "done"
date -Is; echo "node $(hostname)"
