#!/usr/bin/env bash
# V1 / V2 / V3 / V4 decision benchmark -- production-representative.
#
# 4-way extension of decision_bench.sh. Same methodology, but a 4-version
# Latin-square rotation (v1 v2 v3 v4 / v2 v3 v4 v1 / v3 v4 v1 v2 / v4 v1 v2 v3).
#
# For each (workload, variant) in {simple,medium,complex} x {literal,placeholder}:
#   1. Build all four executables
#   2. Extract codegen metrics per version
#   3. Equivalence check (V1 == V2 == V3 == V4 exit code per workload)
#   4. Compile-time: paired rebuilds in 4-way rotation
#   5. Runtime: warmup + paired rounds in 4-way rotation
#
# Outputs (under data/<commit>/decision_v4/):
#   compile_<wl>_<var>.csv        cols: round,order_idx,version,compile_ms
#   runtime_<wl>_<var>.csv        cols: round,order_idx,version,runtime_ms,sclk_mhz
#   codegen_<wl>_<var>.csv        cols: version,asm_lines,vgpr,sgpr,scratch
#   verify_<wl>_<var>.csv         cols: version,exit_code
#
# Usage: decision_bench_v4.sh
# Run inside the rocm72-patched-clang container; repo at /workspace/ck.

set -euo pipefail

# ---------- Config (env-overridable) ----------
N_COMPILE="${N_COMPILE:-10}"
N_RUNTIME="${N_RUNTIME:-20}"
WARMUP_RUNTIME=3
LOOP_ITERS="${LOOP_ITERS:-10000}"

REPO_ROOT="${REPO_ROOT:-/workspace/ck}"
EXP_DIR="$REPO_ROOT/experiments/transform_graph"
WORKLOADS_DIR="$EXP_DIR/workloads"
COMMIT="${COMMIT:-$(git -C "$REPO_ROOT" rev-parse --short=10 HEAD 2>/dev/null || echo unknown)}"
DATA_DIR="$EXP_DIR/data/$COMMIT/decision_v4"
mkdir -p "$DATA_DIR"

BUILD_DIR="${BUILD_DIR:-/tmp/cmillett-decision-bench-v4}"
mkdir -p "$BUILD_DIR"

COMPILER="${COMPILER:-/opt/llvm-patched/bin/clang++}"
LLVM_OBJDUMP="${LLVM_OBJDUMP:-/opt/llvm-patched/bin/llvm-objdump}"
LLVM_READOBJ="${LLVM_READOBJ:-/opt/llvm-patched/bin/llvm-readobj}"
GFX="${GFX:-gfx90a}"

HOST_CFLAGS=( -std=c++20 -O3
              -D__HIP_PLATFORM_AMD__ -D__HIPCC__
              -isystem /opt/rocm/include
              -I "$REPO_ROOT/include"
              --offload-arch="$GFX"
              -x hip
              -B/opt/rocm/llvm/bin )
COMPILE_FLAGS=( "${HOST_CFLAGS[@]}" -c )
DEVICE_FLAGS=(  "${HOST_CFLAGS[@]}" -c --cuda-device-only )
LINK_FLAGS=(    "${HOST_CFLAGS[@]}" --hip-link -L/opt/rocm/lib -lamdhip64 )

read -ra WORKLOADS <<< "${WORKLOADS:-simple medium complex fa higharity}"
VARIANTS=( literal placeholder )
VERSIONS=( v1 v2 v3 v4 )
# 4-way Latin-square rotations (cyclic).
ORDERS=( "v1 v2 v3 v4" "v2 v3 v4 v1" "v3 v4 v1 v2" "v4 v1 v2 v3" )
N_ORDERS=${#ORDERS[@]}

# ---------- helpers ----------
time_ms() {
    local src="$1" out="$2"
    local s e
    s=$(date +%s%N)
    "$COMPILER" "${COMPILE_FLAGS[@]}" "$src" -o "$out" >/dev/null 2>&1
    e=$(date +%s%N)
    echo $(( (e - s) / 1000000 ))
}

scrub_objs() { rm -f "$BUILD_DIR"/v[1234].o; }

extract_codegen() {
    local src="$1" version="$2"
    local bundle="$BUILD_DIR/${version}_dev.o"
    local elf="$BUILD_DIR/${version}_${GFX}.elf"
    rm -f "$bundle" "$elf"
    "$COMPILER" "${DEVICE_FLAGS[@]}" "$src" -o "$bundle" >/dev/null 2>&1 || { echo "$version,0,0,0,0"; return; }
    /opt/llvm-patched/bin/clang-offload-bundler --type=o --unbundle \
        "--input=$bundle" "--output=$elf" \
        "--targets=hipv4-amdgcn-amd-amdhsa--$GFX" >/dev/null 2>&1 || true
    [[ -f "$elf" ]] || { echo "$version,0,0,0,0"; return; }
    local asm vgpr sgpr scratch
    asm=$( "$LLVM_OBJDUMP" -d --mcpu="$GFX" "$elf" 2>/dev/null | grep -cE "^[[:space:]]+[a-z]" ) || asm=0
    local notes
    notes=$( "$LLVM_READOBJ" --notes "$elf" 2>/dev/null || true )
    vgpr=$(   echo "$notes" | grep -m1 "vgpr_count:"                | awk '{print $NF}' )
    sgpr=$(   echo "$notes" | grep -m1 "sgpr_count:"                | awk '{print $NF}' )
    scratch=$( echo "$notes" | grep -m1 "private_segment_fixed_size:" | awk '{print $NF}' )
    : "${vgpr:=0}" "${sgpr:=0}" "${scratch:=0}"
    echo "$version,$asm,$vgpr,$sgpr,$scratch"
}

read_sclk() {
    rocm-smi --showclocks 2>/dev/null | awk -F'[(): ]+' '/sclk clock level/ {print $NF; exit}' \
        || echo "?"
}

# ---------- Phase A: ensure GPU is in a stable state ----------
echo "== Setting GPU perflevel high =="
rocm-smi --setperflevel high 2>&1 | tail -3 || echo "(rocm-smi setperflevel skipped -- no permission?)"
echo

# ---------- main loop over (workload, variant) ----------
echo "Commit: $COMMIT  N_COMPILE=$N_COMPILE  N_RUNTIME=$N_RUNTIME  LOOP_ITERS=$LOOP_ITERS  GFX=$GFX"
echo "Out dir: $DATA_DIR"
echo

for WL in "${WORKLOADS[@]}"; do
    for VAR in "${VARIANTS[@]}"; do
        echo "=== $WL / $VAR ==="

        # Resolve sources, copy into ramdisk for clean rebuilds
        for V in "${VERSIONS[@]}"; do
            SRC="$WORKLOADS_DIR/test_${WL}_${V}_${VAR}.cpp"
            [[ -f "$SRC" ]] || { echo "missing $SRC" >&2; exit 1; }
            cp "$SRC" "$BUILD_DIR/${V}.cpp"
        done

        # ---- Compile-time: paired rebuilds in Latin-square rotation ----
        OUT="$DATA_DIR/compile_${WL}_${VAR}.csv"
        echo "round,order_idx,version,compile_ms" > "$OUT"
        # 1 cold-cache warmup (discarded)
        scrub_objs
        for V in "${VERSIONS[@]}"; do _=$(time_ms "$BUILD_DIR/${V}.cpp" "$BUILD_DIR/${V}.o"); done
        for ((r=1; r<=N_COMPILE; r++)); do
            order_idx=$(( (r - 1) % N_ORDERS ))
            order="${ORDERS[$order_idx]}"
            for V in $order; do
                scrub_objs
                ms=$(time_ms "$BUILD_DIR/${V}.cpp" "$BUILD_DIR/${V}.o")
                echo "$r,$order_idx,$V,$ms" >> "$OUT"
            done
        done
        echo "  compile: $(wc -l < "$OUT") rows  -> $OUT"

        # ---- Codegen (deterministic) ----
        OUT="$DATA_DIR/codegen_${WL}_${VAR}.csv"
        echo "version,asm_lines,vgpr,sgpr,scratch" > "$OUT"
        for V in "${VERSIONS[@]}"; do
            extract_codegen "$BUILD_DIR/${V}.cpp" "$V" >> "$OUT"
        done
        echo "  codegen: -> $OUT"

        # ---- Runtime: build executables, then Latin-square rotation ----
        for V in "${VERSIONS[@]}"; do
            "$COMPILER" "${LINK_FLAGS[@]}" "$BUILD_DIR/${V}.cpp" -o "$BUILD_DIR/${V}_run" 2>"$BUILD_DIR/link.err" || true
        done

        # Equivalence: all 4 versions must agree on exit code
        VERIFY="$DATA_DIR/verify_${WL}_${VAR}.csv"
        echo "version,exit_code" > "$VERIFY"
        declare -A rcs=()
        for V in "${VERSIONS[@]}"; do
            set +e
            N_TRIALS=0 LOOP_ITERS=$LOOP_ITERS "$BUILD_DIR/${V}_run" >/dev/null 2>/dev/null
            rcs[$V]=$?
            set -e
            echo "$V,${rcs[$V]}" >> "$VERIFY"
        done
        equiv=1
        ref="${rcs[v1]}"
        for V in "${VERSIONS[@]}"; do
            [ "${rcs[$V]}" = "$ref" ] || equiv=0
        done
        if [ $equiv -eq 1 ]; then
            echo "  verify: PASS (exit $ref)  -> $VERIFY"
        else
            echo "  verify: MISMATCH v1=${rcs[v1]} v2=${rcs[v2]} v3=${rcs[v3]} v4=${rcs[v4]}  -> $VERIFY"
        fi

        # Warmup (discarded): WARMUP_RUNTIME launches per version
        for V in "${VERSIONS[@]}"; do
            for ((w=1; w<=WARMUP_RUNTIME; w++)); do
                set +e
                N_TRIALS=1 LOOP_ITERS=$LOOP_ITERS "$BUILD_DIR/${V}_run" >/dev/null 2>/dev/null
                set -e
            done
        done

        # Measurement
        OUT="$DATA_DIR/runtime_${WL}_${VAR}.csv"
        echo "round,order_idx,version,runtime_ms,sclk_mhz" > "$OUT"
        for ((r=1; r<=N_RUNTIME; r++)); do
            order_idx=$(( (r - 1) % N_ORDERS ))
            order="${ORDERS[$order_idx]}"
            sclk=$(read_sclk)
            for V in $order; do
                set +e
                N_TRIALS=1 LOOP_ITERS=$LOOP_ITERS "$BUILD_DIR/${V}_run" 2>"$BUILD_DIR/run.err" >/dev/null
                set -e
                # Last trial line: "<wl> <V> trial 1: T ms"
                ms=$(grep -oE "trial 1: [0-9.]+" "$BUILD_DIR/run.err" | tail -1 | awk '{print $NF}')
                : "${ms:=0}"
                echo "$r,$order_idx,$V,$ms,$sclk" >> "$OUT"
            done
        done
        echo "  runtime: $(wc -l < "$OUT") rows  -> $OUT"
        echo
    done
done

echo "Done.  All decision CSVs under $DATA_DIR"
