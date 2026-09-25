# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Compile a bf16 RMSEpilogue LibraryLogic YAML and benchmark it.
# Usage: ./run_benchmark.sh <logic_yaml>

set -euo pipefail

die() { echo "error: $*" >&2; exit 1; }

[[ $# -eq 1 ]] || die "usage: $0 <logic_yaml>"
LOGIC_YAML=$(realpath "$1")
[[ -f "$LOGIC_YAML" ]] || die "not found: $LOGIC_YAML"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
TENSILELITE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
TENSILE_CLIENT="$TENSILELITE_ROOT/build_tmp/tensilelite/client/tensilelite-client"

ARCH=$(grep -m1 '^ArchitectureName:' "$LOGIC_YAML" | awk '{print $2}')

BUILD_DIR=$(mktemp -d "/tmp/$(basename "$LOGIC_YAML" .yaml)_XXXXXX")
trap 'rm -rf "$BUILD_DIR"' EXIT
mkdir -p "$BUILD_DIR/logic"
cp "$LOGIC_YAML" "$BUILD_DIR/logic/"

echo "Compiling library for $ARCH..."
python3 -m Tensile.TensileCreateLibrary \
    --architecture "$ARCH" --library-format yaml --no-lazy-library-loading \
    --jobs "$(nproc)" "$BUILD_DIR/logic" "$BUILD_DIR/output" HIP \
    2>&1 | tee "$BUILD_DIR/tcl.log"

LIB_YAML=$(find "$BUILD_DIR/output" -name "TensileLibrary_*.yaml" | head -1)
LIB_CO=$(find   "$BUILD_DIR/output" -name "TensileLibrary_*.co"   | head -1)
[[ -f "$LIB_YAML" && -f "$LIB_CO" ]] || die "build failed (see $BUILD_DIR/tcl.log)"

echo "Running benchmark..."
"$TENSILE_CLIENT" \
    --library-file "$LIB_YAML" --code-object "$LIB_CO" \
    --problem-identifier Contraction_l_Alik_Bljk_Cijk_Dijk \
    --problem-size 2048,16384,1,2048 \
    --type BFloat16 --a-type BFloat16 --b-type BFloat16 --c-type BFloat16 --d-type BFloat16 \
    --alpha-type Float --beta-type Float \
    --compute-input-type-A BFloat16 --compute-input-type-B BFloat16 \
    --f32-xdl-math-op Float \
    --high-precision-accumulate --strided-batched \
    --use-rms-epilogue --rms-epilogue-gamma-type BFloat16 --rms-epilogue-residual-type BFloat16 \
    --init-beta Zero \
    --init-a TrigSin --init-b TrigCos \
    --rotating-buffer-size 1024 --icache-rotate-copies -1 \
    --num-warmups 20 --num-enqueues-per-sync 20 --num-syncs-per-benchmark 1 --num-benchmarks 1 \
    --sleep-percent 0 \
    --best-solution

"$TENSILE_CLIENT" \
    --library-file "$LIB_YAML" --code-object "$LIB_CO" \
    --problem-identifier Contraction_l_Alik_Bljk_Cijk_Dijk \
    --problem-size 2048,16384,1,2048 \
    --type BFloat16 --a-type BFloat16 --b-type BFloat16 --c-type BFloat16 --d-type BFloat16 \
    --alpha-type Float --beta-type Float \
    --compute-input-type-A BFloat16 --compute-input-type-B BFloat16 \
    --f32-xdl-math-op Float \
    --high-precision-accumulate --strided-batched \
    --use-rms-epilogue --rms-epilogue-gamma-type BFloat16 --rms-epilogue-residual-type BFloat16 \
    --init-beta Zero \
    --init-a TrigSin --init-b TrigCos \
    --rotating-buffer-size 1024 --icache-rotate-copies -1 \
    --num-warmups 20 --num-enqueues-per-sync 20 --num-syncs-per-benchmark 1 --num-benchmarks 1 \
    --sleep-percent 0 
