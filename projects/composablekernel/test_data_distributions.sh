#!/bin/bash
# Test GEMM benchmark with different data distributions on 8192x8192x8192

set -e

KERNEL="build/bin/benchmark_gemm_universal_fp16_ccr_compv3_cshuffle_intrawave_False_False_False_False_64x64x64_2x2x1_16x16x32"
SIZE=8192
WARMUP=5
REPEAT=10

echo "================================================================================"
echo "Testing Data Distribution Impact on GEMM Performance"
echo "================================================================================"
echo "Kernel: $KERNEL"
echo "Matrix Size: ${SIZE}x${SIZE}x${SIZE}"
echo "Warmup: $WARMUP iterations"
echo "Repeat: $REPEAT iterations"
echo "================================================================================"
echo ""

if [ ! -f "$KERNEL" ]; then
    echo "Error: Kernel binary not found: $KERNEL"
    exit 1
fi

# Test each data distribution
declare -A DISTRIBUTIONS=(
    ["0"]="Uniform Random [-1,1]"
    ["1"]="Monotonic Sequence"
    ["2"]="Constant (1.0)"
    ["3"]="Zeros"
    ["4"]="Constant (π)"
    ["5"]="Checkerboard Pattern"
)

for init in 0 1 2 3 4 5; do
    dist_name="${DISTRIBUTIONS[$init]}"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Distribution #$init: $dist_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    $KERNEL \
        -m=$SIZE \
        -n=$SIZE \
        -k=$SIZE \
        -init=$init \
        -warmup=$WARMUP \
        -repeat=$REPEAT \
        -verify=0 \
        -json_output=true 2>&1 | grep -E "latency|tflops|bandwidth" || echo "Failed to run benchmark for init=$init"

    echo ""
done

echo "================================================================================"
echo "Test Complete!"
echo "================================================================================"
