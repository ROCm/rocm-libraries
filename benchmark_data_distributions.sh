#!/bin/bash
# Benchmark GEMM kernel with different data distributions
# This script tests performance variations based on input data patterns

set -e

# Configuration
KERNEL_BINARY="${1:-./gemm_benchmark}"
SIZE="${2:-8192}"
WARMUP="${3:-50}"
REPEAT="${4:-100}"
OUTPUT_DIR="${5:-./benchmark_results}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "$OUTPUT_DIR"
RESULTS_FILE="${OUTPUT_DIR}/distribution_benchmark_${SIZE}_${TIMESTAMP}.json"
CSV_FILE="${OUTPUT_DIR}/distribution_benchmark_${SIZE}_${TIMESTAMP}.csv"

# Check if kernel binary exists
if [ ! -f "$KERNEL_BINARY" ]; then
    echo "Error: Kernel binary not found: $KERNEL_BINARY"
    echo "Usage: $0 <kernel_binary> [size] [warmup] [repeat] [output_dir]"
    exit 1
fi

echo "=================================================================="
echo "Benchmarking ${SIZE}x${SIZE}x${SIZE} GEMM with Different Data Distributions"
echo "=================================================================="
echo "Kernel Binary: $KERNEL_BINARY"
echo "Matrix Size: ${SIZE}x${SIZE}x${SIZE}"
echo "Warmup Iterations: $WARMUP"
echo "Benchmark Iterations: $REPEAT"
echo "Output Directory: $OUTPUT_DIR"
echo "=================================================================="
echo ""

# Initialize JSON results file
echo "[" > "$RESULTS_FILE"

# Initialize CSV file
echo "Distribution,Init_Method,Latency_ms,TFlops,Bandwidth_GBps" > "$CSV_FILE"

# Array of distributions to test
declare -a DISTRIBUTIONS=(
    "0:Uniform_Random_[-1,1]"
    "1:Monotonic_Sequence"
    "2:Constant_1.0"
    "3:Zeros"
    "4:Constant_Pi"
    "5:Checkerboard_Pattern"
)

first_entry=true

# Benchmark each distribution
for dist_info in "${DISTRIBUTIONS[@]}"; do
    IFS=':' read -r init_method dist_name <<< "$dist_info"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Distribution: $dist_name (init=$init_method)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Run benchmark
    output=$($KERNEL_BINARY \
        --m "$SIZE" \
        --n "$SIZE" \
        --k "$SIZE" \
        --init "$init_method" \
        --warmup "$WARMUP" \
        --repeat "$REPEAT" \
        --verify 0 \
        --json_output true 2>&1)

    echo "$output"

    # Extract performance metrics (adjust parsing based on actual output format)
    latency=$(echo "$output" | grep -oP '"latency\(ms\)"\s*:\s*\K[0-9.]+' | head -1)
    tflops=$(echo "$output" | grep -oP '"tflops\(TFlops\)"\s*:\s*\K[0-9.]+' | head -1)
    bandwidth=$(echo "$output" | grep -oP '"bandwidth\(GB/s\)"\s*:\s*\K[0-9.]+' | head -1)

    # Add to CSV
    if [ -n "$latency" ] && [ -n "$tflops" ] && [ -n "$bandwidth" ]; then
        echo "$dist_name,$init_method,$latency,$tflops,$bandwidth" >> "$CSV_FILE"

        # Add to JSON (handle comma separation)
        if [ "$first_entry" = true ]; then
            first_entry=false
        else
            echo "," >> "$RESULTS_FILE"
        fi

        cat >> "$RESULTS_FILE" << EOF
  {
    "distribution": "$dist_name",
    "init_method": $init_method,
    "matrix_size": $SIZE,
    "performance": {
      "latency_ms": $latency,
      "tflops": $tflops,
      "bandwidth_gbps": $bandwidth
    }
  }
EOF
    else
        echo "Warning: Could not extract performance metrics for $dist_name"
    fi

    echo ""
done

# Close JSON array
echo "" >> "$RESULTS_FILE"
echo "]" >> "$RESULTS_FILE"

echo ""
echo "=================================================================="
echo "Benchmark Complete!"
echo "=================================================================="
echo "Results saved to:"
echo "  JSON: $RESULTS_FILE"
echo "  CSV:  $CSV_FILE"
echo ""
echo "Summary:"
echo "--------"
cat "$CSV_FILE"

# Generate a simple comparison report
echo ""
echo "=================================================================="
echo "Performance Comparison (sorted by TFlops)"
echo "=================================================================="
tail -n +2 "$CSV_FILE" | sort -t',' -k4 -rn | \
    awk -F',' 'BEGIN {print "Distribution                    | Latency (ms) | TFlops | Bandwidth (GB/s)"}
                     {printf "%-30s | %12.2f | %6.2f | %16.2f\n", $1, $3, $4, $5}'

echo ""
echo "=================================================================="
