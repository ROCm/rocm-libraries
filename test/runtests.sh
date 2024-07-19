#!/bin/bash -ex

# bash runtests.sh

# Path to the rocRollerTests executable
RRTESTS=$(realpath build/rocRollerTests)
# Path to the rrperf script
RRPERF=$(realpath scripts/rrperf)

# Tests for gfx950
F8TESTS=("*GPU_MatrixMultiplyMacroTileF8_16x16x32_NN*"
"*GPU_MatrixMultiplyMacroTileF8_32x32x16_NN*"
"*GPU_MatrixMultiplyMacroTileF8_16x16x32_TN*"
"*GPU_MatrixMultiplyMacroTileF8_32x32x64_TN*"
"*GPU_MatrixMultiplyMacroTileF8_16x16x128_TN*"
"*GPU_MatrixMultiplyABF8_16x16x32*"
"*GPU_MatrixMultiplyABF8_32x32x16*"
"*GPU_MatrixMultiplyABF8_16x16x128*"
"*GPU_MatrixMultiplyABF8_32x32x64*"
"*GPU_BasicGEMMFP8_16x16x32_NT*"
"*GPU_BasicGEMMFP8_16x16x128_NT"
"*GPU_BasicGEMMBF8_16x16x128_NT"
"*GPU_BasicGEMMFP8_32x32x64_NT"
"*GPU_BasicGEMMBF8_32x32x64_NT"
"*GPU_BasicGEMMFP8_16x16x128_TN"
"*GPU_BasicGEMMBF8_16x16x128_TN"
"*GPU_BasicGEMMFP8_32x32x64_TN"
"*GPU_BasicGEMMBF8_32x32x64_TN"
"*GPU_MatrixMultiplyMixed*"
)

F6TESTS=("*GPU_MatrixMultiplyMacroTileF6_16x16x128_TN*"
"*GPU_MatrixMultiplyMacroTileF6_32x32x64_TN*"
"*GPU_BasicGEMMFP6_16x16x128_TN"
"*GPU_BasicGEMMFP6_32x32x64_TN"
"*GPU_BasicGEMMBF6_16x16x128_TN"
"*GPU_BasicGEMMBF6_32x32x64_TN"
)

F4TESTS=("*GPU_MatrixMultiplyMacroTileFP4_16x16x128_TN*"
"*GPU_MatrixMultiplyMacroTileFP4_32x32x64_TN*"
"*GPU_BasicGEMMFP4_16x16x128_TN"
"*GPU_BasicGEMMFP4_32x32x64_TN"
)

MISCTESTS=("*ScaledMatrixMultiplyTestGPU*"
)

SKIPTESTS=("*BasicGEMMFP16Prefetch3*"
"*VectorAddBenchmark*"
)

RRPERF_F8TESTS=("f8gemm_16x16x128_f8f6f4"
"f8gemm_32x32x64_f8f6f4"
)

RRPERF_F6TESTS=("f6gemm_16x16x128_f8f6f4"
"f6gemm_32x32x64_f8f6f4"
)

RRPERF_F4TESTS=("f4gemm_16x16x128_f8f6f4"
"f4gemm_32x32x64_f8f6f4"
)





suite="small"
while getopts "t:" opt; do
    case "${opt}" in
    t) suite="${OPTARG,,}";;
    [?]) echo >&2 "Usage: $0 [-t option]
             option: {f8 | f6 | f4 | small | full}"
         exit 1;;
    esac
done


if [ "$suite" = "full" ]; then
    skip=${SKIPTESTS[*]}
    $RRTESTS --gtest_filter="-${skip// /:}"

    # Run rrperf tests
    for testName in "${RRPERF_F8TESTS[@]}"; do
        $RRPERF run --suite "$testName"
    done
    for testName in "${RRPERF_F4TESTS[@]}"; do
        $RRPERF run --suite "$testName"
    done
    for testName in "${RRPERF_F6TESTS[@]}"; do
        $RRPERF run --suite "$testName"
    done
elif [ "$suite" = "f8" ]; then
    for testName in "${F8TESTS[@]}"; do
        $RRTESTS --gtest_filter="$testName"
    done
    for testName in "${RRPERF_F8TESTS[@]}"; do
        $RRPERF run --suite "$testName"
    done
elif [ "$suite" = "f6" ]; then
    for testName in "${F6TESTS[@]}"; do
        $RRTESTS --gtest_filter="$testName"
    done
    for testName in "${RRPERF_F6TESTS[@]}"; do
        $RRPERF run --suite "$testName"
    done
elif [ "$suite" = "f4" ]; then
    for testName in "${F4TESTS[@]}"; do
        $RRTESTS --gtest_filter="$testName"
    done
    for testName in "${RRPERF_F4TESTS[@]}"; do
        $RRPERF run --suite "$testName"
    done
else
    # Loop through each test and execute it
    for testName in "${F8TESTS[@]}"; do
        $RRTESTS --gtest_filter="$testName"
    done
    for testName in "${F6TESTS[@]}"; do
        $RRTESTS --gtest_filter="$testName"
    done
    for testName in "${F4TESTS[@]}"; do
        $RRTESTS --gtest_filter="$testName"
    done
    for testName in "${MISCTESTS[@]}"; do
        $RRTESTS --gtest_filter="$testName"
    done

    # Run rrperf tests
    for testName in "${RRPERF_F8TESTS[@]}"; do
        $RRPERF run --suite "$testName"
    done
    for testName in "${RRPERF_F4TESTS[@]}"; do
        $RRPERF run --suite "$testName"
    done
    for testName in "${RRPERF_F6TESTS[@]}"; do
        $RRPERF run --suite "$testName"
    done
fi

