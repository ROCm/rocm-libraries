#!/bin/bash -ex

# bash runtests.sh

# Path to the rocRollerTests executable
RRTESTS=$(realpath build/rocRollerTests)
# Path to the rrperf script
RRPERF=$(realpath scripts/rrperf)

# Tests for gfx950
TESTS=("*GPU_MatrixMultiplyMacroTileF8_16x16x32_NN*"
"*GPU_MatrixMultiplyMacroTileF8_32x32x16_NN*"
"*GPU_MatrixMultiplyMacroTileF8_16x16x32_TN*"
"*GPU_MatrixMultiplyMacroTileFP8_32x32x64_TN*"
"*GPU_MatrixMultiplyMacroTileFP8_16x16x128_TN*"
"*GPU_MatrixMultiplyMacroTileFP6_32x32x64_TN*"
"*GPU_MatrixMultiplyMacroTileF6_16x16x128_TN*"
"*GPU_MatrixMultiplyMacroTileF6_32x32x64_TN*"
"*GPU_MatrixMultiplyMacroTileFP4_16x16x128_TN*"
"*GPU_MatrixMultiplyMacroTileFP4_32x32x64_TN*"
"*GPU_MatrixMultiplyABF8_16x16x32*"
"*GPU_MatrixMultiplyABF8_32x32x16*"
"*GPU_MatrixMultiplyABF8_16x16x128*"
"*GPU_MatrixMultiplyABF8_32x32x64*"
"*ScaledMatrixMultiplyTestGPU*"
"*GPU_BasicGEMMFP8_16x16x32_NT*"
"*GPU_BasicGEMMFP8_16x16x128_NT"
"*GPU_BasicGEMMBF8_16x16x128_NT"
"*GPU_BasicGEMMFP8_32x32x64_NT"
"*GPU_BasicGEMMBF8_32x32x64_NT"
"*GPU_BasicGEMMFP8_16x16x128_TN"
"*GPU_BasicGEMMBF8_16x16x128_TN"
"*GPU_BasicGEMMFP8_32x32x64_TN"
"*GPU_BasicGEMMBF8_32x32x64_TN"
"*GPU_BasicGEMMFP4_16x16x128_TN"
"*GPU_BasicGEMMFP4_32x32x64_TN"
)

RRPERF_TESTS=("f8gemm_16x16x128_f8f6f4"
"f8gemm_32x32x64_f8f6f4"
"f4gemm_16x16x128_f8f6f4"
"f4gemm_32x32x64_f8f6f4"
)

# Loop through each test and execute it
for testName in "${TESTS[@]}"; do
    $RRTESTS --gtest_filter="$testName"
done
# Run rrperf tests
for testName in "${RRPERF_TESTS[@]}"; do
    $RRPERF run --suite "$testName"
done
