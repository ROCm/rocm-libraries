#!/bin/bash -ex

# bash runtests.sh

# Path to the rocRollerTests executable
RRTESTS=$(realpath build/rocRollerTests)

# Tests for gfx950
TESTS=("*GPU_MatrixMultiplyMacroTileF8_16x16x32_NN*"
"*GPU_MatrixMultiplyMacroTileF8_32x32x16_NN*"
"*GPU_MatrixMultiplyMacroTileF8_16x16x32_TN*"
"*GPU_MatrixMultiplyMacroTileFP8_32x32x64_TN*"
"*GPU_MatrixMultiplyMacroTileFP8_16x16x128_TN*"
"*GPU_MatrixMultiplyMacroTileFP6_32x32x64_TN*"
"*GPU_MatrixMultiplyMacroTileFP6_16x16x128_TN*"
"*GPU_MatrixMultiplyMacroTileFP4_32x32x64_TN*"
"*GPU_MatrixMultiplyMacroTileFP4_16x16x128_TN*"
"*GPU_MatrixMultiplyABF8_16x16x32*"
"*GPU_MatrixMultiplyABF8_32x32x16*"
"*ScaledMatrixMultiplyTestGPU*"
"*GPU_BasicGEMMFP8_NT*"
)

# Loop through each test and execute it
for testName in "${TESTS[@]}"; do
    $RRTESTS --gtest_filter="$testName"
done
