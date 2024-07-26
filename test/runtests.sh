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
"*GPU_BasicScaledGEMM*"
"*GPU_ScaledGEMMMixed*"
)

TRANSPOSETESTS=(
"*B4Transpose16x128GPUTest"
"*B4Transpose32x64GPUTest"
"*B6AlignedVGPRsTranspose16x128GPUTest"
"*B6AlignedVGPRsTranspose32x64GPUTest"
"*B6UnalignedVGPRsTranspose16x128GPUTest"
"*B6UnalignedVGPRsTranspose32x64GPUTest"
"*B8Transpose16x64GPUTest"
"*B8Transpose32x32GPUTest"
"*B16Transpose16x32GPUTest"
"*B16Transpose32x16GPUTest"
)

MIXEDTESTS=("*GPU_MatrixMultiplyMixed*"
"*GPU_BasicGEMMMixedF8F6F4*"
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

RRPERF_MIXEDTESTS=("gemm_mixed_16x16x128_f8f6f4"
"gemm_mixed_32x32x64_f8f6f4"
)

RRPERF_MISCTESTS=()
RRPERF_TRANSPOSETESTS=()

RRPERF_TESTS_LIST=()
RRTESTS_LIST=()
RUN_CLIENT_TESTS=n
SUITE="small"
while getopts "ct:" opt; do
    case "${opt}" in
    c)  RUN_CLIENT_TESTS=y
        ;;
    t)
        SUITE="${OPTARG,,}"
        ;;
    [?])
        echo >&2 "Usage: $0 [-t option] [-c]
             option: {f8 | f6 | f4 | mixed | scaled | misc | transpose | small | full}
             -c: enables client tests.
                 Default: always enabled with small & full, disabled otherwise."
        exit 1
        ;;
    esac
done

case "${SUITE}" in
  "f8" | "f6" | "f4" | "mixed" | "scaled" | "misc" | "transpose")
      RRTESTS_VARNAME="${SUITE^^}TESTS"
      RRPERF_TESTS_VARNAME="RRPERF_${SUITE^^}TESTS"
      read -r -a RRTESTS_LIST <<<"$(eval echo "\${${RRTESTS_VARNAME}[@]}")"
      if [[ ${RUN_CLIENT_TESTS} == "y" ]]; then
          read -r -a RRPERF_TESTS_LIST <<<"$(eval echo "\${${RRPERF_TESTS_VARNAME}[@]}")"
      fi
      ;;
  "small" | "full")
      if [ "$SUITE" == "full" ]; then
          for t in "${SKIPTESTS[@]}"  \
                   "${MIXEDTESTS[@]}" ; do
              RRTESTS_LIST+=("$t")
          done

          for t in "${RRPERF_MIXEDTESTS[@]}"; do
              RRTESTS_LIST+=("$t")
          done
      fi
      for t in "${F8TESTS[@]}"        \
               "${F6TESTS[@]}"        \
               "${F4TESTS[@]}"        \
               "${MISCTESTS[@]}"      \
               "${TRANSPOSETESTS[@]}" ; do
          RRTESTS_LIST+=("$t")
      done
      for t in "${RRPERF_F8TESTS[@]}"        \
               "${RRPERF_F6TESTS[@]}"        \
               "${RRPERF_F4TESTS[@]}"        \
               "${RRPERF_MISCTESTS[@]}"      \
               "${RRPERF_TRANSPOSETESTS[@]}" ; do
          RRPERF_TESTS_LIST+=("$t")
      done
      ;;
esac

for testName in "${RRTESTS_LIST[@]}"; do
    $RRTESTS --gtest_filter="$testName"
done

for testName in "${RRPERF_TESTS_LIST[@]}"; do
    $RRPERF run --suite "$testName"
done
