#!/bin/bash -x

# bash runtests.sh

# Path to the rocRollerTests executable
RRTESTS=$(realpath build/rocRollerTests)
# Path to the rrperf script
RRPERF=$(realpath scripts/rrperf)

# Tests for gfx950
F16TESTS=("*GPU_MatrixMultiplyMacroTileF16*"
"*GPU_BasicGEMMF16*"
)

F8F6F4TESTS=("*GPU_MatrixMultiplyMacroTileF8F6F4*"
"*GPU_ScaledMatrixMultiplyMacroTileF8F6F4*"
"*GPU_ScaledMatrixMultiplyMacroTileMixed*"
"*GPU_MatrixMultiplyABF8F6F4*"
"*GPU_BasicGEMMF8F6F4*"
"*GPU_ScaledBasicGEMMF8F6F4*"
"*GPU_ScaledMixedBasicGEMMF8F6F4*"
)

F8TESTS=()

F6TESTS=()

F4TESTS=()

SCALEDTESTS=("*GPU_ScaledMatrixMultiplyMacroTileF8F6F4*"
"*GPU_ScaledMatrixMultiplyMacroTileMixed*"
"*GPU_ScaledBasicGEMMF8F6F4*"
"*GPU_ScaledMixedBasicGEMMF8F6F4*"
)

MIXEDTESTS=("*GPU_MatrixMultiplyMacroTileMixed*"
"*GPU_ScaledMatrixMultiplyMacroTileMixed*"
"*GPU_MixedBasicGEMMF8F6F4*"
"*GPU_ScaledMixedBasicGEMMF8F6F4*"
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

SKIPTESTS=("*BasicGEMMFP16Prefetch3*"
"*VectorAddBenchmark*"
)

RRPERF_F16TESTS=("f16gemm_16x16x32_fp16_NN"
"f16gemm_16x16x32_fp16_NT"
"f16gemm_16x16x32_fp16_TN"
"f16gemm_16x16x32_fp16_TT"
"f16gemm_32x32x16_fp16_NN"
"f16gemm_32x32x16_fp16_NT"
"f16gemm_32x32x16_fp16_TN"
"f16gemm_32x32x16_fp16_TT"
)

RRPERF_F8TESTS=("f8gemm_16x16x128_f8f6f4_NN"
"f8gemm_16x16x128_f8f6f4_NT"
"f8gemm_16x16x128_f8f6f4_TN"
"f8gemm_16x16x128_f8f6f4_TT"
"f8gemm_32x32x64_f8f6f4_NN"
"f8gemm_32x32x64_f8f6f4_NT"
"f8gemm_32x32x64_f8f6f4_TN"
"f8gemm_32x32x64_f8f6f4_TT"
)

RRPERF_F6TESTS=("f6gemm_16x16x128_f8f6f4_NN"
"f6gemm_16x16x128_f8f6f4_NT"
"f6gemm_16x16x128_f8f6f4_TN"
"f6gemm_16x16x128_f8f6f4_TT"
"f6gemm_32x32x64_f8f6f4_NN"
"f6gemm_32x32x64_f8f6f4_NT"
"f6gemm_32x32x64_f8f6f4_TN"
"f6gemm_32x32x64_f8f6f4_TT"
)

RRPERF_F4TESTS=("f4gemm_16x16x128_f8f6f4_NN"
"f4gemm_16x16x128_f8f6f4_NT"
"f4gemm_16x16x128_f8f6f4_TN"
"f4gemm_16x16x128_f8f6f4_TT"
"f4gemm_32x32x64_f8f6f4_NN"
"f4gemm_32x32x64_f8f6f4_NT"
"f4gemm_32x32x64_f8f6f4_TN"
"f4gemm_32x32x64_f8f6f4_TT"
)

RRPERF_MIXEDTESTS=("gemm_mixed_16x16x128_f8f6f4"
"gemm_mixed_32x32x64_f8f6f4"
)

RRPERF_F8F6F4TESTS=()
RRPERF_SCALEDTESTS=()
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
             option: {f16 | f8f6f4 | f8 | f6 | f4 | mixed | scaled | transpose | small | full}
             -c: enables client tests.
                 Default: always enabled with small & full, disabled otherwise."
        exit 1
        ;;
    esac
done

case "${SUITE}" in
  "f16" | "f8f6f4" | "f8" | "f6" | "f4" | "mixed" | "scaled" | "transpose")
      RRTESTS_VARNAME="${SUITE^^}TESTS"
      RRPERF_TESTS_VARNAME="RRPERF_${SUITE^^}TESTS"
      read -r -a RRTESTS_LIST <<<"$(eval echo "\${${RRTESTS_VARNAME}[@]}")"
      if [[ ${RUN_CLIENT_TESTS} == "y" ]]; then
          read -r -a RRPERF_TESTS_LIST <<<"$(eval echo "\${${RRPERF_TESTS_VARNAME}[@]}")"
      fi
      ;;
  "small" | "full")
      if [ "$SUITE" == "full" ]; then
          for t in "${SKIPTESTS[@]}"  ; do
              RRTESTS_LIST+=("$t")
          done
      fi
      for t in "${F16TESTS[@]}"       \
               "${F8F6F4TESTS[@]}"    \
               "${F8TESTS[@]}"        \
               "${F6TESTS[@]}"        \
               "${F4TESTS[@]}"        \
               "${SCALEDTESTS[@]}"    \
               "${MIXEDTESTS[@]}"     \
               "${TRANSPOSETESTS[@]}" ; do
          RRTESTS_LIST+=("$t")
      done
      for t in "${RRPERF_F16TESTS[@]}"       \
               "${RRPERF_F8F6F4TESTS[@]}"    \
               "${RRPERF_F8TESTS[@]}"        \
               "${RRPERF_F6TESTS[@]}"        \
               "${RRPERF_F4TESTS[@]}"        \
               "${RRPERF_SCALEDTESTS[@]}"    \
               "${RRPERF_MIXEDTESTS[@]}"     \
               "${RRPERF_TRANSPOSETESTS[@]}" ; do
          RRPERF_TESTS_LIST+=("$t")
      done
      ;;
esac

# remove duplicates
RRTESTS_LIST=($(echo "${RRTESTS_LIST[@]}" | tr " " "\n" | sort -u | tr "\n" " "))
RRPERF_TESTS_LIST=($(echo "${RRPERF_TESTS_LIST[@]}" | tr " " "\n" | sort -u | tr "\n" " "))

for testName in "${RRTESTS_LIST[@]}"; do
    $RRTESTS --gtest_filter="$testName"
done

for testName in "${RRPERF_TESTS_LIST[@]}"; do
    $RRPERF run --suite "$testName"
done
