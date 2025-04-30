#!/bin/bash

################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################


ROCROLLER_BUILD_DIR="$PWD/build"
RUN_CLIENT_TESTS=n
SUITE="small"
while getopts "ct:b:" opt; do
    case "${opt}" in
    b)  ROCROLLER_BUILD_DIR="${OPTARG}"
        ;;
    c)  RUN_CLIENT_TESTS=y
        ;;
    t)
        SUITE="${OPTARG,,}"
        ;;
    [?])
        echo >&2 "Usage: $0 [-t option] [-c]
             option: {f16 | f8f6f4 | f8 | f6 | f4 | mixed | scaled | transpose | small | full}
             -b: path to RR build directory [Default: \"$ROCROLLER_BUILD_DIR\"]
             -c: enables client tests.
                 Default: always enabled with small & full, disabled otherwise."
        exit 1
        ;;
    esac
done

# So that rrperf can find the our binaries
export ROCROLLER_BUILD_DIR

# Path to the rocRollerTests executable
RRTESTS=$(realpath ${ROCROLLER_BUILD_DIR}/bin/rocRollerTests)
if [[ $? -ne 0 ]]; then
  echo "ERROR: could not find rocRollerTests in $ROCROLLER_BUILD_DIR/bin directory."
  echo "WARN: you can specify a custom build path by passing -b <RR build path>."
  exit 1
fi

RRTESTSCATCH=$(realpath ${ROCROLLER_BUILD_DIR}/bin/rocRollerTests_catch)
if [[ $? -ne 0 ]]; then
  echo "ERROR: could not find rocRollerTests_catch in $ROCROLLER_BUILD_DIR/bin directory."
  echo "WARN: you can specify a custom build path by passing -b <RR build path>."
  exit 1
fi

# Path to the rrperf script
RRPERF=$(realpath scripts/rrperf)
if [[ $? -ne 0 ]]; then
  echo "ERROR: could not find rrperf in $PWD/scripts directory."
  echo "WARN: make sure you are running from the root of a RR work directory."
  exit 1
fi

set -x

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

MEMORYTESTS=(
"*MemoryInstructionsTest*"
"*GPU_BufferLoad2LDSTest*"
"*GlobalMemoryInstructionsTest*"
"*MemoryInstructionsLDSTest*"
)

SKIPTESTS=("*BasicGEMMFP16Prefetch3*"
"*VectorAddBenchmark*"
)

RRCATCHTESTS=("[lds][gpu]"
"[largerLDS][gpu]"
"[global-load-store][gpu]"
"[prng][gpu]"
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
# For now just include all listed catch tests.
# Update this in the future as we migrate tests from google tests.
RRCATCHTESTS_LIST="${RRCATCHTESTS[@]}"
case "${SUITE}" in
  "f16" | "f8f6f4" | "f8" | "f6" | "f4" | "mixed" | "scaled" | "transpose" | "memory")
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
               "${MEMORYTESTS[@]}"    \
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
RRCATCHTESTS_LIST=($(echo "${RRCATCHTESTS_LIST[@]}" | tr " " "\n" | sort -u | tr "\n" " "))
RRPERF_TESTS_LIST=($(echo "${RRPERF_TESTS_LIST[@]}" | tr " " "\n" | sort -u | tr "\n" " "))

for testName in "${RRTESTS_LIST[@]}"; do
    $RRTESTS --gtest_filter="$testName"
done

for testName in "${RRCATCHTESTS_LIST[@]}"; do
    $RRTESTSCATCH "$testName"
done

for testName in "${RRPERF_TESTS_LIST[@]}"; do
    $RRPERF run --suite "$testName"
done
