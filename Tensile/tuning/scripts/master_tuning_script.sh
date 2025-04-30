#!/usr/bin/env bash

################################################################################
#
# Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

HELP=false
TENSILE_CLIENT=new
COUNT=false
TILE_AWARE=false
MFMA=false
RK=false
DISABLE_STRIDES=false
DVAL=2
DATA_TYPE=sgemm
PROBLEM_DEFINITION=both
INITIALIZATION=rand_int
DEPENDENCIES=false
REDO=false

HELP_STR="
Pre-Requisites:
  >=python3.6 or higher, python3-pip/pip3, python3-yaml, python3-setuptools, python3-distutils,
      python3-venv, wheel, setuptools, pyyaml, msgpack, matplotlib, pandas, and numpy)
  >=llvm-6.0-dev, >=cmake3.5, zlib1g-dev
  >=rocm3.3 stack for hip-clang

About
This is the master tuning script. It does the following:
1) Generates an input yaml file (ROCBLAS_LAYER=2 and ROCBLAS_LAYER=4 are supported) for each of the NN/NT/TN configurations.
2) Runs Tensile for all 3 input yaml files
3) Merges the tuning results (yaml file in 3_LibraryLogic directory) into the existing rocBLAS
4) Runs massage script (adds ldc=ldd sizes) for vega10 and vega20 only
5) Runs rocblas-bench with untuned sizes and tuned sizes
6) Runs analysis script, providing spreadsheets with performance before and after tuning

Usage: $0 WORKING_DIR LOG_PATH [options]

Options:
-h | --help             Display this help message
-n | --network          String to search for in filenames in log directory
--tensile-path PATH     Path to existing Tensile (will not provision new copy)
--rocblas-path PATH     Path to existing rocBLAS (will not provision new copy)
-y | --data-type        Data type of sizes that you want to tune (sgemm, dgemm, hgemm only)
-t | --tile-aware       Use tile-aware method. (limited support)
-m | --mfma             Use MFMA kernels
-r | --rk               Use replacement kernels (sgemm only)
-s | --disable-strides  Disable leading dimensions and strides in tuning file
-f | --sclk             Frequency of sclk in MHz
-c | --count            Sets all cases where count=1 to count=10
-d | --dependencies     Install required dependencies (dependencies are not installed by default)
--redo                  Force logic preparation, merge, massage, and library build steps to be redone

-l | --library \\
    {arcturus | vega20 | vega10 | mi25 | r9nano | hip}      GPU used for tuning (arcturus, mi25, mi50, mi60, r7, v340 only)
--initialization {rand_int | trig_float | hpl} (=rand_int)  Data initialization for matrices
--problem-definition {gemm | batch | both} (=both)          Which problem types to tune
--client {new | old | both} (=new)                          Which Tensile runtime client to use
"

if ! OPTS=$(getopt -o h,l:,n:,y:,t,m,r,s,i:,f:,c,d \
--long help,library:,network:,tensile-path:,rocblas-path:,data-type:,tile-aware,mfma,rk,\
disable-strides,initialization,problem-definition,client,sclk:,count,dependencies,redo -n 'parse-options' -- "$@")
then
  echo "Failed parsing options"
  exit 1
fi

eval set -- "${OPTS}"

while true; do
  case ${1} in
    -h | --help )               HELP=true; shift ;;
    -l | --library )            LIBRARY=${2}; shift 2;;
    -n | --network )            NETWORK=${2}; shift 2;;
    --tensile-path )            TENSILE_PATH=${2}; shift 2;;
    --rocblas-path )            ROCBLAS_PATH=${2}; shift 2;;
    -y | --data-type )          DATA_TYPE=${2}; shift 2;;
    -t | --tile-aware )         TILE_AWARE=true; shift ;;
    -m | --mfma )               MFMA=true; shift ;;
    -r | --rk )                 RK=true; shift ;;
    -s | --disable-strides )    DISABLE_STRIDES=true; shift ;;
    -i | --initialization )     INITIALIZATION=${2}; shift 2;;
    --problem-definition )      PROBLEM_DEFINITION=${2}; shift 2;;
    --client)                   TENSILE_CLIENT=${2}; shift 2;;
    -f | --sclk )               SCLK=${2}; shift 2;;
    -c | --count )              COUNT=true; shift ;;
    -d | --dependencies )       DEPENDENCIES=true; shift ;;
    --redo )                    REDO=true; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if ${HELP}; then
  echo "${HELP_STR}"
  exit 0
fi

if [ $# != 2 ]; then
  echo "Exactly two positional args required"
  echo "See ${0} --help"
  exit 2
fi

WORKING_PATH=${1}
LOG_PATH=${2}

if [ -z ${TENSILE_PATH} ]; then
  TENSILE_PATH=${WORKING_PATH}/tensile/Tensile
fi

if [ -z ${ROCBLAS_PATH} ]; then
  ROCBLAS_PATH=${WORKING_PATH}/rocblas/rocBLAS
fi

if ${DEPENDENCIES}; then
  # install dependencies for Tensile and rocBLAS (Ubuntu only)
  sudo apt install -y --no-install-recommends cmake make ca-certificates git \
  pkg-config python3 python3-dev python3-matplotlib python3-pandas python3-pip \
  python3-setuptools python3-tk python3-venv python3-yaml libnuma1 llvm-6.0-dev \
  libboost-all-dev zlib1g-dev libomp-dev gfortran libpthread-stubs0-dev libmsgpack-dev \
  libmsgpackc2 wget

  # add required python dependencies
  pip3 install setuptools --upgrade && pip3 install wheel && pip3 install pyyaml msgpack

  # Install Gtest
  if [ -z "$(ls -A /usr/src/gtest/googletest-release-1.11.0)" ]; then
    sudo mkdir -p /usr/src/gtest && pushd /usr/src/gtest && \
    sudo wget https://github.com/google/googletest/archive/release-1.11.0.tar.gz  && \
    sudo tar -xvf release-1.11.0.tar.gz  && \
    pushd googletest-release-1.11.0 && \
    sudo mkdir build && pushd build || exit && sudo cmake .. && sudo make && sudo make install \
      && popd || exit && popd || exit && popd || exit
  fi

  # Install Lapack
  if [ -z "$(ls -A /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3.7.1)" ]; then
    sudo mkdir -p /usr/src/lapack && pushd /usr/src/lapack && \
    sudo wget https://github.com/Reference-LAPACK/lapack-release/archive/lapack-3.7.1.tar.gz  && \
    sudo tar -xvf lapack-3.7.1.tar.gz  && \
    pushd lapack-release-lapack-3.7.1 && \
    sudo mkdir build && pushd build && \
    sudo cmake .. -DCBLAS=ON -DLAPACKE=OFF -DBUILD_TESTING=OFF -DCMAKE_Fortran_FLAGS='-fno-optimize-sibling-calls' && \
    sudo make && sudo make install && popd || exit && popd || exit && popd || exit
  fi
fi

# try to determine data type
if [ "${DATA_TYPE}" == hgemm ] || [ "${DATA_TYPE}" == h ]; then
  DATA_TYPE=hgemm
  DVAL=4
elif [ "${DATA_TYPE}" == dgemm ] || [ "${DATA_TYPE}" == d ]; then
  DATA_TYPE=dgemm
  DVAL=1
else
  if [ "$(grep -c 'r s' "${LOG_PATH}")" -gt 0 ] || [ "$(grep -c 'r f32' "${LOG_PATH}")" -gt 0 ] || \
      [ "$(grep -c 'sgemm' "${LOG_PATH}")" -gt 0 ] || [ "$(grep -c 'a_type f32' "${LOG_PATH}")" -gt 0 ] || \
      [ "$(grep -c '"a_type: "f32_r"' "${LOG_PATH}")" -gt 0 ]; then
    printf "sgemm detected\n"
    DATA_TYPE=sgemm
    DVAL=2
  elif [ "$(grep -c 'r d' "${LOG_PATH}")" -gt 0 ] || [ "$(grep -c 'r f64' "${LOG_PATH}")" -gt 0 ] || \
      [ "$(grep -c 'dgemm' "${LOG_PATH}")" -gt 0 ]; then
    printf "dgemm detected\n"
    DATA_TYPE=dgemm
    DVAL=1
  elif [ "$(grep -c 'r h' "${LOG_PATH}")" -gt 0 ] || [ "$(grep -c 'r f16' "${LOG_PATH}")" -gt 0 ] || \
      [ "$(grep -c 'hgemm' "${LOG_PATH}")" -gt 0 ] || [ "$(grep -c 'a_type f16' "${LOG_PATH}")" -gt 0 ] || \
      [ "$(grep -c 'a_type: "f16_r"' "${LOG_PATH}")" -gt 0 ]; then
    printf "hgemm detected\n"
    DATA_TYPE=hgemm
    DVAL=4
    if [ "$(grep -c 'compute_type f16_r' "${LOG_PATH}")" -gt 0 ] || \
        [ "$(grep -c 'compute_type: "f16_r"' "${LOG_PATH}")" -gt 0 ]; then
      printf "compute_type of f16 detected, disabling HighPrecisionAccumulate\n"
    fi
  else
    printf "Could not detect data type in log file, assuming sgemm\n"
    DATA_TYPE=sgemm
    DVAL=2
  fi
fi

# try to determine GPU
if [ -z ${LIBRARY+x} ]; then
  rocm_agent_enumerator 2>&1 | tee rae.txt
  rocminfo 2>&1 | tee rocminfo.txt
  if [[ "$(grep -c 'gfx900' rae.txt)" -gt 0 ]]; then
    LIBRARY=vega10
    if [[ "$(grep -c 'Compute Unit:            56' rocminfo.txt)" -gt 0 ]]; then
      printf "v340 GPU detected\n"
    else
      printf "mi25 GPU detected\n"
    fi
  elif [[ "$(grep -c 'gfx906' rae.txt)" -gt 0 ]]; then
    LIBRARY=vega20
    if [[ "$(grep -c 'Compute Unit:            60' rocminfo.txt)" -gt 0 ]]; then
      printf "mi50 GPU detected\n"
    else
      printf "mi60 GPU detected\n"
    fi
  elif [[ "$(grep -c 'gfx908' rae.txt)" -gt 0 ]]; then
    printf "arcturus GPU detected\n"
    LIBRARY=arcturus

    # currently, only sgemm mfma kernels are supported in automation
    if [[ "${DATA_TYPE}" == sgemm ]]; then
      MFMA=true
    fi
  else
    printf "Could not detect GPU, assuming vega20\n"
    LIBRARY=vega20
  fi
  rm -rf rae.txt rocminfo.txt
fi

# try to determine clock rate
if [ -z ${SCLK+x} ]; then
  rocm-smi 2>&1 | tee rocmsmi.txt
  SCLK=$(<rocmsmi.txt awk 'FNR == 6 {print $4}' | cut -d M -f 1)
  echo "SCLK: ${SCLK}"
  rm -rf rocmsmi.txt
fi

# run the four individual tuning scripts (and Tensile) in order
SCRIPT_PATH=$(realpath "${0}" | xargs dirname)

# provision_tuning.sh
if [ ! -d "${WORKING_PATH}/library" ] || ${REDO}; then
  PROVISION_TUNNING=${SCRIPT_PATH}/provision_tuning.sh
  echo "Running ${PROVISION_TUNING}"

  TUNING_ARGS=("${WORKING_PATH}" "${LOG_PATH}" foobar.yaml "${LIBRARY}" -p "${TENSILE_PATH}" --problem-definition \
    "${PROBLEM_DEFINITION}" --initialization "${INITIALIZATION}" --client "${TENSILE_CLIENT}")

  if [ -n "${NETWORK}" ]; then
    TUNING_ARGS+=(-n "${NETWORK}")
  fi

  if ${TILE_AWARE}; then
    TUNING_ARGS+=(-a)
  fi

  if ${MFMA}; then
    TUNING_ARGS+=(-m)
  fi

  if ${RK}; then
    TUNING_ARGS+=(-r)
  fi

  if ${DISABLE_STRIDES}; then
    TUNING_ARGS+=(-s)
  fi

  "${PROVISION_TUNNING}" "${TUNING_ARGS[@]}"
else
  echo "make directory non-empty. Assuming provision tuning step done previously"
  echo "Use --redo to force redoing previously done steps"
fi

# ./doit-all.sh (run Tensile for generated configs)
# TODO: this wont run again properly even with --redo
echo "Performing tuning with Tensile"
pushd "${WORKING_PATH}/make" > /dev/null || exit
./doit-all.sh
popd > /dev/null || exit

# provision_verification.sh
if [ ! -d "${WORKING_PATH}/library" ] || ${REDO}; then
  PROVISION_VERIFICATION=${SCRIPT_PATH}/provision_verification.sh
  echo "Running ${PROVISION_VERIFICATION}"

  VERIFICATION_ARGS=("${WORKING_PATH}" "${TENSILE_PATH}" "${LIBRARY}" -p "${ROCBLAS_PATH}" --redo)

  if [ "${LIBRARY}" == arcturus ]; then
    if [ "${DATA_TYPE}" == hgemm ]; then
      VERIFICATION_ARGS+=(--no-massage)
    fi
  fi

  if [ "${LIBRARY}" == arcturus ]; then
    if [ -z "$(ls -A logs/log-efficiency)" ]; then # && ! ${PUBLIC}; then
      VERIFICATION_ARGS+=(--sclk "${SCLK}")
    fi
  fi

  "${PROVISION_VERIFICATION}" "${VERIFICATION_ARGS[@]}"
else
  echo "library directory non-empty. Assuming provision verification step done previously"
  echo "Use --redo to force redoing previously done steps"
fi

# run_validation.sh
if [ ! -d "${WORKING_PATH}/benchmarks" ] || ${REDO}; then
  RUN_VALIDATION=${SCRIPT_PATH}/run_validation.sh
  echo "Running ${RUN_VALIDATION}"
  "${RUN_VALIDATION}" "${WORKING_PATH}" "${ROCBLAS_PATH}"
else
  echo "benchmarks directory non-empty. Assuming run validation step done previously"
  echo "Use --redo to force redoing previously done steps"
fi

# analyze_results.sh
if [ ! -d "${WORKING_PATH}/analysis" ] || ${REDO}; then
  ANALYZE_RESULTS=${SCRIPT_PATH}/analyze_results.sh
  echo "Running ${ANALYZE_RESULTS}"

  ANALYZE_ARGS=("${WORKING_PATH}" "${LOG_PATH}" "${LIBRARY}" -f "${SCLK}" -s "${DVAL}")

  if ${COUNT}; then
    ANALYZE_ARGS+=(-c)
  fi

  if ${MFMA}; then
    ANALYZE_ARGS+=(-m)
  fi

  "${ANALYZE_RESULTS}" "${ANALYZE_ARGS[@]}"
else
  echo "analysis directory non-empty. Assuming analyze results step done previously"
  echo "Use --redo to force redoing previously done steps"
fi
