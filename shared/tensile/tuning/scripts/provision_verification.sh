#!/bin/bash

################################################################################
#
# Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

function provision_rocblas() {

  local EXE="${SCRIPT_ROOT}/provision_repo.sh"
  local ARGS=(-r -w "${WORKING_PATH}/rocblas" -b "${ROCBLAS_BRANCH}" -f "${ROCBLAS_FORK}")

  if [ -n "${ID}" ]; then
    ARGS+=(-i "${ID}")
  fi

  if [ -n "${TAG}" ]; then
    ARGS+=(-t "${TAG}")
  fi

  if [ -n "${COMMIT}" ]; then
    ARGS+=(-c "${COMMIT}")
  fi

  ${EXE} "${ARGS[@]}"
}

HELP=false
ROCBLAS_BRANCH='develop'
ROCBLAS_FORK='RocmSoftwarePlatform'
MASSAGE=true
MERGE=true
REDO=false

HELP_STR="
Usage: ${0} WORKING_PATH TENSILE_PATH LIBRARY [options]

  where LIBRARY = {arcturus | vega20 | vega10 | mi25 | r9nano | hip}

Options:
-h | --help                   Display this help message
-p | --rocblas-path PATH      Path to existing rocBLAS (will not provision new copy)
-f | --rocblas-fork USERNAME  rocBLAS fork to use
-b | --branch BRANCH          rocBLAS branch to use
-c | --commit COMMIT_ID       rocBLAS commit to use
-t | --tag GITHUP_TAG         rocBLAS tag to use
-i | --id ID                  ID to append to rocBLAS directory name
-s | --sclk                   Frequency of sclk in MHz
-n | --no-merge               Skip merge step
--no-massage                  Skip massage step
--log-dir                     Directory for logs
--redo                        Force logic preparation, merge, massage, and library build steps to be redone
"

if ! OPTS=$(getopt -o h,p:,f:,b:,c:,t:,i:,s:,n \
--long help,rocblas-path:,rocblas-fork:,branch:,commit:,tag:,id:,sclk:,no-merge,no-massage,redo \
-n 'parse-options' -- "$@")
then
  echo "Failed parsing options"
  exit 1
fi

eval set -- "${OPTS}"

while true; do
  case ${1} in
    -h | --help )            HELP=true; shift ;;
    -p | --rocblas-path )    ROCBLAS_PATH=${2}; shift 2;;
    -f | --rocblas-fork )    ROCBLAS_FORK=${2}; shift 2;;
    -b | --branch  )         ROCBLAS_BRANCH=${2}; shift 2;;
    -c | --commit )          COMMIT=${2}; shift 2;;
    -t | --tag )             TAG=${2}; shift 2;;
    -i | --id)               ID=${2}; shift 2;;
    -s | --sclk )            SCLK=${2}; shift 2;;
    -n | --no-merge )        MERGE=false; shift ;;
    --no-massage )           MASSAGE=false; shift ;;
    --log-dir )              LOGS=${2}; shift 2;;
    --redo )                 REDO=true; shift 1;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if ${HELP}; then
  echo "${HELP_STR}"
  exit 0
fi

if [ $# != 3 ]; then
  echo "Exactly three positional args required"
  echo "See ${0} --help"
  exit 2
fi

WORKING_PATH=${1}
TENSILE_PATH=${2}
LIBRARY=${3}

# TODO: test options have valid values

if [ -z ${LOGS+x} ]; then
  LOGS=${WORKING_PATH}/logs
fi

if [ ${MERGE} == false ]; then
   MASSAGE=false
fi

# determine full path of tools root
TOOLS_ROOT=$(realpath "${0}" | xargs dirname | xargs dirname)

SCRIPT_ROOT=${TOOLS_ROOT}/scripts
LIBRARY_ROOT=${WORKING_PATH}/library
EXACT_PATH=${LIBRARY_ROOT}/exact
MERGE_PATH=${LIBRARY_ROOT}/merge
ASM_PATH=${LIBRARY_ROOT}/asm_full
ARCHIVE_PATH=${LIBRARY_ROOT}/archive
MASSAGE_PATH=${LIBRARY_ROOT}/massage

TENSILE_LIBRARY_PATH=${LIBRARY_ROOT}/tensile_library
mkdir -p "${TENSILE_LIBRARY_PATH}"
TENSILE_LIBRARY_PATH=$(realpath "${TENSILE_LIBRARY_PATH}")

mkdir -p "${EXACT_PATH}"
mkdir -p "${ASM_PATH}"
mkdir -p "${ARCHIVE_PATH}"
mkdir -p "${LOGS}"

# provision rocblas if path not provided
if [ -z ${ROCBLAS_PATH+x} ]; then
  echo "rocBLAS path not provided. Trying to provision copy"
  ROCBLAS_PATH=$(realpath "${WORKING_PATH}")/rocblas/rocBLAS

  if [ ! -d "${ROCBLAS_PATH}" ]; then
    provision_rocblas
  else
    echo "Path already exists. Assuming rocBLAS previously provisioned"
  fi
else
  echo "Using existing rocBLAS path"
  ROCBLAS_PATH=$(realpath "${ROCBLAS_PATH}")
fi

# TODO: filter here to only get ones from "this" tuning exercise? Or should a dir only be for one exercise anyway?
LOGIC_FILE_PATHS=$(find "${WORKING_PATH}/make" -name 3_LibraryLogic | xargs -I{} printf "%s\n" {})

REFERENCE_LIBRARY_ASM=${ROCBLAS_PATH}/library/src/blas3/Tensile/Logic/asm_full
REFERENCE_LIBRARY_ARCHIVE=${ROCBLAS_PATH}/library/src/blas3/Tensile/Logic/archive

# copy library logic files
if [ -z "$(ls -A "${EXACT_PATH}")" ] || ${REDO}; then
  for PATH_NAME in $LOGIC_FILE_PATHS; do
    cp "${PATH_NAME}"/* "${EXACT_PATH}"
  done

  cp "${REFERENCE_LIBRARY_ASM}"/* "${ASM_PATH}"
  cp "${REFERENCE_LIBRARY_ARCHIVE}"/* "${ARCHIVE_PATH}"
  cp "${ARCHIVE_PATH}"/*yaml "${ASM_PATH}"
else
  echo "Logic directory non-empty. Assuming file copy done previously"
  echo "Use --redo to force redoing previously done library prep/merge/massage/build steps"
fi

# convert to efficiency
if [ "${LIBRARY}" == arcturus ]; then
  if [ "${PUBLIC}" == false ]; then
    if [ ! -f "${LOGS}/log-efficiency" ]; then
        pushd "${WORKING_PATH}"  > /dev/null || exit
        git clone https://github.com/RocmSoftwarePlatform/rocmdevtools.git -b efficiency
        python3 rocmdevtools/scripts/tuning/convertToEfficiency.py \
          "${EXACT_PATH}" "${LIBRARY}" "${SCLK}" 2>&1 | tee "${LOGS}/efficiency.log"
        popd > /dev/null || exit
    fi
  fi
fi

MERGE_SCRIPT=${TENSILE_PATH}/Tensile/Utilities/merge.py
MASSAGE_SCRIPT=${REFERENCE_LIBRARY_ARCHIVE}/massage.py

# perform merge step
if ${MERGE}; then
  mkdir -p "${MERGE_PATH}"
  mkdir -p "${MASSAGE_PATH}"

  if [ "${LIBRARY}" != arcturus ] && ${MASSAGE}; then
    ASM_PATH=${ARCHIVE_PATH}
  fi

  echo "Merging tuned logic with existing logic"
  if [ -z "$(ls -A "${MERGE_PATH}")" ] || ${REDO}; then
    python3 "${MERGE_SCRIPT}" "${ASM_PATH}" "${EXACT_PATH}" "${MERGE_PATH}" 2>&1 \
      | tee "${LOGS}/merge.log"
  else
    echo "Merge directory non-empty. Assuming merge done previously"
    echo "Use --redo to force redoing previously done library prep/merge/massage/build steps"
  fi
else
  echo "Skipping merge step"
  MERGE_PATH=${EXACT_PATH}
fi

# perform massage step
if ${MASSAGE}; then
  echo "Massaging logic"
  if [ -z "$(ls -A "${MASSAGE_PATH}")" ] || ${REDO}; then
    python3 "${MASSAGE_SCRIPT}" "${MERGE_PATH}" "${MASSAGE_PATH}" 2>&1 \
      | tee "${LOGS}/massage.log"
  else
    echo "Massage directory non-empty. Assuming massage done previously"
    echo "Use --redo to force redoing previously done library prep/merge/massage/build steps"
  fi
else
  echo "Skipping massage step"
fi

# build rocBLAS
echo "Building rocBLAS"
ROCBLAS_BENCH_PATH=${ROCBLAS_PATH}/build/release/clients/staging/rocblas-bench
if [ ! -f "${ROCBLAS_BENCH_PATH}" ]; then
  rm -r -f "${ROCBLAS_PATH}/build"
  pushd "${ROCBLAS_PATH}" > /dev/null || exit

  TENSILE_PATH=$(realpath "${TENSILE_PATH}")
  BUILD_ARGS=(-d -c -t "${TENSILE_PATH}")
  BUILD_EXE=./install.sh
  "${BUILD_EXE}" "${BUILD_ARGS[@]}" 2>&1 | tee "${LOGS}/rocblas-install.log"
  popd > /dev/null || exit
else
  echo "rocBLAS already built: skipping"
fi

# TODO: way to set which Tensile to use for create library?
# TODO: get correct Python version
CREATE_LIBRARY_EXE=${ROCBLAS_PATH}/build/release/virtualenv/lib/python3.6/site-packages/Tensile/bin/TensileCreateLibrary
CREATE_LIBRARY_ARGS=(--merge-files --no-short-file-names \
  --no-library-print-debug --cxx-compiler=hipcc \
  --library-format=msgpack "${MERGE_PATH}" "${TENSILE_LIBRARY_PATH}" HIP)

# create new (tuned) Tensile library
echo "Creating new Tensile library"
if [ -f "${CREATE_LIBRARY_EXE}" ]; then
    if [ ! -d "${TENSILE_LIBRARY_PATH}/library" ] || ${REDO}; then
        ${CREATE_LIBRARY_EXE} "${CREATE_LIBRARY_ARGS[@]}"
    else
      echo "Path already exists. Assuming library already built"
      echo "Use --redo to force redoing previously done library prep/merge/massage/build steps"
    fi
else
  echo "Error: could not find TensileCreateLibrary script"
fi
