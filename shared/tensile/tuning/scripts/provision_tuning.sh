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

function provision_tensile() {

  local EXE="${SCRIPT_ROOT}/provision_repo.sh"
  local ARGS=(-w "${WORKING_PATH}/tensile" -b "${TENSILE_BRANCH}" -f "${TENSILE_FORK}")

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
TENSILE_CLIENT=new
TENSILE_FORK='ROCmSoftwarePlatform'
TENSILE_BRANCH='develop'
TILE_AWARE=false
MFMA=false
RK=false
DISABLE_STRIDES=false
PROBLEM_DEFINITION=both
INITIALIZATION=rand_int
DISABLE_HPA=false

HELP_STR="
Usage: ${0} WORKING_PATH LOG_PATH OUTPUT_SUFFIX.yaml LIBRARY [options]

  where LIBRARY = {arcturus | vega20 | vega10 | mi25 | r9nano | hip}

Options:
-h | --help                   Display this help message
-n | --network NAME           Neural network name. If this is set, LOG_PATH should be a directory. \
Will only tune log files with this string in the file name
-p | --tensile-path PATH      Path to existing Tensile (will not provision new copy)

Options for provisioning Tensile:
-f | --tensile-fork USERNAME  Tensile fork to use
-b | --branch BRANCH          Tensile branch to use
-c | --commit COMMIT_ID       Tensile commit to use
-t | --tag GITHUB_TAG         Tensile tag to use
-i | --id ID                  ID to append to Tensile directory name

Options for config generation:
-t | --tile-aware             Use tile-aware method. (limited support)
-m | --mfma                   Use MFMA kernels
-r | --rk                     Use replacement kernels (sgemm only)
-s | --disable-strides        Disable leading dimensions and strides in tuning file

--initialization {rand_int | trig_float | hpl} (=rand_int)  Data initialization for matrices
--problem-definition {gemm | batch | both} (=both)          Which problem types to tune
--client {new | old | both} (=new)                          Which Tensile runtime client to use
"

if ! OPTS=$(getopt -o h,n:,p:,f:,b:,c:,t:,i:,a,m,r,s \
--long help,network:,tensile-path:,tensile-fork:,branch:,commit:,tag:,id:,\
tile-aware,mfma,rk,disable-strides,initialization:,problem-definition:,client: \
 -n "${0}" -- "$@")
then
  echo "Failed parsing options"
  exit 1
fi

eval set -- "${OPTS}"

while true; do
  case ${1} in
    -h | --help )            HELP=true; shift ;;
    -n | --network )         NETWORK=${2}; shift 2;;
    -p | --tensile-path )    TENSILE_PATH=${2}; shift 2;;
    -f | --tensile-fork )    TENSILE_FORK=${2}; shift 2;;
    -b | --branch  )         TENSILE_BRANCH=${2}; shift 2;;
    -c | --commit )          COMMIT=${2}; shift 2;;
    -t | --tag )             TAG=${2}; shift 2;;
    -i | --id )              ID=${2}; shift 2;;
    -a | --tile-aware )      TILE_AWARE=true; shift;;
    -m | --mfma )            MFMA=true; shift;;
    -r | --rk )              RK=true; shift;;
    -s | --disable-strides ) DISABLE_STRIDES=true; shift;;
    --initialization )       INITIALIZATION=${2}; shift 2;;
    --problem-definition )   PROBLEM_DEFINITION=${2}; shift 2;;
    --client )               TENSILE_CLIENT=${2}; shift 2;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if ${HELP}; then
  echo "${HELP_STR}"
  exit 0
fi

if [ $# != 4 ]; then
  echo "Exactly four positional args required"
  echo "See ${0} --help"
  exit 2
fi

WORKING_PATH=${1}
LOG_PATH=${2}
OUTPUT_SUFFIX=${3} # TODO: this is not really used in the rest of the tuning pipeline
LIBRARY=${4}

# TODO: test options have valid values

# determine full path of tools root
TOOLS_ROOT=$(realpath "${0}" | xargs dirname | xargs dirname)

BUILD_ROOT=${WORKING_PATH}/configs
STAGE_ROOT=${WORKING_PATH}/make
OUT_SCRIPT_ROOT=${WORKING_PATH}/scripts
OUT_SCRIPT2_ROOT=${WORKING_PATH}/scripts2
AUTOMATION_ROOT=${TOOLS_ROOT}/automation
SCRIPT_ROOT=${TOOLS_ROOT}/scripts

mkdir -p "${STAGE_ROOT}"

# create configs for sizes in log file/dir
echo "Generating tuning configurations"
ARGS=("${AUTOMATION_ROOT}/GenerateTuningConfigurations.py" "${LOG_PATH}")

if [ -n "${NETWORK}" ]; then
  ARGS+=("${NETWORK}")
fi

ARGS+=("${WORKING_PATH}" "${OUTPUT_SUFFIX}" "${LIBRARY}" "${TILE_AWARE}" "${MFMA}" "${RK}" \
  "${DISABLE_STRIDES}" "${PROBLEM_DEFINITION}" "${INITIALIZATION}" "${TENSILE_CLIENT}" "${DISABLE_HPA}")
python3 "${ARGS[@]}"

# make outputed scripts executable
pushd "${OUT_SCRIPT_ROOT}" > /dev/null || exit
chmod +x -- *
popd > /dev/null || exit

pushd "${OUT_SCRIPT2_ROOT}" > /dev/null || exit
chmod +x -- *
popd > /dev/null || exit

# provision tensile if path not provided
if [ -z ${TENSILE_PATH+x} ]; then
  echo "Tensile path not provided. Trying to provision copy"
  TENSILE_PATH=$(realpath "${WORKING_PATH}")/tensile/Tensile

  if [ ! -d "${TENSILE_PATH}" ]; then
    provision_tensile
  else
    echo "Path already exists. Assuming Tensile previously provisioned"
  fi
else
  echo "Using existing Tensile path"
  TENSILE_PATH=$(realpath "${TENSILE_PATH}")
fi

echo "Preparing scripts to run tuning"
find "${BUILD_ROOT}" -name '*.yaml' -print0 \
  | xargs -0 -n1 basename \
  | xargs "${SCRIPT_ROOT}/stage_tuning.sh" "${BUILD_ROOT}" "${STAGE_ROOT}" "${TENSILE_PATH}"
