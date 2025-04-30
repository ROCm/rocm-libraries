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

HELP_STR="usage: $0 [-w|--working-path <path>] [-z | --size-log <path>] [-o|--output <configuration filename>] [-y| --type <cofiguration type>] [-l|--library <library>] [-f] [-s] [-h|--help]"
HELP=false
SUPPRESS_TENSILE=false

OPTS=`getopt -o hw:z:f:b:o:y:l:f:s: --long help,working-path:,size-log:,tensile-fork:,branch:,output:,library:,type: -n 'parse-options' -- "$@"`

if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; exit 1 ; fi

eval set -- "$OPTS"

while true; do
  case "$1" in
    -h | --help )         HELP=true; shift ;;
    -w | --working-path ) WORKING_PATH="$2"; shift 2;;
    -z | --size-log )     SIZE_LOG="$2"; shift 2;;
    -f | --tensile-fork)  TENSILE_FORK="$2"; shift 2;;
    -b | --branch  )      TENSILE_BRANCH="$2"; shift 2;;
    -o | --output )       OUTPUT_FILE="$2"; shift 2;;
    -y | --type )         CONFIGURATION_TYPE="$2"; shift 2;;
    -l | --library )      LIBRARY="$2"; shift 2;;
    -f )                  FREQ="$2"; shift 2;;
    -s )                  SZ="$2"; shift 2;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if $HELP; then
  echo "${HELP_STR}" >&2
  exit 2
fi

if [ -z ${WORKING_PATH+foo} ]; then
   printf "A working path is required\n"
   exit 2
fi

if [ -z ${TENSILE_FORK+foo} ]; then
   TENSILE_FORK="ROCmSoftwarePlatform"
fi

if [ -z ${TENSILE_BRANCH+foo} ]; then
   TENSILE_BRANCH="develop"
fi

if [ -z ${ROCBLAS_FORK+foo} ]; then
   ROCBLAS_FORK="ROCmSoftwarePlatform"
fi

TENSILE_HOST="https://github.com/${TENSILE_FORK}/Tensile.git"

if [ -z ${SIZE_LOG+foo} ]; then
   printf "A problem specification file is required\n"
   exit 2
fi

if [ -z ${OUTPUT_FILE+foo} ]; then
   printf "Need an output path\n"
   exit 2
fi

if [ -z ${LIBRARY+foo} ]; then
   printf "Need a library type (arcturus|vega20|vega10)\n"
   exit 2
fi

if [ -z ${CONFIGURATION_TYPE+foo} ]; then
   printf "Need a configuration type (hgemm|sgemm|dgemm|igemm)\n"
   exit 2
fi

if [ -z ${FREQ+foo} ]; then
   printf "The clock rate used\n"
   exit 2
fi

if [ -z ${SZ+foo} ]; then
   printf "datatype size (d=1|s=2|h=4)\n"
   exit 2
fi


TOOLS_ROOT=`dirname "$0"`
TOOLS_ROOT=`( cd "${TOOLS_ROOT}" && cd .. && pwd )`

PROVISION_TUNING=${TOOLS_ROOT}/scripts/provision_tuning.sh
PROVISION_VERIFICATION=${TOOLS_ROOT}/scripts/provision_verification.sh
ANALYSE_RESULTS=${TOOLS_ROOT}/scripts/analyze-results.sh

${PROVISION_TUNING} -w ${WORKING_PATH} -z ${SIZE_LOG} -o ${OUTPUT_FILE} -y ${CONFIGURATION_TYPE}  -l ${LIBRARY}

TUNING_PATH=${WORKING_PATH}/tensile/Tensile

pushd ${TUNING_PATH} > /dev/null

./doit-all.sh > make.out 2>&1

popd > /dev/null

${PROVISION_VERIFICATION} -w validate -r ${TUNING_PATH}

SCRIPT_PATH=${WORKING_PATH}/scripts
ROCBLAS_ROOT=validate/rocblas
REFERENCE_PATH=${ROCBLAS_ROOT}/rocBLAS-reference
VERIFY_PATH=${ROCBLAS_ROOT}/rocBLAS-verify

BUILD_PATH=build/release/clients/staging

REFERENCE_BUILD=${REFERENCE_PATH}/${BUILD_PATH}
VERIFY_BUILD=${VERIFY_PATH}/${BUILD_PATH}

cp ${SCRIPT_PATH}/* ${REFERENCE_BUILD}
cp ${SCRIPT_PATH}/* ${VERIFY_BUILD}

REFERENCE=${REFERENCE_BUILD}/results
VERIFY=${VERIFY_BUILD}/results

mkdir -p ${REFERENCE}
mkdir -p ${VERIFY}

pushd  ${REFERENCE_BUILD} > /dev/null

./doit_all.sh > test.out

popd > /dev/null

pushd ${VERIFY_BUILD} > /dev/null

./doit_all.sh > test.out

popd > /dev/null

EXE="${ANALYSE_RESULTS} -o analysis -r ${REFERENCE} -b ${VERIFY} -f ${FREQ} -s ${SZ}"
${EXE}

