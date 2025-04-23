#!/bin/bash

#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#####################################################################################

set -e

WORK_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_PATH=$(dirname $(dirname $(dirname $(readlink -f "$0"))))/test_runner.py
TESTER_SCRIPT="${TESTER:-$SCRIPT_PATH}"
ATOL="${ATOL:-0.001}"
RTOL="${RTOL:-0.001}"
TARGET="${TARGET:-gpu}"

if [[ "${DEBUG:-0}" -eq 1 ]]; then
    PIPE=/dev/stdout
else
    PIPE=/dev/null
fi

if [[ "${VERBOSE:-0}" -eq 1 ]]; then
    set -x
fi

# Iterate through input recursively, process any onnx file
function iterate() {
  local dir="$1"

  for file in "$dir"/*; do
    if [ -f "$file" ]; then
      if [[ $file = *.onnx ]]; then
        process "$file"
      fi
    fi

    if [ -d "$file" ]; then
      iterate "$file"
    fi
  done
}

# Test it with test_runner.py, both fp32 and fp16
function process() {
    local file="$1"
    echo "INFO: process $file started"
    test $file fp32
    test $file fp16
    echo "INFO: process $file finished"
}

# Run test_runner.py and log if something goes wrong
function test() {
    local file="$1"
    echo "INFO: test $file ($2)"
    model_folder="$(dirname $file)"
    model_name="$(basename $model_folder)"
    # avoid name conflict for non-unique submodels
    if [[ "$model_folder" =~ "diffusion" ]]; then
      model_name="$(basename $(dirname $model_folder))_${model_name}"
    fi
    flag="--atol $ATOL --rtol $RTOL --target $TARGET"
    if [[ "$2" = "fp16" ]]; then
        flag="$flag --fp16"
    fi
    EXIT_CODE=0
    python3 $TESTER_SCRIPT ${flag} ${model_folder}/ &> "$WORK_DIR/logs/$2/${model_name//\//_}.log" || EXIT_CODE=$?
    if [[ "${EXIT_CODE:-0}" -ne 0 ]]; then
        echo "WARNING: ${file} failed ($2)"
    fi
}

mkdir -p $WORK_DIR/logs/fp32/ $WORK_DIR/logs/fp16/

for arg in "$@"; do
    iterate "$(dirname $(readlink -e $arg))/$(basename $arg)"
done
