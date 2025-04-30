#!/bin/bash

################################################################################
#
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

########################################
# client parameters
M=5504
N=5504
K=5504
NUM_PRINTOUTS=256
NUM_ENQUEUES_PER_PRINTOUT=64
DEVICES=( 0 1 2 3 )
EXEC=./sgemm_gfx900

########################################
# crtl-c will kill background process and quit
PIDS=()
function kill_procs() {
  kill ${PIDS[@]} 2> /dev/null
  echo "Aborted."
}
trap kill_procs INT

########################################
# launch clients
echo "" > out.cmd.txt
for DEVICE in ${DEVICES[@]}
do
CMD="${EXEC} --device-idx $DEVICE --sizes $M $N $K --num-enqueues-per-sync $NUM_ENQUEUES_PER_PRINTOUT --num-syncs-per-benchmark 1 --num-benchmarks $NUM_PRINTOUTS --use-gpu-timer 1 --function-idx 0 --init-c 3 --init-ab 3 --num-elements-to-validate 0 --print-max 0 | tee out.${DEVICE}.txt &"
echo $CMD >> out.cmd.txt
eval $CMD
PID=$!
PIDS+=($PID)
done

########################################
# wait for background processes
wait ${PIDS[@]}
echo "Done. Data logged to out.*.txt"
