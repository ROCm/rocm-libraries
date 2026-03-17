#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# TODO: run this script from CK root or build directory
set -euo pipefail

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
EXE_NAME=tile_example_fmha_fwd
EXE="$(find . -name $EXE_NAME -type f | head -n 1)"
KNAME=1
GPU_arch=${GPU_arch:-""}
if [ -z "$GPU_arch" ] ; then
    GPU_arch=$(rocminfo | grep -E 'Name:\s+gfx' | head -n1 | awk '{print $2}')
fi

export CK_WARMUP=0
export CK_REPEAT=1

CURR_FAILS_FILE=${CURR_FAILS_FILE:-"fmha_fwd_sink_fails_$GPU_arch.txt"}
rm -f $CURR_FAILS_FILE
touch $CURR_FAILS_FILE
KNOWN_FAILS_FILE=${KNOWN_FAILS_FILE:-"$SCRIPT_DIR/fmha_fwd_sink_known_fails_$GPU_arch.txt"}

COMMON_ARGS='-v=1 -warmup=0 -repeat=1'

run_exe() {
    set +ex
    $EXE $@
    local ret=$?
    if [ $ret -ne 0 ] ; then
        echo "$EXE_NAME $*" >> $CURR_FAILS_FILE
    fi
    set -ex
}

# Sink-specific mask pattern tests (sliding window + sink token).
# Each case corresponds to a specific attention mask layout documented below.
run_sink_mask_tests() {
    # window_size[2,0], sink_size=2  (top-left causal + sink)
    #    before:              after:
    #    1 * * * * * * *      1 * * * * * * *
    #    1 1 * * * * * *      1 1 * * * * * *
    #    1 1 1 * * * * *      1 1 1 * * * * *
    #    * 1 1 1 * * * *      1 1 1 1 * * * *
    #    * * 1 1 1 * * *      1 1 1 1 1 * * *
    #    * * * 1 1 1 * *      1 1 * 1 1 1 * *
    #    * * * * 1 1 1 *      1 1 * * 1 1 1 *
    #    * * * * * 1 1 1      1 1 * * * 1 1 1
    run_exe -prec=fp16 -mode=0 -b=1 -h=1 -d=128 -d_v=128 -s=512   -s_k=512   -bias=n -lse=0 -iperm=0 -operm=0 -vlayout=r -num_splits=1 -page_block_size=128 -cache_batch_idx=0 -kname=$KNAME $COMMON_ARGS -mask=t:2,0,2
    run_exe -prec=bf16 -mode=0 -b=2 -h=2 -d=128 -d_v=128 -s=512   -s_k=512   -bias=n -lse=0 -iperm=1 -operm=1 -vlayout=r -num_splits=1 -page_block_size=0   -cache_batch_idx=0 -kname=$KNAME $COMMON_ARGS -mask=t:2,0,2

    # window_size[0,3], sink_size=2  (top-left + sink)
    #    before:              after:
    #    1 1 1 1 * * * *      1 1 1 1 * * * *
    #    * 1 1 1 1 * * *      1 1 1 1 1 * * *
    #    * * 1 1 1 1 * *      1 1 1 1 1 1 * *
    #    * * * 1 1 1 1 *      1 1 * 1 1 1 1 *
    #    * * * * 1 1 1 1      1 1 * * 1 1 1 1
    run_exe -prec=fp16 -mode=0 -b=1 -h=1 -d=128 -d_v=128 -s=1024  -s_k=1024  -bias=n -lse=0 -iperm=0 -operm=0 -vlayout=r -num_splits=1 -page_block_size=128 -cache_batch_idx=0 -kname=$KNAME $COMMON_ARGS -mask=t:0,3,2
    run_exe -prec=bf16 -mode=1 -b=2 -h=2 -d=128 -d_v=128 -s=1024  -s_k=1024  -bias=n -lse=0 -iperm=1 -operm=1 -vlayout=r -num_splits=1 -page_block_size=0   -cache_batch_idx=0 -kname=$KNAME $COMMON_ARGS -mask=t:0,3,2

    # window_size[1,0], sink_size=2  (bottom-right + sink)
    #    before:              after:
    #    * * 1 1 * * * *      1 1 1 1 * * * *
    #    * * * 1 1 * * *      1 1 * 1 1 * * *
    #    * * * * 1 1 * *      1 1 * * 1 1 * *
    #    * * * * * 1 1 *      1 1 * * * 1 1 *
    #    * * * * * * 1 1      1 1 * * * * 1 1
    run_exe -prec=fp16 -mode=0 -b=1 -h=1 -d=128 -d_v=128 -s=4096  -s_k=4096  -bias=n -lse=0 -iperm=0 -operm=0 -vlayout=r -num_splits=1 -page_block_size=128 -cache_batch_idx=0 -kname=$KNAME $COMMON_ARGS -mask=b:1,0,2
    run_exe -prec=bf16 -mode=0 -b=2 -h=4 -d=64  -d_v=64  -s=2048  -s_k=2048  -bias=n -lse=0 -iperm=0 -operm=0 -vlayout=r -num_splits=1 -page_block_size=0   -cache_batch_idx=0 -kname=$KNAME $COMMON_ARGS -mask=b:1,0,2

    # window_size[2,0], sink_size=2  (bottom-right, group mode + sink)
    #    before:              after:
    #    1 * * * * *          1 * * * * *
    #    1 1 * * * *          1 1 * * * *
    #    1 1 1 * * *   -->    1 1 1 * * *
    #    * 1 1 1 * *          1 1 1 1 * *
    #    * * 1 1 1 *          1 1 1 1 1 *
    #    * * * 1 1 1          1 1 * 1 1 1
    run_exe -prec=fp16 -mode=1 -b=1 -h=1 -d=128 -d_v=128 -s=8192  -s_k=8192  -bias=n -lse=0 -iperm=0 -operm=0 -vlayout=r -num_splits=1 -page_block_size=128 -cache_batch_idx=0 -kname=$KNAME $COMMON_ARGS -mask=b:2,0,2
    run_exe -prec=bf16 -mode=1 -b=2 -h=2 -d=128 -d_v=128 -s=4096  -s_k=4096  -bias=n -lse=0 -iperm=1 -operm=1 -vlayout=r -num_splits=1 -page_block_size=0   -cache_batch_idx=0 -kname=$KNAME $COMMON_ARGS -mask=b:2,0,2

    # window_size[-1,1], sink_size=2  (bottom-right, large seqlen + sink)
    run_exe -prec=fp16 -mode=1 -b=1 -h=1 -d=128 -d_v=128 -s=16384 -s_k=16384 -bias=n -lse=0 -iperm=0 -operm=0 -vlayout=r -num_splits=1 -page_block_size=128 -cache_batch_idx=0 -kname=$KNAME $COMMON_ARGS -mask=b:-1,1,2
    run_exe -prec=bf16 -mode=1 -b=1 -h=2 -d=128 -d_v=128 -s=8192  -s_k=8192  -bias=n -lse=0 -iperm=0 -operm=0 -vlayout=r -num_splits=1 -page_block_size=0   -cache_batch_idx=0 -kname=$KNAME $COMMON_ARGS -mask=b:-1,1,2
}

# init_sink tests: validate sink token initialization path across
# different seqlens, modes, hdims and precisions.
run_sink_init_tests() {
    for prec in "fp16" "bf16" ; do
    for hdim in 64 128 256 ; do
    for mode in 0 1 ; do
    for mask in 0 1 ; do
        run_exe -prec=$prec -mode=$mode -b=1 -h=2 -d=$hdim -d_v=$hdim  -s=512   -s_k=512   -bias=n -lse=0 -iperm=0 -operm=0 -vlayout=r -kname=$KNAME $COMMON_ARGS -init_sink=1 -mask=$mask
        run_exe -prec=$prec -mode=$mode -b=2 -h=4 -d=$hdim -d_v=$hdim  -s=1024  -s_k=1024  -bias=n -lse=0 -iperm=1 -operm=1 -vlayout=r -kname=$KNAME $COMMON_ARGS -init_sink=1 -mask=$mask
        run_exe -prec=$prec -mode=$mode -b=1 -h=2 -d=$hdim -d_v=$hdim  -s=4096  -s_k=4096  -bias=n -lse=0 -iperm=0 -operm=0 -vlayout=r -page_block_size=128 -cache_batch_idx=0 -kname=$KNAME $COMMON_ARGS -init_sink=1
    done
    done
    done
    done
}

set -x
run_sink_mask_tests
run_sink_init_tests
set +x

new_fails_count=0
known_fails_count=0
if [ -f $KNOWN_FAILS_FILE ] ; then
    echo "Comparing current fails ($CURR_FAILS_FILE) against known fails ($KNOWN_FAILS_FILE):"
    while IFS= read -r line; do
        if grep -Fxq "$line" $KNOWN_FAILS_FILE; then
            echo "Known fail: $line"
            known_fails_count=$(($known_fails_count + 1))
        else
            echo "New fail: $line"
            new_fails_count=$(($new_fails_count + 1))
        fi
    done < $CURR_FAILS_FILE
else
    new_fails_count=$(wc -l < $CURR_FAILS_FILE)
    echo "No known fails file, all fails ($new_fails_count) are new:"
    cat $CURR_FAILS_FILE
fi
echo "New fails count: $new_fails_count; Known fails count: $known_fails_count"
exit $(($new_fails_count != 0))
