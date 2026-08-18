#!/bin/bash

# Multi-kernel path variant of bench_latrd.sh.
# Forces the LATRD multi-kernel path (separate kernel launches per step)
# instead of the default fused cooperative kernel.
export LATRD_MULTI_KERNEL=1

bench=$1
func=${2:-latrd --uplo L}
k=${3:-64}
prec=${4:-s}
dev=${5:-0}

if [[ -z "$dev" ]]; then
    dev=0
fi

verify=""
if [[ ! -z ${VERIFY} ]]; then
    verify="--verify 1"
fi

## Small & medium

i=128
while [ $i -le 256 ]
do
    echo -n -e "$i\t"; $bench -f $func -n $i -k $k --perf 1 --iters 51 -r $prec $verify --device $dev
    i=$((i+32))
done

i=320
while [ $i -le 2048 ]
do
    echo -n -e "$i\t"; $bench -f $func -n $i -k $k --perf 1 --iters 31 -r $prec $verify --device $dev
    i=$((i+64))
done

### Large

i=2176
while [ $i -le 4096 ]
do
    echo -n -e "$i\t"; $bench -f $func -n $i -k $k --perf 1 --iters 21 -r $prec $verify --device $dev
    i=$((i+128))
done

i=4352
while [ $i -le 8192 ]
do
    echo -n -e "$i\t"; $bench -f $func -n $i -k $k --perf 1 --iters 11 -r $prec $verify --device $dev
    i=$((i+256))
done

# i=8704
# while [ $i -le 12288 ]
# do
#     echo -n -e "$i\t"; $bench -f $func -n $i -k $k --perf 1 --iters 11 -r $prec $verify --device $dev
#     i=$((i+512))
# done
