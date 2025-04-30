#!/bin/bash

bench=./rocblas-bench
if [ ! -f ${bench} ]; then
        echo ${bench} not found, exit...
        exit 1
else
        echo ">>" $(realpath $(ldd ${bench} | grep rocblas | awk '{print $3;}'))
fi


for m in 512 1024 2048; do
  for n in 200 256 512 1024; do
    for k in 512 1024 2048; do
      ${bench} -f gemm -r s --transposeA N --transposeB N -m $m -n $n -k $k --alpha 1 --lda $m --ldb $k --beta 1 --ldc $m
    done
  done
done

for m in 1024 2048 4096; do
  for n in 200 256 512 1024; do
    for k in 1024 2048 4096; do
      ${bench} -f gemm -r s --transposeA N --transposeB N -m $m -n $n -k $k --alpha 1 --lda $m --ldb $k --beta 1 --ldc $m
    done
  done
done


