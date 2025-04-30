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
    for k in {512..3200..256}; do
      ${bench} -f gemm -r s --transposeA T --transposeB N -m $m -n $n -k $k --alpha 1 --lda $k --ldb $k --beta 1 --ldc $m
    done
    ${bench} -f gemm -r s --transposeA T --transposeB N -m $m -n $n -k 3200 --alpha 1 --lda 3200 --ldb 3200 --beta 1 --ldc $m
  done
done

for m in 32 64 128 256; do
  for n in 200 256 512 1024; do
    for k in 512 1024 2048; do
      ${bench} -f gemm -r s --transposeA T --transposeB N -m $m -n $n -k $k --alpha 1 --lda $k --ldb $k --beta 1 --ldc $m
    done
  done
done

for n in 200 256 512 1024; do
  for k in 1024 2048 4096; do
    ${bench} -f gemm -r s --transposeA T --transposeB N -m 1 -n $n -k $k --alpha 1 --lda $k --ldb $k --beta 1 --ldc 1
  done
done


for m in 1024 2048 4096; do
  for n in 200 256 512 1024; do
    for ki in 32 64 128 256; do
      for f in {4..64..4}; do
        k=$((f*ki))
        ${bench} -f gemm -r s --transposeA T --transposeB N -m $m -n $n -k $k --alpha 1 --lda $k --ldb $k --beta 1 --ldc $m
      done
      k=${ki}
      ${bench} -f gemm -r s --transposeA T --transposeB N -m $m -n $n -k $k --alpha 1 --lda $k --ldb $k --beta 1 --ldc $m
      k=$((ki*65))
      ${bench} -f gemm -r s --transposeA T --transposeB N -m $m -n $n -k $k --alpha 1 --lda $k --ldb $k --beta 1 --ldc $m
    done
  done
done



