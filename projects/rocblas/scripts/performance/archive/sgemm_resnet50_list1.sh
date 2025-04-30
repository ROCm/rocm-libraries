#!/bin/bash
bench=./rocblas-bench
if [ ! -f ${bench} ]; then
        echo ${bench} not found, exit...
        exit 1
else
        echo ">>" $(realpath $(ldd ${bench} | grep rocblas | awk '{print $3;}'))
fi
${bench} -f gemm -r s --transposeA N --transposeB N -m 12544 -n 1024 -k 256 --alpha 1 --lda 12544 --ldb 256 --beta 0 --ldc 12544
${bench} -f gemm -r s --transposeA N --transposeB N -m 12544 -n 256 -k 1024 --alpha 1 --lda 12544 --ldb 1024 --beta 0 --ldc 12544
${bench} -f gemm -r s --transposeA N --transposeB N -m 3136 -n 2048 -k 512 --alpha 1 --lda 3136 --ldb 512 --beta 0 --ldc 3136
${bench} -f gemm -r s --transposeA N --transposeB N -m 3136 -n 512 -k 2048 --alpha 1 --lda 3136 --ldb 2048 --beta 0 --ldc 3136
${bench} -f gemm -r s --transposeA T --transposeB N -m 1024 -n 256 -k 196 --alpha 1 --lda 196 --ldb 196 --beta 1 --ldc 1024
${bench} -f gemm -r s --transposeA T --transposeB N -m 128 -n 512 -k 784 --alpha 1 --lda 784 --ldb 784 --beta 1 --ldc 128
${bench} -f gemm -r s --transposeA T --transposeB N -m 2048 -n 512 -k 49 --alpha 1 --lda 49 --ldb 49 --beta 1 --ldc 2048
${bench} -f gemm -r s --transposeA T --transposeB N -m 256 -n 1024 -k 196 --alpha 1 --lda 196 --ldb 196 --beta 1 --ldc 256
${bench} -f gemm -r s --transposeA T --transposeB N -m 256 -n 64 -k 3136 --alpha 1 --lda 3136 --ldb 3136 --beta 1 --ldc 256
${bench} -f gemm -r s --transposeA T --transposeB N -m 512 -n 128 -k 784 --alpha 1 --lda 784 --ldb 784 --beta 1 --ldc 512
${bench} -f gemm -r s --transposeA T --transposeB N -m 512 -n 2048 -k 49 --alpha 1 --lda 49 --ldb 49 --beta 1 --ldc 512
${bench} -f gemm -r s --transposeA T --transposeB N -m 64 -n 256 -k 3136 --alpha 1 --lda 3136 --ldb 3136 --beta 1 --ldc 64
${bench} -f gemm -r s --transposeA T --transposeB N -m 64 -n 64 -k 3136 --alpha 1 --lda 3136 --ldb 3136 --beta 1 --ldc 64
