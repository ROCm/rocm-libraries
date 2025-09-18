#!/bin/bash
bench=./rocblas-bench
if [ ! -f ${bench} ]; then
	echo ${bench} not found, exit...
	exit 1
else
	echo ">>" $(realpath $(ldd ${bench} | grep rocblas | awk '{print $3;}'))
fi

${bench} -f gemm -r s --transposeA N --transposeB N -m 500 -n 200 -k 1000 --alpha 1 --lda 500 --ldb 1000 --beta 0 --ldc 500
${bench} -f gemm -r s --transposeA N --transposeB N -m 100 -n 200 -k 500 --alpha 1 --lda 100 --ldb 500 --beta 0 --ldc 100
${bench} -f gemm -r s --transposeA N --transposeB N -m 50 -n 200 -k 100 --alpha 1 --lda 50 --ldb 100 --beta 0 --ldc 50
${bench} -f gemm -r s --transposeA T --transposeB N -m 50 -n 200 -k 1 --alpha 1 --lda 1 --ldb 1 --beta 0 --ldc 50
${bench} -f gemm -r s --transposeA N --transposeB T -m 50 -n 100 -k 200 --alpha 1 --lda 50 --ldb 100 --beta 0 --ldc 50
${bench} -f gemm -r s --transposeA T --transposeB N -m 100 -n 200 -k 50 --alpha 1 --lda 50 --ldb 50 --beta 0 --ldc 100
${bench} -f gemm -r s --transposeA T --transposeB N -m 500 -n 200 -k 100 --alpha 1 --lda 100 --ldb 100 --beta 0 --ldc 500
${bench} -f gemm -r s --transposeA N --transposeB T -m 100 -n 500 -k 200 --alpha 1 --lda 100 --ldb 500 --beta 0 --ldc 100
${bench} -f gemm -r s --transposeA T --transposeB N -m 1000 -n 200 -k 500 --alpha 1 --lda 500 --ldb 500 --beta 0 --ldc 1000
${bench} -f gemm -r s --transposeA N --transposeB T -m 500 -n 1000 -k 200 --alpha 1 --lda 500 --ldb 1000 --beta 0 --ldc 500
${bench} -f gemm -r s --transposeA N --transposeB T -m 1000 -n 29532 -k 200 --alpha 1 --lda 1000 --ldb 29532 --beta 0 --ldc 1000
${bench} -f gemm -r s --transposeA N --transposeB N -m 1000 -n 200 -k 29532 --alpha 1 --lda 1000 --ldb 29532 --beta 0 --ldc 1000

${bench} -f gemv -r s --transposeA T -m 50 -n 200 --alpha 1 --lda 50 --incx 1 --beta 0 --incy 1
${bench} -f gemv -r s --transposeA N -m 50 -n 200 --alpha 1 --lda 50 --incx 1 --beta 0 --incy 1
