#!/bin/bash

./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 356305 -n 1 -k 128 --alpha 1.0 --lda 128 --ldb 128 --beta 0.0 --ldc 356305 -i 1 --initialization random_int
./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 64 -n 356305 -k 128 --alpha 1.0 --lda 128 --ldb 128 --beta 0.0 --ldc 64 -i 1 --initialization random_int
./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 128 -n 356305 -k 256 --alpha 1.0 --lda 256 --ldb 256 --beta 0.0 --ldc 128 -i 1 --initialization random_int
./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 256 -n 356305 -k 256 --alpha 1.0 --lda 256 --ldb 256 --beta 0.0 --ldc 256 -i 1 --initialization random_int
./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 64 -n 1048576 -k 128 --alpha 1.0 --lda 128 --ldb 128 --beta 0.0 --ldc 64 -i 107 --initialization random_int
./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 128 -n 1048576 -k 256 --alpha 1.0 --lda 256 --ldb 256 --beta 0.0 --ldc 128 -i 107 --initialization random_int
./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 1048576 -n 1 -k 128 --alpha 1.0 --lda 128 --ldb 128 --beta 0.0 --ldc 1048576 -i 107 --initialization random_int
./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 256 -n 1048576 -k 256 --alpha 1.0 --lda 256 --ldb 256 --beta 0.0 --ldc 256 -i 107 --initialization random_int

./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 128 -n 1048576 -k 64 --alpha 1.0 --lda 128 --ldb 64 --beta 0.0 --ldc 128 -i 94 --initialization random_int
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 256 -n 1048576 -k 256 --alpha 1.0 --lda 256 --ldb 256 --beta 0.0 --ldc 256 -i 94 --initialization random_int
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 128 -n 1048576 -k 1 --alpha 1.0 --lda 128 --ldb 1 --beta 0.0 --ldc 128 -i 94 --initialization random_int
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 128 -n 1 -k 1048576 --alpha 1.0 --lda 128 --ldb 1048576 --beta 0.0 --ldc 128 -i 94 --initialization random_int
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 256 -n 1048576 -k 128 --alpha 1.0 --lda 256 --ldb 128 --beta 0.0 --ldc 256 -i 94 --initialization random_int

./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 128 -n 64 -k 1048576 --alpha 1.0 --lda 128 --ldb 64 --beta 0.0 --ldc 128 -i 94 --initialization random_int
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 256 -n 256 -k 1048576 --alpha 1.0 --lda 256 --ldb 256 --beta 0.0 --ldc 256 -i 94 --initialization random_int
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 256 -n 128 -k 1048576 --alpha 1.0 --lda 256 --ldb 128 --beta 0.0 --ldc 256 -i 94 --initialization random_int
