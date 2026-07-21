#!/bin/bash

#poor
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 2048 -n 512 -k 74 --alpha 1 --lda 74 --ldb 74 --beta 0 --ldc 2048
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 2048 -n 512 -k 100 --alpha 1 --lda 100 --ldb 100 --beta 1.0 --ldc 2048
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 200 -k 560 --alpha -1.0 --lda 560 --ldb 560 --beta 1.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 2048 -n 512 -k 100 --alpha 1 --lda 100 --ldb 100 --beta 0 --ldc 2048
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 1024 -k 200 --alpha -1.0 --lda 1024 --ldb 1024 --beta 1.0 --ldc 1024

#medium
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 1600 -k 560 --alpha 1 --lda 560 --ldb 560 --beta 0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1600 -n 512 -k 1024 --alpha -1.0 --lda 1600 --ldb 1024 --beta 1.0 --ldc 1600
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 560 -n 1600 -k 1024 --alpha 1 --lda 560 --ldb 1024 --beta 0 --ldc 560
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 1600 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024

#good
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 3200 -n 1024 -k 2048 --alpha 1.0 --lda 3200 --ldb 2048 --beta 0.0 --ldc 3200
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 3200 -n 2048 -k 1024 --alpha 1.0 --lda 3200 --ldb 2048 --beta 0.0 --ldc 3200
