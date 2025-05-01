#!/bin/bash

./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 2048 -n 1024 -k 1 --alpha 1.0 --lda 2048 --ldb 1 --beta 0.0 --ldc 2048
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 256 -n 1024 -k 1 --alpha 1.0 --lda 256 --ldb 1 --beta 0.0 --ldc 256
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 4096 -n 1024 -k 1 --alpha 1.0 --lda 4096 --ldb 1 --beta 0.0 --ldc 4096
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 257 -n 1024 -k 4096 --alpha 1.0 --lda 257 --ldb 4096 --beta 0.0 --ldc 257
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 3200 -n 1024 -k 2048 --alpha 1.0 --lda 3200 --ldb 2048 --beta 0.0 --ldc 3200
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 2048 -n 1024 -k 256 --alpha 1.0 --lda 2048 --ldb 256 --beta 0.0 --ldc 2048
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 3200 -n 2048 -k 1024 --alpha 1.0 --lda 3200 --ldb 2048 --beta 0.0 --ldc 3200
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 4096 -n 4096 -k 1024 --alpha 1.0 --lda 4096 --ldb 4096 --beta 0.0 --ldc 4096
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 257 -n 4096 -k 1024 --alpha 1.0 --lda 257 --ldb 4096 --beta 0.0 --ldc 257
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 2048 -n 256 -k 1024 --alpha 1.0 --lda 2048 --ldb 256 --beta 0.0 --ldc 2048
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 2048 -n 2048 -k 1024 --alpha 1.0 --lda 2048 --ldb 2048 --beta 0.0 --ldc 2048
