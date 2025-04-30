#!/bin/bash

./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 1 -k 200 --alpha 1.0 --lda 1024 --ldb 200 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 67 -n 512 -k 2048 --alpha 1.0 --lda 67 --ldb 2048 --beta 0.0 --ldc 67
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 2048 -n 1 -k 512 --alpha 1.0 --lda 2048 --ldb 512 --beta 0.0 --ldc 2048
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 67 -n 2048 -k 512 --alpha 1.0 --lda 67 --ldb 2048 --beta 0.0 --ldc 67
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 200 -n 1 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 200
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 2048 -n 512 -k 67 --alpha 1.0 --lda 67 --ldb 67 --beta 0.0 --ldc 2048
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 512 -n 1 -k 2048 --alpha 1.0 --lda 2048 --ldb 2048 --beta 0.0 --ldc 512
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB N -m 64 -n 3 -k 3 --alpha 1.0 --lda 64 --stride_a 192 --ldb 3 --stride_b 9 --beta 0.0 --ldc 64 --stride_c 192 --batch 512
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB T -m 33 -n 32 -k 33 --alpha 1.0 --lda 32 --stride_a 1089 --ldb 32 --stride_b 1056 --beta 0.0 --ldc 33 --stride_c 1056 --batch 200
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB T -m 3 -n 64 -k 3 --alpha 1.0 --lda 3 --stride_a 9 --ldb 64 --stride_b 92 --beta 0.0 --ldc 3 --stride_c 192 --batch 512
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA T --transposeB N -m 3 -n 3 -k 64 --alpha 1.0 --lda 64 --stride_a 192 --ldb 64 --stride_b 92 --beta 0.0 --ldc 3 --stride_c 9 --batch 512
