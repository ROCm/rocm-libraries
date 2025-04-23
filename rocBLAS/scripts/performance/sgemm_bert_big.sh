#!/bin/bash

./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 2048 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 3072 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 512 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 2048 -k 3072 --alpha 1.0 --lda 1024 --ldb 3072 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 3072 -k 3072 --alpha 1.0 --lda 1024 --ldb 3072 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 512 -k 3072 --alpha 1.0 --lda 1024 --ldb 3072 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 3072 -n 2048 -k 1024 --alpha 1.0 --lda 3072 --ldb 1024 --beta 0.0 --ldc 3072
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 3072 -n 3072 -k 1024 --alpha 1.0 --lda 3072 --ldb 1024 --beta 0.0 --ldc 3072
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 3072 -n 512 -k 1024 --alpha 1.0 --lda 3072 --ldb 1024 --beta 0.0 --ldc 3072
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 4096 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 4096 -k 3072 --alpha 1.0 --lda 1024 --ldb 3072 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 3072 -n 4096 -k 1024 --alpha 1.0 --lda 3072 --ldb 1024 --beta 0.0 --ldc 3072
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 120 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 120 -k 30522 --alpha 1.0 --lda 1024 --ldb 30522 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 1 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 1 -k 2 --alpha 1.0 --lda 1024 --ldb 2 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 2048 -k 2 --alpha 1.0 --lda 1024 --ldb 2 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 20 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 20 -k 30522 --alpha 1.0 --lda 1024 --ldb 30522 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 3072 -k 2 --alpha 1.0 --lda 1024 --ldb 2 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 4 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 4 -k 2 --alpha 1.0 --lda 1024 --ldb 2 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 512 -k 2 --alpha 1.0 --lda 1024 --ldb 2 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 6 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 6 -k 2 --alpha 1.0 --lda 1024 --ldb 2 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 80 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 80 -k 30522 --alpha 1.0 --lda 1024 --ldb 30522 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 160 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 4096 -k 2 --alpha 1.0 --lda 1024 --ldb 2 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 8 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 1024 -k 2048 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 1024 -k 3072 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 1024 -k 512 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 3072 -k 3072 --alpha 1.0 --lda 1024 --ldb 3072 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 3072 -k 512 --alpha 1.0 --lda 1024 --ldb 3072 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 3072 -n 1024 -k 2048 --alpha 1.0 --lda 3072 --ldb 1024 --beta 0.0 --ldc 3072
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 3072 -n 1024 -k 3072 --alpha 1.0 --lda 3072 --ldb 1024 --beta 0.0 --ldc 3072
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 3072 -n 1024 -k 512 --alpha 1.0 --lda 3072 --ldb 1024 --beta 0.0 --ldc 3072
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 3072 -k 2048 --alpha 1.0 --lda 1024 --ldb 3072 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 1024 -k 120 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 1024 -k 1 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 1024 -k 20 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 1024 -k 4 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 1024 -k 6 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 1024 -k 80 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 2 -k 1 --alpha 1.0 --lda 1024 --ldb 2 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 2 -k 2048 --alpha 1.0 --lda 1024 --ldb 2 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 2 -k 3072 --alpha 1.0 --lda 1024 --ldb 2 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 2 -k 4 --alpha 1.0 --lda 1024 --ldb 2 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 2 -k 512 --alpha 1.0 --lda 1024 --ldb 2 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 2 -k 6 --alpha 1.0 --lda 1024 --ldb 2 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 30522 -k 120 --alpha 1.0 --lda 1024 --ldb 30522 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 30522 -k 20 --alpha 1.0 --lda 1024 --ldb 30522 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 30522 -k 80 --alpha 1.0 --lda 1024 --ldb 30522 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB N -m 64 -n 512 -k 512 --alpha 1.0 --lda 64 --stride_a 32768 --ldb 512 --stride_b 262144 --beta 0.0 --ldc 64 --stride_c 32768 --batch 16
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB N -m 64 -n 512 -k 512 --alpha 1.0 --lda 64 --stride_a 32768 --ldb 512 --stride_b 262144 --beta 0.0 --ldc 64 --stride_c 32768 --batch 64
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB N -m 64 -n 512 -k 512 --alpha 1.0 --lda 64 --stride_a 32768 --ldb 512 --stride_b 262144 --beta 0.0 --ldc 64 --stride_c 32768 --batch 96
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB N -m 64 -n 512 -k 512 --alpha 1.0 --lda 64 --stride_a 32768 --ldb 512 --stride_b 262144 --beta 0.0 --ldc 64 --stride_c 32768 --batch 128
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB T -m 64 -n 512 -k 512 --alpha 1.0 --lda 64 --stride_a 32768 --ldb 512 --stride_b 262144 --beta 0.0 --ldc 64 --stride_c 32768 --batch 16
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB T -m 64 -n 512 -k 512 --alpha 1.0 --lda 64 --stride_a 32768 --ldb 512 --stride_b 262144 --beta 0.0 --ldc 64 --stride_c 32768 --batch 64
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB T -m 64 -n 512 -k 512 --alpha 1.0 --lda 64 --stride_a 32768 --ldb 512 --stride_b 262144 --beta 0.0 --ldc 64 --stride_c 32768 --batch 96
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA T --transposeB N -m 512 -n 512 -k 64 --alpha 1.0 --lda 64 --stride_a 32768 --ldb 64 --stride_b 32768 --beta 0.0 --ldc 512 --stride_c 262144 --batch 64
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA T --transposeB N -m 512 -n 512 -k 64 --alpha 0.125 --lda 64 --stride_a 32768 --ldb 64 --stride_b 32768 --beta 1.0 --ldc 512 --stride_c 262144 --batch 16
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA T --transposeB N -m 512 -n 512 -k 64 --alpha 0.125 --lda 64 --stride_a 32768 --ldb 64 --stride_b 32768 --beta 1.0 --ldc 512 --stride_c 262144 --batch 96
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA T --transposeB N -m 512 -n 512 -k 64 --alpha 1.0 --lda 64 --stride_a 32768 --ldb 64 --stride_b 32768 --beta 0.0 --ldc 512 --stride_c 262144 --batch 16
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA T --transposeB N -m 512 -n 512 -k 64 --alpha 1.0 --lda 64 --stride_a 32768 --ldb 64 --stride_b 32768 --beta 0.0 --ldc 512 --stride_c 262144 --batch 96
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA T --transposeB N -m 512 -n 512 -k 64 --alpha 0.125 --lda 64 --stride_a 32768 --ldb 64 --stride_b 32768 --beta 1.0 --ldc 512 --stride_c 262144 --batch 128
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA T --transposeB N -m 512 -n 512 -k 64 --alpha 1.0 --lda 64 --stride_a 32768 --ldb 64 --stride_b 32768 --beta 0.0 --ldc 512 --stride_c 262144 --batch 128
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 2048 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 3072 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 512 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 2048 -k 3072 --alpha 1.0 --lda 3072 --ldb 3072 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 3072 -k 3072 --alpha 1.0 --lda 3072 --ldb 3072 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 512 -k 3072 --alpha 1.0 --lda 3072 --ldb 3072 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 3072 -n 2048 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 3072
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 3072 -n 3072 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 3072
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 3072 -n 512 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 3072
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 80 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 2 -n 1 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 2
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 2 -n 4 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 2
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 2 -n 6 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 2
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 30522 -n 120 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 30522
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 30522 -n 20 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 30522
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 30522 -n 80 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 30522
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 120 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 1 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 20 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 4 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 6 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 2 -n 8 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 2
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 30522 -n 160 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 30522
