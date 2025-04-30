#!/bin/bash

./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 1024 -n 4096 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024 -i 97
./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 4096 -n 4096 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 4096 -i 24
./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 1024 -n 4096 -k 4096 --alpha 1.0 --lda 4096 --ldb 4096 --beta 0.0 --ldc 1024 -i 24
./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 1024 -n 32 -k 1024 --alpha 1.0 --lda 1024 --ldb 131072 --beta 0.0 --ldc 1024 -i 1
./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 30528 -n 4096 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 30528 -i 1
./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 2 -n 32 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 1.0 --ldc 2 -i 1
./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 1024 -n 2048 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024 -i 97
./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 4096 -n 2048 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 4096 -i 24
./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 1024 -n 4 -k 1024 --alpha 1.0 --lda 1024 --ldb 524288 --beta 0.0 --ldc 1024 -i 1
./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 2 -n 4 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 1.0 --ldc 2 -i 1
./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 1024 -n 2048 -k 4096 --alpha 1.0 --lda 4096 --ldb 4096 --beta 0.0 --ldc 1024 -i 24
./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 30528 -n 2048 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 30528 -i 97

./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 1024 -n 32 -k 2 --alpha 1.0 --lda 1024 --ldb 2 --beta 0.0 --ldc 1024 -i 1
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 1024 -n 4096 -k 30528 --alpha 1.0 --lda 1024 --ldb 30528 --beta 0.0 --ldc 1024 -i 1
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 1024 -n 4096 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024 -i 97
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 1024 -n 32 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024 -i 1
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 4096 -n 4096 -k 1024 --alpha 1.0 --lda 4096 --ldb 1024 --beta 0.0 --ldc 4096 -i 24
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 1024 -n 4096 -k 4096 --alpha 1.0 --lda 1024 --ldb 4096 --beta 0.0 --ldc 1024 -i 24
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 1024 -n 4 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024 -i 1
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 1024 -n 2048 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024 -i 97
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 1024 -n 2048 -k 4096 --alpha 1.0 --lda 1024 --ldb 4096 --beta 0.0 --ldc 1024 -i 24
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 1024 -n 4 -k 2 --alpha 1.0 --lda 1024 --ldb 2 --beta 0.0 --ldc 1024 -i 1
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 4096 -n 2048 -k 1024 --alpha 1.0 --lda 4096 --ldb 1024 --beta 0.0 --ldc 4096 -i 24
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 1024 -n 2048 -k 30528 --alpha 1.0 --lda 1024 --ldb 30528 --beta 0.0 --ldc 1024 -i 1

./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 1024 -n 2 -k 32 --alpha 1.0 --lda 1024 --ldb 2 --beta 0.0 --ldc 1024 -i 1
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 1024 -n 30528 -k 4096 --alpha 1.0 --lda 1024 --ldb 30528 --beta 0.0 --ldc 1024 -i 1
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 1024 -n 1024 -k 4096 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024 -i 97
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 1024 -n 1024 -k 32 --alpha 1.0 --lda 131072 --ldb 1024 --beta 0.0 --ldc 1024 -i 1
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 4096 -n 1024 -k 4096 --alpha 1.0 --lda 4096 --ldb 1024 --beta 0.0 --ldc 4096 -i 24
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 1024 -n 4096 -k 4096 --alpha 1.0 --lda 1024 --ldb 4096 --beta 0.0 --ldc 1024 -i 24
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 1024 -n 1024 -k 2048 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024 -i 97
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 1024 -n 2 -k 4 --alpha 1.0 --lda 1024 --ldb 2 --beta 0.0 --ldc 1024 -i 1
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 4096 -n 1024 -k 2048 --alpha 1.0 --lda 4096 --ldb 1024 --beta 0.0 --ldc 4096 -i 24
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 1024 -n 30528 -k 2048 --alpha 1.0 --lda 1024 --ldb 30528 --beta 0.0 --ldc 1024 -i 1
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 1024 -n 1024 -k 4 --alpha 1.0 --lda 524288 --ldb 1024 --beta 0.0 --ldc 1024 -i 1
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 1024 -n 4096 -k 2048 --alpha 1.0 --lda 1024 --ldb 4096 --beta 0.0 --ldc 1024 -i 24

./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB N -m 128 -n 128 -k 64 --alpha 1.0 --lda 128 --stride_a 8192 --ldb 64 --stride_b 8192 --beta 0.0 --ldc 128 --stride_c 16384 --batch_count 512 -i 24
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB N -m 64 -n 128 -k 128 --alpha 1.0 --lda 64 --stride_a 8192 --ldb 128 --stride_b 16384 --beta 0.0 --ldc 64 --stride_c 8192 --batch_count 512 -i 24
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB N -m 64 -n 512 -k 512 --alpha 1.0 --lda 64 --stride_a 32768 --ldb 512 --stride_b 262144 --beta 0.0 --ldc 64 --stride_c 32768 --batch_count 64 -i 24
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB N -m 512 -n 512 -k 64 --alpha 1.0 --lda 512 --stride_a 32768 --ldb 64 --stride_b 32768 --beta 0.0 --ldc 512 --stride_c 262144 --batch_count 64 -i 24

./rocblas-bench -f gemm_strided_batched -r s --transposeA T --transposeB N -m 128 -n 128 -k 64 --alpha 1.0 --lda 64 --stride_a 8192 --ldb 64 --stride_b 8192 --beta 0.0 --ldc 128 --stride_c 16384 --batch_count 512 -i 24
./rocblas-bench -f gemm_strided_batched -r s --transposeA T --transposeB N -m 64 -n 128 -k 128 --alpha 1.0 --lda 1 --stride_a 0 --ldb 1 --stride_b 0 --beta 0.0 --ldc 1 --stride_c 0 --batch_count 512 -i 24
./rocblas-bench -f gemm_strided_batched -r s --transposeA T --transposeB N -m 64 -n 512 -k 512 --alpha 1.0 --lda 512 --stride_a 32768 --ldb 512 --stride_b 262144 --beta 0.0 --ldc 64 --stride_c 32768 --batch_count 64 -i 24
./rocblas-bench -f gemm_strided_batched -r s --transposeA T --transposeB N -m 512 -n 512 -k 64 --alpha 1.0 --lda 64 --stride_a 32768 --ldb 64 --stride_b 32768 --beta 0.0 --ldc 512 --stride_c 262144 --batch_count 64 -i 24

./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB T -m 128 -n 64 -k 128 --alpha 1.0 --lda 128 --stride_a 16384 --ldb 64 --stride_b 8192 --beta 0.0 --ldc 128 --stride_c 8192 --batch_count 512 -i 24
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB T -m 64 -n 128 -k 128 --alpha 1.0 --lda 64 --stride_a 8192 --ldb 128 --stride_b 16384 --beta 0.0 --ldc 64 --stride_c 8192 --batch_count 512 -i 24
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB T -m 512 -n 64 -k 512 --alpha 1.0 --lda 512 --stride_a 262144 --ldb 64 --stride_b 32768 --beta 0.0 --ldc 512 --stride_c 32768 --batch_count 64 -i 24
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB T -m 64 -n 512 -k 512 --alpha 1.0 --lda 64 --stride_a 32768 --ldb 512 --stride_b 262144 --beta 0.0 --ldc 64 --stride_c 32768 --batch_count 64 -i 24
