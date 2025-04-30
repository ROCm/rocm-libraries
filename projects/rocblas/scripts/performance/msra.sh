#!/bin/bash

./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB T -m 64 -n 512 -k 512 --alpha 1.0 --lda 64 --stride_a 32768 --ldb 512 --stride_b 262144 --beta 0.0 --ldc 64 --stride_c 32768 --batch_count 96 -i 48
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB T -m 512 -n 64 -k 512 --alpha 1.0 --lda 512 --stride_a 262144 --ldb 64 --stride_b 32768 --beta 0.0 --ldc 512 --stride_c 32768 --batch_count 96 -i 48
./rocblas-bench -f gemm_strided_batched -r s --transposeA T --transposeB N -m 64 -n 512 -k 512 --alpha 1.0 --lda 512 --stride_a 32768 --ldb 512 --stride_b 262144 --beta 0.0 --ldc 64 --stride_c 32768 --batch_count 96 -i 48
./rocblas-bench -f gemm_strided_batched -r s --transposeA T --transposeB N -m 512 -n 512 -k 64 --alpha 1.0 --lda 64 --stride_a 32768 --ldb 64 --stride_b 32768 --beta 0.0 --ldc 512 --stride_c 262144 --batch_count 96 -i 48
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB N -m 512 -n 512 -k 64 --alpha 1.0 --lda 512 --stride_a 32768 --ldb 64 --stride_b 32768 --beta 0.0 --ldc 512 --stride_c 262144 --batch_count 96 -i 48
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB N -m 64 -n 512 -k 512 --alpha 1.0 --lda 64 --stride_a 32768 --ldb 512 --stride_b 262144 --beta 0.0 --ldc 64 --stride_c 32768 --batch_count 96 -i 48
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 4096 -n 1024 -k 3072 --alpha 1.0 --lda 4096 --ldb 1024 --beta 0.0 --ldc 4096 -i 48
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 1024 -n 2 -k 6 --alpha 1.0 --lda 1024 --ldb 2 --beta 0.0 --ldc 1024 -i 2
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 1024 -n 1024 -k 6 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024 -i 2
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 1024 -n 1024 -k 3072 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024 -i 194
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 30528 -n 1024 -k 3072 --alpha 1.0 --lda 30528 --ldb 1024 --beta 0.0 --ldc 30528 -i 2
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 1024 -n 4096 -k 3072 --alpha 1.0 --lda 1024 --ldb 4096 --beta 0.0 --ldc 1024 -i 48
./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 1024 -n 3072 -k 4096 --alpha 1.0 --lda 4096 --ldb 4096 --beta 0.0 --ldc 1024 -i 48
./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 2 -n 6 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 2 -i 2
./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 4096 -n 3072 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 4096 -i 48
./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 1024 -n 6 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024 -i 2
./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 30528 -n 3072 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 30528 -i 2
./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 1024 -n 3072 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024 -i 194
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 1024 -n 3072 -k 4096 --alpha 1.0 --lda 1024 --ldb 4096 --beta 0.0 --ldc 1024 -i 48
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 1024 -n 3072 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024 -i 194
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 4096 -n 3072 -k 1024 --alpha 1.0 --lda 4096 --ldb 1024 --beta 0.0 --ldc 4096 -i 48
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 1024 -n 6 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024 -i 2
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 1024 -n 3072 -k 30528 --alpha 1.0 --lda 1024 --ldb 30528 --beta 0.0 --ldc 1024 -i 2
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 1024 -n 6 -k 2 --alpha 1.0 --lda 1024 --ldb 2 --beta 0.0 --ldc 1024 -i 2
