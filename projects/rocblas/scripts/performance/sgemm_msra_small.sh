#!/bin/bash

./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 1024 -n 1024 -k 4096 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB N -m 64 -n 128 -k 128 --alpha 1.0 --lda 64 --stride_a 8192 --ldb 128 --stride_b 16384 --beta 0.0 --ldc 64 --stride_c 8192 --batch 512
./rocblas-bench -f gemm_strided_batched -r s --transposeA N --transposeB T -m 64 -n 128 -k 128 --alpha 1.0 --lda 64 --stride_a 8192 --ldb 128 --stride_b 16384 --beta 0.0 --ldc 64 --stride_c 8192 --batch 512
./rocblas-bench -f gemm_strided_batched -r s --transposeA T --transposeB N -m 128 -n 128 -k 64 --alpha 1.0 --lda 64 --stride_a 8192 --ldb 64 --stride_b 8192 --beta 0.0 --ldc 128 --stride_c 16384 --batch 512
