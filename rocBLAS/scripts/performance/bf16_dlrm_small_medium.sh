#!/bin/bash

./rocblas-bench -f gemm_ex -r bf16_r --transposeA N --transposeB N -m 512 -n 1600 -k 512 --alpha 1 --lda 512 --ldb 512 --beta 0 --ldc 512 --ldd 512 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA N --transposeB N -m 512 -n 1600 -k 32 --alpha 1 --lda 512 --ldb 32 --beta 0 --ldc 512 --ldd 512 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA N --transposeB T -m 512 -n 512 -k 1600 --alpha 1 --lda 512 --ldb 512 --beta 0 --ldc 512 --ldd 512 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA N --transposeB T -m 560 -n 1024 -k 1600 --alpha 1 --lda 560 --ldb 1024 --beta 0 --ldc 560 --ldd 560 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA N --transposeB N -m 1024 -n 1600 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024 --ldd 1024 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA N --transposeB N -m 1024 -n 1600 -k 1 --alpha 1 --lda 1024 --ldb 1 --beta 0 --ldc 1024 --ldd 1024 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA N --transposeB N -m 560 -n 1600 -k 1024 --alpha 1 --lda 560 --ldb 1024 --beta 0 --ldc 560 --ldd 560 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA T --transposeB N -m 32 -n 1600 -k 512 --alpha 1 --lda 512 --ldb 512 --beta 0 --ldc 32 --ldd 32 --compute_type s
#./rocblas-bench -f gemm_ex -r bf16_r --transposeA N --transposeB N -m 1024 -n 1 -k 1600 --alpha 1 --lda 1024 --ldb 1600 --beta 0 --ldc 1024 --ldd 1024 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA T --transposeB N -m 512 -n 1600 -k 512 --alpha 1 --lda 512 --ldb 512 --beta 0 --ldc 512 --ldd 512 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA T --transposeB N -m 1024 -n 1600 -k 560 --alpha 1 --lda 560 --ldb 560 --beta 0 --ldc 1024 --ldd 1024 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA N --transposeB T -m 1024 -n 1024 -k 1600 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024 --ldd 1024 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA T --transposeB N -m 1024 -n 1600 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024 --ldd 1024 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA N --transposeB T -m 512 -n 32 -k 1600 --alpha 1 --lda 512 --ldb 32 --beta 0 --ldc 512 --ldd 512 --compute_type s
#./rocblas-bench -f gemm_ex -r bf16_r --transposeA T --transposeB N -m 1600 -n 1 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1600 --ldd 1600 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA N --transposeB N -m 100 -n 512 -k 2048 --alpha 1 --lda 100 --ldb 2048 --beta 0 --ldc 100 --ldd 100 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA N --transposeB N -m 1024 -n 512 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024 --ldd 1024 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA N --transposeB N -m 1024 -n 512 -k 64 --alpha 1 --lda 1024 --ldb 64 --beta 0 --ldc 1024 --ldd 1024 --compute_type s
#./rocblas-bench -f gemm_ex -r bf16_r --transposeA N --transposeB N -m 2048 -n 1 -k 512 --alpha 1 --lda 2048 --ldb 512 --beta 0 --ldc 2048 --ldd 2048 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA N --transposeB N -m 2048 -n 512 -k 1 --alpha 1 --lda 2048 --ldb 1 --beta 0 --ldc 2048 --ldd 2048 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA N --transposeB N -m 2048 -n 512 -k 2048 --alpha 1 --lda 2048 --ldb 2048 --beta 0 --ldc 2048 --ldd 2048 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA N --transposeB N -m 74 -n 512 -k 2048 --alpha 1 --lda 74 --ldb 2048 --beta 0 --ldc 74 --ldd 74 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA N --transposeB T -m 100 -n 2048 -k 512 --alpha 1 --lda 100 --ldb 2048 --beta 0 --ldc 100 --ldd 100 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA N --transposeB T -m 1024 -n 1024 -k 512 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024 --ldd 1024 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA N --transposeB T -m 1024 -n 64 -k 512 --alpha 1 --lda 1024 --ldb 64 --beta 0 --ldc 1024 --ldd 1024 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA N --transposeB T -m 1600 -n 1024 -k 512 --alpha 1 --lda 1600 --ldb 1024 --beta 0 --ldc 1600 --ldd 1600 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA N --transposeB T -m 2048 -n 2048 -k 512 --alpha 1 --lda 2048 --ldb 2048 --beta 0 --ldc 2048 --ldd 2048 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA N --transposeB T -m 74 -n 2048 -k 512 --alpha 1 --lda 74 --ldb 2048 --beta 0 --ldc 74 --ldd 74 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA T --transposeB N -m 1024 -n 512 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024 --ldd 1024 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA T --transposeB N -m 1024 -n 512 -k 1600 --alpha 1 --lda 1600 --ldb 1600 --beta 0 --ldc 1024 --ldd 1024 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA T --transposeB N -m 2048 -n 512 -k 100 --alpha 1 --lda 100 --ldb 100 --beta 0 --ldc 2048 --ldd 2048 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA T --transposeB N -m 2048 -n 512 -k 2048 --alpha 1 --lda 2048 --ldb 2048 --beta 0 --ldc 2048 --ldd 2048 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA T --transposeB N -m 2048 -n 512 -k 74 --alpha 1 --lda 74 --ldb 74 --beta 0 --ldc 2048 --ldd 2048 --compute_type s
#./rocblas-bench -f gemm_ex -r bf16_r --transposeA T --transposeB N -m 512 -n 1 -k 2048 --alpha 1 --lda 2048 --ldb 2048 --beta 0 --ldc 512 --ldd 512 --compute_type s
./rocblas-bench -f gemm_ex -r bf16_r --transposeA T --transposeB N -m 64 -n 512 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 64 --ldd 64 --compute_type s
./rocblas-bench -f gemm_strided_batched_ex -r bf16_r --transposeA N --transposeB N -m 32 -n 33 -k 33 --alpha 1 --lda 32 --stride_a 1056 --ldb 33 --stride_b 1089 --beta 0 --ldc 32 --ldd 32 --stride_c 1056 --batch_count 1600 --compute_type s
./rocblas-bench -f gemm_strided_batched_ex -r bf16_r --transposeA N --transposeB N -m 64 -n 5 -k 5 --alpha 1 --lda 64 --stride_a 320 --ldb 5 --stride_b 25 --beta 0 --ldc 64 --ldd 64 --stride_c 320 --batch_count 512 --compute_type s
./rocblas-bench -f gemm_strided_batched_ex -r bf16_r --transposeA N --transposeB N -m 64 -n 9 -k 9 --alpha 1 --lda 64 --stride_a 576 --ldb 9 --stride_b 81 --beta 0 --ldc 64 --ldd 64 --stride_c 576 --batch_count 512 --compute_type s
./rocblas-bench -f gemm_strided_batched_ex -r bf16_r --transposeA N --transposeB T -m 33 -n 32 -k 33 --alpha 1 --lda 33 --stride_a 1089 --ldb 32 --stride_b 1056 --beta 0 --ldc 33 --ldd 33 --stride_c 1056 --batch_count 1600 --compute_type s
./rocblas-bench -f gemm_strided_batched_ex -r bf16_r --transposeA N --transposeB T -m 5 -n 64 -k 5 --alpha 1 --lda 5 --stride_a 25 --ldb 64 --stride_b 320 --beta 0 --ldc 5 --ldd 5 --stride_c 320 --batch_count 512 --compute_type s
./rocblas-bench -f gemm_strided_batched_ex -r bf16_r --transposeA N --transposeB T -m 9 -n 64 -k 9 --alpha 1 --lda 9 --stride_a 81 --ldb 64 --stride_b 576 --beta 0 --ldc 9 --ldd 9 --stride_c 576 --batch_count 512 --compute_type s
./rocblas-bench -f gemm_strided_batched_ex -r bf16_r --transposeA T --transposeB N -m 33 -n 33 -k 32 --alpha 1 --lda 32 --stride_a 1056 --ldb 32 --stride_b 1056 --beta 0 --ldc 33 --ldd 33 --stride_c 1089 --batch_count 1600 --compute_type s
./rocblas-bench -f gemm_strided_batched_ex -r bf16_r --transposeA T --transposeB N -m 5 -n 5 -k 64 --alpha 1 --lda 64 --stride_a 320 --ldb 64 --stride_b 320 --beta 0 --ldc 5 --ldd 5 --stride_c 25 --batch_count 512 --compute_type s
./rocblas-bench -f gemm_strided_batched_ex -r bf16_r --transposeA T --transposeB N -m 9 -n 9 -k 64 --alpha 1 --lda 64 --stride_a 576 --ldb 64 --stride_b 576 --beta 0 --ldc 9 --ldd 9 --stride_c 81 --batch_count 512 --compute_type s
