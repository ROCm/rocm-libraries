#!/bin/bash

./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 4096 -k 2 --alpha 1 --lda 768 --ldb 2 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 4096 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA T --transposeB N -m 256 -n 256 -k 64 --alpha 1 --lda 64 --stride_a 16384 --ldb 64 --stride_b 16384 --beta 0 --ldc 256 --stride_c 65536 --batch 192
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB N -m 64 -n 256 -k 256 --alpha 1 --lda 64 --stride_a 16384 --ldb 256 --stride_b 65536 --beta 0 --ldc 64 --stride_c 16384 --batch 192
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 3072 -n 4096 -k 768 --alpha 1 --lda 3072 --ldb 768 --beta 0 --ldc 3072
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 4096 -k 3072 --alpha 1 --lda 768 --ldb 3072 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 320 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 16 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 2 -n 16 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 2
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 768 -n 2 -k 16 --alpha 1 --lda 768 --ldb 2 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 16 -k 2 --alpha 1 --lda 768 --ldb 2 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 768 -n 768 -k 16 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 768 -n 16 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 30522 -n 320 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 30522
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 320 -k 30522 --alpha 1 --lda 768 --ldb 30522 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 768 -n 30522 -k 320 --alpha 1 --lda 768 --ldb 30522 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 768 -n 768 -k 320 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 768 -n 320 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 768 -n 3072 -k 4096 --alpha 1 --lda 768 --ldb 3072 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 3072 -n 4096 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 3072
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 3072 -n 768 -k 4096 --alpha 1 --lda 3072 --ldb 768 --beta 0 --ldc 3072
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 768 -n 4096 -k 3072 --alpha 1 --lda 3072 --ldb 3072 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 768 -n 768 -k 4096 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 768 -n 4096 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB T -m 64 -n 256 -k 256 --alpha 1 --lda 64 --stride_a 16384 --ldb 256 --stride_b 65536 --beta 0 --ldc 64 --stride_c 16384 --batch 192
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 768 -n 2 -k 4096 --alpha 1 --lda 768 --ldb 2 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 2048 -k 2 --alpha 1 --lda 768 --ldb 2 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 2048 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA T --transposeB N -m 256 -n 256 -k 64 --alpha 1 --lda 64 --stride_a 16384 --ldb 64 --stride_b 16384 --beta 0 --ldc 256 --stride_c 65536 --batch 96
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB N -m 64 -n 256 -k 256 --alpha 1 --lda 64 --stride_a 16384 --ldb 256 --stride_b 65536 --beta 0 --ldc 64 --stride_c 16384 --batch 96
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 3072 -n 2048 -k 768 --alpha 1 --lda 3072 --ldb 768 --beta 0 --ldc 3072
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 2048 -k 3072 --alpha 1 --lda 768 --ldb 3072 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 160 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 8 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 2 -n 8 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 2
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 30522 -n 160 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 30522
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA T --transposeB N -m 128 -n 128 -k 64 --alpha 1 --lda 64 --stride_a 8192 --ldb 64 --stride_b 8192 --beta 0 --ldc 128 --stride_c 16384 --batch 384
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB N -m 64 -n 128 -k 128 --alpha 1 --lda 64 --stride_a 8192 --ldb 128 --stride_b 16384 --beta 0 --ldc 64 --stride_c 8192 --batch 384
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 640 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 32 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 2 -n 32 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 2
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 768 -n 2 -k 32 --alpha 1 --lda 768 --ldb 2 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 32 -k 2 --alpha 1 --lda 768 --ldb 2 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 768 -n 768 -k 32 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 768 -n 32 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 30522 -n 640 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 30522
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 640 -k 30522 --alpha 1 --lda 768 --ldb 30522 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 768 -n 30522 -k 640 --alpha 1 --lda 768 --ldb 30522 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 768 -n 768 -k 640 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 768 -n 640 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB T -m 64 -n 128 -k 128 --alpha 1 --lda 64 --stride_a 8192 --ldb 128 --stride_b 16384 --beta 0 --ldc 64 --stride_c 8192 --batch 384
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 1024 -k 2 --alpha 1 --lda 768 --ldb 2 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 1024 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA T --transposeB N -m 128 -n 128 -k 64 --alpha 1 --lda 64 --stride_a 8192 --ldb 64 --stride_b 8192 --beta 0 --ldc 128 --stride_c 16384 --batch 96
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB N -m 64 -n 128 -k 128 --alpha 1 --lda 64 --stride_a 8192 --ldb 128 --stride_b 16384 --beta 0 --ldc 64 --stride_c 8192 --batch 96
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 3072 -n 1024 -k 768 --alpha 1 --lda 3072 --ldb 768 --beta 0 --ldc 3072
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 1024 -k 3072 --alpha 1 --lda 768 --ldb 3072 --beta 0 --ldc 768
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA T --transposeB N -m 64 -n 64 -k 64 --alpha 1 --lda 64 --stride_a 4096 --ldb 64 --stride_b 4096 --beta 0 --ldc 64 --stride_c 4096 --batch 768
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB N -m 64 -n 64 -k 64 --alpha 1 --lda 64 --stride_a 4096 --ldb 64 --stride_b 4096 --beta 0 --ldc 64 --stride_c 4096 --batch 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 1280 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 64 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 2 -n 64 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 2
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 768 -n 2 -k 64 --alpha 1 --lda 768 --ldb 2 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 64 -k 2 --alpha 1 --lda 768 --ldb 2 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 768 -n 768 -k 64 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 768 -n 64 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 30522 -n 1280 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 30522
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 1280 -k 30522 --alpha 1 --lda 768 --ldb 30522 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 768 -n 30522 -k 1280 --alpha 1 --lda 768 --ldb 30522 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 768 -n 768 -k 1280 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 768 -n 1280 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB T -m 64 -n 64 -k 64 --alpha 1 --lda 64 --stride_a 4096 --ldb 64 --stride_b 4096 --beta 0 --ldc 64 --stride_c 4096 --batch 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 512 -k 2 --alpha 1 --lda 768 --ldb 2 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 512 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA T --transposeB N -m 64 -n 64 -k 64 --alpha 1 --lda 64 --stride_a 4096 --ldb 64 --stride_b 4096 --beta 0 --ldc 64 --stride_c 4096 --batch 96
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB N -m 64 -n 64 -k 64 --alpha 1 --lda 64 --stride_a 4096 --ldb 64 --stride_b 4096 --beta 0 --ldc 64 --stride_c 4096 --batch 96
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 3072 -n 512 -k 768 --alpha 1 --lda 3072 --ldb 768 --beta 0 --ldc 3072
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 512 -k 3072 --alpha 1 --lda 768 --ldb 3072 --beta 0 --ldc 768
