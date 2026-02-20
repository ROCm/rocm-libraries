#!/bin/bash
bench=./rocblas-bench
if [ ! -f ${bench} ]; then
        echo ${bench} not found, exit...
        exit 1
else
        echo ">>" $(realpath $(ldd ${bench} | grep rocblas | awk '{print $3;}'))
fi
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB N -m 3136 -n 256 -k 64 --alpha 1 --lda 3136 --stride_a 200704 --ldb 64 --stride_b 0 --beta 0 --ldc 3136 --stride_c 802816 --batch 128
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB N -m 3136 -n 256 -k 64 --alpha 1 --lda 3136 --stride_a 200704 --ldb 64 --stride_b 0 --beta 0 --ldc 3136 --stride_c 802816 --batch 256
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB N -m 3136 -n 256 -k 64 --alpha 1 --lda 3136 --stride_a 200704 --ldb 64 --stride_b 0 --beta 0 --ldc 3136 --stride_c 802816 --batch 64
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB N -m 3136 -n 64 -k 256 --alpha 1 --lda 3136 --stride_a 802816 --ldb 256 --stride_b 0 --beta 0 --ldc 3136 --stride_c 200704 --batch 128
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB N -m 3136 -n 64 -k 256 --alpha 1 --lda 3136 --stride_a 802816 --ldb 256 --stride_b 0 --beta 0 --ldc 3136 --stride_c 200704 --batch 256
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB N -m 3136 -n 64 -k 256 --alpha 1 --lda 3136 --stride_a 802816 --ldb 256 --stride_b 0 --beta 0 --ldc 3136 --stride_c 200704 --batch 64
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB N -m 3136 -n 64 -k 64 --alpha 1 --lda 3136 --stride_a 200704 --ldb 64 --stride_b 0 --beta 0 --ldc 3136 --stride_c 200704 --batch 128
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB N -m 3136 -n 64 -k 64 --alpha 1 --lda 3136 --stride_a 200704 --ldb 64 --stride_b 0 --beta 0 --ldc 3136 --stride_c 200704 --batch 256
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB N -m 3136 -n 64 -k 64 --alpha 1 --lda 3136 --stride_a 200704 --ldb 64 --stride_b 0 --beta 0 --ldc 3136 --stride_c 200704 --batch 64
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB N -m 784 -n 128 -k 512 --alpha 1 --lda 784 --stride_a 401408 --ldb 512 --stride_b 0 --beta 0 --ldc 784 --stride_c 100352 --batch 128
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB N -m 784 -n 128 -k 512 --alpha 1 --lda 784 --stride_a 401408 --ldb 512 --stride_b 0 --beta 0 --ldc 784 --stride_c 100352 --batch 256
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB N -m 784 -n 128 -k 512 --alpha 1 --lda 784 --stride_a 401408 --ldb 512 --stride_b 0 --beta 0 --ldc 784 --stride_c 100352 --batch 64
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB N -m 784 -n 512 -k 128 --alpha 1 --lda 784 --stride_a 100352 --ldb 128 --stride_b 0 --beta 0 --ldc 784 --stride_c 401408 --batch 128
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB N -m 784 -n 512 -k 128 --alpha 1 --lda 784 --stride_a 100352 --ldb 128 --stride_b 0 --beta 0 --ldc 784 --stride_c 401408 --batch 256
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB N -m 784 -n 512 -k 128 --alpha 1 --lda 784 --stride_a 100352 --ldb 128 --stride_b 0 --beta 0 --ldc 784 --stride_c 401408 --batch 64
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 196 -n 1024 -k 256 --alpha 1 --lda 196 --stride_a 50176 --ldb 1024 --stride_b 0 --beta 0 --ldc 196 --stride_c 200704 --batch 128
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 196 -n 1024 -k 256 --alpha 1 --lda 196 --stride_a 50176 --ldb 1024 --stride_b 0 --beta 0 --ldc 196 --stride_c 200704 --batch 256
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 196 -n 1024 -k 256 --alpha 1 --lda 196 --stride_a 50176 --ldb 1024 --stride_b 0 --beta 0 --ldc 196 --stride_c 200704 --batch 64
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 196 -n 256 -k 1024 --alpha 1 --lda 196 --stride_a 200704 --ldb 256 --stride_b 0 --beta 0 --ldc 196 --stride_c 50176 --batch 128
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 196 -n 256 -k 1024 --alpha 1 --lda 196 --stride_a 200704 --ldb 256 --stride_b 0 --beta 0 --ldc 196 --stride_c 50176 --batch 256
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 196 -n 256 -k 1024 --alpha 1 --lda 196 --stride_a 200704 --ldb 256 --stride_b 0 --beta 0 --ldc 196 --stride_c 50176 --batch 64
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 3136 -n 256 -k 64 --alpha 1 --lda 3136 --stride_a 200704 --ldb 256 --stride_b 0 --beta 0 --ldc 3136 --stride_c 802816 --batch 128
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 3136 -n 256 -k 64 --alpha 1 --lda 3136 --stride_a 200704 --ldb 256 --stride_b 0 --beta 0 --ldc 3136 --stride_c 802816 --batch 256
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 3136 -n 256 -k 64 --alpha 1 --lda 3136 --stride_a 200704 --ldb 256 --stride_b 0 --beta 0 --ldc 3136 --stride_c 802816 --batch 64
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 3136 -n 64 -k 256 --alpha 1 --lda 3136 --stride_a 802816 --ldb 64 --stride_b 0 --beta 0 --ldc 3136 --stride_c 200704 --batch 128
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 3136 -n 64 -k 256 --alpha 1 --lda 3136 --stride_a 802816 --ldb 64 --stride_b 0 --beta 0 --ldc 3136 --stride_c 200704 --batch 256
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 3136 -n 64 -k 256 --alpha 1 --lda 3136 --stride_a 802816 --ldb 64 --stride_b 0 --beta 0 --ldc 3136 --stride_c 200704 --batch 64
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 3136 -n 64 -k 64 --alpha 1 --lda 3136 --stride_a 200704 --ldb 64 --stride_b 0 --beta 0 --ldc 3136 --stride_c 200704 --batch 128
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 3136 -n 64 -k 64 --alpha 1 --lda 3136 --stride_a 200704 --ldb 64 --stride_b 0 --beta 0 --ldc 3136 --stride_c 200704 --batch 256
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 3136 -n 64 -k 64 --alpha 1 --lda 3136 --stride_a 200704 --ldb 64 --stride_b 0 --beta 0 --ldc 3136 --stride_c 200704 --batch 64
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 49 -n 2048 -k 512 --alpha 1 --lda 49 --stride_a 25088 --ldb 2048 --stride_b 0 --beta 0 --ldc 49 --stride_c 100352 --batch 128
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 49 -n 2048 -k 512 --alpha 1 --lda 49 --stride_a 25088 --ldb 2048 --stride_b 0 --beta 0 --ldc 49 --stride_c 100352 --batch 256
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 49 -n 2048 -k 512 --alpha 1 --lda 49 --stride_a 25088 --ldb 2048 --stride_b 0 --beta 0 --ldc 49 --stride_c 100352 --batch 64
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 49 -n 512 -k 2048 --alpha 1 --lda 49 --stride_a 100352 --ldb 512 --stride_b 0 --beta 0 --ldc 49 --stride_c 25088 --batch 128
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 49 -n 512 -k 2048 --alpha 1 --lda 49 --stride_a 100352 --ldb 512 --stride_b 0 --beta 0 --ldc 49 --stride_c 25088 --batch 256
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 49 -n 512 -k 2048 --alpha 1 --lda 49 --stride_a 100352 --ldb 512 --stride_b 0 --beta 0 --ldc 49 --stride_c 25088 --batch 64
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 784 -n 128 -k 512 --alpha 1 --lda 784 --stride_a 401408 --ldb 128 --stride_b 0 --beta 0 --ldc 784 --stride_c 100352 --batch 128
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 784 -n 128 -k 512 --alpha 1 --lda 784 --stride_a 401408 --ldb 128 --stride_b 0 --beta 0 --ldc 784 --stride_c 100352 --batch 256
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 784 -n 128 -k 512 --alpha 1 --lda 784 --stride_a 401408 --ldb 128 --stride_b 0 --beta 0 --ldc 784 --stride_c 100352 --batch 64
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 784 -n 512 -k 128 --alpha 1 --lda 784 --stride_a 100352 --ldb 512 --stride_b 0 --beta 0 --ldc 784 --stride_c 401408 --batch 128
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 784 -n 512 -k 128 --alpha 1 --lda 784 --stride_a 100352 --ldb 512 --stride_b 0 --beta 0 --ldc 784 --stride_c 401408 --batch 256
${bench} -f gemm_strided_batched -r s --transposeA N --transposeB T -m 784 -n 512 -k 128 --alpha 1 --lda 784 --stride_a 100352 --ldb 512 --stride_b 0 --beta 0 --ldc 784 --stride_c 401408 --batch 64
