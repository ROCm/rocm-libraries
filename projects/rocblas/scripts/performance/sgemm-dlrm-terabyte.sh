#!/bin/bash

./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 256 -n 32768 -k 1 --alpha 1.0 --lda 256 --ldb 32768 --beta 0.0 --ldc 256 -i 10
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 256 -n 1 -k 32768 --alpha 1.0 --lda 256 --ldb 1 --beta 0.0 --ldc 256 -i 10
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 512 -n 256 -k 32768 --alpha 1.0 --lda 512 --ldb 256 --beta 0.0 --ldc 512 -i 10
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 256 -n 128 -k 32768 --alpha 1.0 --lda 256 --ldb 128 --beta 0.0 --ldc 256 -i 10
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 13 -n 512 -k 32768 --alpha 1.0 --lda 13 --ldb 512 --beta 0.0 --ldc 13 -i 10
