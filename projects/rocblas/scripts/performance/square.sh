#!/bin/bash

./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 512 -n 512 -k 1 --alpha 1.0 --lda 512 --ldb 1 --beta 1.0 --ldc 512
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 512 -n 512 -k 512 --alpha 1.0 --lda 512 --ldb 512 --beta 0.0 --ldc 512
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 1024 -k 1 --alpha 1.0 --lda 1024 --ldb 1 --beta 1.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 1024 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 512 -n 512 -k 1 --alpha 1.0 --lda 512 --ldb 1 --beta 1.0 --ldc 512
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 512 -n 512 -k 512 --alpha 1.0 --lda 512 --ldb 512 --beta 0.0 --ldc 512
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 1024 -k 1 --alpha 1.0 --lda 1024 --ldb 1 --beta 1.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 1024 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 512 -n 512 -k 1 --alpha 1.0 --lda 512 --ldb 1 --beta 1.0 --ldc 512
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 512 -n 512 -k 512 --alpha 1.0 --lda 512 --ldb 512 --beta 0.0 --ldc 512
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 1024 -k 1 --alpha 1.0 --lda 1024 --ldb 1 --beta 1.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 1024 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
