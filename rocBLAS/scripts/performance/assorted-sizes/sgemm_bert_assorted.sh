#!/bin/bash

#poor
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 768 -n 1280 -k 768 --alpha 1 --lda 768 --ldb 768 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 768 -n 320 -k 30522 --alpha 1 --lda 768 --ldb 30522 --beta 0 --ldc 768
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 80 -k 30522 --alpha 1.0 --lda 1024 --ldb 30522 --beta 0.0 --ldc 1024


#medium
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 1024 -n 2048 -k 30528 --alpha 1.0 --lda 1024 --ldb 30528 --beta 0.0 --ldc 1024 -i 1
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 3072 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 1024
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 3072 -n 512 -k 1024 --alpha 1.0 --lda 1024 --ldb 1024 --beta 0.0 --ldc 3072
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 3072 -n 1024 -k 2048 --alpha 1.0 --lda 3072 --ldb 1024 --beta 0.0 --ldc 3072
./rocblas-bench -f gemm -r s --transposeA T --transposeB N -m 1024 -n 2048 -k 4096 --alpha 1.0 --lda 4096 --ldb 4096 --beta 0.0 --ldc 1024 -i 24

#good
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 1024 -n 30528 -k 4096 --alpha 1.0 --lda 1024 --ldb 30528 --beta 0.0 --ldc 1024 -i 1
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 1024 -n 30528 -k 2048 --alpha 1.0 --lda 1024 --ldb 30528 --beta 0.0 --ldc 1024 -i 1
./rocblas-bench -f gemm -r s --transposeA N --transposeB N -m 4096 -n 4096 -k 1024 --alpha 1.0 --lda 4096 --ldb 1024 --beta 0.0 --ldc 4096 -i 24
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 3072 -n 3072 -k 1024 --alpha 1.0 --lda 3072 --ldb 1024 --beta 0.0 --ldc 3072
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 3072 -n 2048 -k 1024 --alpha 1.0 --lda 3072 --ldb 1024 --beta 0.0 --ldc 3072
./rocblas-bench -f gemm -r s --transposeA N --transposeB T -m 1024 -n 4096 -k 4096 --alpha 1.0 --lda 1024 --ldb 4096 --beta 0.0 --ldc 1024 -i 24

