#!/bin/bash

#poor
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 512 -n 512 -k 3968 --alpha 1.0 --lda 512 --ldb 512 --beta 0.0 --ldc 512
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 512 -n 512 -k 4005 --alpha 1.0 --lda 512 --ldb 512 --beta 0.0 --ldc 512
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 512 -n 1350 -k 2048 --alpha 1.0 --lda 512 --ldb 2048 --beta 0.0 --ldc 512
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 512 -n 4012 -k 2048 --alpha 1.0 --lda 512 --ldb 2048 --beta 0.0 --ldc 512
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 2048 -n 560 -k 512 --alpha 1.0 --lda 2048 --ldb 512 --beta 0.0 --ldc 2048

#medium
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 4096 -k 7200 --alpha 1 --lda 1024 --ldb 4096 --beta 0 --ldc 1024 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 4096 -n 1024 -k 3968 --alpha 1 --lda 4096 --ldb 1024 --beta 0 --ldc 4096 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 4096 -n 1024 -k 3840 --alpha 1 --lda 4096 --ldb 1024 --beta 0 --ldc 4096 -i 6

#good
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 42720 -k 9520 --alpha 1 --lda 1024 --ldb 42720 --beta 0 --ldc 1024 -i 1
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 4096 -n 9520 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 4096 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 9520 -k 4096 --alpha 1 --lda 1024 --ldb 4096 --beta 0 --ldc 1024 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 4096 -n 8160 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 4096 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 7200 -k 4096 --alpha 1 --lda 1024 --ldb 4096 --beta 0 --ldc 1024 -i 6
