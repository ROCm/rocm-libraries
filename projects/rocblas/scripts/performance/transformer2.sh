#!/bin/bash

./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 9520 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024 -i 36
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 8160 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024 -i 36
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 7200 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024 -i 36
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 3968 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024 -i 36
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 3840 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024 -i 36
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 10200 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024 -i 36
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 1024 -k 9520 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024 -i 36
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 1024 -k 8160 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024 -i 36
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 1024 -k 7200 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024 -i 36
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 1024 -k 3968 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024 -i 36
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 1024 -k 3840 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024 -i 36
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 1024 -k 10200 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024 -i 36
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 9520 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024 -i 36
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 8160 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024 -i 36
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 7200 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024 -i 36
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 3968 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024 -i 36
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 3840 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024 -i 36
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 10200 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 1024 -i 36
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA T --transposeB N -m 31 -n 31 -k 64 --alpha 1 --lda 131072 --stride_a 64 --ldb 131072 --stride_b 64 --beta 0 --ldc 31 --stride_c 961 --batch_count 2048 -i 12
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA T --transposeB N -m 30 -n 31 -k 64 --alpha 1 --lda 131072 --stride_a 64 --ldb 131072 --stride_b 64 --beta 0 --ldc 30 --stride_c 930 --batch_count 2048 -i 12
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA T --transposeB N -m 30 -n 30 -k 64 --alpha 1 --lda 131072 --stride_a 64 --ldb 131072 --stride_b 64 --beta 0 --ldc 30 --stride_c 900 --batch_count 2048 -i 12
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA T --transposeB N -m 17 -n 17 -k 64 --alpha 1 --lda 491520 --stride_a 64 --ldb 491520 --stride_b 64 --beta 0 --ldc 17 --stride_c 289 --batch_count 7680 -i 12
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA T --transposeB N -m 17 -n 15 -k 64 --alpha 1 --lda 491520 --stride_a 64 --ldb 491520 --stride_b 64 --beta 0 --ldc 17 --stride_c 255 --batch_count 7680 -i 12
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA T --transposeB N -m 15 -n 15 -k 64 --alpha 1 --lda 696320 --stride_a 64 --ldb 696320 --stride_b 64 --beta 0 --ldc 15 --stride_c 225 --batch_count 10880 -i 12
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA T --transposeB N -m 15 -n 15 -k 64 --alpha 1 --lda 491520 --stride_a 64 --ldb 491520 --stride_b 64 --beta 0 --ldc 15 --stride_c 225 --batch_count 7680 -i 12
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA T --transposeB N -m 15 -n 14 -k 64 --alpha 1 --lda 696320 --stride_a 64 --ldb 696320 --stride_b 64 --beta 0 --ldc 15 --stride_c 210 --batch_count 10880 -i 12
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA T --transposeB N -m 14 -n 14 -k 64 --alpha 1 --lda 696320 --stride_a 64 --ldb 696320 --stride_b 64 --beta 0 --ldc 14 --stride_c 196 --batch_count 10880 -i 12
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB N -m 64 -n 31 -k 31 --alpha 1 --lda 131072 --stride_a 64 --ldb 31 --stride_b 961 --beta 0 --ldc 64 --stride_c 1984 --batch_count 2048 -i 12
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB N -m 64 -n 31 -k 30 --alpha 1 --lda 131072 --stride_a 64 --ldb 30 --stride_b 930 --beta 0 --ldc 64 --stride_c 1984 --batch_count 2048 -i 12
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB N -m 64 -n 30 -k 30 --alpha 1 --lda 131072 --stride_a 64 --ldb 30 --stride_b 900 --beta 0 --ldc 64 --stride_c 1920 --batch_count 2048 -i 12
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB N -m 64 -n 17 -k 17 --alpha 1 --lda 491520 --stride_a 64 --ldb 17 --stride_b 289 --beta 0 --ldc 64 --stride_c 1088 --batch_count 7680 -i 12
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB N -m 64 -n 15 -k 17 --alpha 1 --lda 491520 --stride_a 64 --ldb 17 --stride_b 255 --beta 0 --ldc 64 --stride_c 960 --batch_count 7680 -i 12
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB N -m 64 -n 15 -k 15 --alpha 1 --lda 696320 --stride_a 64 --ldb 15 --stride_b 225 --beta 0 --ldc 64 --stride_c 960 --batch_count 10880 -i 12
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB N -m 64 -n 15 -k 15 --alpha 1 --lda 491520 --stride_a 64 --ldb 15 --stride_b 225 --beta 0 --ldc 64 --stride_c 960 --batch_count 7680 -i 12
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB N -m 64 -n 14 -k 15 --alpha 1 --lda 696320 --stride_a 64 --ldb 15 --stride_b 210 --beta 0 --ldc 64 --stride_c 896 --batch_count 10880 -i 12
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB N -m 64 -n 14 -k 14 --alpha 1 --lda 696320 --stride_a 64 --ldb 14 --stride_b 196 --beta 0 --ldc 64 --stride_c 896 --batch_count 10880 -i 12
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB T -m 64 -n 31 -k 31 --alpha 1 --lda 131072 --stride_a 64 --ldb 31 --stride_b 961 --beta 0 --ldc 64 --stride_c 1984 --batch_count 2048 -i 6
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB T -m 64 -n 30 -k 31 --alpha 1 --lda 131072 --stride_a 64 --ldb 30 --stride_b 930 --beta 0 --ldc 64 --stride_c 1920 --batch_count 2048 -i 6
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB T -m 64 -n 30 -k 30 --alpha 1 --lda 131072 --stride_a 64 --ldb 30 --stride_b 900 --beta 0 --ldc 64 --stride_c 1920 --batch_count 2048 -i 6
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB T -m 64 -n 17 -k 17 --alpha 1 --lda 491520 --stride_a 64 --ldb 17 --stride_b 289 --beta 0 --ldc 64 --stride_c 1088 --batch_count 7680 -i 6
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB T -m 64 -n 17 -k 15 --alpha 1 --lda 491520 --stride_a 64 --ldb 17 --stride_b 255 --beta 0 --ldc 64 --stride_c 1088 --batch_count 7680 -i 6
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB T -m 64 -n 15 -k 15 --alpha 1 --lda 696320 --stride_a 64 --ldb 15 --stride_b 225 --beta 0 --ldc 64 --stride_c 960 --batch_count 10880 -i 6
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB T -m 64 -n 15 -k 15 --alpha 1 --lda 491520 --stride_a 64 --ldb 15 --stride_b 225 --beta 0 --ldc 64 --stride_c 960 --batch_count 7680 -i 6
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB T -m 64 -n 15 -k 14 --alpha 1 --lda 696320 --stride_a 64 --ldb 15 --stride_b 210 --beta 0 --ldc 64 --stride_c 960 --batch_count 10880 -i 6
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB T -m 64 -n 14 -k 14 --alpha 1 --lda 696320 --stride_a 64 --ldb 14 --stride_b 196 --beta 0 --ldc 64 --stride_c 896 --batch_count 10880 -i 6
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB T -m 31 -n 64 -k 31 --alpha 1 --lda 31 --stride_a 961 --ldb 131072 --stride_b 64 --beta 0 --ldc 31 --stride_c 1984 --batch_count 2048 -i 6
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB T -m 30 -n 64 -k 31 --alpha 1 --lda 30 --stride_a 930 --ldb 131072 --stride_b 64 --beta 0 --ldc 30 --stride_c 1920 --batch_count 2048 -i 6
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB T -m 30 -n 64 -k 30 --alpha 1 --lda 30 --stride_a 900 --ldb 131072 --stride_b 64 --beta 0 --ldc 30 --stride_c 1920 --batch_count 2048 -i 6
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB T -m 17 -n 64 -k 17 --alpha 1 --lda 17 --stride_a 289 --ldb 491520 --stride_b 64 --beta 0 --ldc 17 --stride_c 1088 --batch_count 7680 -i 6
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB T -m 17 -n 64 -k 15 --alpha 1 --lda 17 --stride_a 255 --ldb 491520 --stride_b 64 --beta 0 --ldc 17 --stride_c 1088 --batch_count 7680 -i 6
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB T -m 15 -n 64 -k 15 --alpha 1 --lda 15 --stride_a 225 --ldb 696320 --stride_b 64 --beta 0 --ldc 15 --stride_c 960 --batch_count 10880 -i 6
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB T -m 15 -n 64 -k 15 --alpha 1 --lda 15 --stride_a 225 --ldb 491520 --stride_b 64 --beta 0 --ldc 15 --stride_c 960 --batch_count 7680 -i 6
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB T -m 15 -n 64 -k 14 --alpha 1 --lda 15 --stride_a 210 --ldb 696320 --stride_b 64 --beta 0 --ldc 15 --stride_c 960 --batch_count 10880 -i 6
./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA N --transposeB T -m 14 -n 64 -k 14 --alpha 1 --lda 14 --stride_a 196 --ldb 696320 --stride_b 64 --beta 0 --ldc 14 --stride_c 896 --batch_count 10880 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 4096 -n 9520 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 4096 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 4096 -n 8160 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 4096 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 4096 -n 7200 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 4096 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 4096 -n 3968 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 4096 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 4096 -n 3840 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 4096 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 4096 -n 10200 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 4096 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 9520 -k 4096 --alpha 1 --lda 4096 --ldb 4096 --beta 0 --ldc 1024 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 8160 -k 4096 --alpha 1 --lda 4096 --ldb 4096 --beta 0 --ldc 1024 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 7200 -k 4096 --alpha 1 --lda 4096 --ldb 4096 --beta 0 --ldc 1024 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 3968 -k 4096 --alpha 1 --lda 4096 --ldb 4096 --beta 0 --ldc 1024 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 3840 -k 4096 --alpha 1 --lda 4096 --ldb 4096 --beta 0 --ldc 1024 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 1024 -n 10200 -k 4096 --alpha 1 --lda 4096 --ldb 4096 --beta 0 --ldc 1024 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 4096 -n 1024 -k 9520 --alpha 1 --lda 4096 --ldb 1024 --beta 0 --ldc 4096 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 4096 -n 1024 -k 8160 --alpha 1 --lda 4096 --ldb 1024 --beta 0 --ldc 4096 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 4096 -n 1024 -k 7200 --alpha 1 --lda 4096 --ldb 1024 --beta 0 --ldc 4096 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 4096 -n 1024 -k 3968 --alpha 1 --lda 4096 --ldb 1024 --beta 0 --ldc 4096 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 4096 -n 1024 -k 3840 --alpha 1 --lda 4096 --ldb 1024 --beta 0 --ldc 4096 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 4096 -n 1024 -k 10200 --alpha 1 --lda 4096 --ldb 1024 --beta 0 --ldc 4096 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 4096 -k 9520 --alpha 1 --lda 1024 --ldb 4096 --beta 0 --ldc 1024 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 4096 -k 8160 --alpha 1 --lda 1024 --ldb 4096 --beta 0 --ldc 1024 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 4096 -k 7200 --alpha 1 --lda 1024 --ldb 4096 --beta 0 --ldc 1024 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 4096 -k 3968 --alpha 1 --lda 1024 --ldb 4096 --beta 0 --ldc 1024 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 4096 -k 3840 --alpha 1 --lda 1024 --ldb 4096 --beta 0 --ldc 1024 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 4096 -k 10200 --alpha 1 --lda 1024 --ldb 4096 --beta 0 --ldc 1024 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 4096 -n 9520 -k 1024 --alpha 1 --lda 4096 --ldb 1024 --beta 0 --ldc 4096 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 4096 -n 8160 -k 1024 --alpha 1 --lda 4096 --ldb 1024 --beta 0 --ldc 4096 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 4096 -n 7200 -k 1024 --alpha 1 --lda 4096 --ldb 1024 --beta 0 --ldc 4096 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 4096 -n 3968 -k 1024 --alpha 1 --lda 4096 --ldb 1024 --beta 0 --ldc 4096 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 4096 -n 3840 -k 1024 --alpha 1 --lda 4096 --ldb 1024 --beta 0 --ldc 4096 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 4096 -n 10200 -k 1024 --alpha 1 --lda 4096 --ldb 1024 --beta 0 --ldc 4096 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 9520 -k 4096 --alpha 1 --lda 1024 --ldb 4096 --beta 0 --ldc 1024 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 8160 -k 4096 --alpha 1 --lda 1024 --ldb 4096 --beta 0 --ldc 1024 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 7200 -k 4096 --alpha 1 --lda 1024 --ldb 4096 --beta 0 --ldc 1024 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 3968 -k 4096 --alpha 1 --lda 1024 --ldb 4096 --beta 0 --ldc 1024 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 3840 -k 4096 --alpha 1 --lda 1024 --ldb 4096 --beta 0 --ldc 1024 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 10200 -k 4096 --alpha 1 --lda 1024 --ldb 4096 --beta 0 --ldc 1024 -i 6
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 42720 -n 9520 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 42720 -i 1
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 42720 -n 7200 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 42720 -i 1
./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 42720 -n 3968 -k 1024 --alpha 1 --lda 1024 --ldb 1024 --beta 0 --ldc 42720 -i 1
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 42720 -k 9520 --alpha 1 --lda 1024 --ldb 42720 --beta 0 --ldc 1024 -i 1
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 42720 -k 7200 --alpha 1 --lda 1024 --ldb 42720 --beta 0 --ldc 1024 -i 1
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB T -m 1024 -n 42720 -k 3968 --alpha 1 --lda 1024 --ldb 42720 --beta 0 --ldc 1024 -i 1
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 9520 -k 42720 --alpha 1 --lda 1024 --ldb 42720 --beta 0 --ldc 1024 -i 1
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 7200 -k 42720 --alpha 1 --lda 1024 --ldb 42720 --beta 0 --ldc 1024 -i 1
./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 1024 -n 3968 -k 42720 --alpha 1 --lda 1024 --ldb 42720 --beta 0 --ldc 1024 -i 1
