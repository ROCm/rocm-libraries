#!/bin/bash

./rocblas-bench -f gemm -r f32_r --transposeA T --transposeB N -m 256 -n 1 -k 4 --alpha 1.0 --lda 4 --ldb 4 --beta 1.0 --ldc 256
