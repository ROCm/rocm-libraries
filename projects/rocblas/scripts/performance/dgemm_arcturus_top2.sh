#!/bin/bash

./rocblas-bench -f gemm -r d --transposeA N --transposeB N -m 63360 -n 62977 -k 384 --alpha -1.0 --lda 63360 --ldb 384 --beta 1.0 --ldc 63360 --initialization trig_float
./rocblas-bench -f gemm -r d --transposeA N --transposeB N -m 63744 -n 63361 -k 384 --alpha -1.0 --lda 63744 --ldb 384 --beta 1.0 --ldc 63744 --initialization trig_float
