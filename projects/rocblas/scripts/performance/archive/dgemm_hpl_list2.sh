#!/bin/bash

# number of iterations defaults to 1
iters=${1:-1}

bench=./rocblas-bench
if [ ! -f ${bench} ]; then
	echo ${bench} not found, exit...
	exit 1
else
	echo ">>" $(realpath $(ldd ${bench} | grep rocblas | awk '{print $3;}'))
fi
for i in {384..44928..384}; do
	${bench} -f gemm -r d --transposeA N --transposeB T \
	-m ${i} -n ${i} -k 384 --lda ${i} --ldb ${i} --ldc 44928 \
	--alpha -1 --beta 1 -i ${iters} \
	--initialization trig_float 2>&1 | egrep '^[NT],[NT],|fault'
done
