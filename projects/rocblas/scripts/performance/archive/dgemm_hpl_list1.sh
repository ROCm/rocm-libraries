#!/bin/bash
bench=./rocblas-bench
if [ ! -f ${bench} ]; then
	echo ${bench} not found, exit...
	exit 1
else
	echo ">>" $(realpath $(ldd ${bench} | grep rocblas | awk '{print $3;}'))
fi
for i in {3144..45000..384}; do
	${bench} -f gemm -r d --transposeA N --transposeB T \
	-m ${i} -n ${i} -k 384 --lda ${i} --ldb ${i} --ldc 45000 \
	--alpha 1 --beta 1 -i 1 \
        --initialization trig_float 2>&1 | egrep '^[NT],[NT],|fault'
done
