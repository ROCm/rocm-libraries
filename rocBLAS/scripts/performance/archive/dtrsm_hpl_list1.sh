#!/bin/bash
bench=./rocblas-bench
if [ ! -f ${bench} ]; then
	echo ${bench} not found, exit...
	exit 1
else
	echo ">>" $(realpath $(ldd ${bench} | grep rocblas | awk '{print $3;}'))
fi

for i in {256..2048..256}; do
	${bench} -f trsm -r d --side R --uplo L --transposeA T --diag U \
	-m ${i} -n 256 --lda 256 --ldb ${i} \
	--initialization trig_float \
	--alpha 1 -i 1 2>&1 | egrep '[LR],[UL],[NT],[UN]|fault'
done

for i in {1..1281..256}; do
	${bench} -f trsm -r d --side R --uplo L --transposeA T --diag U \
	-m ${i} -n 256 --lda 256 --ldb ${i} \
	--initialization trig_float \
	--alpha 1 -i 1 2>&1 | egrep '[LR],[UL],[NT],[UN]|fault'
done

for i in {384..3072..384}; do
	${bench} -f trsm -r d --side R --uplo L --transposeA T --diag U \
	-m ${i} -n 384 --lda 384 --ldb ${i} \
	--initialization trig_float \
	--alpha 1 -i 1 2>&1 | egrep '[LR],[UL],[NT],[UN]|fault'
done

for i in {1..1153..384}; do
	${bench} -f trsm -r d --side R --uplo L --transposeA T --diag U \
	-m ${i} -n 384 --lda 384 --ldb ${i} \
	--initialization trig_float \
	--alpha 1 -i 1 2>&1 | egrep '[LR],[UL],[NT],[UN]|fault'
done
