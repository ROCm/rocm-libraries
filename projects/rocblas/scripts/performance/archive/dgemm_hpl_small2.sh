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

extra="--initialization trig_float"
#extra="-v 1"

lda=44160 ldb=384 ldc=44160
n=192 k=192
for i in {384..44160..384}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done
n=96 k=96
for i in {384..44160..384}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done
for i in {288..44064..384}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done
for i in {96..43872..384}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done
n=48 k=48
for i in {384..44160..384}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done
for i in {336..44112..384}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done
for i in {240..44016..384}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done
for i in {144..43920..384}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done
for i in {48..43824..384}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done

#lda=44160,ldb=384,ldc=44160
#n=192,k=192     cnt=115 m={384..44160..384}
#n=96,k=96       cnt=345 m={384..44160..384},{288..44064..384},{96..43872..384}
#n=48,k=48       cnt=575 m={384..44160..384},{336..44112..384},{240..44016..384},
#                          {144..43920..384},{48..43824..384}
