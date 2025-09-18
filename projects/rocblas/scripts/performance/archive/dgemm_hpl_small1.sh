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

lda=44288 ldb=256 ldc=44288
n=124 k=132
for i in {256..44288..256}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done
n=66 k=66
for i in {256..44288..256}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done
for i in {190..44222..256}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done
n=58 k=66
for i in {256..44288..256}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done
for i in {58..44090..256}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done
n=30 k=36
for i in {256..44288..256}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done
for i in {220..44252..256}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done
for i in {154..44186..256}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done
for i in {88..44120..256}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done
n=128 k=128
for i in {256..44288..256}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done
n=64 k=64
for i in {256..44288..256}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done
for i in {192..44244..256}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done
for i in {64..44096..256}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done
n=32 k=32
for i in {256..44288..256}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done
for i in {224..44256..256}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done
for i in {160..44192..256}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done
for i in {96..44128..256}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done
for i in {32..44064..256}; do
	${bench} -f gemm -r d --transposeA N --transposeB N \
	-m ${i} -n ${n} -k ${k} --lda ${lda} --ldb ${ldb} --ldc ${ldc} \
	--alpha -1 --beta 1 -i ${iters} \
	${extra} 2>&1 | egrep '^[NT],[NT],|fault'
done

#lda=44288,ldb=256,ldc=44288
#n=124,k=132     cnt=173 m={256..44288..256}
#n=66,k=66       cnt=346 m={256..44288..256},{190..44222..256}
#n=58,k=66       cnt=346 m={256..44288..256},{58..44090..256}
#n=30,k=36       cnt=692 m={256..44288..256},{220..44252..256},{154..44186..256},
#                          {88..44120..256}
#n=128,k=128     cnt=173 m={256..44288..256}
#n=64,k=64       cnt=519 m={256..44288..256},{192..44224..256},{64..44096..256}
#n=32,k=32       cnt=765 m={256..44288..256},{224..44256..256},{160..44192..256},
#                          {96..44128..256},{32..44064..256}
#subtotal=3114
