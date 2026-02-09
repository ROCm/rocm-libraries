#!/bin/bash
bench=${bench}
if [ ${#bench} -eq 0 ] || [ ! -f ${bench} ]; then
	echo "bench script path ${bench} not set or found, exit..."
	exit 1
else
	echo ">>" $(realpath $(ldd ${bench} | grep rocblas | awk '{print $3;}'))
fi

agent=${agent}
echo ">>agent_enumerator=${agent}"
if [ ${#agent} -eq 0 ] || [ ! -f ${agent} ]; then
        echo "rocm_agent_enumerator path ${agent} not set or found, exit..."
        exit 1
fi

flagsArg=""
echo ">> use un-packed-int8"

echo "-----------------------------------------------------------------------------------------------------"
echo "----------i8_r-i32_r-gemm_ex-------------------------------------------------------------------------"
echo "-----------------------------------------------------------------------------------------------------"
${bench} -f gemm_strided_batched_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA N --transposeB N -m 3025 -n 256 -k 64  --lda 3025 --ldb 64  --ldc 3025 --ldd 3025 --batch 64 ${flagsArg}
${bench} -f gemm_strided_batched_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA N --transposeB N -m 3025 -n 64  -k 64  --lda 3025 --ldb 64  --ldc 3025 --ldd 3025 --batch 64 ${flagsArg}
${bench} -f gemm_strided_batched_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA N --transposeB N -m 3025 -n 64  -k 256 --lda 3025 --ldb 256 --ldc 3025 --ldd 3025 --batch 64 ${flagsArg}
${bench} -f gemm_strided_batched_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA N --transposeB N -m 784  -n 512 -k 128 --lda 784  --ldb 128 --ldc 784  --ldd 784  --batch 64 ${flagsArg}
${bench} -f gemm_strided_batched_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA N --transposeB N -m 784  -n 128 -k 512 --lda 784  --ldb 512 --ldc 784  --ldd 784  --batch 64 ${flagsArg}

echo "-----------------------------------------------------------------------------------------------------"
echo "----------f32_r-gemm_ex------------------------------------------------------------------------------"
echo "-----------------------------------------------------------------------------------------------------"
${bench} -f gemm_strided_batched_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA N --transposeB N -m 3025 -n 256 -k 16  --lda 3025 --ldb 16  --ldc 3025 --ldd 3025 --batch 64
${bench} -f gemm_strided_batched_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA N --transposeB N -m 3025 -n 64  -k 16  --lda 3025 --ldb 16  --ldc 3025 --ldd 3025 --batch 64
${bench} -f gemm_strided_batched_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA N --transposeB N -m 3025 -n 64  -k 64  --lda 3025 --ldb 64  --ldc 3025 --ldd 3025 --batch 64
${bench} -f gemm_strided_batched_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA N --transposeB N -m 784  -n 512 -k 32  --lda 784  --ldb 32  --ldc 784  --ldd 784  --batch 64
${bench} -f gemm_strided_batched_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA N --transposeB N -m 784  -n 128 -k 128 --lda 784  --ldb 128 --ldc 784  --ldd 784  --batch 64
