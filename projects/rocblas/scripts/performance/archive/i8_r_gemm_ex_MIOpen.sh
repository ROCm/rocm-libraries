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
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA N --transposeB N -m 50176 -n 128  -k 256  --lda 50176 --ldb 256  --ldc 50176 --ldd 50176 ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA N --transposeB N -m 50176 -n 512  -k 256  --lda 50176 --ldb 256  --ldc 50176 --ldd 50176 ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA N --transposeB N -m 12544 -n 1024 -k 512  --lda 12544 --ldb 512  --ldc 12544 --ldd 12544 ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA N --transposeB N -m 12544 -n 256  -k 512  --lda 12544 --ldb 512  --ldc 12544 --ldd 12544 ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA N --transposeB N -m 3136  -n 2048 -k 1024 --lda 3136  --ldb 1024 --ldc 3136  --ldd 3136  ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA N --transposeB N -m 3136  -n 512  -k 1024 --lda 3136  --ldb 1024 --ldc 3136  --ldd 3136  ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA N --transposeB N -m 12544 -n 256  -k 1024 --lda 12544 --ldb 1024 --ldc 12544 --ldd 12544 ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA N --transposeB N -m 12544 -n 1024 -k 256  --lda 12544 --ldb 256  --ldc 12544 --ldd 12544 ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA N --transposeB N -m 3136  -n 2048 -k 512  --lda 3136  --ldb 512  --ldc 3136  --ldd 3136  ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA N --transposeB N -m 3136  -n 512  -k 2048 --lda 3136  --ldb 2048 --ldc 3136  --ldd 3136  ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA N --transposeB N -m 50176 -n 63   -k 784  --lda 50176 --ldb 784  --ldc 50176 --ldd 50176 ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA N --transposeB N -m 3025  -n 64   -k 576  --lda 3025  --ldb 576  --ldc 3025  --ldd 3025  ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA N --transposeB N -m 784   -n 128  -k 1152 --lda 784   --ldb 1152 --ldc 784   --ldd 784   ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA N --transposeB N -m 196   -n 256  -k 2304 --lda 196   --ldb 2304 --ldc 196   --ldd 196   ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA N --transposeB N -m 49    -n 512  -k 4608 --lda 49    --ldb 4608 --ldc 49    --ldd 49    ${flagsArg}

${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA T --transposeB N -m 50176 -n 128  -k 256  --lda 256  --ldb 256  --ldc 50176 --ldd 50176 ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA T --transposeB N -m 50176 -n 512  -k 256  --lda 256  --ldb 256  --ldc 50176 --ldd 50176 ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA T --transposeB N -m 12544 -n 1024 -k 512  --lda 512  --ldb 512  --ldc 12544 --ldd 12544 ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA T --transposeB N -m 12544 -n 256  -k 512  --lda 512  --ldb 512  --ldc 12544 --ldd 12544 ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA T --transposeB N -m 3136  -n 2048 -k 1024 --lda 1024 --ldb 1024 --ldc 3136  --ldd 3136  ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA T --transposeB N -m 3136  -n 512  -k 1024 --lda 1024 --ldb 1024 --ldc 3136  --ldd 3136  ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA T --transposeB N -m 3025  -n 256  -k 64   --lda 64   --ldb 64   --ldc 3025  --ldd 3025  ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA T --transposeB N -m 3025  -n 64   -k 64   --lda 64   --ldb 64   --ldc 3025  --ldd 3025  ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA T --transposeB N -m 3025  -n 64   -k 256  --lda 256  --ldb 256  --ldc 3025  --ldd 3025  ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA T --transposeB N -m 784   -n 512  -k 128  --lda 128  --ldb 128  --ldc 784   --ldd 784   ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA T --transposeB N -m 784   -n 128  -k 512  --lda 512  --ldb 512  --ldc 784   --ldd 784   ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA T --transposeB N -m 12544 -n 256  -k 1024 --lda 1024 --ldb 1024 --ldc 12544 --ldd 12544 ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA T --transposeB N -m 12544 -n 1024 -k 256  --lda 256  --ldb 256  --ldc 12544 --ldd 12544 ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA T --transposeB N -m 3136  -n 2048 -k 512  --lda 512  --ldb 512  --ldc 3136  --ldd 3136  ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA T --transposeB N -m 3136  -n 512  -k 2048 --lda 2048 --ldb 2048 --ldc 3136  --ldd 3136  ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA T --transposeB N -m 50176 -n 63   -k 784  --lda 784  --ldb 784  --ldc 50176 --ldd 50176 ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA T --transposeB N -m 3025  -n 64   -k 576  --lda 576  --ldb 576  --ldc 3025  --ldd 3025  ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA T --transposeB N -m 784   -n 128  -k 1152 --lda 1152 --ldb 1152 --ldc 784   --ldd 784   ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA T --transposeB N -m 196   -n 256  -k 2304 --lda 2304 --ldb 2304 --ldc 196   --ldd 196   ${flagsArg}
${bench} -f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --transposeA T --transposeB N -m 49    -n 512  -k 4608 --lda 4608 --ldb 4608 --ldc 49    --ldd 49    ${flagsArg}

echo "-----------------------------------------------------------------------------------------------------"
echo "----------f32_r-gemm_ex------------------------------------------------------------------------------"
echo "-----------------------------------------------------------------------------------------------------"
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA N --transposeB N -m 50176 -n 128  -k 64   --lda 50176 --ldb 64   --ldc 50176 --ldd 50176
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA N --transposeB N -m 50176 -n 512  -k 64   --lda 50176 --ldb 64   --ldc 50176 --ldd 50176
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA N --transposeB N -m 12544 -n 1024 -k 128  --lda 12544 --ldb 128  --ldc 12544 --ldd 12544
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA N --transposeB N -m 12544 -n 256  -k 128  --lda 12544 --ldb 128  --ldc 12544 --ldd 12544
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA N --transposeB N -m 3136  -n 2048 -k 256  --lda 3136  --ldb 256  --ldc 3136  --ldd 3136
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA N --transposeB N -m 3136  -n 512  -k 256  --lda 3136  --ldb 256  --ldc 3136  --ldd 3136
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA N --transposeB N -m 12544 -n 256  -k 256  --lda 12544 --ldb 256  --ldc 12544 --ldd 12544
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA N --transposeB N -m 12544 -n 1024 -k 64   --lda 12544 --ldb 64   --ldc 12544 --ldd 12544
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA N --transposeB N -m 3136  -n 2048 -k 128  --lda 3136  --ldb 128  --ldc 3136  --ldd 3136
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA N --transposeB N -m 3136  -n 512  -k 512  --lda 3136  --ldb 512  --ldc 3136  --ldd 3136
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA N --transposeB N -m 50176 -n 63   -k 196  --lda 50176 --ldb 196  --ldc 50176 --ldd 50176
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA N --transposeB N -m 3025  -n 64   -k 144  --lda 3025  --ldb 144  --ldc 3025  --ldd 3025
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA N --transposeB N -m 784   -n 128  -k 288  --lda 784   --ldb 288  --ldc 784   --ldd 784
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA N --transposeB N -m 196   -n 256  -k 576  --lda 196   --ldb 576  --ldc 196   --ldd 196
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA N --transposeB N -m 49    -n 512  -k 1024 --lda 49    --ldb 1024 --ldc 49    --ldd 49

${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA T --transposeB N -m 50176 -n 128  -k 64   --lda 64   --ldb 64   --ldc 50176 --ldd 50176
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA T --transposeB N -m 50176 -n 512  -k 64   --lda 64   --ldb 64   --ldc 50176 --ldd 50176
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA T --transposeB N -m 12544 -n 1024 -k 128  --lda 128  --ldb 128  --ldc 12544 --ldd 12544
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA T --transposeB N -m 12544 -n 256  -k 128  --lda 128  --ldb 128  --ldc 12544 --ldd 12544
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA T --transposeB N -m 3136  -n 2048 -k 256  --lda 256  --ldb 256  --ldc 3136  --ldd 3136
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA T --transposeB N -m 3136  -n 512  -k 256  --lda 256  --ldb 256  --ldc 3136  --ldd 3136
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA T --transposeB N -m 3025  -n 256  -k 16   --lda 16   --ldb 16   --ldc 3025  --ldd 3025
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA T --transposeB N -m 3025  -n 64   -k 16   --lda 16   --ldb 16   --ldc 3025  --ldd 3025
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA T --transposeB N -m 3025  -n 64   -k 64   --lda 64   --ldb 64   --ldc 3025  --ldd 3025
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA T --transposeB N -m 784   -n 512  -k 32   --lda 32   --ldb 32   --ldc 784   --ldd 784
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA T --transposeB N -m 784   -n 128  -k 128  --lda 128  --ldb 128  --ldc 784   --ldd 784
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA T --transposeB N -m 12544 -n 256  -k 256  --lda 256  --ldb 256  --ldc 12544 --ldd 12544
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA T --transposeB N -m 12544 -n 1024 -k 64   --lda 64   --ldb 64   --ldc 12544 --ldd 12544
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA T --transposeB N -m 3136  -n 2048 -k 128  --lda 128  --ldb 128  --ldc 3136  --ldd 3136
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA T --transposeB N -m 3136  -n 512  -k 512  --lda 512  --ldb 512  --ldc 3136  --ldd 3136
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA T --transposeB N -m 50176 -n 63   -k 196  --lda 196  --ldb 196  --ldc 50176 --ldd 50176
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA T --transposeB N -m 3025  -n 64   -k 144  --lda 144  --ldb 144  --ldc 3025  --ldd 3025
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA T --transposeB N -m 784   -n 128  -k 288  --lda 288  --ldb 288  --ldc 784   --ldd 784
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA T --transposeB N -m 196   -n 256  -k 576  --lda 576  --ldb 576  --ldc 196   --ldd 196
${bench} -f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --transposeA T --transposeB N -m 49    -n 512  -k 1024 --lda 1024 --ldb 1024 --ldc 49    --ldd 49
