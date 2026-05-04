#!/bin/bash

EXECUTABLE="/opt/rocm/bin/rocblas-test"

BATCHED="_batched"
STRIDED_BATCHED="_strided_batched"
EX="_ex"

#---- AUXILIARY -----

TESTS="half_operators complex_operators helper_utilities check_numerics_vector check_numerics_matrix check_numerics_matrix_batched set_get_pointer_mode set_get_atomics_mode logging set_get_vector set_get_vector_async set_get_matrix set_get_matrix_async"

for TEST in $TESTS
do
	$EXECUTABLE --gtest_filter=*$TEST*quick* 2>&1 | tee $TEST.txt
done

#----- L1 BLAS -----

FUNCTIONS="asum axpy copy dot dotc iamax iamin nrm2 rot rotg rotm rotmg scal swap"

for FUNCTION in $FUNCTIONS
do
	FILTER="*$FUNCTION*quick*-*$BATCHED*:*$EX*"
	OUT_FILE="$FUNCTION.quick.txt"
	$EXECUTABLE --gtest_filter=$FILTER 2>&1 | tee $OUT_FILE

	FILTER="*$FUNCTION$BATCHED*quick*-*$EX*"
	OUT_FILE="$FUNCTION$BATCHED.quick.txt"
	$EXECUTABLE --gtest_filter=$FILTER 2>&1 | tee $OUT_FILE

	FILTER="*$FUNCTION$STRIDED_BATCHED*quick*-*$EX*"
	OUT_FILE="$FUNCTION$STRIDED_BATCHED.quick.txt"
	$EXECUTABLE --gtest_filter=$FILTER 2>&1 | tee $OUT_FILE
done

#----- L1 BLAS_EX -----

FUNCTIONS_EX="axpy dot dotc nrm2 rot scal"

for FUNCTION in $FUNCTIONS_EX
do
	FILTER="*$FUNCTION$EX*quick*-*$BATCHED*"
	OUT_FILE="$FUNCTION$EX.quick.txt"
	$EXECUTABLE --gtest_filter=$FILTER 2>&1 | tee $OUT_FILE

	FILTER="*$FUNCTION$BATCHED$EX*quick*"
	OUT_FILE="$FUNCTION$BATCHED$EX.quick.txt"
	$EXECUTABLE --gtest_filter=$FILTER 2>&1 | tee $OUT_FILE

	FILTER="*$FUNCTION$STRIDED_BATCHED$EX*quick*"
	OUT_FILE="$FUNCTION$STRIDED_BATCHED$EX.quick.txt"
	$EXECUTABLE --gtest_filter=$FILTER 2>&1 | tee $OUT_FILE
done

#----- L2 BLAS -----

FUNCTIONS="trsv gbmv gemv hbmv hemv her her2 hpmv hpr hpr2 trmv tpmv tbmv tbsv ger geru gerc spr spr2 syr syr2 sbmv spmv symv"

for FUNCTION in $FUNCTIONS
do
        FILTER="*$FUNCTION*quick*-*$BATCHED*:*$EX*"
        OUT_FILE="$FUNCTION.quick.txt"
        $EXECUTABLE --gtest_filter=$FILTER 2>&1 | tee $OUT_FILE

        FILTER="*$FUNCTION$BATCHED*quick*-*$EX*"
        OUT_FILE="$FUNCTION$BATCHED.quick.txt"
        $EXECUTABLE --gtest_filter=$FILTER 2>&1 | tee $OUT_FILE

        FILTER="*$FUNCTION$STRIDED_BATCHED*quick*-*$EX*"
        OUT_FILE="$FUNCTION$STRIDED_BATCHED.quick.txt"
        $EXECUTABLE --gtest_filter=$FILTER 2>&1 | tee $OUT_FILE
done
