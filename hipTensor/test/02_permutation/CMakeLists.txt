###############################################################################
 #
 # MIT License
 #
 # Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a copy
 # of this software and associated documentation files (the "Software"), to deal
 # in the Software without restriction, including without limitation the rights
 # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 # copies of the Software, and to permit persons to whom the Software is
 # furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in
 # all copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 # THE SOFTWARE.
 #
 ###############################################################################
 set(PermutationCommonSources
     ${HIPTENSOR_COMMON_TEST_SOURCES}
     ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_resource.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/permutation_test.cpp
     )

 set(ElementwiseBinaryOpCommonSources
     ${HIPTENSOR_COMMON_TEST_SOURCES}
     ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_resource.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_binary_op_test.cpp
     )

 set(ElementwiseTrinaryOpCommonSources
     ${HIPTENSOR_COMMON_TEST_SOURCES}
     ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_resource.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_trinary_op_test.cpp
     )

#  tests
set (CpuTestSources
    ${PermutationCommonSources}
    ${CMAKE_CURRENT_SOURCE_DIR}/permutation_cpu_impl_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_binary_cpu_impl_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_trinary_cpu_impl_test.cpp
    )

add_hiptensor_unit_test(elementwise_cpu_test ${CpuTestSources})

# Rank 2 tests
set (PermutationRank2TestSources ${PermutationCommonSources}
     ${CMAKE_CURRENT_SOURCE_DIR}/rank2_permutation_test.cpp)
set (PermutationRank2TestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/permutation/rank2_test_params.yaml)
add_hiptensor_test(rank2_permutation_test ${PermutationRank2TestConfig}  ${PermutationRank2TestSources})

set (ElementwiseBinaryOpRank2TestSources ${ElementwiseBinaryOpCommonSources}
     ${CMAKE_CURRENT_SOURCE_DIR}/rank2_elementwise_binary_op_test.cpp)
set (ElementwiseBinaryOpRank2TestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/binary_op/rank2_binary_op_test_params.yaml)
add_hiptensor_test(rank2_elementwise_binary_op_test ${ElementwiseBinaryOpRank2TestConfig}  ${ElementwiseBinaryOpRank2TestSources})

set (ElementwiseTrinaryOpRank2TestSources ${ElementwiseTrinaryOpCommonSources}
     ${CMAKE_CURRENT_SOURCE_DIR}/rank2_elementwise_trinary_op_test.cpp)
set (ElementwiseTrinaryOpRank2TestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/trinary_op/rank2_trinary_op_test_params.yaml)
add_hiptensor_test(rank2_elementwise_trinary_op_test ${ElementwiseTrinaryOpRank2TestConfig}  ${ElementwiseTrinaryOpRank2TestSources})

# Rank 3 tests
set (PermutationRank3TestSources ${PermutationCommonSources}
	 ${CMAKE_CURRENT_SOURCE_DIR}/rank3_permutation_test.cpp)
set (PermutationRank3TestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/permutation/rank3_test_params.yaml)
add_hiptensor_test(rank3_permutation_test ${PermutationRank3TestConfig}  ${PermutationRank3TestSources})

set (ElementwiseBinaryOpRank3TestSources ${ElementwiseBinaryOpCommonSources}
	 ${CMAKE_CURRENT_SOURCE_DIR}/rank3_elementwise_binary_op_test.cpp)
set (ElementwiseBinaryOpRank3TestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/binary_op/rank3_binary_op_test_params.yaml)
add_hiptensor_test(rank3_elementwise_binary_op_test ${ElementwiseBinaryOpRank3TestConfig}  ${ElementwiseBinaryOpRank3TestSources})

set (ElementwiseTrinaryOpRank3TestSources ${ElementwiseTrinaryOpCommonSources}
     ${CMAKE_CURRENT_SOURCE_DIR}/rank3_elementwise_trinary_op_test.cpp)
set (ElementwiseTrinaryOpRank3TestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/trinary_op/rank3_trinary_op_test_params.yaml)
add_hiptensor_test(rank3_elementwise_trinary_op_test ${ElementwiseTrinaryOpRank3TestConfig}  ${ElementwiseTrinaryOpRank3TestSources})

# Rank 4 tests
set (PermutationRank4TestSources ${PermutationCommonSources}
	 ${CMAKE_CURRENT_SOURCE_DIR}/rank4_permutation_test.cpp)
set (PermutationRank4TestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/permutation/rank4_test_params.yaml)
add_hiptensor_test(rank4_permutation_test ${PermutationRank4TestConfig}  ${PermutationRank4TestSources})

set (ElementwiseBinaryOpRank4TestSources ${ElementwiseBinaryOpCommonSources}
	 ${CMAKE_CURRENT_SOURCE_DIR}/rank4_elementwise_binary_op_test.cpp)
set (ElementwiseBinaryOpRank4TestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/binary_op/rank4_binary_op_test_params.yaml)
add_hiptensor_test(rank4_elementwise_binary_op_test ${ElementwiseBinaryOpRank4TestConfig}  ${ElementwiseBinaryOpRank4TestSources})

set (ElementwiseTrinaryOpRank4TestSources ${ElementwiseTrinaryOpCommonSources}
     ${CMAKE_CURRENT_SOURCE_DIR}/rank4_elementwise_trinary_op_test.cpp)
set (ElementwiseTrinaryOpRank4TestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/trinary_op/rank4_trinary_op_test_params.yaml)
add_hiptensor_test(rank4_elementwise_trinary_op_test ${ElementwiseTrinaryOpRank4TestConfig}  ${ElementwiseTrinaryOpRank4TestSources})

# Rank 5 tests
set (PermutationRank5TestSources ${PermutationCommonSources}
	 ${CMAKE_CURRENT_SOURCE_DIR}/rank5_permutation_test.cpp)
set (PermutationRank5TestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/permutation/rank5_test_params.yaml)
add_hiptensor_test(rank5_permutation_test ${PermutationRank5TestConfig}  ${PermutationRank5TestSources})

set (ElementwiseBinaryOpRank5TestSources ${ElementwiseBinaryOpCommonSources}
	 ${CMAKE_CURRENT_SOURCE_DIR}/rank5_elementwise_binary_op_test.cpp)
set (ElementwiseBinaryOpRank5TestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/binary_op/rank5_binary_op_test_params.yaml)
add_hiptensor_test(rank5_elementwise_binary_op_test ${ElementwiseBinaryOpRank5TestConfig}  ${ElementwiseBinaryOpRank5TestSources})

set (ElementwiseTrinaryOpRank5TestSources ${ElementwiseTrinaryOpCommonSources}
     ${CMAKE_CURRENT_SOURCE_DIR}/rank5_elementwise_trinary_op_test.cpp)
set (ElementwiseTrinaryOpRank5TestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/trinary_op/rank5_trinary_op_test_params.yaml)
add_hiptensor_test(rank5_elementwise_trinary_op_test ${ElementwiseTrinaryOpRank5TestConfig}  ${ElementwiseTrinaryOpRank5TestSources})

# Rank 6 tests
set (PermutationRank6TestSources ${PermutationCommonSources}
	 ${CMAKE_CURRENT_SOURCE_DIR}/rank6_permutation_test.cpp)
set (PermutationRank6TestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/permutation/rank6_test_params.yaml)
add_hiptensor_test(rank6_permutation_test ${PermutationRank6TestConfig}  ${PermutationRank6TestSources})

set (ElementwiseBinaryOpRank6TestSources ${ElementwiseBinaryOpCommonSources}
	 ${CMAKE_CURRENT_SOURCE_DIR}/rank6_elementwise_binary_op_test.cpp)
set (ElementwiseBinaryOpRank6TestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/binary_op/rank6_binary_op_test_params.yaml)
add_hiptensor_test(rank6_elementwise_binary_op_test ${ElementwiseBinaryOpRank6TestConfig}  ${ElementwiseBinaryOpRank6TestSources})

set (ElementwiseTrinaryOpRank6TestSources ${ElementwiseTrinaryOpCommonSources}
     ${CMAKE_CURRENT_SOURCE_DIR}/rank6_elementwise_trinary_op_test.cpp)
set (ElementwiseTrinaryOpRank6TestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/trinary_op/rank6_trinary_op_test_params.yaml)
add_hiptensor_test(rank6_elementwise_trinary_op_test ${ElementwiseTrinaryOpRank6TestConfig}  ${ElementwiseTrinaryOpRank6TestSources})

