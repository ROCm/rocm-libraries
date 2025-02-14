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
     ${CMAKE_CURRENT_SOURCE_DIR}/permutation_resource.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/permutation_test.cpp
     )

#  tests
set (PermutationTestSources
    ${PermutationCommonSources}
    ${CMAKE_CURRENT_SOURCE_DIR}/permutation_cpu_impl_test.cpp
    )

add_hiptensor_unit_test(permutation_cpu_test ${PermutationTestSources})

# Rank 2 tests
set (PermutationRank2TestSources ${PermutationCommonSources}
     ${CMAKE_CURRENT_SOURCE_DIR}/rank2_permutation_test.cpp)
set (PermutationRank2TestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/rank2_test_params.yaml)
add_hiptensor_test(rank2_permutation_test ${PermutationRank2TestConfig}  ${PermutationRank2TestSources})

# Rank 3 tests
set (PermutationRank3TestSources ${PermutationCommonSources}
     ${CMAKE_CURRENT_SOURCE_DIR}/rank3_permutation_test.cpp)
set (PermutationRank3TestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/rank3_test_params.yaml)
add_hiptensor_test(rank3_permutation_test ${PermutationRank3TestConfig}  ${PermutationRank3TestSources})

# Rank 4 tests
set (PermutationRank4TestSources ${PermutationCommonSources}
     ${CMAKE_CURRENT_SOURCE_DIR}/rank4_permutation_test.cpp)
set (PermutationRank4TestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/rank4_test_params.yaml)
add_hiptensor_test(rank4_permutation_test ${PermutationRank4TestConfig}  ${PermutationRank4TestSources})

# Rank 5 tests
set (PermutationRank5TestSources ${PermutationCommonSources}
     ${CMAKE_CURRENT_SOURCE_DIR}/rank5_permutation_test.cpp)
set (PermutationRank5TestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/rank5_test_params.yaml)
add_hiptensor_test(rank5_permutation_test ${PermutationRank5TestConfig}  ${PermutationRank5TestSources})

# Rank 6 tests
set (PermutationRank6TestSources ${PermutationCommonSources}
     ${CMAKE_CURRENT_SOURCE_DIR}/rank6_permutation_test.cpp)
set (PermutationRank6TestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/rank6_test_params.yaml)
add_hiptensor_test(rank6_permutation_test ${PermutationRank6TestConfig}  ${PermutationRank6TestSources})
