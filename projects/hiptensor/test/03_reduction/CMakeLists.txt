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
 set(ReductionCommonSources ${HIPTENSOR_COMMON_TEST_SOURCES}
 ${CMAKE_CURRENT_SOURCE_DIR}/reduction_resource.cpp
 ${CMAKE_CURRENT_SOURCE_DIR}/reduction_test.cpp)

#  tests
set (ReductionTestSources ${ReductionCommonSources}
${CMAKE_CURRENT_SOURCE_DIR}/reduction_cpu_impl_test.cpp
)
set (ReductionConfig ${CMAKE_CURRENT_SOURCE_DIR}/configs/reduction_params.yaml)
add_hiptensor_test(reduction_cpu_impl_test ${ReductionConfig} ${ReductionTestSources})

set (ReductionRank1TestSources ${ReductionCommonSources}
     ${CMAKE_CURRENT_SOURCE_DIR}/rank1_reduction_test.cpp)
set (ReductionRank1TestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/rank1_test_params.yaml)
add_hiptensor_test(rank1_reduction_test ${ReductionRank1TestConfig}  ${ReductionRank1TestSources})

set (ReductionRank2TestSources ${ReductionCommonSources}
     ${CMAKE_CURRENT_SOURCE_DIR}/rank2_reduction_test.cpp)
set (ReductionRank2TestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/rank2_test_params.yaml)
add_hiptensor_test(rank2_reduction_test ${ReductionRank2TestConfig}  ${ReductionRank2TestSources})

set (ReductionRank3TestSources ${ReductionCommonSources}
     ${CMAKE_CURRENT_SOURCE_DIR}/rank3_reduction_test.cpp)
set (ReductionRank3TestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/rank3_test_params.yaml)
add_hiptensor_test(rank3_reduction_test ${ReductionRank3TestConfig}  ${ReductionRank3TestSources})

set (ReductionRank4TestSources ${ReductionCommonSources}
     ${CMAKE_CURRENT_SOURCE_DIR}/rank4_reduction_test.cpp)
set (ReductionRank4TestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/rank4_test_params.yaml)
add_hiptensor_test(rank4_reduction_test ${ReductionRank4TestConfig}  ${ReductionRank4TestSources})

set (ReductionRank5TestSources ${ReductionCommonSources}
     ${CMAKE_CURRENT_SOURCE_DIR}/rank5_reduction_test.cpp)
set (ReductionRank5TestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/rank5_test_params.yaml)
add_hiptensor_test(rank5_reduction_test ${ReductionRank5TestConfig}  ${ReductionRank5TestSources})

set (ReductionRank6TestSources ${ReductionCommonSources}
     ${CMAKE_CURRENT_SOURCE_DIR}/rank6_reduction_test.cpp)
set (ReductionRank6TestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/rank6_test_params.yaml)
add_hiptensor_test(rank6_reduction_test ${ReductionRank6TestConfig}  ${ReductionRank6TestSources})
