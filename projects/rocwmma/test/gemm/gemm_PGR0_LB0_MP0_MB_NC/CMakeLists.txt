###############################################################################
 #
 # MIT License
 #
 # Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 # SOFTWARE.
 #
 ###############################################################################

# Add the current folder to test includes
set(ROCWMMA_TEST_GEMM_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${ROCWMMA_TEST_GEMM_INCLUDE_DIRS})

# Setup kernel test symbols
set(ROCWMMA_KERNEL_BASE_NAME "gemm_PGR0_LB0_MP0_MB_NC")
set(ROCWMMA_TARGET_NAME ${ROCWMMA_KERNEL_BASE_NAME})
set(ROCWMMA_TARGET_SOURCES ${ROCWMMA_TARGET_NAME}_sources)

set(ROCWMMA_AD_HOC_TARGET_NAME ${ROCWMMA_TARGET_NAME}_ad_hoc)
set(ROCWMMA_AD_HOC_TARGET_SOURCES ${ROCWMMA_AD_HOC_TARGET_NAME}_sources)

# Setup test source files
set(${ROCWMMA_TARGET_SOURCES}   ${GemmCommonSources}
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nn_1x1.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nn_2x2.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nn_4x4.cpp

                                ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nt_1x1.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nt_2x2.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nt_4x4.cpp

                                ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tn_1x1.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tn_2x2.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tn_4x4.cpp

                                ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tt_1x1.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tt_2x2.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tt_4x4.cpp

                                ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nn_1x1.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nn_2x2.cpp

                                ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nt_1x1.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nt_2x2.cpp

                                ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tn_1x1.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tn_2x2.cpp

                                ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tt_1x1.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tt_2x2.cpp

                                # Emulation Smoke
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/smoketest-16x16_4x2.cpp

                                # Emulation Regression
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-16x16_4x2.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-32x32_4x2.cpp
                                )

if(ROCWMMA_BUILD_EXTENDED_TESTS)
    set(${ROCWMMA_TARGET_SOURCES}   ${${ROCWMMA_TARGET_SOURCES}}
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nn_1x2.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nn_1x4.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nn_1x8.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nn_2x1.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nn_2x4.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nn_2x8.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nn_4x1.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nn_4x2.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nn_4x8.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nn_8x1.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nn_8x2.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nn_8x4.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nn_8x8.cpp

                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nt_1x2.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nt_1x4.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nt_1x8.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nt_2x1.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nt_2x4.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nt_2x8.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nt_4x1.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nt_4x2.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nt_4x8.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nt_8x1.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nt_8x2.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nt_8x4.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nt_8x8.cpp

                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tn_1x2.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tn_1x4.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tn_1x8.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tn_2x1.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tn_2x4.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tn_2x8.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tn_4x1.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tn_4x2.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tn_4x8.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tn_8x1.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tn_8x2.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tn_8x4.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tn_8x8.cpp

                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tt_1x2.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tt_1x4.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tt_1x8.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tt_2x1.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tt_2x4.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tt_2x8.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tt_4x1.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tt_4x2.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tt_4x8.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tt_8x1.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tt_8x2.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tt_8x4.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tt_8x8.cpp

                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nn_1x2.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nn_1x4.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nn_1x8.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nn_2x1.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nn_2x4.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nn_2x8.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nn_4x1.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nn_4x2.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nn_4x4.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nn_8x1.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nn_8x2.cpp

                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nt_1x2.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nt_1x4.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nt_1x8.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nt_2x1.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nt_2x4.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nt_2x8.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nt_4x1.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nt_4x2.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nt_4x4.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nt_8x1.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nt_8x2.cpp

                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tn_1x2.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tn_1x4.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tn_1x8.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tn_2x1.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tn_2x4.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tn_2x8.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tn_4x1.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tn_4x2.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tn_4x4.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tn_8x1.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tn_8x2.cpp

                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tt_1x2.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tt_1x4.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tt_1x8.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tt_2x1.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tt_2x4.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tt_2x8.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tt_4x1.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tt_4x2.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tt_4x4.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tt_8x1.cpp
                                    ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tt_8x2.cpp

                                    )
endif()

# Ad hoc test sources.
# Note: GemmKernelBase and GemmResource instantiations required.
set(${ROCWMMA_AD_HOC_TARGET_SOURCES} ${ROCWMMA_COMMON_TEST_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/test/ad_hoc_test.cpp)

# Create targets
add_gemm_test(${ROCWMMA_TARGET_NAME}  ${${ROCWMMA_TARGET_SOURCES}})
add_gemm_test(${ROCWMMA_AD_HOC_TARGET_NAME} ${${ROCWMMA_AD_HOC_TARGET_SOURCES}})
