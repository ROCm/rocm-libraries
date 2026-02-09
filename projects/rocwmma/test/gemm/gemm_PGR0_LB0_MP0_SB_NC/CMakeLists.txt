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
set(ROCWMMA_KERNEL_BASE_NAME "gemm_PGR0_LB0_MP0_SB_NC")
set(ROCWMMA_TARGET_NAME ${ROCWMMA_KERNEL_BASE_NAME})
set(ROCWMMA_TARGET_SOURCES ${ROCWMMA_TARGET_NAME}_sources)

set(ROCWMMA_AD_HOC_TARGET_NAME ${ROCWMMA_TARGET_NAME}_ad_hoc)
set(ROCWMMA_AD_HOC_TARGET_SOURCES ${ROCWMMA_AD_HOC_TARGET_NAME}_sources)

set(${ROCWMMA_TARGET_SOURCES} ${GemmCommonSources}
                          ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nn.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_nt.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tn.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/test/16x16_tt.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nn.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_nt.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tn.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/test/32x32_tt.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/smoketest-16x16.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-16x16.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-32x32.cpp
                          )

# Ad hoc test
# Note: GemmKernelBase and GemmResource instantiations required.
set(${ROCWMMA_AD_HOC_TARGET_SOURCES} ${ROCWMMA_COMMON_TEST_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/test/ad_hoc_test.cpp)

# Create targets
add_gemm_test(${ROCWMMA_TARGET_NAME}  ${${ROCWMMA_TARGET_SOURCES}})
add_gemm_test(${ROCWMMA_AD_HOC_TARGET_NAME} ${${ROCWMMA_AD_HOC_TARGET_SOURCES}})
