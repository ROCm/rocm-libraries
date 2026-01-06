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

# Include path for current test files
set(ROCWMMA_TEST_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${ROCWMMA_TEST_INCLUDE_DIRS})

set(LayoutTraitsTestSources ${UnitCommonSources}
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_16_f16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_16_f32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_16_f64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_16_i8.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_32_f16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_32_f32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_32_f64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_32_i8.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_64_f16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_64_f32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_64_f64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_64_i8.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_128_f16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_128_f32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_128_f64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_128_i8.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_256_f16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_256_f32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_256_f64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_256_i8.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_16_mmadim_16_splitk_1.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_16_mmadim_16_splitk_2.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_16_mmadim_16_splitk_4.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_32_mmadim_16_splitk_1.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_32_mmadim_16_splitk_4.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_32_mmadim_32_splitk_2.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_32_mmadim_16_splitk_2.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_32_mmadim_32_splitk_1.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_32_mmadim_32_splitk_4.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_64_mmadim_16_splitk_1.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_64_mmadim_16_splitk_2.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_64_mmadim_16_splitk_4.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_64_mmadim_32_splitk_1.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_64_mmadim_32_splitk_2.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_64_mmadim_32_splitk_4.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_64_mmadim_64_splitk_1.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_64_mmadim_64_splitk_2.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_64_mmadim_64_splitk_4.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_128_mmadim_16_splitk_1.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_128_mmadim_16_splitk_2.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_128_mmadim_16_splitk_4.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_128_mmadim_32_splitk_1.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_128_mmadim_32_splitk_2.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_128_mmadim_32_splitk_4.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_128_mmadim_64_splitk_1.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_128_mmadim_64_splitk_2.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_128_mmadim_64_splitk_4.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_256_mmadim_16_splitk_1.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_256_mmadim_16_splitk_2.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_256_mmadim_16_splitk_4.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_256_mmadim_32_splitk_1.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_256_mmadim_32_splitk_2.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_256_mmadim_32_splitk_4.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_256_mmadim_64_splitk_1.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_256_mmadim_64_splitk_2.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/layout_traits_int_256_mmadim_64_splitk_4.cpp

                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/smoketest_layout_traits.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest_layout_traits.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/extendedtest_layout_traits.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/extendedtest_layout_traits_int_mmadim_16_splitk_1.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/extendedtest_layout_traits_int_mmadim_16_splitk_2.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/extendedtest_layout_traits_int_mmadim_16_splitk_4.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/extendedtest_layout_traits_int_mmadim_32_splitk_1.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/extendedtest_layout_traits_int_mmadim_32_splitk_2.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/extendedtest_layout_traits_int_mmadim_32_splitk_4.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest_layout_traits_int_mmadim_16_splitk_1.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest_layout_traits_int_mmadim_16_splitk_2.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest_layout_traits_int_mmadim_16_splitk_4.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest_layout_traits_int_mmadim_32_splitk_1.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest_layout_traits_int_mmadim_32_splitk_2.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest_layout_traits_int_mmadim_32_splitk_4.cpp
                       )

add_rocwmma_unit_test(layout_traits_test ${LayoutTraitsTestSources})
