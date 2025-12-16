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

set(CrossLaneOpsTestSources ${UnitCommonSources}

                           # DPP
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/dpp_bcast_2.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/dpp_bcast_4.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/dpp_bcast_16.cpp

                           ${CMAKE_CURRENT_SOURCE_DIR}/test/dpp_reverse_2.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/dpp_reverse_4.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/dpp_reverse_8.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/dpp_reverse_16.cpp

                           ${CMAKE_CURRENT_SOURCE_DIR}/test/dpp_rotate_l2.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/dpp_rotate_r2.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/dpp_rotate_l4.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/dpp_rotate_r4.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/dpp_rotate_r16.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/dpp_rotate_wave_l1.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/dpp_rotate_wave_r1.cpp

                           ${CMAKE_CURRENT_SOURCE_DIR}/test/dpp_shift_l16.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/dpp_shift_r16.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/dpp_shift_wave_l1.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/dpp_shift_wave_r1.cpp

                           ${CMAKE_CURRENT_SOURCE_DIR}/test/dpp_shuffle_2.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/dpp_shuffle_4.cpp

                           ${CMAKE_CURRENT_SOURCE_DIR}/test/dpp_swap_2.cpp

                           ${CMAKE_CURRENT_SOURCE_DIR}/test/dpp_wfall_bcast_16.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/dpp_wfall_bcast_32.cpp

                           # Swizzle
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/swizzle_bcast_2.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/swizzle_bcast_4.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/swizzle_bcast_8.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/swizzle_bcast_16.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/swizzle_bcast_32.cpp

                           ${CMAKE_CURRENT_SOURCE_DIR}/test/swizzle_reverse_2.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/swizzle_reverse_4.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/swizzle_reverse_8.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/swizzle_reverse_16.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/swizzle_reverse_32.cpp

                           ${CMAKE_CURRENT_SOURCE_DIR}/test/swizzle_rotate_l2.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/swizzle_rotate_l4.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/swizzle_rotate_l8.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/swizzle_rotate_l16.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/swizzle_rotate_l32.cpp

                           ${CMAKE_CURRENT_SOURCE_DIR}/test/swizzle_rotate_r2.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/swizzle_rotate_r4.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/swizzle_rotate_r8.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/swizzle_rotate_r16.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/swizzle_rotate_r32.cpp

                           ${CMAKE_CURRENT_SOURCE_DIR}/test/swizzle_shuffle_2.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/swizzle_shuffle_4.cpp

                           ${CMAKE_CURRENT_SOURCE_DIR}/test/swizzle_swap_2.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/swizzle_swap_4.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/swizzle_swap_8.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/swizzle_swap_16.cpp

                           # Permute
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/permute_block_bcast_2.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/permute_block_bcast_4.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/permute_block_bcast_8.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/permute_block_bcast_16.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/permute_block_bcast_32.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/permute_rotate.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/permute_gather_scatter.cpp

                           # Blend
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/blend_unpack_byte_hi_lo.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/blend_unpack_byte_hi.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/blend_unpack_byte_lo.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/blend_unpack_word_hi.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/blend_unpack_word_lo.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/blend_zip_byte.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/blend_zip_word.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/blend_zip.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/blend_extract_byte_even.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/blend_extract_byte_odd.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/blend_extract_word_even.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/blend_extract_word_odd.cpp

                           # Emulation Smoke
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/smoketest-blend_extract_byte_even.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/smoketest-blend_zip.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/smoketest-dpp_bcast_2.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/smoketest-permute_gather_scatter.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/smoketest-swizzle_reverse_4.cpp

                           # Emulation regression
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-blend_zip.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-dpp_bcast_2.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-dpp_reverse_2.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-dpp_rotate_l2.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-dpp_rotate_wave_l1.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-dpp_shift_l16.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-blend_unpack_byte_hi.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-dpp_shuffle_2.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-dpp_wfall_bcast_16.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-dpp_shift_wave_l1.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-permute_block_bcast_2.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-permute_gather_scatter.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-permute_rotate.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-swizzle_bcast_2.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-swizzle_reverse_2.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-swizzle_rotate_l2.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-swizzle_shuffle_2.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-swizzle_swap_2.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-blend_extract_byte_even.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-blend_extract_word_even.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-blend_unpack_byte_lo.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-blend_unpack_word_hi.cpp
)

add_rocwmma_unit_test(cross_lane_ops_test ${CrossLaneOpsTestSources})

