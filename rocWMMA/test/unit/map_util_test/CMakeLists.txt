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

set(MapUtilTestSources ${UnitCommonSources}
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_block_to_matrix_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_block_to_matrix_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_block_to_matrix_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_block_to_matrix_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_block_to_matrix_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_block_to_matrix_override_m_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_block_to_matrix_override_m_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_block_to_matrix_override_m_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_block_to_matrix_override_m_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_block_to_matrix_override_m_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_block_to_matrix_override_n_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_block_to_matrix_override_n_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_block_to_matrix_override_n_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_block_to_matrix_override_n_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_block_to_matrix_override_n_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_matrix_to_data_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_matrix_to_data_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_matrix_to_data_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_matrix_to_data_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_matrix_to_data_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_matrix_to_data_override_m_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_matrix_to_data_override_m_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_matrix_to_data_override_m_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_matrix_to_data_override_m_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_matrix_to_data_override_m_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_matrix_to_data_override_n_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_matrix_to_data_override_n_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_matrix_to_data_override_n_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_matrix_to_data_override_n_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_matrix_to_data_override_n_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_thread_to_matrix_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_thread_to_matrix_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_thread_to_matrix_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_thread_to_matrix_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_thread_to_matrix_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_wave_to_matrix_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_wave_to_matrix_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_wave_to_matrix_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_wave_to_matrix_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/map_wave_to_matrix_256.cpp

                       # Emulation Smoke
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/smoketest-map_block_to_matrix_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/smoketest-map_thread_to_matrix_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/smoketest-map_matrix_to_data_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/smoketest-map_wave_to_matrix_16.cpp
                       # Emulation regression
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-map_block_to_matrix_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-map_block_to_matrix_override_m_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-map_block_to_matrix_override_n_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-map_matrix_to_data_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-map_matrix_to_data_override_m_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-map_matrix_to_data_override_n_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-map_thread_to_matrix_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-map_wave_to_matrix_16.cpp
                       )

                   add_rocwmma_unit_test(map_util_test ${MapUtilTestSources})
