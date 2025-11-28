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

set(LayoutTestSources ${UnitCommonSources}
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/col_layout_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/col_layout_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/col_layout_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/col_layout_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/col_layout_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/colnt_layout_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/colnt_layout_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/colnt_layout_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/colnt_layout_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/colnt_layout_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/row_layout_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/row_layout_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/row_layout_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/row_layout_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/row_layout_256.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/rownt_layout_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/rownt_layout_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/rownt_layout_64.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/rownt_layout_128.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/rownt_layout_256.cpp

                       #Emulation Smoke
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/smoketest-col_layout_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/smoketest-colnt_layout_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/smoketest-row_layout_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/smoketest-rownt_layout_16.cpp

                       #Emulation regression
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-col_layout_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-colnt_layout_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-row_layout_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-rownt_layout_16.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-col_layout_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-colnt_layout_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-row_layout_32.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/test/emulation/regressiontest-rownt_layout_32.cpp
                       )

add_rocwmma_unit_test(layout_test ${LayoutTestSources})
