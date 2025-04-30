# ########################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc.
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
# ########################################################################

add_subdirectory(src)

set(HIPBLASLT_TEST_DATA "${CMAKE_CURRENT_BINARY_DIR}/../hipblaslt_gtest.data")
add_custom_command(
    OUTPUT "${HIPBLASLT_TEST_DATA}"
    COMMAND ${Python3_EXECUTABLE} "${TEMP_CLIENTS_SOURCE_DIR}/common/hipblaslt_gentest.py" -I "${TEMP_CLIENTS_SOURCE_DIR}/include" "${TEMP_CLIENTS_SOURCE_DIR}/gtest/hipblaslt_gtest.yaml" -o ${HIPBLASLT_TEST_DATA}
)

add_custom_target(hipblaslt-test-data DEPENDS ${HIPBLASLT_TEST_DATA})
add_dependencies(hipblaslt-test hipblaslt-test-data)
