#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################

function(add_tf_test TEST_NAME)
    rocm_add_test_executable(${TEST_NAME} ${ARGN})
    rocm_clang_tidy_check(${TEST_NAME})
    target_link_libraries(${TEST_NAME} migraphx_tf pb_files)
    target_include_directories(${TEST_NAME} PUBLIC ../include include)
endfunction()

include(Embed)
file(GLOB_RECURSE PB_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/models/*.pb)
add_embed_library(pb_files ${PB_FILES} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/models)

file(GLOB TF_TESTS CONFIGRE_DEPENDS tests/*.cpp)

add_tf_test(test_tf ${TF_TESTS})
