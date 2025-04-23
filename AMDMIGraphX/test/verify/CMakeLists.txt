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

file(GLOB VERIFY_TESTS CONFIGURE_DEPENDS *.cpp)

add_executable(test_verify ${VERIFY_TESTS})
rocm_mark_as_test(test_verify)
rocm_install_test(TARGETS test_verify)
target_link_libraries(test_verify migraphx migraphx_all_targets)
target_include_directories(test_verify PUBLIC ../include)
rocm_clang_tidy_check(test_verify)

foreach(SECTION general reduce rnn conv gemm)
    rocm_add_test(NAME test_verify_${SECTION} COMMAND test_verify ${SECTION})
    set_tests_properties(test_verify_${SECTION} PROPERTIES 
        COST 100 
    )
    if(MIGRAPHX_ENABLE_GPU)
        set_tests_properties(test_verify_${SECTION} PROPERTIES 
            RESOURCE_LOCK gpu
        )
    endif()
endforeach()
