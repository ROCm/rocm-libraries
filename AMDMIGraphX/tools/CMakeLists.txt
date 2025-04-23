#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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

find_package(Python 3 COMPONENTS Interpreter)
if(NOT Python_EXECUTABLE)
    message(WARNING "Python 3 interpreter not found - skipping 'generate' target!")
    return()
endif()

find_program(CLANG_FORMAT clang-format PATHS /opt/rocm/llvm ENV HIP_PATH PATH_SUFFIXES bin)
if(NOT CLANG_FORMAT)
    message(WARNING "clang-format not found - skipping 'generate' target!")
    return()
endif()

add_custom_target(generate ${Python_EXECUTABLE} generate.py -f ${CLANG_FORMAT} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
