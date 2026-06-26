###############################################################################
 #
 # MIT License
 #
 # Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 ###############################################################################

# Registers a source-level test that the host-only stub
# (library/stub/hiptensor_stub.cpp) implements exactly the public API declared
# with HIPTENSOR_EXPORT. Its purpose is to catch a public API change that does
# not update the stub, so the stub build on unsupported architectures never
# ships an unresolved symbol.
#
# This is a pure source check (no compiler, no built library, no GPU), so it is
# registered as part of the regular test suite for any GPU_TARGETS.

find_package(Python3 COMPONENTS Interpreter)
if(NOT Python3_Interpreter_FOUND)
    message(WARNING "Python3 not found; skipping hiptensor_stub_api_coverage test.")
    return()
endif()

add_test(
    NAME hiptensor_stub_api_coverage
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_api_coverage.py
        --header ${PROJECT_SOURCE_DIR}/library/include/hiptensor/hiptensor.h
        --stub ${PROJECT_SOURCE_DIR}/library/stub/hiptensor_stub.cpp
)
