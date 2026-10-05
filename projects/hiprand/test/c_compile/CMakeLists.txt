# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Checks that the public hipRAND headers compile as C (not only C++), including
# when <hip/hip_runtime.h> is included before or after <hiprand/hiprand.h>.

# These checks target the HIP/AMD headers specifically; skip on the CUDA backend.
if(BUILD_WITH_LIB STREQUAL "CUDA")
    return()
endif()

# hipRAND is otherwise a CXX-only project; enable C just for these sources.
enable_language(C)

set(C_COMPILE_SOURCES
    c_compile_hiprand_only.c
    c_compile_hiprand_then_hip.c
    c_compile_hip_then_hiprand.c
)

# Borrow HIP's include directories without pulling in the device/C++ compile
# flags that hip::device would add (those break a plain C compile).
if(TARGET hip::host)
    set(HIP_C_COMPILE_INCLUDE_TARGET hip::host)
else()
    set(HIP_C_COMPILE_INCLUDE_TARGET hip::device)
endif()

# Compile the C translation units in isolation so only include paths -- not
# compile options -- are inherited from the library targets.
add_library(hiprand_c_compile_obj OBJECT ${C_COMPILE_SOURCES})
set_target_properties(hiprand_c_compile_obj
    PROPERTIES
        C_STANDARD          11
        C_STANDARD_REQUIRED ON
)
target_compile_definitions(hiprand_c_compile_obj PRIVATE __HIP_PLATFORM_AMD__)
target_include_directories(hiprand_c_compile_obj
    PRIVATE
        $<TARGET_PROPERTY:hiprand,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:${HIP_C_COMPILE_INCLUDE_TARGET},INTERFACE_INCLUDE_DIRECTORIES>
)

# gtest driver: links the compiled C objects and runs them as CTest cases.
add_executable(test_hiprand_c_compile
    test_hiprand_c_compile.cpp
    $<TARGET_OBJECTS:hiprand_c_compile_obj>
)

if(TARGET GTest::GTest)
    target_link_libraries(test_hiprand_c_compile GTest::GTest GTest::Main)
else()
    target_link_libraries(test_hiprand_c_compile GTest::gtest GTest::gtest_main)
endif()

target_link_libraries(test_hiprand_c_compile
    hiprand
    roc::rocrand
    hip::device
)

set_target_properties(test_hiprand_c_compile
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
)
add_relative_test(test_hiprand_c_compile test_hiprand_c_compile)
rocm_install(TARGETS test_hiprand_c_compile COMPONENT tests)
