# MIT License
#
# Modifications Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

add_custom_target(libcudacxx.test.atomics.ptx)

find_program(filecheck "FileCheck")

if (filecheck)
  message("-- ${filecheck} found... building atomic codegen tests")
else()
  return()
endif()

find_program(cuobjdump "cuobjdump" REQUIRED)
find_program(bash "bash" REQUIRED)

set(libcudacxx_atomic_codegen_tests)
if (NOT "NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  file(GLOB libcudacxx_atomic_codegen_tests "*.cu")
endif()

# For every atomic API compile the TU and check if the SASS/PTX matches the expected result
foreach(test_path IN LISTS libcudacxx_atomic_codegen_tests)
    cmake_path(GET test_path FILENAME test_file)
    cmake_path(REMOVE_EXTENSION test_file LAST_ONLY OUTPUT_VARIABLE test_name)

    add_library(
        atomic_codegen_${test_name}
        STATIC ${test_path}
    )

    set_target_properties(
        atomic_codegen_${test_name}
        PROPERTIES
          CUDA_ARCHITECTURES "80"
          COMPILE_DEFINITIONS "_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE=1"
    )

    target_compile_options(atomic_codegen_${test_name} PRIVATE "-Wno-comment")

    ## Important for testing the local headers
    target_include_directories(atomic_codegen_${test_name} PRIVATE "${libhipcxx_SOURCE_DIR}/include")
    add_dependencies(libcudacxx.test.atomics.ptx atomic_codegen_${test_name})

    # Add output path to object directory
    add_custom_command(
        TARGET libcudacxx.test.atomics.ptx
        POST_BUILD
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/dump_and_check.bash $<TARGET_FILE:atomic_codegen_${test_name}> ${test_path} SM8X
    )
endforeach()
