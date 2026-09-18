# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

## Codegen adds the following build targets
#    libcudacxx.atomics.codegen
#    libcudacxx.atomics.codegen.install
## Test targets:
#    libcudacxx.test.atomics.codegen.diff

cccl_get_fmt()

add_executable(
    codegen
    EXCLUDE_FROM_ALL
    codegen.cpp
)

target_link_libraries(codegen PRIVATE fmt)

set_property(TARGET codegen PROPERTY CXX_STANDARD 17)

set(atomic_generated_output "${libcudacxx_BINARY_DIR}/codegen/cuda_ptx_generated.h")
set(atomic_install_location "${libhipcxx_SOURCE_DIR}/include/cuda/std/__atomic/functions")

add_custom_target(
    libcudacxx.atomics.codegen
    COMMAND codegen "${atomic_generated_output}"
    BYPRODUCTS "${atomic_generated_output}"
)

add_custom_target(
    libcudacxx.atomics.codegen.install
    COMMAND ${CMAKE_COMMAND} -E copy "${atomic_generated_output}" "${atomic_install_location}/cuda_ptx_generated.h"
    DEPENDS libcudacxx.atomics.codegen
    BYPRODUCTS "${atomic_install_location}/cuda_ptx_generated.h"
)

add_test(
    NAME libcudacxx.test.atomics.codegen.diff
    COMMAND ${CMAKE_COMMAND} -E compare_files "${atomic_install_location}/cuda_ptx_generated.h" "${atomic_generated_output}"
)

set_tests_properties(
    libcudacxx.test.atomics.codegen.diff
    PROPERTIES REQUIRED_FILES "${atomic_generated_output}"
)
