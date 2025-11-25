# ########################################################################
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
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

set(HIPSPARSE_BENCHMARK_SOURCES
  client.cpp
  hipsparse_arguments_config.cpp
  hipsparse_bench.cpp
  hipsparse_bench_app.cpp
  hipsparse_bench_cmdlines.cpp
  hipsparse_routine.cpp
)

set(HIPSPARSE_CLIENTS_COMMON
  ../common/arg_check.cpp
  ../common/unit.cpp
  ../common/utility.cpp
  ../common/hipsparse_template_specialization.cpp
)

add_executable(hipsparse-bench ${HIPSPARSE_BENCHMARK_SOURCES} ${HIPSPARSE_CLIENTS_COMMON})

# Target compile options
target_compile_options(hipsparse-bench PRIVATE -Wno-deprecated -Wno-unused-command-line-argument -Wall)

# Internal common header
target_include_directories(hipsparse-bench PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)

if (BUILD_CLIENTS_ONLY)
  target_include_directories(hipsparse-bench PRIVATE ${hipsparse_INCLUDE_DIRS}/hipsparse)
endif()

# Target link libraries
target_link_libraries(hipsparse-bench PRIVATE roc::hipsparse)

# Add OpenMP if available
if(OPENMP_FOUND AND THREADS_FOUND)
  target_link_libraries(hipsparse-bench PRIVATE OpenMP::OpenMP_CXX ${OpenMP_CXX_FLAGS})
endif()

if(NOT USE_CUDA)
  target_link_libraries(hipsparse-bench PRIVATE hip::host)
else()
  target_compile_definitions(hipsparse-bench PRIVATE __HIP_PLATFORM_NVIDIA__)
  target_include_directories(hipsparse-bench PRIVATE ${HIP_INCLUDE_DIRS})
  target_link_libraries(hipsparse-bench PRIVATE ${CUDA_LIBRARIES})
endif()

# Set benchmark output directory
set_target_properties(hipsparse-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

rocm_install(TARGETS hipsparse-bench COMPONENT benchmarks)
