# ########################################################################
# Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

set(ROCALUTION_BENCHMARK_SOURCES
  client.cpp
  rocalution_arguments_config.cpp
  rocalution_bench.cpp
  rocalution_bench_solver_parameters.cpp
  rocalution_bench_solver_results.cpp
  rocalution_enum_coarsening_strategy.cpp
  rocalution_enum_directsolver.cpp
  rocalution_enum_itilu0_alg.cpp
  rocalution_enum_itsolver.cpp
  rocalution_enum_matrix_init.cpp
  rocalution_enum_preconditioner.cpp
  rocalution_enum_smoother.cpp
  rocalution_bench_cmdlines.cpp
  rocalution_bench_app.cpp
)

add_executable(rocalution-bench ${ROCALUTION_BENCHMARK_SOURCES} ${ROCALUTION_CLIENTS_COMMON})

# Target compile options
target_compile_options(rocalution-bench PRIVATE -Wall)

# Internal common header
target_include_directories(rocalution-bench PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)

# Target link libraries
target_link_libraries(rocalution-bench PRIVATE roc::rocalution)

if(NOT TARGET rocalution)
  set_target_properties(rocalution-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
else()
  set_target_properties(rocalution-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")
endif()

rocm_install(TARGETS rocalution-bench COMPONENT benchmarks)
