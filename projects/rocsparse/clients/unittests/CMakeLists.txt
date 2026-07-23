# ########################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#
# rocsparse-unit-test : fast, GPU-independent unit tests for individual
# host-side library components.
#
# This target is intentionally separate from rocsparse-test (the YAML/Arguments
# driven integration suite). It does NOT use the gentest data pipeline, does not
# require a GPU to run, and links only what each unit needs.
#
# Only builds in-tree (NOT BUILD_CLIENTS_ONLY), because it reaches into the
# library's private headers/sources under ../../library/src.
#
if(BUILD_CLIENTS_ONLY)
  message(STATUS "Skipping rocsparse-unit-test (requires in-tree build for library internals)")
  return()
endif()

find_package(GTest 1.11.0 REQUIRED)

# Unit-test sources (the tests themselves + a minimal gtest main).
set(ROCSPARSE_UNIT_TEST_SOURCES
  rocsparse_unit_test_main.cpp
  unit_test_enum_traits.cpp
  unit_test_indextype_utils.cpp
  unit_test_datatype_utils.cpp
  unit_test_csrmv_adaptive_analysis.cpp
)

# Host-pure library translation units under test. These are compiled directly
# into the unit-test binary because librocsparse is built with hidden symbol
# visibility, so their (non-static, header-declared) symbols are not exported
# for external linking. This keeps the prototype free of any library source
# changes (Symbol-access Option A in the design doc).
set(ROCSPARSE_UNIT_TEST_LIB_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/common/rocsparse_indextype_utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/common/rocsparse_datatype_utils.cpp
)

add_executable(rocsparse-unit-test
  ${ROCSPARSE_UNIT_TEST_SOURCES}
  ${ROCSPARSE_UNIT_TEST_LIB_SOURCES}
)

target_compile_definitions(rocsparse-unit-test PRIVATE GOOGLE_TEST)
target_compile_options(rocsparse-unit-test PRIVATE
  -Wno-deprecated
  -Wno-unused-command-line-argument
)

# Private library headers (for the utilities under test) + public interface
# headers (rocsparse-types.h etc.) come transitively from roc::rocsparse.
target_include_directories(rocsparse-unit-test PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/include>
)

# roc::rocsparse is linked for its INTERFACE include dirs (public types) and so
# future unit tests can reach exported symbols; hip::host provides HIP headers
# pulled in transitively by rocsparse-types.h.
target_link_libraries(rocsparse-unit-test PRIVATE
  GTest::gtest
  roc::rocsparse
  hip::host
)

apply_omp_settings(rocsparse-unit-test)

set_target_properties(rocsparse-unit-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)

# CTest registration. enable_testing() is already invoked at the top-level
# project scope (projects/rocsparse/CMakeLists.txt, guarded by BUILD_CLIENTS_TESTS),
# so this test is discoverable via `ctest` from the top build directory. This lets
# CI run the fast, GPU-independent unit tests as an early gate: `ctest -R rocsparse-unit-test`.
add_test(NAME rocsparse-unit-test COMMAND rocsparse-unit-test)

rocm_install(TARGETS rocsparse-unit-test COMPONENT tests)
