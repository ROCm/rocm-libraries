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
#
# Subsequent per-area unit-test branches append their host-pure sources here.
set(ROCSPARSE_UNIT_TEST_SOURCES
  rocsparse_unit_test_main.cpp
  unit_test_enum_traits.cpp
  unit_test_indextype_utils.cpp
  unit_test_datatype_utils.cpp
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

# This target compiles a few library .cpp translation units directly
# (ROCSPARSE_UNIT_TEST_LIB_SOURCES). Those files decorate public entry points
# with ROCSPARSE_EXPORT, which the generated rocsparse-export.h expands to a
# dllexport/dllimport (Windows) or default-visibility (Linux) attribute. Applying
# an export/import attribute to a *definition* compiled into this executable is
# ill-formed on Windows ("dllimport cannot be applied to non-inline function
# definition"). Defining ROCSPARSE_STATIC_DEFINE makes rocsparse-export.h expand
# ROCSPARSE_EXPORT (and *_NO_EXPORT) to nothing on every platform, which is
# correct here: these TUs are compiled *into* the test binary, not consumed from
# the shared library. Test-only; no library change.
target_compile_definitions(rocsparse-unit-test PRIVATE GOOGLE_TEST ROCSPARSE_STATIC_DEFINE)
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

# ---------------------------------------------------------------------------
# rocsparse-unit-test-device : GPU (device) unit tests.
#
# A SEPARATE binary from rocsparse-unit-test so the latter stays fast and
# GPU-independent. This target links hip::device (which supplies `-x hip` and
# `--offload-arch`) and launches individual library kernels in isolation, or
# drives the public API host paths that need a real device. It MUST run on a
# GPU (via the serializer, e.g. HIP_VISIBLE_DEVICES=0 gpu-run).
#
# Subsequent per-area unit-test branches append their device / host-path
# sources to ROCSPARSE_UNIT_TEST_DEVICE_SOURCES (and, for the internal
# primitives, ROCSPARSE_UNIT_TEST_PRIMITIVE_SOURCES) below.
# ---------------------------------------------------------------------------
set(ROCSPARSE_UNIT_TEST_DEVICE_SOURCES
  rocsparse_unit_test_main.cpp
  unit_test_host_conversion_sort.cpp
  unit_test_internal_collectives.cpp
  unit_test_internal_infostructs.cpp
  unit_test_internal_hostblocks.cpp
)

# Host building-block library translation units under test by
# unit_test_internal_hostblocks.cpp. Compiled into the device (GPU) test binary
# because their include chain (rocsparse_control.hpp -> rocsparse_common.hpp)
# requires the HIP compile mode; the routines themselves are host-pure logic.
# This is build wiring only -- NO library source changes. Subsequent leaves
# append here (append-only) so the stacked branches do not conflict.
set(ROCSPARSE_UNIT_TEST_DEVICE_LIB_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/level3/rocsparse_csrmm_default_alg.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/common/rocsparse_determine_indextype.cpp
)

# Internal rocsparse::primitives building blocks under test. Like the host-pure
# utilities compiled into rocsparse-unit-test, these template instantiations are
# hidden in librocsparse (CXX_VISIBILITY_PRESET "hidden") and cannot be reached
# by linking roc::rocsparse, so the primitive translation units are compiled
# directly into this GPU test binary. They pull in rocprim, hence roc::rocprim
# is added to the link libraries below. (No library source changes.)
set(ROCSPARSE_UNIT_TEST_PRIMITIVE_SOURCES
)

add_executable(rocsparse-unit-test-device
  ${ROCSPARSE_UNIT_TEST_DEVICE_SOURCES}
  ${ROCSPARSE_UNIT_TEST_PRIMITIVE_SOURCES}
  ${ROCSPARSE_UNIT_TEST_DEVICE_LIB_SOURCES}
)

# See the note on rocsparse-unit-test above: ROCSPARSE_STATIC_DEFINE makes the
# compiled-in library TUs' ROCSPARSE_EXPORT expand to nothing, avoiding the
# Windows "dllimport cannot be applied to non-inline function definition" error
# (e.g. from library/src/auxiliary/rocsparse_auxiliary.cpp compiled in at the
# infostructs leaf and beyond). Correct on Linux too (these are definitions
# built into the executable, not symbols imported from the shared library).
target_compile_definitions(rocsparse-unit-test-device PRIVATE GOOGLE_TEST ROCSPARSE_STATIC_DEFINE)
target_compile_options(rocsparse-unit-test-device PRIVATE
  -Wno-deprecated
  -Wno-unused-command-line-argument
)

# Private library headers: rocsparse_common.hpp (macros/fma) lives in
# library/src/include; the level-1 device kernels live in library/src/level1.
target_include_directories(rocsparse-unit-test-device PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/level1>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/level3>
)

# hip::device provides the HIP device-compilation flags (-x hip, --offload-arch).
# roc::rocprim supplies the rocprim headers used by the compiled-in
# library/src/primitives translation units.
target_link_libraries(rocsparse-unit-test-device PRIVATE
  GTest::gtest
  roc::rocsparse
  roc::rocprim
  hip::host
  hip::device
)

apply_omp_settings(rocsparse-unit-test-device)

set_target_properties(rocsparse-unit-test-device PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)

# Registered with CTest but tagged so CI can schedule it on a GPU node only
# (it is NOT part of the fast CPU-only gate).
add_test(NAME rocsparse-unit-test-device COMMAND rocsparse-unit-test-device)
set_tests_properties(rocsparse-unit-test-device PROPERTIES LABELS "gpu")

rocm_install(TARGETS rocsparse-unit-test-device COMPONENT tests)

# ---------------------------------------------------------------------------
# Code coverage: instrument the unit-test binaries so the library sources that
# are compiled directly into them (primitives, level-1 device kernels,
# indextype/datatype utils) emit LLVM profile data. Without this the binaries
# run under `make coverage` but produce no .profraw and contribute nothing.
# Matches the flags applied to the rocsparse library target.
# ---------------------------------------------------------------------------
if(BUILD_CODE_COVERAGE)
  foreach(_cov_tgt rocsparse-unit-test rocsparse-unit-test-device)
    if(TARGET ${_cov_tgt})
      target_compile_options(${_cov_tgt} PRIVATE -fprofile-instr-generate -fcoverage-mapping)
      target_link_libraries(${_cov_tgt} PRIVATE -fprofile-instr-generate)
    endif()
  endforeach()
endif()
