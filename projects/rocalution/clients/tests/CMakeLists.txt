# ########################################################################
# Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

# Optional: CTest test categorization when building in rocm-libraries with shared/ctest.
# Skipped when shared/ is not present (e.g. standalone rocalution or sparse checkout in CI).
if(DEFINED ROCM_LIBRARIES_ROOT AND EXISTS "${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake")
  include("${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake")
  set(ROCALUTION_HAS_CTEST_CATEGORIES ON)
endif()

# Find google test and its dependencies
find_package(GTest REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Non MPI tests
set(ROCALUTION_TEST_SOURCES
  rocalution_host_gtest_main.cpp
# Direct solvers
  test_qr.cpp
  test_lu.cpp
  test_inversion.cpp
# Krylov solvers
  test_backend.cpp
  test_bicgstab.cpp
  test_bicgstabl.cpp
  test_cg.cpp
  test_cr.cpp
  test_fcg.cpp
  test_fgmres.cpp
  test_gmres.cpp
  test_idr.cpp
  test_qmrcgstab.cpp
# AMG
  test_pairwise_amg.cpp
  test_ruge_stueben_amg.cpp
  test_saamg.cpp
  test_uaamg.cpp
# Other tests
  test_preconditioner.cpp
  test_itersolver.cpp
  test_chebyshev.cpp
  test_mixed_precision.cpp
)

if(NOT WIN32)
# Local structures - skip on Windows, as google test does not support union for death tests
  list(APPEND ROCALUTION_TEST_SOURCES
    test_local_matrix.cpp
    test_local_matrix_multicoloring.cpp
    test_local_matrix_itsolve.cpp
    test_local_matrix_solve.cpp
    test_local_stencil.cpp
    test_local_vector.cpp
  )
endif()

# MPI tests
if(SUPPORT_MPI)
  list(APPEND ROCALUTION_TEST_SOURCES
    test_global_matrix.cpp
# test_global_stencil.cpp
    test_global_vector.cpp
    test_parallel_manager.cpp
  )
endif()

# Common
list(APPEND ROCALUTION_TEST_SOURCES
  ../include/random.cpp
)

add_executable(rocalution-test ${ROCALUTION_TEST_SOURCES} ${ROCALUTION_CLIENTS_COMMON})

# Google test
target_compile_definitions(rocalution-test PRIVATE GOOGLE_TEST)
target_include_directories(rocalution-test SYSTEM PRIVATE $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>)
target_link_libraries(rocalution-test PRIVATE ${GTEST_BOTH_LIBRARIES} Threads::Threads)

# Include common client headers
target_include_directories(rocalution-test PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)

# Link rocALUTION library
target_link_libraries(rocalution-test PRIVATE roc::rocalution)

if(NOT TARGET rocalution)
  set_target_properties(rocalution-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
else()
  set_target_properties(rocalution-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")
endif()

rocm_install(TARGETS rocalution-test COMPONENT tests)

# ---------------------------------------------------------------------------
# CTest categorization via test_categories.yaml
# Only active when building in the full rocm-libraries tree (ROCM_LIBRARIES_ROOT
# is defined and shared/ctest/TestCategories.cmake exists).
#
# rocALUTION does not use GTest category prefixes; all categories use "*" as
# the test pattern. Test subset selection is done via environment variables
# set below after the YAML-driven suites are generated:
#
#   quick / ffm-quick / ffm-full  -> ROCALUTION_EMULATION_SMOKE=1
#   standard                      -> ROCALUTION_EMULATION_REGRESSION=1
#   comprehensive                 -> ROCALUTION_EMULATION_EXTENDED=1
#   full                          -> (no env var — full parameter sweep)
# ---------------------------------------------------------------------------
set(ROCALUTION_CTEST_INSTALL_FILE "")

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml")
  if(ROCALUTION_HAS_CTEST_CATEGORIES)
    enable_testing()
    message(STATUS "rocALUTION: YAML-based test categorization")

    # Determine staging directory (mirrors the binary output path above)
    if(NOT TARGET rocalution)
      set(ROCALUTION_STAGING_DIR "${PROJECT_BINARY_DIR}/staging")
    else()
      set(ROCALUTION_STAGING_DIR "${PROJECT_BINARY_DIR}/clients/staging")
    endif()

    file(MAKE_DIRECTORY "${ROCALUTION_STAGING_DIR}")

    # Install-time CTestTestfile for TheRock
    set(ROCALUTION_CTEST_INSTALL_FILE "${CMAKE_CURRENT_BINARY_DIR}/install_CTestTestfile.cmake")
    file(WRITE "${ROCALUTION_CTEST_INSTALL_FILE}"
[=[
# This is a test file generated by rocALUTION for install time.
# Tests are defined with relative paths to work in the installed location.
]=]
    )

    apply_test_category_labels(
      rocalution-test
      "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml"
      "${ROCALUTION_STAGING_DIR}"
      "${ROCALUTION_CTEST_INSTALL_FILE}"
    )

    # Set per-suite ROCALUTION_EMULATION_* env vars.
    # The YAML uses test_patterns: ["*"] for most suites; the env var drives
    # which parameter sets actually execute inside the test binary.
    # ffm-quick and ffm-full use an explicit GTest filter (single CG test) instead,
    # so they do not need an emulation env var.
    # We also append these to the install-time file because the Python parser
    # that generates it has no knowledge of our post-hoc set_tests_properties().
    if(TEST rocalution-test_quick_suite)
      set_tests_properties(rocalution-test_quick_suite PROPERTIES
        ENVIRONMENT "ROCALUTION_EMULATION_SMOKE=1"
      )
      if(ROCALUTION_CTEST_INSTALL_FILE)
        file(APPEND "${ROCALUTION_CTEST_INSTALL_FILE}"
          "set_tests_properties(rocalution-test_quick_suite PROPERTIES ENVIRONMENT \"ROCALUTION_EMULATION_SMOKE=1\")\n"
        )
      endif()
    endif()

    if(TEST rocalution-test_standard_suite)
      set_tests_properties(rocalution-test_standard_suite PROPERTIES
        ENVIRONMENT "ROCALUTION_EMULATION_REGRESSION=1"
      )
      if(ROCALUTION_CTEST_INSTALL_FILE)
        file(APPEND "${ROCALUTION_CTEST_INSTALL_FILE}"
          "set_tests_properties(rocalution-test_standard_suite PROPERTIES ENVIRONMENT \"ROCALUTION_EMULATION_REGRESSION=1\")\n"
        )
      endif()
    endif()

    if(TEST rocalution-test_comprehensive_suite)
      set_tests_properties(rocalution-test_comprehensive_suite PROPERTIES
        ENVIRONMENT "ROCALUTION_EMULATION_EXTENDED=1"
      )
      if(ROCALUTION_CTEST_INSTALL_FILE)
        file(APPEND "${ROCALUTION_CTEST_INSTALL_FILE}"
          "set_tests_properties(rocalution-test_comprehensive_suite PROPERTIES ENVIRONMENT \"ROCALUTION_EMULATION_EXTENDED=1\")\n"
        )
      endif()
    endif()
    # rocalution-test_full_suite intentionally has no env var (runs everything).

  else()
    message(STATUS "rocALUTION: shared/ctest not found (e.g. standalone checkout), skipping CTest categories")
    # Fall back to a single unlabelled CTest entry
    add_test(rocalution-test rocalution-test)
  endif()
else()
  message(STATUS "rocALUTION: No test_categories.yaml (CTest categories not used)")
  add_test(rocalution-test rocalution-test)
endif()

# Install CTestTestfile.cmake to bin/rocalution/ for TheRock
if(ROCALUTION_CTEST_INSTALL_FILE)
  rocm_install(
    FILES "${ROCALUTION_CTEST_INSTALL_FILE}"
    DESTINATION "${CMAKE_INSTALL_BINDIR}/rocalution"
    COMPONENT tests
    RENAME "CTestTestfile.cmake"
  )
endif()

if(WIN32)
  # for now adding in all .dll as dependency chain is not cmake based on win32
  file(GLOB third_party_dlls
    LIST_DIRECTORIES OFF
    CONFIGURE_DEPENDS
    ${HIP_DIR}/bin/*.dll
    ${HIP_DIR}/bin/hipinfo.exe
    ${CMAKE_SOURCE_DIR}/rtest.*
    $ENV{rocsparse_DIR}/bin/*.dll
    $ENV{rocrand_DIR}/bin/*.dll
    $ENV{rocblas_DIR}/bin/*.dll
    C:/Windows/System32/libomp140*.dll
  )
  foreach(file_i ${third_party_dlls})
    add_custom_command(TARGET rocalution-test POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} ${PROJECT_BINARY_DIR}/staging/ COMMENT "Copying ${file_i} to staging")
  endforeach(file_i)
endif()
