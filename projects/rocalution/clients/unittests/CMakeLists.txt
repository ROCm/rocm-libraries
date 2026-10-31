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

# The unit tests link the library's object files instead of the library, so
# they can only be built together with it
if(NOT TARGET rocalution_objects)
  message(STATUS "Skipping rocalution-unit-test (requires an in-tree library build)")
  return()
endif()

# clients/tests defines GTest::gtest when it builds googletest from source
if(NOT TARGET GTest::gtest)
  find_package(GTest REQUIRED)
endif()

set(ROCALUTION_UNIT_TEST_SOURCES
  rocalution_unit_test_main.cpp
  unit_test_backend_manager.cpp
  unit_test_backend_vector.cpp
  unit_test_backend_amg.cpp
  unit_test_backend_csr_preconditioners.cpp
  unit_test_backend_matrix.cpp
  unit_test_backend_pairwise_amg.cpp
  unit_test_host_conversion.cpp
  unit_test_local_matrix.cpp
  unit_test_local_matrix_amg.cpp
  unit_test_local_vector.cpp
  unit_test_matrix_csr.cpp
  unit_test_matrix_formats.cpp
  unit_test_matrix_kernels.cpp
  unit_test_matrix_mcsr.cpp
  unit_test_multigrid.cpp
  unit_test_preconditioners.cpp
  unit_test_solvers.cpp
)

if(SUPPORT_HIP)
  list(APPEND ROCALUTION_UNIT_TEST_SOURCES unit_test_backend_hip.cpp unit_test_hip_conversion.cpp unit_test_hip_matrix_csr.cpp unit_test_hip_sparse.cpp)
endif()

add_executable(rocalution-unit-test ${ROCALUTION_UNIT_TEST_SOURCES})

target_include_directories(rocalution-unit-test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

# The test sources are part of the executable, not users of a DLL
target_compile_definitions(rocalution-unit-test PRIVATE ROCALUTION_STATIC_DEFINE)

# Never link roc::rocalution as well: that would put a second copy of the
# library's global state into the process
target_link_libraries(rocalution-unit-test PRIVATE rocalution_objects GTest::gtest)

# rocalution_hip is an object library on Windows and a shared or static
# library elsewhere. In the shared case its references into the host code resolve
# against the host objects in this executable.
if(SUPPORT_HIP)
  target_link_libraries(rocalution-unit-test PRIVATE rocalution_hip roc::rocblas roc::rocsparse)
endif()

set_target_properties(rocalution-unit-test PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging"
)

set(ROCALUTION_UNIT_TEST_LABELS "quick;standard;comprehensive;full")

# TheRock's test artifact includes bin/rocalution/** and bin/rocalution-test*,
# and ctest runs from bin/rocalution. These binaries do not match
# bin/rocalution-test*, so install them beside CTestTestfile.cmake. That is one
# directory deeper than rocalution-test, so the RPATH gains an extra "..".
function(rocalution_install_unit_test target)
  # TheRock's post-configure hook rewrites INSTALL_RPATH for every executable
  # as if it lived in bin/, unless it is told the real directory. Elsewhere these
  # are the directories rocm_install gives rocalution-test, one level up.
  set_property(TARGET ${target} PROPERTY THEROCK_INSTALL_RPATH_ORIGIN
    "${CMAKE_INSTALL_BINDIR}/rocalution")
  set_property(TARGET ${target} PROPERTY INSTALL_RPATH
    "$ORIGIN/../../lib"
    "$ORIGIN/../../lib/${PROJECT_NAME}/lib")
  install(
    TARGETS ${target}
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}/rocalution"
    COMPONENT tests)
endfunction()

add_test(NAME rocalution-unit-test COMMAND rocalution-unit-test)
set_tests_properties(rocalution-unit-test PROPERTIES LABELS "${ROCALUTION_UNIT_TEST_LABELS}" TIMEOUT 3600)

rocalution_install_unit_test(rocalution-unit-test)

# Initialization and shutdown change process-wide state, so they are tested in
# a separate executable without a global environment, one process per test
add_executable(rocalution-unit-test-lifecycle rocalution_unit_test_lifecycle.cpp)

target_include_directories(rocalution-unit-test-lifecycle PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)
target_compile_definitions(rocalution-unit-test-lifecycle PRIVATE ROCALUTION_STATIC_DEFINE)
target_link_libraries(rocalution-unit-test-lifecycle PRIVATE
  rocalution_objects
  GTest::gtest
  GTest::gtest_main
)

if(SUPPORT_HIP)
  target_link_libraries(rocalution-unit-test-lifecycle PRIVATE rocalution_hip)
endif()

# The fault injection relies on the ELF dynamic linker binding the references
# of librocalution_hip to the definitions in the executable
if(SUPPORT_HIP AND NOT WIN32)
  target_sources(rocalution-unit-test-lifecycle PRIVATE hip_fault_injection.cpp)
  target_compile_definitions(rocalution-unit-test-lifecycle PRIVATE ROCALUTION_UNIT_TEST_HIP_FAULTS)
  target_link_libraries(rocalution-unit-test-lifecycle PRIVATE
    roc::rocblas
    roc::rocsparse
    ${CMAKE_DL_LIBS}
  )
endif()

set_target_properties(rocalution-unit-test-lifecycle PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging"
)

# The library objects in the executables are instrumented, the tests are not
if(BUILD_CODE_COVERAGE)
  foreach(target rocalution-unit-test rocalution-unit-test-lifecycle)
    target_link_options(${target} PRIVATE -u__llvm_profile_runtime)
    target_link_libraries(${target} PRIVATE ${ROCALUTION_PROFILE_RUNTIME})
  endforeach()
endif()

include(GoogleTest)
gtest_discover_tests(rocalution-unit-test-lifecycle
  DISCOVERY_MODE PRE_TEST
  PROPERTIES LABELS "${ROCALUTION_UNIT_TEST_LABELS}"
)

rocalution_install_unit_test(rocalution-unit-test-lifecycle)

# The binaries are installed next to the installed CTestTestfile.cmake in bin/rocalution.
# The lifecycle test names are only known after the build, so that binary runs as one test.
if(ROCALUTION_CTEST_INSTALL_FILE)
  file(APPEND "${ROCALUTION_CTEST_INSTALL_FILE}"
    "add_test(rocalution-unit-test \"./rocalution-unit-test\")\n"
    "set_tests_properties(rocalution-unit-test PROPERTIES LABELS \"${ROCALUTION_UNIT_TEST_LABELS}\" TIMEOUT 3600)\n"
    "add_test(rocalution-unit-test-lifecycle \"./rocalution-unit-test-lifecycle\")\n"
    "set_tests_properties(rocalution-unit-test-lifecycle PROPERTIES LABELS \"${ROCALUTION_UNIT_TEST_LABELS}\" TIMEOUT 600)\n"
  )
endif()
