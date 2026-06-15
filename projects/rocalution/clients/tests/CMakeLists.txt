# ########################################################################
# Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#    test_global_stencil.cpp
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

# Add tests
add_test(rocalution-test rocalution-test)

rocm_install(TARGETS rocalution-test COMPONENT tests)

if (WIN32)
  # for now adding in all .dll as dependency chain is not cmake based on win32
  file( GLOB third_party_dlls
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
  foreach( file_i ${third_party_dlls})
    add_custom_command( TARGET rocalution-test POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} ${PROJECT_BINARY_DIR}/staging/ )
  endforeach( file_i )
endif()
