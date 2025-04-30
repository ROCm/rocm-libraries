# ########################################################################
# Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ########################################################################

find_package(GTest REQUIRED)

if(BUILD_HIPBLAS_TESTS)
  if(NOT TARGET hipblas)
    if(CUSTOM_HIPBLAS)
      set(ENV{hipblas_DIR} ${CUSTOM_HIPBLAS})
      find_package(hipblas REQUIRED CONFIG NO_CMAKE_PATH)
    else()
      find_package(hipblas REQUIRED CONFIG PATHS ${ROCM_PATH} /opt/rocm)
    endif()
  endif()
endif()

if(BUILD_HIPSPARSE_TESTS)
  if(NOT TARGET hipsparse)
    if(CUSTOM_HIPSPARSE)
      set(ENV{hipsparse_DIR} ${CUSTOM_HIPSPARSE})
      find_package(hipsparse REQUIRED CONFIG NO_CMAKE_PATH)
    else()
      find_package(hipsparse REQUIRED CONFIG PATHS ${ROCM_PATH} /opt/rocm)
    endif()
  endif()
endif()

set(hipsolverDn_test_source
  hipsolver_gtest_main.cpp
  getrs_gtest.cpp
  getrf_gtest.cpp
  gebrd_gtest.cpp
  gels_gtest.cpp
  geqrf_gtest.cpp
  gesv_gtest.cpp
  gesvd_gtest.cpp
  gesvda_gtest.cpp
  gesvdj_gtest.cpp
  potrf_gtest.cpp
  potri_gtest.cpp
  potrs_gtest.cpp
  syevd_heevd_gtest.cpp
  syevj_heevj_gtest.cpp
  sygvd_hegvd_gtest.cpp
  sygvj_hegvj_gtest.cpp
  syevdx_heevdx_gtest.cpp
  sygvdx_hegvdx_gtest.cpp
  sytrd_hetrd_gtest.cpp
  sytrf_gtest.cpp
  orgbr_ungbr_gtest.cpp
  orgqr_ungqr_gtest.cpp
  orgtr_ungtr_gtest.cpp
  ormqr_unmqr_gtest.cpp
  ormtr_unmtr_gtest.cpp
)

set(hipsolverSp_test_source
  csrlsvchol_gtest.cpp
  csrlsvqr_gtest.cpp
)

set(hipsolverRf_test_source
  csrrf_refactlu_gtest.cpp
  csrrf_solve_gtest.cpp
)

set(others_test_source
  determinism_gtest.cpp
  params_gtest.cpp
)

add_executable(hipsolver-test ${others_test_source} ${hipsolverDn_test_source} ${hipsolverRf_test_source})

target_include_directories(hipsolver-test
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

add_armor_flags(hipsolver-test "${ARMOR_LEVEL}")

target_compile_definitions(hipsolver-test PRIVATE GOOGLE_TEST)

# External header includes included as SYSTEM files
target_include_directories(hipsolver-test
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${CBLAS_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)

target_link_libraries(hipsolver-test PRIVATE
  ${LAPACK_LIBRARIES}
  hipsolver-common
  clients-common
  GTest::GTest
  Threads::Threads
  roc::hipsolver
)
if(BUILD_HIPBLAS_TESTS)
  target_link_libraries(hipsolver-test PRIVATE roc::hipblas)
  target_sources(hipsolver-test PRIVATE hipblas_include1_gtest.cpp hipblas_include2_gtest.cpp)
endif()
if(BUILD_HIPSPARSE_TESTS)
  target_link_libraries(hipsolver-test PRIVATE roc::hipsparse)
  target_sources(hipsolver-test PRIVATE ${hipsolverSp_test_source})
endif()
if(BUILD_FORTRAN_BINDINGS)
  target_link_libraries(hipsolver-test PRIVATE hipsolver_fortran_client)
  target_sources(hipsolver-test PRIVATE ${hipsolver_f90_source})
endif()

if(NOT USE_CUDA)
  target_link_libraries(hipsolver-test PRIVATE hip::host)

  if(CUSTOM_TARGET)
    target_link_libraries(hipsolver-test PRIVATE hip::${CUSTOM_TARGET})
  endif()

  if(CMAKE_CXX_COMPILER MATCHES ".*/hipcc$")
    # hip-clang needs specific flag to turn on pthread and m
    target_link_libraries(hipsolver-test PRIVATE -lpthread -lm)
  endif()
else()
  target_compile_definitions(hipsolver-test PRIVATE __HIP_PLATFORM_NVIDIA__)

  target_include_directories(hipsolver-test
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
  )

  target_link_libraries(hipsolver-test PRIVATE ${CUDA_LIBRARIES} Threads::Threads)
endif()

set_target_properties(hipsolver-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

rocm_install(TARGETS hipsolver-test COMPONENT tests)

if(WIN32)
  file(GLOB third_party_dlls
    LIST_DIRECTORIES OFF
    CONFIGURE_DEPENDS
    ${cblas_DIR}/bin/*.dll
    ${GTest_DIR}/bin/*.dll
    $ENV{rocblas_DIR}/bin/*.dll
    $ENV{rocsolver_DIR}/bin/*.dll
    $ENV{HIP_DIR}/bin/*.dll
    $ENV{HIP_DIR}/bin/hipinfo.exe
    ${CMAKE_SOURCE_DIR}/rtest.*
  )
  foreach(file_i ${third_party_dlls})
    add_custom_command(TARGET hipsolver-test
      POST_BUILD
      COMMAND ${CMAKE_COMMAND}
      ARGS -E copy ${file_i} ${PROJECT_BINARY_DIR}/staging/
    )
  endforeach()
  add_custom_command(TARGET hipsolver-test
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_directory $ENV{rocblas_DIR}/bin/rocblas/library ${PROJECT_BINARY_DIR}/staging/library
  )
endif()
