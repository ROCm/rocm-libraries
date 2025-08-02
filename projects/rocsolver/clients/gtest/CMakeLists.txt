# ##########################################################################
# Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
# ##########################################################################

find_package(GTest REQUIRED)

set(roclapack_test_source
  # linear systems solvers
  lapack/getri_gtest.cpp
  lapack/getrs_gtest.cpp
  lapack/gesv_gtest.cpp
  lapack/potrs_gtest.cpp
  lapack/posv_gtest.cpp
  lapack/potri_gtest.cpp
  lapack/trtri_gtest.cpp
  lapack/geblttrs_gtest.cpp
  # least squares solvers
  lapack/gels_gtest.cpp
  # triangular factorizations
  lapack/getf2_getrf_gtest.cpp
  lapack/getrf_large_gtest.cpp
  lapack/potf2_potrf_gtest.cpp
  lapack/sytf2_sytrf_gtest.cpp
  lapack/geblttrf_gtest.cpp
  # orthogonal factorizations
  lapack/geqr2_geqrf_gtest.cpp
  lapack/gerq2_gerqf_gtest.cpp
  lapack/geql2_geqlf_gtest.cpp
  lapack/gelq2_gelqf_gtest.cpp
  # problem and matrix reductions (diagonalizations)
  lapack/gebd2_gebrd_gtest.cpp
  lapack/sytxx_hetxx_gtest.cpp
  lapack/sygsx_hegsx_gtest.cpp
  # singular value decomposition
  lapack/gesvd_gtest.cpp
  lapack/gesdd_gtest.cpp
  lapack/gesvdj_gtest.cpp
  lapack/gesvdx_gtest.cpp
  # symmetric eigensolvers
  lapack/syev_heev_gtest.cpp
  lapack/syevd_heevd_gtest.cpp
  lapack/sygv_hegv_gtest.cpp
  lapack/sygvd_hegvd_gtest.cpp
  lapack/syevj_heevj_gtest.cpp
  lapack/syevdj_heevdj_gtest.cpp
  lapack/sygvj_hegvj_gtest.cpp
  lapack/sygvdj_hegvdj_gtest.cpp
  lapack/syevx_heevx_gtest.cpp
  lapack/syevdx_heevdx_gtest.cpp
  lapack/sygvx_hegvx_gtest.cpp
  lapack/sygvdx_hegvdx_gtest.cpp
)

set(rocauxiliary_test_source
  # vector & matrix manipulations
  auxiliary/lacgv_gtest.cpp
  auxiliary/laswp_gtest.cpp
  # householder reflections
  auxiliary/larf_gtest.cpp
  auxiliary/larfg_gtest.cpp
  auxiliary/larft_gtest.cpp
  auxiliary/larfb_gtest.cpp
  # plane rotations
  auxiliary/lasr_gtest.cpp
  # orthonormal/unitary matrices
  auxiliary/orgxr_ungxr_gtest.cpp
  auxiliary/orglx_unglx_gtest.cpp
  auxiliary/orgxl_ungxl_gtest.cpp
  auxiliary/orgbr_ungbr_gtest.cpp
  auxiliary/orgtr_ungtr_gtest.cpp
  auxiliary/ormxr_unmxr_gtest.cpp
  auxiliary/ormlx_unmlx_gtest.cpp
  auxiliary/ormxl_unmxl_gtest.cpp
  auxiliary/ormbr_unmbr_gtest.cpp
  auxiliary/ormtr_unmtr_gtest.cpp
  # bidiagonal matrices
  auxiliary/labrd_gtest.cpp
  auxiliary/bdsqr_gtest.cpp
  auxiliary/bdsvdx_gtest.cpp
  # tridiagonal matrices
  auxiliary/sterf_gtest.cpp
  auxiliary/steqr_gtest.cpp
  auxiliary/stedc_gtest.cpp
  auxiliary/stedcj_gtest.cpp
  auxiliary/stedcx_gtest.cpp
  auxiliary/stebz_gtest.cpp
  auxiliary/stein_gtest.cpp
  auxiliary/latrd_gtest.cpp
  # symmetric matrices
  auxiliary/lasyf_gtest.cpp
  # triangular matrices
  auxiliary/lauum_gtest.cpp
)

set(rocrefact_test_source
  # rfinfo analysis
  refact/csrrf_analysis_gtest.cpp
  refact/csrrf_workflow_gtest.cpp
  # lu refactorization
  refact/csrrf_sumlu_gtest.cpp
  refact/csrrf_splitlu_gtest.cpp
  refact/csrrf_refactlu_gtest.cpp
  refact/csrrf_refactchol_gtest.cpp
  # sparse solver
  refact/csrrf_solve_gtest.cpp
)

set(rocunit_test_source
  # gemm
  unit/gemm_gtest.cpp
)

set(others_test_source
  # unified memory model
  managed_malloc_gtest.cpp
  # rocblas memory model
  memory_model_gtest.cpp
  # rocsolver logging
  logging_gtest.cpp
  # helpers
  #common/client_environment_helpers.cpp
)

set(rocsolver_test_source
  rocsolver_gtest_main.cpp
)

set(rocsolver_test_common_source
  ${PROJECT_SOURCE_DIR}/common/src/common_host_helpers.cpp
)

if(USE_HIPCXX)
  set_source_files_properties(
    ${common_source_paths}
    ${rocsolver_test_common_source}
    ${roclapack_test_source}
    ${rocauxiliary_test_source}
    ${rocrefact_test_source}
    ${others_test_source}
    ${rocsolver_test_source}
    PROPERTIES
      LANGUAGE HIP
  )
endif()

add_executable(rocsolver-test
  ${common_source_paths}
  ${rocsolver_test_common_source}
  ${roclapack_test_source}
  ${rocauxiliary_test_source}
  ${rocrefact_test_source}
  ${rocunit_test_source}
  ${others_test_source}
  ${rocsolver_test_source}
)

add_armor_flags(rocsolver-test "${ARMOR_LEVEL}")

if(WIN32)
  file(GLOB third_party_dlls
    LIST_DIRECTORIES OFF
    CONFIGURE_DEPENDS
    ${ROCSOLVER_LAPACK_PATH}/bin/*.dll
    ${GTest_DIR}/bin/*.dll
    $ENV{rocblas_DIR}/bin/*.dll
    $ENV{HIP_DIR}/bin/*.dll
    $ENV{HIP_DIR}/bin/hipinfo.exe
    ${CMAKE_SOURCE_DIR}/rtest.*
  )
  foreach(file_i ${third_party_dlls})
    add_custom_command(TARGET rocsolver-test
      POST_BUILD
      COMMAND ${CMAKE_COMMAND}
      ARGS -E copy ${file_i} $<TARGET_FILE_DIR:rocsolver-test>
    )
  endforeach()
  if(DEFINED ENV{rocblas_DIR})
    add_custom_command(TARGET rocsolver-test
      POST_BUILD
      COMMAND ${CMAKE_COMMAND}
      ARGS -E copy_directory $ENV{rocblas_DIR}/bin/rocblas/library $<TARGET_FILE_DIR:rocsolver-test>/library
    )
  else()
    message(WARNING "rocblas_DIR not set. rocBLAS Tensile runtime kernels will not be copied to client staging directory (insitu testing affected).")
  endif()
endif()

target_link_libraries(rocsolver-test PRIVATE
  $<IF:$<TARGET_EXISTS:GTest::gtest>,GTest::gtest,GTest::GTest>
  hip::host
  rocsolver-common
  clients-common
  $<$<PLATFORM_ID:Linux>:stdc++fs>
  $<$<PLATFORM_ID:Linux>:m>
  roc::rocsolver
  roc::rocblas
)
if(NOT USE_HIPCXX)
  target_link_libraries(rocsolver-test PRIVATE hip::device)
endif()

target_compile_definitions(rocsolver-test PRIVATE
  ROCM_USE_FLOAT16
  ROCSOLVER_CLIENTS_TEST
)

add_test(
  NAME rocsolver-test
  COMMAND rocsolver-test
)

rocm_install(TARGETS rocsolver-test COMPONENT tests)
