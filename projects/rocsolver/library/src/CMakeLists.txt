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

# package_targets is used as a list of install target
set(package_targets rocsolver)

set(rocsolver_lapack_source
  #### Linear Systems Solvers ####
  ################################
  #- general inverse
  lapack/roclapack_getri.cpp
  lapack/roclapack_getri_batched.cpp
  lapack/roclapack_getri_strided_batched.cpp
  lapack/roclapack_getri_outofplace.cpp
  lapack/roclapack_getri_outofplace_batched.cpp
  lapack/roclapack_getri_outofplace_strided_batched.cpp
  #- symmetric positive definite inverse
  lapack/roclapack_potri.cpp
  lapack/roclapack_potri_batched.cpp
  lapack/roclapack_potri_strided_batched.cpp
  #- triangular inverse
  lapack/roclapack_trtri.cpp
  lapack/roclapack_trtri_batched.cpp
  lapack/roclapack_trtri_strided_batched.cpp
  #- general systems
  lapack/roclapack_getrs.cpp
  lapack/roclapack_getrs_batched.cpp
  lapack/roclapack_getrs_strided_batched.cpp
  lapack/roclapack_gesv.cpp
  lapack/roclapack_gesv_batched.cpp
  lapack/roclapack_gesv_strided_batched.cpp
  lapack/roclapack_gesv_outofplace.cpp
  #- symmetric positive definite systems
  lapack/roclapack_potrs.cpp
  lapack/roclapack_potrs_batched.cpp
  lapack/roclapack_potrs_strided_batched.cpp
  lapack/roclapack_posv.cpp
  lapack/roclapack_posv_batched.cpp
  lapack/roclapack_posv_strided_batched.cpp
  #- block-tridiagonal systems
  lapack/roclapack_geblttrs_npvt.cpp
  lapack/roclapack_geblttrs_npvt_batched.cpp
  lapack/roclapack_geblttrs_npvt_strided_batched.cpp
  lapack/roclapack_geblttrs_npvt_interleaved_batched.cpp
  #### Least-Squares Solvers ####
  ###############################
  lapack/roclapack_gels.cpp
  lapack/roclapack_gels_batched.cpp
  lapack/roclapack_gels_strided_batched.cpp
  lapack/roclapack_gels_outofplace.cpp
  #### Triangular Factorizations ####
  ###################################
  #- general matrices
  lapack/roclapack_getf2.cpp
  lapack/roclapack_getf2_batched.cpp
  lapack/roclapack_getf2_strided_batched.cpp
  lapack/roclapack_getrf.cpp
  lapack/roclapack_getrf_info32.cpp
  lapack/roclapack_getrf_batched.cpp
  lapack/roclapack_getrf_strided_batched.cpp
  #- symmetric positive definite matrices
  lapack/roclapack_potf2.cpp
  lapack/roclapack_potf2_batched.cpp
  lapack/roclapack_potf2_strided_batched.cpp
  lapack/roclapack_potrf.cpp
  lapack/roclapack_potrf_info32.cpp
  lapack/roclapack_potrf_batched.cpp
  lapack/roclapack_potrf_strided_batched.cpp
  #- symmetric indefinite matrices
  lapack/roclapack_sytf2.cpp
  lapack/roclapack_sytf2_batched.cpp
  lapack/roclapack_sytf2_strided_batched.cpp
  lapack/roclapack_sytrf.cpp
  lapack/roclapack_sytrf_batched.cpp
  lapack/roclapack_sytrf_strided_batched.cpp
  #- block-tridiagonal matrices
  lapack/roclapack_geblttrf_npvt.cpp
  lapack/roclapack_geblttrf_npvt_batched.cpp
  lapack/roclapack_geblttrf_npvt_strided_batched.cpp
  lapack/roclapack_geblttrf_npvt_interleaved_batched.cpp
  #### Orthogonal Factorizations ####
  ###################################
  #- bottom row compression
  lapack/roclapack_geqr2.cpp
  lapack/roclapack_geqr2_batched.cpp
  lapack/roclapack_geqr2_strided_batched.cpp
  lapack/roclapack_geqrf.cpp
  lapack/roclapack_geqrf_batched.cpp
  lapack/roclapack_geqrf_ptr_batched.cpp
  lapack/roclapack_geqrf_strided_batched.cpp
  #- top row compression
  lapack/roclapack_geql2.cpp
  lapack/roclapack_geql2_batched.cpp
  lapack/roclapack_geql2_strided_batched.cpp
  lapack/roclapack_geqlf.cpp
  lapack/roclapack_geqlf_batched.cpp
  lapack/roclapack_geqlf_strided_batched.cpp
  #- right column compression
  lapack/roclapack_gelq2.cpp
  lapack/roclapack_gelq2_batched.cpp
  lapack/roclapack_gelq2_strided_batched.cpp
  lapack/roclapack_gelqf.cpp
  lapack/roclapack_gelqf_batched.cpp
  lapack/roclapack_gelqf_strided_batched.cpp
  #- left column compression
  lapack/roclapack_gerq2.cpp
  lapack/roclapack_gerq2_batched.cpp
  lapack/roclapack_gerq2_strided_batched.cpp
  lapack/roclapack_gerqf.cpp
  lapack/roclapack_gerqf_batched.cpp
  lapack/roclapack_gerqf_strided_batched.cpp
  #### Problem and Matrix Reductions (Diagonalizations) ####
  ##########################################################
  #- bidiagonalization
  lapack/roclapack_gebd2.cpp
  lapack/roclapack_gebd2_batched.cpp
  lapack/roclapack_gebd2_strided_batched.cpp
  lapack/roclapack_gebrd.cpp
  lapack/roclapack_gebrd_batched.cpp
  lapack/roclapack_gebrd_strided_batched.cpp
  #- tridiagonalization
  lapack/roclapack_sytd2_hetd2.cpp
  lapack/roclapack_sytd2_hetd2_batched.cpp
  lapack/roclapack_sytd2_hetd2_strided_batched.cpp
  lapack/roclapack_sytrd_hetrd.cpp
  lapack/roclapack_sytrd_hetrd_batched.cpp
  lapack/roclapack_sytrd_hetrd_strided_batched.cpp
  #- reduction of generalized eigenproblems
  lapack/roclapack_sygs2_hegs2.cpp
  lapack/roclapack_sygs2_hegs2_batched.cpp
  lapack/roclapack_sygs2_hegs2_strided_batched.cpp
  lapack/roclapack_sygst_hegst.cpp
  lapack/roclapack_sygst_hegst_batched.cpp
  lapack/roclapack_sygst_hegst_strided_batched.cpp
  #### Singular Value Decomposition ####
  ######################################
  #- classic
  lapack/roclapack_gesvd.cpp
  lapack/roclapack_gesvd_batched.cpp
  lapack/roclapack_gesvd_strided_batched.cpp
  #- divide-and-conquer
  lapack/roclapack_gesdd.cpp
  lapack/roclapack_gesdd_batched.cpp
  lapack/roclapack_gesdd_strided_batched.cpp
  # lapack/roclapack_gesdd_notransv.cpp
  # lapack/roclapack_gesdd_notransv_strided_batched.cpp
  #- jacobi
  lapack/roclapack_gesvdj.cpp
  lapack/roclapack_gesvdj_batched.cpp
  lapack/roclapack_gesvdj_strided_batched.cpp
  lapack/roclapack_gesvdj_notransv.cpp
  lapack/roclapack_gesvdj_notransv_strided_batched.cpp
  #- bisection
  lapack/roclapack_gesvdx.cpp
  lapack/roclapack_gesvdx_batched.cpp
  lapack/roclapack_gesvdx_strided_batched.cpp
  lapack/roclapack_gesvdx_notransv_strided_batched.cpp
  #### Symmetric Eigensolvers ####
  ################################
  #- classic
  lapack/roclapack_syev_heev.cpp
  lapack/roclapack_syev_heev_batched.cpp
  lapack/roclapack_syev_heev_strided_batched.cpp
  lapack/roclapack_sygv_hegv.cpp
  lapack/roclapack_sygv_hegv_batched.cpp
  lapack/roclapack_sygv_hegv_strided_batched.cpp
  #- divide & conquer
  lapack/roclapack_syevd_heevd.cpp
  lapack/roclapack_syevd_heevd_batched.cpp
  lapack/roclapack_syevd_heevd_strided_batched.cpp
  lapack/roclapack_sygvd_hegvd.cpp
  lapack/roclapack_sygvd_hegvd_batched.cpp
  lapack/roclapack_sygvd_hegvd_strided_batched.cpp
  #- jacobi
  lapack/roclapack_syevj_heevj.cpp
  lapack/roclapack_syevj_heevj_batched.cpp
  lapack/roclapack_syevj_heevj_strided_batched.cpp
  lapack/roclapack_sygvj_hegvj.cpp
  lapack/roclapack_sygvj_hegvj_batched.cpp
  lapack/roclapack_sygvj_hegvj_strided_batched.cpp
  #- bisection
  lapack/roclapack_syevx_heevx.cpp
  lapack/roclapack_syevx_heevx_batched.cpp
  lapack/roclapack_syevx_heevx_strided_batched.cpp
  lapack/roclapack_sygvx_hegvx.cpp
  lapack/roclapack_sygvx_hegvx_batched.cpp
  lapack/roclapack_sygvx_hegvx_strided_batched.cpp
  #- d&c + jacobi
  lapack/roclapack_syevdj_heevdj.cpp
  lapack/roclapack_syevdj_heevdj_batched.cpp
  lapack/roclapack_syevdj_heevdj_strided_batched.cpp
  lapack/roclapack_sygvdj_hegvdj.cpp
  lapack/roclapack_sygvdj_hegvdj_batched.cpp
  lapack/roclapack_sygvdj_hegvdj_strided_batched.cpp
  #- d&c + bisection
  lapack/roclapack_syevdx_heevdx.cpp
  lapack/roclapack_syevdx_heevdx_batched.cpp
  lapack/roclapack_syevdx_heevdx_strided_batched.cpp
  lapack/roclapack_sygvdx_hegvdx.cpp
  lapack/roclapack_sygvdx_hegvdx_batched.cpp
  lapack/roclapack_sygvdx_hegvdx_strided_batched.cpp
  lapack/roclapack_syevdx_heevdx_inplace.cpp
  lapack/roclapack_sygvdx_hegvdx_inplace.cpp
)

set(rocsolver_auxiliary_source
  # vector & matrix manipulations
  auxiliary/rocauxiliary_aliases.cpp
  auxiliary/rocauxiliary_lacgv.cpp
  auxiliary/rocauxiliary_laswp.cpp
  # householder reflections
  auxiliary/rocauxiliary_larfg.cpp
  auxiliary/rocauxiliary_larf.cpp
  auxiliary/rocauxiliary_larft.cpp
  auxiliary/rocauxiliary_larfb.cpp
  # plane rotations
  auxiliary/rocauxiliary_lasr.cpp
  # orthonormal/unitary matrices
  auxiliary/rocauxiliary_org2r_ung2r.cpp
  auxiliary/rocauxiliary_orgqr_ungqr.cpp
  auxiliary/rocauxiliary_orgl2_ungl2.cpp
  auxiliary/rocauxiliary_orglq_unglq.cpp
  auxiliary/rocauxiliary_org2l_ung2l.cpp
  auxiliary/rocauxiliary_orgql_ungql.cpp
  auxiliary/rocauxiliary_orgbr_ungbr.cpp
  auxiliary/rocauxiliary_orgtr_ungtr.cpp
  auxiliary/rocauxiliary_orm2r_unm2r.cpp
  auxiliary/rocauxiliary_ormqr_unmqr.cpp
  auxiliary/rocauxiliary_orml2_unml2.cpp
  auxiliary/rocauxiliary_ormlq_unmlq.cpp
  auxiliary/rocauxiliary_orm2l_unm2l.cpp
  auxiliary/rocauxiliary_ormql_unmql.cpp
  auxiliary/rocauxiliary_ormbr_unmbr.cpp
  auxiliary/rocauxiliary_ormtr_unmtr.cpp
  # bidiagonal matrices
  auxiliary/rocauxiliary_bdsqr.cpp
  auxiliary/rocauxiliary_bdsvdx.cpp
  auxiliary/rocauxiliary_labrd.cpp
  # tridiagonal matrices
  auxiliary/rocauxiliary_sterf.cpp
  auxiliary/rocauxiliary_stebz.cpp
  auxiliary/rocauxiliary_steqr.cpp
  auxiliary/rocauxiliary_stedc.cpp
  auxiliary/rocauxiliary_stedcj.cpp
  auxiliary/rocauxiliary_stedcx.cpp
  auxiliary/rocauxiliary_stein.cpp
  auxiliary/rocauxiliary_latrd.cpp
  # symmetric matrices
  auxiliary/rocauxiliary_lasyf.cpp
  # triangular matrices
  auxiliary/rocauxiliary_lauum.cpp
)

set(rocsolver_refact_source
  # rfinfo create, analysis, destroy
  refact/rocsolver_rfinfo.cpp
  refact/rocrefact_csrrf_analysis.cpp
  # re-factorization
  refact/rocrefact_csrrf_sumlu.cpp
  refact/rocrefact_csrrf_splitlu.cpp
  refact/rocrefact_csrrf_refactlu.cpp
  refact/rocrefact_csrrf_refactchol.cpp
  # direct solver
  refact/rocrefact_csrrf_solve.cpp
)

set(rocsolver_specialized_source
  # larf
  specialized/rocauxiliary_larf_specialized_kernels_s.cpp
  specialized/rocauxiliary_larf_specialized_kernels_d.cpp
  specialized/rocauxiliary_larf_specialized_kernels_c.cpp
  specialized/rocauxiliary_larf_specialized_kernels_z.cpp
  # larfg
  specialized/rocauxiliary_larfg_specialized_kernels_s.cpp
  specialized/rocauxiliary_larfg_specialized_kernels_d.cpp
  specialized/rocauxiliary_larfg_specialized_kernels_c.cpp
  specialized/rocauxiliary_larfg_specialized_kernels_z.cpp
  # gemm
  specialized/roclapack_gemm_specialized_kernels_s.cpp
  specialized/roclapack_gemm_specialized_kernels_d.cpp
  specialized/roclapack_gemm_specialized_kernels_c.cpp
  specialized/roclapack_gemm_specialized_kernels_z.cpp
  # ger
  specialized/roclapack_ger_specialized_kernels_s.cpp
  specialized/roclapack_ger_specialized_kernels_d.cpp
  specialized/roclapack_ger_specialized_kernels_c.cpp
  specialized/roclapack_ger_specialized_kernels_z.cpp
  # trsm
  specialized/roclapack_trsm_specialized_kernels_s.cpp
  specialized/roclapack_trsm_specialized_kernels_d.cpp
  specialized/roclapack_trsm_specialized_kernels_c.cpp
  specialized/roclapack_trsm_specialized_kernels_z.cpp
  # potf2
  specialized/roclapack_potf2_specialized_kernels_s.cpp
  specialized/roclapack_potf2_specialized_kernels_d.cpp
  specialized/roclapack_potf2_specialized_kernels_c.cpp
  specialized/roclapack_potf2_specialized_kernels_z.cpp
)

if(OPTIMAL)
  list(APPEND rocsolver_specialized_source
    # getf2
    specialized/roclapack_getf2_specialized_kernels_s.cpp
    specialized/roclapack_getf2_specialized_kernels_d.cpp
    specialized/roclapack_getf2_specialized_kernels_c.cpp
    specialized/roclapack_getf2_specialized_kernels_z.cpp
    specialized/roclapack_getf2_small_s.cpp
    specialized/roclapack_getf2_small_d.cpp
    specialized/roclapack_getf2_small_c.cpp
    specialized/roclapack_getf2_small_z.cpp
    specialized/roclapack_getf2_small_sb.cpp
    specialized/roclapack_getf2_small_db.cpp
    specialized/roclapack_getf2_small_cb.cpp
    specialized/roclapack_getf2_small_zb.cpp
    # getri
    specialized/roclapack_getri_specialized_kernels_s.cpp
    specialized/roclapack_getri_specialized_kernels_d.cpp
    specialized/roclapack_getri_specialized_kernels_c.cpp
    specialized/roclapack_getri_specialized_kernels_z.cpp
    # trtri
    specialized/roclapack_trtri_specialized_kernels_s.cpp
    specialized/roclapack_trtri_specialized_kernels_d.cpp
    specialized/roclapack_trtri_specialized_kernels_c.cpp
    specialized/roclapack_trtri_specialized_kernels_z.cpp
  )
endif()

set(auxiliaries
  common/buildinfo.cpp
  common/rocsolver_handle.cpp
  common/rocsolver_logger.cpp
  common/rocsparse.cpp
)

set(common_source
  ${PROJECT_SOURCE_DIR}/common/src/common_host_helpers.cpp
)

if(USE_HIPCXX)
  set_source_files_properties(
    ${common_source}
    ${auxiliaries}
    ${rocsolver_specialized_source}
    ${rocsolver_auxiliary_source}
    ${rocsolver_lapack_source}
    ${rocsolver_refact_source}
    PROPERTIES
      LANGUAGE HIP
  )
endif()

add_library(rocsolver
  ${common_source}
  ${auxiliaries}
  ${rocsolver_specialized_source}
  ${rocsolver_auxiliary_source}
  ${rocsolver_lapack_source}
  ${rocsolver_refact_source}
)

add_library(roc::rocsolver ALIAS rocsolver)

target_link_libraries(rocsolver
  PUBLIC
    roc::rocblas
  PRIVATE
    $<BUILD_INTERFACE:rocsolver-common> # https://gitlab.kitware.com/cmake/cmake/-/issues/15415
    roc::rocprim
    $<$<PLATFORM_ID:Linux>:--rtlib=compiler-rt>
    $<$<PLATFORM_ID:Linux>:--unwindlib=libgcc>
)
if(NOT USE_HIPCXX)
  target_link_libraries(rocsolver PRIVATE hip::device)
endif()

if(NOT BUILD_WITH_SPARSE)
  target_link_libraries(rocsolver PRIVATE $<$<PLATFORM_ID:Linux>:${CMAKE_DL_LIBS}>)
endif()

set(static_depends)

if(BUILD_WITH_SPARSE)
  target_link_libraries(rocsolver PRIVATE roc::rocsparse)
  list(APPEND static_depends PACKAGE rocsparse)
  set_source_files_properties(${rocsolver_refact_source}
    PROPERTIES
      COMPILE_DEFINITIONS HAVE_ROCSPARSE
  )
endif()

if(ROCSOLVER_EMBED_FMT)
  target_link_libraries(rocsolver PRIVATE $<BUILD_INTERFACE:fmt::fmt-header-only>)
else()
  list(APPEND static_depends PACKAGE fmt)
  target_link_libraries(rocsolver PRIVATE fmt::fmt)
endif()

# In ROCm 4.0 and earlier, the default maximum threads per block is 256
if(USE_HIPCXX)
  target_compile_options(rocsolver PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--gpu-max-threads-per-block=1024>)
else()
  target_compile_options(rocsolver PRIVATE --gpu-max-threads-per-block=1024)
endif()

# Ignore loop unrolling failures
target_compile_options(rocsolver PRIVATE -Wno-pass-failed)

# hipcc adds these options automatically. These options might have an effect
# on performance, so add them when building with clang directly, just in case.
if(USE_HIPCXX)
  target_compile_options(rocsolver PRIVATE
    "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-mllvm -amdgpu-early-inline-all=true>"
    "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-mllvm -amdgpu-function-calls=false>"
  )
elseif(CMAKE_CXX_COMPILER MATCHES ".*clang\\+\\+.*")
  target_compile_options(rocsolver PRIVATE
    "SHELL:-mllvm -amdgpu-early-inline-all=true"
    "SHELL:-mllvm -amdgpu-function-calls=false"
  )
endif()

if(BUILD_OFFLOAD_COMPRESS)
  target_compile_options(rocsolver PRIVATE "--offload-compress")
endif()

target_include_directories(rocsolver
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/rocsolver>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(rocsolver PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
rocm_set_soversion(rocsolver ${rocsolver_SOVERSION})

# set visibility for function names exported from shared library
include(GenerateExportHeader)
set_target_properties(rocsolver PROPERTIES
  CXX_VISIBILITY_PRESET "hidden"
  VISIBILITY_INLINES_HIDDEN ON
)
generate_export_header(rocsolver EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/rocsolver/rocsolver-export.h)

if(NOT BUILD_SHARED_LIBS)
  # Following Boost conventions of prefixing 'lib' on static built libraries, across all platforms
  set_target_properties(rocsolver PROPERTIES PREFIX "lib")
  target_compile_definitions(rocsolver PRIVATE ROCSOLVER_STATIC_LIB)
endif()

if(OPTIMAL)
  target_compile_definitions(rocsolver PRIVATE OPTIMAL)
endif()

if(ROCSOLVER_USE_INTERNAL_BLAS)
  target_compile_definitions(rocsolver PRIVATE USE_INTERNAL_GEMM USE_INTERNAL_TRSM)
endif()

if(ROCSOLVER_USE_REFERENCE_SECULAR_EQUATIONS_SOLVER)
  target_compile_definitions(rocsolver PRIVATE ROCSOLVER_USE_REFERENCE_SECULAR_EQUATIONS_SOLVER)
endif()

target_compile_definitions(rocsolver PRIVATE
  ROCM_USE_FLOAT16
  ROCBLAS_INTERNAL_API
  ROCSOLVER_LIBRARY
)

add_armor_flags(rocsolver "${ARMOR_LEVEL}")

if(WIN32)
  if(BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_TESTS)
    add_custom_command(TARGET rocsolver
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_BINARY_DIR}/staging/$<TARGET_FILE_NAME:rocsolver>" "${PROJECT_BINARY_DIR}/clients/staging/$<TARGET_FILE_NAME:rocsolver>"
    )
    if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
      add_custom_command(TARGET rocsolver
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_BINARY_DIR}/staging/rocsolver.pdb" "${PROJECT_BINARY_DIR}/clients/staging/rocsolver.pdb"
      )
    endif()
  endif()
endif()

############################################################
# Installation

rocm_install(
	DIRECTORY
		"${PROJECT_SOURCE_DIR}/library/include/rocsolver/"
		"${PROJECT_BINARY_DIR}/include/rocsolver/"
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocsolver
	FILES_MATCHING
	PATTERN "*.h"
	PATTERN "*.hpp"
	PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
)

rocm_install_targets(
  TARGETS ${package_targets}
  INCLUDE
    ${CMAKE_SOURCE_DIR}/library/include
    ${CMAKE_BINARY_DIR}/include
)

rocm_export_targets(
  TARGETS roc::rocsolver
  DEPENDS
    PACKAGE hip
    PACKAGE rocblas
  STATIC_DEPENDS
    ${static_depends}
  NAMESPACE roc::
)

