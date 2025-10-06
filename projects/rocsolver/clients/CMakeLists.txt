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

if(UNIX)
  enable_language(Fortran)
endif()

# Specify where to put the client binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/staging")

# The rocsolver target will exist if the library is being built along with the clients,
# but if this is a clients-only build, we'll have to search for it.
if(NOT TARGET rocsolver)
  find_package(rocsolver REQUIRED CONFIG PATHS ${ROCM_PATH}/rocsolver /opt/rocm/rocsolver)
  get_imported_target_location(location roc::rocsolver)
  message(STATUS "Found rocSOLVER: ${location}")
endif()

if(BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_TESTS)
  if(ROCSOLVER_FIND_PACKAGE_LAPACK_CONFIG)
    find_package(LAPACK 3.7 REQUIRED CONFIG)
  else()
    find_package(LAPACK 3.7 REQUIRED)
  endif()

  if(NOT LAPACK_LIBRARIES)
    set(LAPACK_LIBRARIES
      ${LAPACK_blas_LIBRARIES}
      ${LAPACK_lapack_LIBRARIES}
    )
  endif()

  add_library(clients-common INTERFACE)
  target_include_directories(clients-common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../library/src/include>
  )
  target_link_libraries(clients-common INTERFACE
    ${LAPACK_LIBRARIES}
    $<$<PLATFORM_ID:Linux>:stdc++fs>
    fmt::fmt-header-only
  )
  target_link_options(clients-common INTERFACE
    ${LAPACK_LINKER_FLAGS}
  )

  set(rocauxiliary_inst_files
    common/auxiliary/testing_lacgv.cpp
    common/auxiliary/testing_laswp.cpp
    common/auxiliary/testing_larfg.cpp
    common/auxiliary/testing_larf.cpp
    common/auxiliary/testing_larft.cpp
    common/auxiliary/testing_larfb.cpp
    common/auxiliary/testing_lasr.cpp
    common/auxiliary/testing_latrd.cpp
    common/auxiliary/testing_labrd.cpp
    common/auxiliary/testing_lauum.cpp
    common/auxiliary/testing_bdsqr.cpp
    common/auxiliary/testing_bdsvdx.cpp
    common/auxiliary/testing_steqr.cpp
    common/auxiliary/testing_stedc.cpp
    common/auxiliary/testing_stedcj.cpp
    common/auxiliary/testing_stedcx.cpp
    common/auxiliary/testing_stein.cpp
    common/auxiliary/testing_lasyf.cpp
    common/auxiliary/testing_sterf.cpp
    common/auxiliary/testing_stebz.cpp
    common/auxiliary/testing_orgxr_ungxr.cpp
    common/auxiliary/testing_orgxl_ungxl.cpp
    common/auxiliary/testing_orglx_unglx.cpp
    common/auxiliary/testing_orgbr_ungbr.cpp
    common/auxiliary/testing_orgtr_ungtr.cpp
    common/auxiliary/testing_ormxr_unmxr.cpp
    common/auxiliary/testing_ormxl_unmxl.cpp
    common/auxiliary/testing_ormlx_unmlx.cpp
    common/auxiliary/testing_ormbr_unmbr.cpp
    common/auxiliary/testing_ormtr_unmtr.cpp
  )

  set(roclapack_inst_files
    common/lapack/testing_potf2_potrf.cpp
    common/lapack/testing_potrs.cpp
    common/lapack/testing_posv.cpp
    common/lapack/testing_potri.cpp
    common/lapack/testing_getf2_getrf_npvt.cpp
    common/lapack/testing_getf2_getrf.cpp
    common/lapack/testing_geqr2_geqrf.cpp
    common/lapack/testing_gerq2_gerqf.cpp
    common/lapack/testing_geql2_geqlf.cpp
    common/lapack/testing_gelq2_gelqf.cpp
    common/lapack/testing_getrs.cpp
    common/lapack/testing_gesv.cpp
    common/lapack/testing_gesvd.cpp
    common/lapack/testing_gesdd.cpp
    common/lapack/testing_gesvdj.cpp
    common/lapack/testing_gesvdx.cpp
    common/lapack/testing_trtri.cpp
    common/lapack/testing_getri.cpp
    common/lapack/testing_getri_npvt.cpp
    common/lapack/testing_getri_outofplace.cpp
    common/lapack/testing_getri_npvt_outofplace.cpp
    common/lapack/testing_gels.cpp
    common/lapack/testing_gebd2_gebrd.cpp
    common/lapack/testing_sytf2_sytrf.cpp
    common/lapack/testing_sytxx_hetxx.cpp
    common/lapack/testing_sygsx_hegsx.cpp
    common/lapack/testing_syev_heev.cpp
    common/lapack/testing_syevd_heevd.cpp
    common/lapack/testing_syevdj_heevdj.cpp
    common/lapack/testing_syevdx_heevdx.cpp
    common/lapack/testing_syevj_heevj.cpp
    common/lapack/testing_syevx_heevx.cpp
    common/lapack/testing_sygv_hegv.cpp
    common/lapack/testing_sygvd_hegvd.cpp
    common/lapack/testing_sygvdj_hegvdj.cpp
    common/lapack/testing_sygvdx_hegvdx.cpp
    common/lapack/testing_sygvj_hegvj.cpp
    common/lapack/testing_sygvx_hegvx.cpp
    common/lapack/testing_geblttrf_npvt.cpp
    common/lapack/testing_geblttrf_npvt_interleaved.cpp
    common/lapack/testing_geblttrs_npvt.cpp
    common/lapack/testing_geblttrs_npvt_interleaved.cpp
  )

  set(rocrefact_inst_files
    common/refact/testing_csrrf_analysis.cpp
    common/refact/testing_csrrf_sumlu.cpp
    common/refact/testing_csrrf_splitlu.cpp
    common/refact/testing_csrrf_refactlu.cpp
    common/refact/testing_csrrf_refactchol.cpp
    common/refact/testing_csrrf_solve.cpp
  )

  set(rocunit_inst_files
    common/unit/testing_gemm.cpp
  )

  set(common_source_files
    common/misc/lapack_host_reference.cpp
    common/misc/rocsolver_test.cpp
    common/misc/clients_utility.cpp
    common/misc/program_options.cpp
    common/misc/client_environment_helpers.cpp
    common/matrix_utils/matrix_utils.cpp
    ${rocauxiliary_inst_files}
    ${roclapack_inst_files}
    ${rocrefact_inst_files}
    ${rocunit_inst_files}
  )

  prepend_path("${CMAKE_CURRENT_SOURCE_DIR}/" common_source_files common_source_paths)

  # Copy and point to sparse test data
  file(COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/sparsedata/
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/staging/sparsedata/
  )
  install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/sparsedata/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/rocsolver/test
    COMPONENT tests
  )

  target_compile_definitions(clients-common INTERFACE
    -DROCBLAS_NO_DEPRECATED_WARNINGS
  )

  if(BUILD_CLIENTS_BENCHMARKS)
    add_subdirectory(benchmarks)
  endif()

  if(BUILD_CLIENTS_TESTS)
    add_subdirectory(gtest)
  endif()
endif()

if(BUILD_CLIENTS_SAMPLES)
  add_subdirectory(samples)
endif()

if(BUILD_CLIENTS_EXTRA_TESTS)
  add_subdirectory(extras)
endif()
