# ########################################################################
# Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
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


# ########################################################################
# Main
# ########################################################################

# package_targets is used as a list of install target
set( package_targets rocblas )


# Set up Tensile Dependency
if( BUILD_WITH_TENSILE )
  # If we want to build a shared rocblas lib, force Tensile to build as a static lib to absorb into rocblas
  if( BUILD_SHARED_LIBS )
    set( ROCBLAS_SHARED_LIBS ON )
    set( BUILD_SHARED_LIBS OFF )
  else( )
    set( ROCBLAS_SHARED_LIBS OFF )
  endif( )

  set( Tensile_RUNTIME_LANGUAGE "HIP" )

  #TODO update when this feature has been validated
  #set( PACKAGE_TENSILE_LIBRARY ON )
  set( PACKAGE_TENSILE_LIBRARY OFF )

  # Build options list
  if(Tensile_MERGE_FILES)
    set(Tensile_Options ${Tensile_Options} MERGE_FILES)
  endif()
  if(Tensile_SEPARATE_ARCHITECTURES)
    set(Tensile_Options ${Tensile_Options} SEPARATE_ARCHITECTURES)
  endif()
  if(Tensile_LAZY_LIBRARY_LOADING)
  set(Tensile_Options ${Tensile_Options} LAZY_LIBRARY_LOADING)
  endif()
  if(Tensile_SHORT_FILENAMES)
    set(Tensile_Options ${Tensile_Options} SHORT_FILE_NAMES)
  endif()
  if(Tensile_PRINT_DEBUG)
    set(Tensile_Options ${Tensile_Options} PRINT_DEBUG)
  endif()
  if(PACKAGE_TENSILE_LIBRARY)
    set(Tensile_Options ${Tensile_Options} GENERATE_PACKAGE)
  endif()

  # Add a build target for Tensile kernel library
  # Runtime language is HIP by default
  # warning our Tensile_ variables may shadow variable in TensileCreateLibraryFiles
  # thus bypassing the function argument parameter system (mainly the options list) and CPU_THREADS
  if(Tensile_CPU_THREADS MATCHES "^[0-9]+$")
    # only including threads argument if number
    TensileCreateLibraryFiles(
      "${CMAKE_CURRENT_SOURCE_DIR}/blas3/Tensile/Logic/${Tensile_LOGIC}"
      "${PROJECT_BINARY_DIR}/Tensile"
      ARCHITECTURE        ${Tensile_ARCHITECTURE}
      CODE_OBJECT_VERSION ${Tensile_CODE_OBJECT_VERSION}
      COMPILER            ${Tensile_COMPILER}
      LIBRARY_FORMAT      ${Tensile_LIBRARY_FORMAT}
      CPU_THREADS         ${Tensile_CPU_THREADS}
      ${Tensile_Options}
    )
  else()
    TensileCreateLibraryFiles(
      "${CMAKE_CURRENT_SOURCE_DIR}/blas3/Tensile/Logic/${Tensile_LOGIC}"
      "${PROJECT_BINARY_DIR}/Tensile"
      ARCHITECTURE        ${Tensile_ARCHITECTURE}
      CODE_OBJECT_VERSION ${Tensile_CODE_OBJECT_VERSION}
      COMPILER            ${Tensile_COMPILER}
      LIBRARY_FORMAT      ${Tensile_LIBRARY_FORMAT}
      ${Tensile_Options}
    )
  endif()

  # Create a unique name for TensileHost compiled for rocBLAS
  set_target_properties( TensileHost PROPERTIES OUTPUT_NAME rocblas-tensile CXX_EXTENSIONS NO )

  # Tensile host depends on libs build target
  add_dependencies( TensileHost TENSILE_LIBRARY_TARGET )

  if( ROCBLAS_SHARED_LIBS )
    set( BUILD_SHARED_LIBS ON )
    set_target_properties( TensileHost PROPERTIES POSITION_INDEPENDENT_CODE ON )
  endif()

  set( Tensile_SRC
    tensile_host.cpp
  )

  if(BUILD_WITH_HIPBLASLT)
    set( HipBlasLT_SRC
      hipblaslt_host.cpp
    )
  endif()

  set( rocblas_ex_source
    ${Tensile_SRC}
    ${HipBlasLT_SRC}
  )

endif() # BUILD_WITH_TENSILE

# tensile includes have internal guards for BUILD_WITH_TENSILE to allow source gemm
set( Tensile_INC
  ${CMAKE_CURRENT_SOURCE_DIR}/blas3/Tensile
)

set(rocblas_ex_source_no_tensile
    blas_ex/rocblas_axpy_ex.cpp
    blas_ex/rocblas_axpy_ex_kernels.cpp
    blas_ex/rocblas_axpy_batched_ex.cpp
    blas_ex/rocblas_axpy_strided_batched_ex.cpp
    blas_ex/rocblas_dot_ex.cpp
    blas_ex/rocblas_dot_ex_kernels.cpp
    blas_ex/rocblas_dot_batched_ex.cpp
    blas_ex/rocblas_dot_strided_batched_ex.cpp
    blas_ex/rocblas_rot_ex.cpp
    blas_ex/rocblas_rot_ex_kernels.cpp
    blas_ex/rocblas_rot_batched_ex.cpp
    blas_ex/rocblas_rot_strided_batched_ex.cpp
    blas_ex/rocblas_scal_ex.cpp
    blas_ex/rocblas_scal_ex_kernels.cpp
    blas_ex/rocblas_scal_batched_ex.cpp
    blas_ex/rocblas_scal_strided_batched_ex.cpp
    blas_ex/rocblas_nrm2_ex.cpp
    blas_ex/rocblas_nrm2_ex_kernels.cpp
    blas_ex/rocblas_nrm2_batched_ex.cpp
    blas_ex/rocblas_nrm2_strided_batched_ex.cpp
    blas_ex/rocblas_geam_ex.cpp
    blas_ex/rocblas_geam_ex_kernels.cpp
    blas_ex/rocblas_gemmt.cpp
    blas_ex/rocblas_gemmt_batched.cpp
    blas_ex/rocblas_gemmt_strided_batched.cpp
    blas_ex/rocblas_gemmt_kernels.cpp

    # these require may use Tensile or source gemm
    blas_ex/rocblas_gemm_ex.cpp
    blas_ex/rocblas_gemm_batched_ex.cpp
    blas_ex/rocblas_gemm_strided_batched_ex.cpp
    blas_ex/rocblas_gemm_ex_kernels.cpp
    blas_ex/rocblas_trsv_ex.cpp
    blas_ex/rocblas_trsv_strided_batched_ex.cpp
    blas_ex/rocblas_trsv_batched_ex.cpp
)

set( rocblas_blas3_source_no_tensile
    blas3/rocblas_dgmm.cpp
    blas3/rocblas_dgmm_kernels.cpp
    blas3/rocblas_dgmm_batched.cpp
    blas3/rocblas_dgmm_strided_batched.cpp
    blas3/rocblas_geam.cpp
    blas3/rocblas_geam_kernels.cpp
    blas3/rocblas_geam_batched.cpp
    blas3/rocblas_geam_strided_batched.cpp
)

# rocblas L3 that use tensile but can use source gemm as fallback
set( rocblas_blas3_source
    # trsm first as currently slowest
    blas3/rocblas_trsm.cpp
    blas3/rocblas_trsm_batched.cpp
    blas3/rocblas_trsm_strided_batched.cpp
    blas3/rocblas_trsm_kernels.cpp
    blas3/rocblas_trsm_batched_kernels.cpp
    #
    blas3/rocblas_hemm.cpp
    blas3/rocblas_hemm_batched.cpp
    blas3/rocblas_hemm_strided_batched.cpp
    blas3/rocblas_herk.cpp
    blas3/rocblas_herk_batched.cpp
    blas3/rocblas_herk_strided_batched.cpp
    blas3/rocblas_her2k.cpp
    blas3/rocblas_her2k_batched.cpp
    blas3/rocblas_her2k_strided_batched.cpp
    blas3/rocblas_herkx.cpp
    blas3/rocblas_herkx_batched.cpp
    blas3/rocblas_herkx_strided_batched.cpp
    blas3/rocblas_symm.cpp
    blas3/rocblas_symm_hemm_kernels.cpp
    blas3/rocblas_symm_batched.cpp
    blas3/rocblas_symm_strided_batched.cpp
    blas3/rocblas_syrk.cpp
    blas3/rocblas_syrk_herk_kernels.cpp
    blas3/rocblas_syrk_batched.cpp
    blas3/rocblas_syrk_strided_batched.cpp
    blas3/rocblas_syr2k.cpp
    blas3/rocblas_syr2k_her2k_kernels.cpp
    blas3/rocblas_syr2k_batched.cpp
    blas3/rocblas_syr2k_strided_batched.cpp
    blas3/rocblas_gemm.cpp
    blas3/rocblas_gemm_batched.cpp
    blas3/rocblas_gemm_strided_batched.cpp
    blas3/Tensile/gemm_templates.cpp
    blas3/rocblas_syrkx.cpp
    blas3/rocblas_syrkx_herkx_kernels.cpp
    blas3/rocblas_syrkx_batched.cpp
    blas3/rocblas_syrkx_strided_batched.cpp
    blas3/rocblas_trmm.cpp
    blas3/rocblas_trmm_kernels.cpp
    blas3/rocblas_trmm_batched.cpp
    blas3/rocblas_trmm_strided_batched.cpp
    blas3/rocblas_trtri.cpp
    blas3/rocblas_trtri_batched.cpp
    blas3/rocblas_trtri_strided_batched.cpp
    blas3/rocblas_trtri_kernels.cpp
)

set( rocblas_blas2_source
  # trsv first as slower
  blas2/rocblas_trsv.cpp
  blas2/rocblas_trsv_kernels.cpp
  blas2/rocblas_trsv_strided_batched.cpp
  blas2/rocblas_trsv_batched.cpp
  #
  blas2/rocblas_gemv.cpp
  blas2/rocblas_gemv_kernels.cpp
  blas2/rocblas_gemv_batched.cpp
  blas2/rocblas_gemv_strided_batched.cpp
  blas2/rocblas_tpmv.cpp
  blas2/rocblas_tpmv_kernels.cpp
  blas2/rocblas_tpmv_batched.cpp
  blas2/rocblas_tpmv_strided_batched.cpp
  blas2/rocblas_gbmv.cpp
  blas2/rocblas_gbmv_kernels.cpp
  blas2/rocblas_gbmv_batched.cpp
  blas2/rocblas_gbmv_strided_batched.cpp
  blas2/rocblas_tbsv.cpp
  blas2/rocblas_tbsv_kernels.cpp
  blas2/rocblas_tbsv_batched.cpp
  blas2/rocblas_tbsv_strided_batched.cpp
  blas2/rocblas_trmv.cpp
  blas2/rocblas_trmv_kernels.cpp
  blas2/rocblas_trmv_batched.cpp
  blas2/rocblas_trmv_strided_batched.cpp
  blas2/rocblas_ger.cpp
  blas2/rocblas_ger_kernels.cpp
  blas2/rocblas_ger_batched.cpp
  blas2/rocblas_ger_strided_batched.cpp
  blas2/rocblas_hbmv.cpp
  blas2/rocblas_hbmv_kernels.cpp
  blas2/rocblas_hbmv_batched.cpp
  blas2/rocblas_hbmv_strided_batched.cpp
  blas2/rocblas_hemv.cpp
  blas2/rocblas_hemv_symv_kernels.cpp
  blas2/rocblas_hemv_batched.cpp
  blas2/rocblas_hemv_strided_batched.cpp
  blas2/rocblas_her.cpp
  blas2/rocblas_her_kernels.cpp
  blas2/rocblas_her_batched.cpp
  blas2/rocblas_her_strided_batched.cpp
  blas2/rocblas_her2.cpp
  blas2/rocblas_her2_kernels.cpp
  blas2/rocblas_her2_batched.cpp
  blas2/rocblas_her2_strided_batched.cpp
  blas2/rocblas_hpmv.cpp
  blas2/rocblas_hpmv_kernels.cpp
  blas2/rocblas_hpmv_batched.cpp
  blas2/rocblas_hpmv_strided_batched.cpp
  blas2/rocblas_hpr.cpp
  blas2/rocblas_hpr_kernels.cpp
  blas2/rocblas_hpr_batched.cpp
  blas2/rocblas_hpr_strided_batched.cpp
  blas2/rocblas_hpr2.cpp
  blas2/rocblas_hpr2_kernels.cpp
  blas2/rocblas_hpr2_batched.cpp
  blas2/rocblas_hpr2_strided_batched.cpp
  blas2/rocblas_spr.cpp
  blas2/rocblas_spr_kernels.cpp
  blas2/rocblas_spr_batched.cpp
  blas2/rocblas_spr_strided_batched.cpp
  blas2/rocblas_spr2.cpp
  blas2/rocblas_spr2_kernels.cpp
  blas2/rocblas_spr2_batched.cpp
  blas2/rocblas_spr2_strided_batched.cpp
  blas2/rocblas_syr.cpp
  blas2/rocblas_syr_kernels.cpp
  blas2/rocblas_syr_batched.cpp
  blas2/rocblas_syr_strided_batched.cpp
  blas2/rocblas_syr2.cpp
  blas2/rocblas_syr2_kernels.cpp
  blas2/rocblas_syr2_batched.cpp
  blas2/rocblas_syr2_strided_batched.cpp
  blas2/rocblas_tbmv.cpp
  blas2/rocblas_tbmv_kernels.cpp
  blas2/rocblas_tbmv_batched.cpp
  blas2/rocblas_tbmv_strided_batched.cpp
  blas2/rocblas_tpsv.cpp
  blas2/rocblas_tpsv_kernels.cpp
  blas2/rocblas_tpsv_batched.cpp
  blas2/rocblas_tpsv_strided_batched.cpp
  blas2/rocblas_sbmv.cpp
  blas2/rocblas_sbmv_kernels.cpp
  blas2/rocblas_sbmv_batched.cpp
  blas2/rocblas_sbmv_strided_batched.cpp
  blas2/rocblas_spmv.cpp
  blas2/rocblas_spmv_kernels.cpp
  blas2/rocblas_spmv_batched.cpp
  blas2/rocblas_spmv_strided_batched.cpp
  blas2/rocblas_symv.cpp
  blas2/rocblas_symv_batched.cpp
  blas2/rocblas_symv_strided_batched.cpp
)

set( rocblas_auxiliary_source
  handle.cpp
  logging.cpp
  rocblas_auxiliary.cpp
  buildinfo.cpp
  rocblas_ostream.cpp
  check_numerics_vector.cpp
  check_numerics_matrix.cpp
  utility.cpp
)

set( rocblas_blas1_source
  blas1/rocblas_iamax_iamin_kernels.cpp
  blas1/rocblas_iamin.cpp
  blas1/rocblas_iamin_batched.cpp
  blas1/rocblas_iamin_strided_batched.cpp
  blas1/rocblas_iamax.cpp
  blas1/rocblas_iamax_batched.cpp
  blas1/rocblas_iamax_strided_batched.cpp
  blas1/rocblas_asum.cpp
  blas1/rocblas_asum_batched.cpp
  blas1/rocblas_asum_strided_batched.cpp
  blas1/rocblas_axpy.cpp
  blas1/rocblas_axpy_kernels.cpp
  blas1/rocblas_axpy_batched.cpp
  blas1/rocblas_axpy_strided_batched.cpp
  blas1/rocblas_copy.cpp
  blas1/rocblas_copy_kernels.cpp
  blas1/rocblas_copy_batched.cpp
  blas1/rocblas_copy_strided_batched.cpp
  blas1/rocblas_dot.cpp
  blas1/rocblas_dot_kernels.cpp
  blas1/rocblas_dot_strided_batched.cpp
  blas1/rocblas_dot_batched.cpp
  blas1/rocblas_nrm2.cpp
  blas1/rocblas_nrm2_batched.cpp
  blas1/rocblas_nrm2_strided_batched.cpp
  blas1/rocblas_asum_nrm2_kernels.cpp
  blas1/rocblas_rot.cpp
  blas1/rocblas_rot_kernels.cpp
  blas1/rocblas_rot_batched.cpp
  blas1/rocblas_rot_strided_batched.cpp
  blas1/rocblas_rotg.cpp
  blas1/rocblas_rotg_kernels.cpp
  blas1/rocblas_rotg_batched.cpp
  blas1/rocblas_rotg_strided_batched.cpp
  blas1/rocblas_rotm.cpp
  blas1/rocblas_rotm_kernels.cpp
  blas1/rocblas_rotm_batched.cpp
  blas1/rocblas_rotm_strided_batched.cpp
  blas1/rocblas_rotmg.cpp
  blas1/rocblas_rotmg_kernels.cpp
  blas1/rocblas_rotmg_batched.cpp
  blas1/rocblas_rotmg_strided_batched.cpp
  blas1/rocblas_scal.cpp
  blas1/rocblas_scal_kernels.cpp
  blas1/rocblas_scal_batched.cpp
  blas1/rocblas_scal_strided_batched.cpp
  blas1/rocblas_swap.cpp
  blas1/rocblas_swap_kernels.cpp
  blas1/rocblas_swap_batched.cpp
  blas1/rocblas_swap_strided_batched.cpp
)

#
# src64 sub-directory adds ILP64 API

set(rocblas64_ex_source_no_tensile
  src64/blas_ex/rocblas_axpy_ex_64.cpp
  src64/blas_ex/rocblas_axpy_ex_kernels_64.cpp
  src64/blas_ex/rocblas_axpy_batched_ex_64.cpp
  src64/blas_ex/rocblas_axpy_strided_batched_ex_64.cpp
  src64/blas_ex/rocblas_dot_ex_64.cpp
  src64/blas_ex/rocblas_dot_ex_kernels_64.cpp
  src64/blas_ex/rocblas_dot_batched_ex_64.cpp
  src64/blas_ex/rocblas_dot_strided_batched_ex_64.cpp
  src64/blas_ex/rocblas_rot_ex_64.cpp
  src64/blas_ex/rocblas_rot_ex_kernels_64.cpp
  src64/blas_ex/rocblas_rot_batched_ex_64.cpp
  src64/blas_ex/rocblas_rot_strided_batched_ex_64.cpp
  src64/blas_ex/rocblas_scal_ex_64.cpp
  src64/blas_ex/rocblas_scal_ex_kernels_64.cpp
  src64/blas_ex/rocblas_scal_batched_ex_64.cpp
  src64/blas_ex/rocblas_scal_strided_batched_ex_64.cpp
  src64/blas_ex/rocblas_nrm2_ex_64.cpp
  src64/blas_ex/rocblas_nrm2_ex_kernels_64.cpp
  src64/blas_ex/rocblas_nrm2_batched_ex_64.cpp
  src64/blas_ex/rocblas_nrm2_strided_batched_ex_64.cpp
  src64/blas_ex/rocblas_gemmt_64.cpp
  src64/blas_ex/rocblas_gemmt_kernels_64.cpp
  src64/blas_ex/rocblas_gemmt_batched_64.cpp
  src64/blas_ex/rocblas_gemmt_strided_batched_64.cpp
  #     src64/blas_ex/rocblas_geam_ex.cpp
  #     src64/blas_ex/rocblas_geam_ex_kernels.cpp
)

set( rocblas64_ex_source
  src64/blas_ex/rocblas_gemm_ex_64.cpp
  src64/blas_ex/rocblas_gemm_batched_ex_64.cpp
  src64/blas_ex/rocblas_gemm_strided_batched_ex_64.cpp
  src64/blas_ex/rocblas_gemm_ex_kernels_64.cpp
)

set( rocblas64_blas3_source_no_tensile
  src64/blas3/rocblas_dgmm_64.cpp
  src64/blas3/rocblas_dgmm_kernels_64.cpp
  src64/blas3/rocblas_dgmm_batched_64.cpp
  src64/blas3/rocblas_dgmm_strided_batched_64.cpp
  src64/blas3/rocblas_geam_64.cpp
  src64/blas3/rocblas_geam_kernels_64.cpp
  src64/blas3/rocblas_geam_batched_64.cpp
  src64/blas3/rocblas_geam_strided_batched_64.cpp
)

# rocblas L3 that use tensile but can use source gemm as fallback
set( rocblas64_blas3_source
    src64/blas3/rocblas_hemm_64.cpp
    src64/blas3/rocblas_hemm_batched_64.cpp
    src64/blas3/rocblas_hemm_strided_batched_64.cpp
    src64/blas3/rocblas_herk_64.cpp
    src64/blas3/rocblas_herk_batched_64.cpp
    src64/blas3/rocblas_herk_strided_batched_64.cpp
    src64/blas3/rocblas_her2k_64.cpp
    src64/blas3/rocblas_her2k_batched_64.cpp
    src64/blas3/rocblas_her2k_strided_batched_64.cpp
    src64/blas3/rocblas_herkx_64.cpp
    src64/blas3/rocblas_herkx_batched_64.cpp
    src64/blas3/rocblas_herkx_strided_batched_64.cpp
    src64/blas3/rocblas_symm_64.cpp
    src64/blas3/rocblas_symm_hemm_kernels_64.cpp
    src64/blas3/rocblas_symm_batched_64.cpp
    src64/blas3/rocblas_symm_strided_batched_64.cpp
    src64/blas3/rocblas_syrk_64.cpp
    src64/blas3/rocblas_syrk_herk_kernels_64.cpp
    src64/blas3/rocblas_syrk_batched_64.cpp
    src64/blas3/rocblas_syrk_strided_batched_64.cpp
    src64/blas3/rocblas_syr2k_64.cpp
    src64/blas3/rocblas_syr2k_her2k_kernels_64.cpp
    src64/blas3/rocblas_syr2k_batched_64.cpp
    src64/blas3/rocblas_syr2k_strided_batched_64.cpp
    src64/blas3/rocblas_gemm_64.cpp
    src64/blas3/rocblas_gemm_kernels_64.cpp
    src64/blas3/rocblas_gemm_batched_64.cpp
    src64/blas3/rocblas_gemm_strided_batched_64.cpp
    src64/blas3/rocblas_syrkx_64.cpp
    src64/blas3/rocblas_syrkx_herkx_kernels_64.cpp
    src64/blas3/rocblas_syrkx_batched_64.cpp
    src64/blas3/rocblas_syrkx_strided_batched_64.cpp
    src64/blas3/rocblas_trmm_64.cpp
    src64/blas3/rocblas_trmm_kernels_64.cpp
    src64/blas3/rocblas_trmm_batched_64.cpp
    src64/blas3/rocblas_trmm_strided_batched_64.cpp
    src64/blas3/rocblas_trsm_64.cpp
    src64/blas3/rocblas_trsm_kernels_64.cpp
    src64/blas3/rocblas_trsm_batched_64.cpp
    src64/blas3/rocblas_trsm_strided_batched_64.cpp
#     src64/blas3/rocblas_trtri.cpp
#     src64/blas3/rocblas_trtri_batched.cpp
#     src64/blas3/rocblas_trtri_strided_batched.cpp
)

set( rocblas64_blas2_source
  src64/blas2/rocblas_gemv_64.cpp
  src64/blas2/rocblas_gemv_kernels_64.cpp
  src64/blas2/rocblas_gemv_batched_64.cpp
  src64/blas2/rocblas_gemv_strided_batched_64.cpp
  src64/blas2/rocblas_tpmv_64.cpp
  src64/blas2/rocblas_tpmv_kernels_64.cpp
  src64/blas2/rocblas_tpmv_batched_64.cpp
  src64/blas2/rocblas_tpmv_strided_batched_64.cpp
  src64/blas2/rocblas_gbmv_64.cpp
  src64/blas2/rocblas_gbmv_kernels_64.cpp
  src64/blas2/rocblas_gbmv_batched_64.cpp
  src64/blas2/rocblas_gbmv_strided_batched_64.cpp
  src64/blas2/rocblas_tbsv_64.cpp
  src64/blas2/rocblas_tbsv_kernels_64.cpp
  src64/blas2/rocblas_tbsv_batched_64.cpp
  src64/blas2/rocblas_tbsv_strided_batched_64.cpp
  src64/blas2/rocblas_trmv_64.cpp
  src64/blas2/rocblas_trmv_kernels_64.cpp
  src64/blas2/rocblas_trmv_batched_64.cpp
  src64/blas2/rocblas_trmv_strided_batched_64.cpp
  src64/blas2/rocblas_ger_64.cpp
  src64/blas2/rocblas_ger_kernels_64.cpp
  src64/blas2/rocblas_ger_batched_64.cpp
  src64/blas2/rocblas_ger_strided_batched_64.cpp
  src64/blas2/rocblas_hbmv_64.cpp
  src64/blas2/rocblas_hbmv_kernels_64.cpp
  src64/blas2/rocblas_hbmv_batched_64.cpp
  src64/blas2/rocblas_hbmv_strided_batched_64.cpp
  src64/blas2/rocblas_hemv_64.cpp
  src64/blas2/rocblas_hemv_symv_kernels_64.cpp
  src64/blas2/rocblas_hemv_batched_64.cpp
  src64/blas2/rocblas_hemv_strided_batched_64.cpp
  src64/blas2/rocblas_her_64.cpp
  src64/blas2/rocblas_her_kernels_64.cpp
  src64/blas2/rocblas_her_batched_64.cpp
  src64/blas2/rocblas_her_strided_batched_64.cpp
  src64/blas2/rocblas_her2_64.cpp
  src64/blas2/rocblas_her2_kernels_64.cpp
  src64/blas2/rocblas_her2_batched_64.cpp
  src64/blas2/rocblas_her2_strided_batched_64.cpp
  src64/blas2/rocblas_hpmv_64.cpp
  src64/blas2/rocblas_hpmv_kernels_64.cpp
  src64/blas2/rocblas_hpmv_batched_64.cpp
  src64/blas2/rocblas_hpmv_strided_batched_64.cpp
  src64/blas2/rocblas_hpr_64.cpp
  src64/blas2/rocblas_hpr_kernels_64.cpp
  src64/blas2/rocblas_hpr_batched_64.cpp
  src64/blas2/rocblas_hpr_strided_batched_64.cpp
  src64/blas2/rocblas_hpr2_64.cpp
  src64/blas2/rocblas_hpr2_kernels_64.cpp
  src64/blas2/rocblas_hpr2_batched_64.cpp
  src64/blas2/rocblas_hpr2_strided_batched_64.cpp
  src64/blas2/rocblas_spr_64.cpp
  src64/blas2/rocblas_spr_kernels_64.cpp
  src64/blas2/rocblas_spr_batched_64.cpp
  src64/blas2/rocblas_spr_strided_batched_64.cpp
  src64/blas2/rocblas_spr2_64.cpp
  src64/blas2/rocblas_spr2_kernels_64.cpp
  src64/blas2/rocblas_spr2_batched_64.cpp
  src64/blas2/rocblas_spr2_strided_batched_64.cpp
  src64/blas2/rocblas_syr2_64.cpp
  src64/blas2/rocblas_syr2_kernels_64.cpp
  src64/blas2/rocblas_syr2_batched_64.cpp
  src64/blas2/rocblas_syr2_strided_batched_64.cpp
  src64/blas2/rocblas_tbmv_64.cpp
  src64/blas2/rocblas_tbmv_kernels_64.cpp
  src64/blas2/rocblas_tbmv_batched_64.cpp
  src64/blas2/rocblas_tbmv_strided_batched_64.cpp
  src64/blas2/rocblas_syr_64.cpp
  src64/blas2/rocblas_syr_kernels_64.cpp
  src64/blas2/rocblas_syr_batched_64.cpp
  src64/blas2/rocblas_syr_strided_batched_64.cpp
  src64/blas2/rocblas_tpsv_64.cpp
  src64/blas2/rocblas_tpsv_kernels_64.cpp
  src64/blas2/rocblas_tpsv_batched_64.cpp
  src64/blas2/rocblas_tpsv_strided_batched_64.cpp
  src64/blas2/rocblas_sbmv_64.cpp
  src64/blas2/rocblas_sbmv_kernels_64.cpp
  src64/blas2/rocblas_sbmv_batched_64.cpp
  src64/blas2/rocblas_sbmv_strided_batched_64.cpp
  src64/blas2/rocblas_spmv_64.cpp
  src64/blas2/rocblas_spmv_kernels_64.cpp
  src64/blas2/rocblas_spmv_batched_64.cpp
  src64/blas2/rocblas_spmv_strided_batched_64.cpp
  src64/blas2/rocblas_symv_64.cpp
  src64/blas2/rocblas_symv_batched_64.cpp
  src64/blas2/rocblas_symv_strided_batched_64.cpp
  src64/blas2/rocblas_trsv_64.cpp
  src64/blas2/rocblas_trsv_kernels_64.cpp
  src64/blas2/rocblas_trsv_strided_batched_64.cpp
  src64/blas2/rocblas_trsv_batched_64.cpp
)

set( rocblas64_auxiliary_source
  # src64/handle.cpp
  # src64/rocblas_auxiliary.cpp
  # src64/buildinfo.cpp
  # src64/rocblas_ostream.cpp
  # src64/check_numerics_vector.cpp
  # src64/check_numerics_matrix.cpp
)

set( rocblas64_blas1_source
  src64/blas1/rocblas_iamax_iamin_kernels_64.cpp
  src64/blas1/rocblas_iamin_64.cpp
  src64/blas1/rocblas_iamin_batched_64.cpp
  src64/blas1/rocblas_iamin_strided_batched_64.cpp
  src64/blas1/rocblas_iamax_64.cpp
  src64/blas1/rocblas_iamax_batched_64.cpp
  src64/blas1/rocblas_iamax_strided_batched_64.cpp
  src64/blas1/rocblas_axpy_64.cpp
  src64/blas1/rocblas_axpy_kernels_64.cpp
  src64/blas1/rocblas_axpy_batched_64.cpp
  src64/blas1/rocblas_axpy_strided_batched_64.cpp
  src64/blas1/rocblas_copy_64.cpp
  src64/blas1/rocblas_copy_kernels_64.cpp
  src64/blas1/rocblas_copy_batched_64.cpp
  src64/blas1/rocblas_copy_strided_batched_64.cpp
  src64/blas1/rocblas_asum_64.cpp
  src64/blas1/rocblas_asum_nrm2_kernels_64.cpp
  src64/blas1/rocblas_asum_batched_64.cpp
  src64/blas1/rocblas_asum_strided_batched_64.cpp
  src64/blas1/rocblas_dot_64.cpp
  src64/blas1/rocblas_dot_kernels_64.cpp
  src64/blas1/rocblas_dot_strided_batched_64.cpp
  src64/blas1/rocblas_dot_batched_64.cpp
  src64/blas1/rocblas_nrm2_64.cpp
  src64/blas1/rocblas_nrm2_batched_64.cpp
  src64/blas1/rocblas_nrm2_strided_batched_64.cpp
  src64/blas1/rocblas_rot_64.cpp
  src64/blas1/rocblas_rot_kernels_64.cpp
  src64/blas1/rocblas_rot_batched_64.cpp
  src64/blas1/rocblas_rot_strided_batched_64.cpp
  src64/blas1/rocblas_rotg_64.cpp
  src64/blas1/rocblas_rotg_kernels_64.cpp
  src64/blas1/rocblas_rotg_batched_64.cpp
  src64/blas1/rocblas_rotg_strided_batched_64.cpp
  src64/blas1/rocblas_rotm_64.cpp
  src64/blas1/rocblas_rotm_kernels_64.cpp
  src64/blas1/rocblas_rotm_batched_64.cpp
  src64/blas1/rocblas_rotm_strided_batched_64.cpp
  src64/blas1/rocblas_rotmg_64.cpp
  src64/blas1/rocblas_rotmg_kernels_64.cpp
  src64/blas1/rocblas_rotmg_batched_64.cpp
  src64/blas1/rocblas_rotmg_strided_batched_64.cpp
  src64/blas1/rocblas_scal_64.cpp
  src64/blas1/rocblas_scal_kernels_64.cpp
  src64/blas1/rocblas_scal_batched_64.cpp
  src64/blas1/rocblas_scal_strided_batched_64.cpp
  src64/blas1/rocblas_swap_64.cpp
  src64/blas1/rocblas_swap_kernels_64.cpp
  src64/blas1/rocblas_swap_batched_64.cpp
  src64/blas1/rocblas_swap_strided_batched_64.cpp
)

set( subdir_src64_list
  ${rocblas64_ex_source}
  ${rocblas64_ex_source_no_tensile}
  ${rocblas64_blas3_source}
  ${rocblas64_blas3_source_no_tensile}
  ${rocblas64_blas2_source}
  ${rocblas64_blas1_source}
  ${rocblas64_auxiliary_source}
)

prepend_path( ".." rocblas_headers_public relative_rocblas_headers_public )

add_library( rocblas
  # ordered from generally slowest to compile to faster to increase parallelism
  ${rocblas_blas3_source}
  ${rocblas_blas3_source_no_tensile}
  ${rocblas_blas2_source}
  ${rocblas_ex_source}
  ${rocblas_ex_source_no_tensile}
  ${rocblas_blas1_source}
  ${relative_rocblas_headers_public}
  ${rocblas_auxiliary_source}
  ${subdir_src64_list}
)

foreach( file ${subdir_src64_list} )
  SET_SOURCE_FILES_PROPERTIES( ${file} PROPERTIES COMPILE_DEFINITIONS ROCBLAS_INTERNAL_ILP64 )
endforeach()

message(STATUS "** NOTE: blas2/rocblas_ger_kernels.cpp is compiled with the verbose flag -v for QC purposes.")
SET_SOURCE_FILES_PROPERTIES( blas2/rocblas_ger_kernels.cpp PROPERTIES COMPILE_FLAGS "-v" )

#if( WIN32 )
#  set_target_properties(rocblas_fortran PROPERTIES LINKER_LANGUAGE CXX)
#  target_link_directories(rocblas_fortran PRIVATE "C:\\cygwin64\\lib\\gcc\\x86_64-pc-cygwin\\9.3.0" "C:\\cygwin64\\lib" "C:\\cygwin64\\lib\\w32api")
#endif( )

add_library( roc::rocblas ALIAS rocblas )

rocblas_library_settings( rocblas )

target_link_libraries( rocblas PRIVATE "-Xlinker --exclude-libs=ALL" ) # HIDE symbols


if( BUILD_WITH_TENSILE )

  if( BUILD_SHARED_LIBS )
    target_link_libraries( rocblas PRIVATE TensileHost )
  endif()

  if(BUILD_WITH_HIPBLASLT)
    target_link_libraries( rocblas PRIVATE roc::hipblaslt )
  endif()

  if( NOT BUILD_SHARED_LIBS )
    target_compile_definitions( rocblas PRIVATE ROCBLAS_STATIC_LIB )

    # bypassing cmake dependencies chain for static link as it won't allow target from different directory

    # including tensile headers into rocblas tensileHost client so get compile properties
    get_target_property(TensileHost_INCLUDES TensileHost INCLUDE_DIRECTORIES)
    target_include_directories( rocblas PRIVATE ${TensileHost_INCLUDES} )
    get_target_property(TensileHost_DEFINES TensileHost COMPILE_DEFINITIONS)
    target_compile_definitions( rocblas PRIVATE ${TensileHost_DEFINES} )

    get_target_property( TensileHost_LIBDIR TensileHost BINARY_DIR )

    message (STATUS "TensileHost_INCLUDES == ${TensileHost_INCLUDES}")
    message (STATUS "TensileHost_DEFINES == ${TensileHost_DEFINES}")
    message (STATUS "TensileHost_LIBDIR == ${TensileHost_LIBDIR}")

    # recreate LLVM static dependencies
    if (${Tensile_LIBRARY_FORMAT} STREQUAL "yaml")
      find_package(LLVM REQUIRED CONFIG)
      find_library(LLVMObjectYAML_LIBRARY
        NAMES LLVMObjectYAML
        PATHS ${LLVM_LIBRARY_DIR})
      message("LLVMObjectYAML_LIBRARY: ${LLVMObjectYAML_LIBRARY}")

      target_link_libraries(rocblas PRIVATE LLVMObjectYAML )  # match tensile
    endif()

    # to get TensileHost built first, not to link target
    # as dependency chain can not be created
    add_dependencies(rocblas TensileHost)

  endif()

  target_compile_definitions( rocblas PRIVATE ${TENSILE_DEFINES} ${HIPBLASLT_DEFINES})
endif()

#set_target_properties( rocblas PROPERTIES CXX_VISIBILITY_PRESET "hidden" C_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON )
#set_target_properties( rocblas PROPERTIES CXX_VISIBILITY_PRESET "default" C_VISIBILITY_PRESET "default" VISIBILITY_INLINES_HIDDEN ON )

# The export header generation is now done in parent CMakeLists.txt
# Package that helps me set visibility for function names exported from shared library
include( GenerateExportHeader )
generate_export_header( rocblas EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/rocblas/internal/rocblas-export.h )

# generate header with prototypes for export reuse
file( GLOB rocblas_prototype_inputs
  LIST_DIRECTORIES OFF
  CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/blas3/Tensile/*.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/blas3/*.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/blas2/*.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/blas1/*.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/*.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/*.hpp
)
set( ROCBLAS_PROTO_TEMPLATES "${PROJECT_BINARY_DIR}/include/rocblas/internal/rocblas-exported-proto.hpp" )
add_custom_command(OUTPUT ${ROCBLAS_PROTO_TEMPLATES}
  COMMAND ${python} template-proto.py ${CMAKE_CURRENT_SOURCE_DIR}/include/ ${CMAKE_CURRENT_SOURCE_DIR}/blas3/Tensile/ ${CMAKE_CURRENT_SOURCE_DIR}/blas3/ ${CMAKE_CURRENT_SOURCE_DIR}/blas2/ ${CMAKE_CURRENT_SOURCE_DIR}/blas1/ ${CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/ ${CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/ ${CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/ > ${ROCBLAS_PROTO_TEMPLATES}
  #DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/blas3/Tensile/ ${CMAKE_CURRENT_SOURCE_DIR}/blas3/ ${CMAKE_CURRENT_SOURCE_DIR}/blas2/ ${CMAKE_CURRENT_SOURCE_DIR}/blas1/
  DEPENDS ${rocblas_prototype_inputs}
  COMMENT "Generating prototypes from ${CMAKE_CURRENT_SOURCE_DIR}."
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
add_custom_target( rocblas_proto_templates DEPENDS ${ROCBLAS_PROTO_TEMPLATES} )
add_dependencies( rocblas rocblas_proto_templates )

# generate rocblas_device_malloc.hpp header for device memory allocation
set( ROCBLAS_DEVICE_MALLOC "${PROJECT_BINARY_DIR}/include/rocblas/internal/rocblas_device_malloc.hpp" )
add_custom_command( OUTPUT ${ROCBLAS_DEVICE_MALLOC}
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_device_malloc.hpp ${ROCBLAS_DEVICE_MALLOC}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_device_malloc.hpp
  COMMENT "Generating ${ROCBLAS_DEVICE_MALLOC}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
add_custom_target( rocblas_device_malloc DEPENDS ${ROCBLAS_DEVICE_MALLOC} )
add_dependencies( rocblas rocblas_device_malloc )



############################################################
# Post build

if( NOT BUILD_SHARED_LIBS )

  if( BUILD_WITH_TENSILE )
    add_custom_command( TARGET rocblas POST_BUILD
      COMMAND
        ${python} ${CMAKE_CURRENT_SOURCE_DIR}/merge_archives.py
      ARGS
        -v
        -o "$<TARGET_LINKER_FILE:rocblas>"
        --ar "${CMAKE_AR}"
        -L "${TensileHost_LIBDIR}"
        "$<TARGET_LINKER_FILE:rocblas>"
        "librocblas-tensile.a"
      WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
      COMMENT "Merging rocblas-tensile library into rocblas"
      )
  endif()

endif( )

if (WIN32 AND BUILD_CLIENTS)
  add_custom_command( TARGET rocblas POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/staging/$<TARGET_FILE_NAME:rocblas> ${PROJECT_BINARY_DIR}/clients/staging/$<TARGET_FILE_NAME:rocblas> )
  if( ${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_custom_command( TARGET rocblas POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/staging/rocblas.pdb ${PROJECT_BINARY_DIR}/clients/staging/rocblas.pdb )
  endif()
endif()

############################################################
# Installation

rocm_install_targets(
  TARGETS ${package_targets}
  INCLUDE
    ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}
)

if( BUILD_WITH_TENSILE )
  if (WIN32)
    set( ROCBLAS_TENSILE_LIBRARY_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/bin/rocblas" CACHE PATH "path to tensile library" )
  else()
    set( ROCBLAS_TENSILE_LIBRARY_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}${CMAKE_INSTALL_LIBDIR}/rocblas" CACHE PATH "path to tensile library" )
  endif()
  # For ASAN package, Tensile library files(which are not shared libraries) are not required
  if( NOT ENABLE_ASAN_PACKAGING )
    if( BUILD_SHARED_LIBS )
      set( TENSILE_DATA_COMPONENT_NAME ${CMAKE_INSTALL_DEFAULT_COMPONENT_NAME} )
    else()
      set( TENSILE_DATA_COMPONENT_NAME devel )
    endif()
    rocm_install(
      DIRECTORY ${CMAKE_BINARY_DIR}/Tensile/library
      DESTINATION ${ROCBLAS_TENSILE_LIBRARY_DIR}
      COMPONENT ${TENSILE_DATA_COMPONENT_NAME}) # Use this cmake variable to be compatible with rocm-cmake 0.6 and 0.7
  endif()
endif()

if(NOT WIN32)
  if(RUN_HEADER_TESTING)
  # Compilation tests to ensure that header files work independently,
  # and that public header files work across several languages
  add_custom_command(
    TARGET rocblas
    POST_BUILD
    COMMAND ${CMAKE_HOME_DIRECTORY}/header_compilation_tests.sh
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  endif()
endif()
