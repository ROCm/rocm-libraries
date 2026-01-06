# ########################################################################
# Copyright (C) 2018-2020 Advanced Micro Devices, Inc. All rights Reserved.
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

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Print verbose compiler flags
if(BUILD_VERBOSE)
  include(../cmake/Verbose.cmake)
endif()

# Configure a header file to pass the hipSPARSE version
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/hipsparse-version.h.in"
               "${PROJECT_BINARY_DIR}/include/hipsparse/hipsparse-version.h")

# Copy Header files to build directory
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/hipsparse.h"
               "${PROJECT_BINARY_DIR}/include/hipsparse/hipsparse.h" COPYONLY)

set(HEADER_FILES
  # Level1
  internal/level1/hipsparse_axpyi.h
  internal/level1/hipsparse_dotci.h
  internal/level1/hipsparse_doti.h
  internal/level1/hipsparse_gthr.h
  internal/level1/hipsparse_gthrz.h
  internal/level1/hipsparse_roti.h
  internal/level1/hipsparse_sctr.h
  # Level2
  internal/level2/hipsparse_bsrmv.h
  internal/level2/hipsparse_bsrsv.h
  internal/level2/hipsparse_bsrxmv.h
  internal/level2/hipsparse_csrmv.h
  internal/level2/hipsparse_csrsv.h
  internal/level2/hipsparse_gemvi.h
  internal/level2/hipsparse_hybmv.h
  # Level3
  internal/level3/hipsparse_bsrmm.h
  internal/level3/hipsparse_bsrsm.h
  internal/level3/hipsparse_csrmm.h
  internal/level3/hipsparse_csrsm.h
  internal/level3/hipsparse_gemmi.h
  # Extra
  internal/extra/hipsparse_csrgeam.h
  internal/extra/hipsparse_csrgemm.h
  # Precond
  internal/precond/hipsparse_bsric0.h
  internal/precond/hipsparse_bsrilu0.h
  internal/precond/hipsparse_csric0.h
  internal/precond/hipsparse_csrilu0.h
  internal/precond/hipsparse_gpsv_interleaved_batch.h
  internal/precond/hipsparse_gtsv_interleaved_batch.h
  internal/precond/hipsparse_gtsv_nopivot.h
  internal/precond/hipsparse_gtsv_strided_batch.h
  internal/precond/hipsparse_gtsv.h
  # Reorder
  internal/reorder/hipsparse_csrcolor.h
  # Conversion
  internal/conversion/hipsparse_bsr2csr.h
  internal/conversion/hipsparse_coo2csr.h
  internal/conversion/hipsparse_coosort.h
  internal/conversion/hipsparse_create_identity_permutation.h
  internal/conversion/hipsparse_csc2dense.h
  internal/conversion/hipsparse_cscsort.h
  internal/conversion/hipsparse_csr2bsr.h
  internal/conversion/hipsparse_csr2coo.h
  internal/conversion/hipsparse_csr2csc.h
  internal/conversion/hipsparse_csr2csr_compress.h
  internal/conversion/hipsparse_csr2csru.h
  internal/conversion/hipsparse_csr2dense.h
  internal/conversion/hipsparse_csr2gebsr.h
  internal/conversion/hipsparse_csr2hyb.h
  internal/conversion/hipsparse_csrsort.h
  internal/conversion/hipsparse_csru2csr.h
  internal/conversion/hipsparse_dense2csc.h
  internal/conversion/hipsparse_dense2csr.h
  internal/conversion/hipsparse_gebsr2csr.h
  internal/conversion/hipsparse_gebsr2gebsc.h
  internal/conversion/hipsparse_gebsr2gebsr.h
  internal/conversion/hipsparse_hyb2csr.h
  internal/conversion/hipsparse_nnz_compress.h
  internal/conversion/hipsparse_nnz.h
  internal/conversion/hipsparse_prune_csr2csr_by_percentage.h
  internal/conversion/hipsparse_prune_csr2csr.h
  internal/conversion/hipsparse_prune_dense2csr_by_percentage.h
  internal/conversion/hipsparse_prune_dense2csr.h
  # Generic
  internal/generic/hipsparse_axpby.h
  internal/generic/hipsparse_dense2sparse.h
  internal/generic/hipsparse_gather.h
  internal/generic/hipsparse_rot.h
  internal/generic/hipsparse_scatter.h
  internal/generic/hipsparse_sddmm.h
  internal/generic/hipsparse_sparse2dense.h
  internal/generic/hipsparse_spgemm_reuse.h
  internal/generic/hipsparse_spgemm.h
  internal/generic/hipsparse_spmm.h
  internal/generic/hipsparse_spmv.h
  internal/generic/hipsparse_spsm.h
  internal/generic/hipsparse_spsv.h
  internal/generic/hipsparse_spvv.h
  # Auxiliary
  hipsparse-bfloat16.h
  hipsparse-types.h
  hipsparse-auxiliary.h
  hipsparse-generic-types.h
  hipsparse-generic-auxiliary.h)

foreach( file_i ${HEADER_FILES})
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/${file_i}"
    "${PROJECT_BINARY_DIR}/include/hipsparse/${file_i}" COPYONLY)
endforeach()

# Public hipSPARSE headers
set(hipsparse_headers_public
  include/hipsparse.h
  ${PROJECT_BINARY_DIR}/include/hipsparse/hipsparse-version.h)

source_group("Header Files\\Public" FILES ${hipsparse_headers_public})

# Include sources
include(src/CMakeLists.txt)

if (NOT WIN32)
# Set Fortran module output directory
set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/include/hipsparse)

# Create hipSPARSE Fortran module
add_library(hipsparse_fortran OBJECT ${hipsparse_fortran_source})

# Target compile options
target_compile_options(hipsparse_fortran PRIVATE -ffree-form -cpp)
endif()

# Create hipSPARSE library
add_library(hipsparse ${hipsparse_source} ${hipsparse_headers_public})
add_library(roc::hipsparse ALIAS hipsparse)

# Target compile options
target_compile_options(hipsparse PRIVATE -Wno-unused-command-line-argument -Wall)

# External header includes included as system files
target_include_directories( hipsparse
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${ROCSPARSE_INCLUDE_DIRS}>
)

# Target include directories
target_include_directories(hipsparse
                           PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
                           PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
                                   $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/hipsparse>
                                   $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
                                   $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(NOT USE_CUDA)
  target_link_libraries(hipsparse PUBLIC hip::host)
  target_link_libraries(hipsparse PRIVATE roc::rocsparse)
else()
  target_compile_definitions(hipsparse PUBLIC __HIP_PLATFORM_NVIDIA__)
  target_include_directories(hipsparse
                               PUBLIC
                                 $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
                                 $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>)

  SET(SEARCH_SYMBOL "cusparseCreateSpVec")
  SET(SEARCH_RESULTS "")
  if(WIN32)
    EXECUTE_PROCESS(
        COMMAND
        dumpbin /EXPORTS ${CUDA_cusparse_LIBRARY}
        COMMAND
        findstr ${SEARCH_SYMBOL}
        OUTPUT_VARIABLE SEARCH_RESULTS
        ERROR_VARIABLE ERR)
  else()
    EXECUTE_PROCESS(
        COMMAND
        objdump -T ${CUDA_cusparse_LIBRARY}
        COMMAND
        grep ${SEARCH_SYMBOL}
        OUTPUT_VARIABLE SEARCH_RESULTS
        ERROR_VARIABLE ERR)
  endif()

  if(SEARCH_RESULTS STREQUAL "")
    # symbol not found
    target_compile_definitions(hipsparse PRIVATE CUDART_10_1_UPDATE_VERSION=0)
  else()
    # symbol found
    target_compile_definitions(hipsparse PRIVATE CUDART_10_1_UPDATE_VERSION=1)
  endif()

  target_link_libraries(hipsparse PRIVATE ${CUDA_cusparse_LIBRARY})
endif()

# Target properties
rocm_set_soversion(hipsparse ${hipsparse_SOVERSION})
set_target_properties(hipsparse PROPERTIES CXX_EXTENSIONS NO)
set_target_properties(hipsparse PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON)
set_target_properties(hipsparse PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
set_target_propertieS(hipsparse PROPERTIES DEBUG_POSTFIX "-d")

if (WIN32 AND BUILD_SHARED_LIBS)
  add_custom_command( TARGET hipsparse POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/staging/$<TARGET_FILE_NAME:hipsparse> ${PROJECT_BINARY_DIR}/clients/staging/$<TARGET_FILE_NAME:hipsparse> )
  if( ${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_custom_command( TARGET hipsparse POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/staging/hipsparse.pdb ${PROJECT_BINARY_DIR}/clients/staging/hipsparse.pdb )
  endif()
endif()

# Following boost conventions of prefixing 'lib' on static built libraries
if(NOT BUILD_SHARED_LIBS)
  set_target_properties(hipsparse PROPERTIES PREFIX "lib")
endif()

# Generate export header
include(GenerateExportHeader)
generate_export_header(hipsparse EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/hipsparse/hipsparse-export.h)

# Install targets
rocm_install_targets(TARGETS hipsparse
                     INCLUDE
                       ${CMAKE_BINARY_DIR}/include
                     )

# Export targets
if(NOT USE_CUDA)
  rocm_export_targets(TARGETS roc::hipsparse
                      DEPENDS PACKAGE hip
                      STATIC_DEPENDS PACKAGE rocsparse
                      NAMESPACE roc::)
else()
  rocm_export_targets(TARGETS roc::hipsparse
                      NAMESPACE roc::)
endif()
