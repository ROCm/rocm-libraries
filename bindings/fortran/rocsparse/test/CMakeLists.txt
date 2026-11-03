# ########################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
#
# SPDX-License-Identifier: MIT
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

# rocSPARSE Fortran binding tests: one CTest executable per test, GPU required.

# ifx/ifort skip .f03/.f08 sources unless told the language with -Tf.
if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  string(REPLACE "<SOURCE>" "-Tf <SOURCE>"
         CMAKE_Fortran_COMPILE_OBJECT "${CMAKE_Fortran_COMPILE_OBJECT}")
endif()

# Keep only the binding wrappers a test calls (the archive has one section per
# procedure), so a test links even when another wrapper names a C symbol the
# installed library does not export. Spelled -Wl, rather than LINKER:, which
# CMake 3.27 passes to LLVMFlang as a bare --gc-sections.
if(NOT APPLE AND NOT WIN32)
  add_link_options(-Wl,--gc-sections)
endif()

# The tests `use hip`: hip::hip_fortran comes from find_package(hip-fortran),
# built with this same compiler.
if(NOT TARGET hip::hip_fortran)
  find_package(hip-fortran QUIET)
endif()
set(_have_hip_binding TRUE)
if(NOT TARGET hip::hip_fortran)
  set(_have_hip_binding FALSE)
  message(STATUS
    "rocSPARSE: no hip::hip_fortran target, skipping the Fortran binding tests, "
    "every one of which uses it. Point CMAKE_PREFIX_PATH at a ROCm whose HIP "
    "Fortran bindings were built with ${CMAKE_Fortran_COMPILER}.")
endif()

# Standalone librocsparse has to be found; its find_dependency(hip) needs C.
if(NOT TARGET roc::rocsparse)
  if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
    enable_language(C)
  endif()
  find_package(rocsparse QUIET)
endif()
if(NOT TARGET roc::rocsparse)
  message(STATUS
    "rocSPARSE: no roc::rocsparse target, skipping the Fortran binding tests "
    "(they link librocsparse). Point CMAKE_PREFIX_PATH at an installed ROCm to "
    "build them standalone.")
  return()
endif()

# f2003 tests (<name>.f03) call the C-binding interfaces; f2008 tests
# (<name>.f08) the array overloads, which exist only when
# FORTRAN_ARRAY_INTERFACES is not none. Their CTest names carry an _f2008 suffix.
function(rocsparse_fortran_add_test name std)
  if(std STREQUAL "f2008")
    set(ext "f08")
    set(suffix "_f2008")
  else()
    set(ext "f03")
    set(suffix "")
  endif()
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/${name}.${ext}")
  if(NOT EXISTS "${src}")
    message(FATAL_ERROR "Missing test source: ${src}")
  endif()
  set(tgt rocsparse_fortran_test_${name}${suffix})
  set_source_files_properties("${src}" PROPERTIES LANGUAGE Fortran)
  add_executable(${tgt} "${src}")
  set_target_properties(${tgt} PROPERTIES LINKER_LANGUAGE Fortran)
  target_link_libraries(${tgt} PRIVATE
    rocsparse_fortran hip::hip_fortran roc::rocsparse)
  # The tests call EXIT, a GNU extension in gfortran.
  target_compile_options(${tgt} PRIVATE
    $<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-std=gnu>)
  add_test(NAME rocsparse_fortran_${name}${suffix} COMMAND ${tgt})
  # Needs a GPU; a GPU-less machine can run ctest -LE gpu.
  set_tests_properties(rocsparse_fortran_${name}${suffix} PROPERTIES LABELS "gpu")
endfunction()

set(ROCSPARSE_FORTRAN_TESTS
    ddoti
    ddoti_typed
    rocsparse_cbsrmv
    rocsparse_cbsrmv_devptr
    rocsparse_ccsr2csc
    rocsparse_ccsrgeam
    rocsparse_ccsrgeam_devptr
    rocsparse_ccsrgemm
    rocsparse_ccsrgemm_devptr
    rocsparse_ccsric0
    rocsparse_ccsrilu0
    rocsparse_cgebsrmv
    rocsparse_cgebsrmv_devptr
    rocsparse_cgemvi
    rocsparse_cgemvi_devptr
    rocsparse_cgthr
    rocsparse_coo2csr
    rocsparse_csctr
    rocsparse_csddmm
    rocsparse_csddmm_devptr
    rocsparse_cspmm
    rocsparse_cspmm_devptr
    rocsparse_cspmv
    rocsparse_cspmv_devptr
    rocsparse_csptrsm
    rocsparse_csptrsm_devptr
    rocsparse_csptrsv
    rocsparse_csptrsv_devptr
    rocsparse_csr2coo
    rocsparse_dbsrmv
    rocsparse_dbsrmv_devptr
    rocsparse_dcsr2csc
    rocsparse_dcsrgeam
    rocsparse_dcsrgeam_devptr
    rocsparse_dcsrgemm
    rocsparse_dcsrgemm_devptr
    rocsparse_dcsrgemm_typed
    rocsparse_dcsric0
    rocsparse_dcsrilu0
    rocsparse_descr_get
    rocsparse_dgebsrmv
    rocsparse_dgebsrmv_devptr
    rocsparse_dgemvi
    rocsparse_dgemvi_devptr
    rocsparse_dgemvi_typed
    rocsparse_dgthr
    rocsparse_dsctr
    rocsparse_dsddmm
    rocsparse_dsddmm_devptr
    rocsparse_dspmm
    rocsparse_dspmm_devptr
    rocsparse_dspmv
    rocsparse_dspmv_devptr
    rocsparse_dsptrsm
    rocsparse_dsptrsm_devptr
    rocsparse_dsptrsv
    rocsparse_dsptrsv_devptr
    rocsparse_handle_create
    rocsparse_pointer_mode
    rocsparse_sbsrmv
    rocsparse_sbsrmv_devptr
    rocsparse_scsr2csc
    rocsparse_scsrgeam
    rocsparse_scsrgeam_devptr
    rocsparse_scsrgemm
    rocsparse_scsrgemm_devptr
    rocsparse_scsric0
    rocsparse_scsrilu0
    rocsparse_sgebsrmv
    rocsparse_sgebsrmv_devptr
    rocsparse_sgemvi
    rocsparse_sgemvi_devptr
    rocsparse_sgthr
    rocsparse_spildlt0
    rocsparse_ssctr
    rocsparse_ssddmm
    rocsparse_ssddmm_devptr
    rocsparse_sspmm
    rocsparse_sspmm_devptr
    rocsparse_sspmv
    rocsparse_sspmv_devptr
    rocsparse_ssptrsm
    rocsparse_ssptrsm_devptr
    rocsparse_ssptrsv
    rocsparse_ssptrsv_devptr
    rocsparse_version
    rocsparse_zbsrmv
    rocsparse_zbsrmv_devptr
    rocsparse_zcsr2csc
    rocsparse_zcsrgeam
    rocsparse_zcsrgeam_devptr
    rocsparse_zcsrgemm
    rocsparse_zcsrgemm_devptr
    rocsparse_zcsric0
    rocsparse_zcsrilu0
    rocsparse_zgebsrmv
    rocsparse_zgebsrmv_devptr
    rocsparse_zgemvi
    rocsparse_zgemvi_devptr
    rocsparse_zgthr
    rocsparse_zsctr
    rocsparse_zsddmm
    rocsparse_zsddmm_devptr
    rocsparse_zspmm
    rocsparse_zspmm_devptr
    rocsparse_zspmv
    rocsparse_zspmv_devptr
    rocsparse_zsptrsm
    rocsparse_zsptrsm_devptr
    rocsparse_zsptrsv
    rocsparse_zsptrsv_devptr
    zgpsv_interleaved_batch)

set(ROCSPARSE_FORTRAN_F2008_TESTS
    ddoti
    rocsparse_cbsrmv
    rocsparse_ccsr2csc
    rocsparse_ccsrgeam
    rocsparse_ccsrgemm
    rocsparse_ccsric0
    rocsparse_ccsrilu0
    rocsparse_cgebsrmv
    rocsparse_cgemvi
    rocsparse_cgthr
    rocsparse_coo2csr
    rocsparse_csctr
    rocsparse_csddmm
    rocsparse_cspmm
    rocsparse_cspmv
    rocsparse_csptrsm
    rocsparse_csptrsv
    rocsparse_csr2coo
    rocsparse_dbsrmv
    rocsparse_dcsr2csc
    rocsparse_dcsrgeam
    rocsparse_dcsrgemm
    rocsparse_dcsric0
    rocsparse_dcsrilu0
    rocsparse_descr_get
    rocsparse_dgebsrmv
    rocsparse_dgemvi
    rocsparse_dgthr
    rocsparse_dsctr
    rocsparse_dsddmm
    rocsparse_dspmm
    rocsparse_dspmv
    rocsparse_dsptrsm
    rocsparse_dsptrsv
    rocsparse_pointer_mode
    rocsparse_sbsrmv
    rocsparse_scsr2csc
    rocsparse_scsrgeam
    rocsparse_scsrgemm
    rocsparse_scsric0
    rocsparse_scsrilu0
    rocsparse_sgebsrmv
    rocsparse_sgemvi
    rocsparse_sgthr
    rocsparse_sgtsv
    rocsparse_spildlt0
    rocsparse_ssctr
    rocsparse_ssddmm
    rocsparse_sspmm
    rocsparse_sspmv
    rocsparse_ssptrsm
    rocsparse_ssptrsv
    rocsparse_version
    rocsparse_zbsrmv
    rocsparse_zcsr2csc
    rocsparse_zcsrgeam
    rocsparse_zcsrgemm
    rocsparse_zcsric0
    rocsparse_zcsrilu0
    rocsparse_zgebsrmv
    rocsparse_zgemvi
    rocsparse_zgthr
    rocsparse_zsctr
    rocsparse_zsddmm
    rocsparse_zspmm
    rocsparse_zspmv
    rocsparse_zsptrsm
    rocsparse_zsptrsv
    zgpsv_interleaved_batch)

if(_have_hip_binding)
  foreach(_t IN LISTS ROCSPARSE_FORTRAN_TESTS)
    rocsparse_fortran_add_test(${_t} f2003)
  endforeach()
  if(NOT FORTRAN_ARRAY_INTERFACES STREQUAL "none")
    foreach(_t IN LISTS ROCSPARSE_FORTRAN_F2008_TESTS)
      rocsparse_fortran_add_test(${_t} f2008)
    endforeach()
  endif()
endif()
