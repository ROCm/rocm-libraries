# ########################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
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

# ifx/ifort skip .f03/.f08 files; -Tf forces Fortran, which the LANGUAGE property alone does not.
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

# The tests say `use hip`, so a hip.mod from this compiler is needed to compile them.
if(NOT TARGET hip::hip_fortran)
  find_package(hip-fortran QUIET)
endif()
set(_have_hip_binding TRUE)
if(NOT TARGET hip::hip_fortran)
  set(_have_hip_binding FALSE)
  message(STATUS
    "hipSOLVER: no hip::hip_fortran target, skipping the Fortran binding tests "
    "(they `use hip`). Point CMAKE_PREFIX_PATH at a ROCm whose HIP Fortran "
    "bindings were built with ${CMAKE_Fortran_COMPILER}.")
endif()

# C is enabled because hip-config.cmake reaches FindThreads.
if(NOT TARGET roc::hipsolver)
  if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
    enable_language(C)
  endif()
  find_package(hipsolver QUIET)
endif()
if(NOT TARGET roc::hipsolver)
  message(STATUS
    "hipSOLVER: no roc::hipsolver target, skipping the Fortran binding tests "
    "(they link libhipsolver). Point CMAKE_PREFIX_PATH at an installed ROCm to "
    "build them standalone.")
  return()
endif()

# A .f03 test calls the C interfaces; a .f08 test calls the array overloads and
# registers as <name>_f2008.
function(hipsolver_fortran_add_test name ext)
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/${name}.${ext}")
  if(ext STREQUAL "f08")
    set(name "${name}_f2008")
  endif()
  if(NOT EXISTS "${src}")
    message(FATAL_ERROR "Missing test source: ${src}")
  endif()
  set_source_files_properties("${src}" PROPERTIES LANGUAGE Fortran)
  add_executable(hipsolver_fortran_test_${name} "${src}")
  set_target_properties(hipsolver_fortran_test_${name} PROPERTIES LINKER_LANGUAGE Fortran)
  target_link_libraries(hipsolver_fortran_test_${name} PRIVATE
    hipsolver_fortran hip::hip_fortran roc::hipsolver)
  # The tests call EXIT, a GNU extension in gfortran.
  if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    target_compile_options(hipsolver_fortran_test_${name} PRIVATE -std=gnu)
  endif()
  add_test(NAME hipsolver_fortran_${name} COMMAND hipsolver_fortran_test_${name})
  # Lets a GPU-less machine skip them with ctest -LE gpu.
  set_tests_properties(hipsolver_fortran_${name} PROPERTIES LABELS "gpu")
endfunction()

set(HIPSOLVER_FORTRAN_TESTS
    hipsolver_cgeqrf
    hipsolver_cgesvd
    hipsolver_cgesvdj
    hipsolver_cgetrf
    hipsolver_cheevd
    hipsolver_cheevdx
    hipsolver_chegvd
    hipsolver_chegvdx
    hipsolver_cpotrfbatched
    hipsolver_cpotrs
    hipsolver_cungqr
    hipsolver_cunmqr
    hipsolver_deterministic_mode
    hipsolver_dgeqrf
    hipsolver_dgesvdj
    hipsolver_dgetrf
    hipsolver_dorgqr
    hipsolver_dormqr
    hipsolver_dpotrfbatched
    hipsolver_dpotrs
    hipsolver_dsyevdx
    hipsolver_dsyevj
    hipsolver_dsygvd
    hipsolver_dsygvdx
    hipsolver_rf_params
    hipsolver_sgeqrf
    hipsolver_sgesvd
    hipsolver_sgesvdj
    hipsolver_sgetrf
    hipsolver_sorgqr
    hipsolver_sormqr
    hipsolver_spotrfbatched
    hipsolver_spotrs
    hipsolver_ssyevd
    hipsolver_ssyevdx
    hipsolver_ssyevj
    hipsolver_ssygvd
    hipsolver_ssygvdx
    hipsolver_syevj_info
    hipsolver_zgeqrf
    hipsolver_zgesvd
    hipsolver_zgesvdj
    hipsolver_zgetrf
    hipsolver_zheevd
    hipsolver_zheevdx
    hipsolver_zhegvd
    hipsolver_zhegvdx
    hipsolver_zpotrfbatched
    hipsolver_zpotrs
    hipsolver_zungqr
    hipsolver_zunmqr)

set(HIPSOLVER_FORTRAN_F2008_TESTS
    hipsolver_cgeqrf
    hipsolver_cgesvd
    hipsolver_cgesvdj
    hipsolver_cgetrf
    hipsolver_cheevd
    hipsolver_cheevdx
    hipsolver_chegvd
    hipsolver_chegvdx
    hipsolver_cpotrfbatched
    hipsolver_cpotrs
    hipsolver_cungqr
    hipsolver_cunmqr
    hipsolver_deterministic_mode
    hipsolver_dgeqrf
    hipsolver_dgesvd
    hipsolver_dgesvdj
    hipsolver_dgetrf
    hipsolver_dgetrs
    hipsolver_dorgqr
    hipsolver_dormqr
    hipsolver_dpotrf
    hipsolver_dpotrfbatched
    hipsolver_dpotrs
    hipsolver_dsyevd
    hipsolver_dsyevdx
    hipsolver_dsyevj
    hipsolver_dsygvd
    hipsolver_dsygvdx
    hipsolver_rf_params
    hipsolver_sgeqrf
    hipsolver_sgesvd
    hipsolver_sgesvdj
    hipsolver_sgetrf
    hipsolver_sorgqr
    hipsolver_sormqr
    hipsolver_spotrfbatched
    hipsolver_spotrs
    hipsolver_ssyevd
    hipsolver_ssyevdx
    hipsolver_ssyevj
    hipsolver_ssygvd
    hipsolver_ssygvdx
    hipsolver_syevj_info
    hipsolver_zgeqrf
    hipsolver_zgesvd
    hipsolver_zgesvdj
    hipsolver_zgetrf
    hipsolver_zheevd
    hipsolver_zheevdx
    hipsolver_zhegvd
    hipsolver_zhegvdx
    hipsolver_zpotrfbatched
    hipsolver_zpotrs
    hipsolver_zungqr
    hipsolver_zunmqr)

# The .f08 tests need the array overloads, which FORTRAN_ARRAY_INTERFACES=none leaves out.
if(_have_hip_binding)
  foreach(_t IN LISTS HIPSOLVER_FORTRAN_TESTS)
    hipsolver_fortran_add_test(${_t} f03)
  endforeach()
  if(NOT FORTRAN_ARRAY_INTERFACES STREQUAL "none")
    foreach(_t IN LISTS HIPSOLVER_FORTRAN_F2008_TESTS)
      hipsolver_fortran_add_test(${_t} f08)
    endforeach()
  endif()
endif()
