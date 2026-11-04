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
    "rocSOLVER: no hip::hip_fortran target, skipping the Fortran binding tests "
    "(they `use hip`). Point CMAKE_PREFIX_PATH at a ROCm whose HIP Fortran "
    "bindings were built with ${CMAKE_Fortran_COMPILER}.")
endif()

if(NOT TARGET roc::rocsolver)
  message(STATUS
    "rocSOLVER: no roc::rocsolver target, skipping the Fortran binding tests "
    "(they link librocsolver). Point CMAKE_PREFIX_PATH at an installed ROCm.")
  return()
endif()

# A .f03 test calls the C interfaces; a .f08 test calls the array overloads and
# registers as <name>_f2008.
function(rocsolver_fortran_add_test name ext)
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/${name}.${ext}")
  if(ext STREQUAL "f08")
    set(name "${name}_f2008")
  endif()
  if(NOT EXISTS "${src}")
    message(FATAL_ERROR "Missing test source: ${src}")
  endif()
  set_source_files_properties("${src}" PROPERTIES LANGUAGE Fortran)
  add_executable(rocsolver_fortran_test_${name} "${src}")
  set_target_properties(rocsolver_fortran_test_${name} PROPERTIES LINKER_LANGUAGE Fortran)
  target_link_libraries(rocsolver_fortran_test_${name} PRIVATE
    rocsolver_fortran hip::hip_fortran roc::rocsolver)
  # The tests call EXIT, a GNU extension in gfortran.
  if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    target_compile_options(rocsolver_fortran_test_${name} PRIVATE -std=gnu)
  endif()
  add_test(NAME rocsolver_fortran_${name} COMMAND rocsolver_fortran_test_${name})
  # Lets a GPU-less machine skip them with ctest -LE gpu.
  set_tests_properties(rocsolver_fortran_${name} PROPERTIES LABELS "gpu")
endfunction()

set(ROCSOLVER_FORTRAN_TESTS
    rocsolver_alg_mode
    rocsolver_cgebrd
    rocsolver_cgels
    rocsolver_cgeqrf
    rocsolver_cgesvd
    rocsolver_cgesvdj
    rocsolver_cgetrf
    rocsolver_cgetrs
    rocsolver_cheevd
    rocsolver_cheevdx
    rocsolver_cheevj
    rocsolver_chegvd
    rocsolver_chegvdx
    rocsolver_cpotrf
    rocsolver_cpotrs
    rocsolver_csytrf
    rocsolver_csytrs
    rocsolver_cungqr
    rocsolver_cunmqr
    rocsolver_dgebrd
    rocsolver_dgels
    rocsolver_dgeqrf
    rocsolver_dgesv
    rocsolver_dgesvd
    rocsolver_dgesvdj
    rocsolver_dgetrf
    rocsolver_dgetrf_strided_batched
    rocsolver_dgetri
    rocsolver_dgetrs
    rocsolver_dlarft
    rocsolver_dlatrd
    rocsolver_dorgqr
    rocsolver_dormqr
    rocsolver_dposv
    rocsolver_dpotrf
    rocsolver_dpotrs
    rocsolver_dsterf
    rocsolver_dsyev
    rocsolver_dsyevd
    rocsolver_dsyevdx
    rocsolver_dsyevj
    rocsolver_dsygvd
    rocsolver_dsygvdx
    rocsolver_dsytrd
    rocsolver_dsytrf
    rocsolver_dsytrs
    rocsolver_dtrtri
    rocsolver_sgebrd
    rocsolver_sgels
    rocsolver_sgeqrf
    rocsolver_sgesvd
    rocsolver_sgesvdj
    rocsolver_sgetrf
    rocsolver_sgetrs
    rocsolver_sorgqr
    rocsolver_sormqr
    rocsolver_spotrf
    rocsolver_spotrs
    rocsolver_ssyevd
    rocsolver_ssyevdx
    rocsolver_ssyevj
    rocsolver_ssygvd
    rocsolver_ssygvdx
    rocsolver_ssytrf
    rocsolver_ssytrs
    rocsolver_version
    rocsolver_zgebrd
    rocsolver_zgels
    rocsolver_zgeqrf
    rocsolver_zgesvd
    rocsolver_zgesvdj
    rocsolver_zgetrf
    rocsolver_zgetrs
    rocsolver_zheevd
    rocsolver_zheevdx
    rocsolver_zheevj
    rocsolver_zhegvd
    rocsolver_zhegvdx
    rocsolver_zpotrf
    rocsolver_zpotrs
    rocsolver_zsytrf
    rocsolver_zsytrs
    rocsolver_zungqr
    rocsolver_zunmqr)

set(ROCSOLVER_FORTRAN_F2008_TESTS
    rocsolver_alg_mode
    rocsolver_cgebrd
    rocsolver_cgels
    rocsolver_cgeqrf
    rocsolver_cgesvd
    rocsolver_cgesvdj
    rocsolver_cgetrf
    rocsolver_cgetrs
    rocsolver_cheevd
    rocsolver_cheevdx
    rocsolver_cheevj
    rocsolver_chegvd
    rocsolver_chegvdx
    rocsolver_cpotrf
    rocsolver_cpotrs
    rocsolver_csytrf
    rocsolver_csytrs
    rocsolver_cungqr
    rocsolver_cunmqr
    rocsolver_dgebrd
    rocsolver_dgels
    rocsolver_dgeqrf
    rocsolver_dgesv
    rocsolver_dgesvd
    rocsolver_dgesvdj
    rocsolver_dgetf2
    rocsolver_dgetf2_64
    rocsolver_dgetf2_npvt
    rocsolver_dgetrf
    rocsolver_dgetrf_64
    rocsolver_dgetrf_batched
    rocsolver_dgetrf_npvt
    rocsolver_dgetrf_strided_batched
    rocsolver_dgetri
    rocsolver_dgetri_npvt
    rocsolver_dgetrs
    rocsolver_dlarft
    rocsolver_dlatrd
    rocsolver_dorgqr
    rocsolver_dormqr
    rocsolver_dposv
    rocsolver_dpotf2
    rocsolver_dpotf2_64
    rocsolver_dpotrf
    rocsolver_dpotrf_64
    rocsolver_dpotri
    rocsolver_dpotrs
    rocsolver_dstedc
    rocsolver_dsteqr
    rocsolver_dsterf
    rocsolver_dsyev
    rocsolver_dsyevd
    rocsolver_dsyevdx
    rocsolver_dsyevj
    rocsolver_dsygvd
    rocsolver_dsygvdx
    rocsolver_dsytf2
    rocsolver_dsytrd
    rocsolver_dsytrf
    rocsolver_dsytrs
    rocsolver_dtrtri
    rocsolver_sgebrd
    rocsolver_sgels
    rocsolver_sgeqrf
    rocsolver_sgesvd
    rocsolver_sgesvdj
    rocsolver_sgetrf
    rocsolver_sgetrs
    rocsolver_sorgqr
    rocsolver_sormqr
    rocsolver_spotrf
    rocsolver_spotrs
    rocsolver_ssyevd
    rocsolver_ssyevdx
    rocsolver_ssyevj
    rocsolver_ssygvd
    rocsolver_ssygvdx
    rocsolver_ssytrf
    rocsolver_ssytrs
    rocsolver_version
    rocsolver_zgebrd
    rocsolver_zgels
    rocsolver_zgeqrf
    rocsolver_zgesvd
    rocsolver_zgesvdj
    rocsolver_zgetrf
    rocsolver_zgetrs
    rocsolver_zheev
    rocsolver_zheevd
    rocsolver_zheevdx
    rocsolver_zheevj
    rocsolver_zhegvd
    rocsolver_zhegvdx
    rocsolver_zpotrf
    rocsolver_zpotrs
    rocsolver_zsytrf
    rocsolver_zsytrs
    rocsolver_zungqr
    rocsolver_zunmqr)

# The .f08 tests need the array overloads, which FORTRAN_ARRAY_INTERFACES=none leaves out.
if(_have_hip_binding)
  foreach(_t IN LISTS ROCSOLVER_FORTRAN_TESTS)
    rocsolver_fortran_add_test(${_t} f03)
  endforeach()
  if(NOT FORTRAN_ARRAY_INTERFACES STREQUAL "none")
    foreach(_t IN LISTS ROCSOLVER_FORTRAN_F2008_TESTS)
      rocsolver_fortran_add_test(${_t} f08)
    endforeach()
  endif()
endif()

# Installed tests: the executables go to bin/, and a CTestTestfile.cmake with
# the same tests and properties to bin/rocsolver-fortran/, for ctest --test-dir.
include(GNUInstallDirs)
get_property(_rocsolver_fortran_tests DIRECTORY PROPERTY TESTS)
if(_rocsolver_fortran_tests)
  set(_ctest "${CMAKE_CURRENT_BINARY_DIR}/install-CTestTestfile.cmake")
  file(WRITE "${_ctest}" "")
  foreach(_test IN LISTS _rocsolver_fortran_tests)
    string(REPLACE "_fortran_" "_fortran_test_" _target "${_test}")
    install(TARGETS ${_target} RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT tests)
    file(APPEND "${_ctest}" "add_test(${_test} \"../${_target}\")\n")
    foreach(_prop LABELS SKIP_RETURN_CODE ENVIRONMENT PASS_REGULAR_EXPRESSION FAIL_REGULAR_EXPRESSION)
      get_test_property(${_test} ${_prop} _value)
      if(_value)
        file(APPEND "${_ctest}" "set_tests_properties(${_test} PROPERTIES ${_prop} [=[${_value}]=])\n")
      endif()
    endforeach()
  endforeach()
  install(
    FILES "${_ctest}"
    DESTINATION "${CMAKE_INSTALL_BINDIR}/rocsolver-fortran"
    COMPONENT tests
    RENAME CTestTestfile.cmake
  )
endif()
