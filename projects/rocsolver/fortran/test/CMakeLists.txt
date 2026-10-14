# ##########################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

# ifx/ifort skip .f03 files; -Tf forces Fortran, which the LANGUAGE property alone does not.
if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  string(REPLACE "<SOURCE>" "-Tf <SOURCE>"
         CMAKE_Fortran_COMPILE_OBJECT "${CMAKE_Fortran_COMPILE_OBJECT}")
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

# C is enabled because hip-config.cmake reaches FindThreads.
if(NOT TARGET roc::rocsolver)
  if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
    enable_language(C)
  endif()
  find_package(rocsolver QUIET)
endif()
if(NOT TARGET roc::rocsolver)
  message(STATUS
    "rocSOLVER: no roc::rocsolver target, skipping the Fortran binding tests "
    "(they link librocsolver). Point CMAKE_PREFIX_PATH at an installed ROCm to "
    "build them standalone.")
  return()
endif()

function(rocsolver_fortran_add_test name)
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/${name}.f03")
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

if(_have_hip_binding)
  foreach(_t IN LISTS ROCSOLVER_FORTRAN_TESTS)
    rocsolver_fortran_add_test(${_t})
  endforeach()
endif()
