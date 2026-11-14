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

# Intel's ifx/ifort do not treat .f03/.f08 as Fortran source by default.
if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  string(REPLACE "<SOURCE>" "-Tf <SOURCE>"
         CMAKE_Fortran_COMPILE_OBJECT "${CMAKE_Fortran_COMPILE_OBJECT}")
endif()

# The archive has one section per procedure: keep only the wrappers a test calls.
if(NOT APPLE AND NOT WIN32)
  add_link_options(-Wl,--gc-sections)
endif()

if(NOT TARGET hip::hip_fortran)
  find_package(hip-fortran QUIET)
endif()
if(NOT TARGET hip::hip_fortran)
  message(STATUS
    "rocBLAS: no hip::hip_fortran target, skipping the Fortran binding tests "
    "(they `use hip`). Point CMAKE_PREFIX_PATH at a ROCm whose HIP Fortran "
    "bindings were built with ${CMAKE_Fortran_COMPILER}.")
  return()
endif()

if(NOT TARGET roc::rocblas)
  message(STATUS
    "rocBLAS: no roc::rocblas target, skipping the Fortran binding tests "
    "(they link librocblas). Point CMAKE_PREFIX_PATH at an installed ROCm.")
  return()
endif()

# rocblas_fortran_add_test(<name> <ext>): a .f03 test calls the C interfaces; a
# .f08 test calls the array overloads and registers as <name>_f2008.
function(rocblas_fortran_add_test name ext)
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/${name}.${ext}")
  if(ext STREQUAL "f08")
    set(name "${name}_f2008")
  endif()
  set_source_files_properties("${src}" PROPERTIES LANGUAGE Fortran)
  add_executable(rocblas_fortran_test_${name} "${src}")
  set_target_properties(rocblas_fortran_test_${name} PROPERTIES LINKER_LANGUAGE Fortran)
  target_link_libraries(rocblas_fortran_test_${name} PRIVATE
    rocblas_fortran hip::hip_fortran roc::rocblas)
  # The tests call EXIT, a GNU extension in gfortran.
  if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    target_compile_options(rocblas_fortran_test_${name} PRIVATE -std=gnu)
  endif()
  add_test(NAME rocblas_fortran_${name} COMMAND rocblas_fortran_test_${name})
  set_tests_properties(rocblas_fortran_${name} PROPERTIES LABELS "gpu")
endfunction()

set(ROCBLAS_FORTRAN_TESTS
    caxpy
    caxpy_devptr
    caxpy_typed
    cdotc
    cdotu
    cgeam
    cgeam_devptr
    cgemm
    cgemm_batched
    cgemm_batched_devptr
    cgemm_devptr
    cgemm_strided_batched
    cgemm_strided_batched_devptr
    cgemm_trans
    cgemv
    cgemv_devptr
    cgemv_trans
    cgerc
    cgerc_devptr
    cgeru
    cgeru_devptr
    chemm
    chemm_devptr
    cherk
    cherk_devptr
    cscal
    cscal_devptr
    csscal
    csscal_devptr
    csymm
    csymm_devptr
    csyrk
    csyrk_devptr
    ctrsm
    ctrsm_devptr
    ctrsv
    dasum
    daxpy
    daxpy_devptr
    ddot
    ddot_typed
    dgeam
    dgeam_devptr
    dgemm
    dgemm_batched
    dgemm_batched_devptr
    dgemm_devptr
    dgemm_strided_batched
    dgemm_strided_batched_devptr
    dgemm_trans
    dgemm_typed
    dgemv
    dgemv_devptr
    dgemv_trans
    dger
    dger_devptr
    dnrm2
    dscal
    dscal_devptr
    dsymm
    dsymm_devptr
    dsyrk
    dsyrk_devptr
    dtrmm
    dtrmm_devptr
    dtrsm
    dtrsm_devptr
    dtrsv
    dzasum
    dznrm2
    gemm_ex
    gemm_ex_devptr
    icamax
    icamin
    idamax
    idamin
    isamax
    isamin
    izamax
    izamin
    rocblas_handle_modes
    rocblas_handle_query
    rocblas_version
    sasum
    saxpy
    saxpy_devptr
    scasum
    scnrm2
    sdot
    sgeam
    sgeam_devptr
    sgemm
    sgemm_batched
    sgemm_batched_devptr
    sgemm_devptr
    sgemm_strided_batched
    sgemm_strided_batched_devptr
    sgemm_trans
    sgemv
    sgemv_devptr
    sgemv_trans
    sger
    sger_devptr
    snrm2
    sscal
    sscal_devptr
    ssymm
    ssymm_devptr
    ssyrk
    ssyrk_devptr
    strsm
    strsm_devptr
    strsv
    zaxpy
    zaxpy_devptr
    zdotc
    zdotu
    zdscal
    zdscal_devptr
    zgeam
    zgeam_devptr
    zgemm
    zgemm_batched
    zgemm_batched_devptr
    zgemm_devptr
    zgemm_strided_batched
    zgemm_strided_batched_devptr
    zgemm_trans
    zgemv
    zgemv_devptr
    zgemv_trans
    zgerc
    zgerc_devptr
    zgeru
    zgeru_devptr
    zhemm
    zhemm_devptr
    zherk
    zherk_devptr
    zscal
    zscal_devptr
    zsymm
    zsymm_devptr
    zsyrk
    zsyrk_devptr
    ztrsm
    ztrsm_devptr
    ztrsv)

set(ROCBLAS_FORTRAN_F2008_TESTS
    caxpy
    cdotc
    cdotu
    cgeam
    cgemm
    cgemm_batched
    cgemm_strided_batched
    cgemm_trans
    cgemv
    cgemv_trans
    cgerc
    cgeru
    chemm
    cherk
    cscal
    csscal
    csymm
    csyrk
    ctrsm
    ctrsv
    dasum
    daxpy
    ddot
    ddot_strided_batched_result
    dgeam
    dgemm
    dgemm_batched
    dgemm_ex_fitness_query
    dgemm_strided_batched
    dgemm_strided_batched_ranks
    dgemm_trans
    dgemv
    dgemv_batched
    dgemv_trans
    dger
    dnrm2
    drotm_param
    dscal
    dsymm
    dsyrk
    dtrmm
    dtrsm
    dtrsv
    dzasum
    dznrm2
    gemm_ex
    icamax
    icamin
    idamax
    idamin
    isamax
    isamin
    izamax
    izamin
    rocblas_handle_modes
    rocblas_handle_query
    rocblas_version
    sasum
    saxpy
    scasum
    scnrm2
    sdot
    sgeam
    sgemm
    sgemm_batched
    sgemm_strided_batched
    sgemm_trans
    sgemv
    sgemv_trans
    sger
    snrm2
    sscal
    ssymm
    ssyrk
    stpsv
    strsm
    strsv
    zaxpy
    zdotc
    zdotu
    zdscal
    zgeam
    zgemm
    zgemm_batched
    zgemm_strided_batched
    zgemm_trans
    zgemv
    zgemv_trans
    zgerc
    zgeru
    zhemm
    zherk
    zscal
    zsymm
    zsyrk
    ztrsm
    ztrsv)

foreach(_t IN LISTS ROCBLAS_FORTRAN_TESTS)
  rocblas_fortran_add_test(${_t} f03)
endforeach()
# The .f08 tests need the array overloads, which FORTRAN_ARRAY_INTERFACES=none leaves out.
if(NOT FORTRAN_ARRAY_INTERFACES STREQUAL "none")
  foreach(_t IN LISTS ROCBLAS_FORTRAN_F2008_TESTS)
    rocblas_fortran_add_test(${_t} f08)
  endforeach()
endif()

# Installed tests: the executables go to bin/, and a CTestTestfile.cmake with
# the same tests and properties to bin/rocblas-fortran/, for ctest --test-dir.
include(GNUInstallDirs)
get_property(_rocblas_fortran_tests DIRECTORY PROPERTY TESTS)
if(_rocblas_fortran_tests)
  set(_ctest "${CMAKE_CURRENT_BINARY_DIR}/install-CTestTestfile.cmake")
  file(WRITE "${_ctest}" "")
  foreach(_test IN LISTS _rocblas_fortran_tests)
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
    DESTINATION "${CMAKE_INSTALL_BINDIR}/rocblas-fortran"
    COMPONENT tests
    RENAME CTestTestfile.cmake
  )
endif()
