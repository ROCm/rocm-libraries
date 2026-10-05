# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# hipSPARSE Fortran binding tests: one CTest executable per test, GPU required.

# ifx/ifort skip .f03 sources unless told the language with -Tf.
if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  string(REPLACE "<SOURCE>" "-Tf <SOURCE>"
         CMAKE_Fortran_COMPILE_OBJECT "${CMAKE_Fortran_COMPILE_OBJECT}")
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
    "hipSPARSE: no hip::hip_fortran target, skipping the Fortran binding tests "
    "(they `use hip`). Point CMAKE_PREFIX_PATH at a ROCm whose HIP Fortran "
    "bindings were built with ${CMAKE_Fortran_COMPILER}.")
endif()

# Standalone libhipsparse has to be found; its find_dependency(hip) needs C.
if(NOT TARGET roc::hipsparse)
  if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
    enable_language(C)
  endif()
  find_package(hipsparse QUIET)
endif()
if(NOT TARGET roc::hipsparse)
  message(STATUS
    "hipSPARSE: no roc::hipsparse target, skipping the Fortran binding tests "
    "(they link libhipsparse). Point CMAKE_PREFIX_PATH at an installed ROCm to "
    "build them standalone.")
  return()
endif()

function(hipsparse_fortran_add_test name)
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/${name}.f03")
  if(NOT EXISTS "${src}")
    message(FATAL_ERROR "Missing test source: ${src}")
  endif()
  set_source_files_properties("${src}" PROPERTIES LANGUAGE Fortran)
  add_executable(hipsparse_fortran_test_${name} "${src}")
  set_target_properties(hipsparse_fortran_test_${name} PROPERTIES LINKER_LANGUAGE Fortran)
  target_link_libraries(hipsparse_fortran_test_${name} PRIVATE
    hipsparse_fortran hip::hip_fortran roc::hipsparse)
  # The tests call EXIT, a GNU extension in gfortran.
  target_compile_options(hipsparse_fortran_test_${name} PRIVATE
    $<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-std=gnu>)
  add_test(NAME hipsparse_fortran_${name} COMMAND hipsparse_fortran_test_${name})
  # Needs a GPU; a GPU-less machine can run ctest -LE gpu.
  set_tests_properties(hipsparse_fortran_${name} PROPERTIES LABELS "gpu")
endfunction()

set(HIPSPARSE_FORTRAN_TESTS
    hipsparse_ccsr2csc
    hipsparse_ccsrgemm
    hipsparse_ccsrilu02
    hipsparse_ccsrsv2
    hipsparse_cgemvi
    hipsparse_cgthr
    hipsparse_csctr
    hipsparse_csddmm
    hipsparse_cspmm
    hipsparse_cspmv
    hipsparse_csptrsm
    hipsparse_csptrsv
    hipsparse_dcsr2csc
    hipsparse_dcsrgemm
    hipsparse_dcsrilu02
    hipsparse_dcsrsv2
    hipsparse_descr_get
    hipsparse_dgemvi
    hipsparse_dgthr
    hipsparse_dsctr
    hipsparse_dsddmm
    hipsparse_dspmm
    hipsparse_dspmv
    hipsparse_dsptrsm
    hipsparse_dsptrsv
    hipsparse_pointer_mode
    hipsparse_scsrgemm
    hipsparse_scsrilu02
    hipsparse_scsrsv2
    hipsparse_sgemvi
    hipsparse_sgthr
    hipsparse_ssctr
    hipsparse_ssddmm
    hipsparse_sspmm
    hipsparse_sspmv
    hipsparse_ssptrsm
    hipsparse_ssptrsv
    hipsparse_version
    hipsparse_xcoo2csr
    hipsparse_xcsr2coo
    hipsparse_zcsr2csc
    hipsparse_zcsrgemm
    hipsparse_zcsrilu02
    hipsparse_zcsrsv2
    hipsparse_zgemvi
    hipsparse_zgthr
    hipsparse_zsctr
    hipsparse_zsddmm
    hipsparse_zspmm
    hipsparse_zspmv
    hipsparse_zsptrsm
    hipsparse_zsptrsv)

if(_have_hip_binding)
  foreach(_t IN LISTS HIPSPARSE_FORTRAN_TESTS)
    hipsparse_fortran_add_test(${_t})
  endforeach()
endif()
