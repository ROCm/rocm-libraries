# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# hipSPARSE Fortran binding tests: one CTest executable per test, GPU required.

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

# f2003 tests (<name>.f03) call the C-binding interfaces; f2008 tests
# (<name>.f08) the array overloads, which exist only when
# FORTRAN_ARRAY_INTERFACES is not none. Their CTest names carry an _f2008 suffix.
function(hipsparse_fortran_add_test name std)
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
  set(tgt hipsparse_fortran_test_${name}${suffix})
  set_source_files_properties("${src}" PROPERTIES LANGUAGE Fortran)
  add_executable(${tgt} "${src}")
  set_target_properties(${tgt} PROPERTIES LINKER_LANGUAGE Fortran)
  target_link_libraries(${tgt} PRIVATE
    hipsparse_fortran hip::hip_fortran roc::hipsparse)
  # The tests call EXIT, a GNU extension in gfortran.
  target_compile_options(${tgt} PRIVATE
    $<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-std=gnu>)
  add_test(NAME hipsparse_fortran_${name}${suffix} COMMAND ${tgt})
  # Needs a GPU; a GPU-less machine can run ctest -LE gpu.
  set_tests_properties(hipsparse_fortran_${name}${suffix} PROPERTIES LABELS "gpu")
endfunction()

set(HIPSPARSE_FORTRAN_TESTS
    hipsparse_ccsr2csc
    hipsparse_ccsrgemm
    hipsparse_ccsrilu02
    hipsparse_ccsrsv2
    hipsparse_ccsrsv2_devptr
    hipsparse_cgemvi
    hipsparse_cgemvi_devptr
    hipsparse_cgthr
    hipsparse_csctr
    hipsparse_csddmm
    hipsparse_csddmm_devptr
    hipsparse_cspmm
    hipsparse_cspmm_devptr
    hipsparse_cspmv
    hipsparse_cspmv_devptr
    hipsparse_csptrsm
    hipsparse_csptrsm_devptr
    hipsparse_csptrsv
    hipsparse_csptrsv_devptr
    hipsparse_dcsr2csc
    hipsparse_dcsrgemm
    hipsparse_dcsrilu02
    hipsparse_dcsrsv2
    hipsparse_dcsrsv2_devptr
    hipsparse_descr_get
    hipsparse_dgemvi
    hipsparse_dgemvi_devptr
    hipsparse_dgemvi_typed
    hipsparse_dgthr
    hipsparse_dsctr
    hipsparse_dsddmm
    hipsparse_dsddmm_devptr
    hipsparse_dspmm
    hipsparse_dspmm_devptr
    hipsparse_dspmv
    hipsparse_dspmv_devptr
    hipsparse_dsptrsm
    hipsparse_dsptrsm_devptr
    hipsparse_dsptrsv
    hipsparse_dsptrsv_devptr
    hipsparse_pointer_mode
    hipsparse_scsrgemm
    hipsparse_scsrilu02
    hipsparse_scsrsv2
    hipsparse_scsrsv2_devptr
    hipsparse_sgemvi
    hipsparse_sgemvi_devptr
    hipsparse_sgthr
    hipsparse_ssctr
    hipsparse_ssddmm
    hipsparse_ssddmm_devptr
    hipsparse_sspmm
    hipsparse_sspmm_devptr
    hipsparse_sspmv
    hipsparse_sspmv_devptr
    hipsparse_ssptrsm
    hipsparse_ssptrsm_devptr
    hipsparse_ssptrsv
    hipsparse_ssptrsv_devptr
    hipsparse_version
    hipsparse_xcoo2csr
    hipsparse_xcsr2coo
    hipsparse_zcsr2csc
    hipsparse_zcsrgemm
    hipsparse_zcsrilu02
    hipsparse_zcsrsv2
    hipsparse_zcsrsv2_devptr
    hipsparse_zcsrsv2_typed
    hipsparse_zgemvi
    hipsparse_zgemvi_devptr
    hipsparse_zgthr
    hipsparse_zsctr
    hipsparse_zsddmm
    hipsparse_zsddmm_devptr
    hipsparse_zspmm
    hipsparse_zspmm_devptr
    hipsparse_zspmv
    hipsparse_zspmv_devptr
    hipsparse_zsptrsm
    hipsparse_zsptrsm_devptr
    hipsparse_zsptrsv
    hipsparse_zsptrsv_devptr)

set(HIPSPARSE_FORTRAN_F2008_TESTS
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
    hipsparse_scsr2csc
    hipsparse_scsrgemm
    hipsparse_scsrilu02
    hipsparse_scsrsv2
    hipsparse_sgemvi
    hipsparse_sgthr
    hipsparse_sgtsv
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
    hipsparse_fortran_add_test(${_t} f2003)
  endforeach()
  if(NOT FORTRAN_ARRAY_INTERFACES STREQUAL "none")
    foreach(_t IN LISTS HIPSPARSE_FORTRAN_F2008_TESTS)
      hipsparse_fortran_add_test(${_t} f2008)
    endforeach()
  endif()
endif()
