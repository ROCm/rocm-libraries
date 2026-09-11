# Tests for the rocSPARSE Fortran bindings.
#
# One executable per test, driven by CTest. The runtime tests create a generator
# and draw numbers on a device, so they build anywhere but only pass on a machine
# with a GPU.
#
# Reached only from the parent CMakeLists, under BUILD_FORTRAN_TESTS.

# The Intel compilers (ifx/ifort) do not recognize the .f03 source extension and
# skip such files ("no action performed"). Force Fortran compilation by inserting
# -Tf immediately before the source in the compile rule; the LANGUAGE Fortran
# source property alone is not enough.
if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  string(REPLACE "<SOURCE>" "-Tf <SOURCE>"
         CMAKE_Fortran_COMPILE_OBJECT "${CMAKE_Fortran_COMPILE_OBJECT}")
endif()

# Unlike the binding itself, the RUNTIME tests are not self-contained: they
# allocate and copy device memory, so each says `use hip` as well as
# `use rocsparse`. That makes the HIP Fortran binding a COMPILE-time dependency of
# those tests: its .mod must exist, where librocsparse is only a link-time one.
# In a monorepo build hip::hip_fortran comes from projects/hip/fortran; a
# standalone build has to find it in an installed ROCm, built with this same
# compiler.
#
# The exhaustive symbol test is the exception, and the exception matters: 65 of
# the 66 sources here use hip, test_rocsparse.F03 uses only iso_c_binding and
# rocsparse. So it is registered BELOW, outside this guard. Returning early here
# would take it down with the rest, and ROCm does not currently ship a HIP
# Fortran binding at all, so against a stock /opt/rocm that early return is not a
# partial skip but a total one: ctest finds zero tests, including the only one
# that proves every bind(C) symbol resolves against librocsparse.
if(NOT TARGET hip::hip_fortran)
  find_package(hip-fortran QUIET)
endif()
set(_have_hip_binding TRUE)
if(NOT TARGET hip::hip_fortran)
  set(_have_hip_binding FALSE)
  message(STATUS
    "rocSPARSE: no hip::hip_fortran target, skipping the Fortran binding tests "
    "that use it (65 of 66; the symbol test does not). Point CMAKE_PREFIX_PATH "
    "at a ROCm whose HIP Fortran bindings were built with "
    "${CMAKE_Fortran_COMPILER}.")
endif()

# librocsparse itself, needed to link but not to compile. In-tree it arrives
# through the rocsparse_fortran link interface; standalone it has to be found.
#
# rocsparse-config.cmake does find_dependency(hip), and hip-config.cmake reaches
# FindThreads, which hard-errors unless C or CXX is enabled. The bindings are
# Fortran-only and standalone this project declares LANGUAGES NONE, so C is
# enabled here purely to satisfy that dependency: it compiles none of the
# sources below. Without it, a standalone test build stops at "FindThreads only
# works if either C or CXX language is enabled", which says nothing about what
# is actually wrong.
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

# Build <name>.f03 into one executable and register it with CTest as
# rocsparse_fortran_<name>, labelled gpu. Expects the source to sit in this
# directory and to link rocsparse_fortran, hip::hip_fortran and roc::rocsparse.
function(rocsparse_fortran_add_test name)
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/${name}.f03")
  if(NOT EXISTS "${src}")
    message(FATAL_ERROR "Missing test source: ${src}")
  endif()
  # Force language detection for .f03 in toolchains where CMake misses it.
  set_source_files_properties("${src}" PROPERTIES LANGUAGE Fortran)
  add_executable(rocsparse_fortran_test_${name} "${src}")
  set_target_properties(rocsparse_fortran_test_${name} PROPERTIES LINKER_LANGUAGE Fortran)
  target_link_libraries(rocsparse_fortran_test_${name} PRIVATE
    rocsparse_fortran hip::hip_fortran roc::rocsparse)
  # The tests call EXIT, a GNU extension in gfortran.
  target_compile_options(rocsparse_fortran_test_${name} PRIVATE
    $<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-std=gnu>)
  add_test(NAME rocsparse_fortran_${name} COMMAND rocsparse_fortran_test_${name})
  # Every one of these creates a generator and draws numbers on a device, so it
  # fails without a GPU, which is not the same thing as the binding being
  # broken. The label lets a build machine with no GPU still run the suite
  # meaningfully: ctest -LE gpu.
  set_tests_properties(rocsparse_fortran_${name} PROPERTIES LABELS "gpu")
endfunction()

# One per routine family and precision, carried over from hipFORT's rocSPARSE
# suite: the SpMV/SpMM kernels, the conversion routines, the incomplete
# factorizations and the triangular solves, each across the s/d/c/z types.
set(ROCSPARSE_FORTRAN_TESTS
    ddoti
    rocsparse_cbsrmv
    rocsparse_ccsr2csc
#   rocsparse_ccsrgeam
#   rocsparse_ccsrgemm
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
#   rocsparse_dcsrgeam
#   rocsparse_dcsrgemm
    rocsparse_dcsric0
    rocsparse_dcsrilu0
    rocsparse_dgebsrmv
    rocsparse_dgemvi
    rocsparse_dgthr
    rocsparse_dsctr
    rocsparse_dsddmm
    rocsparse_dspmm
    rocsparse_dspmv
    rocsparse_dsptrsm
    rocsparse_dsptrsv
    rocsparse_sbsrmv
    rocsparse_scsr2csc
#   rocsparse_scsrgeam
#   rocsparse_scsrgemm
    rocsparse_scsric0
    rocsparse_scsrilu0
    rocsparse_sgebsrmv
    rocsparse_sgemvi
    rocsparse_sgthr
    rocsparse_spildlt0
    rocsparse_ssctr
    rocsparse_ssddmm
    rocsparse_sspmm
    rocsparse_sspmv
    rocsparse_ssptrsm
    rocsparse_ssptrsv
    rocsparse_zbsrmv
    rocsparse_zcsr2csc
#   rocsparse_zcsrgeam
#   rocsparse_zcsrgemm
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

# The csrgeam and csrgemm tests are commented out above, all four precisions of
# each, because they segfault. The cause is in the tests, not in the binding, and
# is worth stating so nobody re-adds them unchanged.
#
# Both routine families select what they compute from whether alpha and beta are
# NULL POINTERS, not from their values. rocsparse_csrgemm.h is explicit about it:
# "csr_val_C, csr_row_ptr_C, or csr_col_ind_C are invalid if beta is valid".
# These tests pass beta = 0.0_c_double: a perfectly valid pointer, while
# passing c_null_ptr for descr_D and every D array. rocSPARSE is therefore told
# that a D term exists, goes looking for it, and dereferences null. That is
# exactly why the failures are these two families and no others.
#
# The binding can express the NULL: the module carries a _dptr specific taking
# type(c_ptr),value :: alpha,beta alongside the real(c_double) one, which is also
# why these tests never compiled against the old hand-written binding, where
# alpha and beta were type(c_ptr) throughout. Fixing them means moving the whole
# call onto the c_ptr specific consistently, not just swapping the two scalars;
# a partial switch fails to resolve the generic. Left for a follow-up rather than
# guessed at here.

if(_have_hip_binding)
  foreach(_t IN LISTS ROCSPARSE_FORTRAN_TESTS)
    rocsparse_fortran_add_test(${_t})
  endforeach()
endif()

# Exhaustive "symbol" test, generated by the rocm-fortran generator alongside the
# binding. It references EVERY bind(C) routine of the module, guarded so nothing
# executes: the point is purely to prove each interface compiles and each C
# symbol links. Because it references every symbol, it turns a symbol missing
# from the installed rocSPARSE into a link error, which is useful to audit a ROCm
# version but too strict for the default test build, hence the separate option.
# The name carries hipFORT's HIPFORT_EXTENDED_TESTS forward, because this is the
# same test under a new home and anyone who knew the old flag looks for it.
option(BUILD_FORTRAN_EXTENDED_TESTS
       "Build the exhaustive rocSPARSE Fortran symbol test (needs a complete ROCm)" OFF)

if(BUILD_FORTRAN_EXTENDED_TESTS)
  set(_symsrc "${CMAKE_CURRENT_SOURCE_DIR}/test_rocsparse.F03")
  set_source_files_properties("${_symsrc}" PROPERTIES LANGUAGE Fortran)
  add_executable(rocsparse_fortran_test_extended "${_symsrc}")
  set_target_properties(rocsparse_fortran_test_extended PROPERTIES
    LINKER_LANGUAGE Fortran
    # .F03 is not one of the extensions CMake maps to "preprocess me", so the
    # uppercase F is honoured explicitly rather than left to the extension.
    Fortran_PREPROCESS ON)
  # No hip::hip_fortran here: unlike the runtime tests, the extended test touches
  # nothing but the rocsparse module, because it never actually allocates.
  target_link_libraries(rocsparse_fortran_test_extended PRIVATE
    rocsparse_fortran roc::rocsparse)
  target_compile_options(rocsparse_fortran_test_extended PRIVATE
    $<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-std=gnu>)
  add_test(NAME rocsparse_fortran_extended COMMAND rocsparse_fortran_test_extended)
  # Nothing in it executes, so unlike the tests above it does not need a GPU:
  # the whole point is what it proves at compile and link time.
  set_tests_properties(rocsparse_fortran_extended PROPERTIES LABELS "symbols")
endif()
