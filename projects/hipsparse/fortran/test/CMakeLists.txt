# Tests for the hipSPARSE Fortran bindings.
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
# `use hipsparse`. That makes the HIP Fortran binding a COMPILE-time dependency of
# those tests: its .mod must exist, where libhipsparse is only a link-time one.
# In a monorepo build hip::hip_fortran comes from projects/hip/fortran; a
# standalone build has to find it in an installed ROCm, built with this same
# compiler.
#
# The exhaustive symbol test is the exception, and the exception matters: 65 of
# the 66 sources here use hip, test_hipsparse.F03 uses only iso_c_binding and
# hipsparse. So it is registered BELOW, outside this guard. Returning early here
# would take it down with the rest, and ROCm does not currently ship a HIP
# Fortran binding at all, so against a stock /opt/rocm that early return is not a
# partial skip but a total one: ctest finds zero tests, including the only one
# that proves every bind(C) symbol resolves against libhipsparse.
if(NOT TARGET hip::hip_fortran)
  find_package(hip-fortran QUIET)
endif()
set(_have_hip_binding TRUE)
if(NOT TARGET hip::hip_fortran)
  set(_have_hip_binding FALSE)
  message(STATUS
    "hipSPARSE: no hip::hip_fortran target, skipping the Fortran binding tests "
    "that use it (65 of 66; the symbol test does not). Point CMAKE_PREFIX_PATH "
    "at a ROCm whose HIP Fortran bindings were built with "
    "${CMAKE_Fortran_COMPILER}.")
endif()

# libhipsparse itself, needed to link but not to compile. In-tree it arrives
# through the hipsparse_fortran link interface; standalone it has to be found.
#
# hipsparse-config.cmake does find_dependency(hip), and hip-config.cmake reaches
# FindThreads, which hard-errors unless C or CXX is enabled. The bindings are
# Fortran-only and standalone this project declares LANGUAGES NONE, so C is
# enabled here purely to satisfy that dependency: it compiles none of the
# sources below. Without it, a standalone test build stops at "FindThreads only
# works if either C or CXX language is enabled", which says nothing about what
# is actually wrong.
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

# Build <name>.f03 into one executable and register it with CTest as
# hipsparse_fortran_<name>, labelled gpu. Expects the source to sit in this
# directory and to link hipsparse_fortran, hip::hip_fortran and roc::hipsparse.
function(hipsparse_fortran_add_test name)
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/${name}.f03")
  if(NOT EXISTS "${src}")
    message(FATAL_ERROR "Missing test source: ${src}")
  endif()
  # Force language detection for .f03 in toolchains where CMake misses it.
  set_source_files_properties("${src}" PROPERTIES LANGUAGE Fortran)
  add_executable(hipsparse_fortran_test_${name} "${src}")
  set_target_properties(hipsparse_fortran_test_${name} PROPERTIES LINKER_LANGUAGE Fortran)
  target_link_libraries(hipsparse_fortran_test_${name} PRIVATE
    hipsparse_fortran hip::hip_fortran roc::hipsparse)
  # The tests call EXIT, a GNU extension in gfortran.
  target_compile_options(hipsparse_fortran_test_${name} PRIVATE
    $<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-std=gnu>)
  add_test(NAME hipsparse_fortran_${name} COMMAND hipsparse_fortran_test_${name})
  # Every one of these creates a generator and draws numbers on a device, so it
  # fails without a GPU, which is not the same thing as the binding being
  # broken. The label lets a build machine with no GPU still run the suite
  # meaningfully: ctest -LE gpu.
  set_tests_properties(hipsparse_fortran_${name} PROPERTIES LABELS "gpu")
endfunction()

# One per routine family and precision, carried over from hipFORT's hipSPARSE
# suite: the SpMV/SpMM kernels, the conversion routines, the incomplete
# factorizations and the triangular solves, each across the s/d/c/z types.
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
    hipsparse_dgemvi
    hipsparse_dgthr
    hipsparse_dsctr
    hipsparse_dsddmm
    hipsparse_dspmm
    hipsparse_dspmv
    hipsparse_dsptrsm
    hipsparse_dsptrsv
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

# Exhaustive "symbol" test, generated by the rocm-fortran generator alongside the
# binding. It references EVERY bind(C) routine of the module, guarded so nothing
# executes: the point is purely to prove each interface compiles and each C
# symbol links. Because it references every symbol, it turns a symbol missing
# from the installed hipSPARSE into a link error, which is useful to audit a ROCm
# version but too strict for the default test build, hence the separate option.
# The name carries hipFORT's HIPFORT_EXTENDED_TESTS forward, because this is the
# same test under a new home and anyone who knew the old flag looks for it.
option(BUILD_FORTRAN_EXTENDED_TESTS
       "Build the exhaustive hipSPARSE Fortran symbol test (needs a complete ROCm)" OFF)

if(BUILD_FORTRAN_EXTENDED_TESTS)
  set(_symsrc "${CMAKE_CURRENT_SOURCE_DIR}/test_hipsparse.F03")
  set_source_files_properties("${_symsrc}" PROPERTIES LANGUAGE Fortran)
  add_executable(hipsparse_fortran_test_extended "${_symsrc}")
  set_target_properties(hipsparse_fortran_test_extended PROPERTIES
    LINKER_LANGUAGE Fortran
    # .F03 is not one of the extensions CMake maps to "preprocess me", so the
    # uppercase F is honoured explicitly rather than left to the extension.
    Fortran_PREPROCESS ON)
  # No hip::hip_fortran here: unlike the runtime tests, the extended test touches
  # nothing but the hipsparse module, because it never actually allocates.
  target_link_libraries(hipsparse_fortran_test_extended PRIVATE
    hipsparse_fortran roc::hipsparse)
  target_compile_options(hipsparse_fortran_test_extended PRIVATE
    $<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-std=gnu>)
  add_test(NAME hipsparse_fortran_extended COMMAND hipsparse_fortran_test_extended)
  # Nothing in it executes, so unlike the tests above it does not need a GPU:
  # the whole point is what it proves at compile and link time.
  set_tests_properties(hipsparse_fortran_extended PROPERTIES LABELS "symbols")
endif()
