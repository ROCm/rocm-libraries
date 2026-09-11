# Tests for the rocSOLVER Fortran bindings.
#
# One executable per test, driven by CTest. The runtime tests factor or
# diagonalize a small matrix on a device, so they build anywhere but only pass on
# a machine with a GPU.
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
# `use rocsolver` and `use rocblas`. That makes the HIP Fortran binding a
# COMPILE-time dependency of those tests: its .mod must exist, where librocsolver
# is only a link-time one. In a monorepo build hip::hip_fortran comes from
# projects/hip/fortran; a standalone build has to find it in an installed ROCm,
# built with this same compiler.
#
# The exhaustive symbol test is the exception, and the exception matters: 82 of
# the 83 sources here use hip, test_rocsolver.F03 uses only iso_c_binding,
# rocsolver and rocblas. So it is registered BELOW, outside this guard. Returning
# early here would take it down with the rest, and ROCm does not currently ship a
# HIP Fortran binding at all, so against a stock /opt/rocm that early return is
# not a partial skip but a total one: ctest finds zero tests, including the only
# one that proves every bind(C) symbol resolves against librocsolver.
if(NOT TARGET hip::hip_fortran)
  find_package(hip-fortran QUIET)
endif()
set(_have_hip_binding TRUE)
if(NOT TARGET hip::hip_fortran)
  set(_have_hip_binding FALSE)
  message(STATUS
    "rocSOLVER: no hip::hip_fortran target, skipping the Fortran binding tests "
    "that use it (82 of 83; the symbol test does not). Point CMAKE_PREFIX_PATH "
    "at a ROCm whose HIP Fortran bindings were built with "
    "${CMAKE_Fortran_COMPILER}.")
endif()

# librocsolver itself, needed to link but not to compile. In-tree it arrives
# through the rocsolver_fortran link interface; standalone it has to be found.
#
# rocsolver-config.cmake does find_dependency(hip), and hip-config.cmake reaches
# FindThreads, which hard-errors unless C or CXX is enabled. The bindings are
# Fortran-only and standalone this project declares LANGUAGES NONE, so C is
# enabled here purely to satisfy that dependency: it compiles none of the
# sources below. Without it, a standalone test build stops at "FindThreads only
# works if either C or CXX language is enabled", which says nothing about what
# is actually wrong.
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

# Build <name>.f03 into one executable and register it with CTest as
# rocsolver_fortran_<name>, labelled gpu. Expects the source to sit in this
# directory and to link rocsolver_fortran, hip::hip_fortran and roc::rocsolver.
#
# No explicit rocblas_fortran here: rocsolver_fortran carries it PUBLIC, so the
# rocblas.mod the tests need for `use rocblas` comes with it.
function(rocsolver_fortran_add_test name)
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/${name}.f03")
  if(NOT EXISTS "${src}")
    message(FATAL_ERROR "Missing test source: ${src}")
  endif()
  # Force language detection for .f03 in toolchains where CMake misses it.
  set_source_files_properties("${src}" PROPERTIES LANGUAGE Fortran)
  add_executable(rocsolver_fortran_test_${name} "${src}")
  set_target_properties(rocsolver_fortran_test_${name} PROPERTIES LINKER_LANGUAGE Fortran)
  target_link_libraries(rocsolver_fortran_test_${name} PRIVATE
    rocsolver_fortran hip::hip_fortran roc::rocsolver)
  # The tests call EXIT, a GNU extension in gfortran.
  target_compile_options(rocsolver_fortran_test_${name} PRIVATE
    $<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-std=gnu>)
  add_test(NAME rocsolver_fortran_${name} COMMAND rocsolver_fortran_test_${name})
  # Every one of these factors or diagonalizes a matrix on a device, so it fails
  # without a GPU, which is not the same thing as the binding being broken. The
  # label lets a build machine with no GPU still run the suite meaningfully:
  # ctest -LE gpu.
  set_tests_properties(rocsolver_fortran_${name} PROPERTIES LABELS "gpu")
endfunction()

# One per routine family and precision, carried over from hipFORT's rocSOLVER
# suite: the factorizations (getrf, geqrf, potrf, sytrf, gebrd), the solvers
# (getrs, potrs, sytrs, gels, gesv, posv, getri, trtri), the orthogonal factor
# routines (orgqr/ungqr, ormqr/unmqr, larft, latrd), the symmetric and Hermitian
# eigensolvers (syev, syevd/heevd, syevdx/heevdx, syevj/heevj, sygvd/hegvd,
# sygvdx/hegvdx, sytrd, sterf) and the SVDs (gesvd, gesvdj), each across the
# s/d/c/z types the API defines them for.
set(ROCSOLVER_FORTRAN_TESTS
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

# Exhaustive "symbol" test, generated by the rocm-fortran generator alongside the
# binding. It references EVERY bind(C) routine of the module, guarded so nothing
# executes: the point is purely to prove each interface compiles and each C
# symbol links. Because it references every symbol, it turns a symbol missing
# from the installed rocSOLVER into a link error, which is useful to audit a ROCm
# version but too strict for the default test build, hence the separate option.
# The name carries hipFORT's HIPFORT_EXTENDED_TESTS forward, because this is the
# same test under a new home and anyone who knew the old flag looks for it.
option(BUILD_FORTRAN_EXTENDED_TESTS
       "Build the exhaustive rocSOLVER Fortran symbol test (needs a complete ROCm)" OFF)

if(BUILD_FORTRAN_EXTENDED_TESTS)
  set(_symsrc "${CMAKE_CURRENT_SOURCE_DIR}/test_rocsolver.F03")
  set_source_files_properties("${_symsrc}" PROPERTIES LANGUAGE Fortran)
  add_executable(rocsolver_fortran_test_extended "${_symsrc}")
  set_target_properties(rocsolver_fortran_test_extended PROPERTIES
    LINKER_LANGUAGE Fortran
    # .F03 is not one of the extensions CMake maps to "preprocess me", so the
    # uppercase F is honoured explicitly rather than left to the extension.
    Fortran_PREPROCESS ON)
  # No hip::hip_fortran here: unlike the runtime tests, the extended test touches
  # nothing but the rocsolver and rocblas modules, because it never actually
  # allocates.
  target_link_libraries(rocsolver_fortran_test_extended PRIVATE
    rocsolver_fortran roc::rocsolver)
  target_compile_options(rocsolver_fortran_test_extended PRIVATE
    $<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-std=gnu>)
  add_test(NAME rocsolver_fortran_extended COMMAND rocsolver_fortran_test_extended)
  # Nothing in it executes, so unlike the tests above it does not need a GPU:
  # the whole point is what it proves at compile and link time.
  set_tests_properties(rocsolver_fortran_extended PROPERTIES LABELS "symbols")
endif()
