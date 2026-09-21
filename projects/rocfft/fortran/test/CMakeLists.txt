# Tests for the rocFFT Fortran bindings.
#
# One executable per test, driven by CTest. These are runtime tests: they call
# into rocFFT and the HIP runtime and need a GPU, so they build anywhere but only
# pass on a machine with one.
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

# Everything looked for below -- hip-fortran, rocfft, hip -- lands in
# hip-config.cmake, which reaches FindThreads, which hard-errors unless C or CXX
# is enabled. The bindings are Fortran-only and standalone this project declares
# LANGUAGES NONE, so C is enabled here purely to satisfy that: it compiles none
# of the sources below. Without it, a standalone test build stops at
# "FindThreads only works if either C or CXX language is enabled", which says
# nothing about what is actually wrong.
if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
  enable_language(C)
endif()

# Every RUNTIME test allocates device memory and copies buffers, so each one
# writes `use hip` next to its `use rocfft`. That makes the HIP BINDING (not just
# the HIP runtime) a compile-time dependency of those tests: without hip.mod they
# cannot even be parsed. It is built by rocm-systems' projects/hip and, like this
# one, only exists when that repo was configured with a Fortran compiler, hence a
# skip rather than an error.
#
# The exhaustive symbol test is the exception, and the exception matters: 18 of
# the 19 sources here use hip, test_rocfft.F03 uses only iso_c_binding and
# rocfft. So it is registered BELOW, outside this guard. Returning early here
# would take it down with the rest, and ROCm does not currently ship a HIP
# Fortran binding at all, so against a stock /opt/rocm that early return is not a
# partial skip but a total one: ctest finds zero tests, including the only one
# that proves every bind(C) symbol resolves against librocfft.
if(NOT TARGET hip::hip_fortran)
  find_package(hip-fortran QUIET)
endif()
set(_have_hip_binding TRUE)
if(NOT TARGET hip::hip_fortran)
  set(_have_hip_binding FALSE)
  message(STATUS
    "rocFFT: no hip::hip_fortran target, skipping the Fortran binding tests "
    "that use it (18 of 19; the symbol test does not). Point CMAKE_PREFIX_PATH "
    "at a ROCm whose HIP Fortran bindings were built with this same compiler to "
    "build them.")
endif()

# librocfft and the HIP runtime are link-time dependencies of every test. In-tree
# roc::rocfft comes from rocFFT's own library/src; standalone both have to be
# found.
if(NOT TARGET roc::rocfft)
  find_package(rocfft QUIET)
endif()
if(NOT TARGET hip::host)
  find_package(hip QUIET)
endif()
if(NOT TARGET roc::rocfft OR NOT TARGET hip::host)
  message(STATUS
    "rocFFT: no roc::rocfft or hip::host target, skipping the Fortran binding "
    "tests (they link librocfft and the HIP runtime). Point CMAKE_PREFIX_PATH "
    "at an installed ROCm to build them standalone.")
  return()
endif()

function(rocfft_fortran_add_test name)
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/${name}.f03")
  if(NOT EXISTS "${src}")
    message(FATAL_ERROR "Missing test source: ${src}")
  endif()
  # Force language detection for .f03 in toolchains where CMake misses it.
  set_source_files_properties("${src}" PROPERTIES LANGUAGE Fortran)
  add_executable(rocfft_fortran_test_${name} "${src}")
  set_target_properties(rocfft_fortran_test_${name} PROPERTIES LINKER_LANGUAGE Fortran)
  target_link_libraries(rocfft_fortran_test_${name} PRIVATE
    rocfft_fortran hip::hip_fortran roc::rocfft hip::host)
  # The tests call EXIT, a GNU extension in gfortran.
  target_compile_options(rocfft_fortran_test_${name} PRIVATE
    $<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-std=gnu>)
  add_test(NAME rocfft_fortran_${name} COMMAND rocfft_fortran_test_${name})
  # Every one of these calls into rocFFT and the HIP runtime and fails without a
  # GPU, which is not the same thing as the binding being broken. The label lets
  # a build machine with no GPU still run the suite meaningfully: ctest -LE gpu.
  set_tests_properties(rocfft_fortran_${name} PROPERTIES LABELS "gpu")
endfunction()

set(ROCFFT_FORTRAN_TESTS
    rocfft
    rocfft_c2c_1d_batched_z
    rocfft_c2c_1d_c
    rocfft_c2c_1d_notinplace_z
    rocfft_c2c_1d_z
    rocfft_c2c_2d_z
    rocfft_c2c_3d_z
    rocfft_cache_z
    rocfft_callback_z
    rocfft_field_brick_z
    rocfft_plan_print_z
    rocfft_r2c_c2r_1d_d
    rocfft_r2c_c2r_1d_inplace_d
    rocfft_r2c_c2r_1d_s
    rocfft_scale_factor_z
    rocfft_stream_z
    rocfft_version
    rocfft_work_buffer_z)

if(_have_hip_binding)
  foreach(_t IN LISTS ROCFFT_FORTRAN_TESTS)
    rocfft_fortran_add_test(${_t})
  endforeach()
endif()

# Exhaustive "symbol" test, generated by the rocm-fortran generator alongside the
# binding. It references EVERY bind(C) routine of the module, guarded so nothing
# executes: the point is purely to prove each interface compiles and each C
# symbol links. Because it references every symbol, it turns a symbol missing
# from the installed rocFFT into a link error, which is useful to audit a ROCm
# version but too strict for the default test build -- hence the separate option.
# The name carries hipFORT's HIPFORT_EXTENDED_TESTS forward, because this is the
# same test under a new home and anyone who knew the old flag looks for it.
option(BUILD_FORTRAN_EXTENDED_TESTS
       "Build the exhaustive rocFFT Fortran symbol test (needs a complete ROCm)" OFF)

if(BUILD_FORTRAN_EXTENDED_TESTS)
  set(_symsrc "${CMAKE_CURRENT_SOURCE_DIR}/test_rocfft.F03")
  set_source_files_properties("${_symsrc}" PROPERTIES LANGUAGE Fortran)
  add_executable(rocfft_fortran_test_extended "${_symsrc}")
  set_target_properties(rocfft_fortran_test_extended PROPERTIES
    LINKER_LANGUAGE Fortran
    # Uppercase .F03, and the generator emits the same preprocessor guards here
    # as in the module, so this one source does need the C preprocessor.
    Fortran_PREPROCESS ON)
  # No hip::hip_fortran here, unlike the runtime tests above: the generated
  # symbol test is self-contained and does not `use hip`.
  target_link_libraries(rocfft_fortran_test_extended PRIVATE
    rocfft_fortran roc::rocfft hip::host)
  target_compile_options(rocfft_fortran_test_extended PRIVATE
    $<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-std=gnu>)
  add_test(NAME rocfft_fortran_extended COMMAND rocfft_fortran_test_extended)
  # Nothing in it executes, so unlike the tests above it does not need a GPU:
  # the whole point is what it proves at compile and link time.
  set_tests_properties(rocfft_fortran_extended PROPERTIES LABELS "symbols")
endif()
