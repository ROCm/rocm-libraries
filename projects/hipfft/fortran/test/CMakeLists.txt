# #############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ############################################################################

# hipFFT and hipFFTW Fortran binding tests: one GPU executable per test. Each
# half guards itself on its own binding target and C library.

# ifx/ifort skip .f03/.f08 files unless -Tf precedes the source.
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

# hip-config-amd.cmake reaches FindThreads, which needs C or CXX enabled.
if(NOT TARGET hip::hipfft AND NOT TARGET hip::hipfftw)
  if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
    enable_language(C)
  endif()
  find_package(hipfft QUIET)
endif()

# The tests `use hip`, so hip.mod (rocm-systems, same compiler) is required.
if(NOT TARGET hip::hip_fortran)
  find_package(hip-fortran QUIET)
endif()
if(NOT TARGET hip::hip_fortran)
  message(STATUS
    "hipFFT: no hip::hip_fortran target, skipping the Fortran runtime tests "
    "of both bindings (they `use hip`). Install the HIP Fortran bindings for "
    "this compiler to build them.")
endif()

if(NOT _hipfft_tests)
  # Deliberately silent: the parent was told not to build this half.
elseif(NOT TARGET hipfft_fortran)
  message(STATUS
    "hipFFT: the Fortran binding target was not built, skipping its tests.")
elseif(NOT TARGET hip::hipfft)
  message(STATUS
    "hipFFT: no hip::hipfft target, skipping the hipFFT Fortran binding tests "
    "(they link libhipfft). Point CMAKE_PREFIX_PATH at an installed ROCm to "
    "build them standalone.")
else()
  # hipfft_fortran_add_test(<name> [F2008]): F2008 builds <name>.f08, which
  # exercises the array overloads, as test hipfft_fortran_<name>_f2008.
  function(hipfft_fortran_add_test name)
    if("F2008" IN_LIST ARGN)
      set(src "${CMAKE_CURRENT_SOURCE_DIR}/${name}.f08")
      set(test_name "${name}_f2008")
    else()
      set(src "${CMAKE_CURRENT_SOURCE_DIR}/${name}.f03")
      set(test_name "${name}")
    endif()
    if(NOT EXISTS "${src}")
      message(FATAL_ERROR "Missing test source: ${src}")
    endif()
    set_source_files_properties("${src}" PROPERTIES LANGUAGE Fortran)
    add_executable(hipfft_fortran_test_${test_name} "${src}")
    set_target_properties(hipfft_fortran_test_${test_name} PROPERTIES LINKER_LANGUAGE Fortran)
    target_link_libraries(hipfft_fortran_test_${test_name}
                          PRIVATE hipfft_fortran hip::hip_fortran hip::hipfft)
    # EXIT is a GNU extension.
    target_compile_options(hipfft_fortran_test_${test_name} PRIVATE
      $<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-std=gnu>)
    add_test(NAME hipfft_fortran_${test_name} COMMAND hipfft_fortran_test_${test_name})
    # ctest -LE gpu skips them on a GPU-less machine.
    set_tests_properties(hipfft_fortran_${test_name} PROPERTIES LABELS "gpu")
  endfunction()

  if(TARGET hip::hip_fortran)
    set(HIPFFT_FORTRAN_TESTS
        hipfft
        hipfft_c2c_1d_batched_z
        hipfft_c2c_1d_c
        hipfft_c2c_1d_z
        hipfft_c2c_2d_z
        hipfft_c2c_3d_z
        hipfft_estimate_getsize_d
        hipfft_makeplanmany_z
        hipfft_planmany_2d_z2z
        hipfft_r2c_c2r_1d_d
        hipfft_r2c_c2r_1d_s
        hipfft_setstream_z
        hipfft_version
        hipfft_xt_descriptor_z
        hipfft_xt_makeplanmany_c)

    foreach(_t IN LISTS HIPFFT_FORTRAN_TESTS)
      hipfft_fortran_add_test(${_t})
    endforeach()
    # Exits 77 when fewer than two GPUs are present.
    set_tests_properties(hipfft_fortran_hipfft_xt_descriptor_z PROPERTIES
                         SKIP_RETURN_CODE 77)

    # The .f08 variants use the array overloads, present in both the
    # assumed-shape and the assumed-rank tier.
    # The Xt tests have none: the Xt API passes descriptors, not arrays.
    if(NOT FORTRAN_ARRAY_INTERFACES STREQUAL "none")
      set(HIPFFT_FORTRAN_F2008_TESTS ${HIPFFT_FORTRAN_TESTS})
      list(FILTER HIPFFT_FORTRAN_F2008_TESTS EXCLUDE REGEX "^hipfft_xt_")
      foreach(_t IN LISTS HIPFFT_FORTRAN_F2008_TESTS)
        hipfft_fortran_add_test(${_t} F2008)
      endforeach()
    endif()
  endif()
endif()

if(NOT _hipfftw_tests)
  # Deliberately silent: the parent was told not to build this half.
elseif(NOT TARGET hipfftw_fortran)
  message(STATUS
    "hipFFTW: the Fortran binding target was not built, skipping its tests.")
elseif(NOT TARGET hip::hipfftw)
  message(STATUS
    "hipFFTW: no hip::hipfftw target, skipping the hipFFTW Fortran binding "
    "tests (they link libhipfftw). Point CMAKE_PREFIX_PATH at an installed "
    "ROCm to build them standalone.")
else()
  function(hipfftw_fortran_add_test name)
    set(src "${CMAKE_CURRENT_SOURCE_DIR}/${name}.f03")
    if(NOT EXISTS "${src}")
      message(FATAL_ERROR "Missing test source: ${src}")
    endif()
    set_source_files_properties("${src}" PROPERTIES LANGUAGE Fortran)
    add_executable(hipfftw_fortran_test_${name} "${src}")
    set_target_properties(hipfftw_fortran_test_${name} PROPERTIES
                          LINKER_LANGUAGE Fortran)
    target_link_libraries(hipfftw_fortran_test_${name} PRIVATE
                          hipfftw_fortran hip::hipfftw hip::hip_fortran)
    # EXIT is a GNU extension.
    target_compile_options(hipfftw_fortran_test_${name} PRIVATE
      $<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-std=gnu>)
    add_test(NAME hipfftw_fortran_${name} COMMAND hipfftw_fortran_test_${name})
    set_tests_properties(hipfftw_fortran_${name} PROPERTIES LABELS "gpu")
  endfunction()

  if(TARGET hip::hip_fortran)
    set(HIPFFTW_FORTRAN_TESTS
        hipfftw_alloc
        hipfftw_c2c
        hipfftw_dft_2d
        hipfftw_dft_3d
        hipfftw_guru
        hipfftw_guru64
        hipfftw_many
        hipfftw_r2c_c2r)

    foreach(_t IN LISTS HIPFFTW_FORTRAN_TESTS)
      hipfftw_fortran_add_test(${_t})
    endforeach()
  endif()
endif()
