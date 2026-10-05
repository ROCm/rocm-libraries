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
# #############################################################################

# rocFFT Fortran binding tests: one GPU executable per test.

# ifx/ifort skip .f03 files unless -Tf precedes the source.
if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  string(REPLACE "<SOURCE>" "-Tf <SOURCE>"
         CMAKE_Fortran_COMPILE_OBJECT "${CMAKE_Fortran_COMPILE_OBJECT}")
endif()

# hip-config.cmake reaches FindThreads, which needs C or CXX enabled.
if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
  enable_language(C)
endif()

# The tests `use hip`, so hip.mod (rocm-systems) is a compile-time dependency.
if(NOT TARGET hip::hip_fortran)
  find_package(hip-fortran QUIET)
endif()
set(_have_hip_binding TRUE)
if(NOT TARGET hip::hip_fortran)
  set(_have_hip_binding FALSE)
  message(STATUS
    "rocFFT: no hip::hip_fortran target, skipping the Fortran binding tests "
    "(they `use hip`). Point CMAKE_PREFIX_PATH at a ROCm whose HIP Fortran "
    "bindings were built with this same compiler to build them.")
endif()

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
  set_source_files_properties("${src}" PROPERTIES LANGUAGE Fortran)
  add_executable(rocfft_fortran_test_${name} "${src}")
  set_target_properties(rocfft_fortran_test_${name} PROPERTIES LINKER_LANGUAGE Fortran)
  target_link_libraries(rocfft_fortran_test_${name} PRIVATE
    rocfft_fortran hip::hip_fortran roc::rocfft hip::host)
  # EXIT is a GNU extension.
  target_compile_options(rocfft_fortran_test_${name} PRIVATE
    $<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-std=gnu>)
  add_test(NAME rocfft_fortran_${name} COMMAND rocfft_fortran_test_${name})
  # ctest -LE gpu skips them on a GPU-less machine.
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
  # Exits 77 when fewer than two GPUs are present.
  set_tests_properties(rocfft_fortran_rocfft_field_brick_z PROPERTIES
                       SKIP_RETURN_CODE 77)
endif()
