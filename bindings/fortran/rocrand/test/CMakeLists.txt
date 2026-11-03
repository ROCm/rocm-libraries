# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# rocRAND Fortran binding tests: the .f03 programs, their .f08 variants (array
# overloads) and the FRUIT suite.

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

# Every test but the version one `use hip` (hip.mod: rocm-systems, same compiler).
if(NOT TARGET hip::hip_fortran)
  find_package(hip-fortran QUIET)
endif()
set(_have_hip_binding TRUE)
if(NOT TARGET hip::hip_fortran)
  set(_have_hip_binding FALSE)
  message(STATUS
    "rocRAND: no hip::hip_fortran target, skipping all but the version test "
    "(the others `use hip`). Point CMAKE_PREFIX_PATH at a ROCm whose HIP Fortran "
    "bindings were built with ${CMAKE_Fortran_COMPILER}.")
endif()

# hip-config.cmake reaches FindThreads, which needs C or CXX enabled.
if(NOT TARGET roc::rocrand)
  if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
    enable_language(C)
  endif()
  find_package(rocrand QUIET)
endif()
if(NOT TARGET roc::rocrand)
  message(STATUS
    "rocRAND: no roc::rocrand target, skipping the Fortran binding tests "
    "(they link librocrand). Point CMAKE_PREFIX_PATH at an installed ROCm to "
    "build them standalone.")
  return()
endif()

# rocrand_fortran_add_test(<name> [sources...]); default source <name>.f03.
function(rocrand_fortran_add_test name)
  set(srcs ${ARGN})
  if(NOT srcs)
    set(srcs "${name}.f03")
  endif()
  set(abs_srcs "")
  foreach(src IN LISTS srcs)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
      message(FATAL_ERROR "Missing test source: ${CMAKE_CURRENT_SOURCE_DIR}/${src}")
    endif()
    list(APPEND abs_srcs "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
  endforeach()
  set_source_files_properties(${abs_srcs} PROPERTIES LANGUAGE Fortran)
  add_executable(rocrand_fortran_test_${name} ${abs_srcs})
  set_target_properties(rocrand_fortran_test_${name} PROPERTIES
    LINKER_LANGUAGE Fortran
    # The FRUIT sources write .mod files; one directory per test avoids races.
    Fortran_MODULE_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/rocrand_fortran_test_${name}.mod.dir")
  target_link_libraries(rocrand_fortran_test_${name} PRIVATE
    rocrand_fortran roc::rocrand)
  # EXIT is a GNU extension.
  target_compile_options(rocrand_fortran_test_${name} PRIVATE
    $<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-std=gnu>)
  add_test(NAME rocrand_fortran_${name} COMMAND rocrand_fortran_test_${name})
  # ctest -LE gpu skips them on a GPU-less machine.
  set_tests_properties(rocrand_fortran_${name} PROPERTIES LABELS "gpu")
endfunction()

# Log-normal and Poisson do not vary with the generator: philox only.
set(ROCRAND_FORTRAN_TESTS
    mrg32k3a_normal
    mrg32k3a_normal_double
    mrg32k3a_uniform
    mrg32k3a_uniform_double
    mtgp32_normal
    mtgp32_normal_double
    mtgp32_uniform
    mtgp32_uniform_double
    philox_log_normal
    philox_log_normal_double
    philox_normal
    philox_normal_double
    philox_poisson
    philox_uniform
    philox_uniform_double
    sobol32_normal
    sobol32_normal_double
    sobol32_uniform
    sobol32_uniform_double
    xorwow_normal
    xorwow_normal_double
    xorwow_uniform
    xorwow_uniform_double)

# The .f08 variants use the array overloads, present in both the assumed-shape
# and the assumed-rank tier; they are registered as <name>_f2008.
set(_rocrand_version_tests rocrand_version)
rocrand_fortran_add_test(rocrand_version)
if(NOT FORTRAN_ARRAY_INTERFACES STREQUAL "none")
  rocrand_fortran_add_test(rocrand_version_f2008 rocrand_version.f08)
  list(APPEND _rocrand_version_tests rocrand_version_f2008)
endif()
# In-tree the module and the library come from the same headers.
get_target_property(_rocrand_aliased roc::rocrand ALIASED_TARGET)
foreach(_t IN LISTS _rocrand_version_tests)
  # The version query touches no device.
  set_tests_properties(rocrand_fortran_${_t} PROPERTIES LABELS "")
  if(_rocrand_aliased STREQUAL "rocrand")
    set_tests_properties(rocrand_fortran_${_t} PROPERTIES
                         ENVIRONMENT "ROCRAND_FORTRAN_EXACT_VERSION=1")
  endif()
endforeach()

if(_have_hip_binding)
  foreach(_t IN LISTS ROCRAND_FORTRAN_TESTS)
    rocrand_fortran_add_test(${_t})
    target_link_libraries(rocrand_fortran_test_${_t} PRIVATE hip::hip_fortran)
  endforeach()
  if(NOT FORTRAN_ARRAY_INTERFACES STREQUAL "none")
    foreach(_t IN LISTS ROCRAND_FORTRAN_TESTS)
      rocrand_fortran_add_test(${_t}_f2008 ${_t}.f08)
      target_link_libraries(rocrand_fortran_test_${_t}_f2008 PRIVATE
                            hip::hip_fortran)
    endforeach()
  endif()

  # One CTest test: FRUIT reports its own per-case results.
  rocrand_fortran_add_test(fruit
      fruit/fruit_util.f90
      fruit/fruit.f90
      fruit_helpers.f90
      test_rocrand.f90
      fruit_test_driver_rocrand.f90)
  target_link_libraries(rocrand_fortran_test_fruit PRIVATE hip::hip_fortran)
endif()
