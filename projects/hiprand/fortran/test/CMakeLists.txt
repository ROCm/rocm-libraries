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

# hipRAND Fortran binding tests: the .f03 programs and the FRUIT suite.

# ifx/ifort skip .f03 files unless -Tf precedes the source.
if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  string(REPLACE "<SOURCE>" "-Tf <SOURCE>"
         CMAKE_Fortran_COMPILE_OBJECT "${CMAKE_Fortran_COMPILE_OBJECT}")
endif()

# Every test but the version one `use hip` (hip.mod: rocm-systems, same compiler).
if(NOT TARGET hip::hip_fortran)
  find_package(hip-fortran QUIET)
endif()
set(_have_hip_binding TRUE)
if(NOT TARGET hip::hip_fortran)
  set(_have_hip_binding FALSE)
  message(STATUS
    "hipRAND: no hip::hip_fortran target, skipping all but the version test "
    "(the others `use hip`). Point CMAKE_PREFIX_PATH at a ROCm whose HIP Fortran "
    "bindings were built with ${CMAKE_Fortran_COMPILER}.")
endif()

# hip-config.cmake reaches FindThreads, which needs C or CXX enabled.
if(NOT TARGET hip::hiprand)
  if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
    enable_language(C)
  endif()
  find_package(hiprand QUIET)
endif()
if(NOT TARGET hip::hiprand)
  message(STATUS
    "hipRAND: no hip::hiprand target, skipping the Fortran binding tests "
    "(they link libhiprand). Point CMAKE_PREFIX_PATH at an installed ROCm to "
    "build them standalone.")
  return()
endif()

# hiprand_fortran_add_test(<name> [sources...]); default source <name>.f03.
function(hiprand_fortran_add_test name)
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
  add_executable(hiprand_fortran_test_${name} ${abs_srcs})
  # The FRUIT sources write .mod files; one directory per test avoids races.
  set_target_properties(hiprand_fortran_test_${name} PROPERTIES
    LINKER_LANGUAGE Fortran
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/modules/${name}")
  target_link_libraries(hiprand_fortran_test_${name} PRIVATE
    hiprand_fortran hip::hiprand)
  # EXIT is a GNU extension.
  target_compile_options(hiprand_fortran_test_${name} PRIVATE
    $<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-std=gnu>)
  add_test(NAME hiprand_fortran_${name} COMMAND hiprand_fortran_test_${name})
  # ctest -LE gpu skips them on a GPU-less machine.
  set_tests_properties(hiprand_fortran_${name} PROPERTIES LABELS "gpu")
endfunction()

# Log-normal and Poisson do not vary with the generator: philox only.
set(HIPRAND_FORTRAN_TESTS
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

hiprand_fortran_add_test(hiprand_version)
# The version query touches no device.
set_tests_properties(hiprand_fortran_hiprand_version PROPERTIES LABELS "")
# In-tree the module must carry this tree's version (hiprandGetVersion
# reports rocRAND's on ROCm, so the runtime value cannot serve).
get_target_property(_hiprand_aliased hip::hiprand ALIASED_TARGET)
if(_hiprand_aliased STREQUAL "hiprand"
   AND hiprand_VERSION MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)")
  math(EXPR _hiprand_code
       "${CMAKE_MATCH_1} * 100000 + ${CMAKE_MATCH_2} * 100 + ${CMAKE_MATCH_3}")
  set_tests_properties(hiprand_fortran_hiprand_version PROPERTIES
                       ENVIRONMENT "HIPRAND_FORTRAN_EXPECTED_VERSION=${_hiprand_code}")
endif()

if(_have_hip_binding)
  foreach(_t IN LISTS HIPRAND_FORTRAN_TESTS)
    hiprand_fortran_add_test(${_t})
    target_link_libraries(hiprand_fortran_test_${_t} PRIVATE hip::hip_fortran)
  endforeach()

  # One CTest test: FRUIT reports its own per-case results; it also reaches the
  # entry points the programs above do not (Generate, GenerateSeeds, ...).
  hiprand_fortran_add_test(fruit
      fruit/fruit_util.f90
      fruit/fruit.f90
      fruit_helpers.f90
      test_hiprand.f90
      fruit_test_driver_hiprand.f90)
  target_link_libraries(hiprand_fortran_test_fruit PRIVATE hip::hip_fortran)
endif()
