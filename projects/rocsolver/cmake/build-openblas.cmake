# ##########################################################################
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
# ##########################################################################

# Builds a static, C-only OpenBLAS (which provides both BLAS and LAPACK) at
# configure/build time via ExternalProject_Add. Used as a fallback when no
# system LAPACK is found. Because OpenBLAS is built with C_LAPACK=ON and
# NOFORTRAN=ON, no Fortran compiler is required.
#
# On return this module defines an IMPORTED target `openblas_lapack` and sets
# LAPACK_LIBRARIES to it, so callers can link exactly as they would against a
# system LAPACK. Nothing is installed to the system.

include_guard(GLOBAL)

set(OPENBLAS_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/openblas-src)
set(OPENBLAS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/openblas-bin)

get_property(_openblas_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

# The static library location differs between single- and multi-config
# generators (multi-config OpenBLAS upper-cases the per-config subdirectory).
if(_openblas_is_multi_config)
  if(NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;MinSizeRel")
  endif()
  set(_openblas_byproducts)
  foreach(_cfg ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${_cfg} _cfg_upper)
    list(APPEND _openblas_byproducts
      ${OPENBLAS_BINARY_DIR}/lib/${_cfg_upper}/${CMAKE_STATIC_LIBRARY_PREFIX}openblas${CMAKE_STATIC_LIBRARY_SUFFIX})
  endforeach()
else()
  set(_openblas_byproducts
    ${OPENBLAS_BINARY_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}openblas${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

set(_openblas_cmake_args
  -DC_LAPACK=ON
  -DNOFORTRAN=ON
  -DBUILD_STATIC_LIBS=ON
  -DBUILD_LAPACK_DEPRECATED=OFF
  -DBUILD_TESTING=OFF
  -DBUILD_BENCHMARKS=OFF
  -DUSE_OPENMP=OFF
  -DUSE_THREAD=OFF
  -DCMAKE_C_VISIBILITY_PRESET=hidden
  # Force GENERIC target so the binary runs on any CPU.
  -DTARGET=GENERIC
)

# CMAKE_BUILD_TYPE is ignored by multi-config generators.
if(NOT _openblas_is_multi_config)
  list(APPEND _openblas_cmake_args -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
endif()

# Match position-independence so OpenBLAS can be linked into shared clients.
list(APPEND _openblas_cmake_args -DCMAKE_POSITION_INDEPENDENT_CODE=ON)

# Use the same compilers as the parent build. This is not just to avoid runtime
# mismatches -- on Windows CMake may otherwise default the sub-build to MSVC and
# link an MSVC-built OpenBLAS against amdclang-built rocSOLVER binaries, which is
# ABI-incompatible and fails at link/run time. Forwarding pins both to the same
# toolchain.
if(CMAKE_C_COMPILER)
  list(APPEND _openblas_cmake_args -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER})
endif()
if(CMAKE_CXX_COMPILER)
  list(APPEND _openblas_cmake_args -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
endif()

# OpenBLAS is third-party code we build but do not modify, and its f2c-translated
# LAPACK is very noisy (unused-variable/function warnings, and on recent Clang it
# even trips -Werror=incompatible-pointer-types). Suppress warnings for the
# sub-build using the spelling appropriate to the forwarded compiler.
if(CMAKE_C_COMPILER_ID MATCHES "MSVC")
  set(_openblas_c_flags "/w")
else()
  set(_openblas_c_flags "-w")
  if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    string(APPEND _openblas_c_flags " -Wno-error=incompatible-pointer-types")
  endif()
endif()
list(APPEND _openblas_cmake_args "-DCMAKE_C_FLAGS=${_openblas_c_flags}")

include(ExternalProject)
# OpenBLAS v0.3.33 (commit 62bcfb0)
ExternalProject_Add(rocsolver-static-lapack
  BUILD_BYPRODUCTS ${_openblas_byproducts}
  GIT_REPOSITORY   https://github.com/OpenMathLib/OpenBLAS
  GIT_TAG          62bcfb0dc9f1cfa685fc04135c50e2780c303137
  SOURCE_DIR       ${OPENBLAS_SOURCE_DIR}
  BINARY_DIR       ${OPENBLAS_BINARY_DIR}
  CMAKE_ARGS       ${_openblas_cmake_args}
  INSTALL_COMMAND  ""
  # Redirect the (very verbose) sub-build output to log files under the build
  # tree; CMake prints the log path only if a step fails.
  LOG_DOWNLOAD     ON
  LOG_CONFIGURE    ON
  LOG_BUILD        ON
  LOG_OUTPUT_ON_FAILURE ON
)

add_library(openblas_lapack STATIC IMPORTED)
if(_openblas_is_multi_config)
  foreach(_cfg ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${_cfg} _cfg_upper)
    set_property(TARGET openblas_lapack PROPERTY
      IMPORTED_LOCATION_${_cfg_upper}
      ${OPENBLAS_BINARY_DIR}/lib/${_cfg_upper}/${CMAKE_STATIC_LIBRARY_PREFIX}openblas${CMAKE_STATIC_LIBRARY_SUFFIX})
  endforeach()
else()
  set_target_properties(openblas_lapack PROPERTIES
    IMPORTED_LOCATION
    ${OPENBLAS_BINARY_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}openblas${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

add_dependencies(openblas_lapack rocsolver-static-lapack)

# Expose as LAPACK_LIBRARIES so callers link exactly as against system LAPACK.
set(LAPACK_LIBRARIES openblas_lapack)
