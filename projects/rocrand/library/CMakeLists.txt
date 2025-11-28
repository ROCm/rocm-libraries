# MIT License
#
# Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

# rocRAND library
# C++17 is used internally
set(CMAKE_CXX_STANDARD 17)
# Get sources
set(rocRAND_HIP_SRCS
    "src/rng/generator_type_lfsr113.cpp"
    "src/rng/generator_type_mrg31k3p.cpp"
    "src/rng/generator_type_mrg32k3a.cpp"
    "src/rng/generator_type_mt19937.cpp"
    "src/rng/generator_type_mtgp32.cpp"
    "src/rng/generator_type_philox4x32_10.cpp"
    "src/rng/generator_type_scrambled_sobol32.cpp"
    "src/rng/generator_type_scrambled_sobol64.cpp"
    "src/rng/generator_type_sobol32.cpp"
    "src/rng/generator_type_sobol64.cpp"
    "src/rng/generator_type_threefry2x32_20.cpp"
    "src/rng/generator_type_threefry2x64_20.cpp"
    "src/rng/generator_type_threefry4x32_20.cpp"
    "src/rng/generator_type_threefry4x64_20.cpp"
    "src/rng/generator_type_xorwow.cpp"
    "src/rocrand_mt19937_precomputed.cpp"
    "src/rocrand.cpp")

# Configure a header file to pass the rocRAND version
configure_file(
    "${PROJECT_SOURCE_DIR}/library/include/rocrand/rocrand_version.h.in"
    "${PROJECT_BINARY_DIR}/library/include/rocrand/rocrand_version.h"
    @ONLY
)

add_library(rocrand ${rocRAND_HIP_SRCS})

set(SOBOL_PRECOMPUTED_SOURCES
    "src/rocrand_scrambled_sobol32_precomputed"
    "src/rocrand_scrambled_sobol64_precomputed"
    "src/rocrand_sobol32_precomputed"
    "src/rocrand_sobol64_precomputed")

if (ROCRAND_HAVE_ASM_INCBIN)
    enable_language(ASM)
    list(TRANSFORM SOBOL_PRECOMPUTED_SOURCES APPEND ".S")
    set_source_files_properties(${SOBOL_PRECOMPUTED_SOURCES} PROPERTIES
                                INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/src)
else()
    list(TRANSFORM SOBOL_PRECOMPUTED_SOURCES APPEND ".cpp")
endif()
target_sources(rocrand PRIVATE ${SOBOL_PRECOMPUTED_SOURCES})

set_target_properties(rocrand PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

if (NOT BUILD_SHARED_LIBS)
    target_compile_definitions(rocrand PUBLIC ROCRAND_STATIC_BUILD)
endif()

if (HIP_COMPILER STREQUAL "nvcc")
    set_target_properties(rocrand PROPERTIES CUDA_VISIBILITY_PRESET hidden)
endif()

# Add interface include directory so that other CMake applications can maintain previous behaviour.
# This will be removed with upcoming packaging changes.
target_include_directories(rocrand INTERFACE $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/rocrand>)

target_include_directories(
    rocrand
    PRIVATE
        "${PROJECT_SOURCE_DIR}/library/include"
        "${PROJECT_BINARY_DIR}/library/include"
)

# Build library
if(HIP_COMPILER STREQUAL "nvcc")
    set_source_files_properties(${rocRAND_HIP_SRCS}
        PROPERTIES LANGUAGE CUDA CUDA_STANDARD 17)
    if(${CMAKE_VERSION} VERSION_LESS "3.18")
        set_source_files_properties(${rocRAND_HIP_SRCS} PROPERTIES COMPILE_OPTIONS "-std=c++17")
    endif()
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
else()
    target_link_libraries(rocrand PRIVATE hip::device)
    target_link_libraries(rocrand PUBLIC hip::host)
endif()

rocm_set_soversion(rocrand ${rocrand_SOVERSION})
set_target_properties(rocrand
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/library"
        DEBUG_POSTFIX "-d"
)

rocm_install(
    TARGETS rocrand
    INCLUDE
        "${PROJECT_SOURCE_DIR}/library/include"
        "${PROJECT_BINARY_DIR}/library/include"
)

set(FORTRAN_SRCS_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/rocrand/src/fortran")
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
set(CONFIG_PACKAGE_INSTALL_DIR ${LIB_INSTALL_DIR}/cmake/rocrand)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    src/rocrand-fortran-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rocrand-fortran-config.cmake
    INSTALL_DESTINATION ${CONFIG_PACKAGE_INSTALL_DIR}
)

if(HIP_COMPILER STREQUAL "nvcc")
    rocm_export_targets(
        TARGETS roc::rocrand
        NAMESPACE roc::
        INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/rocrand-fortran-config.cmake"
    )
else()
    rocm_export_targets(
        TARGETS roc::rocrand
        NAMESPACE roc::
        DEPENDS PACKAGE hip
        INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/rocrand-fortran-config.cmake"
    )
endif()

# install library to C:\hipSDK\bin
if (WIN32)
    install (TARGETS rocrand DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
    if (BUILD_TEST)
	    add_custom_command(
		    TARGET rocrand
		    POST_BUILD
		    COMMAND ${CMAKE_COMMAND} -E copy
			    $<TARGET_FILE:rocrand>
			    ${PROJECT_BINARY_DIR}/test/$<TARGET_FILE_NAME:rocrand>
	    )
    endif()
    if (BUILD_BENCHMARK)
	    add_custom_command(
		    TARGET rocrand
		    POST_BUILD
		    COMMAND ${CMAKE_COMMAND} -E copy
			    $<TARGET_FILE:rocrand>
			    ${PROJECT_BINARY_DIR}/benchmark/$<TARGET_FILE_NAME:rocrand>
	    )
    endif()
endif()

# Fortran wrappers for rocRAND
if(BUILD_FORTRAN_WRAPPER)
    add_subdirectory(src/fortran)
endif()
