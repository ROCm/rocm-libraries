# MIT License
#
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

# hipRAND library
# Get hipRAND sources
if (BUILD_WITH_LIB STREQUAL "CUDA")
    set(hipRAND_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/nvidia_detail/hiprand.cpp)
else()
    set(hipRAND_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/amd_detail/hiprand.cpp)
endif()

# Configure a header file to pass the hipRAND version
configure_file(
    "${PROJECT_SOURCE_DIR}/library/include/hiprand/hiprand_version.h.in"
    "${PROJECT_BINARY_DIR}/library/include/hiprand/hiprand_version.h"
    @ONLY
)

# Build
if (BUILD_WITH_LIB STREQUAL "CUDA")
    set_source_files_properties(${hipRAND_SRCS} PROPERTIES LANGUAGE CUDA)
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
endif()
add_library(hiprand ${hipRAND_SRCS})
add_library(hip::hiprand ALIAS hiprand)

# Add interface include directory so that other CMake applications can maintain previous behaviour.
# This will be removed with upcoming packaging changes.
target_include_directories(hiprand INTERFACE $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/hiprand>)

if (BUILD_WITH_LIB STREQUAL "CUDA")
    target_link_libraries(hiprand
        ${CUDA_curand_LIBRARY}
    )
else()
    target_link_libraries(hiprand PRIVATE hip::device)
    target_link_libraries(hiprand PUBLIC roc::rocrand hip::host)
endif()

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(hiprand PUBLIC HIPRAND_STATIC_BUILD)
endif()

rocm_set_soversion(hiprand ${hiprand_SOVERSION})
set_target_properties(hiprand
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/library"
        DEBUG_POSTFIX "-d"
)

rocm_install(
    TARGETS hiprand
    INCLUDE 
        "${PROJECT_SOURCE_DIR}/library/include"
        "${PROJECT_BINARY_DIR}/library/include"
    EXPORT hiprand-targets
)

set(FORTRAN_SRCS_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/hiprand/src/fortran")
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
set(CONFIG_PACKAGE_INSTALL_DIR ${LIB_INSTALL_DIR}/cmake/hiprand)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    src/hiprand-fortran-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/hiprand-fortran-config.cmake
    INSTALL_DESTINATION ${CONFIG_PACKAGE_INSTALL_DIR}
)

if (BUILD_WITH_LIB STREQUAL "CUDA")
    rocm_export_targets(
        TARGETS hip::hiprand
        NAME hiprand
        NAMESPACE hip::
        INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/hiprand-fortran-config.cmake"
    )
else()
    rocm_export_targets(
        TARGETS hip::hiprand
        NAME hiprand
        NAMESPACE hip::
        DEPENDS PACKAGE hip
        DEPENDS PACKAGE rocrand
        STATIC_DEPENDS PACKAGE rocrand
        INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/hiprand-fortran-config.cmake"
    )
endif()

# install library to C:\hipSDK\bin
if (WIN32)
    install (TARGETS hiprand DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
    if (BUILD_TEST)
	    add_custom_command(
		    TARGET hiprand 
		    POST_BUILD
		    COMMAND ${CMAKE_COMMAND} -E copy
			    $<TARGET_FILE:hiprand>
			    ${PROJECT_BINARY_DIR}/test/$<TARGET_FILE_NAME:hiprand>
	    )
    endif()
    if (BUILD_BENCHMARK)
	    add_custom_command(
		    TARGET hiprand 
		    POST_BUILD
		    COMMAND ${CMAKE_COMMAND} -E copy
			    $<TARGET_FILE:hiprand>
			    ${PROJECT_BINARY_DIR}/benchmark/$<TARGET_FILE_NAME:hiprand>
	    )
    endif()	
endif()

# Fortran wrappers for hipRAND
if(BUILD_FORTRAN_WRAPPER)
    add_subdirectory(src/fortran)
endif()
