###############################################################################
 #
 # MIT License
 #
 # Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 # THE SOFTWARE.
 #
 ###############################################################################

#Prevent CMake to add building path to RUNPATH
set(CMAKE_SKIP_RPATH  TRUE)

# Check whether building within hiptensor context
if( CMAKE_PROJECT_NAME STREQUAL "hiptensor" )
    # Target that will trigger build of all samples
    add_custom_target(hiptensor_samples)

    # Create sample executables and deploy
    function(add_hiptensor_sample BINARY_NAME FILE_NAME)

        message( STATUS "adding hiptensor sample: ${BINARY_NAME}")
        add_executable(${BINARY_NAME} ${FILE_NAME})

        target_compile_options(${BINARY_NAME} PRIVATE ${CLANG_DRIVER_MODE})
        target_link_options(${BINARY_NAME} PRIVATE ${CLANG_DRIVER_MODE})

        # Sample must propagate the build interface includes to make sure
        # hiptensor includes are captured at runtime.
        target_link_libraries(${BINARY_NAME} PRIVATE hiptensor::hiptensor "-L${HIP_CLANG_ROOT}/lib" "-Wl,-rpath=$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
                              "-fno-rtlib-add-rpath")
        target_include_directories(${BINARY_NAME} PRIVATE
                                ${CMAKE_CURRENT_SOURCE_DIR}
                                ${PROJECT_SOURCE_DIR}/samples
                                ${PROJECT_SOURCE_DIR}/library/src/include/
                                ${PROJECT_SOURCE_DIR}/library/include)

        # Build this sample under custom target
        add_dependencies(hiptensor_samples ${BINARY_NAME})

        # Install with rocm pkg
        rocm_install_targets(
        TARGETS ${BINARY_NAME}
        COMPONENT samples
        )
    endfunction()
# If building hipTensor samples as a standalone Cmake project
else()
    if(NOT CMAKE_CXX_COMPILER)
        set(CMAKE_CXX_COMPILER "/opt/rocm/bin/amdclang++" CACHE STRING
            "Choose the type of compiler to build: Default point to amdclang++"
            FORCE)
    endif()

    cmake_minimum_required(VERSION 3.14)
    project(hiptensor_app)
    add_compile_options(-std=c++17)

    # Requires linking installed composable kernel and hipTensor library
    find_package(hiptensor 0.2.0 REQUIRED PATHS /opt/rocm /opt/rocm/hiptensor $ENV{HIPTENSOR_DIR}/lib/cmake COMPONENTS hiptensor)

    # This helps cmake properly find hip-config.cmake
    list( APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} ${ROCM_PATH} /opt/rocm )
    # Append our library helper cmake path and the cmake path for hip (for convenience).
    # Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
    list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

    find_package(ROCM 0.8 CONFIG REQUIRED)
    find_package( hip REQUIRED )

    message(STATUS "Build with HIP ${hip_VERSION}")
endif()
add_subdirectory(01_contraction)
add_subdirectory(02_permutation)
add_subdirectory(03_reduction)
