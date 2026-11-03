# MIT License
#
# Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

# Package test
cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

if(DEFINED ENV{ROCM_PATH})
  set (ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "The path to the ROCm installation.")
endif()
set (ROCM_PATH "/opt/rocm" CACHE PATH "The path to the ROCm installation.")

project(hiprand_package_test CXX)

# CMake modules
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)
list(APPEND CMAKE_MODULE_PATH
    ${HIP_DIR}/cmake ${ROCM_PATH}/hip/cmake # FindHIP.cmake
)

# Find HIP
if (BUILD_WITH_LIB STREQUAL "CUDA")
    find_package(hip QUIET CONFIG PATHS ${ROCM_PATH})
    if(NOT hip_FOUND)
        find_package(HIP REQUIRED)
    endif()
else()
  find_package(hip REQUIRED CONFIG PATHS ${ROCM_PATH})
endif()

if (NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_CPP_CONFIG}")

if (CMAKE_CXX_STANDARD EQUAL 14)
  message(WARNING "C++14 will be deprecated in the next major release")
elseif(NOT CMAKE_CXX_STANDARD EQUAL 17)
  message(FATAL_ERROR "Only C++14 and C++17 are supported")
endif()

# CUDA
if (BUILD_WITH_LIB STREQUAL "CUDA")
    set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
    find_package(CUDA REQUIRED)
endif()

# Find hipRAND
find_package(hiprand REQUIRED CONFIG HINTS ${hiprand_DIR} PATHS "${ROCM_PATH}/hiprand")

# TODO: Fix fortran wrapper
# Check if Fortran wrapper is installed
#if(NOT EXISTS "${hiprand_FORTRAN_SRC_DIRS}/hiprand_m.f90")
#    message(FATAL_ERROR "${rocrand_FORTRAN_SRC_DIRS}/hiprand_m.f90 does not exist")
#endif()
#if(NOT EXISTS "${hiprand_FORTRAN_SRC_DIRS}/hip_m.f90")
#    message(FATAL_ERROR "${hiprand_FORTRAN_SRC_DIRS}/hip_m.f90 does not exist")
#endif()
#if(NOT EXISTS "${rocrand_FORTRAN_SRC_DIRS}/rocrand_m.f90")
#    message(FATAL_ERROR "${rocrand_FORTRAN_SRC_DIRS}/rocrand_m.f90 does not exist")
#endif()
#if(NOT EXISTS "${rocrand_FORTRAN_SRC_DIRS}/hip_m.f90")
#    message(FATAL_ERROR "${rocrand_FORTRAN_SRC_DIRS}/hip_m.f90 does not exist")
#endif()

# Get sources
file(GLOB hiprand_pkg_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/test_hiprand*.cpp)

# Enable testing (ctest)
enable_testing()

function(add_relative_test test_name test_target)
    get_target_property(EXE_PATH ${test_target} RUNTIME_OUTPUT_DIRECTORY)
    if(EXE_PATH STREQUAL "EXE_PATH-NOTFOUND")
        set(EXE_PATH ".")
    endif()
    get_filename_component(EXE_PATH "${EXE_PATH}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    get_target_property(EXE_NAME ${test_target} RUNTIME_OUTPUT_NAME)
    if(EXE_NAME STREQUAL "EXE_NAME-NOTFOUND")
        get_target_property(EXE_NAME ${test_target} OUTPUT_NAME)
        if(EXE_NAME STREQUAL "EXE_NAME-NOTFOUND")
            set(EXE_NAME "${test_target}")
        endif()
    endif()
    file(RELATIVE_PATH rel_path "${CMAKE_CURRENT_BINARY_DIR}" "${EXE_PATH}/${EXE_NAME}")
    add_test(NAME "${test_name}" COMMAND "./${rel_path}")
endfunction()

# Build
foreach(test_src ${hiprand_pkg_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} "${test_src}")
    if(HIP_COMPILER STREQUAL "clang")
        target_link_libraries(${test_name}
            PRIVATE
                hip::hiprand
                hip::host
        )
        if(WIN32)
            target_compile_options(${test_name} PRIVATE "-xhip")
        endif()
    else()
        target_link_libraries(${test_name}
            PRIVATE
                hip::hiprand
                ${CUDA_curand_LIBRARY}
                ${CUDA_LIBRARIES}
        )
        # Get HIP options
        execute_process(
            COMMAND ${HIP_HIPCONFIG_EXECUTABLE} --cpp_config
            OUTPUT_VARIABLE HIP_CPP_CONFIG
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE
        )
        separate_arguments(HIP_CPP_CONFIG UNIX_COMMAND "${HIP_CPP_CONFIG}")
        target_compile_options(${test_name} PRIVATE ${HIP_CPP_CONFIG})
    endif()
    set_target_properties(${TEST_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
    )
    add_relative_test(${test_name} ${test_name})
    if (WIN32 AND NOT DEFINED DLLS_COPIED)
      set(DLLS_COPIED "YES")
      set(DLLS_COPIED ${DLLS_COPIED} PARENT_SCOPE)
      # for now adding in all .dll as dependency chain is not cmake based on win32
      file( GLOB third_party_dlls
      LIST_DIRECTORIES ON
      CONFIGURE_DEPENDS
      ${HIP_DIR}/bin/*.dll
      ${CMAKE_SOURCE_DIR}/rtest.*
      )
      foreach( file_i ${third_party_dlls})
        add_custom_command( TARGET ${test_name} POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${file_i} ${PROJECT_BINARY_DIR}/test/package )
      endforeach( file_i )
    endif()         
endforeach()
