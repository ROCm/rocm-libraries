################################################################################
#
# Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

cmake_minimum_required(VERSION 3.25)

project(TensileHostLibraryTest)

set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_EXTENSIONS OFF )

list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} /opt/rocm)

option(TENSILE_USE_HIP       "Use the Hip runtime." ON)
option(TENSILE_USE_LLVM      "Use LLVM for parsing config files." ON)
option(TENSILE_USE_MSGPACK   "Use msgpack for parsing config files." ON)
option(TENSILE_USE_OPENMP    "Enable OpenMP multithreading." ON)
option(TENSILE_STATIC_ONLY   "Disable exposing Tensile symbols in a shared library." ON)
option(TENSILE_DISABLE_CTEST "Disable CTest integration." ON)

if(TENSILE_USE_HIP)
    set(TENSILE_COMPONENTS HIP)
    if(TENSILE_USE_LLVM)
        set(TENSILE_COMPONENTS ${TENSILE_COMPONENTS} Client)
    endif()
endif()
if(TENSILE_USE_LLVM)
    set(TENSILE_COMPONENTS ${TENSILE_COMPONENTS} LLVM)
endif()
if(TENSILE_USE_OPENMP)
    set(TENSILE_COMPONENTS ${TENSILE_COMPONENTS} OpenMP)
endif()
if(TENSILE_STATIC_ONLY)
    set(TENSILE_COMPONENTS ${TENSILE_COMPONENTS} STATIC_ONLY)
endif()

if(NOT Tensile_FOUND)
    find_package(Tensile 4.33.0 EXACT REQUIRED ${TENSILE_COMPONENTS} PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../Tensile")
endif()

if(NOT TENSILE_DISABLE_CTEST)
    enable_testing()
    find_package( GTest )
    if(GTest_FOUND)
        option(GTEST_INTEGRATION "Enable Googletest integration with CMake." ON)
    else()
        option(GTEST_INTEGRATION "Enable Googletest integration with CMake." OFF)
    endif()
endif()


find_package(Boost COMPONENTS filesystem)

#if(TENSILE_USE_HIP AND NOT HIP_DIR)
if(TENSILE_USE_HIP)
    find_package(HIP REQUIRED CONFIG PATHS $ENV{ROCM_PATH} /opt/rocm)
endif()

if(TENSILE_USE_OPENMP AND NOT TARGET custom_openmp_cxx)

    # Workaround for https://gitlab.kitware.com/cmake/cmake/-/issues/21787
    # ensures we link to HIP's libomp and get an rpath to it.
    add_library(custom_openmp_cxx INTERFACE)

    if(TENSILE_USE_HIP)
        target_compile_options(custom_openmp_cxx INTERFACE "-fopenmp")
        target_link_options(custom_openmp_cxx INTERFACE "-fopenmp")
    else ()
        find_package(OpenMP REQUIRED)
        target_link_libraries(custom_openmp_cxx INTERFACE OpenMP::OpenMP_CXX)
    endif ()
endif()

add_subdirectory(configs)

set(TEST_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")
file(MAKE_DIRECTORY "${TEST_DATA_DIR}")

foreach(FILE ${SOLUTION_LIBRARY_FILES})
    if(EXISTS ${FILE})
            endif()
    execute_process(COMMAND gzip --keep --decompress --force "${FILE}.gz")
    file(COPY ${FILE} DESTINATION "${TEST_DATA_DIR}")
endforeach(FILE)

add_subdirectory(testlib)

set(test_sources ${test_sources}
    test.cpp
    ContractionSelectionLibrary_test.cpp
    DataTypes_test.cpp
    Predicates_test.cpp
)

if(TENSILE_USE_LLVM)
    find_package(LLVM REQUIRED CONFIG)

    set(test_sources ${test_sources}
        llvm/LibraryPerformance_test.cpp
        )
endif()

# if(TENSILE_USE_HIP)
#     add_subdirectory(hip)
# endif()

add_executable(TensileTests ${test_sources})

if(NOT TENSILE_DISABLE_CTEST)
    if(GTEST_INTEGRATION)
        gtest_discover_tests(TensileTests WORKING_DIRECTORY ${CMAKE_BINARY_DIR} TIMEOUT 60)
    else()
        add_test(NAME TensileHostLibraryTests COMMAND TensileTests WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    endif()
endif()

target_link_libraries(TensileTests PUBLIC TensileHost TensileTestLib ${Boost_LIBRARIES})

if(GTEST_INTEGRATION)
    target_link_libraries(TensileTests PUBLIC GTest::gtest)
endif()

if(TENSILE_USE_LLVM)
    find_library(LLVMObjectYAML_LIBRARY
        NAMES LLVMObjectYAML
        PATHS ${LLVM_LIBRARY_DIR})
    message("LLVMObjectYAML_LIBRARY: ${LLVMObjectYAML_LIBRARY}")
    #Use LLVM shared libs if LLVMObjectYAML static lib cannot be found
    if(LLVMObjectYAML_LIBRARY)
        target_link_libraries(TensileTests PUBLIC LLVMObjectYAML )
    else()
        target_link_libraries(TensileTests PUBLIC LLVM )
    endif()
    target_include_directories(TensileTests PUBLIC ${LLVM_INCLUDE_DIRS})
endif()

if(TENSILE_USE_HIP)

    target_link_libraries(TensileTests PRIVATE "-Xlinker --whole-archive")
    target_link_libraries(TensileTests PUBLIC ${HIP_TEST_LIBRARIES})
    target_link_libraries(TensileTests PRIVATE "-Xlinker --no-whole-archive")
    foreach(arch IN LISTS TENSILE_GPU_ARCHS)
        target_link_libraries(TensileTests PRIVATE "--offload-arch=${arch}")
    endforeach(arch)
endif()

if(TENSILE_USE_OPENMP)
    target_link_libraries(TensileTests PRIVATE custom_openmp_cxx)
endif()

