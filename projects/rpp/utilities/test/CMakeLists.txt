#[[
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
]]

cmake_minimum_required(VERSION 3.14)

project(rpp-test CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ROCm path (only used to locate the installed rpp package by default)
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "ROCm installation path")
else()
    set(ROCM_PATH /opt/rocm CACHE PATH "ROCm installation path")
endif()
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)

# Standalone build finds an installed rpp; in-tree build reuses the existing target.
if(NOT TARGET rpp)
    find_package(rpp 3.0.0 REQUIRED)
    set(RPP_TEST_LINK_LIB rpp::rpp)
else()
    set(RPP_TEST_LINK_LIB rpp)
endif()

include(cmake/gtest.cmake)

enable_testing()
include(GoogleTest)
include(CTest)

file(GLOB_RECURSE RPP_TEST_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/*.cpp)

if(NOT RPP_TEST_SOURCES)
    message(FATAL_ERROR "No test sources found under ${CMAKE_CURRENT_SOURCE_DIR}/src/tests")
endif()

add_executable(rpp_tests ${RPP_TEST_SOURCES})
target_include_directories(rpp_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_property(TARGET rpp_tests PROPERTY COMPILE_WARNING_AS_ERROR ON)
if(rpp_INCLUDE_DIR)
    target_include_directories(rpp_tests PRIVATE ${rpp_INCLUDE_DIR}/rpp)
endif()
target_link_libraries(rpp_tests PRIVATE ${RPP_TEST_LINK_LIB} GTest::gtest_main)

if(rpp_BACKEND_TYPE STREQUAL "HIP" OR BACKEND STREQUAL "HIP")
    # GPU_SUPPORT exposes the HIP-only API symbols guarded behind it in the installed
    # headers (e.g. rppt_erode/rppt_dilate); the legacy HIP suite defines it the same way.
    target_compile_definitions(rpp_tests PRIVATE RPP_TEST_HAVE_HIP=1 GPU_SUPPORT=1)
endif()

gtest_discover_tests(rpp_tests)
