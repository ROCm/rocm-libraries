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

# Location of golden/input data (defaults to the legacy suite for now).
set(RPP_TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../test_suite"
    CACHE PATH "Directory containing test input/golden data")

add_executable(rpp_tests
    core/handle_test.cpp
    core/descriptor_test.cpp
    image/brightness_test.cpp
)
target_include_directories(rpp_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(rpp_INCLUDE_DIR)
    target_include_directories(rpp_tests PRIVATE ${rpp_INCLUDE_DIR}/rpp)
endif()
target_link_libraries(rpp_tests PRIVATE ${RPP_TEST_LINK_LIB} GTest::gtest_main)
target_compile_definitions(rpp_tests PRIVATE RPP_TEST_DATA_DIR="${RPP_TEST_DATA_DIR}")

if(rpp_BACKEND_TYPE STREQUAL "HIP" OR BACKEND STREQUAL "HIP")
    target_compile_definitions(rpp_tests PRIVATE RPP_TEST_HAVE_HIP=1)
endif()

gtest_discover_tests(rpp_tests)
