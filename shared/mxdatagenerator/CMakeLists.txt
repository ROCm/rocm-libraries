
################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

cmake_minimum_required(VERSION 3.16)
project(FPDataGeneration VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 20)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(OpenMP)
add_library(mxDataGenerator INTERFACE)
add_library(roc::mxDataGenerator ALIAS mxDataGenerator)
target_link_libraries(mxDataGenerator INTERFACE OpenMP::OpenMP_CXX)
target_include_directories(mxDataGenerator
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_sources(mxDataGenerator
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include/DataGenerator.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include/bf16.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include/bf16_impl.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include/dataTypeInfo.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include/dataTypeInfo_impl.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include/data_generation_utils.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include/f32.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include/f32_impl.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include/fp16.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include/fp16_impl.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include/fp6.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include/ocp_e2m1_mxfp4.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include/ocp_e2m1_mxfp4_impl.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include/ocp_e2m3_mxfp6.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include/ocp_e2m3_mxfp6_impl.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include/ocp_e3m2_mxfp6.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include/ocp_e3m2_mxfp6_impl.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include/ocp_e4m3_mxfp8.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include/ocp_e4m3_mxfp8_impl.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include/ocp_e5m2_mxfp8.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include/ocp_e5m2_mxfp8_impl.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include/packing.hpp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/DataGenerator.hpp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/bf16.hpp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/bf16_impl.hpp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/dataTypeInfo.hpp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/dataTypeInfo_impl.hpp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/data_generation_utils.hpp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/f32.hpp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/f32_impl.hpp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fp16.hpp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fp16_impl.hpp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fp6.hpp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ocp_e2m1_mxfp4.hpp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ocp_e2m1_mxfp4_impl.hpp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ocp_e2m3_mxfp6.hpp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ocp_e2m3_mxfp6_impl.hpp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ocp_e3m2_mxfp6.hpp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ocp_e3m2_mxfp6_impl.hpp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ocp_e4m3_mxfp8.hpp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ocp_e4m3_mxfp8_impl.hpp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ocp_e5m2_mxfp8.hpp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ocp_e5m2_mxfp8_impl.hpp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/packing.hpp>
)

if(MXDATAGENERATOR_BUILD_TESTING)
    find_package(GTest 1.12.1)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
          googletest
          URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
        )
        FetchContent_MakeAvailable(googletest)
    endif()
    enable_testing()

    add_executable(
      mxDataGeneratorTests
      test/data_generator_test.cpp
      test/f32_test.cpp
      test/fp16_test.cpp
      test/bf16_test.cpp
      test/ocp_e4m3_mxfp8_test.cpp
      test/ocp_e5m2_mxfp8_test.cpp
      test/ocp_e2m3_mxfp6_test.cpp
      test/ocp_e3m2_mxfp6_test.cpp
      test/ocp_e2m1_mxfp4_test.cpp
    )
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(mxDataGeneratorTests PRIVATE -Wall -Wextra -Wpedantic -Werror -Wno-strict-aliasing -Wno-error=strict-aliasing)
        target_link_options(mxDataGeneratorTests PRIVATE -fsanitize=undefined)
    endif()
    target_link_libraries(
      mxDataGeneratorTests PRIVATE
      GTest::gtest_main
      mxDataGenerator
      OpenMP::OpenMP_CXX
    )
    include(GoogleTest)
    gtest_discover_tests(mxDataGeneratorTests DISCOVERY_MODE PRE_TEST)
endif()

get_target_property(mxdatagenerator_headers_public mxDataGenerator INTERFACE_SOURCES)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/mxDataGeneratorConfig-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY ExactVersion
  )
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/mxDataGeneratorConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/mxDataGeneratorConfig.cmake"
    COPYONLY
)
install(
    TARGETS mxDataGenerator
    EXPORT mxDataGenerator-targets
    COMPONENT mxDataGenerator_Development
)
install(
    EXPORT mxDataGenerator-targets
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/mxDataGenerator"
    NAMESPACE roc::
    COMPONENT mxDataGenerator_Development
    FILE mxDataGenerator-targets.cmake
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/mxDataGeneratorConfig-version.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/mxDataGeneratorConfig.cmake"
    COMPONENT mxDataGenerator_Development 
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mxDataGenerator
)
install(
    FILES ${mxdatagenerator_headers_public}
    COMPONENT mxDataGenerator_Development
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md"
    COMPONENT mxDataGenerator_Docs
    DESTINATION "${CMAKE_INSTALL_DOCDIR}"
)
export(EXPORT mxDataGenerator-targets FILE mxDataGeneratorConfig.cmake)
