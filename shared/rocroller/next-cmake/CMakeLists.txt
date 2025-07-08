################################################################################
#
# MIT License
#
# Copyright 2025 AMD ROCm(TM) Software
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

cmake_minimum_required(VERSION 3.21.0)
project(rocroller VERSION 1.0.0)

set(ROCROLLER_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../lib")
set(ROCROLLER_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../test")
set(ROCROLLER_GPUARCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../GPUArchitectureGenerator")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CMakeRC)
include(FetchContent)
include(CMakeDependentOption)

FetchContent_Declare(
    ROCmCMakeBuildTools
    GIT_REPOSITORY https://github.com/ROCm/rocm-cmake.git
    GIT_TAG rocm-6.3.0
    SOURCE_SUBDIR "DISABLE ADDING TO BUILD"
    # Don't consume the build/test targets of ROCmCMakeBuildTools
)
FetchContent_MakeAvailable(ROCmCMakeBuildTools)
list(
    APPEND CMAKE_MODULE_PATH
    ${rocmcmakebuildtools_SOURCE_DIR}/share/rocmcmakebuildtools/cmake
)
include(ROCMInstallTargets)

option(ROCROLLER_ENABLE_CLIENT "Build the rocRoller client." ON)
option(ROCROLLER_ENABLE_YAML_CPP "Enable yaml-cpp backend." ON)
cmake_dependent_option(ROCROLLER_ENABLE_LLVM "Enable llvm yaml backend." ON "NOT ROCROLLER_ENABLE_YAML_CPP" OFF)
option(ROCROLLER_BUILD_TESTING "Build rocRoller testing." ON)
option(ROCROLLER_ENABLE_CATCH "Build rocRoller catch unit tests" ON)
option(ROCROLLER_ENABLE_ARCH_GEN_TEST "Build rocRoller architecture generator test" ON)
option(ROCROLLER_ENABLE_TEST_DISCOVERY "Use gtest and catch2 test discovery functions." ON)
option(ROCROLLER_ENABLE_COVERAGE "Build code coverage." OFF)
cmake_dependent_option(ROCROLLER_TESTS_SKIP_SLOW "Disable slow running tests." ON "ROCROLLER_ENABLE_COVERAGE" OFF)
option(ROCROLLER_EMBED_ARCH_DEF "Embed msgpack architecture data in library." ON)
option(ROCROLLER_BUILD_SHARED_LIBS "Build rocRoller as a shared library." ON)
option(ROCROLLER_ENABLE_FETCH "Enable fetch content for dependencies if find_package fails." OFF)
option(ROCROLLER_ENABLE_LLD "Build rocroller functionality requiring LLD" OFF)
option(ROCROLLER_ENABLE_TIMERS "Enable rocRoller timer code." OFF)

find_package(fmt 11.1.3)
if(NOT fmt_FOUND)
    if(ROCROLLER_ENABLE_FETCH)
        set(FMT_SYSTEM_HEADERS ON)
        FetchContent_Declare(
            fmt
            GIT_REPOSITORY https://github.com/fmtlib/fmt.git
            GIT_TAG 11.1.3
        )
        FetchContent_MakeAvailable(fmt)
    else()
        message(FATAL_ERROR "Failed to find fmt")
    endif()
endif()

find_package(spdlog)
if(NOT spdlog_FOUND)
    if(ROCROLLER_ENABLE_FETCH)
        FetchContent_Declare(
            spdlog
            GIT_REPOSITORY https://github.com/gabime/spdlog.git
            GIT_TAG v1.x
        )
        FetchContent_MakeAvailable(spdlog)
    else()
        message(FATAL_ERROR "Failed to find spdlog")
    endif()
endif()

find_package(libdivide)
if(NOT libdivide_FOUND)
if(ROCROLLER_ENABLE_FETCH)
    FetchContent_Declare(
        libdivide
        GIT_REPOSITORY https://github.com/ridiculousfish/libdivide.git
        GIT_TAG master
    )
    FetchContent_MakeAvailable(libdivide)
    else()
        message(FATAL_ERROR "Failed to find libdivide")
    endif()
endif()

find_package(msgpack-cxx CONFIG)
if(msgpack-cxx_FOUND)
    message(STATUS "Found msgpack-cxx (>=6.x)")
    set(msgpack_libs msgpack-cxx)
else()
    find_package(msgpackc-cxx CONFIG REQUIRED NAMES msgpackc-cxx msgpack)
    message(STATUS "Found msgpack (<=5.x)")
    set(msgpack_libs msgpackc)
endif()

find_package(Boost 1.74.0)
if(NOT Boost_FOUND)
    if(ROCROLLER_ENABLE_FETCH)
        FetchContent_Declare(
            boost
            URL https://github.com/boostorg/boost/releases/download/boost-1.87.0/boost-1.87.0-cmake.tar.gz
        )
        set(boost_temp_testing ${BUILD_TESTING})
        set(BUILD_TESTING OFF)
        set(BUILD_SHARED_LIBS OFF)
        set(boost_temp_shared ${BUILD_SHARED_LIBS})
        set(Boost_USE_STATIC_LIBS ON)
        set(BOOST_INCLUDE_LIBRARIES multi_index)
        FetchContent_MakeAvailable(boost)
        set(BUILD_TESTING ${boost_temp_testing})
        set(BUILD_SHARED_LIBS ${boost_temp_shared})
    else()
        message(FATAL_ERROR "Failed to find boost")
    endif()
endif()

find_package(hip REQUIRED)
find_package(amd_comgr REQUIRED)
if(ROCROLLER_ENABLE_LLVM)
    find_package(LLVM REQUIRED)
endif()

if(ROCROLLER_ENABLE_LLD)
    find_package(LLD REQUIRED)
    target_include_directories(rocroller PRIVATE "${LLD_INCLUDES}")
    target_compile_definitions(rocroller PUBLIC ROCROLLER_USE_LLD)
endif()

if(ROCROLLER_BUILD_SHARED_LIBS)
    add_library(rocroller SHARED)
else()
    add_library(rocroller STATIC)
    set_target_properties(rocroller PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
add_library(roc::rocroller ALIAS rocroller)

add_library(rocroller-interface INTERFACE)
add_library(roc::rocroller-interface ALIAS rocroller-interface)

if(ROCROLLER_BUILD_SHARED_LIBS)
    target_link_libraries(rocroller-interface INTERFACE roc::rocroller)
else()
    target_link_libraries(rocroller-interface INTERFACE $<LINK_LIBRARY:WHOLE_ARCHIVE,roc::rocroller>)
endif()

if(ROCROLLER_ENABLE_TIMERS)
    target_compile_definitions(rocroller PUBLIC ROCROLLER_ENABLE_TIMERS)
    target_compile_options(rocroller PRIVATE -pg -fno-omit-frame-pointer)
endif()

if(ROCROLLER_ENABLE_YAML_CPP)
    find_package(yaml-cpp 0.8.0)
    if(NOT yaml-cpp_FOUND)
        if(ROCROLLER_ENABLE_FETCH)
            set(YAML_CPP_BUILD_TOOLS OFF)
            set(YAML_BUILD_SHARED_LIBS OFF)
            option(YAML_CPP_INSTALL "" ON)
            FetchContent_Declare(
                yaml_cpp
                GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
                GIT_TAG 0.8.0
            )
            FetchContent_MakeAvailable(yaml_cpp)
        else()
            message(FATAL_ERROR "Failed to find yaml-cpp")
        endif()
    endif()
    target_link_libraries(rocroller PUBLIC yaml-cpp::yaml-cpp)
    target_compile_definitions(rocroller PUBLIC ROCROLLER_USE_YAML_CPP)
endif()

target_compile_definitions(rocroller
    PUBLIC
        ROCROLLER_USE_HIP
        __HIP_PLATFORM_AMD__
)

if(ROCROLLER_ENABLE_LLVM)
    llvm_map_components_to_libnames(llvm_yaml_libs objectyaml)
    llvm_map_components_to_libnames(llvm_libs mc support)
endif()

target_link_libraries(rocroller
    PUBLIC
       Boost::headers
       hip::host
    PRIVATE
        ${msgpack_libs}
        amd_comgr
)

if(ROCROLLER_ENABLE_LLVM)
    # ROCROLLER_USE_LLVM enables the llvm yaml API
    target_compile_definitions(rocroller PUBLIC ROCROLLER_USE_LLVM)
endif()

target_link_libraries(rocroller
    PUBLIC
        fmt::fmt-header-only
    PRIVATE
        $<BUILD_INTERFACE:spdlog::spdlog_header_only>
        $<BUILD_INTERFACE:libdivide::libdivide>
        ${llvm_yaml_libs}
        ${llvm_libs}
)

target_include_directories(rocroller PUBLIC $<BUILD_INTERFACE:${ROCROLLER_LIB_DIR}/include>)

target_compile_features(rocroller PUBLIC cxx_std_20)

configure_file(
    "${ROCROLLER_LIB_DIR}/source/Utilities/Version.cpp.in"
    "${PROJECT_BINARY_DIR}/rocRoller/src/Version.cpp"
)
target_sources(rocroller PRIVATE "${PROJECT_BINARY_DIR}/rocRoller/src/Version.cpp")

add_subdirectory(src)
add_subdirectory(include)
add_subdirectory(GPUArchitectureGenerator)

if(ROCROLLER_EMBED_ARCH_DEF)
    target_link_libraries(rocroller PUBLIC $<BUILD_INTERFACE:rocroller::GPUArchitecture_def>)
    target_compile_definitions(rocroller PRIVATE ROCROLLER_EMBED_ARCH_DEF)
endif()

if(ROCROLLER_BUILD_TESTING)
    add_subdirectory(test)
endif()

if(ROCROLLER_ENABLE_CLIENT)
    add_subdirectory(client)
endif()

rocm_install(
    TARGETS rocroller rocroller-interface
    EXPORT rocroller-targets
    INCLUDE
        "${ROCROLLER_LIB_DIR}/include"
)

if(ROCROLLER_BUILD_TESTING OR BUILD_TESTING)
    rocm_install(
        TARGETS rocroller-tests
        COMPONENT tests
    )
    if(ROCROLLER_ENABLE_CATCH)
        rocm_install(
            TARGETS rocroller-tests-catch
            COMPONENT tests
        )
    endif()
endif()

rocm_export_targets(
    TARGETS roc::rocroller roc::rocroller-interface
    NAMESPACE roc::
    EXPORT rocroller-targets
    DEPENDS PACKAGE yaml-cpp
    DEPENDS PACKAGE Boost
    DEPENDS PACKAGE fmt
    DEPENDS PACKAGE hip
)
