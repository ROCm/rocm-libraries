# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.24.4)
project(rocroller VERSION 1.0.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CMakeRC)
include(FetchContent)
include(CMakeDependentOption)
include(rocroller_target_configure_sanitizers)

FetchContent_Declare(
    ROCmCMakeBuildTools
    GIT_REPOSITORY https://github.com/ROCm/rocm-cmake.git
    GIT_TAG rocm-6.3.0
    SOURCE_SUBDIR "DISABLE ADDING TO BUILD"
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
option(ROCROLLER_ENABLE_DOCS "Build rocRoller documentation." OFF)
option(ROCROLLER_ENABLE_CATCH "Build rocRoller catch unit tests" ON)
option(ROCROLLER_ENABLE_ARCH_GEN_TEST "Build rocRoller architecture generator test" ON)
option(ROCROLLER_ENABLE_TEST_DISCOVERY "Use gtest and catch2 test discovery functions." ON)
option(ROCROLLER_ENABLE_COVERAGE "Build code coverage." OFF)
cmake_dependent_option(ROCROLLER_ENABLE_SLOW_TESTS "Disable slow running tests." ON "NOT ROCROLLER_ENABLE_COVERAGE" OFF)
option(ROCROLLER_EMBED_ARCH_DEF "Embed msgpack architecture data in library." ON)
option(ROCROLLER_BUILD_SHARED_LIBS "Build rocRoller as a shared library." ON)
option(ROCROLLER_ENABLE_FETCH "Enable fetch content for dependencies if find_package fails." OFF)
option(ROCROLLER_ENABLE_LLD "Build rocroller functionality requiring LLD" OFF)
option(ROCROLLER_ENABLE_TIMERS "Enable rocRoller timer code." OFF)
option(ROCROLLER_ENABLE_CPPCHECK "Enable cppcheck." OFF)
option(ROCROLLER_ENABLE_ASAN "Enable addresss sanitizers." OFF)
option(ROCROLLER_ENABLE_PREGENERATED_ARCH_DEF "Use the pregenerates GPU architecture definition YAML file(s) in the repository." ON)
set(ROCROLLER_MRISAS_DIR "${CMAKE_CURRENT_BINARY_DIR}/GPUArchitectureGenerator/amd-mrisa" CACHE STRING "Path to directory containing MRISA XML files.")
set(MXDATAGENERATOR_GIT_TAG "3ac153eec4c1a1e0da7546abc9c0c4f54e180f43" CACHE STRING "mxDataGenerator tag/commit hash to checkout")
set(MXDATAGENERATOR_GIT_URL "https://github.com/ROCm/mxDataGenerator.git" CACHE STRING "Base Git URL to fetch mxDataGenerator from.")

if(ROCROLLER_ENABLE_TEST_DISCOVERY)
    enable_testing()
endif()

if(ROCROLLER_ENABLE_CPPCHECK)
    find_program(CMAKE_CXX_CPPCHECK NAMES cppcheck)
    if(CMAKE_CXX_CPPCHECK)
        list(
            APPEND
            CMAKE_CXX_CPPCHECK
            "--enable=warning"
            "--enable=performance"
            "--enable=portability"
            "--enable=style"
            "--error-exitcode=10"
            "--force"
            "--inline-suppr"
            "--library=googletest"
            "--suppressions-list=${PROJECT_SOURCE_DIR}/CppCheckSuppressions.txt"
        )
    endif()
else()
    unset(CMAKE_CXX_CPPCHECK CACHE)
endif()

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

if(ROCROLLER_BUILD_SHARED_LIBS)
    add_library(rocroller SHARED)
else()
    add_library(rocroller STATIC)
    set_target_properties(rocroller PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
add_library(roc::rocroller ALIAS rocroller)

add_library(rocroller-interface INTERFACE)
add_library(roc::rocroller-interface ALIAS rocroller-interface)

if(ROCROLLER_ENABLE_LLD)
    find_package(LLD REQUIRED)
    target_include_directories(rocroller PRIVATE "${LLD_INCLUDES}")
    target_compile_definitions(rocroller PUBLIC ROCROLLER_USE_LLD)
endif()

if(ROCROLLER_ENABLE_ASAN AND NOT WIN32)
    rocroller_target_configure_sanitizers(rocroller PRIVATE)
    set(ASAN_LIB_PATH "unset")
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} --print-file-name=libclang_rt.asan-x86_64.so
        OUTPUT_VARIABLE ASAN_LIB_PATH
    )
    # If ASAN_LIB_PATH is libclang_rt.asan-x86_64.so
    # rather than /path/to/libclang_rt.asan-x86_64.so then
    # we failed to locate it and HAS_PARENT_PATH is false.
    string(STRIP ${ASAN_LIB_PATH} ASAN_LIB_PATH)
    cmake_path(HAS_PARENT_PATH ASAN_LIB_PATH result)
    if(NOT result)
        message(FATAL_ERROR "Failed to locate libclang_rt.asan-x86_64.so ")
    endif()
    set(asan_opts "LD_PRELOAD=${ASAN_LIB_PATH};ASAN_OPTIONS=detect_leaks=0")
endif()

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
    target_include_directories(rocroller PUBLIC "${LLVM_INCLUDE_DIRS}")
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

target_include_directories(rocroller PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include>)

target_compile_features(rocroller PUBLIC cxx_std_20)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/lib/source/Utilities/Version.cpp.in"
    "${PROJECT_BINARY_DIR}/rocRoller/src/Version.cpp"
)
target_sources(rocroller PRIVATE "${PROJECT_BINARY_DIR}/rocRoller/src/Version.cpp")

add_subdirectory(lib/source)
add_subdirectory(lib/include)
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

if(ROCROLLER_ENABLE_DOCS)
    add_subdirectory(docs)
endif()