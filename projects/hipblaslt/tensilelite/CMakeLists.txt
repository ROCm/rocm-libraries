# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.24.4)
project(tensilelite LANGUAGES CXX)

option(TENSILELITE_ENABLE_AUTOBUILD "Build the tensilelite host library tests." ON)
option(HIPBLASLT_ENABLE_HOST "Build hipBLASLt host library." OFF)
option(TENSILELITE_ENABLE_HOST "Build the tensilelite host library." OFF)
option(HIPBLASLT_ENABLE_MSGPACK "Use msgpack for parsing configuration files." ON)
option(HIPBLASLT_ENABLE_OPENMP "Use OpenMP to improve performance." ON)
option(HIPBLASLT_ENABLE_LLVM "Use msgpack for parsing configuration files." OFF)

if(TENSILELITE_ENABLE_AUTOBUILD)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    include(tensilelite_auto_build)
endif()

if(TENSILELITE_ENABLE_HOST OR HIPBLASLT_ENABLE_HOST)
    if(HIPBLASLT_ENABLE_OPENMP)
        find_package(OpenMP REQUIRED)
    endif()
    if(HIPBLASLT_ENABLE_MSGPACK)
        # See: https://github.com/msgpack/msgpack-c/wiki/Q%26A#how-to-support-both-msgpack-c-c-version-5x-and-6x-
        # Prefer 6.x (msgpack-cxx)
        find_package(msgpack-cxx CONFIG)
        if(msgpack-cxx_FOUND)
            message(STATUS "Found msgpack-cxx (>=6.x)")
        else()
            # TODO: set minimum version requied for *-cxx library
            find_package(msgpackc-cxx CONFIG REQUIRED NAMES msgpackc-cxx msgpack)
            message(STATUS "Found msgpack (<=5.x)")
        endif()
    endif()
    if(HIPBLASLT_ENABLE_LLVM)
        find_package(LLVM REQUIRED)
    endif()

    find_package(hip REQUIRED)
    add_subdirectory(rocisa/rocisa-cpp)

    add_library(tensilelite-host OBJECT)
    add_library(tensilelite::tensilelite-host ALIAS tensilelite-host)

    set_target_properties(tensilelite-host PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_compile_features(tensilelite-host PUBLIC cxx_std_17)
    target_compile_definitions(tensilelite-host
        PUBLIC
            $<BUILD_INTERFACE:TENSILE_DEFAULT_SERIALIZATION>
            __HIP_PLATFORM_AMD__
    )

    target_compile_definitions(tensilelite-host PUBLIC $<BUILD_INTERFACE:TENSILE_USE_HIP>)

    if(HIPBLASLT_ENABLE_LLVM)
        target_link_libraries(tensilelite-host PRIVATE LLVMObjectYAML)
        target_include_directories(tensilelite-host PRIVATE ${LLVM_INCLUDE_DIRS})
        target_compile_definitions(tensilelite-host PUBLIC TENSILE_YAML)
    endif()

    if(HIPBLASLT_ENABLE_MSGPACK)
        if(msgpack-cxx_FOUND)
            target_link_libraries(tensilelite-host PRIVATE msgpack-cxx)
        else()
            target_link_libraries(tensilelite-host PRIVATE msgpackc-cxx)
        endif()
        target_compile_definitions(tensilelite-host PUBLIC $<BUILD_INTERFACE:TENSILE_MSGPACK>) # make TENSILELITE_MSGPACK
    endif()

    target_include_directories(tensilelite-host
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            "${CMAKE_CURRENT_BINARY_DIR}/include"
    )

    target_link_libraries(tensilelite-host 
        PUBLIC
            rocisa::rocisa-cpp
            hip::device
    )
    add_subdirectory(src)
    add_subdirectory(include)

    if(TENSILELITE_ENABLE_CLIENT)
        add_subdirectory(client)
    endif()

    if(TENSILELITE_BUILD_TESTING OR BUILD_TESTING)
        add_subdirectory(tests)
    endif()
endif()
