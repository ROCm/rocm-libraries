# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.25.2)
    set(TENSILELITE_STANDALONE ON)
    cmake_path(SET ROCM_LIBRARIES_ROOT NORMALIZE "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
    include("${ROCM_LIBRARIES_ROOT}/cmake/modules/default_amdclang.cmake")
else()
    set(TENSILELITE_STANDALONE OFF)
endif()

set(TENSILELITE_PROJECT_VERSION "1.5.0" CACHE STRING "Semantic version string.")
project(tensilelite VERSION "${TENSILELITE_PROJECT_VERSION}" LANGUAGES CXX)
include(GNUInstallDirs)
include(CMakeDependentOption)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(TENSILELITE_STANDALONE)
    list(APPEND CMAKE_MODULE_PATH "${ROCM_LIBRARIES_ROOT}/cmake/modules")
    include(fetch_rocm_cmake)

    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        if(WIN32)
            set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path prefix." FORCE)
        else()
            set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix." FORCE)
        endif()
    endif()

    set(TENSILELITE_RUNTIME_COMPONENT runtime)
    set(TENSILELITE_DEVEL_COMPONENT devel)
else()
    set(TENSILELITE_RUNTIME_COMPONENT tensilelite)
    set(TENSILELITE_DEVEL_COMPONENT tensilelite-devel)
endif()

option(TENSILELITE_ENABLE_HOST "Build the TensileLite host library." ON)
option(TENSILELITE_BUILD_SHARED_LIBS "Build the TensileLite host library as shared vs static." ON)
option(TENSILELITE_ENABLE_YAML "Use YAML for parsing configuration files." OFF)
cmake_dependent_option(TENSILELITE_ENABLE_MXDATAGENERATOR
    "Use mxDataGenerator for pre-swizzled MX scale strides." ON "NOT WIN32" OFF)
set(BUILD_SHARED_LIBS "${TENSILELITE_BUILD_SHARED_LIBS}")

if(TENSILELITE_STANDALONE)
    if(TENSILELITE_ENABLE_CLIENT OR TENSILELITE_BUILD_TESTING OR TENSILELITE_ENABLE_AUTOBUILD OR BUILD_TESTING)
        message(FATAL_ERROR "The TensileLite client, tests, and autobuild scripts require the hipBLASLt build.")
    endif()

    find_package(hip REQUIRED)
    find_package(origami CONFIG REQUIRED)
    if(TENSILELITE_ENABLE_MXDATAGENERATOR)
        find_package(mxDataGenerator CONFIG REQUIRED)
    endif()
endif()

if(NOT TENSILELITE_ENABLE_YAML)
    find_package(msgpack-cxx CONFIG)
    if(msgpack-cxx_FOUND)
        message(STATUS "Found msgpack-cxx (>=6.x)")
        set(tensilelite_msgpack_target "msgpack-cxx")
    else()
        find_package(msgpackc-cxx CONFIG REQUIRED NAMES msgpackc-cxx msgpack)
        set(tensilelite_msgpack_target "msgpackc-cxx")
        message(STATUS "Found msgpack (<=5.x)")
    endif()
endif()

if(TENSILELITE_ENABLE_YAML AND TENSILELITE_STANDALONE)
    find_package(LLVM REQUIRED)
endif()

if(TENSILELITE_ENABLE_MXDATAGENERATOR AND NOT TARGET roc::mxDataGeneratorPreSwizzle)
    message(FATAL_ERROR "TENSILELITE_ENABLE_MXDATAGENERATOR requires roc::mxDataGeneratorPreSwizzle")
endif()

option(TENSILELITE_ENABLE_HOST_ASAN "Enable AddressSanitizer for host code" OFF)
option(TENSILELITE_ENABLE_HOST_TSAN "Enable ThreadSanitizer for host code" OFF)
option(TENSILELITE_ENABLE_COVERAGE "Enable code coverage instrumentation" OFF)
mark_as_advanced(TENSILELITE_ENABLE_HOST_ASAN)
mark_as_advanced(TENSILELITE_ENABLE_HOST_TSAN)
mark_as_advanced(TENSILELITE_ENABLE_COVERAGE)

if(TENSILELITE_ENABLE_HOST_ASAN AND TENSILELITE_ENABLE_HOST_TSAN)
    message(FATAL_ERROR "ASAN and TSAN cannot be enabled simultaneously")
endif()

add_subdirectory(rocisa)

if(TENSILELITE_ENABLE_AUTOBUILD)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    include(tensilelite_auto_build)
endif()

if(TENSILELITE_ENABLE_HOST)
    find_package(ZLIB REQUIRED)

    add_library(tensilelite-host)
    add_library(roc::tensilelite-host ALIAS tensilelite-host)

    set_target_properties(tensilelite-host
        PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN ON
    )
    rocm_set_soversion(tensilelite-host "1.0")

    if(BUILD_SHARED_LIBS AND NOT WIN32)
        target_link_options(tensilelite-host PRIVATE "LINKER:--exclude-libs,ALL")
    endif()

    target_compile_features(tensilelite-host PUBLIC cxx_std_17)

    include(GenerateExportHeader)
    generate_export_header(tensilelite-host
        BASE_NAME TENSILELITEHOST
        EXPORT_MACRO_NAME TENSILELITEHOST_EXPORT
        EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/tensilelitehost/export.h"
    )
    target_compile_definitions(tensilelite-host
        PUBLIC
            $<BUILD_INTERFACE:__HIP_HCC_COMPAT_MODE__=1> # if we don't have this we fail tests on 942_304cu
    )

    if(TENSILELITE_ENABLE_HOST_ASAN)
        # Apply sanitizer only to host code, not device code
        # Use SHELL: to pass flags that should not be applied to device offload
        target_compile_options(tensilelite-host PUBLIC
            $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:SHELL:-Xarch_host -fsanitize=address>>
            $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:SHELL:-Xarch_host -fno-omit-frame-pointer>>
        )
        target_link_options(tensilelite-host PUBLIC
            $<BUILD_INTERFACE:SHELL:-Xarch_host -fsanitize=address>
        )
    endif()

    if(TENSILELITE_ENABLE_HOST_TSAN)
        # Apply sanitizer only to host code, not device code
        # Use SHELL: to pass flags that should not be applied to device offload
        target_compile_options(tensilelite-host PUBLIC
            $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:SHELL:-Xarch_host -fsanitize=thread>>
        )
        target_link_options(tensilelite-host PUBLIC
            $<BUILD_INTERFACE:SHELL:-Xarch_host -fsanitize=thread>
        )
    endif()

    if(TENSILELITE_ENABLE_YAML)
        target_link_libraries(tensilelite-host PRIVATE LLVMObjectYAML)
        target_include_directories(tensilelite-host PRIVATE ${LLVM_INCLUDE_DIRS})
        target_compile_definitions(tensilelite-host PUBLIC $<BUILD_INTERFACE:TENSILE_YAML>)
    else()
        target_link_libraries(tensilelite-host PRIVATE ${tensilelite_msgpack_target})
        target_compile_definitions(tensilelite-host PUBLIC $<BUILD_INTERFACE:TENSILE_MSGPACK>)
    endif()

    target_include_directories(tensilelite-host
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

    target_link_libraries(tensilelite-host
        PUBLIC
            $<BUILD_INTERFACE:rocisa::rocisa-cpp>
            roc::origami
            $<BUILD_INTERFACE:hip::host>
        PRIVATE
            hip::device
            ZLIB::ZLIB
    )

    if(TENSILELITE_ENABLE_MXDATAGENERATOR)
        target_link_libraries(tensilelite-host
            PRIVATE $<BUILD_INTERFACE:roc::mxDataGeneratorPreSwizzle>)
        target_compile_definitions(tensilelite-host PRIVATE HIPBLASLT_ENABLE_MXDATAGENERATOR)
    endif()

    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES ${HIP_INCLUDE_DIRS})
    set(CMAKE_REQUIRED_LIBRARIES hip::device)
    check_cxx_source_compiles("
        #include <hip/hip_runtime.h>
        int main() {
            hipLaunchAttributeID id = hipLaunchAttributeClusterDimension;
            return 0;
        }
    " HIP_HAS_CLUSTER_LAUNCH)
    if(HIP_HAS_CLUSTER_LAUNCH)
        target_compile_definitions(tensilelite-host PRIVATE HIP_HAS_CLUSTER_LAUNCH)
    endif()

    if(TENSILELITE_ENABLE_COVERAGE)
        message(STATUS "Enabling C++ code coverage instrumentation for tensilelite-host")
        target_compile_options(tensilelite-host PRIVATE -fprofile-instr-generate -fcoverage-mapping)
        target_link_options(tensilelite-host PUBLIC $<BUILD_INTERFACE:-fprofile-instr-generate -fcoverage-mapping>)
    endif()

    add_subdirectory(src)
    add_subdirectory(include)

    install(TARGETS tensilelite-host
        EXPORT tensilelite-targets
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT ${TENSILELITE_RUNTIME_COMPONENT}
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT ${TENSILELITE_RUNTIME_COMPONENT}
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT ${TENSILELITE_DEVEL_COMPONENT}
        FILE_SET tensilelite_public_headers
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" COMPONENT ${TENSILELITE_DEVEL_COMPONENT}
        FILE_SET tensilelite_generated_headers
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" COMPONENT ${TENSILELITE_DEVEL_COMPONENT}
        FILE_SET rocisa_public_headers
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" COMPONENT ${TENSILELITE_DEVEL_COMPONENT}
    )

    set(_tensilelite_config_dir "${CMAKE_INSTALL_LIBDIR}/cmake/tensilelite")
    install(EXPORT tensilelite-targets
        NAMESPACE roc::
        DESTINATION "${_tensilelite_config_dir}"
        COMPONENT ${TENSILELITE_DEVEL_COMPONENT}
    )

    include(CMakePackageConfigHelpers)
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tensilelite-config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/tensilelite-config.cmake"
        INSTALL_DESTINATION "${_tensilelite_config_dir}"
    )
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/tensilelite-config-version.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY SameMajorVersion
    )
    install(
        FILES
            "${CMAKE_CURRENT_BINARY_DIR}/tensilelite-config.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/tensilelite-config-version.cmake"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipblaslt_codegen.cmake"
        DESTINATION "${_tensilelite_config_dir}"
        COMPONENT ${TENSILELITE_DEVEL_COMPONENT}
    )
    unset(_tensilelite_config_dir)

    if(TENSILELITE_BUILD_SHARED_LIBS AND NOT WIN32)
        set_property(TARGET tensilelite-host PROPERTY
            INSTALL_RPATH "$ORIGIN;$ORIGIN/../llvm/lib")
    endif()

    if(TENSILELITE_ENABLE_CLIENT)
        add_subdirectory(client)
    endif()

    if(NOT TENSILELITE_STANDALONE AND (TENSILELITE_BUILD_TESTING OR BUILD_TESTING))
        add_subdirectory(tests)
    endif()
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/tensilelite_packaging.cmake")
