
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

cmake_minimum_required(VERSION 3.16...3.22)

set(ROCROLLER_VERSION "1.0.0")
set(ROCROLLER_SOVERSION 1)

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if(WIN32)
    set(CMAKE_INSTALL_PREFIX
        "${PROJECT_BINARY_DIR}/package"
        CACHE PATH "Install path prefix, prepended onto install directories"
    )
else()
    set(CMAKE_INSTALL_PREFIX
        "/opt/rocm"
        CACHE PATH "Install path prefix, prepended onto install directories"
    )
endif()

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if(NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE
        Release
        CACHE
            STRING
            "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
    )
endif()

project(
    rocRoller
    VERSION ${ROCROLLER_VERSION}
    LANGUAGES CXX
)
if(CMAKE_C_COMPILER)
    # If the compiler is specified but the C language isn't enabled,
    # LLVM pitches a fit, so for easy compatibility just enable the C language
    enable_language(C)
endif()

set(CMAKE_CXX_STANDARD 20)

# Append our library helper cmake path and the cmake path for hip (for convenience)
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list(
    APPEND
    CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    ${ROCM_PATH}/lib/cmake/hip
    /opt/rocm/lib/cmake/hip
    ${HIP_DIR}/cmake
)

# NOTE:  workaround until hip cmake modules fixes symlink logic in their config files; remove when fixed
list(
    APPEND
    CMAKE_PREFIX_PATH
    ${ROCM_PATH}/lib/cmake/hip
    /opt/rocm
    /opt/rocm/llvm
    /opt/rocm/hip
)

include(cmake/Dependencies.cmake)
rocroller_add_dependency(
    ROCmCMakeBuildTools
    VERSION 0.11.0
    FIND_PACKAGE_ARGS NO_DEFAULT_PATH # PATHS "${ROCM_PATH}"
)

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMClients)

rocm_setup_version(VERSION ${ROCROLLER_VERSION})

option(ROCROLLER_ENABLE_TIMERS "Enable rocRoller timer code." OFF)
option(ROCROLLER_USE_HIP "Use HIP within the rocRoller." ON)
option(ROCROLLER_EMBED_ARCH_DEF
       "Embed the generated GPU architecture msgpack in the library" ON
)
option(ROCROLLER_USE_PREGENERATED_ARCH_DEF
       "Use the pregenerates GPU architecture definition YAML file(s) in the repository." ON
)
option(SKIP_CPPCHECK "Skip CPPCHECK." ON)

option(CODE_ANALYSIS "When building, don't query for tests etc." OFF)
option(CODE_COVERAGE "Build with code coverage flags (clang only)" OFF)

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
option(BUILD_CLIENTS "Build performance testing clients" ON)
option(BUILD_TESTING "Build rocRoller test clients" ON)
option(BUILD_DOCS "Build rocRoller documentation" ON)
option(BUILD_VERBOSE "Output additional build information" OFF)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

set(YAML_BACKEND
    "YAML_CPP"
    CACHE STRING "YAML backend used for serialization"
)
set_property(CACHE YAML_BACKEND PROPERTY STRINGS YAML_CPP LLVM)

# For paths given when reporting errors
add_compile_options(-fmacro-prefix-map=${PROJECT_SOURCE_DIR}=.)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(COROUTINES_COMPILE_OPTION)
    set(EXTRA_COMPILE_OPTIONS)
    set(EXTRA_LINK_OPTIONS -fuse-ld=lld -frtti)
else()
    set(COROUTINES_COMPILE_OPTION -fcoroutines)
    set(EXTRA_COMPILE_OPTIONS)
    set(EXTRA_LINK_OPTIONS -frtti)
endif()

if(ROCROLLER_ENABLE_TIMERS)
    add_definitions(-DROCROLLER_ENABLE_TIMERS)
    add_compile_options(-pg -fno-omit-frame-pointer)
endif()

if(CODE_COVERAGE)
    list(PREPEND EXTRA_COMPILE_OPTIONS -fprofile-instr-generate
         -fcoverage-mapping
    )
    list(PREPEND EXTRA_LINK_OPTIONS -fprofile-instr-generate -fcoverage-mapping)
endif()

if(ROCROLLER_USE_HIP)
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip)
    find_package(hip)

    find_package(LLVM REQUIRED HINTS /opt/rocm/llvm/)

    if(YAML_BACKEND STREQUAL "LLVM")
        llvm_map_components_to_libnames(llvm_yaml_libs objectyaml)
    endif()

    llvm_map_components_to_libnames(llvm_libs mc support)
    find_package(
        LLD
        REQUIRED
        CONFIG
        HINTS
        ${LLVM_DIR}/../lld/
    )
endif()

# shows up in headers, system package required
rocroller_add_dependency(
    msgpack
    VERSION 3.3.0
    COMPONENTS devel
)

# does not show up in headers, no system package required
rocroller_add_dependency(spdlog VERSION 1.14.1)

if(YAML_BACKEND STREQUAL "YAML_CPP")
    # shows up in headers, system package required
    rocroller_add_dependency(
        yaml-cpp
        VERSION 0.7.0
        COMPONENTS devel runtime
    )
    add_definitions(-DROCROLLER_USE_YAML_CPP)
elseif(YAML_BACKEND STREQUAL "LLVM")
    add_definitions(-DROCROLLER_USE_LLVM)
endif()

if(SKIP_CPPCHECK)
    unset(CMAKE_CXX_CPPCHECK CACHE)
else()
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
endif()

# Git hash
execute_process(
    COMMAND git describe --always --dirty
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    OUTPUT_VARIABLE ROCROLLER_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(ROCROLLER_GIT_VERSION_CPP_FILE
    ${PROJECT_BINARY_DIR}/source/rocRoller/Utilities/Version.cpp
)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/source/Utilities/Version.cpp.in
    ${ROCROLLER_GIT_VERSION_CPP_FILE}
)

add_subdirectory(GPUArchitectureGenerator)
add_subdirectory(lib)

rocm_package_add_dependencies(COMPONENT runtime DEPENDS "libstdc++6 >= 13.1.0")
rocm_package_add_deb_dependencies(
    COMPONENT devel DEPENDS liboost-container1.74-dev
)
rocm_package_add_rpm_dependencies(
    COMPONENT devel DEPENDS liboost-container1.74-devel
)

if(BUILD_CLIENTS OR BUILD_TESTING)
    if(NOT CLIENTS_OS)
        rocm_set_os_id(CLIENTS_OS)
        string(TOLOWER "${CLIENTS_OS}" CLIENTS_OS)
        rocm_read_os_release(CLIENTS_OS_VERSION VERSION_ID)
    endif()
    message(STATUS "OS: ${CLIENTS_OS} ${CLIENTS_OS_VERSION}")
    set(OPENMP_DEB "libomp-dev")
    set(OPENMP_RPM "libgomp")

    rocm_package_setup_component(clients)
endif()

if(BUILD_TESTING)
    enable_testing()
    rocm_package_setup_client_component(
        tests
        DEPENDS DEB
                "${OPENMP_DEB}"
                "libopenblas-dev"
                RPM
                "${OPENMP_RPM}"
                "libopenblas-devel"
    )
    add_subdirectory(test)
endif()

if(BUILD_DOCS)
    # Add docs directory
    add_subdirectory(docs)
endif()

if(BUILD_CLIENTS)
    rocm_package_setup_client_component(
        benchmarks DEPENDS DEB "${OPENMP_DEB}" RPM "${OPENMP_RPM}"
    )
    # Add client directory
    add_subdirectory(client)
endif()

include(GNUInstallDirs)

if(WIN32)
    set(CPACK_SOURCE_GENERATOR "ZIP")
    set(CPACK_GENERATOR "ZIP")
else()
    set(CPACK_GENERATOR "DEB")
endif()

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

if(WIN32)
    set(CMAKE_INSTALL_PREFIX
        "C:/hipSDK"
        CACHE PATH "Install path" FORCE
    )
    set(INSTALL_PREFIX "C:/hipSDK")
    set(CPACK_SET_DESTDIR FALSE)
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "C:/hipSDK")
    set(CPACK_PACKAGING_INSTALL_PREFIX "")
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
else()
    if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
        set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
    endif()
    set(CPACK_SET_DESTDIR
        OFF
        CACHE
            BOOL
            "Boolean toggle to make CPack use DESTDIR mechanism when packaging"
    )
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
    "\${CPACK_PACKAGING_INSTALL_PREFIX}"
)

set(package_name rocroller)

rocm_package_add_deb_dependencies(STATIC_DEPENDS "rocm-llvm-dev >= 18.0.0")
rocm_package_add_rpm_dependencies(STATIC_DEPENDS "rocm-llvm-devel >= 18.0.0")

set(ROCROLLER_CONFIG_DIR
    "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
    CACHE PATH "Path placed into ldconfig file"
)

rocm_create_package(
    NAME ${package_name}
    DESCRIPTION "A library for generating AMDGPU kernels"
    MAINTAINER "rocRoller Maintainer <rocroller-maintainer@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${ROCROLLER_CONFIG_DIR}
)
