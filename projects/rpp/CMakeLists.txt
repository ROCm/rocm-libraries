#[[
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

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
cmake_minimum_required(VERSION 3.10)
include( utils.cmake )
set( CORE_TARGET "rpp" )
set( BUILD_ENABLE_LINTIAN_OVERRIDES OFF CACHE BOOL "Enable/Disable Lintian Overrides" )
set( BUILD_DEBIAN_PKGING_FLAG OFF CACHE BOOL "Internal Status Flag to indicate Debian Packaging Build" )
option(DISABLE_LINTIAN "Disable Lintian" ON)

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
    message("-- rpp: INFO - ROCM_PATH Set -- ${ROCM_PATH}")
else()
    set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()

# Check if lib/rocm_sysdeps/lib exists in the ROCm path which indicates ROCm installation via TheRock
set(USING_THE_ROCK OFF)
if(EXISTS "${ROCM_PATH}/lib/rocm_sysdeps/lib")
  set(USING_THE_ROCK ON)
endif()

if(USING_THE_ROCK)
  if(NOT DEFINED ENV{HIP_DEVICE_LIB_PATH})
    set(ENV{HIP_DEVICE_LIB_PATH} ${ROCM_PATH}/lib/llvm/amdgcn/bitcode)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "HIP_DEVICE_LIB_PATH=${ROCM_PATH}/lib/llvm/amdgcn/bitcode")
  endif()
endif()

# Set AMD Clang as default compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/lib/llvm/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++)
elseif(NOT DEFINED CMAKE_CXX_COMPILER AND NOT EXISTS "${ROCM_PATH}/lib/llvm/bin/amdclang++")
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)
endif()

# RPP Version
# NOTE: package version and rpp_version.h is generated with this version
set(VERSION "3.1.2")

# Set Project Version and Language
project(rpp VERSION ${VERSION} LANGUAGES CXX)
# Set RPP source directory
set(RPP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")
include(GNUInstallDirs)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

find_program(MAKE_NSIS_EXE makensis)
find_program(RPMBUILD_EXE rpmbuild)
find_program(DPKG_EXE dpkg)

# Avoid setting the default path to /usr/local
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${ROCM_PATH} CACHE PATH "RPP default ROCm installation path" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# RPP Default Options
set(DEFAULT_BUILD_TYPE "Release")

### RPP_AUDIO_SUPPORT - default = ON, NOTE: support currently only on Ubuntu - user to set to OFF otherwise
option(RPP_AUDIO_SUPPORT      "Build RPP with Audio Support"         ON)

# Set message options
if(NOT WIN32)
    string(ASCII 27 Esc)
    set(ColourReset "${Esc}[m")
    set(Red         "${Esc}[31m")
    set(Green       "${Esc}[32m")
    set(Yellow      "${Esc}[33m")
    set(Blue        "${Esc}[34m")
    set(BoldBlue    "${Esc}[1;34m")
    set(Magenta     "${Esc}[35m")
    set(Cyan        "${Esc}[36m")
    set(White       "${Esc}[37m")
endif()

# Set backend options
if(APPLE)
    set(CMAKE_MACOSX_RPATH 1)
    set(BACKEND "CPU")
    set(RPP_AUDIO_SUPPORT OFF)
    message("-- ${Magenta}Apple macOS Detected -- GPU Support turned OFF${ColourReset}")
endif()

if(NOT DEFINED BACKEND)
    set(BACKEND "HIP")   # set default backend to HIP
else()
    if("${BACKEND}" STREQUAL "HIP" OR "${BACKEND}" STREQUAL "hip")
        set(BACKEND "HIP")
    elseif("${BACKEND}" STREQUAL "host" OR "${BACKEND}" STREQUAL "HOST" OR "${BACKEND}" STREQUAL "CPU" OR "${BACKEND}" STREQUAL "cpu")
        set(BACKEND "CPU")
    else()
        message("-- ${Yellow}Warning: ${PROJECT_NAME} backend option: ${BACKEND} -- unknown${ColourReset}")
        message("-- ${Yellow}Warning: ${PROJECT_NAME} default backend will be enforced${ColourReset}")
        set(BACKEND "HIP")
    endif()
endif()

message("-- ${BoldBlue}AMD ROCm Performance Primitives (RPP) Version -- ${VERSION}${ColourReset}")
message("-- ${BoldBlue}AMD RPP install path set to -- ${CMAKE_INSTALL_PREFIX}${ColourReset}")
message("-- ${BoldBlue}AMD RPP Backend set to -- ${BACKEND}${ColourReset}")

# RPP Build Type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "RPP Default Build Type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()
if(CMAKE_BUILD_TYPE MATCHES Debug)
    # -O0  -- Don't Optimize output file
    # -g -- generate debugging information
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
else()
    # -O3 -- Optimize output file
    # TBD:Verify Requirement -Ofast - Optimize for speed disregarding exact standards compliance
    # -DNDEBUG -- turn off asserts
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Ofast -DNDEBUG")
endif()

# Enable -Wall warnings for both release and debug builds
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Clang-specific warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # These warnings are suppressed since we are always using a Clang compiler, which supports VLA.
    # TBD: Determine if we must remove this flag later for standard compliance.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-vla-cxx-extension")
endif()

message("-- ${BoldBlue}AMD RPP Build Type -- ${CMAKE_BUILD_TYPE}${ColourReset}")

if(WIN32 AND CMAKE_CROSSCOMPILING)
    set(WINE_CMD "wine")
else()
    set(WINE_CMD)
endif()

# CMake checks and setups
if("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "${Red}${PROJECT_NAME} -- In-source builds are not allowed. Invoke cmake from build directory${ColourReset}")
endif()
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Half (HALF::half carries INTERFACE include dirs for transitive use via rpp::rpp)
find_package(HALF REQUIRED)

if (RPP_AUDIO_SUPPORT)
    find_package(FFTS QUIET)
    if(FFTS_FOUND)
        add_definitions(-DAUDIO_SUPPORT)        # For compile flags in RPP
        set(RPP_AUDIO_AUGMENTATIONS_SUPPORT 1)  # For cmakedefine01 in rpp_audio_augmentations_support.h.in
        message("-- ${Green}${PROJECT_NAME} set to build with RPP_AUDIO_SUPPORT${ColourReset}")
    else()
        set(RPP_AUDIO_SUPPORT OFF)              # Disable audio support if FFTS not found
        message("-- ${Yellow}FFTS not found. ${PROJECT_NAME} set to build without RPP_AUDIO_SUPPORT${ColourReset}")
    endif()
else()
    message("-- ${Yellow}${PROJECT_NAME} set to build without RPP_AUDIO_SUPPORT${ColourReset}")
endif()

message("-- ${Cyan}RPP Developer Options${ColourReset}")
message("-- ${Cyan}     -D BACKEND=${BACKEND} [Select RPP Backend [options:CPU/HIP](default:HIP)]${ColourReset}")
message("-- ${Cyan}     -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} [Select RPP build type [options:Debug/Release](default:Release)]${ColourReset}")
message("-- ${Cyan}     -D RPP_AUDIO_SUPPORT=${RPP_AUDIO_SUPPORT} [Select RPP audio support [options:ON/OFF](default:ON)]${ColourReset}")

# OpenMP
find_package(OpenMP COMPONENTS CXX REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# STD Filesystem (private to rpp; not part of the link interface)
set(RPP_PRIVATE_LINK_LIBS "")
if(NOT APPLE)
    list(APPEND RPP_PRIVATE_LINK_LIBS stdc++fs)
endif()

# set global compiler flags for the project
# -fPIC -- Generate position-independent code if possible.
# -mavx2 -- Support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX and AVX2 built-in functions and code generation.
# -mfma -- Support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX and FMA built-in functions and code generation.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -mavx2 -mf16c -mfma")

# Check AVX/SSE support
include(CheckCXXCompilerFlag)

# Check AVX flag
check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
if(COMPILER_SUPPORTS_AVX)
    add_definitions(-D__AVX2__)
else()
    # Check SSE flag
    check_cxx_compiler_flag("-msse4" COMPILER_SUPPORTS_SSE)
    if(COMPILER_SUPPORTS_SSE)
        add_definitions(-D__SSE4_1__)
    endif()
endif()

#Generate version info
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/api/rpp_version.h.in ${CMAKE_CURRENT_BINARY_DIR}/rpp_version.h @ONLY)

set(RPP_HIP_HOST_TARGET "")

# Backend specific settings
if("${BACKEND}" STREQUAL "HIP")
    # TBD: to be removed once STATIC becomes the only option within HIP backend, and all STATIC compile time switches are removed.
    add_definitions(-DSTATIC)

    if(NOT DEFINED HIP_PATH)
        if(NOT DEFINED ENV{HIP_PATH})
            set(HIP_PATH ${ROCM_PATH} CACHE PATH "Path to which HIP has been installed")
        else()
            set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
        endif()
    endif()

    # Set supported GPU Targets
    if(NOT GPU_TARGETS AND NOT AMDGPU_TARGETS)
        find_package(ROCmCMakeBuildTools QUIET)
        if(NOT ROCmCMakeBuildTools_FOUND)
            find_package(ROCM QUIET)
        endif()
        include(ROCMCheckTargetIds OPTIONAL RESULT_VARIABLE HAS_ROCM_CHECK_TARGET_IDS)

        set(OPTIONAL_GPU_TARGETS "gfx1250")
        if(HAS_ROCM_CHECK_TARGET_IDS)
            rocm_check_target_ids(OPTIONAL_GPU_TARGETS_AVAILABLE TARGETS ${OPTIONAL_GPU_TARGETS})
        else() # if we don't have rocm_check_target_ids, just assume the targets are available
            set(OPTIONAL_GPU_TARGETS_AVAILABLE "${OPTIONAL_GPU_TARGETS}")
        endif()
        set(DEFAULT_GPU_TARGETS "gfx908;gfx90a;gfx942;gfx1030;gfx1031;gfx1032;gfx1100;gfx1101;gfx1151;gfx1102;gfx950;gfx1200;gfx1201;${OPTIONAL_GPU_TARGETS_AVAILABLE}")
    endif()

    # Set AMD GPU_TARGETS
    if((AMDGPU_TARGETS OR DEFINED ENV{AMDGPU_TARGETS}) AND (NOT GPU_TARGETS))
        message("-- ${Red}${PROJECT_NAME} DEPRECATION -- AMDGPU_TARGETS use is deprecated. Use GPU_TARGETS${ColourReset}")
        if(DEFINED ENV{AMDGPU_TARGETS})
            set(GPU_TARGETS $ENV{AMDGPU_TARGETS} CACHE STRING "List of specific machine types for library to target")
        else()
            set(GPU_TARGETS ${AMDGPU_TARGETS})
        endif()
    endif()

    if(DEFINED ENV{GPU_ARCHS})
        set(GPU_TARGETS $ENV{GPU_ARCHS} CACHE STRING "List of specific machine types for library to target")
    elseif(GPU_TARGETS)
        message("-- ${White}${PROJECT_NAME} -- GPU_TARGETS set with -D option${ColourReset}")
    else()
        set(GPU_TARGETS "${DEFAULT_GPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")
    endif()
    message("-- ${White}${PROJECT_NAME} -- AMD GPU_TARGETS: ${GPU_TARGETS}${ColourReset}")

    set (HIP_PLATFORM amd CACHE STRING "HIP platform")
    find_package(HIP QUIET)
    if(HIP_FOUND)
        message("-- ${White}${PROJECT_NAME} -- Using HIP - Path:" ${HIP_PATH} "\tVersion:" ${HIP_VERSION} "\tCompiler:" ${HIP_COMPILER}${ColourReset})
        # TBD: Set external flag to notify GPU_SUPPORT and HIP backend availability - Compatible with rocAL and MIVisionX
        add_definitions(-DGPU_SUPPORT)
        # For cmakedefine01 in rpp_backend.h.in
        set(RPP_BACKEND_HIP 1)
        # To use RPP_BACKEND_HIP
        add_definitions(-DRPP_BACKEND_HIP=${RPP_BACKEND_HIP})
        link_directories(${HIP_PATH}/${CMAKE_INSTALL_LIBDIR})
        # hip::host supplies HIP/ROCm host headers and is linked PUBLIC on rpp::rpp for transitive usage
        set(RPP_HIP_HOST_TARGET hip::host)

        # Add modules
        set(MODULES_LIST modules)
        foreach(MOD_NAME ${MODULES_LIST})
            add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/${MOD_NAME})
        endforeach(MOD_NAME)

        message("-- ${Green}${PROJECT_NAME} set to build with HIP backend${ColourReset}")
    else()
        set(BACKEND "CPU")
        message("-- ${Yellow}HIP not found! ${PROJECT_NAME} set to build with CPU backend${ColourReset}")
    endif()

endif()
# CPU backend specific settings
if("${BACKEND}" STREQUAL "CPU")
    # Add modules
    set(MODULES_LIST modules)
    foreach(MOD_NAME ${MODULES_LIST})
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/${MOD_NAME})
    endforeach(MOD_NAME)
endif()

configure_file("${PROJECT_SOURCE_DIR}/api/rpp_backend.h.in" "${PROJECT_BINARY_DIR}/api/rpp_backend.h")
configure_file("${PROJECT_SOURCE_DIR}/api/rpp_audio_augmentations_support.h.in" "${PROJECT_BINARY_DIR}/api/rpp_audio_augmentations_support.h")

# Enable SIMD for HOST code (even in HIP backends)
if(NOT DEFINED SIMD_ENABLE)
    SET(SIMD_ENABLE "1")
endif()

add_library(${PROJECT_NAME} SHARED api/rpp.h $<TARGET_OBJECTS:modules>)
if("${SIMD_ENABLE}" STREQUAL "1")
    target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_SIMD_INTRINSICS=1)
    message("-- ${Green}${PROJECT_NAME} set to build with ENABLE_SIMD_INTRINSICS${ColourReset}")
else()
    target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_SIMD_INTRINSICS=0)
    message("-- ${Red}${PROJECT_NAME} set to build without ENABLE_SIMD_INTRINSICS${ColourReset}")
endif()

message("-- ${White}${PROJECT_NAME} -- Using Compiler - Path:" ${CMAKE_CXX_COMPILER} "\tVersion:" ${CMAKE_CXX_COMPILER_VERSION} "\tCompiler:" ${CMAKE_CXX_COMPILER_ID}${ColourReset})
message("-- ${White}${PROJECT_NAME} -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}${ColourReset}")
target_link_libraries(${PROJECT_NAME} PRIVATE
    OpenMP::OpenMP_CXX
    HALF::half
)
if(RPP_HIP_HOST_TARGET)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${RPP_HIP_HOST_TARGET})
endif()
if(RPP_PRIVATE_LINK_LIBS)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${RPP_PRIVATE_LINK_LIBS})
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

if(USING_THE_ROCK)
    set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN;$ORIGIN/rocm_sysdeps/lib" BUILD_WITH_INSTALL_RPATH TRUE)
endif()

if(RPP_AUDIO_SUPPORT)
    target_link_libraries(${PROJECT_NAME} PRIVATE FFTS::FFTS)
    message("-- ${Green}${PROJECT_NAME} set to build with RPP_AUDIO_SUPPORT${ColourReset}")
else()
    message("-- ${Yellow}${PROJECT_NAME} set to build without RPP_AUDIO_SUPPORT${ColourReset}")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})

# Public headers: build tree vs install tree (GNUInstallDirs paths are relative to the install prefix).
target_include_directories(${PROJECT_NAME}
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/api>"
        "${ROCM_PATH}/include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/include/cpu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/include/common"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/include/func_specific"
        "${INCLUDE_LIST}"
)

if( "${BACKEND}" STREQUAL "HIP")
    target_include_directories(${PROJECT_NAME} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/include/hip"
    )
endif()

add_subdirectory(addkernels)

# Test suite
enable_testing()
include(CTest)
add_subdirectory(utilities/test_suite)

# Installation
# install RPP libs -- {ROCM_PATH}/lib
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime NAMELINK_COMPONENT dev)
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)

# install RPP include files -- {ROCM_PATH}/include/rpp
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/api/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
        COMPONENT dev
        PATTERN "*.h.in" EXCLUDE)
install(FILES ${PROJECT_BINARY_DIR}/api/rpp_backend.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
        COMPONENT dev)
install(FILES ${PROJECT_BINARY_DIR}/api/rpp_audio_augmentations_support.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
        COMPONENT dev)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/rpp_version.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
        COMPONENT dev)

# install Test
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/test COMPONENT test)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/utilities/test_suite/ DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/test COMPONENT test)

# set license information
set(CPACK_RESOURCE_FILE_LICENSE  "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
install(FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION ${CMAKE_INSTALL_DOCDIR} COMPONENT runtime)
install(FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION ${CMAKE_INSTALL_DOCDIR}-asan COMPONENT asan)
if(RPP_AUDIO_SUPPORT AND FFTS_FOUND)
    set(CPACK_RESOURCE_FILE_FFTS_LICENSE  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ffts/COPYRIGHT")
    if(EXISTS ${CPACK_RESOURCE_FILE_FFTS_LICENSE})
        install(FILES ${CPACK_RESOURCE_FILE_FFTS_LICENSE} DESTINATION ${CMAKE_INSTALL_DOCDIR} COMPONENT runtime)
        install(FILES ${CPACK_RESOURCE_FILE_FFTS_LICENSE} DESTINATION ${CMAKE_INSTALL_DOCDIR}-asan COMPONENT asan)
    endif()
endif()

# Cmake module config file configurations
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/" CACHE INTERNAL "Default module path.")

set(CONF_LIB_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")

# Create the rpp-config.cmake and rpp-config-version files
include(CMakePackageConfigHelpers)
set(CONFIG_PACKAGE_INSTALL_DIR ${CONF_LIB_DIR}/cmake/${PROJECT_NAME})
set(LIB_INSTALL_DIR  "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
set(INCLUDE_INSTALL_DIR  "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")
set(BIN_INSTALL_DIR  "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}")

# Backend string for package config (HIP or CPU after HIP fallback)
set(RPP_PACKAGE_BACKEND_TYPE "${BACKEND}")
# Audio augmentations (same effective flag as RPP_AUDIO_SUPPORT / rpp_audio_augmentations_support.h)
set(RPP_PACKAGE_AUDIO_AUGMENTATIONS_SUPPORT "${RPP_AUDIO_SUPPORT}")

# Generate Config File for rpp
configure_package_config_file(
    ${CMAKE_MODULE_PATH}/${PROJECT_NAME}-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
    INSTALL_DESTINATION ${CONFIG_PACKAGE_INSTALL_DIR}
    PATH_VARS LIB_INSTALL_DIR INCLUDE_INSTALL_DIR BIN_INSTALL_DIR
)

# Generate config Version File for rpp
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
    VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
    COMPATIBILITY SameMajorVersion
)

# Install the rpp-config.cmake and rpp-config-version.cmake
install(FILES
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
    COMPONENT dev
)

# Install the export set for use with the install-tree
install(EXPORT ${PROJECT_NAME}-targets
    FILE ${PROJECT_NAME}-targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
    COMPONENT dev
)

# Set package information
set(PKG_MAINTAINER_NM  "mivisionx support")
set(PKG_MAINTAINER_EMAIL  "mivisionx.support@amd.com")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
set(CPACK_PACKAGE_NAME          "rpp")
set(CPACK_PACKAGE_LICENSE       "MIT")
set(CPACK_PACKAGE_CONTACT "${PKG_MAINTAINER_NM} <${PKG_MAINTAINER_EMAIL}>")
set(CPACK_PACKAGE_VENDOR        "AMD ROCm")
set(CPACK_PACKAGE_GROUP         "Development/Tools")
set(CPACK_PACKAGE_HOMEPAGE      "https://github.com/ROCm/rpp")

if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
    set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.$ENV{ROCM_LIBPATCH_VERSION}")
endif()

# Set the dependent packages
set(RPP_DEBIAN_PACKAGE_LIST  "hip-runtime-amd, openmp-extras-runtime")
set(RPP_RPM_PACKAGE_LIST     "hip-runtime-amd, openmp-extras-runtime")
set(RPP_DEBIAN_DEV_PACKAGE_LIST  "hip-dev, openmp-extras-dev, half")
set(RPP_RPM_DEV_PACKAGE_LIST     "hip-devel, openmp-extras-devel, half")
set(RPP_DEBIAN_TEST_PACKAGE_LIST  "cmake, python3-dev, python3-pip, python3-pandas, libopencv-dev, libsndfile1-dev")
set(RPP_RPM_TEST_PACKAGE_LIST     "cmake, python3-devel, python3-pip") # TBD: pandas, OpenCV & libsnd packages missing on RPM

# Add OS specific dependencies
if(EXISTS "/etc/os-release")
    file(READ "/etc/os-release" OS_RELEASE)
    string(REGEX MATCH "22.04" UBUNTU_22_FOUND ${OS_RELEASE})
    if(UBUNTU_22_FOUND)
        set(RPP_DEBIAN_DEV_PACKAGE_LIST "${RPP_DEBIAN_DEV_PACKAGE_LIST}, libstdc++-12-dev")
    endif()
endif()

# package release
set(CPACK_DEBIAN_FILE_NAME       "DEB-DEFAULT")
set(CPACK_RPM_FILE_NAME          "RPM-DEFAULT")
set(CPACK_DEBIAN_PACKAGE_RELEASE "local")
set(CPACK_RPM_PACKAGE_RELEASE    "local")
if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
    set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
endif()
if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
    set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
endif()

# '%{?dist}' breaks manual builds on debian systems due to empty Provides
execute_process(COMMAND rpm --eval %{?dist}
                RESULT_VARIABLE PROC_RESULT
                OUTPUT_VARIABLE EVAL_RESULT
                OUTPUT_STRIP_TRAILING_WHITESPACE)
if(PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "")
    string (APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}")
endif()

set(CPACK_PACKAGING_INSTALL_PREFIX  ${CMAKE_INSTALL_PREFIX})
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE   ${CPACK_PACKAGE_HOMEPAGE})
set(CPACK_RPM_PACKAGE_URL           ${CPACK_PACKAGE_HOMEPAGE})
set(CPACK_RPM_PACKAGE_AUTOREQPROV   "no")

# Debian package
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_DEBIAN_RUNTIME_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS "rocm-core, ${RPP_DEBIAN_PACKAGE_LIST}")
set(CPACK_DEBIAN_DEV_PACKAGE_NAME "${PROJECT_NAME}-dev")
set(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS "rocm-core, ${PROJECT_NAME}, ${RPP_DEBIAN_DEV_PACKAGE_LIST}")
set(CPACK_DEBIAN_TEST_PACKAGE_NAME "${PROJECT_NAME}-test")
set(CPACK_DEBIAN_TEST_PACKAGE_DEPENDS "rocm-core, ${PROJECT_NAME}-dev, ${RPP_DEBIAN_TEST_PACKAGE_LIST}")
set(CPACK_DEBIAN_PACKAGE_LICENSE "MIT" )
# Debian package specific variable for ASAN
set(CPACK_DEBIAN_ASAN_PACKAGE_NAME "${PROJECT_NAME}-asan" )
set(CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS "rocm-core-asan" )
# RPM package
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_RPM_RUNTIME_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_RPM_RUNTIME_PACKAGE_REQUIRES "rocm-core, ${RPP_RPM_PACKAGE_LIST}")
set(CPACK_RPM_DEV_PACKAGE_NAME "${PROJECT_NAME}-devel")
set(CPACK_RPM_DEV_PACKAGE_REQUIRES "rocm-core, ${PROJECT_NAME}, ${RPP_RPM_DEV_PACKAGE_LIST}")
set(CPACK_RPM_TEST_PACKAGE_NAME "${PROJECT_NAME}-test")
set(CPACK_RPM_TEST_PACKAGE_REQUIRES "rocm-core, ${PROJECT_NAME}-devel, ${RPP_RPM_TEST_PACKAGE_LIST}")
set(CPACK_RPM_PACKAGE_LICENSE "MIT" )
# RPM package specific variable for ASAN
set(CPACK_RPM_ASAN_PACKAGE_NAME "${PROJECT_NAME}-asan" )
set(CPACK_RPM_ASAN_PACKAGE_REQUIRES "rocm-core-asan" )

if(NOT ROCM_DEP_ROCMCORE)
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_RPM_RUNTIME_PACKAGE_REQUIRES ${CPACK_RPM_RUNTIME_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core-asan," "" CPACK_RPM_ASAN_PACKAGE_REQUIRES ${CPACK_RPM_ASAN_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_RPM_DEV_PACKAGE_REQUIRES ${CPACK_RPM_DEV_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_RPM_TEST_PACKAGE_REQUIRES ${CPACK_RPM_TEST_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS ${CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core-asan," "" CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_DEBIAN_DEV_PACKAGE_DEPENDS ${CPACK_DEBIAN_DEV_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core," "" CPACK_DEBIAN_TEST_PACKAGE_DEPENDS ${CPACK_DEBIAN_TEST_PACKAGE_DEPENDS})
endif()

if(ENABLE_ASAN_PACKAGING)
    # ASAN Package requires asan component with only libraries and license file
    set(CPACK_COMPONENTS_ALL asan)
    set( COMP_TYPE "asan" )
else()
    set(CPACK_COMPONENTS_ALL runtime dev test)
    set( COMP_TYPE "runtime" )
endif()

if(APPLE)
    set(CPACK_GENERATOR "Bundle")
    set(CPACK_BUNDLE_NAME ${CPACK_PACKAGE_NAME})
    set(CPACK_BUNDLE_ICON "${CMAKE_CURRENT_SOURCE_DIR}/.github/rpp.png")
    set(CPACK_BUNDLE_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/.github/info.plist")
else()
    set(CPACK_GENERATOR "TGZ;ZIP")
    if(EXISTS ${MAKE_NSIS_EXE})
        list(APPEND CPACK_GENERATOR "NSIS")
    endif()
    if(EXISTS ${RPMBUILD_EXE})
        list(APPEND CPACK_GENERATOR "RPM")
        message("-- ${White}AMD ROCm RPP RunTime Package -- ${CPACK_RPM_DEV_PACKAGE_REQUIRES}${ColourReset}")
        message("-- ${White}AMD ROCm RPP Dev Package -- ${CPACK_RPM_DEV_PACKAGE_DEPENDS}${ColourReset}")
        message("-- ${White}AMD ROCm RPP Test Package -- ${CPACK_RPM_TEST_PACKAGE_DEPENDS}${ColourReset}")
    endif()
    if(EXISTS ${DPKG_EXE})
        list(APPEND CPACK_GENERATOR "DEB")
        message("-- ${White}AMD ROCm RPP RunTime Package -- ${CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS}${ColourReset}")
        message("-- ${White}AMD ROCm RPP Dev Package -- ${CPACK_DEBIAN_DEV_PACKAGE_DEPENDS}${ColourReset}")
        message("-- ${White}AMD ROCm RPP Test Package -- ${CPACK_DEBIAN_TEST_PACKAGE_DEPENDS}${ColourReset}")
    endif()
endif()

if(NOT DISABLE_LINTIAN)
    configure__pkg( ${CORE_TARGET} ${COMP_TYPE} ${CPACK_PACKAGE_VERSION} ${PKG_MAINTAINER_NM} ${PKG_MAINTAINER_EMAIL} )
endif()

include(CPack)

cpack_add_component(runtime
    DISPLAY_NAME "RPP Runtime Package"
    DESCRIPTION "Computer vision library for AMD CPUs and GPUs with HOST/HIP back-ends")

cpack_add_component(dev
    DISPLAY_NAME "RPP Develop Package"
    DESCRIPTION "Computer vision library for AMD CPUs and GPUs with HOST/HIP back-ends")

cpack_add_component(test
    DISPLAY_NAME "RPP Test Package"
    DESCRIPTION "ROCm Performance Primitives library is a comprehensive high performance \
computer vision library for AMD CPUs and GPUs with HOST/HIP back-ends. \
RPP test package provides rpp test suite")

cpack_add_component(asan
    DISPLAY_NAME "RPP ASAN Package"
    DESCRIPTION "ROCm Performance Primitives library is a comprehensive high performance \
computer vision library for AMD CPUs and GPUs with HOST/HIP back-ends. \
RPP ASAN package provides rpp ASAN libraries")
