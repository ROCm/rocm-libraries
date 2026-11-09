# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.25.2)
project(rocm-arch VERSION 0.4.0 LANGUAGES NONE)

include(CMakePackageConfigHelpers)
# A header-only package needs no target compiler to configure/install.
set(CMAKE_INSTALL_INCLUDEDIR include CACHE PATH "Header installation directory")
set(CMAKE_INSTALL_DATADIR share CACHE PATH "Data installation directory")
find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter)

set(_arch_definitions "${CMAKE_CURRENT_SOURCE_DIR}/python/rocm_arch/architectures")
file(GLOB _arch_json CONFIGURE_DEPENDS "${_arch_definitions}/*.json")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${_arch_json}
    "${CMAKE_CURRENT_SOURCE_DIR}/python/rocm_arch/__init__.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate.py"
)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate.py"
            --output "${CMAKE_CURRENT_BINARY_DIR}/generated"
    COMMAND_ERROR_IS_FATAL ANY
)
include("${CMAKE_CURRENT_BINARY_DIR}/generated/rocm-arch-catalog.cmake")
include(cmake/rocm-arch-functions.cmake)
set_property(GLOBAL PROPERTY ROCM_ARCH_PYTHON_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python")

add_library(rocm-arch INTERFACE)
add_library(rocm::arch ALIAS rocm-arch)
set_target_properties(rocm-arch PROPERTIES EXPORT_NAME arch)
target_compile_features(rocm-arch INTERFACE cxx_std_14)
target_include_directories(rocm-arch INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

set(_arch_package_dir "${CMAKE_INSTALL_DATADIR}/rocm-arch/cmake")
configure_package_config_file(cmake/rocm-arch-config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/rocm-arch-config.cmake"
    INSTALL_DESTINATION "${_arch_package_dir}"
    PATH_VARS CMAKE_INSTALL_DATADIR
)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/rocm-arch-config-version.cmake"
    VERSION "${PROJECT_VERSION}" COMPATIBILITY SameMajorVersion ARCH_INDEPENDENT
)
install(TARGETS rocm-arch EXPORT rocm-arch-targets)
install(EXPORT rocm-arch-targets NAMESPACE rocm:: DESTINATION "${_arch_package_dir}")
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/generated/rocm-arch-catalog.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/rocm-arch-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/rocm-arch-config-version.cmake"
    cmake/rocm-arch-functions.cmake
    cmake/rocm-arch-dependency.cmake
    DESTINATION "${_arch_package_dir}"
)
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(FILES ${_arch_json} DESTINATION "${CMAKE_INSTALL_DATADIR}/rocm-arch/architectures")
install(DIRECTORY python/rocm_arch
    DESTINATION "${CMAKE_INSTALL_DATADIR}/rocm-arch/python"
    PATTERN "__pycache__" EXCLUDE
)

option(ROCM_ARCH_BUILD_TESTING "Build host-only prototype tests" ${PROJECT_IS_TOP_LEVEL})
option(ROCM_ARCH_BUILD_INTEGRATION_TESTING "Test FFT consumers in a full monorepo checkout" OFF)
if(ROCM_ARCH_BUILD_TESTING)
    enable_language(CXX)
    enable_testing()
    add_subdirectory(tests)
endif()
