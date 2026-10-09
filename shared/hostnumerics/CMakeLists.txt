# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Use the monorepo baseline, also used by hipBLASLt and rocRoller.
# CMake 3.25.2 was released on 2023-01-19.
cmake_minimum_required(VERSION 3.25.2)
project(HostNumerics VERSION 0.1.0 LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

option(HOSTNUMERICS_BUILD_TESTING "Build HostNumerics tests." ${PROJECT_IS_TOP_LEVEL})
option(HOSTNUMERICS_BUILD_PYTHON "Build the HostNumerics Python module."
       ${PROJECT_IS_TOP_LEVEL})

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/roc/hostnumerics/version.hpp"
    @ONLY
)

# hostnumerics-core will provide scalar types and conversions, plus the tensor
# model (shape, layout, and storage), independently of numerical operations.
# This bootstrap uses INTERFACE targets that only expose the version header;
# the prototype implements both targets as static libraries.
add_library(hostnumerics-core INTERFACE)
add_library(roc::hostnumerics-core ALIAS hostnumerics-core)
target_compile_features(hostnumerics-core INTERFACE cxx_std_20)
target_sources(hostnumerics-core
    INTERFACE
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/include"
        FILES "${CMAKE_CURRENT_BINARY_DIR}/include/roc/hostnumerics/version.hpp"
)

# hostnumerics will add input generation, reference operations, and comparison.
# Linking it also brings in core; consumers needing only the tensor/scalar model
# can link hostnumerics-core directly.
add_library(hostnumerics INTERFACE)
add_library(roc::hostnumerics ALIAS hostnumerics)
target_link_libraries(hostnumerics INTERFACE roc::hostnumerics-core)

if(HOSTNUMERICS_BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
if(HOSTNUMERICS_BUILD_PYTHON)
    add_subdirectory(python)
endif()

install(TARGETS hostnumerics-core hostnumerics
    EXPORT hostnumerics-targets
    FILE_SET HEADERS DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
set(_hostnumerics_cmake_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/HostNumerics")
install(EXPORT hostnumerics-targets
    FILE HostNumericsTargets.cmake
    DESTINATION "${_hostnumerics_cmake_install_dir}"
    NAMESPACE roc::
)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HostNumericsConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/HostNumericsConfig.cmake"
    INSTALL_DESTINATION "${_hostnumerics_cmake_install_dir}"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/HostNumericsConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/HostNumericsConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/HostNumericsConfigVersion.cmake"
    DESTINATION "${_hostnumerics_cmake_install_dir}"
)
install(FILES LICENSE.md
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/HostNumerics"
)
