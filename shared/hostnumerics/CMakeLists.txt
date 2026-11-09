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

# Scalar and layout APIs, including the generic physical-layout descriptor.
add_library(hostnumerics-core STATIC src/reshape_permutation.cpp)
add_library(roc::hostnumerics-core ALIAS hostnumerics-core)
target_compile_features(hostnumerics-core PUBLIC cxx_std_20)
target_sources(hostnumerics-core
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_BINARY_DIR}/include"
        FILES
            "${CMAKE_CURRENT_BINARY_DIR}/include/roc/hostnumerics/version.hpp"
            include/roc/hostnumerics/scalar_type.hpp
            include/roc/hostnumerics/integer_conversion.hpp
            include/roc/hostnumerics/scalar_storage.hpp
            include/roc/hostnumerics/detail/small_float.hpp
            include/roc/hostnumerics/permutation.hpp
            include/roc/hostnumerics/shape.hpp
            include/roc/hostnumerics/index_range.hpp
            include/roc/hostnumerics/sampling.hpp
            include/roc/hostnumerics/detail/prime.hpp
            include/roc/hostnumerics/expected.hpp
            include/roc/hostnumerics/layout.hpp
            include/roc/hostnumerics/reshape_permutation.hpp
)

set_target_properties(hostnumerics-core
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        POSITION_INDEPENDENT_CODE ON
        VISIBILITY_INLINES_HIDDEN ON
)

add_library(hostnumerics INTERFACE)
add_library(roc::hostnumerics ALIAS hostnumerics)
target_link_libraries(hostnumerics INTERFACE roc::hostnumerics-core)

# CPU conversions to the physical storage layouts that AMD GPU kernels read.
# Kernel presets use the generic reshape/permutation descriptor in Core.
add_library(hostnumerics-amd-gpu-layout STATIC
    src/amd_gpu_layout/mx.cpp
    src/amd_gpu_layout/operand.cpp
)
add_library(roc::hostnumerics-amd-gpu-layout ALIAS hostnumerics-amd-gpu-layout)
target_link_libraries(hostnumerics-amd-gpu-layout PUBLIC roc::hostnumerics-core)
target_include_directories(hostnumerics-amd-gpu-layout
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_sources(hostnumerics-amd-gpu-layout
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include"
        FILES
            include/roc/hostnumerics/amd_gpu_layout/mx.hpp
            include/roc/hostnumerics/amd_gpu_layout/operand.hpp
)
set_target_properties(hostnumerics-amd-gpu-layout
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        POSITION_INDEPENDENT_CODE ON
        VISIBILITY_INLINES_HIDDEN ON
)

if(HOSTNUMERICS_BUILD_TESTING)
    list(APPEND CMAKE_MODULE_PATH "${HostNumerics_SOURCE_DIR}/cmake")
    enable_testing()
    add_subdirectory(tests)
endif()
if(HOSTNUMERICS_BUILD_PYTHON)
    add_subdirectory(python)
endif()

install(TARGETS hostnumerics-core hostnumerics hostnumerics-amd-gpu-layout
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
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/HostNumericsConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/HostNumericsConfigVersion.cmake"
    DESTINATION "${_hostnumerics_cmake_install_dir}"
)
install(FILES LICENSE.md
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/HostNumerics"
)
