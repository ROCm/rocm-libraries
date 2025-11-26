# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(hipsparse_shim)

option(CMAKE_EXPORT_COMPILE_COMMANDS "Export compile_commands.json for clang tooling support" ON)

project(hipsparse VERSION 4.2.0 LANGUAGES CXX C)

if(POLICY CMP0066)
    cmake_policy(SET CMP0066 NEW) # Honor per-config flags in try_compile()
endif()

if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW) # option() honors normal variables
endif()

# Set cpack license file here to avoid rocm-cmake warning
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")

include(fetch_rocm_cmake)
include(GNUInstallDirs)

rocm_setup_version(VERSION ${PROJECT_VERSION})
set(hipsparse_SOVERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(WIN32)
        set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix." FORCE)
    else()
        set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix." FORCE)
    endif()
endif()

# --- Build options and validation ---
option(HIPSPARSE_ENABLE_CUDA "Build hipSPARSE with CUDA backend." OFF)
option(HIPSPARSE_ENABLE_HIP "Build hipSPARSE with HIP backend." ON)

if(HIPSPARSE_ENABLE_CUDA AND HIPSPARSE_ENABLE_HIP)
    message(FATAL_ERROR "Cannot enable both CUDA and HIP backends simultaneously.")
endif()

option(HIPSPARSE_BUILD_SHARED_LIBS "Build hipSPARSE as a shared library." ON)
option(HIPSPARSE_BUILD_CLIENTS_ONLY "Build only clients." OFF)
option(HIPSPARSE_BUILD_TESTING "Build tests (requires googletest)." OFF)
option(HIPSPARSE_ENABLE_BENCHMARKS "Build benchmarks." ON)
option(HIPSPARSE_ENABLE_SAMPLES "Build examples." ON)
option(HIPSPARSE_ENABLE_FORTRAN "Enable Fortran client support." ON)
option(HIPSPARSE_ENABLE_OPENMP "Enable OpenMP support in clients for parallel execution." ON)
option(HIPSPARSE_ENABLE_COVERAGE "Build with code coverage enabled." OFF)
option(HIPSPARSE_ENABLE_ASAN "Build with address sanitizer enabled." OFF)
option(HIPSPARSE_BUILD_DOCS "Build documentation." OFF)

# Validate incompatible option combinations
if(HIPSPARSE_ENABLE_COVERAGE)
    # if(HIPSPARSE_ENABLE_FORTRAN)
        # message(FATAL_ERROR "Cannot enable fortran clients and code coverage.")
    if((NOT CMAKE_BUILD_TYPE STREQUAL "Debug") AND (NOT CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
        message(FATAL_ERROR "Code coverage is only supported for CMAKE_BUILD_TYPE=Debug and CMAKE_BUILD_TYPE=RelWithDebInfo.") # rel with deb info also
    endif()
endif()



# Set default build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Conditionally enable Fortran
# NOTE: Behavior change - Fortran now controlled by HIPSPARSE_ENABLE_FORTRAN option (default ON)
# Previously enabled automatically when building tests or benchmarks
if(HIPSPARSE_ENABLE_FORTRAN)
    enable_language(Fortran)
endif()

# --- Dependency Setup ---

find_package(Python3 3.9 COMPONENTS Interpreter REQUIRED)

if(HIPSPARSE_ENABLE_CUDA)
    find_package(HIP REQUIRED CONFIG)
    # list(APPEND HIP_INCLUDE_DIRS "${HIP_ROOT_DIR}/include")
    find_package(CUDA REQUIRED)
elseif(HIPSPARSE_ENABLE_HIP)
    find_package(hip REQUIRED CONFIG)
    find_package(rocsparse 4.0.1 REQUIRED CONFIG)
endif()

if(HIPSPARSE_BUILD_TESTING OR BUILD_TESTING OR HIPSPARSE_ENABLE_BENCHMARKS OR HIPSPARSE_ENABLE_SAMPLES)
    if(HIPSPARSE_ENABLE_OPENMP)
        find_package(OpenMP REQUIRED)
    endif()
endif()

# --- Library target ---

if(NOT HIPSPARSE_BUILD_CLIENTS_ONLY)
    add_subdirectory(library)
else()
    # When building clients-only, find the installed hipSPARSE library
    find_package(hipsparse 4.0.0 REQUIRED CONFIG)
endif()

# --- Client targets ---

if(HIPSPARSE_BUILD_CLIENTS_ONLY OR HIPSPARSE_ENABLE_SAMPLES OR HIPSPARSE_BUILD_TESTING OR BUILD_TESTING OR HIPSPARSE_ENABLE_BENCHMARKS)
    if(NOT CLIENTS_OS)
        rocm_set_os_id(CLIENTS_OS)
        string(TOLOWER "${CLIENTS_OS}" CLIENTS_OS)
        rocm_read_os_release(CLIENTS_OS_VERSION VERSION_ID)
    endif()
    message(STATUS "OS: ${CLIENTS_OS} ${CLIENTS_OS_VERSION}")
    
    set(GFORTRAN_RPM "libgfortran4")
    set(GFORTRAN_DEB "libgfortran4")
    
    if(CLIENTS_OS STREQUAL "centos" OR CLIENTS_OS STREQUAL "rhel" OR CLIENTS_OS STREQUAL "almalinux")
        if(CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "8")
            set(GFORTRAN_RPM "libgfortran")
        endif()
    elseif(CLIENTS_OS STREQUAL "ubuntu" AND CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "20.04")
        set(GFORTRAN_DEB "libgfortran5")
    elseif(CLIENTS_OS STREQUAL "mariner" OR CLIENTS_OS STREQUAL "azurelinux")
        set(GFORTRAN_RPM "gfortran")
    endif()
    
    # Set OpenMP package dependencies only if enabled
    if(HIPSPARSE_ENABLE_OPENMP)
        set(OPENMP_RPM "libgomp")
        set(OPENMP_DEB "libomp-dev")
        if(CLIENTS_OS STREQUAL "sles")
            set(OPENMP_RPM "libgomp1")
        endif()
    else()
        set(OPENMP_RPM "")
        set(OPENMP_DEB "")
    endif()
    
    add_subdirectory(clients)
endif()

# --- Packaging and installation ---

if(HIPSPARSE_BUILD_CLIENTS_ONLY OR HIPSPARSE_ENABLE_SAMPLES OR HIPSPARSE_BUILD_TESTING OR BUILD_TESTING OR HIPSPARSE_ENABLE_BENCHMARKS)
    rocm_package_setup_component(clients)
    rocm_package_setup_client_component(clients-common)
    
    if(HIPSPARSE_BUILD_TESTING OR BUILD_TESTING)
        rocm_package_setup_client_component(
            tests
            DEPENDS
            COMPONENT clients-common
            DEB "${GFORTRAN_DEB}"
            RPM "${GFORTRAN_RPM}"
        )
        rocm_package_setup_client_component(
            tests
            DEPENDS
            COMPONENT clients-common
            DEB "${OPENMP_DEB}"
            RPM "${OPENMP_RPM}"
        )
    endif()
    
    if(HIPSPARSE_ENABLE_BENCHMARKS)
        rocm_package_setup_client_component(
            benchmarks
            DEPENDS
            COMPONENT clients-common
            DEB "${OPENMP_DEB}"
            RPM "${OPENMP_RPM}"
        )
    endif()
    
    if(NOT WIN32)
        if(HIPSPARSE_BUILD_TESTING OR BUILD_TESTING)
            rocm_package_add_rpm_dependencies(COMPONENT tests DEPENDS "${GFORTRAN_RPM}")
            rocm_package_add_deb_dependencies(COMPONENT tests DEPENDS "${GFORTRAN_DEB}")
        endif()
        if(HIPSPARSE_ENABLE_BENCHMARKS)
            rocm_package_add_rpm_dependencies(COMPONENT benchmarks DEPENDS "${GFORTRAN_RPM}")
            rocm_package_add_deb_dependencies(COMPONENT benchmarks DEPENDS "${GFORTRAN_DEB}")
        endif()
        if(HIPSPARSE_ENABLE_OPENMP AND (HIPSPARSE_BUILD_TESTING OR BUILD_TESTING))
            rocm_package_add_rpm_dependencies(COMPONENT tests DEPENDS "${OPENMP_RPM}")
            rocm_package_add_deb_dependencies(COMPONENT tests DEPENDS "${OPENMP_DEB}")
        endif()
    endif()
endif()

# Package specific dependencies
if(HIPSPARSE_ENABLE_HIP)
    set(ROCSPARSE_MINIMUM "4.0.1")
    rocm_package_add_dependencies(SHARED_DEPENDS "rocsparse >= ${ROCSPARSE_MINIMUM}")
    rocm_package_add_deb_dependencies(STATIC_DEPENDS "rocsparse-static-dev >= ${ROCSPARSE_MINIMUM}")
    rocm_package_add_rpm_dependencies(STATIC_DEPENDS "rocsparse-static-devel >= ${ROCSPARSE_MINIMUM}")
endif()

set(CPACK_RPM_PACKAGE_LICENSE "MIT")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

if(WIN32)
    set(CPACK_SOURCE_GENERATOR "ZIP")
    set(CPACK_GENERATOR "ZIP")
    set(CPACK_SET_DESTDIR OFF)
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "C:/hipSDK")
    set(CPACK_PACKAGING_INSTALL_PREFIX "")
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
else()
    if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
        set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
    endif()
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}")

if(HIPSPARSE_ENABLE_HIP)
    set(package_name hipsparse)
elseif(HIPSPARSE_ENABLE_CUDA)
    set(package_name hipsparse-alt)
endif()

set(HIPSPARSE_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")

rocm_create_package(
    NAME ${package_name}
    DESCRIPTION "ROCm SPARSE library"
    MAINTAINER "hipSPARSE Maintainer <hipsparse-maintainer@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${HIPSPARSE_CONFIG_DIR}
)

# --- Documentation ---

if(HIPSPARSE_BUILD_DOCS)
    add_subdirectory(docs)
endif()

# --- Code coverage targets ---

if(HIPSPARSE_ENABLE_COVERAGE)
    add_custom_target(coverage_cleanup
        COMMAND find ${CMAKE_BINARY_DIR} -name *.gcda -delete
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Cleaning up coverage data files"
    )
    
    add_custom_target(coverage_analysis
        COMMAND echo Coverage GTEST_FILTER=\${GTEST_FILTER}
        COMMAND ./clients/staging/hipsparse-test --gtest_filter=\"\${GTEST_FILTER}\"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running tests for coverage analysis"
    )
    
    add_dependencies(coverage_analysis hipsparse)
    
    add_custom_target(coverage_output
        DEPENDS coverage_analysis
        COMMAND mkdir -p lcoverage
        COMMAND lcov --directory . --base-directory . --capture -o lcoverage/raw_main_coverage.info
        COMMAND lcov --remove lcoverage/raw_main_coverage.info "'/opt/*'" "'/usr/*'" -o lcoverage/main_coverage.info
        COMMAND genhtml lcoverage/main_coverage.info --output-directory lcoverage
        COMMENT "Generating coverage report"
    )
    
    add_custom_target(coverage DEPENDS coverage_output)
endif()
