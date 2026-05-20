# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(hipsparse_shim)

option(CMAKE_EXPORT_COMPILE_COMMANDS "Export compile_commands.json for clang tooling support" ON)

project(hipsparse VERSION 4.6.0 LANGUAGES CXX C)

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

# Configure Windows version resource file
if(WIN32)
    string(TIMESTAMP HIPSPARSE_COPYRIGHT_YEAR "%Y")
    if(hipsparse_VERSION_TWEAK)
        set(HIPSPARSE_PRODUCT_VERSION_DISPLAY "${hipsparse_VERSION_MAJOR}.${hipsparse_VERSION_MINOR} (${hipsparse_VERSION_TWEAK})")
    else()
        set(HIPSPARSE_PRODUCT_VERSION_DISPLAY "${hipsparse_VERSION_MAJOR}.${hipsparse_VERSION_MINOR}")
    endif()
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/library/src/version.rc.in"
                   "${PROJECT_BINARY_DIR}/version.rc" @ONLY)
endif()

# --- Build options and validation ---
option(HIPSPARSE_ENABLE_CUDA "Build hipSPARSE with CUDA backend." OFF)
option(HIPSPARSE_ENABLE_HIP "Build hipSPARSE with HIP backend." ON)

option(HIPSPARSE_BUILD_SHARED_LIBS "Build hipSPARSE as a shared library." ON)
set(BUILD_SHARED_LIBS ${HIPSPARSE_BUILD_SHARED_LIBS})
option(HIPSPARSE_ENABLE_HOST "Build hipSPARSE library." ON)

option(HIPSPARSE_ENABLE_CLIENT "Build hipSPARSE clients." OFF)
if(HIPSPARSE_ENABLE_CLIENT)
    option(HIPSPARSE_ENABLE_BENCHMARKS "Build benchmark client." ON)
    option(HIPSPARSE_ENABLE_SAMPLES "Build sample programs." ON)
    if(WIN32)
        option(HIPSPARSE_ENABLE_FORTRAN "Enable Fortran client support." OFF)
    else()
        option(HIPSPARSE_ENABLE_FORTRAN "Enable Fortran client support." ON)
    endif()
    option(HIPSPARSE_ENABLE_OPENMP "Enable OpenMP support in clients for parallel execution." ON)
endif()

option(HIPSPARSE_BUILD_TESTING "Build test client." OFF)
option(HIPSPARSE_ENABLE_COVERAGE "Build with code coverage enabled." OFF)
option(HIPSPARSE_ENABLE_ASAN "Build with address sanitizer enabled." OFF)
option(HIPSPARSE_BUILD_DOCS "Build documentation." OFF)

# Validate incompatible option combinations
include(hipsparse_compatibility)

# Set default build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(HIPSPARSE_ENABLE_FORTRAN)
    enable_language(Fortran)
endif()

# --- Dependency Setup ---
find_package(Python3 3.9 COMPONENTS Interpreter REQUIRED)

if(HIPSPARSE_ENABLE_CUDA)
    find_package(HIP REQUIRED CONFIG)
    find_package(CUDAToolkit REQUIRED)
elseif(HIPSPARSE_ENABLE_HIP)
    find_package(hip REQUIRED CONFIG)
    find_package(rocsparse REQUIRED)
endif()

if(HIPSPARSE_ENABLE_CLIENT OR HIPSPARSE_BUILD_TESTING OR BUILD_TESTING)
    if(HIPSPARSE_ENABLE_OPENMP)
        find_package(OpenMP REQUIRED)
    endif()
endif()

# --- Library target ---

if(HIPSPARSE_ENABLE_HOST)
    add_subdirectory(library)
endif()

# Must be set before add_subdirectory(clients): clients/tests needs it for CTest categories
# (shared/ctest/TestCategories.cmake and test_categories.yaml).
if(NOT DEFINED ROCM_LIBRARIES_ROOT)
    set(ROCM_LIBRARIES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
endif()

# --- Client targets ---

if(HIPSPARSE_ENABLE_CLIENT OR HIPSPARSE_BUILD_TESTING OR BUILD_TESTING)
    # Required at project root so ${CMAKE_BINARY_DIR}/CTestTestfile.cmake exists.
    # Otherwise ctest run from the build root finds no tests (they only lived under clients/).
    enable_testing()
    include(rocm_client_prerequisites)
    add_subdirectory(clients)
endif()

# --- Packaging and installation ---

if(HIPSPARSE_ENABLE_HOST)
    rocm_package_setup_component(runtime)
endif()

if(HIPSPARSE_ENABLE_CLIENT OR HIPSPARSE_BUILD_TESTING OR BUILD_TESTING)
    rocm_package_setup_component(clients)
    rocm_package_setup_client_component(clients-common)
endif()

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

if(HIPSPARSE_BUILD_TESTING OR BUILD_TESTING)
    if(NOT WIN32)
        rocm_package_add_rpm_dependencies(COMPONENT tests DEPENDS "${GFORTRAN_RPM}")
        rocm_package_add_deb_dependencies(COMPONENT tests DEPENDS "${GFORTRAN_DEB}")
        if(HIPSPARSE_ENABLE_OPENMP)
            rocm_package_add_rpm_dependencies(COMPONENT tests DEPENDS "${OPENMP_RPM}")
            rocm_package_add_deb_dependencies(COMPONENT tests DEPENDS "${OPENMP_DEB}")
        endif()
    endif()
endif()

if(HIPSPARSE_ENABLE_BENCHMARKS)
    if(NOT WIN32)
        rocm_package_add_rpm_dependencies(COMPONENT benchmarks DEPENDS "${GFORTRAN_RPM}")
        rocm_package_add_deb_dependencies(COMPONENT benchmarks DEPENDS "${GFORTRAN_DEB}")
        if(HIPSPARSE_ENABLE_OPENMP)
            rocm_package_add_rpm_dependencies(COMPONENT benchmarks DEPENDS "${OPENMP_RPM}")
            rocm_package_add_deb_dependencies(COMPONENT benchmarks DEPENDS "${OPENMP_DEB}")
        endif()
    endif()
endif()

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
    set(hipsparse_coverage_dir "${CMAKE_BINARY_DIR}/lcoverage")
    
    # Apply coverage to main library
    target_compile_options(hipsparse PRIVATE -fprofile-arcs -ftest-coverage)
    target_link_options(hipsparse PRIVATE --coverage)
    
    # Apply coverage to client common libraries
    if(HIPSPARSE_ENABLE_CLIENT)
        target_compile_options(hipsparse-clients-common PUBLIC -fprofile-arcs -ftest-coverage)
        target_link_options(hipsparse-clients-common PUBLIC --coverage)
        
        target_compile_options(hipsparse-samples-common PUBLIC -fprofile-arcs -ftest-coverage)
        target_link_options(hipsparse-samples-common PUBLIC --coverage)
    endif()
    
    add_custom_target(coverage_cleanup
        COMMAND find ${CMAKE_BINARY_DIR} -name *.gcda -delete
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Cleaning up coverage data files"
    )
    
    add_custom_target(coverage_analysis
        COMMAND echo Coverage GTEST_FILTER=\${GTEST_FILTER}
        COMMAND $<TARGET_FILE:hipsparse-test> --gtest_filter=\"\${GTEST_FILTER}\"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running tests for coverage analysis"
    )
    
    add_dependencies(coverage_analysis hipsparse-test)
    
    add_custom_target(coverage_output
        DEPENDS coverage_analysis
        COMMAND mkdir -p lcoverage
        COMMAND lcov --directory . --base-directory . --capture -o "${hipsparse_coverage_dir}/raw_main_coverage.info"
        COMMAND lcov --remove "${hipsparse_coverage_dir}/raw_main_coverage.info" "'/opt/*'" "'/usr/*'" -o "${hipsparse_coverage_dir}/main_coverage.info"
        COMMAND genhtml "${hipsparse_coverage_dir}/main_coverage.info" --output-directory "${hipsparse_coverage_dir}"
        COMMENT "Generating coverage report"
    )
    
    add_custom_target(coverage DEPENDS coverage_output)
endif()

# --- Address Sanitizer targets ---

if(HIPSPARSE_ENABLE_ASAN)
    # Apply ASAN to main library
    target_compile_options(hipsparse PRIVATE -fsanitize=address -shared-libasan)
    target_link_options(hipsparse PRIVATE -fsanitize=address -shared-libasan -fuse-ld=lld)
    
    # Apply ASAN to client common libraries
    if(HIPSPARSE_ENABLE_CLIENT OR HIPSPARSE_BUILD_TESTING OR BUILD_TESTING)
        target_compile_options(hipsparse-clients-common PUBLIC -fsanitize=address -shared-libasan)
        target_link_options(hipsparse-clients-common PUBLIC -fsanitize=address -shared-libasan -fuse-ld=lld)
        
        target_compile_options(hipsparse-samples-common PUBLIC -fsanitize=address -shared-libasan)
        target_link_options(hipsparse-samples-common PUBLIC -fsanitize=address -shared-libasan -fuse-ld=lld)
    endif()
endif()