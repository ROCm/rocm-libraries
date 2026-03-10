# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)

# Default install prefix (before project() so it doesn't use FORCE)
if(NOT DEFINED CMAKE_INSTALL_PREFIX OR CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(WIN32)
        set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path prefix." FORCE)
    else()
        set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix." FORCE)
    endif()
endif()

# Default build type for single-config generators
if(NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
    )
endif()

if(NOT DEFINED CMAKE_Fortran_COMPILER AND NOT DEFINED ENV{FC})
    set(CMAKE_Fortran_COMPILER "gfortran")
endif()

project(hipsolver LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Set cpack license file early to avoid rocm-cmake warning
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")

# --- Backward-compatibility shim for renamed options ---
include(hipsolver-options-shim)

# --- rocm-cmake ---
find_package(ROCM 0.7.3 REQUIRED CONFIG)

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMClients)
include(ROCMHeaderWrapper)
include(CMakeDependentOption)
include(GNUInstallDirs)

set(VERSION_STRING "3.3.0")
rocm_setup_version(VERSION ${VERSION_STRING})

# --- Build options ---

option(BUILD_SHARED_LIBS "Build hipSOLVER as a shared library." ON)
option(HIPSOLVER_ENABLE_CUDA "Build with CUDA backend instead of HIP." OFF)
option(HIPSOLVER_ENABLE_ASAN "Build with address sanitizer enabled." OFF)
option(HIPSOLVER_BUILD_COVERAGE "Build with code coverage enabled." OFF)
option(HIPSOLVER_ENABLE_HIPBLAS_TESTS "Build tests for hipBLAS/hipSOLVER compatibility (requires hipBLAS)." OFF)
option(HIPSOLVER_ENABLE_HIPSPARSE_TESTS "Build tests for sparse functionality (requires hipSPARSE)." ON)
option(HIPSOLVER_ENABLE_SPARSE "Build with sparse functionality at build time (requires dependencies)." OFF)
option(HIPSOLVER_ENABLE_VERBOSE "Output additional build information." OFF)
option(HIPSOLVER_FIND_PACKAGE_LAPACK_CONFIG "Skip module mode search for LAPACK." ON)
option(HIPSOLVER_ENABLE_FORTRAN "Build the Fortran bindings." "${UNIX}")
option(HIPSOLVER_BUILD_TESTING "Build hipSOLVER unit tests." OFF)
option(HIPSOLVER_ENABLE_BENCHMARKS "Build hipSOLVER benchmarks." OFF)
option(HIPSOLVER_ENABLE_SAMPLES "Build hipSOLVER samples." OFF)
option(CMAKE_EXPORT_COMPILE_COMMANDS "Export compile_commands.json for clang tooling support." ON)

set(HIPSOLVER_ARMOR_LEVEL "0" CACHE STRING
    "Runtime correctness check level (0=none, 1=assertions, 2=debug iterators)."
)

# --- Language setup ---

if(HIPSOLVER_ENABLE_FORTRAN)
    enable_language(Fortran)
endif()

# --- Dependencies ---

if(HIPSOLVER_ENABLE_CUDA)
    find_package(CUDA REQUIRED)
    find_package(HIP MODULE REQUIRED)
else()
    find_package(hip REQUIRED CONFIG)
endif()

# --- Armor configuration ---
include(armor-config)

# --- Backward compatibility for file reorg ---
cmake_dependent_option(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY
    "Build with file/folder reorg backward compatibility enabled." OFF "NOT WIN32" OFF
)
if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY)
    rocm_wrap_header_dir(
        "${PROJECT_SOURCE_DIR}/library/include"
        PATTERNS "*.h"
        GUARDS SYMLINK WRAPPER
        WRAPPER_LOCATIONS ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()

# --- Library ---
add_subdirectory(library)

# --- OS detection for packaging ---
if(NOT SYSTEM_OS)
    rocm_set_os_id(SYSTEM_OS)
    string(TOLOWER "${SYSTEM_OS}" SYSTEM_OS)
    rocm_read_os_release(SYSTEM_OS_VERSION VERSION_ID)
endif()

# --- Clients ---
if(HIPSOLVER_BUILD_TESTING OR HIPSOLVER_ENABLE_BENCHMARKS OR HIPSOLVER_ENABLE_SAMPLES)
    set(GFORTRAN_RPM "libgfortran4")
    set(GFORTRAN_DEB "libgfortran4")
    if(SYSTEM_OS STREQUAL "centos" OR SYSTEM_OS STREQUAL "rhel" OR SYSTEM_OS STREQUAL "almalinux")
        if(SYSTEM_OS_VERSION VERSION_GREATER_EQUAL "8")
            set(GFORTRAN_RPM "libgfortran")
        endif()
    elseif(SYSTEM_OS STREQUAL "ubuntu" AND SYSTEM_OS_VERSION VERSION_GREATER_EQUAL "20.04")
        set(GFORTRAN_DEB "libgfortran5")
    elseif(SYSTEM_OS STREQUAL "mariner" OR SYSTEM_OS STREQUAL "azurelinux")
        set(GFORTRAN_RPM "gfortran")
    endif()
    rocm_package_setup_component(clients)
    if(UNIX)
        set(DEP_ARGS DEPENDS RPM "${GFORTRAN_RPM}" DEB "${GFORTRAN_DEB}")
    endif()
    if(HIPSOLVER_BUILD_TESTING)
        rocm_package_setup_client_component(tests ${DEP_ARGS})
    endif()
    if(HIPSOLVER_ENABLE_BENCHMARKS)
        rocm_package_setup_client_component(benchmarks ${DEP_ARGS})
    endif()
    add_subdirectory(clients)
endif()

# --- Packaging ---

if(NOT HIPSOLVER_ENABLE_CUDA)
    set(rocblas_minimum 4.2.0)
    set(rocsolver_minimum 3.33.0)
    rocm_package_add_dependencies(SHARED_DEPENDS
        "rocblas >= ${rocblas_minimum}" "rocsolver >= ${rocsolver_minimum}"
    )
    rocm_package_add_rpm_dependencies(STATIC_DEPENDS
        "rocblas-static-devel >= ${rocblas_minimum}" "rocsolver-static-devel >= ${rocsolver_minimum}"
    )
    rocm_package_add_deb_dependencies(STATIC_DEPENDS
        "rocblas-static-dev >= ${rocblas_minimum}" "rocsolver-static-dev >= ${rocsolver_minimum}"
    )

    if(SYSTEM_OS STREQUAL "centos" OR SYSTEM_OS STREQUAL "rhel"
       OR SYSTEM_OS STREQUAL "mariner" OR SYSTEM_OS STREQUAL "azurelinux")
        list(APPEND hipsolver_pkgdeps "suitesparse")
    elseif(SYSTEM_OS STREQUAL "ubuntu" AND SYSTEM_OS_VERSION VERSION_GREATER_EQUAL "24.04")
        list(APPEND hipsolver_pkgdeps "libcholmod5" "libsuitesparseconfig7")
    else()
        list(APPEND hipsolver_pkgdeps "libcholmod3" "libsuitesparseconfig5")
    endif()

    if(HIPSOLVER_ENABLE_SPARSE)
        set(rocsparse_minimum 2.3.0)
        rocm_package_add_dependencies(SHARED_DEPENDS "rocsparse >= ${rocsparse_minimum}")
        rocm_package_add_rpm_dependencies(STATIC_DEPENDS "rocsparse-static-devel >= ${rocsparse_minimum}")
        rocm_package_add_deb_dependencies(STATIC_DEPENDS "rocsparse-static-dev >= ${rocsparse_minimum}")
        rocm_package_add_dependencies(DEPENDS ${hipsolver_pkgdeps})
    else()
        string(JOIN "," CPACK_DEBIAN_RUNTIME_PACKAGE_RECOMMENDS "rocsparse" ${hipsolver_pkgdeps})
        string(JOIN "," CPACK_RPM_RUNTIME_PACKAGE_SUGGESTS "rocsparse" ${hipsolver_pkgdeps})
    endif()
endif()

set(CPACK_RPM_PACKAGE_LICENSE "MIT")

if(WIN32)
    set(CPACK_SOURCE_GENERATOR "ZIP")
    set(CPACK_GENERATOR "ZIP")
    set(INSTALL_PREFIX "C:/hipSDK")
    set(CPACK_SET_DESTDIR OFF)
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "C:/hipSDK")
    set(CPACK_PACKAGING_INSTALL_PREFIX "")
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
else()
    if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
        set(CPACK_PACKAGING_INSTALL_PREFIX "${ROCM_PATH}")
    endif()
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}")

if(NOT HIPSOLVER_ENABLE_CUDA)
    set(package_name hipsolver)
else()
    set(package_name hipsolver-alt)
endif()

set(HIPSOLVER_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
    CACHE PATH "Path placed into ldconfig file"
)

rocm_create_package(
    NAME ${package_name}
    DESCRIPTION "ROCm LAPACK marshalling library"
    MAINTAINER "hipSOLVER Maintainer <hipsolver-maintainer@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${HIPSOLVER_CONFIG_DIR}
)

# --- Code coverage targets ---

if(HIPSOLVER_BUILD_COVERAGE)
    find_program(LLVM_PROFDATA llvm-profdata
        REQUIRED HINTS ${ROCM_PATH}/llvm/bin
    )
    find_program(LLVM_COV llvm-cov
        REQUIRED HINTS ${ROCM_PATH}/llvm/bin
    )

    set(_coverage_dir "${CMAKE_CURRENT_BINARY_DIR}/coverage-report")

    add_custom_target(code_cov_tests
        DEPENDS hipsolver-test
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${_coverage_dir}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_coverage_dir}/profraw"
        COMMAND ${CMAKE_COMMAND} -E env
            LLVM_PROFILE_FILE="${_coverage_dir}/profraw/hipSOLVER-coverage_%p.profraw"
            GTEST_LISTENER=NO_PASS_LINE_IN_LOG
            $<TARGET_FILE:hipsolver-test> --gtest_filter="\${GTEST_FILTER}"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    add_custom_target(coverage
        DEPENDS code_cov_tests
        COMMAND ${LLVM_PROFDATA} merge -sparse
            "${_coverage_dir}/profraw/hipSOLVER-coverage_*.profraw"
            -o "${_coverage_dir}/hipSOLVER.profdata"
        COMMAND ${LLVM_COV} report
            -object $<TARGET_FILE:hipsolver>
            -instr-profile="${_coverage_dir}/hipSOLVER.profdata"
        COMMAND ${LLVM_COV} show
            -object $<TARGET_FILE:hipsolver>
            -instr-profile="${_coverage_dir}/hipSOLVER.profdata"
            -format=html -output-dir="${_coverage_dir}"
        COMMAND ${LLVM_COV} export
            -object $<TARGET_FILE:hipsolver>
            -instr-profile="${_coverage_dir}/hipSOLVER.profdata"
            -format=lcov > "${_coverage_dir}/coverage.info"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating coverage report in ${_coverage_dir}"
    )

    add_custom_target(coverage_cleanup
        COMMAND ${CMAKE_COMMAND} -E rm -f
            "${CMAKE_BINARY_DIR}/*.profraw"
            "${_coverage_dir}/hipSOLVER.profdata"
            "${_coverage_dir}/coverage.info"
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${_coverage_dir}"
    )
endif()
