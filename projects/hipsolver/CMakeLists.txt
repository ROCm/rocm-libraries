# ########################################################################
# Copyright (C) 2016-2026 Advanced Micro Devices, Inc. All rights reserved.
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
# ########################################################################

cmake_minimum_required(VERSION 3.14)

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if(WIN32)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories")
else()
  set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")
endif()

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if(NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")
endif()

if(NOT DEFINED CMAKE_Fortran_COMPILER AND NOT DEFINED ENV{FC})
  set(CMAKE_Fortran_COMPILER  "gfortran")
endif()

project(hipsolver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add our CMake helper files to the lookup path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Get rocm-cmake
include(get-rocm-cmake)

# Include the rocm-cmake components we use
include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMClients)
include(ROCMHeaderWrapper)

set(VERSION_STRING "3.5.0")
rocm_setup_version(VERSION ${VERSION_STRING})

if(NOT DEFINED ENV{HIP_PATH})
  set(HIP_PATH "/opt/rocm/hip")
else()
  file(TO_CMAKE_PATH "$ENV{HIP_PATH}" HIP_PATH)
endif()

# NOTE:  workaround until hip cmake modules fixes symlink logic in their config files; remove when fixed
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake/hip /opt/rocm /opt/rocm/llvm /opt/rocm/hip)

option(BUILD_SHARED_LIBS "Build hipSOLVER as a shared library" ON)
option(BUILD_ADDRESS_SANITIZER "Build with address sanitizer enabled" OFF)
option(BUILD_CODE_COVERAGE "Build hipSOLVER with code coverage enabled" OFF)
option(BUILD_HIPBLAS_TESTS "Build additional tests to ensure hipBLAS and hipSOLVER are compatible (requires installed hipBLAS)" OFF)
option(BUILD_HIPSPARSE_TESTS "Build additional tests to cover sparse functionality (requires installed hipSPARSE)" ON)
# BUILD_SHARED_LIBS is a cmake built-in; we make it an explicit option such that it shows in cmake-gui
option(BUILD_WITH_SPARSE "Build hipSOLVER with sparse functionality available at build time (requires installed dependencies)" OFF)
option(BUILD_VERBOSE "Output additional build information" OFF)
option(HIPSOLVER_FIND_PACKAGE_LAPACK_CONFIG "Skip module mode search for LAPACK" ON)
option(BUILD_FORTRAN_BINDINGS "Build the Fortran bindings" "${UNIX}")
option(EXPORT_FORTRAN_BINDINGS "Install hipsolver_fortran as an exported CMake target" ON)

if(NOT BUILD_SHARED_LIBS)
  add_compile_definitions(HIPSOLVER_STATIC_LIB)
endif()

if(BUILD_FORTRAN_BINDINGS)
  enable_language(Fortran)
endif()

add_library(hipsolver-common INTERFACE)
target_compile_options(hipsolver-common INTERFACE
  -Wno-unused-result # TODO: address [[nodiscard]] warnings
)
if(BUILD_ADDRESS_SANITIZER)
  target_compile_options(hipsolver-common INTERFACE
    -fsanitize=address
    -shared-libasan
  )
  target_link_options(hipsolver-common INTERFACE
    -fsanitize=address
    -shared-libasan
    -fuse-ld=lld
  )
endif()


# Deprecated USE_CUDA option
if(DEFINED USE_CUDA)
  if(USE_CUDA)
    set(ENV{HIP_PLATFORM} nvidia)
    set(HIP_PLATFORM nvidia CACHE STRING "HIP platform (e.g., amd or nvidia)")
    message(DEPRECATION "USE_CUDA is deprecated (use environment variable HIP_PLATFORM=nvidia)")
  else()
    set(ENV{HIP_PLATFORM} amd)
    set(HIP_PLATFORM amd CACHE STRING "HIP platform (e.g., amd or nvidia)")
    message(DEPRECATION "USE_CUDA is deprecated (use environment variable HIP_PLATFORM=amd)")
  endif()
endif()

# Hip headers required of all clients; clients use hip to allocate device memory
find_package(hip CONFIG PATHS ${HIP_PATH} ${ROCM_PATH} /opt/rocm)

# support for cuda backend with hip < 6.0
if(NOT hip_FOUND)
    find_package(HIP MODULE REQUIRED)
    list(APPEND HIP_INCLUDE_DIRS "${HIP_ROOT_DIR}/include")
    set(ENV{HIP_PLATFORM} nvidia)
    if (NOT HIP_PLATFORM)
      set(HIP_PLATFORM nvidia)
    endif()
else()
    if(NOT HIP_PLATFORM)
      set(HIP_PLATFORM amd)
    endif()
endif()

if(HIP_PLATFORM STREQUAL nvidia)
  find_package(CUDA REQUIRED)
endif()

if(NOT "${HIP_PLATFORM}" STREQUAL "nvidia")
    # TODO: This LAPACK dependency is only required to support geev
    # It should be removed once geev is implemented in rocSOLVER
    option(HIPSOLVER_INTERNAL_LAPACK_BUILD "Link to a locally built, static, OpenBLAS library -- required to support geev, will be removed once geev is implemented in rocSOLVER" ON)
    if(HIPSOLVER_INTERNAL_LAPACK_BUILD)
        set(LAPACK_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/lapack-src)
        set(LAPACK_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/lapack-bin)

        # Detect if we're using a multi-config generator (focus on Ninja Multi-Config and Visual Studio)
        get_property(IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

        # Set BYPRODUCTS based on generator type
        if(IS_MULTI_CONFIG)
            # NOTE: OpenBLAS converts the usual config-specific subdirectories
            # to upper-case (e.g., lib/DEBUG/ instead of lib/Debug) in its
            # CMakeLists.txt file.
            if(NOT CMAKE_CONFIGURATION_TYPES)
                set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;MinSizeRel")
            endif()

            set(LAPACK_BUILD_BYPRODUCTS)
            foreach(config ${CMAKE_CONFIGURATION_TYPES})
                string(TOUPPER ${config} config_upper)
                list(APPEND LAPACK_BUILD_BYPRODUCTS
                    ${LAPACK_BINARY_DIR}/lib/${config_upper}/${CMAKE_STATIC_LIBRARY_PREFIX}openblas${CMAKE_STATIC_LIBRARY_SUFFIX})
            endforeach()
        else()
            set(LAPACK_BUILD_BYPRODUCTS
                ${LAPACK_BINARY_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}openblas${CMAKE_STATIC_LIBRARY_SUFFIX})
        endif()

        set(LAPACK_CMAKE_ARGS
            -DC_LAPACK=ON
            -DNOFORTRAN=ON
            -DBUILD_STATIC_LIBS=ON
            -DBUILD_LAPACK_DEPRECATED=OFF
            -DBUILD_TESTING=OFF
            -DBUILD_BENCHMARKS=OFF
            -DUSE_OPENMP=OFF
            -DUSE_THREAD=OFF
            -DCMAKE_C_VISIBILITY_PRESET=hidden
        )

        # Only pass CMAKE_BUILD_TYPE for single-config generators (it is
        # ignored in multi-config generators)
        if(NOT IS_MULTI_CONFIG)
            list(APPEND LAPACK_CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
        endif()

        # Build OpenBLAS with PIC if hipSOLVER is a shared library
        if(BUILD_SHARED_LIBS)
            list(APPEND LAPACK_CMAKE_ARGS -DCMAKE_POSITION_INDEPENDENT_CODE=ON)
        endif()

        # Force OpenBLAS to use the same compiler as hipSOLVER to avoid
        # runtime library mismatches (e.g., MinGW vs MSVCRT on Windows)
        if(CMAKE_C_COMPILER)
            list(APPEND LAPACK_CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER})
        endif()
        if(CMAKE_CXX_COMPILER)
            list(APPEND LAPACK_CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
        endif()

        # Force GENERIC target to ensure binaries work on all CPUs
        list(APPEND LAPACK_CMAKE_ARGS -DTARGET=GENERIC)

        # Downgrade incompatible-pointer-types errors to warnings when using Clang
        if(CMAKE_C_COMPILER_ID MATCHES "Clang")
            list(APPEND LAPACK_CMAKE_ARGS -DCMAKE_C_FLAGS=-Wno-error=incompatible-pointer-types)
        endif()

        include(ExternalProject)
        # Use OpenBLAS v0.3.33 (commit 62bcfb0)
        ExternalProject_Add(hipsolver-static-lapack
            BUILD_BYPRODUCTS    ${LAPACK_BUILD_BYPRODUCTS}
            GIT_REPOSITORY      https://github.com/OpenMathLib/OpenBLAS
            GIT_TAG             62bcfb0dc9f1cfa685fc04135c50e2780c303137
            SOURCE_DIR          ${LAPACK_SOURCE_DIR}
            BINARY_DIR          ${LAPACK_BINARY_DIR}
            CMAKE_ARGS          ${LAPACK_CMAKE_ARGS}
            INSTALL_COMMAND     ""
        )
        add_library(lapack_libraries STATIC IMPORTED)

        if(IS_MULTI_CONFIG)
            foreach(config ${CMAKE_CONFIGURATION_TYPES})
                string(TOUPPER ${config} config_upper)
                set_property(TARGET lapack_libraries PROPERTY
                    IMPORTED_LOCATION_${config_upper}
                    ${LAPACK_BINARY_DIR}/lib/${config_upper}/${CMAKE_STATIC_LIBRARY_PREFIX}openblas${CMAKE_STATIC_LIBRARY_SUFFIX}
                )
            endforeach()
        else()
            set_target_properties(lapack_libraries PROPERTIES
                IMPORTED_LOCATION ${LAPACK_BINARY_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}openblas${CMAKE_STATIC_LIBRARY_SUFFIX}
            )
        endif()

        add_dependencies(lapack_libraries hipsolver-static-lapack)
    else() # NOT HIPSOLVER_INTERNAL_LAPACK_BUILD
        # Some CPU side functionality is provided by external lapack
        # Linking lapack library requires fortran flags
        set(THREADS_PREFER_PTHREAD_FLAG ON)
        find_package(Threads REQUIRED)
        if(HIPSOLVER_FIND_PACKAGE_LAPACK_CONFIG)
            find_package(LAPACK 3.7 REQUIRED CONFIG)
        else()
            find_package(LAPACK 3.7 REQUIRED)
        endif()

        if(NOT LAPACK_LIBRARIES)
            set(LAPACK_LIBRARIES
                ${LAPACK_blas_LIBRARIES}
                ${LAPACK_lapack_LIBRARIES}
            )
        endif()

        if(NOT TARGET lapack_libraries)
            add_library(lapack_libraries INTERFACE)
            message(STATUS "Using manually specified LAPACK")
            target_link_libraries(lapack_libraries INTERFACE
                ${LAPACK_LIBRARIES}
            )
        endif()
    endif()
endif()


# Append our library helper cmake path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(DEFAULT_ARMOR_LEVEL 1)
else()
  set(DEFAULT_ARMOR_LEVEL 0)
endif()
set(ARMOR_LEVEL "${DEFAULT_ARMOR_LEVEL}" CACHE STRING "Enables increasingly expensive runtime correctness checks")
include(armor-config)

# FOR OPTIONAL CODE COVERAGE
if(BUILD_CODE_COVERAGE)
  target_compile_options(hipsolver-common
    INTERFACE
    -g
    -O0
    -fprofile-instr-generate
    -fcoverage-mapping)
  target_link_options(hipsolver-common
    INTERFACE
    -fprofile-instr-generate)
endif()

include(CMakeDependentOption)
cmake_dependent_option(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY
  "Build with file/folder reorg backward compatibility enabled" OFF "NOT WIN32" OFF)
if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY)
  rocm_wrap_header_dir(
    ${CMAKE_SOURCE_DIR}/library/include
    PATTERNS "*.h"
    GUARDS SYMLINK WRAPPER
    WRAPPER_LOCATIONS ${CMAKE_INSTALL_INCLUDEDIR}
  )
endif()

if(WIN32)
  add_compile_definitions(
    WIN32_LEAN_AND_MEAN
    _CRT_SECURE_NO_WARNINGS
    NOMINMAX
  )
endif()

add_subdirectory(library)

include(clients/cmake/build-options.cmake)

if(NOT SYSTEM_OS)
  rocm_set_os_id(SYSTEM_OS)
  string(TOLOWER "${SYSTEM_OS}" SYSTEM_OS)
  rocm_read_os_release(SYSTEM_OS_VERSION VERSION_ID)
endif()

# Build clients of the library
if(BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS)
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
  if(BUILD_CLIENTS_TESTS)
    rocm_package_setup_client_component(tests ${DEP_ARGS})
  endif()
  if(BUILD_CLIENTS_BENCHMARKS)
    rocm_package_setup_client_component(benchmarks ${DEP_ARGS})
  endif()
  add_subdirectory(clients)
endif()

# Package specific CPACK vars
if(NOT "${HIP_PLATFORM}" STREQUAL "nvidia")
  set(rocblas_minimum 4.2.0)
  set(rocsolver_minimum 3.35.0)
  rocm_package_add_dependencies(SHARED_DEPENDS "rocblas >= ${rocblas_minimum}" "rocsolver >= ${rocsolver_minimum}")
  rocm_package_add_rpm_dependencies(STATIC_DEPENDS "rocblas-static-devel >= ${rocblas_minimum}" "rocsolver-static-devel >= ${rocsolver_minimum}")
  rocm_package_add_deb_dependencies(STATIC_DEPENDS "rocblas-static-dev >= ${rocblas_minimum}" "rocsolver-static-dev >= ${rocsolver_minimum}")

  if(SYSTEM_OS STREQUAL "centos" OR SYSTEM_OS STREQUAL "rhel" OR SYSTEM_OS STREQUAL "mariner" OR SYSTEM_OS STREQUAL "azurelinux")
    list(APPEND hipsolver_pkgdeps "suitesparse")
  elseif(SYSTEM_OS STREQUAL "ubuntu" AND SYSTEM_OS_VERSION VERSION_GREATER_EQUAL "24.04")
    list(APPEND hipsolver_pkgdeps "libcholmod5" "libsuitesparseconfig7")
  else()
    list(APPEND hipsolver_pkgdeps "libcholmod3" "libsuitesparseconfig5")
  endif()

  if(BUILD_WITH_SPARSE)
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

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

if(WIN32)
  set(CPACK_SOURCE_GENERATOR "ZIP")
  set(CPACK_GENERATOR "ZIP")
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path" FORCE)
  endif()
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

# Give hipsolver compiled for CUDA backend a different name
if(NOT "${HIP_PLATFORM}" STREQUAL "nvidia")
  set(package_name hipsolver)
else()
  set(package_name hipsolver-alt)
endif()

set(HIPSOLVER_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")

rocm_create_package(
    NAME ${package_name}
    DESCRIPTION "ROCm LAPACK marshalling library"
    MAINTAINER "hipSOLVER Maintainer <hipsolver-maintainer@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${HIPSOLVER_CONFIG_DIR}
)

# ADDITIONAL TARGETS FOR CODE COVERAGE
#
# make coverage
if(BUILD_CODE_COVERAGE)
  set(coverage_test ./clients/staging/hipsolver-test)

  add_custom_target(
    code_cov_tests
    DEPENDS hipsolver-test
    COMMAND echo Coverage GTEST_FILTER=\${GTEST_FILTER}
    COMMAND ${CMAKE_COMMAND} -E rm -rf ./coverage-report
    COMMAND ${CMAKE_COMMAND} -E make_directory ./coverage-report/profraw
    COMMAND ${CMAKE_COMMAND} -E env LLVM_PROFILE_FILE="./coverage-report/profraw/hipSOLVER-coverage_%p.profraw" GTEST_LISTENER=NO_PASS_LINE_IN_LOG ${coverage_test} --gtest_filter=\"\${GTEST_FILTER}\"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  find_program(
    LLVM_PROFDATA
    llvm-profdata
    REQUIRED
    HINTS ${ROCM_PATH}/llvm/bin
    PATHS /opt/rocm/llvm/bin
  )

  find_program(
    LLVM_COV
    llvm-cov
    REQUIRED
    HINTS ${ROCM_PATH}/llvm/bin
    PATHS /opt/rocm/llvm/bin
  )

  add_custom_target(
    coverage
    DEPENDS code_cov_tests
    COMMAND ${LLVM_PROFDATA} merge -sparse ./coverage-report/profraw/hipSOLVER-coverage_*.profraw -o ./coverage-report/hipSOLVER.profdata
    COMMAND ${LLVM_COV} report -object ./library/src/libhipsolver.so -instr-profile=./coverage-report/hipSOLVER.profdata
    COMMAND ${LLVM_COV} show -object ./library/src/libhipsolver.so -instr-profile=./coverage-report/hipSOLVER.profdata -format=html -output-dir=coverage-report
    COMMAND ${LLVM_COV} export -object ./library/src/libhipsolver.so -instr-profile=./coverage-report/hipSOLVER.profdata -format=lcov > ./coverage-report/coverage.info
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  # Coverage cleanup
  add_custom_target(coverage_cleanup
      COMMAND find ${CMAKE_BINARY_DIR} -name *.gcda -delete
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endif()

