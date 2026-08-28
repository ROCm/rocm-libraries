# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)

option(CMAKE_EXPORT_COMPILE_COMMANDS "Generate compile commands json file." ON)

project(hipsparselt VERSION 0.2.9 LANGUAGES CXX)

# ROCm libraries root (for shared/ctest and other shared resources)
# Only set default when not already defined (e.g. by superproject / rocm-libraries).
if(NOT DEFINED ROCM_LIBRARIES_ROOT)
  set(ROCM_LIBRARIES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
endif()

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW) # FetchContent timestamp handling
endif()

if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW) # option() honors normal variables
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Set cpack license file here to avoid rocm-cmake warning
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")

include(fetch_rocm_cmake)
include(hipsparselt_supported_architectures)
include(hipsparselt_link_blas_libraries)
include(CMakeDependentOption)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-mf16c" HIPSPARSELT_CXX_HAS_MF16C)

rocm_setup_version(VERSION ${PROJECT_VERSION})
set(hipsparselt_SOVERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(WIN32)
        set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path prefix." FORCE)
    else()
        set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix." FORCE)
    endif()
endif()

# --- Build options and validation ---

option(HIPSPARSELT_ENABLE_CUDA "Build hipSPARSELt with CUDA backend." OFF)
cmake_dependent_option(
    HIPSPARSELT_ENABLE_HIP "Build hipSPARSELt with HIP backend." ON "NOT HIPSPARSELT_ENABLE_CUDA"
    OFF
)
if(HIPSPARSELT_ENABLE_CUDA AND HIPSPARSELT_ENABLE_HIP)
    message(STATUS "HIPSPARSELT_ENABLE_CUDA: ${HIPSPARSELT_ENABLE_CUDA}")
    message(STATUS "HIPSPARSELT_ENABLE_HIP: ${HIPSPARSELT_ENABLE_HIP}")
    message(FATAL_ERROR "Cannot enable CUDA and HIP support simultaneously.")
endif()

set(_hipsparselt_build_shared_default ON)
if(DEFINED BUILD_SHARED_LIBS AND NOT "${BUILD_SHARED_LIBS}" STREQUAL "")
    set(_hipsparselt_build_shared_default "${BUILD_SHARED_LIBS}")
endif()
option(HIPSPARSELT_BUILD_SHARED_LIBS
    "Build the hipSPARSELt shared library."
    "${_hipsparselt_build_shared_default}"
)
unset(_hipsparselt_build_shared_default)
set(BUILD_SHARED_LIBS "${HIPSPARSELT_BUILD_SHARED_LIBS}")
option(HIPSPARSELT_BUILD_TESTING "Build test client." ON)
option(HIPSPARSELT_BUILD_COVERAGE "Build tests with coverage enabled." OFF)

option(HIPSPARSELT_ENABLE_DEVICE "Build hipSPARSELt device libraries." ON)
option(HIPSPARSELT_ENABLE_CLIENT "Build hipSPARSELt clients." ON)
if(HIPSPARSELT_ENABLE_CLIENT)
    option(HIPSPARSELT_ENABLE_BENCHMARKS "Build benchmark client." ON)
    option(HIPSPARSELT_ENABLE_SAMPLES "Build client samples." ON)
    option(HIPSPARSELT_ENABLE_FORTRAN "Build Fortran clients." OFF)
    option(HIPSPARSELT_ENABLE_BLIS "Enable BLIS support for reference implementations." OFF)
endif()

cmake_dependent_option(HIPSPARSELT_ENABLE_MARKER "Use the marker library." ON "NOT WIN32" OFF)
option(HIPSPARSELT_ENABLE_OPENMP "Enable OpenMP support." ON)
option(HIPSPARSELT_ENABLE_ASAN "Build with address sanitizer enabled." OFF)
option(HIPSPARSELT_ENABLE_FETCH "Fetch dependencies." OFF)
option(HIPSPARSELT_ENABLE_THEROCK "Enable TheRock support." OFF)
set(HIPSPARSELT_COVERAGE_GTEST_FILTER "" CACHE STRING "GTest filter for coverage tests.")

if(NOT HIPSPARSELT_ENABLE_CUDA)
    set(GPU_TARGETS "all" CACHE STRING "AMD GFX targets to cross-compile")

    if(NOT GPU_TARGETS)
        set(HIPSPARSELT_ENABLE_DEVICE OFF)
        message(WARNING "No GPU targets specified, device libraries will not be built.")
    elseif(GPU_TARGETS STREQUAL "all")
        hipsparselt_get_base_architectures(GPU_TARGETS)
    else()
        hipsparselt_validate_gpu_targets("${GPU_TARGETS}")
    endif()
    message(STATUS "Building for GPU targets: ${GPU_TARGETS}")
endif()

if(HIPSPARSELT_ENABLE_FORTRAN AND HIPSPARSELT_ENABLE_ASAN)
    message(FATAL_ERROR "Cannot enable fortran clients and address sanitizers.")
endif()

if(HIPSPARSELT_BUILD_COVERAGE)
    if(HIPSPARSELT_ENABLE_FORTRAN)
        message(FATAL_ERROR "Cannot enable fortran clients and code coverage.")
    elseif(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(FATAL_ERROR "Code coverage is only supported for CMAKE_BUILD_TYPE=Debug.")
    endif()
endif()

if(HIPSPARSELT_ENABLE_FORTRAN)
    enable_language(Fortran)
endif()

if(HIPSPARSELT_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

# --- Dependency Setup ---

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(hip REQUIRED)
find_package(hipsparse REQUIRED)

if(HIPSPARSELT_ENABLE_MARKER)
    if(HIPSPARSELT_ENABLE_CUDA)
        message(FATAL_ERROR "rocTracer marker is not supported with CUDA backend.")
    else()
        find_library(ROCTX64_LIBRARY REQUIRED HINTS lib NAMES roctx64 libroctx64)
        get_filename_component(ROCTRACER_LIB_DIR ${ROCTX64_LIBRARY} DIRECTORY)
        get_filename_component(ROCTRACER_ROOT_DIR ${ROCTRACER_LIB_DIR} DIRECTORY)
        find_path(ROCTRACER_INCLUDE_DIR "roctracer/roctx.h"
            HINTS ${ROCTRACER_ROOT_DIR}
            PATH_SUFFIXES include
            REQUIRED)
    endif()
endif()

if(HIPSPARSELT_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(VERBOSE "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
    find_path(cusparselt_INCLUDE_DIR REQUIRED NAMES cusparseLt.h)
    message(VERBOSE "cusparselt_INCLUDE_DIR: ${cusparselt_INCLUDE_DIR}")
    find_library(cusparselt_LIBRARY REQUIRED NAMES cusparseLt)
    message(VERBOSE "cusparselt_LIBRARY: ${cusparselt_LIBRARY}")

    message(STATUS "Found cusparseLt: ${cusparselt_LIBRARY}")
endif()

if(HIPSPARSELT_ENABLE_HIP)
    message(STATUS "hipSPARSELt links the installed hipBLASLt package.")
    find_package(hipblaslt CONFIG REQUIRED)

    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md"
        CACHE FILEPATH "License file for package" FORCE
    )

    if(HIPSPARSELT_ENABLE_DEVICE)
        if(NOT HIPBLASLT_TENSILELITE_PATH)
            set(HIPBLASLT_TENSILELITE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../hipblaslt/tensilelite")
        endif()
        if(NOT EXISTS "${HIPBLASLT_TENSILELITE_PATH}/Tensile/bin/TensileLogic")
            message(FATAL_ERROR
                "hipSPARSELt device codegen needs the hipBLASLt tensilelite source at "
                "'${HIPBLASLT_TENSILELITE_PATH}'; set -DHIPBLASLT_TENSILELITE_PATH.")
        endif()
        set(HIPBLASLT_CODEGEN_ROOT "${HIPBLASLT_TENSILELITE_PATH}")

        find_package(origami CONFIG REQUIRED)

        set(_hipsparselt_rocisa_dir "$<IF:$<PLATFORM_ID:Windows>,$<TARGET_FILE_DIR:roc::tensilelite-host>/../lib/hipblaslt/rocisa,$<TARGET_FILE_DIR:roc::tensilelite-host>/hipblaslt/rocisa>")
        get_filename_component(_hipsparselt_hip_bindir "${hip_DIR}/../../../bin" ABSOLUTE)
        get_filename_component(_hipsparselt_cxx_bindir "${CMAKE_CXX_COMPILER}" DIRECTORY)

        set(_hipsparselt_tensile_out "${CMAKE_BINARY_DIR}/Tensile")
        set(_hipsparselt_codegen_flags "")
        if(HIPSPARSELT_ENABLE_ASAN)
            list(APPEND _hipsparselt_codegen_flags ASAN)
        endif()
        hipblaslt_make_python_command(HIPBLASLT_PYTHON_COMMAND
            PYTHONPATH_DIRS "$<IF:$<PLATFORM_ID:Windows>,$<TARGET_FILE_DIR:roc::tensilelite-host>/../lib/hipblaslt,$<TARGET_FILE_DIR:roc::tensilelite-host>/hipblaslt>" "${HIPBLASLT_CODEGEN_ROOT}"
            RUNTIME_LIB_DIRS "$<TARGET_FILE_DIR:roc::origami>" "${_hipsparselt_rocisa_dir}"
            TOOL_BIN_DIRS "${_hipsparselt_hip_bindir}" "${_hipsparselt_cxx_bindir}"
            ${_hipsparselt_codegen_flags}
        )
        create_device_library(
            LOGIC_PATH "${TENSILELITE_LIBLOGIC_PATH}"
            OUTPUT_DIR "${_hipsparselt_tensile_out}"
            PYTHON_COMMAND "${HIPBLASLT_PYTHON_COMMAND}"
            ASAN "${HIPSPARSELT_ENABLE_ASAN}"
        )

        if(WIN32)
            set(_hipsparselt_tensile_install_dir "${CMAKE_INSTALL_BINDIR}/hipsparselt")
        else()
            set(_hipsparselt_tensile_install_dir "${CMAKE_INSTALL_LIBDIR}/hipsparselt")
        endif()
        install(
            DIRECTORY "${_hipsparselt_tensile_out}/library"
            DESTINATION "${_hipsparselt_tensile_install_dir}"
            COMPONENT runtime
        )
    endif()
endif()

add_subdirectory(library)

if(HIPSPARSELT_BUILD_TESTING)
  enable_testing()
endif()

if(HIPSPARSELT_ENABLE_CLIENT)
    if(NOT HIPSPARSELT_ENABLE_THEROCK)
        if(NOT DEFINED CMAKE_Fortran_COMPILER AND NOT DEFINED ENV{FC})
            set(CMAKE_Fortran_COMPILER "gfortran")
        endif()
        enable_language(Fortran)
        find_package(LAPACK)
        include(fetch_lapack)
    endif()
    add_subdirectory(clients)
endif()

# --- Packaging and installation ---

if(HIPSPARSELT_ENABLE_HIP)
    if(HIPSPARSELT_BUILD_SHARED_LIBS)
        rocm_package_add_dependencies(
            COMPONENT runtime
            DEPENDS "hipblaslt >= ${hipblaslt_VERSION}"
        )
    else()
        rocm_package_add_deb_dependencies(
            COMPONENT devel
            DEPENDS "hipblaslt-static-dev >= ${hipblaslt_VERSION}"
        )
        rocm_package_add_rpm_dependencies(
            COMPONENT devel
            DEPENDS "hipblaslt-static-devel >= ${hipblaslt_VERSION}"
        )
    endif()
endif()

if(HIPSPARSELT_ENABLE_CLIENT)
    include(rocm_client_prerequisites)
    rocm_set_client_package_prerequisites()
    rocm_package_setup_component(clients)
    rocm_package_setup_client_component(clients-common)
endif()

if(HIPSPARSELT_BUILD_TESTING OR BUILD_TESTING)
    rocm_package_setup_client_component(tests DEPENDS COMPONENT clients-common)
endif()

if(HIPSPARSELT_ENABLE_BENCHMARKS)
    rocm_package_setup_client_component(benchmarks DEPENDS COMPONENT clients-common)
endif()

if(HIPSPARSELT_ENABLE_SAMPLES)
    rocm_package_setup_client_component(samples)
endif()

if(HIPSPARSELT_ENABLE_FORTRAN)
    rocm_package_add_rpm_dependencies(COMPONENT tests DEPENDS "${GFORTRAN_RPM}")
    rocm_package_add_deb_dependencies(COMPONENT tests DEPENDS "${GFORTRAN_DEB}")
    rocm_package_add_rpm_dependencies(COMPONENT benchmarks DEPENDS "${GFORTRAN_RPM}")
    rocm_package_add_deb_dependencies(COMPONENT benchmarks DEPENDS "${GFORTRAN_DEB}")
endif()

# Work around code object stripping failure if using /usr/bin/strip
find_program(LLVM_STRIP NAMES llvm-strip PATH_SUFFIXES lib/llvm/bin llvm/bin bin)
if(LLVM_STRIP)
    set(CPACK_RPM_SPEC_MORE_DEFINE "%define __strip ${LLVM_STRIP}")
else()
    message(
        WARNING
            "llvm-strip not found, using default strip tool. This may cause stripping to fail during packaging."
    )
endif()

rocm_install(
    TARGETS
    hipsparselt
    EXPORT
    hipsparselt-targets
    ARCHIVE
    COMPONENT
    "devel"
    LIBRARY
    COMPONENT
    "runtime"
    RUNTIME
    COMPONENT
    "runtime"
)

rocm_install(
    DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT "devel"
)

set(_hipsparselt_export_depends PACKAGE hip)
if(HIPSPARSELT_ENABLE_HIP)
    list(APPEND _hipsparselt_export_depends PACKAGE hipblaslt)
endif()

rocm_export_targets(
    NAMESPACE
    roc::
    TARGETS
    hipsparselt
    EXPORT
    hipsparselt-targets
    DEPENDS
    ${_hipsparselt_export_depends}
)
unset(_hipsparselt_export_depends)

if(HIPSPARSELT_ENABLE_CLIENT)
    if(HIPSPARSELT_BUILD_TESTING OR BUILD_TESTING)
        rocm_install(
            TARGETS hipsparselt-test COMPONENT "tests" DESTINATION "${CMAKE_INSTALL_BINDIR}"
        )

        if(TARGET hipsparselt-test-files)
            get_target_property(
                TEST_YAML_FILES hipsparselt-test-files HIPSPARSELT_INSTALL_YAML_FILES
            )
            get_target_property(TEST_DATA_FILE hipsparselt-test-files HIPSPARSELT_INSTALL_DATA_FILE)
            get_target_property(TEST_PROGRAM hipsparselt-test-files HIPSPARSELT_INSTALL_PROGRAM)
            if(TEST_YAML_FILES)
                rocm_install(
                    FILES ${TEST_YAML_FILES} DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT tests
                )
            endif()
            if(TEST_DATA_FILE)
                rocm_install(
                    FILES ${TEST_DATA_FILE} DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT tests
                )
            endif()
            if(TEST_PROGRAM)
                rocm_install(
                    PROGRAMS ${TEST_PROGRAM} DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT tests
                )
            endif()
        endif()
    endif()

    if(HIPSPARSELT_ENABLE_BENCHMARKS)
        rocm_install(
            TARGETS hipsparselt-bench COMPONENT benchmarks DESTINATION "${CMAKE_INSTALL_BINDIR}"
        )
    endif()

    if(HIPSPARSELT_ENABLE_SAMPLES)
        rocm_install(
            TARGETS
            example_spmm_strided_batched
            example_prune_strip
            example_compress
            COMPONENT
            samples
            DESTINATION
            ${CMAKE_INSTALL_BINDIR}
        )
    endif()
endif()

if(ROCM_LIBS_SUPERBUILD OR NOT HIPSPARSELT_IS_SUBPROJECT)
    # rocm_create_package makes decisions on what type of package to produce based on
    # BUILD_SHARED_LIBS. If we don't set this here we will only get a dev package.
    set(BUILD_SHARED_LIBS ${HIPSPARSELT_BUILD_SHARED_LIBS})
    set(HIPSPARSELT_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
        CACHE PATH "Path placed into ldconfig file"
    )
    rocm_create_package(
        NAME hipsparselt
        DESCRIPTION "ROCm structured sparsity matrix multiplication marshalling library"
        MAINTAINER "hipSPARSELt Maintainer <hipsparselt-maintainer@amd.com>"
        LDCONFIG
        LDCONFIG_DIR ${HIPSPARSELT_CONFIG_DIR}
    )
endif()

# --- Coverage setup ---

if(HIPSPARSELT_BUILD_COVERAGE OR BUILD_CODE_COVERAGE)
    message(STATUS "Configuring code coverage target")

    # Apply coverage flags to library target
    target_compile_options(hipsparselt PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(hipsparselt PRIVATE -fprofile-instr-generate)

    set(_coverage_dir "${CMAKE_CURRENT_BINARY_DIR}/coverage-report")
    set(_coverage_test_arguments "--precompile=hipsparselt-test-precompile.db")
    if(HIPSPARSELT_COVERAGE_GTEST_FILTER)
        list(APPEND _coverage_test_arguments "--gtest_filter=${HIPSPARSELT_COVERAGE_GTEST_FILTER}")
    endif()

    add_custom_target(
        code_cov_tests
        DEPENDS hipsparselt-test
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${_coverage_dir}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_coverage_dir}/profraw"
        COMMAND
            ${CMAKE_COMMAND} -E env
            LLVM_PROFILE_FILE="${_coverage_dir}/profraw/hipsparselt-coverage_%p.profraw"
            $<TARGET_FILE:hipsparselt-test> ${_coverage_test_arguments}
        COMMENT
            "Running coverage with test command: $<TARGET_FILE:hipsparselt-test> ${_coverage_test_arguments}"
    )

    set(_llvm_tool_hints)
    foreach(_prefix IN LISTS CMAKE_PREFIX_PATH)
        list(APPEND _llvm_tool_hints "${_prefix}/lib/llvm/bin")
    endforeach()

    find_program(LLVM_PROFDATA llvm-profdata HINTS ${_llvm_tool_hints} REQUIRED)
    find_program(LLVM_COV llvm-cov HINTS ${_llvm_tool_hints} REQUIRED)

    add_custom_target(
        coverage
        DEPENDS code_cov_tests
        COMMAND
            ${LLVM_PROFDATA} merge -sparse "${_coverage_dir}/profraw/hipsparselt-coverage_*.profraw"
            -o "${_coverage_dir}/hipsparselt.profdata"
        COMMAND ${LLVM_COV} report -object $<TARGET_FILE:hipsparselt>
                -instr-profile="${_coverage_dir}/hipsparselt.profdata"
        COMMAND
            ${LLVM_COV} show -object $<TARGET_FILE:hipsparselt>
            -instr-profile="${_coverage_dir}/hipsparselt.profdata" -format=html
            -output-dir="${_coverage_dir}"
        COMMENT "Generating coverage... report can be found in ${_coverage_dir}/index.html"
    )
endif()
