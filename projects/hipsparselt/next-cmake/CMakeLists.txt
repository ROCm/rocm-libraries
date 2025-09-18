# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(hipsparselt VERSION 0.2.4 LANGUAGES CXX)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW) # FetchContent timestamp handling
endif()

if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW) # option() honors normal variables
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Set cpack license file here to avoid rocm-cmake warning
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md")

include(FetchROCmCMake)
include(hipSPARSELtSupportedArchitectures)
include(hipSPARSELtLinkBLASLibraries)
include(CMakeDependentOption)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

rocm_setup_version(VERSION ${PROJECT_VERSION})
set(hipsparselt_SOVERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(WIN32)
        set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path prefix." FORCE)
    else()
        set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix." FORCE)
    endif()
endif()

# Build Options and Validation
# ============================================================================
option(CMAKE_EXPORT_COMPILE_COMMANDS "Export compile_commands.json for clang tooling support" ON)

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

option(HIPSPARSELT_BUILD_SHARED_LIBS "Build the hipSPARSELt shared library." ON)
option(HIPSPARSELT_BUILD_TESTING "Build test client." ON)
option(HIPSPARSELT_BUILD_COVERAGE "Build tests with coverage enabled." OFF)

option(HIPSPARSELT_ENABLE_CLIENT "Build hipSPARSELt clients." ON)
if(HIPSPARSELT_ENABLE_CLIENT)
    option(HIPSPARSELT_ENABLE_BENCHMARKS "Build benchmark client." ON)
    option(HIPSPARSELT_ENABLE_SAMPLES "Build client samples." ON)
    option(HIPSPARSELT_ENABLE_FORTRAN "Build Fortran clients." OFF)
    option(HIPSPARSELT_ENABLE_BLIS "Enable BLIS support for reference implementations." ON)
endif()

option(HIPSPARSELT_ENABLE_MARKER "Enable rocTracer marker support." ON)
option(HIPSPARSELT_ENABLE_ASAN "Build with address sanitizer enabled." OFF)
option(HIPSPARSELT_ENABLE_VERBOSE "Output additional build information." OFF)

set(HIPSPARSELT_HIPBLASLT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../hipblaslt/next-cmake"
    CACHE PATH "Path to hipblaslt directory."
)
set(HIPSPARSELT_COVERAGE_GTEST_FILTER "*pre_checkin*" CACHE STRING
                                                            "GTest filter for coverage tests."
)

if(NOT HIPSPARSELT_ENABLE_CUDA)
    set(GPU_TARGETS "" CACHE STRING "AMD GFX targets to cross-compile")

    if(NOT GPU_TARGETS OR GPU_TARGETS STREQUAL "all")
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

# Dependency Setup
# ============================================================================
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(hip REQUIRED)
find_package(hipsparse REQUIRED)

if(HIPSPARSELT_ENABLE_MARKER)
    if(HIPSPARSELT_ENABLE_CUDA)
        message(FATAL_ERROR "rocTracer marker is not supported with CUDA backend.")
    else()
        find_library(ROCTX64_LIBRARY REQUIRED NAMES roctx64 libroctx64)
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
    set(HIPBLASLT_PATH "${HIPSPARSELT_HIPBLASLT_PATH}")
    if(NOT EXISTS "${HIPBLASLT_PATH}")
        message(
            FATAL_ERROR
                "hipBLASLt not found at ${HIPBLASLT_PATH}. Please ensure hipblaslt is available."
        )
    endif()

    block(SCOPE_FOR VARIABLES)
    set(HIPBLASLT_ENABLE_HOST OFF CACHE BOOL "" FORCE)
    set(HIPBLASLT_ENABLE_CLIENT OFF CACHE BOOL "" FORCE)
    set(HIPBLASLT_ENABLE_DEVICE ON CACHE BOOL "" FORCE)
    set(TENSILELITE_ENABLE_HOST ON CACHE BOOL "" FORCE)
    set(HIPBLASLT_DEVICE_LIBRARY_PATH "${CMAKE_BINARY_DIR}/Tensile" CACHE PATH "" FORCE)
    set(HIPBLASLT_LIBLOGIC_PATH
        "${CMAKE_CURRENT_SOURCE_DIR}/../library/src/hcc_detail/rocsparselt/src/spmm/Tensile/Logic/asm_full"
        CACHE STRING "" FORCE
    )
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md"
        CACHE FILEPATH "License file for package" FORCE
    )

    if(WIN32)
        set(HIPBLASLT_TENSILE_LIBRARY_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}hipsparselt/bin"
            CACHE PATH "path to tensile library" FORCE
        )
    else()
        set(HIPBLASLT_TENSILE_LIBRARY_DIR
            "\${CPACK_PACKAGING_INSTALL_PREFIX}${CMAKE_INSTALL_LIBDIR}/hipsparselt"
            CACHE PATH "path to tensile library" FORCE
        )
    endif()

    add_subdirectory("${HIPBLASLT_PATH}" hipblaslt)
    endblock()
endif()

add_subdirectory(library)

if(HIPSPARSELT_ENABLE_CLIENT)
    add_subdirectory(clients)
endif()

# Packaging and installation
# ============================================================================
if(HIPSPARSELT_ENABLE_CLIENT)
    include(ROCmClientPackages)
    rocm_setup_client_components(clients clients-common)
endif()

if(HIPSPARSELT_BUILD_TESTING OR BUILD_TESTING)
    rocm_package_setup_client_component(
        tests
        DEPENDS
        COMPONENT
        clients-common
        DEB
        "${OPENMP_DEB}"
        RPM
        "${OPENMP_RPM}"
    )
endif()

if(HIPSPARSELT_ENABLE_BENCHMARKS)
    rocm_package_setup_client_component(
        benchmarks
        DEPENDS
        COMPONENT
        clients-common
        DEB
        "${OPENMP_DEB}"
        RPM
        "${OPENMP_RPM}"
    )
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

rocm_export_targets(
    NAMESPACE
    roc::
    TARGETS
    hipsparselt
    EXPORT
    hipsparselt-targets
    DEPENDS
    PACKAGE
    hip
)

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

rocm_create_package(
    NAME ${PROJECT_NAME}
    DESCRIPTION "ROCm structured sparsity matrix multiplication marshalling library"
    MAINTAINER "hipSPARSELt Maintainer <hipsparselt-maintainer@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${HIPSPARSELT_CONFIG_DIR}
)

# Coverage Setup
# ============================================================================
if(HIPSPARSELT_BUILD_COVERAGE OR BUILD_CODE_COVERAGE)
    message(STATUS "Configuring code coverage target")
    target_compile_options(hipsparselt PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(hipsparselt PRIVATE -fprofile-instr-generate)

    set(coverage_dir "${CMAKE_CURRENT_BINARY_DIR}/coverage-report")
    set(coverage_test_arguments "--precompile=hipsparselt-test-precompile.db")
    if(HIPSPARSELT_COVERAGE_GTEST_FILTER)
        list(APPEND coverage_test_arguments "--gtest_filter=${HIPSPARSELT_COVERAGE_GTEST_FILTER}")
    endif()

    add_custom_target(
        code_cov_tests
        DEPENDS hipsparselt-test
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${coverage_dir}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${coverage_dir}/profraw"
        COMMAND
            ${CMAKE_COMMAND} -E env
            LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipsparselt-coverage_%p.profraw"
            $<TARGET_FILE:hipsparselt-test> ${coverage_test_arguments}
        COMMENT
            "Running coverage with test command: $<TARGET_FILE:hipsparselt-test> ${coverage_test_arguments}"
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
            ${LLVM_PROFDATA} merge -sparse "${coverage_dir}/profraw/hipsparselt-coverage_*.profraw"
            -o "${coverage_dir}/hipsparselt.profdata"
        COMMAND ${LLVM_COV} report -object $<TARGET_FILE:hipsparselt>
                -instr-profile="${coverage_dir}/hipsparselt.profdata"
        COMMAND
            ${LLVM_COV} show -object $<TARGET_FILE:hipsparselt>
            -instr-profile="${coverage_dir}/hipsparselt.profdata" -format=html
            -output-dir="${coverage_dir}"
        COMMENT "Generating coverage... report can be found in ${coverage_dir}/index.html"
    )
endif()
