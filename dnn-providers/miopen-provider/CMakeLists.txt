# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)
message(STATUS "Building miopen provider plugin standalone")
add_compile_definitions(__HIP_PLATFORM_AMD__)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/cmake/ClangToolChain.cmake"
    CACHE STRING "Toolchain file"
)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Include version utilities (must be before project() call)
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/VersionUtils.cmake)

# Version handling
dnn_provider_setup_version(miopen_provider ${CMAKE_CURRENT_LIST_DIR})

# Set project with version
project(miopen-provider VERSION ${MIOPEN_PROVIDER_VERSION} LANGUAGES CXX)

include(CMakePackageConfigHelpers)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/CheckPreCommit.cmake)

if(NOT WIN32)
    # Only set if not already specified by user
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX "${ROCM_PATH}" CACHE PATH "Install path prefix" FORCE)
    endif()
    message(STATUS "Setting CMAKE_INSTALL_PREFIX to ${CMAKE_INSTALL_PREFIX}")
endif()

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE
    CACHE BOOL "Add paths to linker search and installed rpath"
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake
     ${CMAKE_CURRENT_SOURCE_DIR}/../cmake ${ROCM_PATH}/lib/cmake/hip ${HIP_DIR}/cmake
)

# Migrate HIP_DNN_SKIP_TESTS -> MIOPENPROVIDER_SKIP_TESTS
if(DEFINED HIP_DNN_SKIP_TESTS)
    set(MIOPENPROVIDER_SKIP_TESTS ${HIP_DNN_SKIP_TESTS} CACHE BOOL "Skip building tests" FORCE)
    unset(HIP_DNN_SKIP_TESTS CACHE)
    message(DEPRECATION "HIP_DNN_SKIP_TESTS has been renamed to MIOPENPROVIDER_SKIP_TESTS. Please update your build scripts to use -DMIOPENPROVIDER_SKIP_TESTS instead.")
endif()
option(MIOPENPROVIDER_SKIP_TESTS "Skip building tests" OFF)
option(MIOPENPROVIDER_ENABLE_COVERAGE "Build with code coverage flags" OFF)

option(BUILD_ADDRESS_SANITIZER "Build with Address Sanitizer enabled" OFF)
option(ENABLE_CLANG_FORMAT "Enable Clang-Format checks" ON)
if(NOT ENABLE_CLANG_FORMAT)
    message(WARNING "Clang-Format checks are disabled. CI requires passing clang-format checks.")
endif()
if(WIN32)
    option(ENABLE_CLANG_TIDY "Enable Clang-Tidy checks" OFF)
    if(ENABLE_CLANG_TIDY)
        message(WARNING "Clang-Tidy enabled on Windows - this significantly increases build time.")
    endif()
else()
    option(ENABLE_CLANG_TIDY "Enable Clang-Tidy checks" ON)
    if(NOT ENABLE_CLANG_TIDY)
        message(WARNING "Clang-Tidy checks are disabled. CI requires passing clang-tidy checks.")
    endif()
endif()

list(
    PREPEND
    MIOPENPROVIDER_WARNING_COMPILE_OPTIONS
    -Werror # Treat all warnings as errors
    -Wall # Enable most common warnings
    -Wextra # Enable additional warnings not covered by -Wall
    -Wpedantic # Enforce strict ISO C++ compliance
    -Wshadow # Warn about variable shadowing
    -Wnon-virtual-dtor # Warn if a class with virtual functions has a non-virtual destructor
    -Wold-style-cast # Warn about C-style casts
    -Wcast-align # Warn about potential performance issues with misaligned casts
    -Woverloaded-virtual # Warn if a base class function is hidden by a derived class function
                         # with the same name
    -Wconversion # Warn about implicit type conversions that may alter a value
    -Wsign-conversion # Warn about implicit conversions between signed and unsigned types
    -Wnull-dereference # Warn about dereferencing null pointers
    -Wdouble-promotion # Warn when a float is implicitly promoted to a double
    -Wformat=2 # Enable stricter format string checks
    -Winit-self # Warn about variables initialized with itself
    -Wunreachable-code # Warn about unreachable code
    -Wno-error=unused-command-line-argument # Disable error for unused command line arguments
    -Wswitch-default # Warn if a switch statement does not have a default case
)

# NOTE: Workaround until llvm & hip cmake modules fixes symlink logic in their config files;
# remove when fixed. Added to prevent issues with not finding hip cmake.
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/llvm ${ROCM_PATH} ${ROCM_PATH}/hip)

    find_package(hip REQUIRED)

    # hipdnn SDK dependencies - use targets directly if available (superbuild),
    # otherwise find installed packages (true standalone)
    if(NOT TARGET hipdnn_data_sdk)
        find_package(hipdnn_data_sdk CONFIG REQUIRED)
    endif()
    if(NOT TARGET hipdnn_flatbuffers_sdk)
        find_package(hipdnn_flatbuffers_sdk CONFIG REQUIRED)
    endif()
    if(NOT TARGET hipdnn_plugin_sdk)
        find_package(hipdnn_plugin_sdk CONFIG REQUIRED)
    endif()

include(CheckToolVersion)
include(ClangCheck)
include(ClangTidy)
include(Sanitizers)

include(Dependencies)
include(Tests)

if(MIOPENPROVIDER_ENABLE_COVERAGE)
    message(STATUS "Enabling compile flags for coverage-mapping")
    add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
    add_link_options(-fprofile-instr-generate -fcoverage-mapping)
endif()

if(WIN32)
    set(HIPDNN_PLUGIN_ENGINE_ROOTDIR "bin")
else()
    set(HIPDNN_PLUGIN_ENGINE_ROOTDIR "lib")
endif()

set(HIPDNN_BUILD_PLUGIN_ENGINE_DIR
    "${CMAKE_BINARY_DIR}/${HIPDNN_PLUGIN_ENGINE_ROOTDIR}/${HIPDNN_PLUGIN_ENGINE_SUBDIR}"
)

message(STATUS "Plugin build directory: ${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}")
message(STATUS "Plugin install directory: ${HIPDNN_INSTALL_PLUGIN_ENGINE_DIR}")
file(MAKE_DIRECTORY ${HIPDNN_BUILD_PLUGIN_ENGINE_DIR})

find_package(miopen CONFIG REQUIRED)

add_library(
    miopen_plugin_impl OBJECT
    engines/MiopenEngine.cpp
    engines/plans/MiopenBatchnormApplicabilityChecks.cpp
    engines/plans/MiopenBatchnormPlanBuilder.cpp
    engines/plans/MiopenBatchnormBwdPlan.cpp
    engines/plans/MiopenBatchnormFwdInferencePlan.cpp
    engines/plans/MiopenBatchnormFwdInferenceWithVariancePlan.cpp
    engines/plans/MiopenBatchnormFwdTrainingPlan.cpp
    engines/plans/MiopenBatchnormFwdTrainingPlanBuilder.cpp
    engines/plans/MiopenConvBwdPlan.cpp
    engines/plans/MiopenConvFwdBiasActivPlan.cpp
    engines/plans/MiopenConvFwdBiasActivPlanBuilder.cpp
    engines/plans/MiopenConvFwdPlan.cpp
    engines/plans/MiopenConvPlanBuilder.cpp
    engines/plans/MiopenConvWrwPlan.cpp
    HipdnnMiopenHandle.cpp
    MiopenActivationDescriptor.cpp
    MiopenContainer.cpp
    MiopenConvDescriptor.cpp
    MiopenUtils.cpp
    MiopenTensor.cpp
    MiopenPluginPublic.cpp
)
target_include_directories(miopen_plugin_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(miopen_plugin_impl PUBLIC MIOpen hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk)

target_compile_definitions(miopen_plugin_impl PUBLIC MIOPEN_BETA_API)
target_compile_options(miopen_plugin_impl PRIVATE ${MIOPENPROVIDER_WARNING_COMPILE_OPTIONS})
set_target_properties(miopen_plugin_impl PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)

clang_tidy_check(miopen_plugin_impl)

# Static library for unit tests to link against (provides access to internal implementation)
add_library(miopen_plugin_private STATIC $<TARGET_OBJECTS:miopen_plugin_impl>)
target_include_directories(miopen_plugin_private PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# Link to miopen_plugin_impl to inherit its PUBLIC properties
target_link_libraries(miopen_plugin_private PUBLIC MIOpen hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk miopen_plugin_impl)

add_library(miopen_plugin SHARED $<TARGET_OBJECTS:miopen_plugin_impl>)
target_link_libraries(miopen_plugin PRIVATE MIOpen hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk)

if(NOT WIN32)
    target_link_libraries(miopen_plugin PRIVATE "-Xlinker --exclude-libs=ALL")
endif()

target_compile_options(miopen_plugin PRIVATE ${MIOPENPROVIDER_WARNING_COMPILE_OPTIONS})
set_target_properties(
    miopen_plugin
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               LIBRARY_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               RUNTIME_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               ARCHIVE_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               INSTALL_RPATH "${ROCM_PATH}/lib"
               INSTALL_RPATH_USE_LINK_PATH TRUE
               BUILD_WITH_INSTALL_RPATH TRUE
)

# Set RPATH origin for TheRock integration. The plugin is installed to
# lib/hipdnn_plugins/engines/ which is 2 levels below lib/, so TheRock needs
# this hint to calculate correct relative RPATHs.
set_target_properties(miopen_plugin PROPERTIES
    THEROCK_INSTALL_RPATH_ORIGIN "lib/hipdnn_plugins/engines")

# Generate version header
dnn_provider_generate_version_header(miopen_provider miopen_plugin_impl ${CMAKE_CURRENT_LIST_DIR})

if(NOT MIOPENPROVIDER_SKIP_TESTS)
    add_subdirectory(tests)
    add_subdirectory(integration_tests)

    # Finalize test targets with prefix to avoid collisions in superbuild mode
    # Keep this after all build folders have been added so they have a chance to register their tests
    # using add_*_test_target()
    finalize_test_targets("${PROJECT_NAME}")

    # For standalone builds we need to install the ctest files
    if(NOT ROCM_LIBS_SUPERBUILD)
        install_miopen_plugin_ctest_files()
    endif()

    if(NOT TARGET hipdnn_integration_tests)
        find_package(hipdnn_integration_tests CONFIG QUIET)
    endif()

    if(TARGET hipdnn_integration_tests)
        add_external_integration_test_target(
            TARGET_NAME   ${PROJECT_NAME}-external-integration-check
            PLUGIN_TARGET miopen_plugin
            ENGINE_NAME   MIOPEN_ENGINE
            TEST_CONFIG   ${CMAKE_CURRENT_SOURCE_DIR}/config/MIOPEN_ENGINE.toml
            GTEST_FILTER
                *IsSupported*
                *ConvFwd*
        )
        message(STATUS "Created ${PROJECT_NAME}-external-integration-check target")
    else()
        message(STATUS "hipdnn_integration_tests not found - skipping external-integration-check")
    endif()
endif()

install(TARGETS miopen_plugin EXPORT miopen-plugin
        LIBRARY DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
        RUNTIME DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
)

add_clang_tidy_custom_target()

if(MIOPENPROVIDER_ENABLE_COVERAGE)
    findandcheckllvmtools()

    set(CODE_COVERAGE_IGNORE_REGEX ".*deps.*|.*tests.*")
    set(CODE_COVERAGE_OBJECTS
        -object ./${HIPDNN_PLUGIN_ENGINE_ROOTDIR}/${HIPDNN_PLUGIN_ENGINE_SUBDIR}/libmiopen_plugin.so
        -object ./${CMAKE_INSTALL_BINDIR}/miopen_plugin_tests
        -object ./${CMAKE_INSTALL_BINDIR}/miopen_plugin_integration_tests
    )

    # Helper function to create a code coverage target
    # ~~~
    # Parameters:
    #   TARGET_NAME - Name of the coverage target to create
    #   DEPENDS_TARGET - The target that must be built before generating
    #                    coverage (required, use "" for no dependency)
    # ~~~
    function(add_custom_coverage_target TARGET_NAME DEPENDS_TARGET)
        # Build the DEPENDS clause
        set(DEPENDS_CLAUSE "")
        if(NOT "${DEPENDS_TARGET}" STREQUAL "")
            set(DEPENDS_CLAUSE DEPENDS ${DEPENDS_TARGET})
        endif()

        # cmake-format: off
        add_custom_target(
            ${TARGET_NAME}
            COMMAND ${CMAKE_COMMAND} -E make_directory ./coverage-report
            COMMAND find . -name "*.profraw" -print0 | xargs -0 ${LLVM_PROFDATA_BINARY} merge
                    -sparse -o ./coverage-report/miopen_plugin.profdata
            COMMAND
                ${LLVM_COV_BINARY} report ${CODE_COVERAGE_OBJECTS}
                -instr-profile=./coverage-report/miopen_plugin.profdata
                -ignore-filename-regex=${CODE_COVERAGE_IGNORE_REGEX} >
                ./coverage-report/code_cov_miopen_plugin.report
            COMMAND
                ${LLVM_COV_BINARY} show -Xdemangler=${LLVM_CXXFILT_BINARY} ${CODE_COVERAGE_OBJECTS}
                -instr-profile=./coverage-report/miopen_plugin.profdata
                -ignore-filename-regex=${CODE_COVERAGE_IGNORE_REGEX} --format=html
                --output-dir=coverage-report
            COMMAND
                ${LLVM_COV_BINARY} export ${CODE_COVERAGE_OBJECTS}
                -instr-profile=./coverage-report/miopen_plugin.profdata
                -ignore-filename-regex=${CODE_COVERAGE_IGNORE_REGEX} --format=lcov >
                ./coverage-report/coverage.info
            ${DEPENDS_CLAUSE}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating code coverage report"
            VERBATIM
            USES_TERMINAL
        )
        # cmake-format: on
    endfunction()

    if(MIOPENPROVIDER_SKIP_TESTS)
        message(
            WARNING
                "\nThe code coverage targets depend on test targets but test targets don't exist because MIOPENPROVIDER_SKIP_TESTS is enabled."
        )
    else()
        add_custom_coverage_target(code_coverage check) # code_coverage depends on 'check'
        add_custom_coverage_target(unit-coverage unit-check) # unit-coverage depends on 'unit-check'
        add_custom_coverage_target(integration-coverage integration-check) # ... 'integration-check'

        # Make alias to target coverage
        add_custom_target(coverage DEPENDS code_coverage COMMENT "Alias for code_coverage")
    endif()

    # Create a coverage target that does not depend on any other target and will use the coverage
    # data that already exists on disk from prior / manual target runs performed before this target
    # is run.
    add_custom_coverage_target(current-coverage "")

endif()
