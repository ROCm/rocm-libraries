# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)

# hipDNN dependency versions to use
set(HIPDNN_SPDLOG_VERSION "1.15.3" CACHE STRING "Version of spdlog to use")
set(HIPDNN_FLATBUFFERS_VERSION "25.9.23" CACHE STRING "Version of flatbuffers to use")
set(HIPDNN_NLOHMANN_JSON_VERSION "3.12.0" CACHE STRING "Version of nlohmann_json to use")
set(HIPDNN_GTEST_VERSION "1.16.0" CACHE STRING "Version of googletest to use")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_compile_definitions(__HIP_PLATFORM_AMD__)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/cmake/ClangToolChain.cmake" CACHE STRING "Toolchain file")

set(HIPDNN_VERSION "0.0.1")

project(
    hipDNN
    VERSION ${HIPDNN_VERSION}
    LANGUAGES CXX)

if(NOT WIN32)
    if(CMAKE_GENERATOR STREQUAL "Ninja")
        message(STATUS "The CMake generator is Ninja.")
    else()
        message(WARNING "The CMake generator is not Ninja. Ninja is preferred over Make.")
    endif()
endif()

if(NOT WIN32)
    # Only set if not already specified by user
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX "${ROCM_PATH}" CACHE PATH "Install path prefix" FORCE)
    endif()
    message(STATUS "Setting CMAKE_INSTALL_PREFIX to ${CMAKE_INSTALL_PREFIX}")
endif()

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

set(HIPDNN_PLUGIN_ENGINE_SUBDIR "hipdnn_plugins/engines" CACHE STRING "Subdirectory for official hipDNN engine plugins")
set(HIPDNN_BUILD_PLUGIN_ENGINE_DIR "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/${HIPDNN_PLUGIN_ENGINE_SUBDIR}" CACHE PATH "Build directory for official hipDNN engine plugins")
set(HIPDNN_FULL_INSTALL_PLUGIN_ENGINE_DIR "${CMAKE_INSTALL_FULL_LIBDIR}/${HIPDNN_PLUGIN_ENGINE_SUBDIR}" CACHE STRING "Full install directory for official hipDNN engine plugins. Uses the CMAKE_INSTALL_PREFIX set at configure time (defaults to /opt/rocm/).")
set(HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR "${CMAKE_INSTALL_LIBDIR}/${HIPDNN_PLUGIN_ENGINE_SUBDIR}" CACHE STRING "Relative install directory for official hipDNN engine plugins")

set(HIPDNN_TEST_PLUGIN_DIR "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/test_plugins" CACHE INTERNAL "Build directory for test plugins")
set(HIPDNN_TEST_PLUGIN_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/test_plugins" CACHE INTERNAL "Install directory for test plugins")

set(HIPDNN_TEST_REFERENCE_DIR "${CMAKE_BINARY_DIR}/lib/hipdnn_reference_data" CACHE INTERNAL "Directory for test reference data")
set(HIPDNN_TEST_REFERENCE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}" CACHE INTERNAL "Install directory for test reference data")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/hipdnn_reference_data/" DESTINATION ${HIPDNN_TEST_REFERENCE_DIR}) # Copies only on calls to cmake
install(DIRECTORY "${CMAKE_BINARY_DIR}/lib/hipdnn_reference_data" DESTINATION ${HIPDNN_TEST_REFERENCE_INSTALL_DIR})

option(HIP_DNN_BUILD_BACKEND "Build the backend code" ON)
option(HIP_DNN_BUILD_FRONTEND "Build the frontend code" ON)
option(HIP_DNN_BUILD_PLUGINS "Builds the plugins as part of the main cmake" ON)
option(HIP_DNN_GENERATE_SDK_HEADERS "Automatically generates sdk headers from schema during build" ON)
option(HIP_DNN_SKIP_TESTS "Skips building all tests" OFF)

option(CODE_COVERAGE "Build with code coverage flags" OFF)
option(ENABLE_CLANG_TIDY "Enable Clang-Tidy checks" ON)
option(BUILD_ADDRESS_SANITIZER "Build with Address Sanitizer enabled" OFF)
option(ENABLE_CLANG_FORMAT "Enable Clang-Format checks" ON)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOL "Add paths to linker search and installed rpath")

list(
    APPEND
    CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    ${ROCM_PATH}/${CMAKE_INSTALL_LIBDIR}/cmake/hip
    ${HIP_DIR}/cmake
)

# NOTE: Workaround until llvm & hip cmake modules fixes symlink logic in their config files; remove when fixed.
#       Added to prevent issues with not finding hip cmake.
list(
    APPEND
    CMAKE_PREFIX_PATH
    ${ROCM_PATH}/llvm
    ${ROCM_PATH}
    ${ROCM_PATH}/hip
)

include(cmake/Dependencies.cmake)
include(cmake/CheckToolVersion.cmake)
include(cmake/ClangCheck.cmake)
include(cmake/ClangTidy.cmake)
include(cmake/Sanitizers.cmake)
include(cmake/Tests.cmake)

# Add global compile options
set(EXTRA_COMPILE_OPTIONS)
set(EXTRA_LINK_OPTIONS)
if(CODE_COVERAGE)
    list(PREPEND EXTRA_COMPILE_OPTIONS 
        -fprofile-instr-generate
        -fcoverage-mapping
    )
    list(PREPEND EXTRA_LINK_OPTIONS 
        -fprofile-instr-generate 
        -fcoverage-mapping)
endif()

list(PREPEND HIPDNN_WARNING_COMPILE_OPTIONS
    -Werror                                  # Treat all warnings as errors
    -Wall                                    # Enable most common warnings
    -Wextra                                  # Enable additional warnings not covered by -Wall
    -Wpedantic                               # Enforce strict ISO C++ compliance
    -Wshadow                                 # Warn about variable shadowing
    -Wnon-virtual-dtor                       # Warn if a class with virtual functions has a non-virtual destructor
    -Wold-style-cast                         # Warn about C-style casts
    -Wcast-align                             # Warn about potential performance issues with misaligned casts
    -Woverloaded-virtual                     # Warn if a base class function is hidden by a derived class function with the same name
    -Wconversion                             # Warn about implicit type conversions that may alter a value
    -Wsign-conversion                        # Warn about implicit conversions between signed and unsigned types
    -Wnull-dereference                       # Warn about dereferencing null pointers
    -Wdouble-promotion                       # Warn when a float is implicitly promoted to a double
    -Wformat=2                               # Enable stricter format string checks
    -Winit-self                              # Warn about variables initialized with itself
    -Wunreachable-code                       # Warn about unreachable code
    -Wno-error=unused-command-line-argument  # Disable error for unused command line arguments
    -Wswitch-default                         # Warn if a switch statement does not have a default case
)

add_compile_options(${EXTRA_COMPILE_OPTIONS})
add_link_options(${EXTRA_LINK_OPTIONS})

add_subdirectory(sdk)

if(HIP_DNN_BUILD_BACKEND)
    add_subdirectory(backend)
endif()

if(HIP_DNN_BUILD_FRONTEND)
    add_subdirectory(frontend)
endif()

add_subdirectory(tests)

if(HIP_DNN_BUILD_PLUGINS)
    file(MAKE_DIRECTORY ${HIPDNN_BUILD_PLUGIN_ENGINE_DIR})
    add_subdirectory(plugins)
endif()

# keep this after all build folder have been added so they have a chance to register their tests
# using append_test_to_check_target

if(NOT HIP_DNN_SKIP_TESTS)
    create_test_name_validation_target()

    finalize_custom_check_target()
    finalize_unit_check_target()
    finalize_integration_check_target()

    add_dependencies(check validate_test_names)
    add_dependencies(check_ctest validate_test_names)
endif()

if(CODE_COVERAGE)
    findAndCheckLlvmTools()
    
    if(HIP_DNN_BUILD_PLUGINS)
        set(MIOPEN_PLUGIN_CODE_COVERAGE_OBJECT -object ${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}/libmiopen_legacy_plugin.so -object ./${CMAKE_INSTALL_BINDIR}/miopen_legacy_plugin_tests)
    endif()

    set(CODE_COVERAGE_IGNORE_REGEX "\(.*deps.*\)|\(.*tests.*\)|\(.*sdk/include/hipdnn_sdk/data_objects.*\)|\(.*sdk/include/hipdnn_sdk/test_utilities.*\)|\(.*sdk/include/hipdnn_sdk/plugin/test_utils.*\)")
    set(CODE_COVERAGE_OBJECTS -object ./${CMAKE_INSTALL_LIBDIR}/libhipdnn_backend.so -object ./${CMAKE_INSTALL_BINDIR}/hipdnn_backend_tests -object ./${CMAKE_INSTALL_BINDIR}/hipdnn_frontend_tests -object ./${CMAKE_INSTALL_BINDIR}/public_hipdnn_backend_tests -object ./${CMAKE_INSTALL_BINDIR}/hipdnn_sdk_tests ${MIOPEN_PLUGIN_CODE_COVERAGE_OBJECT})

    add_custom_target(
        code_coverage
        COMMAND cd ${CMAKE_BINARY_DIR} && find . -name \"*.profraw\" -print0 | xargs -0 ${LLVM_PROFDATA_BINARY} merge -sparse -o ./hipdnn.profdata
        COMMAND ${LLVM_COV_BINARY} report ${CODE_COVERAGE_OBJECTS} -instr-profile=./hipdnn.profdata -ignore-filename-regex=\"${CODE_COVERAGE_IGNORE_REGEX}\" > ./code_cov_hipdnn.report
        COMMAND ${LLVM_COV_BINARY} show -Xdemangler=${LLVM_CXXFILT_BINARY} ${CODE_COVERAGE_OBJECTS} -instr-profile=./hipdnn.profdata -ignore-filename-regex=\"${CODE_COVERAGE_IGNORE_REGEX}\" --format=html --output-dir=./hipdnn_coverage_html
        DEPENDS check
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating code coverage report"
    )
endif()

if(NOT HIP_DNN_SKIP_TESTS)
    install_hipdnn_ctest_files()
endif()
