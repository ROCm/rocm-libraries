# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)
project(hipdnn-integration-tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

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

option(BUILD_ADDRESS_SANITIZER "Build with Address Sanitizer enabled" OFF)

include(ClangTidy)
include(ClangCheck)
include(Sanitizers)

findandcheckllvmsymbolizer()

list(
    PREPEND
    INTEGRATION_TESTS_WARNING_COMPILE_OPTIONS
    -Werror # Treat all warnings as errors
    -Wall # Enable most common warnings
    -Wextra # Enable additional warnings not covered by -Wall
    -Wpedantic # Enforce strict ISO C++ compliance
    -Wshadow # Warn about variable shadowing
    -Wnon-virtual-dtor # Warn if a class with virtual functions has a non-virtual destructor
    -Wold-style-cast # Warn about C-style casts
    -Wcast-align # Warn about potential performance issues with misaligned casts
    -Woverloaded-virtual # Warn if a base class function is hidden by a derived class function with
                         # the same name
    -Wconversion # Warn about implicit type conversions that may alter a value
    -Wsign-conversion # Warn about implicit conversions between signed and unsigned types
    -Wnull-dereference # Warn about dereferencing null pointers
    -Wdouble-promotion # Warn when a float is implicitly promoted to a double
    -Wformat=2 # Enable stricter format string checks
    -Winit-self # Warn about variables initialized with itself
    -Wunreachable-code # Warn about unreachable code
    -Wswitch-default # Warn if a switch statement does not have a default case
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    list(APPEND INTEGRATION_TESTS_WARNING_COMPILE_OPTIONS
        -Wno-error=unused-command-line-argument # Clang-specific: disable error for unused CLI args
    )
endif()

find_package(hip REQUIRED)
find_package(hipdnn_test_sdk REQUIRED)
find_package(hipdnn_plugin_sdk REQUIRED)
find_package(Threads REQUIRED)
find_package(nlohmann_json REQUIRED)

include(Dependencies)

add_executable(hipdnn_integration_tests
    src/main.cpp
    src/IntegrationGpuBatchnormBackwardActivation.cpp
    src/IntegrationGpuConvForward.cpp
    src/IntegrationIsSupportedExtPerformance.cpp
)

target_include_directories(hipdnn_integration_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(hipdnn_integration_tests
    PRIVATE
        hip::host
        hipdnn_test_sdk
        hipdnn_plugin_sdk
        GTest::gtest
        Threads::Threads
        nlohmann_json::nlohmann_json
)

target_compile_options(hipdnn_integration_tests PRIVATE ${INTEGRATION_TESTS_WARNING_COMPILE_OPTIONS})

clang_tidy_check(hipdnn_integration_tests)

include(GNUInstallDirs)
install(TARGETS hipdnn_integration_tests
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

enable_testing()
add_test(NAME hipdnn_integration_tests COMMAND hipdnn_integration_tests)
if(DEFINED TEST_ENVIRONMENT)
    set_tests_properties(hipdnn_integration_tests PROPERTIES ENVIRONMENT "${TEST_ENVIRONMENT}")
endif()

# Generate CTestTestfile for installed tests (used by CI test runners).
# We manually create this file with relative paths - CMake's CTest module
# generates files with absolute paths which don't work for relocated installs.
set(_INSTALLED_CTEST_FILE "${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake.install")
file(WRITE "${_INSTALLED_CTEST_FILE}"
    "# CTestTestfile for installed hipdnn-integration-tests\n"
    "# Used by CI test runners to execute tests from the installed location\n\n"
    "add_test(hipdnn_integration_tests \"../hipdnn_integration_tests\")\n"
)

install(
    FILES "${_INSTALLED_CTEST_FILE}"
    DESTINATION "${CMAKE_INSTALL_BINDIR}/hipdnn_integration_tests_test_infra"
    RENAME CTestTestfile.cmake
)

add_clang_tidy_custom_target()

install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/hipdnn_test_config.toml"
    DESTINATION "${CMAKE_INSTALL_BINDIR}/hipdnn_integration_tests_test_infra"
)
