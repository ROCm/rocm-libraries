# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)
project(hipdnn-integration-tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

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

include(ClangTidy)

find_package(hip REQUIRED)
find_package(hipdnn_test_sdk REQUIRED)
find_package(hipdnn_plugin_sdk REQUIRED)
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.16.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(INSTALL_GTEST OFF)
    FetchContent_MakeAvailable(googletest)
endif()
find_package(Threads REQUIRED)
find_package(nlohmann_json REQUIRED)

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

clang_tidy_check(hipdnn_integration_tests)

include(GNUInstallDirs)
install(TARGETS hipdnn_integration_tests
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

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
