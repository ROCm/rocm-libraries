# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# This test intentionally exercises unclean shutdown with threads still running during
# static destruction. ASAN flags the resulting heap-use-after-free, which is expected
# behavior for this scenario. Disable under ASAN since the test validates graceful
# degradation, not memory safety.
if(BUILD_ADDRESS_SANITIZER OR THEROCK_SANITIZER STREQUAL "ASAN" OR THEROCK_SANITIZER STREQUAL "HOST_ASAN")
    message(STATUS "Skipping backend_logging_shutdown_tests (incompatible with AddressSanitizer)")
    return()
endif()

find_package(Threads REQUIRED)

# Standalone integration test (no gtest) that verifies shutdown safety when multiple
# threads with active user log callbacks are still running during static destruction.
add_executable(hipdnn_backend_logging_shutdown_tests BackendLoggingShutdownTest.cpp)

target_compile_options(hipdnn_backend_logging_shutdown_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

target_link_libraries(
    hipdnn_backend_logging_shutdown_tests
    PRIVATE
        Threads::Threads
        hipdnn_backend
        hipdnn_data_sdk
)

target_compile_definitions(
    hipdnn_backend_logging_shutdown_tests
    PRIVATE TEST_GOOD_DEFAULT_PLUGIN_NAME="${TEST_GOOD_DEFAULT_PLUGIN_NAME}"
)

add_dependencies(hipdnn_backend_logging_shutdown_tests test_good_default_plugin)

clang_tidy_check(hipdnn_backend_logging_shutdown_tests)
add_integration_test_target(hipdnn_backend_logging_shutdown_tests ${CMAKE_CURRENT_BINARY_DIR})
