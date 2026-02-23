# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(Threads REQUIRED)

# Standalone integration test (no gtest) that verifies shutdown safety when multiple
# threads with active user log callbacks are still running during static destruction.
add_executable(hipdnn_backend_logging_shutdown_test BackendLoggingShutdownTest.cpp)

target_compile_options(hipdnn_backend_logging_shutdown_test PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

target_link_libraries(
    hipdnn_backend_logging_shutdown_test
    PRIVATE
        Threads::Threads
        hipdnn_backend
)

clang_tidy_check(hipdnn_backend_logging_shutdown_test)
add_integration_test_target(hipdnn_backend_logging_shutdown_test ${CMAKE_CURRENT_BINARY_DIR})
