# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(GoogleTest)
find_package(Threads REQUIRED)

# Integration tests
add_executable(fusilli_plugin_integration_tests
    test_convfprop.cpp
    test_basic.cpp
)
target_compile_options(fusilli_plugin_integration_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
target_link_libraries(fusilli_plugin_integration_tests PRIVATE
    GTest::gtest_main
    hip::host
    hipdnn_frontend
    hipdnn_sdk
    Threads::Threads
)
target_compile_definitions(fusilli_plugin_integration_tests PRIVATE
    FUSILLI_PLUGIN_DIR="${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
    FUSILLI_PLUGIN_NAME="${FUSILLI_PLUGIN_NAME}"
    FUSILLI_PLUGIN_ENGINE_ID="${FUSILLI_PLUGIN_ENGINE_ID}"
)
# Register with CTest
gtest_discover_tests(fusilli_plugin_integration_tests
    # Ensure that tests pick up libhipdnn_backend.so in build directory, not
    # from the global TheRock install.
    PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib"
)
