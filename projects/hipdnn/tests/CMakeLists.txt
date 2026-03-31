# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(HIPDNN_SKIP_TESTS)
    message(STATUS "Skipping HIP-DNN Integration tests")
    return()
endif()

set(HIPDNN_TEST_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/backend/tests/descriptors ${PROJECT_SOURCE_DIR}/tests/backend
    ${PROJECT_SOURCE_DIR}/backend/src ${PROJECT_SOURCE_DIR}/tests
    CACHE INTERNAL "Common test include directories"
)

# Test plugins are used by both frontend and backend tests.
if(HIPDNN_BUILD_FRONTEND OR HIPDNN_BUILD_BACKEND)
    add_subdirectory(test_plugins)
endif()

if(HIPDNN_BUILD_BACKEND)
    add_subdirectory(backend)
    add_subdirectory(backend_logging)
endif()

if(HIPDNN_BUILD_FRONTEND)
    add_subdirectory(frontend)
endif()
