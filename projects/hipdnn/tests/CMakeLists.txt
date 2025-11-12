# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(HIP_DNN_SKIP_TESTS)
    message(STATUS "Skipping HIP-DNN Integration tests")
    return()
endif()

set(HIPDNN_TEST_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/backend/tests/descriptors
    ${CMAKE_SOURCE_DIR}/tests/backend
    ${CMAKE_SOURCE_DIR}/backend/src
    ${CMAKE_SOURCE_DIR}/tests
    CACHE INTERNAL "Common test include directories"
)

# Test plugins are used by both frontend and backend tests.
if(HIP_DNN_BUILD_FRONTEND OR HIP_DNN_BUILD_BACKEND)
    add_subdirectory(test_plugins)
endif()

if(HIP_DNN_BUILD_BACKEND)
    add_subdirectory(backend)
endif()

if(HIP_DNN_BUILD_FRONTEND)
    add_subdirectory(frontend)
endif()
