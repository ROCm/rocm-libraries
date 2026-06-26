# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/rocke_aot.cmake")

add_custom_target(rocke_client_aot_artifacts
    COMMENT "Build rocKE client AOT artifacts"
)

set(_ROCKE_CLIENT_ENABLE_TESTS OFF)
if((DEFINED BUILD_TESTING AND BUILD_TESTING)
   OR (DEFINED HIPKERNELPROVIDER_ENABLE_TESTS AND HIPKERNELPROVIDER_ENABLE_TESTS)
)
    set(_ROCKE_CLIENT_ENABLE_TESTS ON)
endif()

add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/sdpa/fmha_fwd_mfma"
    "${CMAKE_CURRENT_BINARY_DIR}/kernels/sdpa/fmha_fwd_mfma"
)

if(_ROCKE_CLIENT_ENABLE_TESTS)
    rocke_client_aot_pythonpath_environment(_ROCKE_CLIENT_AOT_TEST_ENVIRONMENT)
    add_test(
        NAME rocke_client_aot_pytest
        COMMAND "${Python3_EXECUTABLE}" -m pytest "${CMAKE_CURRENT_SOURCE_DIR}/tests"
    )
    set_tests_properties(rocke_client_aot_pytest PROPERTIES
        LABELS "rocKE-client;sdpa;unit_test"
        ENVIRONMENT "${_ROCKE_CLIENT_AOT_TEST_ENVIRONMENT}"
    )
endif()

add_custom_target(rocke_client_aot_check
    COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure -L rocKE-client
    DEPENDS rocke_client_aot_artifacts
    COMMENT "Run rocKE client AOT tests"
)
