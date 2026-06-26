# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Common helpers define the Python execution environment and the
# rocke_client_add_aot_instances() registration function used by kernel-family
# subdirectories.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/rocke_aot.cmake")

# Aggregate target for all checked-in AOT instance families. Kernel-family
# CMake files add one target per NAME and attach it as a dependency here.
add_custom_target(rocke_client_aot_artifacts
    COMMENT "Build rocKE client AOT artifacts"
)

# Support both standalone rocKE-client configures (BUILD_TESTING via CTest) and
# hip-kernel-provider parent configures (HIPKERNELPROVIDER_ENABLE_TESTS).
set(_ROCKE_CLIENT_ENABLE_TESTS OFF)
if((DEFINED BUILD_TESTING AND BUILD_TESTING)
   OR (DEFINED HIPKERNELPROVIDER_ENABLE_TESTS AND HIPKERNELPROVIDER_ENABLE_TESTS)
)
    set(_ROCKE_CLIENT_ENABLE_TESTS ON)
endif()

# Register the first provider-owned kernel family. Additional families should
# add sibling subdirectories here so they share the same AOT helper policy.
add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/sdpa/fmha_fwd_mfma"
    "${CMAKE_CURRENT_BINARY_DIR}/kernels/sdpa/fmha_fwd_mfma"
)

# Common AOT Python tests validate the operation-independent schema and sidecar
# helpers. Kernel-family tests are registered inside their family directories.
if(_ROCKE_CLIENT_ENABLE_TESTS)
    rocke_client_aot_pythonpath_environment(_ROCKE_CLIENT_AOT_TEST_ENVIRONMENT)
    add_test(
        NAME rocke_client_aot_pytest
        COMMAND "${Python3_EXECUTABLE}" -m pytest "${CMAKE_CURRENT_SOURCE_DIR}/tests"
    )
    set_tests_properties(rocke_client_aot_pytest PROPERTIES
        LABELS "rocKE-client;unit_test"
        ENVIRONMENT "${_ROCKE_CLIENT_AOT_TEST_ENVIRONMENT}"
    )
endif()

# Convenience target for local validation: build artifacts first, then run only
# rocKE-client-labeled CTest entries.
add_custom_target(rocke_client_aot_check
    COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure -L rocKE-client
    DEPENDS rocke_client_aot_artifacts
    COMMENT "Run rocKE client AOT tests"
)
