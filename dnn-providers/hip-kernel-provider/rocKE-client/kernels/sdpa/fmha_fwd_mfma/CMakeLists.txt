# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Each call registers all checked-in .instance.json files for one architecture.
# The helper creates both a per-architecture target and build-tree HSACO/sidecar
# artifacts under ${PROJECT_BINARY_DIR}/rocKE-client/aot/<arch>/<name>/.
rocke_client_add_aot_instances(
    NAME sdpa_fwd_fmha_mfma_gfx1151
    ARCH gfx1151
    INSTANCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/instances/gfx1151"
)

rocke_client_add_aot_instances(
    NAME sdpa_fwd_fmha_mfma_gfx942
    ARCH gfx942
    INSTANCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/instances/gfx942"
)

# Family-local tests cover SDPA-specific instance parsing and optional numeric
# execution. The numeric tests point at the helper's artifact layout and skip
# with return code 77 when no matching HIP device is visible.
if(_ROCKE_CLIENT_ENABLE_TESTS)
    rocke_client_aot_pythonpath_environment(_ROCKE_CLIENT_AOT_TEST_ENVIRONMENT)
    add_test(
        NAME rocke_client_sdpa_aot_pytest
        COMMAND "${Python3_EXECUTABLE}" -m pytest
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_sdpa_aot_instance.py"
    )
    set_tests_properties(rocke_client_sdpa_aot_pytest PROPERTIES
        LABELS "rocKE-client;sdpa;unit_test"
        ENVIRONMENT "${_ROCKE_CLIENT_AOT_TEST_ENVIRONMENT}"
    )

    add_test(
        NAME rocke_client_sdpa_aot_numeric_gfx1151
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/sdpa_aot_numeric.py"
                --arch gfx1151
                --artifact-dir "${PROJECT_BINARY_DIR}/rocKE-client/aot/gfx1151/sdpa_fwd_fmha_mfma_gfx1151"
    )
    set_tests_properties(rocke_client_sdpa_aot_numeric_gfx1151 PROPERTIES
        LABELS "gpu;rocKE-client;sdpa"
        REQUIRED_FILES "${PROJECT_BINARY_DIR}/rocKE-client/aot/gfx1151/sdpa_fwd_fmha_mfma_gfx1151/build.stamp"
        SKIP_RETURN_CODE 77
        ENVIRONMENT "${_ROCKE_CLIENT_AOT_TEST_ENVIRONMENT}"
    )

    add_test(
        NAME rocke_client_sdpa_aot_numeric_gfx942
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/sdpa_aot_numeric.py"
                --arch gfx942
                --artifact-dir "${PROJECT_BINARY_DIR}/rocKE-client/aot/gfx942/sdpa_fwd_fmha_mfma_gfx942"
    )
    set_tests_properties(rocke_client_sdpa_aot_numeric_gfx942 PROPERTIES
        LABELS "gpu;rocKE-client;sdpa"
        REQUIRED_FILES "${PROJECT_BINARY_DIR}/rocKE-client/aot/gfx942/sdpa_fwd_fmha_mfma_gfx942/build.stamp"
        SKIP_RETURN_CODE 77
        ENVIRONMENT "${_ROCKE_CLIENT_AOT_TEST_ENVIRONMENT}"
    )
endif()
