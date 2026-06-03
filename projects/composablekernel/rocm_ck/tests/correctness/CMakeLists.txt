# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Correctness tests for rocm_ck kernel families.
#
# Test tier:
#   ROCM_CK_CORRECTNESS — CPU reference validation (no GPU required)
#   ROCM_CK_KERNEL      — GPU kernel tests (require HIP and a GPU, Phase 2+)

# ---------------------------------------------------------------------------
# CPU reference validation (Phase 1 — no GPU needed)
# ---------------------------------------------------------------------------
add_rocm_ck_test(rocm_ck_cpu_ref_validation
    test_cpu_ref_validation.cpp
)
target_include_directories(rocm_ck_cpu_ref_validation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}                      # local headers (fmha_bwd_cpu_ref.hpp)
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include         # rocm_ck headers
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include      # ck_tile headers (reference_batched_*)
)
if(TARGET hip::device)
    target_compile_options(rocm_ck_cpu_ref_validation PRIVATE --offload-host-only)
endif()
add_test(NAME rocm_ck_cpu_ref_validation COMMAND rocm_ck_cpu_ref_validation)
set_tests_properties(rocm_ck_cpu_ref_validation PROPERTIES
    LABELS "ROCM_CK_CORRECTNESS"
)
