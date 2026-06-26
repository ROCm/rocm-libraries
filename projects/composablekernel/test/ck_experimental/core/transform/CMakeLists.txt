# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# ck_experimental V4 transform-graph tests.
#
# Positive (gtest):
#   test_transform_subgraph        -- constexpr subgraph/splice correctness
#   test_transform_cross_tu        -- cross-TU NTTP equality, 3 graph TUs + main
#   test_transform_p64_correctness -- Precision64 (>2^32) runtime correctness
#
# Negative (compile-fail): efail_* TUs that MUST fail to compile with a specific
# diagnostic. See add_efail_test below.
#
# The bench/ subdirectory (V1 + V4 decision-bench cells + driver) is a manual
# performance harness, not a CTest target -- intentionally not added here.

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_transform_subgraph test_transform_subgraph.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_transform_subgraph PRIVATE utility)
    endif()

    # Cross-TU NTTP equality: 3 graph TUs + 1 gtest main. `--icf=none` keeps the
    # discriminating-power check (a != c) from trivially passing via lld
    # identical-code-folding.
    add_gtest_executable(test_transform_cross_tu
        cross_tu_a.cpp
        cross_tu_b.cpp
        cross_tu_c_alt.cpp
        cross_tu_main.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_transform_cross_tu PRIVATE utility)
        target_include_directories(test_transform_cross_tu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_options(test_transform_cross_tu PRIVATE "LINKER:--icf=none")
    endif()

    add_gtest_executable(test_transform_p64_correctness test_transform_p64_correctness.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_transform_p64_correctness PRIVATE utility)
    endif()

    # Negative-compile tests. Each efail_*.cpp constructs a malformed graph in a
    # constant expression, which invokes a non-constexpr trap stub and so MUST
    # fail to compile naming that stub. The target is EXCLUDE_FROM_ALL (the
    # normal build never compiles these); a CTest builds it on demand and passes
    # iff EXPECTED_DIAG appears in the build output. PASS_REGULAR_EXPRESSION
    # overrides the non-zero build exit code, so a clean compile (regression)
    # produces no match and fails the test.
    function(add_efail_test TEST_NAME EXPECTED_DIAG)
        add_executable(${TEST_NAME} EXCLUDE_FROM_ALL ${TEST_NAME}.cpp)
        set_source_files_properties(${TEST_NAME}.cpp PROPERTIES LANGUAGE HIP)
        set_property(TARGET ${TEST_NAME} PROPERTY HIP_ARCHITECTURES ${SUPPORTED_GPU_TARGETS})
        set_target_properties(${TEST_NAME} PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD TRUE)
        add_test(NAME ${TEST_NAME}
                 COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                                          --target ${TEST_NAME} --config $<CONFIG>)
        set_tests_properties(${TEST_NAME} PROPERTIES
            PASS_REGULAR_EXPRESSION "${EXPECTED_DIAG}"
            LABELS "SMOKE_TEST")
    endfunction()

    add_efail_test(efail_splice_transform_overflow     graphValidationErrorSpliceTransformOverflow)
    add_efail_test(efail_subgraph_read_arity_mismatch  transformErrorReadCountArityMismatch)
    add_efail_test(efail_subgraph_write_arity_mismatch transformErrorWriteCountArityMismatch)
endif()
