# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

message("-- Adding: test/ck_tile/direct_conv/")

# Unit tests (CPU-only, compile with HIP for type support)
add_gtest_executable(test_direct_conv_mathutil test_mathutil.cpp)
add_gtest_executable(test_direct_conv_params test_conv_params.cpp)
add_gtest_executable(test_direct_conv_matrix_layout test_matrix_layout.cpp)
add_gtest_executable(test_direct_conv_swizzle test_swizzle.cpp)
add_gtest_executable(test_direct_conv_transpose_lds test_transpose_lds_layout.cpp)
add_gtest_executable(test_direct_conv_tile_distribution_max_thread_id test_tile_distribution_max_thread_id.cpp)

# Integration test (gfx950 / CDNA4 only)
if(GPU_TARGETS MATCHES "gfx950")
    add_gtest_executable(test_direct_conv_grouped_4c_fp16 test_grouped_4c_fp16.cpp)
    # The direct conv kernels are ported from hipconv and use C-style casts, have unused
    # parameters, and other patterns that trigger -Weverything. Suppress these warnings for now.
    # TODO: Fix the code such that warnign can be re-enabled.
    if(TARGET test_direct_conv_grouped_4c_fp16)
        target_compile_options(test_direct_conv_grouped_4c_fp16 PRIVATE
            -Wno-old-style-cast
            -Wno-unused-parameter
            -Wno-shadow
        )
    endif()

    add_gtest_executable(test_direct_conv_grouped_4c_fp16_tile_conv test_grouped_4c_fp16_tile_conv.cpp)
    if(TARGET test_direct_conv_grouped_4c_fp16_tile_conv)
        target_compile_options(test_direct_conv_grouped_4c_fp16_tile_conv PRIVATE
            -Wno-old-style-cast
            -Wno-unused-parameter
            -Wno-shadow
        )
    endif()

    add_gtest_executable(test_direct_conv_grouped_16c_fp16 test_grouped_16c_fp16.cpp)
    if(TARGET test_direct_conv_grouped_16c_fp16)
        target_compile_options(test_direct_conv_grouped_16c_fp16 PRIVATE
            -Wno-old-style-cast
            -Wno-unused-parameter
            -Wno-shadow
        )
    endif()

    add_gtest_executable(test_direct_conv_grouped_16c_fp16_tile_conv test_grouped_16c_fp16_tile_conv.cpp)
    if(TARGET test_direct_conv_grouped_16c_fp16_tile_conv)
        target_compile_options(test_direct_conv_grouped_16c_fp16_tile_conv PRIVATE
            -Wno-old-style-cast
            -Wno-unused-parameter
            -Wno-shadow
        )
    endif()

    add_gtest_executable(test_direct_conv_grouped_8c_fp16 test_grouped_8c_fp16.cpp)
    if(TARGET test_direct_conv_grouped_8c_fp16)
        target_compile_options(test_direct_conv_grouped_8c_fp16 PRIVATE
            -Wno-old-style-cast
            -Wno-unused-parameter
            -Wno-shadow
        )
    endif()

    add_gtest_executable(test_direct_conv_grouped_8c_fp16_tile_conv test_grouped_8c_fp16_tile_conv.cpp)
    if(TARGET test_direct_conv_grouped_8c_fp16_tile_conv)
        target_compile_options(test_direct_conv_grouped_8c_fp16_tile_conv PRIVATE
            -Wno-old-style-cast
            -Wno-unused-parameter
            -Wno-shadow
        )
    endif()

    add_gtest_executable(test_direct_conv_mfma test_mfma.cpp)
    if(TARGET test_direct_conv_mfma)
        target_compile_options(test_direct_conv_mfma PRIVATE
            -Wno-old-style-cast
            -Wno-unused-parameter
            -Wno-shadow
        )
    endif()

    add_gtest_executable(test_direct_conv_lds_bank_conflicts test_lds_bank_conflicts.cpp)
    if(TARGET test_direct_conv_lds_bank_conflicts)
        target_compile_options(test_direct_conv_lds_bank_conflicts PRIVATE
            -Wno-old-style-cast
            -Wno-unused-parameter
            -Wno-shadow
        )
    endif()

    # Umbrella target: build and run all gfx950 integration tests
    set(CK_TILE_DIRECT_CONV_INTEGRATION_TEST_TARGETS
        test_direct_conv_grouped_4c_fp16_tile_conv
        test_direct_conv_grouped_16c_fp16_tile_conv
        test_direct_conv_grouped_8c_fp16_tile_conv
        test_direct_conv_mfma
        test_direct_conv_lds_bank_conflicts
    )

    foreach(test_target ${CK_TILE_DIRECT_CONV_INTEGRATION_TEST_TARGETS})
        if(TARGET ${test_target})
            set_tests_properties(${test_target} PROPERTIES LABELS "CK_TILE_DIRECT_CONV_TESTS")
        endif()
    endforeach()

    add_custom_target(ck_tile_direct_conv_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR} -L "CK_TILE_DIRECT_CONV_TESTS"
        DEPENDS ${CK_TILE_DIRECT_CONV_INTEGRATION_TEST_TARGETS}
        USES_TERMINAL
        COMMENT "Running all ck_tile direct conv integration tests..."
    )

endif()
