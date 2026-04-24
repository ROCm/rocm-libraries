# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(CK_TILE_SWIGLU_TEST_COMPILE_OPTIONS)
set(CK_TILE_SWIGLU_TEST_TARGETS)

if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx90a|gfx11|gfx12")
    # Collect all test targets for umbrella label

    if(GPU_TARGETS MATCHES "gfx94|gfx95")
        add_gtest_executable(test_ck_tile_swiglu test_swiglu.cpp)

        target_compile_options(test_ck_tile_swiglu PRIVATE ${CK_TILE_SWIGLU_TEST_COMPILE_OPTIONS})

        list(APPEND CK_TILE_SWIGLU_TEST_TARGETS
            test_ck_tile_swiglu
        )
    endif()
endif()

# Label all ck_tile swiglu tests with CK_TILE_GEMM_TESTS for selective execution
foreach(test_target ${CK_TILE_SWIGLU_TEST_TARGETS})
    set_tests_properties(${test_target} PROPERTIES LABELS "CK_TILE_SWIGLU_TESTS")
endforeach()

if(!${CK_TILE_SWIGLU_TEST_TARGETS})
    message(DEBUG "Skipping ck_tile_swiglu tests for current target! ")
endif()
