# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Keep in sync with example/ck_tile/50_sparse_attn/CMakeLists.txt (gfx9-only: async LDS path).
if(NOT SUPPORTED_GPU_TARGETS MATCHES "gfx9")
    return()
endif()

set(SPARSE_ATTN_EXAMPLE_DIR ${PROJECT_SOURCE_DIR}/example/ck_tile/50_sparse_attn)

# The device dispatch (jenga/vsa/sparge/sparge_sage_sparse_attention<T>) lives in the example's
# sparse_attention.cpp; the generated kernels live in the four instance object libraries. The test
# compiles the former and links the latter -- the same set the example executable links.
set(SPARSE_ATTN_INSTANCE_LIBS
    tile_sparse_attn_jenga_instances
    tile_sparse_attn_vsa_instances
    tile_sparse_attn_sparge_instances
    tile_sparse_attn_sparge_sage_instances)

set(TEST_NAME "test_ck_tile_sparse_attn")

function(add_gtest_fwd test_group)
    set(V_TYPES "fp16" "bf16")
    set(CPP_TYPE_fp16 "FmhaSparseFwdFp16")
    set(CPP_TYPE_bf16 "FmhaSparseFwdBf16")

    set(all_tests)
    foreach(type ${V_TYPES})
        set(name "${test_group}_${type}")
        add_gtest_executable(${name}
            test_sparse_attn_fwd.cpp
            ${SPARSE_ATTN_EXAMPLE_DIR}/sparse_attention.cpp)
        # add_gtest_executable skips (result != 0) when the datatype is filtered out for this build
        if(NOT TARGET ${name})
            message(STATUS "Skipping sparse_attn test ${name}")
            continue()
        endif()
        get_test_property(${name} LABELS COMMON_LABELS)
        set_tests_properties(${name} PROPERTIES
            LABELS "${COMMON_LABELS};${TEST_NAME};${test_group};CK_TILE_SPARSE_ATTN_TESTS")
        set_tests_properties(${name} PROPERTIES TIMEOUT 3600)
        target_compile_definitions(${name} PRIVATE
            DataTypeConfig=${CPP_TYPE_${type}}
            CK_TILE_USE_BUFFER_ADDRESSING_BUILTIN
            CK_TILE_FMHA_FWD_FAST_EXP2)
        target_include_directories(${name} PRIVATE
            ${SPARSE_ATTN_EXAMPLE_DIR}
            ${PROJECT_SOURCE_DIR}/include/ck_tile/ops/sparse_attn)
        target_compile_options(${name} PRIVATE
            -Wno-undefined-func-template
            -Wno-float-equal)
        target_link_libraries(${name} PRIVATE ${SPARSE_ATTN_INSTANCE_LIBS})
        list(APPEND all_tests ${name})
    endforeach()
    message(STATUS "Sparse Attention FWD tests: ${all_tests}")
    add_custom_target(${test_group} DEPENDS ${all_tests})
endfunction()

add_gtest_fwd(${TEST_NAME}_fwd)
add_custom_target(${TEST_NAME} DEPENDS ${TEST_NAME}_fwd)

# Umbrella target to build and run all ck_tile sparse attention tests
# Usage: ninja ck_tile_sparse_attn_tests
add_custom_target(ck_tile_sparse_attn_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR} -L "CK_TILE_SPARSE_ATTN_TESTS"
    DEPENDS ${TEST_NAME}
    USES_TERMINAL
    COMMENT "Running all ck_tile sparse attention tests..."
)
