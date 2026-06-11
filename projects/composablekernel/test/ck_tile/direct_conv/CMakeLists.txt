# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

message("-- Adding: test/ck_tile/direct_conv/")

# Direct conv config headers.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Warning suppressions shared by all direct-conv integration tests.
set(CK_TILE_DIRECT_CONV_TEST_WARNINGS
    # We have __global__ functions in the normal .cpp test files, which are guarded by #ifdef __HIP_DEVICE_COMPILE__.
    # In the host compiler pass, the function arguments appear as "unused", which is why we need to suppress 
    # this warning globally for all direct-conv tests.
    # TODO: Fix the test code such that we can remove this suppression.
    -Wno-unused-parameter
)

# add_direct_conv_test(<target> <source> [WARNINGS <flag>...] [DEFINES <def>...])
#
# Adds a direct-conv integration test executable with the shared warning
# suppressions. WARNINGS appends extra per-target suppressions (e.g.
# -Wno-float-equal) and DEFINES adds per-target compile definitions.
function(add_direct_conv_test target source)
    cmake_parse_arguments(ARG "" "" "WARNINGS;DEFINES" ${ARGN})
    add_gtest_executable(${target} ${source})
    if(TARGET ${target})
        target_compile_options(${target} PRIVATE
            ${CK_TILE_DIRECT_CONV_TEST_WARNINGS} ${ARG_WARNINGS})
        if(ARG_DEFINES)
            target_compile_definitions(${target} PRIVATE ${ARG_DEFINES})
        endif()
    endif()
endfunction()

# add_direct_conv_umbrella_target(<target>...)
#
# Labels the given test targets and creates the ck_tile_direct_conv_tests
# umbrella target that builds and runs them via ctest.
function(add_direct_conv_umbrella_target)
    foreach(test_target ${ARGN})
        if(TARGET ${test_target})
            set_tests_properties(${test_target} PROPERTIES LABELS "CK_TILE_DIRECT_CONV_TESTS")
        endif()
    endforeach()

    add_custom_target(ck_tile_direct_conv_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR} -L "CK_TILE_DIRECT_CONV_TESTS"
        DEPENDS ${ARGN}
        USES_TERMINAL
        COMMENT "Running all ck_tile direct conv integration tests..."
    )
endfunction()

# Unit tests (CPU-only, compile with HIP for type support)
add_gtest_executable(test_direct_conv_params test_conv_params.cpp)
add_gtest_executable(test_direct_conv_matrix_layout test_matrix_layout.cpp)
add_gtest_executable(test_direct_conv_swizzle test_swizzle.cpp)
add_gtest_executable(test_direct_conv_transpose_lds test_transpose_lds_layout.cpp)

# Integration tests. gfx950 (CDNA4) supports the full set; gfx942 (CDNA3) only
# the 4c-forward subset plus the component tests that fit its feature set.
if(GPU_TARGETS MATCHES "gfx950")
    add_direct_conv_test(test_direct_conv_grouped_4c_fp16_tile_conv test_grouped_4c_fp16_tile_conv.cpp)
    add_direct_conv_test(test_direct_conv_grouped_16c_fp16_tile_conv test_grouped_16c_fp16_tile_conv.cpp)
    add_direct_conv_test(test_direct_conv_grouped_8c_fp16_tile_conv test_grouped_8c_fp16_tile_conv.cpp)
    add_direct_conv_test(test_direct_conv_grouped_32c_fp16_tile_conv test_grouped_32c_fp16_tile_conv.cpp)
    add_direct_conv_test(test_direct_conv_non_grouped_32c_fp16_tile_conv test_non_grouped_32c_fp16_tile_conv.cpp)
    add_direct_conv_test(test_direct_conv_mfma test_mfma.cpp)
    add_direct_conv_test(test_direct_conv_lds_bank_conflicts test_lds_bank_conflicts.cpp)
    add_direct_conv_test(test_direct_conv_weight_loader test_weight_loader.cpp WARNINGS -Wno-float-equal)
    add_direct_conv_test(test_direct_conv_input_loader test_input_loader.cpp WARNINGS -Wno-float-equal)
    add_direct_conv_test(test_direct_conv_output_writer test_output_writer.cpp WARNINGS -Wno-float-equal)
    add_direct_conv_test(test_direct_conv_conv_input_loader_chunked test_conv_input_loader_chunked.cpp WARNINGS -Wno-float-equal)
    add_direct_conv_test(test_direct_conv_conv_weight_loader_chunked test_conv_weight_loader_chunked.cpp WARNINGS -Wno-float-equal)

    # BF16 CK Tile direct convolution integration tests
    add_direct_conv_test(test_direct_conv_grouped_4c_bf16_tile_conv test_grouped_4c_bf16_tile_conv.cpp)
    add_direct_conv_test(test_direct_conv_grouped_8c_bf16_tile_conv test_grouped_8c_bf16_tile_conv.cpp)
    add_direct_conv_test(test_direct_conv_grouped_16c_bf16_tile_conv test_grouped_16c_bf16_tile_conv.cpp)
    add_direct_conv_test(test_direct_conv_grouped_32c_bf16_tile_conv test_grouped_32c_bf16_tile_conv.cpp)

    add_direct_conv_umbrella_target(
        test_direct_conv_grouped_4c_fp16_tile_conv
        test_direct_conv_grouped_16c_fp16_tile_conv
        test_direct_conv_grouped_8c_fp16_tile_conv
        test_direct_conv_grouped_32c_fp16_tile_conv
        test_direct_conv_non_grouped_32c_fp16_tile_conv
        test_direct_conv_grouped_4c_bf16_tile_conv
        test_direct_conv_grouped_8c_bf16_tile_conv
        test_direct_conv_grouped_16c_bf16_tile_conv
        test_direct_conv_grouped_32c_bf16_tile_conv
        test_direct_conv_mfma
        test_direct_conv_lds_bank_conflicts
        test_direct_conv_input_loader
        test_direct_conv_weight_loader
        test_direct_conv_output_writer
    )
elseif(GPU_TARGETS MATCHES "gfx942")
    # Dgrad (backward-data) direct conv is gfx950-only (CDNA4 ds_read_b64_tr_b16
    # transpose read), so its tests are skipped on gfx942 via CK_TILE_TEST_NO_DGRAD.
    add_direct_conv_test(test_direct_conv_grouped_4c_fp16_tile_conv test_grouped_4c_fp16_tile_conv.cpp
        DEFINES CK_TILE_TEST_NO_DGRAD)
    add_direct_conv_test(test_direct_conv_grouped_4c_bf16_tile_conv test_grouped_4c_bf16_tile_conv.cpp
        DEFINES CK_TILE_TEST_NO_DGRAD)
    add_direct_conv_test(test_direct_conv_lds_bank_conflicts test_lds_bank_conflicts.cpp)
    # The 16c weight loader stages 72KB into LDS (> gfx942's 64KB limit); 16c is
    # a gfx950-only variant, so disable its tests on gfx942.
    add_direct_conv_test(test_direct_conv_weight_loader test_weight_loader.cpp
        WARNINGS -Wno-float-equal DEFINES CK_TILE_TEST_NO_16C)
    add_direct_conv_test(test_direct_conv_input_loader test_input_loader.cpp WARNINGS -Wno-float-equal)
    add_direct_conv_test(test_direct_conv_output_writer test_output_writer.cpp WARNINGS -Wno-float-equal)

    add_direct_conv_umbrella_target(
        test_direct_conv_grouped_4c_fp16_tile_conv
        test_direct_conv_grouped_4c_bf16_tile_conv
        test_direct_conv_lds_bank_conflicts
        test_direct_conv_input_loader
        test_direct_conv_weight_loader
        test_direct_conv_output_writer
    )
else()
    message(INFO "No CK Tile direct convolution integration tests will be built for GPU targets: ${GPU_TARGETS}")
endif()
