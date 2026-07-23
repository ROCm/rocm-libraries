# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

################################################################################
# CK Tile Test Organization
################################################################################
# CK Tile tests can be run using several methods:
#
# 1. Global test labels (run tests across all operations):
#    - ninja smoke       - Fast tests (~30s on gfx90a)
#    - ninja regression  - Slower comprehensive tests
#    - ninja check       - All available tests
#
# 2. Operation-specific umbrella targets (run all tests for a specific operation):
#    - ninja ck_tile_gemm_tests                  - All basic GEMM tests
#    - ninja ck_tile_gemm_block_scale_tests      - All GEMM with block-scale quantization tests
#    - ninja ck_tile_gemm_streamk_tests          - All GEMM StreamK tests
#    - ninja ck_tile_grouped_gemm_quant_tests    - All grouped GEMM quantization tests
#    - ninja ck_tile_reduce_tests                - All reduce operation tests
#    - ninja ck_tile_fmha_tests                  - All FMHA (Flash Attention) tests
#
# 3. Individual test executables:
#    - ninja test_<test_name>                    - Build specific test executable
#    - ./build/bin/test_<test_name>              - Run specific test directly
#
# These umbrella targets are useful when working on specific operations to quickly
# validate all related tests without running the entire test suite.
################################################################################

add_subdirectory(image_to_column)
add_subdirectory(gemm)
add_subdirectory(gemm_persistent_async_input)
add_subdirectory(gemm_weight_preshuffle)
add_subdirectory(batched_gemm)
add_subdirectory(grouped_gemm)
add_subdirectory(grouped_gemm_preshuffle)
add_subdirectory(grouped_gemm_multi_d)
add_subdirectory(grouped_gemm_quant)
add_subdirectory(grouped_gemm_abquant)
add_subdirectory(gemm_multi_d)
add_subdirectory(gemm_multi_abd)
add_subdirectory(gemm_streamk)
add_subdirectory(data_type)
add_subdirectory(container)
add_subdirectory(elementwise)
# Not including these tests as there is a bug on gfx90a and gfx942
# resulting in "GPU core dump"
#add_subdirectory(moe_smoothquant)
add_subdirectory(permute)
add_subdirectory(moe_sorting)
add_subdirectory(slice_tile)
add_subdirectory(memory_copy)
add_subdirectory(batched_transpose)
add_subdirectory(smoothquant)
add_subdirectory(topk_softmax)
add_subdirectory(add_rmsnorm2d_rdquant)
# add_subdirectory(layernorm2d)
# add_subdirectory(rmsnorm2d)
add_subdirectory(gemm_block_scale)
add_subdirectory(flatmm)
add_subdirectory(gemm_mx)
add_subdirectory(grouped_gemm_mx)
add_subdirectory(utility)
add_subdirectory(warp_gemm)
add_subdirectory(reduce)
add_subdirectory(tdm)
add_subdirectory(core)
add_subdirectory(epilogue)
add_subdirectory(atomic_add_op)
if(BUILD_CK_TILE_FMHA_TESTS)
    add_subdirectory(fmha)
endif()
if(BUILD_CK_TILE_ENGINE_TESTS)
# TODO: The Universal GEMM tile engine test will be either removed
# or moved to the appropriate location in future work.
#   add_subdirectory(gemm_tile_engine)
    add_subdirectory(pooling_tile_engine)
endif()
add_subdirectory(pooling)
add_subdirectory(grouped_conv)
add_subdirectory(load_and_convert_tile)
add_subdirectory(multicast_load)
add_subdirectory(quick_examples)

# Compile only the executable targets used by quick tests. This is kept
# separate from execution so CPU-only builders can produce the test payload
# before a GPU job runs CTest. Direct quick tests and filtered aliases are both
# included; duplicate backing executables are collapsed.
get_property(_ck_tile_quick_targets GLOBAL PROPERTY CK_TILE_CTEST_ALIAS_TARGETS)
foreach(test_name IN LISTS CK_TILE_CTEST_QUICK_TESTS)
    if(TARGET "${test_name}")
        list(APPEND _ck_tile_quick_targets "${test_name}")
    endif()
endforeach()
if(_ck_tile_quick_targets)
    list(REMOVE_DUPLICATES _ck_tile_quick_targets)
    add_custom_target(ck_tile_quick_build DEPENDS ${_ck_tile_quick_targets})
endif()

# Install a CTest entry point for the curated tiered subset. Test executables
# are already installed by add_test_executable()/add_gtest_executable().
if(COMMAND apply_ctest_category_labels)
    set(CK_TILE_INSTALLED_CTEST_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/install_CTestTestfile.cmake"
    )
    set(CK_TILE_INSTALLED_CTEST_TESTS "")
    set(_ck_tile_installed_test_lines "")

    foreach(test_name IN LISTS CK_TILE_CTEST_CATEGORY_TESTS)
        if(TARGET "${test_name}")
            list(APPEND _ck_tile_installed_test_lines
                "add_test(\"${test_name}\" \"../${test_name}${CMAKE_EXECUTABLE_SUFFIX}\")"
            )
            list(APPEND CK_TILE_INSTALLED_CTEST_TESTS "${test_name}")
        endif()
    endforeach()

    get_property(_ck_tile_alias_tests GLOBAL PROPERTY CK_TILE_CTEST_ALIAS_TESTS)
    foreach(test_name IN LISTS _ck_tile_alias_tests)
        get_property(
            target_name GLOBAL PROPERTY "CK_TILE_CTEST_ALIAS_TARGET_${test_name}"
        )
        get_property(
            command_args GLOBAL PROPERTY "CK_TILE_CTEST_ALIAS_ARGS_${test_name}"
        )

        set(_command_parts
            "add_test(\"${test_name}\""
            "\"../${target_name}${CMAKE_EXECUTABLE_SUFFIX}\""
        )
        foreach(command_arg IN LISTS command_args)
            list(APPEND _command_parts "\"${command_arg}\"")
        endforeach()
        list(APPEND _command_parts ")")
        list(JOIN _command_parts " " _command_line)
        list(APPEND _ck_tile_installed_test_lines "${_command_line}")
        list(APPEND CK_TILE_INSTALLED_CTEST_TESTS "${test_name}")
    endforeach()

    if(CK_TILE_INSTALLED_CTEST_TESTS)
        list(JOIN _ck_tile_installed_test_lines "\n" CK_TILE_INSTALLED_CTEST_ENTRIES)
        string(CONFIGURE [=[
# Autogenerated CTestTestfile for installed CK-Tile tests

@CK_TILE_INSTALLED_CTEST_ENTRIES@
]=] _ck_tile_installed_ctest_content @ONLY)
        file(WRITE
            "${CK_TILE_INSTALLED_CTEST_FILE}"
            "${_ck_tile_installed_ctest_content}"
        )

        apply_ctest_category_labels(
            "${CK_TILE_TEST_CATEGORIES_YAML}"
            INSTALL_TEST_FILE "${CK_TILE_INSTALLED_CTEST_FILE}"
        )
        install(
            FILES "${CK_TILE_INSTALLED_CTEST_FILE}"
            RENAME CTestTestfile.cmake
            DESTINATION "${CMAKE_INSTALL_BINDIR}/composablekernel"
            COMPONENT tests
        )
    endif()
endif()
