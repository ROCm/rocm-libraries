# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Keep in sync with example/ck_tile/01_fmha/CMakeLists.txt
if(NOT SUPPORTED_GPU_TARGETS MATCHES "gfx9|gfx12")
    return()
endif()

set(FMHA_BWD_INSTANCES "tile_fmha_bwd_instances")
set(FMHA_FWD_INSTANCES "tile_fmha_fwd_instances")

set(TEST_NAME "test_ck_tile_fmha")

function(add_gtest_fwd test_group)
    set(V_TYPES "fp16" "bf16" "fp8bf16" "fp32" "mxfp8" "mxfp4")
    if(GPU_TARGETS MATCHES "gfx908|gfx90a" AND NOT CK_USE_FP8_ON_UNSUPPORTED_ARCH)
        # fp8 instances are built for all gfx9, do not test on archs without hardware support
        list(REMOVE_ITEM V_TYPES "fp8bf16")
    endif()
    set(CPP_TYPE_fp16 "FmhaFwdFp16")
    set(CPP_TYPE_bf16 "FmhaFwdBf16")
    set(CPP_TYPE_fp8bf16 "FmhaFwdFp8Bf16")
    set(CPP_TYPE_fp32 "FmhaFwdFp32")
    set(CPP_TYPE_mxfp8 "FmhaFwdMxFp8")
    set(CPP_TYPE_mxfp4 "FmhaFwdMxFp4")

    set(sources)
    if(TARGET ${FMHA_FWD_INSTANCES})
        get_target_property(sources ${FMHA_FWD_INSTANCES} SOURCES)
        message(VERBOSE "${FMHA_FWD_INSTANCES} SOURCES ${sources}")
    endif()

    set(all_tests)
    foreach(type ${V_TYPES})
        set(name "${test_group}_${type}")
        if(NOT sources MATCHES "_${type}_")
            message(STATUS "No FMHA FWD instances for ${type}, skip ${name}")
            continue()
        endif()
        add_gtest_executable(${name} test_fmha_fwd.cpp)
        get_test_property(${name} LABELS COMMON_LABELS)
        set_tests_properties(${name} PROPERTIES LABELS "${COMMON_LABELS};${TEST_NAME};${test_group};CK_TILE_FMHA_TESTS")
        # increase timeout limit for fmha_fwd tests to prevent timeout on a slower gfx11 machine
        set_tests_properties(${name} PROPERTIES TIMEOUT 3600)
        target_compile_definitions(${name} PRIVATE DataTypeConfig=${CPP_TYPE_${type}})
        target_link_libraries(${name} PRIVATE ${FMHA_FWD_INSTANCES})
        list(APPEND all_tests ${name})
    endforeach()
    message(STATUS "FMHA FWD tests: ${all_tests}")
    add_custom_target(${test_group} DEPENDS ${all_tests})
endfunction()

function(add_gtest_bwd test_group)
    set(V_TYPES "fp16" "bf16" "fp32")
    set(CPP_TYPE_fp16 "FmhaBwdFp16")
    set(CPP_TYPE_bf16 "FmhaBwdBf16")
    set(CPP_TYPE_fp32 "FmhaBwdFp32")

    set(sources)
    if(TARGET ${FMHA_BWD_INSTANCES})
        get_target_property(sources ${FMHA_BWD_INSTANCES} SOURCES)
        message(VERBOSE "${FMHA_BWD_INSTANCES} SOURCES ${sources}")
    endif()

    set(all_tests)
    foreach(type ${V_TYPES})
        set(name "${test_group}_${type}")
        if(NOT sources MATCHES "_${type}_")
            message(STATUS "No FMHA BWD instances for ${type}, skip ${name}")
            continue()
        endif()
        add_gtest_executable(${name} test_fmha_bwd.cpp)
        get_test_property(${name} LABELS COMMON_LABELS)
        set_tests_properties(${name} PROPERTIES LABELS "${COMMON_LABELS};${TEST_NAME};${test_group};CK_TILE_FMHA_TESTS")
        # increase timeout limit for fmha_bwd tests to prevent occasional timeout on slower machines
        set_tests_properties(${name} PROPERTIES TIMEOUT 3600)
        target_compile_definitions(${name} PRIVATE DataTypeConfig=${CPP_TYPE_${type}})
        target_link_libraries(${name} PRIVATE ${FMHA_BWD_INSTANCES})
        list(APPEND all_tests ${name})
    endforeach()
    message(STATUS "FMHA BWD tests: ${all_tests}")
    add_custom_target(${test_group} DEPENDS ${all_tests})
endfunction()


function(add_gtest_fwd_quick)
    if(NOT TARGET ${FMHA_FWD_INSTANCES})
        return()
    endif()

    set(quick_instance_target tile_fmha_fwd_quick_instances)
    set(quick_test_target test_ck_tile_fmha_fwd_quick)
    set(quick_codegen_dir "${CMAKE_CURRENT_BINARY_DIR}/quick_codegen")
    set(quick_blob_list "${quick_codegen_dir}/fwd_blob_list.txt")

    set(quick_targets ${SUPPORTED_GPU_TARGETS})
    list(FILTER quick_targets INCLUDE REGEX "gfx9|gfx1[12]")
    list(JOIN quick_targets "," quick_targets_arg)

    # Generate only the FP16 traits exercised below. The API-specific filters
    # are ordered as fwd, split-KV combine@kernel, append-KV, paged-KV. The
    # regular FWD filter admits d32 plus d64 for the head-dimension padding
    # case; gfx9 paged-KV starts at d128.
    set(quick_codegen_filter
        "fmha_fwd_d[36][24]_fp16_*_nlogits*_nskip*_nsink*,*d32_fp16_batch*lse*@*d32_fp16_batch*_nlogits_nbias_nmask_lse*_npagedkv_nsink*,*d32_fp16*,*d128_fp16_batch*_nlogits_nbias_nmask_nlse_nskip*_pagedkv_nsink*"
    )
    set(quick_codegen_args
        "${PROJECT_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py"
        --targets "${quick_targets_arg}"
        --api "fwd,fwd_splitkv,fwd_appendkv,pagedkv_prefill"
        --optdim 32,64,128
        --filter "${quick_codegen_filter}"
    )
    file(GLOB_RECURSE quick_codegen_scripts CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py"
        "${PROJECT_SOURCE_DIR}/example/ck_tile/01_fmha/codegen/*.py"
    )

    file(MAKE_DIRECTORY "${quick_codegen_dir}")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${quick_codegen_args}
                --list_blobs "${quick_blob_list}"
        RESULT_VARIABLE quick_codegen_result
    )
    if(quick_codegen_result AND NOT quick_codegen_result EQUAL 0)
        message(FATAL_ERROR "Failed to list quick CK-Tile FMHA kernels")
    endif()
    file(STRINGS "${quick_blob_list}" quick_fmha_blobs)

    add_custom_command(
        OUTPUT ${quick_fmha_blobs}
        COMMAND ${Python3_EXECUTABLE} ${quick_codegen_args}
                --output_dir "${quick_codegen_dir}"
        DEPENDS ${quick_codegen_scripts}
        COMMENT "Generate quick CK-Tile FMHA FWD kernels"
    )

    add_library(${quick_instance_target} OBJECT EXCLUDE_FROM_ALL)
    target_sources(${quick_instance_target} PRIVATE ${quick_fmha_blobs})
    set_source_files_properties(${quick_fmha_blobs} PROPERTIES LANGUAGE HIP)
    set_property(
        TARGET ${quick_instance_target} PROPERTY HIP_ARCHITECTURES ${quick_targets}
    )
    target_include_directories(
        ${quick_instance_target}
        PRIVATE "${PROJECT_SOURCE_DIR}/example/ck_tile/01_fmha"
    )

    get_target_property(
        quick_private_options ${FMHA_FWD_INSTANCES} COMPILE_OPTIONS
    )
    get_target_property(
        quick_interface_options ${FMHA_FWD_INSTANCES} INTERFACE_COMPILE_OPTIONS
    )
    target_compile_options(
        ${quick_instance_target}
        PRIVATE ${quick_private_options}
        INTERFACE ${quick_interface_options}
    )

    set_source_files_properties(test_fmha_fwd.cpp PROPERTIES LANGUAGE HIP)
    add_executable(${quick_test_target} test_fmha_fwd.cpp)
    set_property(TARGET ${quick_test_target} PROPERTY HIP_ARCHITECTURES ${quick_targets})
    target_compile_options(
        ${quick_test_target}
        PRIVATE -Wno-global-constructors -Wno-undef
    )
    target_compile_definitions(${quick_test_target} PRIVATE DataTypeConfig=FmhaFwdFp16)
    target_link_libraries(
        ${quick_test_target}
        PRIVATE ${quick_instance_target} gtest_main getopt::getopt
    )
    add_dependencies(tests ${quick_test_target})
    add_dependencies(check ${quick_test_target})
    rocm_install(TARGETS ${quick_test_target} COMPONENT tests)

    # Vary batch/group mode and input permutation, then sample head-dimension
    # padding, bias/dropout, and the paged/split/append KV APIs.
    set(quick_cases
        "TestCkTileFmhaFwd/General.FmhaFwdFp16/0"
        "TestCkTileFmhaFwd/General.FmhaFwdFp16/7"
        "TestCkTileFmhaFwd/General.FmhaFwdFp16/14"
        "TestCkTileFmhaFwd/HDimPadding.FmhaFwdFp16/0"
        "TestCkTileFmhaFwd/ElementwiseBias.FmhaFwdFp16/0"
        "TestCkTileFmhaFwd/Alibi.FmhaFwdFp16/0"
        "TestCkTileFmhaFwd/Dropout.FmhaFwdFp16/0"
        "TestCkTileFmhaFwd/PagedKV.FmhaFwdFp16/72"
        "TestCkTileFmhaFwd/SplitKV.FmhaFwdFp16/0"
        "TestCkTileFmhaFwd/AppendKV.FmhaFwdFp16/1"
    )
    list(JOIN quick_cases ":" quick_filter)
    add_ck_tile_category_test(
        test_ck_tile_fmha_quick
        ${quick_test_target}
        "--gtest_filter=${quick_filter}"
    )
    set_property(
        TEST test_ck_tile_fmha_quick APPEND PROPERTY LABELS CK_TILE_FMHA_TESTS
    )
endfunction()


add_gtest_fwd(${TEST_NAME}_fwd)
add_gtest_bwd(${TEST_NAME}_bwd)
add_custom_target(${TEST_NAME} DEPENDS ${TEST_NAME}_fwd ${TEST_NAME}_bwd)
add_gtest_fwd_quick()

# Umbrella target to build and run all ck_tile fmha tests
# Usage: ninja ck_tile_fmha_tests
add_custom_target(ck_tile_fmha_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR} -L "CK_TILE_FMHA_TESTS"
    DEPENDS ${TEST_NAME}
    USES_TERMINAL
    COMMENT "Running all ck_tile fmha tests..."
)
