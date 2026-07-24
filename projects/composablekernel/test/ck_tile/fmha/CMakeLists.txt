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


function(add_fmha_quick_codegen
         quick_instance_target full_instance_target quick_api quick_optdim
         quick_codegen_filter)
    if(NOT TARGET ${full_instance_target})
        return()
    endif()

    set(quick_targets ${ARGN})
    list(JOIN quick_targets "," quick_targets_arg)
    set(quick_codegen_dir
        "${CMAKE_CURRENT_BINARY_DIR}/${quick_instance_target}_codegen"
    )
    set(quick_blob_list "${quick_codegen_dir}/blob_list.txt")

    set(quick_codegen_args
        "${PROJECT_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py"
        --targets "${quick_targets_arg}"
        --api "${quick_api}"
        --optdim "${quick_optdim}"
        --filter "${quick_codegen_filter}"
    )
    file(GLOB_RECURSE quick_codegen_scripts CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/example/ck_tile/01_fmha/generate.py"
        "${PROJECT_SOURCE_DIR}/example/ck_tile/01_fmha/codegen/*.py"
    )

    file(MAKE_DIRECTORY "${quick_codegen_dir}")
    file(REMOVE "${quick_blob_list}")
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
        COMMENT "Generate quick CK-Tile FMHA kernels"
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
        quick_private_options ${full_instance_target} COMPILE_OPTIONS
    )
    get_target_property(
        quick_interface_options ${full_instance_target} INTERFACE_COMPILE_OPTIONS
    )
    if("${full_instance_target}" STREQUAL "${FMHA_FWD_INSTANCES}")
        foreach(api_option
                "fwd_splitkv;CK_TILE_FMHA_FWD_SPLITKV_API"
                "fwd_appendkv;CK_TILE_FMHA_FWD_APPENDKV_API"
                "pagedkv_prefill;CK_TILE_FMHA_FWD_PAGEDKV_API"
                "batch_prefill;CK_TILE_FMHA_FWD_BATCH_PREFILL_API")
            list(GET api_option 0 api_name)
            list(GET api_option 1 api_macro)
            list(REMOVE_ITEM quick_interface_options
                "-D${api_macro}=0" "-D${api_macro}=1"
            )
            if("${quick_api}" MATCHES "(^|,)${api_name}(,|$)")
                list(APPEND quick_interface_options "-D${api_macro}=1")
            else()
                list(APPEND quick_interface_options "-D${api_macro}=0")
            endif()
        endforeach()
    endif()
    target_compile_options(
        ${quick_instance_target}
        PRIVATE ${quick_private_options}
        INTERFACE ${quick_interface_options}
    )
endfunction()

function(add_fmha_quick_executable
         quick_test_target quick_source data_type_config quick_instance_target)
    get_target_property(quick_targets ${quick_instance_target} HIP_ARCHITECTURES)
    set_source_files_properties(${quick_source} PROPERTIES LANGUAGE HIP)
    add_executable(${quick_test_target} ${quick_source})
    set_property(TARGET ${quick_test_target} PROPERTY HIP_ARCHITECTURES ${quick_targets})
    target_compile_options(
        ${quick_test_target}
        PRIVATE -Wno-global-constructors -Wno-undef
    )
    target_compile_definitions(
        ${quick_test_target} PRIVATE DataTypeConfig=${data_type_config}
    )
    target_link_libraries(
        ${quick_test_target}
        PRIVATE ${quick_instance_target} gtest_main getopt::getopt
    )
    add_dependencies(tests ${quick_test_target})
    add_dependencies(check ${quick_test_target})
    rocm_install(TARGETS ${quick_test_target} COMPONENT tests)
endfunction()

function(add_fmha_quick_category_test alias_name quick_test_target)
    set(quick_cases ${ARGN})
    list(JOIN quick_cases ":" quick_filter)
    add_ck_tile_category_test(
        ${alias_name}
        ${quick_test_target}
        "--gtest_filter=${quick_filter}"
    )
    set_property(TEST ${alias_name} APPEND PROPERTY LABELS CK_TILE_FMHA_TESTS)
endfunction()

function(add_gtest_fwd_quick)
    if(NOT TARGET ${FMHA_FWD_INSTANCES})
        return()
    endif()

    set(quick_targets ${SUPPORTED_GPU_TARGETS})
    list(FILTER quick_targets INCLUDE REGEX "gfx9|gfx1[12]")

    set(quick_instance_target tile_fmha_fwd_quick_instances)
    set(quick_test_target test_ck_tile_fmha_fwd_quick)
    # Generate only the FP16 feature traits exercised below. The API-specific
    # filters are ordered as fwd, split-KV combine@kernel, append-KV, paged-KV.
    set(quick_codegen_filter
        "fmha_fwd_d[36][24]_fp16_*_nlogits*_nskip*_nsink*,*d32_fp16_batch*lse*@*d32_fp16_batch*_nlogits_nbias_nmask_lse*_npagedkv_nsink*,*d32_fp16*,*d128_fp16_*_nlogits_nbias_*mask_nlse_nskip*_pagedkv_nsink*"
    )
    add_fmha_quick_codegen(
        ${quick_instance_target}
        ${FMHA_FWD_INSTANCES}
        "fwd,fwd_splitkv,fwd_appendkv,pagedkv_prefill"
        "32,64,128"
        "${quick_codegen_filter}"
        ${quick_targets}
    )
    add_fmha_quick_executable(
        ${quick_test_target} test_fmha_fwd.cpp FmhaFwdFp16
        ${quick_instance_target}
    )

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
        "TestCkTileFmhaFwd/SplitKV.FmhaFwdFp16/0"
        "TestCkTileFmhaFwd/AppendKV.FmhaFwdFp16/1"
    )
    add_fmha_quick_category_test(
        test_ck_tile_fmha_quick
        ${quick_test_target}
        ${quick_cases}
    )

    # Keep paged attention visible as its own quick entry. These cases span
    # batch/group modes, 128/256-token pages, contiguous/permuted KV layouts,
    # and regular plus masked/odd sequence shapes while sharing one d128
    # bounded code-generation family.
    add_fmha_quick_category_test(
        test_ck_tile_fmha_paged_quick
        ${quick_test_target}
        "TestCkTileFmhaFwd/PagedKV.FmhaFwdFp16/72"
        "TestCkTileFmhaFwd/PagedKV.FmhaFwdFp16/75"
        "TestCkTileFmhaFwd/PagedKV.FmhaFwdFp16/83"
        "TestCkTileFmhaFwd/PagedKV.FmhaFwdFp16/84"
        "TestCkTileFmhaFwd/PagedKV.FmhaFwdFp16/94"
    )
endfunction()

function(add_gtest_fwd_tuning_quick)
    set(quick_targets ${SUPPORTED_GPU_TARGETS})
    list(FILTER quick_targets INCLUDE REGEX "gfx9|gfx1[12]")

    set(quick_instance_target tile_fmha_fwd_tuning_quick_instances)
    set(quick_test_target test_ck_tile_fmha_fwd_tuning_quick_bin)
    set(quick_codegen_filter
        "fmha_fwd_d*_fp16_*_nlogits_nbias_nmask_lse_ndropout_nskip_nqscale_ntrload_nsink*"
    )
    add_fmha_quick_codegen(
        ${quick_instance_target} ${FMHA_FWD_INSTANCES} "fwd" "32,64,128,256"
        "${quick_codegen_filter}" ${quick_targets}
    )
    add_fmha_quick_executable(
        ${quick_test_target} test_fmha_fwd.cpp FmhaFwdFp16
        ${quick_instance_target}
    )

    set(quick_cases)
    foreach(case_index 28 35 84 91 168 175)
        list(APPEND quick_cases
            "TestCkTileFmhaFwd/General.FmhaFwdFp16/${case_index}"
        )
    endforeach()
    add_fmha_quick_category_test(
        test_ck_tile_fmha_fwd_tuning_quick ${quick_test_target} ${quick_cases}
    )

    # The asymmetric 192x128 QK/V head dimensions are representative of
    # multi-latent-attention decode. Reuse the d256 padded instances above.
    add_fmha_quick_category_test(
        test_ck_tile_fmha_mla_quick
        ${quick_test_target}
        "TestCkTileFmhaFwd/General.FmhaFwdFp16/112"
        "TestCkTileFmhaFwd/General.FmhaFwdFp16/119"
    )
endfunction()

function(add_gtest_fwd_sink_quick)
    set(quick_targets ${SUPPORTED_GPU_TARGETS})
    list(FILTER quick_targets INCLUDE REGEX "gfx9|gfx1[12]")

    set(quick_instance_target tile_fmha_fwd_sink_quick_instances)
    set(quick_test_target test_ck_tile_fmha_fwd_sink_quick_bin)
    set(quick_codegen_filter
        "fmha_fwd_d128_fp16_*_nlogits_nbias_*mask_lse_ndropout_nskip_nqscale_ntrload_sink*"
    )
    add_fmha_quick_codegen(
        ${quick_instance_target} ${FMHA_FWD_INSTANCES} "fwd" "128"
        "${quick_codegen_filter}" ${quick_targets}
    )
    add_fmha_quick_executable(
        ${quick_test_target} test_fmha_fwd.cpp FmhaFwdFp16
        ${quick_instance_target}
    )
    target_compile_definitions(
        ${quick_test_target} PRIVATE CK_TILE_FMHA_FWD_SINK_QUICK_TESTS=1
    )
    add_fmha_quick_category_test(
        test_ck_tile_fmha_fwd_sink_quick
        ${quick_test_target}
        "TestCkTileFmhaFwdSinkQuick.StreamLlmMaskBatch"
        "TestCkTileFmhaFwdSinkQuick.LearnedSinkGroup"
    )
endfunction()

function(add_gtest_fwd_fp8_quick)
    set(quick_targets ${SUPPORTED_GPU_TARGETS})
    list(FILTER quick_targets INCLUDE REGEX "gfx9|gfx1[12]")
    list(FILTER quick_targets EXCLUDE REGEX "gfx908|gfx90a")
    if(NOT quick_targets)
        return()
    endif()

    set(quick_instance_target tile_fmha_fwd_fp8_quick_instances)
    set(quick_test_target test_ck_tile_fmha_fwd_fp8_quick_bin)
    set(quick_codegen_filter
        "fmha_fwd_d*_fp8bf16_*_nlogits_nbias_nmask_nlse_ndropout_nskip_pertensor_ntrload_nsink*"
    )
    add_fmha_quick_codegen(
        ${quick_instance_target} ${FMHA_FWD_INSTANCES} "fwd" "64,128,256"
        "${quick_codegen_filter}" ${quick_targets}
    )
    add_fmha_quick_executable(
        ${quick_test_target} test_fmha_fwd.cpp FmhaFwdFp8Bf16
        ${quick_instance_target}
    )

    set(quick_cases)
    foreach(case_index 0 7 28 35 56 63)
        list(APPEND quick_cases
            "TestCkTileFmhaFwd/General.FmhaFwdFp8Bf16/${case_index}"
        )
    endforeach()
    add_fmha_quick_category_test(
        test_ck_tile_fmha_fwd_fp8_quick ${quick_test_target} ${quick_cases}
    )
endfunction()

function(add_gtest_bwd_tuning_quick)
    set(quick_targets ${SUPPORTED_GPU_TARGETS})
    list(FILTER quick_targets INCLUDE REGEX "gfx9|gfx1[12]")

    set(quick_instance_target tile_fmha_bwd_tuning_quick_instances)
    set(quick_test_target test_ck_tile_fmha_bwd_tuning_quick_bin)
    set(quick_codegen_filter
        "fmha_bwd_dot_do_o_d*_fp16_*@fmha_bwd_convert_dq_d*_fp16_*_ndeterministic@fmha_bwd_d*_fp16_*_npad_nbias_ndbias_*mask_ndropout_ndeterministic_ntrload*"
    )
    add_fmha_quick_codegen(
        ${quick_instance_target} ${FMHA_BWD_INSTANCES} "bwd" "32,64,128,256"
        "${quick_codegen_filter}" ${quick_targets}
    )
    add_fmha_quick_executable(
        ${quick_test_target} test_fmha_bwd.cpp FmhaBwdFp16
        ${quick_instance_target}
    )

    set(quick_cases)
    foreach(case_index RANGE 0 15)
        list(APPEND quick_cases
            "TestCkTileFmhaBwd/Tuning.FmhaBwdFp16/${case_index}"
        )
    endforeach()
    add_fmha_quick_category_test(
        test_ck_tile_fmha_bwd_tuning_quick ${quick_test_target} ${quick_cases}
    )

    # Sink-gradient uses the same backward kernels; select group-mode d64/d128
    # regression cases without expanding the generated source graph.
    add_fmha_quick_category_test(
        test_ck_tile_fmha_bwd_sink_quick
        ${quick_test_target}
        "TestCkTileFmhaBwd/SinkGradGroupMode.FmhaBwdFp16/5"
        "TestCkTileFmhaBwd/SinkGradGroupMode.FmhaBwdFp16/7"
        "TestCkTileFmhaBwd/SinkGradGroupMode.FmhaBwdFp16/10"
        "TestCkTileFmhaBwd/SinkGradGroupMode.FmhaBwdFp16/12"
    )
endfunction()


add_gtest_fwd(${TEST_NAME}_fwd)
add_gtest_bwd(${TEST_NAME}_bwd)
add_custom_target(${TEST_NAME} DEPENDS ${TEST_NAME}_fwd ${TEST_NAME}_bwd)
add_gtest_fwd_quick()
add_gtest_fwd_tuning_quick()
add_gtest_fwd_fp8_quick()
add_gtest_fwd_sink_quick()
add_gtest_bwd_tuning_quick()

# Umbrella target to build and run all ck_tile fmha tests
# Usage: ninja ck_tile_fmha_tests
add_custom_target(ck_tile_fmha_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR} -L "CK_TILE_FMHA_TESTS"
    DEPENDS ${TEST_NAME}
    USES_TERMINAL
    COMMENT "Running all ck_tile fmha tests..."
)
