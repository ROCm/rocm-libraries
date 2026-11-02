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
    # fp8/fp8fp32 targets and the quantization-scale sweep only have instances on gfx125x.
    # Keep them off every other target so their test time does not grow.
    if(GPU_TARGETS MATCHES "gfx125")
        list(APPEND V_TYPES "fp8" "fp8fp32")
    endif()
    if(GPU_TARGETS MATCHES "gfx908|gfx90a" AND NOT CK_USE_FP8_ON_UNSUPPORTED_ARCH)
        # fp8 instances are built for all gfx9, do not test on archs without hardware support
        list(REMOVE_ITEM V_TYPES "fp8bf16")
    endif()
    set(CPP_TYPE_fp16 "FmhaFwdFp16")
    set(CPP_TYPE_bf16 "FmhaFwdBf16")
    set(CPP_TYPE_fp8 "FmhaFwdFp8")
    set(CPP_TYPE_fp8bf16 "FmhaFwdFp8Bf16")
    set(CPP_TYPE_fp8fp32 "FmhaFwdFp8Fp32")
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
        # fp8-input types with a wider output only; an fp8 output cannot meet the test's bound.
        if(GPU_TARGETS MATCHES "gfx125" AND (type STREQUAL "fp8bf16" OR type STREQUAL "fp8fp32"))
            set_source_files_properties(test_fmha_fwd_sink_exact.cpp PROPERTIES LANGUAGE HIP)
            target_sources(${name} PRIVATE test_fmha_fwd_sink_exact.cpp)
        endif()
        get_test_property(${name} LABELS COMMON_LABELS)
        set_tests_properties(${name} PROPERTIES LABELS "${COMMON_LABELS};${TEST_NAME};${test_group};CK_TILE_FMHA_TESTS")
        # increase timeout limit for fmha_fwd tests to prevent timeout on a slower gfx11 machine
        set_tests_properties(${name} PROPERTIES TIMEOUT 3600)
        target_compile_definitions(${name} PRIVATE DataTypeConfig=${CPP_TYPE_${type}})
        if(GPU_TARGETS MATCHES "gfx125")
            target_compile_definitions(${name} PRIVATE CK_TILE_TEST_FMHA_QSCALE_SWEEP)
            target_compile_definitions(${name} PRIVATE CK_TILE_TEST_FMHA_GPTOSS_SINK)
        endif()
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


add_gtest_fwd(${TEST_NAME}_fwd)
add_gtest_bwd(${TEST_NAME}_bwd)
add_custom_target(${TEST_NAME} DEPENDS ${TEST_NAME}_fwd ${TEST_NAME}_bwd)

set(FMHA_QR_TDM_TEST_TARGETS)
if(SUPPORTED_GPU_TARGETS MATCHES "gfx125")
    add_gtest_executable(test_ck_tile_fmha_bwd_tdm_padding
        test_fmha_bwd_tdm_padding.cpp
        test_fmha_bwd_tdm_layout.cpp)
    set_tests_properties(test_ck_tile_fmha_bwd_tdm_padding PROPERTIES
        LABELS "${TEST_NAME};CK_TILE_FMHA_TESTS")
    add_gtest_executable(test_ck_tile_fmha_qr_tdm_lds_padding
        test_qr_tdm_lds_padding.cpp)
    target_link_libraries(test_ck_tile_fmha_qr_tdm_lds_padding PRIVATE utility)
    # the qr_tdm pipeline folds log2(e) into scale_s and has no exp() fallback
    target_compile_definitions(test_ck_tile_fmha_qr_tdm_lds_padding
        PRIVATE CK_TILE_FMHA_FWD_FAST_EXP2=1)
    set_tests_properties(test_ck_tile_fmha_qr_tdm_lds_padding PROPERTIES
        LABELS "${TEST_NAME};CK_TILE_FMHA_TESTS")

    add_gtest_executable(test_ck_tile_fmha_qr_tdm_progressive_k_lds
        test_qr_tdm_progressive_k_lds.cpp)
    # the qr_tdm pipeline folds log2(e) into scale_s and has no exp() fallback
    target_compile_definitions(test_ck_tile_fmha_qr_tdm_progressive_k_lds
        PRIVATE CK_TILE_FMHA_FWD_FAST_EXP2=1)
    # The causal pipeline deliberately compares the masked-row -infinity sentinel.
    target_compile_options(test_ck_tile_fmha_qr_tdm_progressive_k_lds PRIVATE -Wno-float-equal)
    set_tests_properties(test_ck_tile_fmha_qr_tdm_progressive_k_lds PROPERTIES
        LABELS "${TEST_NAME};CK_TILE_FMHA_TESTS")

    add_gtest_executable(test_ck_tile_fmha_qr_tdm_causal_mapping
        test_qr_tdm_causal_mapping.cpp)
    target_compile_definitions(test_ck_tile_fmha_qr_tdm_causal_mapping
        PRIVATE CK_TILE_FMHA_FWD_FAST_EXP2=1)
    set_tests_properties(test_ck_tile_fmha_qr_tdm_causal_mapping PROPERTIES
        LABELS "${TEST_NAME};CK_TILE_FMHA_TESTS")

    add_gtest_executable(test_ck_tile_fmha_qr_tdm_row_reduce
        test_qr_tdm_row_reduce.cpp)
    target_compile_definitions(test_ck_tile_fmha_qr_tdm_row_reduce
        PRIVATE CK_TILE_FMHA_FWD_FAST_EXP2=1)
    # Hand-derived sums/maxima are exactly representable integers; require exact equality.
    target_compile_options(test_ck_tile_fmha_qr_tdm_row_reduce PRIVATE -Wno-float-equal)
    set_tests_properties(test_ck_tile_fmha_qr_tdm_row_reduce PROPERTIES
        LABELS "${TEST_NAME};CK_TILE_FMHA_TESTS")

    set(FMHA_QR_TDM_TEST_TARGETS
        test_ck_tile_fmha_bwd_tdm_padding
        test_ck_tile_fmha_qr_tdm_lds_padding
        test_ck_tile_fmha_qr_tdm_progressive_k_lds
        test_ck_tile_fmha_qr_tdm_causal_mapping
        test_ck_tile_fmha_qr_tdm_row_reduce)
endif()

# Scheduled FMHA tests share the existing FMHA gate and umbrella.
set(FMHA_QR_TDM_SCHED_TEST_TARGETS)
if(SUPPORTED_GPU_TARGETS MATCHES "gfx125")
    set(FMHA_QR_TDM_SCHED_GPU_TARGETS ${SUPPORTED_GPU_TARGETS})
    list(FILTER FMHA_QR_TDM_SCHED_GPU_TARGETS INCLUDE REGEX "^gfx125")

    function(add_qr_tdm_sched_gtest target source)
        add_gtest_executable(${target} ${source})
        if(TARGET ${target})
            target_link_libraries(${target} PRIVATE utility)
            target_compile_definitions(${target} PRIVATE CK_TILE_FMHA_FWD_FAST_EXP2=1)
            set_property(TARGET ${target} PROPERTY HIP_ARCHITECTURES
                ${FMHA_QR_TDM_SCHED_GPU_TARGETS})
            get_test_property(${target} LABELS common_labels)
            set_tests_properties(${target} PROPERTIES
                LABELS "${common_labels};${TEST_NAME};CK_TILE_FMHA_TESTS")
            set(FMHA_QR_TDM_SCHED_TEST_TARGETS
                ${FMHA_QR_TDM_SCHED_TEST_TARGETS} ${target} PARENT_SCOPE)
        endif()
    endfunction()

    add_qr_tdm_sched_gtest(test_ck_tile_fmha_qr_tdm_sched_config
        test_qr_tdm_sched_config.cpp)
    add_qr_tdm_sched_gtest(test_ck_tile_fmha_qr_tdm_sched_schedule
        test_qr_tdm_sched_schedule.cpp)

    add_test(NAME test_ck_tile_fmha_qr_tdm_sched_read_first
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_qr_tdm_sched_read_first.py
            --compiler=${CMAKE_HIP_COMPILER}
            --ck-root=${PROJECT_SOURCE_DIR}
            --ck-build=${PROJECT_BINARY_DIR} -v)
    set_tests_properties(test_ck_tile_fmha_qr_tdm_sched_read_first PROPERTIES
        LABELS "${TEST_NAME};CK_TILE_FMHA_TESTS;CK_TILE_FMHA_CODEGEN_TESTS"
        TIMEOUT 180)

    # Check GenericAttentionMask compatibility across pipeline paths with one blob
    # per path (device -fsyntax-only, no GPU); numerical coverage is separate below.
    list(GET FMHA_QR_TDM_SCHED_GPU_TARGETS 0 FMHA_QR_TDM_SCHED_FIRST_GPU_TARGET)
    # The blobs need the configure-level GPU feature macros (CK_USE_GFX1250, CK_TILE_USE_WMMA, ...).
    get_directory_property(FMHA_QR_TDM_SCHED_DEFINITIONS COMPILE_DEFINITIONS)
    string(REPLACE ";" "\n" FMHA_QR_TDM_SCHED_DEFINITIONS "${FMHA_QR_TDM_SCHED_DEFINITIONS}")
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/qr_tdm_sched_generic_mask_defines.txt
        "${FMHA_QR_TDM_SCHED_DEFINITIONS}\n")
    add_test(NAME test_ck_tile_fmha_qr_tdm_sched_generic_mask
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_qr_tdm_sched_generic_mask.py
            --compiler=${CMAKE_HIP_COMPILER}
            --ck-root=${PROJECT_SOURCE_DIR}
            --ck-build=${PROJECT_BINARY_DIR}
            --gpu-target=${FMHA_QR_TDM_SCHED_FIRST_GPU_TARGET}
            --defines-file=${CMAKE_CURRENT_BINARY_DIR}/qr_tdm_sched_generic_mask_defines.txt -v)
    set_tests_properties(test_ck_tile_fmha_qr_tdm_sched_generic_mask PROPERTIES
        LABELS "${TEST_NAME};CK_TILE_FMHA_TESTS;CK_TILE_FMHA_CODEGEN_TESTS"
        TIMEOUT 1200)

    # Numerical coverage uses the actual generated generic mc specializations and
    # dispatcher, with the same compiler options as production. Keep this library
    # separate from the default simplified-mask forward instances.
    if(TARGET ${FMHA_FWD_INSTANCES})
        set(generic_fwd_example ${PROJECT_SOURCE_DIR}/example/ck_tile/01_fmha)
        set(generic_fwd_dir ${CMAKE_CURRENT_BINARY_DIR}/qr_tdm_sched_generic_fwd)
        set(generic_fwd_list ${generic_fwd_dir}/fwd_blob_list.txt)
        set(generic_fwd_filter
            "fmha_fwd_d64_*_batch_b128x128x32x64x32x64_r4x1x1_r4x1x1_w16x16x32_w16x16x32_o3_qr_tdm_sched_vr_npad_nlogits_nbias_mc_*_ndropout_nskip_nqscale_ntrload_nsink")
        set(generic_fwd_codegen_args
            ${generic_fwd_example}/generate.py
            --targets ${FMHA_QR_TDM_SCHED_FIRST_GPU_TARGET}
            --api fwd --optdim 64 -m generic --filter ${generic_fwd_filter})
        file(MAKE_DIRECTORY ${generic_fwd_dir})
        execute_process(
            COMMAND ${Python3_EXECUTABLE} ${generic_fwd_codegen_args}
                --list_blobs ${generic_fwd_list}
            RESULT_VARIABLE generic_fwd_result)
        if(NOT generic_fwd_result EQUAL 0)
            message(FATAL_ERROR "Failed to list generic mc D64 N128 forward test instances")
        endif()
        file(STRINGS ${generic_fwd_list} generic_fwd_blobs)
        list(LENGTH generic_fwd_blobs generic_fwd_blob_count)
        # Four kernels: BF16/FP16 x LSE off/on, plus the generated API.
        if(NOT generic_fwd_blob_count EQUAL 5)
            message(FATAL_ERROR "Expected four generic mc D64 N128 kernels and their API")
        endif()
        file(GLOB_RECURSE generic_fwd_codegen_scripts CONFIGURE_DEPENDS
            ${generic_fwd_example}/codegen/*.py)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            ${generic_fwd_example}/generate.py ${generic_fwd_codegen_scripts})
        add_custom_command(
            OUTPUT ${generic_fwd_blobs}
            COMMAND ${Python3_EXECUTABLE} ${generic_fwd_codegen_args}
                --output_dir ${generic_fwd_dir}
            DEPENDS ${generic_fwd_example}/generate.py ${generic_fwd_codegen_scripts}
            VERBATIM)
        set_source_files_properties(${generic_fwd_blobs} PROPERTIES LANGUAGE HIP)
        set(generic_fwd_instances test_qr_tdm_sched_generic_fwd_instances)
        add_library(${generic_fwd_instances} OBJECT EXCLUDE_FROM_ALL ${generic_fwd_blobs})
        target_include_directories(${generic_fwd_instances} PRIVATE ${generic_fwd_example})
        set_property(TARGET ${generic_fwd_instances} PROPERTY HIP_ARCHITECTURES
            ${FMHA_QR_TDM_SCHED_GPU_TARGETS})
        get_target_property(generic_fwd_private_options ${FMHA_FWD_INSTANCES} COMPILE_OPTIONS)
        get_target_property(generic_fwd_interface_options ${FMHA_FWD_INSTANCES}
            INTERFACE_COMPILE_OPTIONS)
        if(generic_fwd_private_options)
            target_compile_options(${generic_fwd_instances} PRIVATE ${generic_fwd_private_options})
        endif()
        if(generic_fwd_interface_options)
            target_compile_options(${generic_fwd_instances} INTERFACE ${generic_fwd_interface_options})
        endif()
        # Production applies the RDNA BF16 conversion definition per source.
        get_target_property(production_fwd_blobs ${FMHA_FWD_INSTANCES} SOURCES)
        foreach(blob IN LISTS production_fwd_blobs)
            if(blob MATCHES "_gfx125\\.cpp$")
                get_source_file_property(generic_fwd_definitions ${blob}
                    TARGET_DIRECTORY ${FMHA_FWD_INSTANCES} COMPILE_DEFINITIONS)
                break()
            endif()
        endforeach()
        include(${generic_fwd_example}/cmake/fmha_fwd_source_flags.cmake)
        foreach(blob IN LISTS generic_fwd_blobs)
            get_filename_component(blob_name ${blob} NAME)
            if(blob_name MATCHES "_gfx125\\.cpp$" AND generic_fwd_definitions)
                set_property(SOURCE ${blob} APPEND PROPERTY COMPILE_DEFINITIONS
                    ${generic_fwd_definitions})
            endif()
            ck_tile_fmha_fwd_get_source_compile_options(${blob_name} generic_fwd_source_options)
            set_property(SOURCE ${blob} APPEND PROPERTY COMPILE_OPTIONS
                ${generic_fwd_source_options})
        endforeach()
    endif()

    foreach(dtype IN ITEMS bf16 fp16)
        set(window_target test_ck_tile_fmha_qr_tdm_bhsd_window_${dtype})
        add_qr_tdm_sched_gtest(${window_target} test_qr_tdm_bhsd_window.cpp)
        if(dtype STREQUAL "fp16")
            target_compile_definitions(${window_target} PRIVATE CK_TILE_FMHA_BHSD_WINDOW_FP16=1)
        endif()
        target_compile_options(${window_target} PRIVATE -Wno-float-equal
            -mllvm -amdgpu-expert-scheduling-mode -fgpu-flush-denormals-to-zero)
        if(TARGET ${FMHA_FWD_INSTANCES})
            set(target test_ck_tile_fmha_qr_tdm_sched_fwd_${dtype})
            add_qr_tdm_sched_gtest(${target} test_qr_tdm_sched_fwd.cpp)
            target_link_libraries(${target} PRIVATE ${FMHA_FWD_INSTANCES})
            if(dtype STREQUAL "fp16")
                target_compile_definitions(${target} PRIVATE DataTypeConfig=FmhaFwdFp16)
            else()
                target_compile_definitions(${target} PRIVATE DataTypeConfig=FmhaFwdBf16)
            endif()
            target_compile_options(${target} PRIVATE -Wno-float-equal)

            set(target test_ck_tile_fmha_qr_tdm_sched_generic_fwd_${dtype})
            add_qr_tdm_sched_gtest(${target} test_qr_tdm_sched_generic_fwd.cpp)
            target_link_libraries(${target} PRIVATE ${generic_fwd_instances})
            if(dtype STREQUAL "fp16")
                target_compile_definitions(${target} PRIVATE DataTypeConfig=FmhaFwdFp16)
            else()
                target_compile_definitions(${target} PRIVATE DataTypeConfig=FmhaFwdBf16)
            endif()
            set_tests_properties(${target} PROPERTIES TIMEOUT 300)
        endif()
        set(target test_ck_tile_fmha_qr_tdm_sched_layout_${dtype})
        add_qr_tdm_sched_gtest(${target} test_qr_tdm_sched_layout.cpp)
        if(dtype STREQUAL "fp16")
            target_compile_definitions(${target} PRIVATE CK_TILE_FMHA_TDM_V128_TEST_FP16=1)
        endif()
    endforeach()

    add_qr_tdm_sched_gtest(test_ck_tile_fmha_qr_tdm_sched_softmax
        test_qr_tdm_sched_softmax.cpp)
    target_compile_options(test_ck_tile_fmha_qr_tdm_sched_softmax PRIVATE -Wno-float-equal)
    add_qr_tdm_sched_gtest(test_ck_tile_fmha_qr_tdm_sched_fragments
        test_qr_tdm_sched_fragments.cpp)
    target_compile_options(test_ck_tile_fmha_qr_tdm_sched_fragments PRIVATE -Wno-float-equal)

    foreach(geometry IN ITEMS d128_n128 d192_n64 d192_n128)
        if(geometry MATCHES "d128")
            set(head_dim 128)
        else()
            set(head_dim 192)
        endif()
        if(geometry MATCHES "n64$")
            set(tile_n 64)
        else()
            set(tile_n 128)
        endif()
        foreach(dtype IN ITEMS bf16 fp16)
            set(target test_ck_tile_fmha_qr_tdm_sched_sink_${geometry}_${dtype})
            add_qr_tdm_sched_gtest(${target} test_qr_tdm_sched_sink.cpp)
            target_compile_definitions(${target} PRIVATE
                CK_TILE_FMHA_SINK_TEST_HEAD_DIM=${head_dim}
                CK_TILE_FMHA_SINK_TEST_TILE_N=${tile_n})
            if(dtype STREQUAL "fp16")
                target_compile_definitions(${target} PRIVATE CK_TILE_FMHA_SINK_TEST_FP16=1)
            endif()
            target_compile_options(${target} PRIVATE -Wno-float-equal
                -mllvm -amdgpu-expert-scheduling-mode -fgpu-flush-denormals-to-zero)
        endforeach()
    endforeach()
endif()

# The independent arithmetic reference is a host test on every FMHA architecture.
add_gtest_executable(test_ck_tile_fmha_reference test_fmha_reference.cpp)
add_dependencies(${TEST_NAME} test_ck_tile_fmha_reference)
target_compile_options(test_ck_tile_fmha_reference PRIVATE -Wno-float-equal)
get_test_property(test_ck_tile_fmha_reference LABELS FMHA_REFERENCE_COMMON_LABELS)
set_tests_properties(test_ck_tile_fmha_reference PROPERTIES
    LABELS "${FMHA_REFERENCE_COMMON_LABELS};${TEST_NAME};CK_TILE_FMHA_TESTS")
list(APPEND FMHA_QR_TDM_SCHED_TEST_TARGETS test_ck_tile_fmha_reference)
# script/dependency-parser/smart_build_ci.sh force-builds the ninja target
# `${TEST_NAME}` and runs `ctest -R ${TEST_NAME}` whenever the ck_tile FMHA
# example changes. That regex is unanchored, so it also selects this test;
# without the edge below the binary is never built and ctest reports Not Run.
# Keep the umbrella a superset of what the regex matches.
add_dependencies(${TEST_NAME} test_ck_tile_fmha_reference)

# Host test of the simplified attention mask's edge-tile check for partial query tiles.
add_gtest_executable(test_ck_tile_fmha_mask_edge_tile test_fmha_mask_edge_tile.cpp)
add_dependencies(${TEST_NAME} test_ck_tile_fmha_mask_edge_tile)
get_test_property(test_ck_tile_fmha_mask_edge_tile LABELS FMHA_MASK_COMMON_LABELS)
set_tests_properties(test_ck_tile_fmha_mask_edge_tile PROPERTIES
    LABELS "${FMHA_MASK_COMMON_LABELS};${TEST_NAME};CK_TILE_FMHA_TESTS")
list(APPEND FMHA_QR_TDM_SCHED_TEST_TARGETS test_ck_tile_fmha_mask_edge_tile)

# Python integration test for the fmha_fwd codegen path.
#
# Drives the real `generate.py` with a JSON tune-config
# (CK_TILE_FMHA_FWD_CUSTOM_TUNE_CONFIG_FILE) that declares two tiles
# identical in every field except F_occupancy, and asserts that codegen
# emits two distinct .cpp blobs (one with `_o2` suffix, one without).
# This covers both:
#   1. CK_TILE_FMHA_FWD_CUSTOM_TUNE_CONFIG_FILE JSON loading & dynamic
#      factory subclassing.
#   2. Per-occupancy kernel instantiation (kOccupancy_ template parameter)
#      exposed via FmhaFwdTileSize.name symbol suffix.
# Pure Python, no GPU required.
set(FMHA_FWD_TUNE_CONFIG_PY_TEST test_ck_tile_fmha_fwd_tune_config)
add_test(
    NAME ${FMHA_FWD_TUNE_CONFIG_PY_TEST}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_fmha_fwd_tune_config.py -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(${FMHA_FWD_TUNE_CONFIG_PY_TEST}
    PROPERTIES LABELS "${TEST_NAME};CK_TILE_FMHA_TESTS")

# Codegen tests for the per-architecture FMHA BWD decode pipeline selection:
# gfx950 must stay off the TDM decode pipeline (it has no tensor loads), gfx1250
# must take it, and the two trailing problem-template flags must be emitted in
# the declared order -- swapping them compiles and silently picks the wrong
# pipeline. Pure Python, no GPU required.
set(FMHA_BWD_CODEGEN_PY_TEST test_ck_tile_fmha_bwd_codegen)
add_test(
    NAME ${FMHA_BWD_CODEGEN_PY_TEST}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_fmha_bwd_codegen.py -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(${FMHA_BWD_CODEGEN_PY_TEST}
    PROPERTIES LABELS "${TEST_NAME};CK_TILE_FMHA_TESTS")

# Codegen and compiled host-dispatch regression tests require no GPU.
add_test(
    NAME test_ck_tile_fmha_fwd_codegen
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_fmha_fwd_codegen.py -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(test_ck_tile_fmha_fwd_codegen PROPERTIES
    LABELS "${TEST_NAME};CK_TILE_FMHA_TESTS;CK_TILE_FMHA_CODEGEN_TESTS")

# Umbrella target to build and run all ck_tile fmha tests
# Usage: ninja ck_tile_fmha_tests
add_custom_target(ck_tile_fmha_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR} -L "CK_TILE_FMHA_TESTS"
    DEPENDS ${TEST_NAME} ${FMHA_QR_TDM_TEST_TARGETS} ${FMHA_QR_TDM_SCHED_TEST_TARGETS}
    USES_TERMINAL
    COMMENT "Running all ck_tile fmha tests..."
)
