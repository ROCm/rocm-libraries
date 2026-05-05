# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ============================================================================
# Pooling Bwd Tile Engine Unit Tests
#
# Mirrors test/ck_tile/pooling_tile_engine/ but for the max-pool backward op.
# Each generated kernel configuration gets its own test executable.
# ============================================================================

set(TILE_ENGINE_POOLING_BWD_DIR "${PROJECT_SOURCE_DIR}/tile_engine/ops/pooling_bwd")

if(NOT EXISTS ${TILE_ENGINE_POOLING_BWD_DIR})
    message(WARNING "Tile engine pooling_bwd directory not found: ${TILE_ENGINE_POOLING_BWD_DIR}")
    return()
endif()

function(create_individual_pool_bwd_test_target datatype config_name kernel_name trait tile_config config_json)
    set(target_name "test_pooling_bwd_tile_engine_${datatype}_${config_name}_${trait}_${tile_config}")
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${config_name}")

    set(test_header "${working_path}/pooling_bwd_single_${kernel_name}.hpp")

    # trait format: pooling_dim_overlap-or-nooverlap
    string(REGEX MATCH "^[23]d" kernel_pooling_dim "${trait}")
    if(kernel_pooling_dim STREQUAL "3d")
        set(test_params_header "${working_path}/test_params_3d.hpp")
        set(pooling_dim_value 3)
    else()
        set(test_params_header "${working_path}/test_params_2d.hpp")
        set(pooling_dim_value 2)
    endif()

    if(NOT EXISTS ${test_header})
        message(WARNING "Generated header not found: ${test_header}")
        return()
    endif()

    if(NOT EXISTS ${test_params_header})
        message(WARNING "Test parameters header not found: ${test_params_header}")
        return()
    endif()

    add_gtest_executable(${target_name}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_pooling_bwd_simple.cpp
    )

    set_property(TARGET ${target_name} PROPERTY HIP_ARCHITECTURES ${POOLING_BWD_TEST_GPU_TARGETS})

    target_compile_definitions(${target_name} PRIVATE
        POOLING_BWD_SINGLE_INSTANCE_HPP="${test_header}"
        POOLING_BWD_TEST_PARAMS_HPP="${test_params_header}"
        POOLING_BWD_DIM_VALUE=${pooling_dim_value}
    )

    target_include_directories(${target_name} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/include
        ${PROJECT_SOURCE_DIR}
        ${TILE_ENGINE_POOLING_BWD_DIR}
        ${GTEST_INCLUDE_DIRS}
    )

    target_compile_options(${target_name} PRIVATE
        -Wno-undefined-func-template
        -Wno-float-equal
        --offload-compress
        -include ${test_header}
    )

    if(CK_USE_OCP_FP8)
        target_compile_options(${target_name} PRIVATE -DCK_TILE_USE_OCP_FP8)
    endif()

    message(STATUS "  Created pooling bwd test target: ${target_name}")
endfunction()

function(build_pool_bwd_test_targets datatype config_name)
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${config_name}")

    set(config_filename "${config_name}.json")
    set(json_blob "${CMAKE_CURRENT_SOURCE_DIR}/configs/${config_filename}")

    if(NOT EXISTS ${json_blob})
        message(WARNING "Pooling bwd test config file not found: ${json_blob}")
        return()
    endif()

    file(MAKE_DIRECTORY ${working_path})

    execute_process(
        COMMAND ${Python3_EXECUTABLE} -u ${TILE_ENGINE_POOLING_BWD_DIR}/pooling_bwd_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --config_json ${json_blob}
                --list_kernels
        WORKING_DIRECTORY ${TILE_ENGINE_POOLING_BWD_DIR}
        RESULT_VARIABLE ret
        OUTPUT_VARIABLE list_output
        ERROR_VARIABLE list_error
    )

    if(NOT ret EQUAL 0)
        message(WARNING "Failed to list pooling bwd kernels for ${datatype}_${config_name}: ${list_error}")
        return()
    endif()

    if(NOT EXISTS ${working_path}/pool_bwd_kernel_list.txt)
        message(STATUS "No pooling bwd kernels found for ${datatype}_${config_name}")
        return()
    endif()

    message(STATUS "Building pooling bwd tests for ${datatype}_${config_name}")

    set(test_params_file_2d "${working_path}/test_params_2d.hpp")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -u ${CMAKE_CURRENT_SOURCE_DIR}/extract_test_params.py
                --config_file ${json_blob}
                --output_file ${test_params_file_2d}
                --pooling_dim 2d
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE extract_ret_2d
        OUTPUT_VARIABLE extract_output_2d
        ERROR_VARIABLE extract_error_2d
    )
    if(NOT extract_ret_2d EQUAL 0)
        message(WARNING "Failed to extract 2D test parameters for pooling bwd ${datatype}: ${extract_error_2d}")
        return()
    endif()

    set(test_params_file_3d "${working_path}/test_params_3d.hpp")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -u ${CMAKE_CURRENT_SOURCE_DIR}/extract_test_params.py
                --config_file ${json_blob}
                --output_file ${test_params_file_3d}
                --pooling_dim 3d
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE extract_ret_3d
        OUTPUT_VARIABLE extract_output_3d
        ERROR_VARIABLE extract_error_3d
    )
    if(NOT extract_ret_3d EQUAL 0)
        message(WARNING "Failed to extract 3D test parameters for pooling bwd ${datatype}: ${extract_error_3d}")
        return()
    endif()

    message(STATUS "  Generating pooling bwd headers using --gen_single...")

    file(STRINGS ${working_path}/pool_bwd_kernel_list.txt kernel_lines)
    set(gen_count 0)

    foreach(line IN LISTS kernel_lines)
        string(REPLACE "|" ";" parts "${line}")
        list(LENGTH parts parts_len)
        if(parts_len EQUAL 3)
            list(GET parts 0 kernel_name)
            list(GET parts 1 tile_config)
            list(GET parts 2 trait_combo)

            execute_process(
                COMMAND ${Python3_EXECUTABLE} -u ${TILE_ENGINE_POOLING_BWD_DIR}/pooling_bwd_instance_builder.py
                        --working_path ${working_path}
                        --datatype ${datatype}
                        --config_json ${json_blob}
                        --gen_single
                        --kernel_name "${kernel_name}"
                        --tile_config "${tile_config}"
                        --trait_combo "${trait_combo}"
                WORKING_DIRECTORY ${TILE_ENGINE_POOLING_BWD_DIR}
                RESULT_VARIABLE gen_ret
                OUTPUT_VARIABLE gen_output
                ERROR_VARIABLE gen_error
            )

            if(NOT gen_ret EQUAL 0)
                message(WARNING "Failed to generate pooling bwd header for ${kernel_name}: ${gen_error}")
            else()
                math(EXPR gen_count "${gen_count} + 1")
            endif()
        endif()
    endforeach()

    message(STATUS "  Generated ${gen_count} pooling bwd headers for ${datatype}")

    message(STATUS "  Creating pooling bwd test targets...")
    file(STRINGS ${working_path}/pool_bwd_kernel_list.txt kernel_lines)
    set(test_count 0)
    foreach(line IN LISTS kernel_lines)
        string(REPLACE "|" ";" parts "${line}")
        list(LENGTH parts parts_len)
        if(parts_len EQUAL 3)
            list(GET parts 0 kernel_name)
            list(GET parts 1 tile_config)
            list(GET parts 2 trait_combo)

            create_individual_pool_bwd_test_target("${datatype}" "${config_name}" "${kernel_name}" "${trait_combo}" "${tile_config}" "${json_blob}")
            math(EXPR test_count "${test_count} + 1")
        endif()
    endforeach()
    message(STATUS "  Created ${test_count} pooling bwd test targets for ${datatype}")
endfunction()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

message(STATUS "=== Starting Pooling Bwd Tile Engine Test Configuration ===")
message(STATUS "SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")

set(POOLING_BWD_TEST_GPU_TARGETS "")
set(DESIRED_TARGETS "gfx90a;gfx942;gfx950;gfx1201")

foreach(target IN LISTS SUPPORTED_GPU_TARGETS)
    if(target IN_LIST DESIRED_TARGETS)
        list(APPEND POOLING_BWD_TEST_GPU_TARGETS ${target})
        message(STATUS "  Adding GPU target for pooling bwd tests: ${target}")
    endif()
endforeach()

if(NOT POOLING_BWD_TEST_GPU_TARGETS)
    message(WARNING "Skipping Pooling Bwd Tile Engine tests: No supported GPU targets (gfx90a, gfx942, gfx950, gfx1201) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
    return()
endif()

message(STATUS "Building Pooling bwd tile engine tests for GPU targets: ${POOLING_BWD_TEST_GPU_TARGETS}")

set(TEST_DATATYPES "fp16;fp32")

set(SIMPLE_TEST_CONFIG "simple_test_config")
set(SIMPLE_TEST_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/configs/${SIMPLE_TEST_CONFIG}.json")

if(EXISTS ${SIMPLE_TEST_CONFIG_FILE})
    message(STATUS "Processing pooling bwd simple test config: ${SIMPLE_TEST_CONFIG}")
    foreach(datatype IN LISTS TEST_DATATYPES)
        build_pool_bwd_test_targets("${datatype}" "${SIMPLE_TEST_CONFIG}")
    endforeach()
else()
    message(WARNING "Pooling bwd simple test config file not found: ${SIMPLE_TEST_CONFIG_FILE}")
endif()

set(POOLING_BWD_COVERAGE_LEVEL "quick" CACHE STRING "Pooling bwd coverage level: quick or comprehensive")
set_property(CACHE POOLING_BWD_COVERAGE_LEVEL PROPERTY STRINGS "quick" "comprehensive")

if(POOLING_BWD_COVERAGE_LEVEL STREQUAL "comprehensive")
    set(COMPREHENSIVE_CONFIG "comprehensive_coverage_config")
    set(COMPREHENSIVE_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/configs/${COMPREHENSIVE_CONFIG}.json")

    if(EXISTS ${COMPREHENSIVE_CONFIG_FILE})
        message(STATUS "Processing pooling bwd comprehensive coverage config: ${COMPREHENSIVE_CONFIG}")
        foreach(datatype IN LISTS TEST_DATATYPES)
            build_pool_bwd_test_targets("${datatype}" "${COMPREHENSIVE_CONFIG}")
        endforeach()
    else()
        message(WARNING "Pooling bwd comprehensive config file not found: ${COMPREHENSIVE_CONFIG_FILE}")
    endif()
elseif(NOT POOLING_BWD_COVERAGE_LEVEL STREQUAL "quick")
    message(FATAL_ERROR "Invalid POOLING_BWD_COVERAGE_LEVEL: ${POOLING_BWD_COVERAGE_LEVEL}. Must be 'quick' or 'comprehensive'")
endif()

message(STATUS "Pooling bwd tile engine tests configured:")
message(STATUS "  - Simple test: fp16/fp32 (always)")
message(STATUS "  - Coverage level: ${POOLING_BWD_COVERAGE_LEVEL}")
message(STATUS "    Use -DPOOLING_BWD_COVERAGE_LEVEL=comprehensive for extensive testing")
