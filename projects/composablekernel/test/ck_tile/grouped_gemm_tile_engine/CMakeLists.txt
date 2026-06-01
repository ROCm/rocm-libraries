# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ============================================================================
# Grouped GEMM Tile Engine Unit Tests
#
# This CMake file creates unit tests for tile_engine generated Grouped GEMM
# kernels. It follows the exact same build patterns as tile_engine for
# consistency and reliability. Each kernel configuration gets its own test
# executable.
# ============================================================================

# Locate tile_engine Grouped GEMM scripts directory
set(TILE_ENGINE_GROUPED_GEMM_DIR "${PROJECT_SOURCE_DIR}/tile_engine/ops/gemm/grouped_gemm")

if(NOT EXISTS ${TILE_ENGINE_GROUPED_GEMM_DIR})
    message(WARNING "Tile engine grouped GEMM directory not found: ${TILE_ENGINE_GROUPED_GEMM_DIR}")
    return()
endif()

# ============================================================================
# create_individual_grouped_gemm_test_target
#
# Creates a single test executable for a specific kernel configuration.
# Mirrors tile_engine's create_individual_grouped_gemm_target for consistency.
#
# Parameters:
#   datatype     - Data type (fp16, bf16, fp8, bf8)
#   layout       - Matrix layout (rcr, rrr, ccr, crr)
#   config_name  - Configuration file name without .json extension
#   trait        - Kernel trait combination string
#   tile_config  - Tile configuration parameters
#   config_json  - Full path to JSON configuration file
# ============================================================================
function(create_individual_grouped_gemm_test_target datatype layout config_name trait tile_config config_json)
    set(target_name "test_grouped_gemm_tile_engine_${datatype}_${layout}_${config_name}_${trait}_${tile_config}")
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}/${config_name}")

    # Generated header path (already created during cmake configuration)
    set(test_header "${working_path}/grouped_gemm_single_${datatype}_${layout}_${trait}_${tile_config}.hpp")
    set(test_params_header "${working_path}/test_params.hpp")

    # Verify header exists (should have been generated during cmake configuration)
    if(NOT EXISTS ${test_header})
        message(WARNING "Generated header not found: ${test_header}")
        return()
    endif()

    # Verify test parameters header exists
    if(NOT EXISTS ${test_params_header})
        message(WARNING "Test parameters header not found: ${test_params_header}")
        return()
    endif()

    # Create GTest executable for this kernel configuration
    add_gtest_executable(${target_name}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_grouped_gemm_simple.cpp
    )

    # Configure GPU architectures for HIP compilation
    set_property(TARGET ${target_name} PROPERTY HIP_ARCHITECTURES ${GROUPED_GEMM_TEST_GPU_TARGETS})

    # Define preprocessor macros for generated header location and test parameters
    target_compile_definitions(${target_name} PRIVATE
        GROUPED_GEMM_SINGLE_INSTANCE_HPP="${test_header}"
        GROUPED_GEMM_TEST_PARAMS_HPP="${test_params_header}"
    )

    # Include directories for headers and dependencies
    target_include_directories(${target_name} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/include
        ${PROJECT_SOURCE_DIR}  # Root directory for tile_engine access
        ${TILE_ENGINE_GROUPED_GEMM_DIR}
        ${GTEST_INCLUDE_DIRS}
    )

    # Compiler options matching tile_engine requirements
    target_compile_options(${target_name} PRIVATE
        -Wno-undefined-func-template  # Suppress template warnings
        -Wno-float-equal              # Allow floating point comparisons
        --offload-compress            # Enable GPU code compression
        -include ${test_header}       # Auto-include generated header
    )

    # Add FP8 format definitions for proper data type interpretation
    if(CK_USE_OCP_FP8)
        target_compile_options(${target_name} PRIVATE -DCK_TILE_USE_OCP_FP8)
    endif()

    message(DEBUG "  Created grouped GEMM test target: ${target_name}")
endfunction()

# ============================================================================
# build_grouped_gemm_test_targets
#
# Builds all test targets for a specific datatype/layout/config combination.
# Uses tile_engine's two-step process: list kernels, then generate tests.
#
# Parameters:
#   datatype     - Data type (fp16, bf16, fp8, bf8)
#   layout       - Matrix layout (rcr, rrr, ccr, crr)
#   config_name  - Configuration file name without .json extension
# ============================================================================
function(build_grouped_gemm_test_targets datatype layout config_name)
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}/${config_name}")

    # Locate and validate configuration file
    set(config_filename "${config_name}.json")
    set(json_blob "${CMAKE_CURRENT_SOURCE_DIR}/configs/${config_filename}")

    if(NOT EXISTS ${json_blob})
        message(WARNING "Test config file not found: ${json_blob}")
        return()
    endif()

    # Prepare build directory for this configuration
    file(MAKE_DIRECTORY ${working_path})

    # STEP 1: Discovery phase - list all valid kernel configurations
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -u ${TILE_ENGINE_GROUPED_GEMM_DIR}/grouped_gemm_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --config_json ${json_blob}
                --list_kernels
                --gpu_target "${GROUPED_GEMM_TEST_GPU_TARGETS}"
        WORKING_DIRECTORY ${TILE_ENGINE_GROUPED_GEMM_DIR}
        RESULT_VARIABLE ret
        OUTPUT_VARIABLE list_output
        ERROR_VARIABLE list_error
    )

    if(NOT ret EQUAL 0)
        message(WARNING "Failed to list grouped GEMM kernels for ${datatype}_${layout}_${config_name}: ${list_error}")
        return()
    endif()

    # Verify kernel list file was generated
    if(NOT EXISTS ${working_path}/grouped_gemm_kernel_list.txt)
        message(DEBUG "No kernels found for ${datatype}_${layout}_${config_name} (validation filtered out all combinations)")
        return()
    endif()

    message(DEBUG "Building grouped GEMM tests for ${datatype}_${layout}_${config_name}")

    # STEP 2a: Extract test parameters from config
    set(test_params_file "${working_path}/test_params.hpp")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -u ${CMAKE_CURRENT_SOURCE_DIR}/extract_test_params.py
                --config_file ${json_blob}
                --output_file ${test_params_file}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE extract_ret
        OUTPUT_VARIABLE extract_output
        ERROR_VARIABLE extract_error
    )

    if(NOT extract_ret EQUAL 0)
        message(WARNING "Failed to extract test parameters for ${datatype}_${layout}: ${extract_error}")
        return()
    endif()

    # STEP 2b: Header generation phase - generate headers using --gen_single
    message(STATUS "  Generating grouped GEMM headers using --gen_single...")

    file(STRINGS ${working_path}/grouped_gemm_kernel_list.txt kernel_lines)
    set(gen_count 0)

    foreach(line IN LISTS kernel_lines)
        # Parse kernel specification format: kernel_name|tile_config|trait_combo
        string(REPLACE "|" ";" parts "${line}")
        list(LENGTH parts parts_len)
        if(parts_len EQUAL 3)
            list(GET parts 0 kernel_name)
            list(GET parts 1 tile_config)
            list(GET parts 2 trait_combo)

            # Generate header using --gen_single
            execute_process(
                COMMAND ${Python3_EXECUTABLE} -u ${TILE_ENGINE_GROUPED_GEMM_DIR}/grouped_gemm_instance_builder.py
                        --working_path ${working_path}
                        --gpu_target "${GROUPED_GEMM_TEST_GPU_TARGETS}"
                        --datatype ${datatype}
                        --layout ${layout}
                        --config_json ${json_blob}
                        --gen_single
                        --kernel_name "${kernel_name}"
                        --tile_config "${tile_config}"
                        --trait_combo "${trait_combo}"
                WORKING_DIRECTORY ${TILE_ENGINE_GROUPED_GEMM_DIR}
                RESULT_VARIABLE gen_ret
                OUTPUT_VARIABLE gen_output
                ERROR_VARIABLE gen_error
            )

            if(NOT gen_ret EQUAL 0)
                message(WARNING "Failed to generate header for ${kernel_name}: ${gen_error}")
            else()
                math(EXPR gen_count "${gen_count} + 1")
            endif()
        endif()
    endforeach()

    message(STATUS "  Generated ${gen_count} grouped GEMM headers for ${datatype}_${layout}")

    # STEP 3: Target creation phase - create test targets
    message(STATUS "  Creating grouped GEMM test targets...")
    file(STRINGS ${working_path}/grouped_gemm_kernel_list.txt kernel_lines)
    set(test_count 0)
    foreach(line IN LISTS kernel_lines)
        # Parse kernel specification format: kernel_name|tile_config|trait_combo
        string(REPLACE "|" ";" parts "${line}")
        list(LENGTH parts parts_len)
        if(parts_len EQUAL 3)
            list(GET parts 0 kernel_name)
            list(GET parts 1 tile_config)
            list(GET parts 2 trait_combo)

            # Generate test target for this kernel configuration
            create_individual_grouped_gemm_test_target("${datatype}" "${layout}" "${config_name}" "${trait_combo}" "${tile_config}" "${json_blob}")
            math(EXPR test_count "${test_count} + 1")
        endif()
    endforeach()
    message(STATUS "  Created ${test_count} grouped GEMM test targets for ${datatype}_${layout}")
endfunction()

# ============================================================================
# MAIN EXECUTION - Test Target Generation
# ============================================================================

message(STATUS "=== Starting Grouped GEMM Tile Engine Test Configuration ===")
message(STATUS "SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")

# GPU architecture filtering - only build tests for supported architectures
set(GROUPED_GEMM_TEST_GPU_TARGETS "")
set(DESIRED_TARGETS "gfx942;gfx950;gfx1250")

foreach(target IN LISTS SUPPORTED_GPU_TARGETS)
    if(target IN_LIST DESIRED_TARGETS)
        list(APPEND GROUPED_GEMM_TEST_GPU_TARGETS ${target})
        message(STATUS "  Adding GPU target for grouped GEMM tests: ${target}")
    endif()
endforeach()

# Early exit if no compatible GPU architectures are available
if(NOT GROUPED_GEMM_TEST_GPU_TARGETS)
    message(WARNING "Skipping Grouped GEMM Tile Engine tests: No supported GPU targets (gfx942, gfx950, gfx1250) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
    return()
endif()

message(STATUS "Building Grouped GEMM tile engine tests for GPU targets: ${GROUPED_GEMM_TEST_GPU_TARGETS}")

# ============================================================================
# Test Configuration Matrix
# ============================================================================

set(TEST_DATATYPES "fp16;fp8;bf16;bf8")
set(TEST_LAYOUTS "rcr;rrr;ccr;crr")

# Simple test: basic functionality validation
set(SIMPLE_TEST_CONFIG "simple_test_config")
set(SIMPLE_TEST_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/configs/${SIMPLE_TEST_CONFIG}.json")

if(EXISTS ${SIMPLE_TEST_CONFIG_FILE})
    message(STATUS "Processing grouped GEMM simple test config: ${SIMPLE_TEST_CONFIG}")
    foreach(datatype IN LISTS TEST_DATATYPES)
        foreach(layout IN LISTS TEST_LAYOUTS)
            build_grouped_gemm_test_targets("${datatype}" "${layout}" "${SIMPLE_TEST_CONFIG}")
        endforeach()
    endforeach()
else()
    message(WARNING "Simple test config file not found: ${SIMPLE_TEST_CONFIG_FILE}")
endif()

message(STATUS "Grouped GEMM tile engine tests configured")
message(STATUS "  - Simple test: all datatypes and layouts")
