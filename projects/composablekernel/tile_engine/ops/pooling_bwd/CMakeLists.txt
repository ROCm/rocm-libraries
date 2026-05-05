# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ============================================================================
# Pooling Backward Tile Engine Build Configuration
#
# Generates individual benchmark executables for max-pool backward kernels.
# ============================================================================

set(POOLING_BWD_DATATYPE "fp16;fp32" CACHE STRING "List of datatypes for Pooling Bwd (semicolon-separated)")
set(POOLING_BWD_CONFIG_FILE "" CACHE STRING "Custom config file name (without path, must be in configs/ folder)")
option(ENABLE_CCACHE_POOLING_BWD "Enable ccache for pooling bwd ops compilation" OFF)

set(POOLING_BWD_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

# ============================================================================
# create_individual_pool_bwd_target
#
# Creates a single benchmark executable for a specific pooling bwd kernel
# configuration.
# ============================================================================
function(create_individual_pool_bwd_target datatype kernel_name trait tile_config config_json)
    if(NOT POOLING_BWD_GPU_TARGETS)
        message(WARNING "Skipping individual pooling bwd target: No supported GPU targets")
        return()
    endif()

    set(target_name "benchmark_pooling_bwd_${datatype}_${trait}_${tile_config}")
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}")
    set(target_source "${CMAKE_CURRENT_BINARY_DIR}/${target_name}_pooling_bwd_benchmark_single.cpp")

    set(instance_header "${working_path}/pooling_bwd_single_${kernel_name}.hpp")

    add_custom_command(
        OUTPUT ${instance_header}
        COMMAND ${Python3_EXECUTABLE} ${POOLING_BWD_SOURCE_DIR}/pooling_bwd_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --config_json ${config_json}
                --gen_single
                --kernel_name "${kernel_name}"
                --tile_config "${tile_config}"
                --trait_combo "${trait}"
        DEPENDS ${POOLING_BWD_SOURCE_DIR}/pooling_bwd_instance_builder.py ${config_json}
        COMMENT "Generating ${instance_header}"
    )

    configure_file(${POOLING_BWD_SOURCE_DIR}/pooling_bwd_benchmark_single.cpp ${target_source} COPYONLY)

    add_executable(${target_name}
        EXCLUDE_FROM_ALL
        ${target_source}
        ${instance_header}
    )

    set_property(TARGET ${target_name} PROPERTY HIP_ARCHITECTURES ${POOLING_BWD_GPU_TARGETS})

    target_compile_definitions(${target_name} PRIVATE
        POOLING_BWD_SINGLE_INSTANCE_HPP="${instance_header}"
    )

    target_include_directories(${target_name} PRIVATE
        ${POOLING_BWD_SOURCE_DIR}
        ${working_path}
    )

    target_compile_options(${target_name} PRIVATE
        -Wno-undefined-func-template
        -Wno-float-equal
        --offload-compress
        -include ${instance_header}
    )

    if(CK_USE_OCP_FP8)
        target_compile_options(${target_name} PRIVATE -DCK_TILE_USE_OCP_FP8)
    endif()

    add_dependencies(benchmark_pooling_bwd_all ${target_name})
    add_dependencies(benchmark_pooling_bwd_${datatype} ${target_name})

    message(DEBUG "  Created pooling bwd benchmark target: ${target_name}")
endfunction()

# ============================================================================
# build_individual_pool_bwd_targets
#
# Builds all benchmark targets for a specific datatype.
# ============================================================================
function(build_individual_pool_bwd_targets datatype)
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}")

    if(DEFINED ENV{POOLING_BWD_CONFIG_FILE} AND NOT "$ENV{POOLING_BWD_CONFIG_FILE}" STREQUAL "")
        set(config_filename "$ENV{POOLING_BWD_CONFIG_FILE}")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/${config_filename}")
        message(VERBOSE "  Using config from environment variable: ${config_filename}")
    elseif(NOT "${POOLING_BWD_CONFIG_FILE}" STREQUAL "")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/${POOLING_BWD_CONFIG_FILE}")
        message(VERBOSE "  Using custom config: ${POOLING_BWD_CONFIG_FILE}")
    else()
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/default_config.json")
        message(VERBOSE "  Using default config for pooling bwd")
    endif()

    if(NOT EXISTS ${json_blob})
        message(FATAL_ERROR "Config file not found: ${json_blob}")
    endif()

    file(MAKE_DIRECTORY ${working_path})

    message(VERBOSE "  Listing pooling bwd kernel configurations for ${datatype}...")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -u ${CMAKE_CURRENT_LIST_DIR}/pooling_bwd_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --config_json ${json_blob}
                --list_kernels
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        RESULT_VARIABLE ret
        OUTPUT_VARIABLE list_output
        ERROR_VARIABLE list_error
    )

    if(NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to list pooling bwd kernels for ${datatype}: ${list_error}")
    endif()

    if(EXISTS ${working_path}/pool_bwd_kernel_count.txt)
        file(READ ${working_path}/pool_bwd_kernel_count.txt kernel_count)
        string(STRIP "${kernel_count}" kernel_count)
        message(VERBOSE "  Found ${kernel_count} pooling bwd kernel configurations")
    else()
        message(FATAL_ERROR "Pooling bwd kernel count file not found")
    endif()

    if(EXISTS ${working_path}/pool_bwd_kernel_list.txt)
        file(STRINGS ${working_path}/pool_bwd_kernel_list.txt kernel_lines)
        foreach(line IN LISTS kernel_lines)
            string(REPLACE "|" ";" parts "${line}")
            list(LENGTH parts parts_len)
            if(parts_len EQUAL 3)
                list(GET parts 0 kernel_name)
                list(GET parts 1 tile_config)
                list(GET parts 2 trait_combo)
                create_individual_pool_bwd_target("${datatype}" "${kernel_name}" "${trait_combo}" "${tile_config}" "${json_blob}")
            endif()
        endforeach()
    else()
        message(FATAL_ERROR "Pooling bwd kernel list file not found")
    endif()
endfunction()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

message(VERBOSE "=== Starting Tile Engine Pooling Bwd Configuration ===")
message(VERBOSE "POOLING_BWD_DATATYPE: ${POOLING_BWD_DATATYPE}")
message(VERBOSE "SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")

set(POOLING_BWD_GPU_TARGETS "")
set(DESIRED_TARGETS "gfx90a;gfx942;gfx950;gfx1201")

foreach(target IN LISTS SUPPORTED_GPU_TARGETS)
    if(target IN_LIST DESIRED_TARGETS)
        list(APPEND POOLING_BWD_GPU_TARGETS ${target})
        message(VERBOSE "  Adding GPU target for pooling bwd: ${target}")
    endif()
endforeach()

if(NOT POOLING_BWD_GPU_TARGETS)
    message(WARNING "Skipping Tile Engine Pooling Bwd build: No supported GPU targets (gfx90a, gfx942, gfx950, gfx1201) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
else()
    message(VERBOSE "Building pooling bwd targets for GPU targets: ${POOLING_BWD_GPU_TARGETS}")

    if(ENABLE_CCACHE_POOLING_BWD)
        find_program(CCACHE_PROGRAM ccache)
        if(CCACHE_PROGRAM)
            set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
            message(VERBOSE "Using ccache for pooling bwd compilation")
        endif()
    endif()

    add_custom_target(benchmark_pooling_bwd_all)

    foreach(dt IN LISTS POOLING_BWD_DATATYPE)
        add_custom_target(benchmark_pooling_bwd_${dt})
    endforeach()

    foreach(dt IN LISTS POOLING_BWD_DATATYPE)
        build_individual_pool_bwd_targets(${dt})
    endforeach()
endif()
