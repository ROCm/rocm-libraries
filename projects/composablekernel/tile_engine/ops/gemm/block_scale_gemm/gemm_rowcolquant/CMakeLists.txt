# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(GEMM_ROWCOLQUANT_DATATYPE "fp8;bf8" CACHE STRING "List of datatypes for GEMM RowColQuant (semicolon-separated)")
set(GEMM_ROWCOLQUANT_LAYOUT "rcr" CACHE STRING "List of layouts for GEMM RowColQuant (semicolon-separated)")
set(GEMM_ROWCOLQUANT_CONFIG_FILE "" CACHE STRING "Custom config file name (without path, must be in configs/ folder)")
option(ENABLE_CCACHE_GEMM_ROWCOLQUANT "Enable ccache for GEMM RowColQuant ops compilation" OFF)

# Store the directory path for use in functions
set(GEMM_ROWCOLQUANT_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

# Function to create individual GEMM RowColQuant targets
function(create_individual_gemm_rowcolquant_target datatype layout trait tile_config config_json)
    # GEMM_ROWCOLQUANT_GPU_TARGETS_INDIVIDUAL is guaranteed non-empty here (caller checks at line 192)
    if(NOT GEMM_ROWCOLQUANT_GPU_TARGETS_INDIVIDUAL)
        message(FATAL_ERROR "BUG: create_individual_gemm_rowcolquant_target called with no GPU targets")
    endif()

    # Parse tile configuration: format is tile_mxtile_nxtile_k_warp_mxwarp_nxwarp_k_warp_tile_mxwarp_tile_nxwarp_tile_k
    string(REPLACE "_" ";" config_groups ${tile_config})
    list(GET config_groups 0 tile_dims)

    set(target_name "benchmark_gemm_rowcolquant_${datatype}_${layout}_${trait}_${tile_config}")
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")

    # Generate the single instance header for this kernel
    set(instance_header "${working_path}/gemm_rowcolquant_single_${datatype}_${layout}_${trait}_${tile_config}.hpp")

    # Add custom command to generate the header file at build time
    add_custom_command(
        OUTPUT ${instance_header}
        COMMAND ${Python3_EXECUTABLE} ${GEMM_ROWCOLQUANT_SOURCE_DIR}/gemm_rowcolquant_instance_builder.py
            --working_path ${working_path}
            --datatype ${datatype}
            --layout ${layout}
            --config_json ${config_json}
            --gen_single
            --kernel_name "gemm_rowcolquant_${datatype}_${layout}_${trait}_${tile_config}"
            --tile_config "${tile_config}"
            --trait_combo "${trait}"
            --gpu_target "${GEMM_ROWCOLQUANT_GPU_TARGETS_INDIVIDUAL}"
        DEPENDS ${GEMM_ROWCOLQUANT_SOURCE_DIR}/gemm_rowcolquant_instance_builder.py ${config_json}
        COMMENT "Generating ${instance_header}"
    )

    # Create the executable
    add_executable(${target_name}
        EXCLUDE_FROM_ALL
        ${GEMM_ROWCOLQUANT_SOURCE_DIR}/gemm_rowcolquant_benchmark_single.cpp
        ${instance_header}
    )

    # Set GPU architectures
    set_property(TARGET ${target_name} PROPERTY HIP_ARCHITECTURES ${GEMM_ROWCOLQUANT_GPU_TARGETS_INDIVIDUAL})

    # Set compile definitions
    target_compile_definitions(${target_name} PRIVATE
        GEMM_ROWCOLQUANT_SINGLE_INSTANCE_HPP="${instance_header}"
    )

    # Include directories
    target_include_directories(${target_name} PRIVATE
        ${GEMM_ROWCOLQUANT_SOURCE_DIR}
        ${working_path}
    )

    # Compile options
    target_compile_options(${target_name} PRIVATE
        -Wno-undefined-func-template
        -Wno-float-equal
        --offload-compress
        -include ${instance_header}
    )

    # Add to collection targets
    add_dependencies(benchmark_gemm_rowcolquant_all ${target_name})
    add_dependencies(benchmark_gemm_rowcolquant_${datatype} ${target_name})
    add_dependencies(benchmark_gemm_rowcolquant_${layout} ${target_name})
    add_dependencies(benchmark_gemm_rowcolquant_${datatype}_${layout} ${target_name})

    # Add to pipeline-specific targets
    string(REPLACE "_" ";" trait_parts ${trait})
    list(GET trait_parts 0 pipeline)

    add_dependencies(benchmark_gemm_rowcolquant_${pipeline}_pipeline ${target_name})
endfunction()

# Function to build individual GEMM RowColQuant targets
function(build_individual_gemm_rowcolquant_targets datatype layout)
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")

    # Choose config file (priority: env var > cmake var > default)
    if(DEFINED ENV{GEMM_ROWCOLQUANT_CONFIG_FILE} AND NOT "$ENV{GEMM_ROWCOLQUANT_CONFIG_FILE}" STREQUAL "")
        set(config_filename "$ENV{GEMM_ROWCOLQUANT_CONFIG_FILE}")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/${config_filename}")
        message(VERBOSE " Using config from environment variable: ${config_filename}")
    elseif(NOT "${GEMM_ROWCOLQUANT_CONFIG_FILE}" STREQUAL "")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/${GEMM_ROWCOLQUANT_CONFIG_FILE}")
        message(VERBOSE " Using custom config: ${GEMM_ROWCOLQUANT_CONFIG_FILE}")
    else()
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/default_config.json")
        message(VERBOSE " Using default config for layout ${layout}")
    endif()

    if(NOT EXISTS ${json_blob})
        message(FATAL_ERROR "Config file not found: ${json_blob}")
    endif()

    # Create working directory
    file(MAKE_DIRECTORY ${working_path})

    # List kernels
    message(VERBOSE " Listing RowColQuant kernel configurations for ${datatype} ${layout}...")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -u ${CMAKE_CURRENT_LIST_DIR}/gemm_rowcolquant_instance_builder.py
            --working_path ${working_path}
            --datatype ${datatype}
            --layout ${layout}
            --config_json ${json_blob}
            --gpu_target "${GEMM_ROWCOLQUANT_GPU_TARGETS_INDIVIDUAL}"
            --list_kernels
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        RESULT_VARIABLE ret
        OUTPUT_VARIABLE list_output
        ERROR_VARIABLE list_error
    )

    if(NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to list RowColQuant kernels for ${datatype} ${layout}: ${list_error}")
    endif()

    # Read kernel count
    if(EXISTS ${working_path}/gemm_rowcolquant_kernel_count.txt)
        file(READ ${working_path}/gemm_rowcolquant_kernel_count.txt kernel_count)
        string(STRIP "${kernel_count}" kernel_count)
        message(VERBOSE " Found ${kernel_count} RowColQuant kernel configurations")
    else()
        message(FATAL_ERROR "Kernel count file not found")
    endif()

    # Read kernel list and create targets
    if(EXISTS ${working_path}/gemm_rowcolquant_kernel_list.txt)
        file(STRINGS ${working_path}/gemm_rowcolquant_kernel_list.txt kernel_lines)
        foreach(line IN LISTS kernel_lines)
            string(REPLACE "|" ";" parts "${line}")
            list(GET parts 0 kernel_name)
            list(GET parts 1 tile_config)
            list(GET parts 2 trait_combo)

            create_individual_gemm_rowcolquant_target("${datatype}" "${layout}" "${trait_combo}" "${tile_config}" "${json_blob}")
        endforeach()
    else()
        message(FATAL_ERROR "Kernel list file not found")
    endif()
endfunction()

# Main build logic
message(VERBOSE "=== Starting Tile Engine GEMM RowColQuant Configuration ===")
message(VERBOSE "GEMM_ROWCOLQUANT_DATATYPE: ${GEMM_ROWCOLQUANT_DATATYPE}")
message(VERBOSE "GEMM_ROWCOLQUANT_LAYOUT: ${GEMM_ROWCOLQUANT_LAYOUT}")
message(VERBOSE "SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")

# Filter GPU targets to supported architectures
set(GEMM_ROWCOLQUANT_GPU_TARGETS_INDIVIDUAL "")
set(DESIRED_TARGETS "gfx90a;gfx942;gfx950;gfx1201")

foreach(target IN LISTS SUPPORTED_GPU_TARGETS)
    if(target IN_LIST DESIRED_TARGETS)
        list(APPEND GEMM_ROWCOLQUANT_GPU_TARGETS_INDIVIDUAL ${target})
        message(VERBOSE " Adding GPU target: ${target}")
    endif()
endforeach()

# Skip build if no matching targets found
if(NOT GEMM_ROWCOLQUANT_GPU_TARGETS_INDIVIDUAL)
    message(WARNING "Skipping Tile Engine GEMM RowColQuant build: No supported GPU targets (gfx90a, gfx942, gfx950, gfx1201) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
else()
    message(VERBOSE "Building individual GEMM RowColQuant targets for GPU targets: ${GEMM_ROWCOLQUANT_GPU_TARGETS_INDIVIDUAL}")

    # Enable parallel compilation optimizations
    # Set up job pools for better parallel compilation control
    set_property(GLOBAL PROPERTY JOB_POOLS
        compile_heavy=4    # Limit heavy compilations to prevent OOM
        compile_normal=16  # Allow more parallel normal compilations
    )

    # Enable compiler cache if requested
    if(ENABLE_CCACHE_GEMM_ROWCOLQUANT)
        find_program(CCACHE_PROGRAM ccache)
        if(CCACHE_PROGRAM)
            set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
            message(VERBOSE "Using ccache for faster compilation")
        endif()
    endif()

    # Create master collection target
    add_custom_target(benchmark_gemm_rowcolquant_all)

    # Create datatype collection targets
    foreach(dt IN LISTS GEMM_ROWCOLQUANT_DATATYPE)
        add_custom_target(benchmark_gemm_rowcolquant_${dt})
    endforeach()

    # Create layout collection targets
    foreach(l IN LISTS GEMM_ROWCOLQUANT_LAYOUT)
        add_custom_target(benchmark_gemm_rowcolquant_${l})
    endforeach()

    # Create combined collection targets
    foreach(dt IN LISTS GEMM_ROWCOLQUANT_DATATYPE)
        foreach(l IN LISTS GEMM_ROWCOLQUANT_LAYOUT)
            add_custom_target(benchmark_gemm_rowcolquant_${dt}_${l})
        endforeach()
    endforeach()

    # Create pipeline-specific collection targets
    set(GEMM_ROWCOLQUANT_PIPELINES "compv3")
    foreach(pipeline IN LISTS GEMM_ROWCOLQUANT_PIPELINES)
        add_custom_target(benchmark_gemm_rowcolquant_${pipeline}_pipeline)
    endforeach()

    # Build individual targets for each datatype/layout combination
    foreach(dt IN LISTS GEMM_ROWCOLQUANT_DATATYPE)
        foreach(l IN LISTS GEMM_ROWCOLQUANT_LAYOUT)
            build_individual_gemm_rowcolquant_targets(${dt} ${l})
        endforeach()
    endforeach()
endif()
