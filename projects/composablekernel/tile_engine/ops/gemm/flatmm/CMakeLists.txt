# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(FLATMM_DATATYPE "fp16;fp8;bf16;bf8" CACHE STRING "List of datatypes for FLATMM (semicolon-separated)")
set(FLATMM_LAYOUT "rcr" CACHE STRING "List of layouts for FLATMM (semicolon-separated)")
set(FLATMM_CONFIG_FILE "" CACHE STRING "Custom config file name (without path, must be in configs/ folder)")
option(ENABLE_CCACHE_FLATMM "Enable ccache for FLATMM ops compilation" OFF)

set(FLATMM_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

function(create_individual_flatmm_target datatype layout trait tile_config config_json)
    # Use the parent scope FLATMM_GPU_TARGETS_INDIVIDUAL variable
    if(NOT FLATMM_GPU_TARGETS_INDIVIDUAL)
        message(WARNING "Skipping individual FLATMM target ${datatype}_${layout}_${trait}_${tile_config}: No supported GPU targets")
        return()
    endif()

    set(target_name "benchmark_flatmm_${datatype}_${layout}_${trait}_${tile_config}")
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")
    set(instance_header "${working_path}/flatmm_single_${datatype}_${layout}_${trait}_${tile_config}.hpp")

    add_custom_command(
        OUTPUT ${instance_header}
        COMMAND ${Python3_EXECUTABLE} ${FLATMM_SOURCE_DIR}/flatmm_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --config_json ${config_json}
                --gpu_target "${FLATMM_GPU_TARGETS_INDIVIDUAL}"
                --gen_single
                --kernel_name "flatmm_${datatype}_${layout}_${trait}_${tile_config}"
                --tile_config "${tile_config}"
                --trait_combo "${trait}"
        DEPENDS ${FLATMM_SOURCE_DIR}/flatmm_instance_builder.py ${config_json}
        COMMENT "Generating ${instance_header}"
    )

    add_executable(${target_name}
        EXCLUDE_FROM_ALL
        ${FLATMM_SOURCE_DIR}/flatmm_benchmark_single.cpp
        ${instance_header}
    )

    set_property(TARGET ${target_name} PROPERTY HIP_ARCHITECTURES ${target_archs})

    target_compile_definitions(${target_name} PRIVATE
        FLATMM_SINGLE_INSTANCE_HPP="${instance_header}"
    )

    target_include_directories(${target_name} PRIVATE
        ${FLATMM_SOURCE_DIR}
        ${working_path}
    )

    target_compile_options(${target_name} PRIVATE
        -Wno-undefined-func-template
        -Wno-float-equal
        --offload-compress
        -include ${instance_header}
    )

    add_dependencies(benchmark_flatmm_all ${target_name})
    add_dependencies(benchmark_flatmm_${datatype} ${target_name})
    add_dependencies(benchmark_flatmm_${layout} ${target_name})
    add_dependencies(benchmark_flatmm_${datatype}_${layout} ${target_name})

    string(REPLACE "_" ";" trait_parts ${trait})
    list(GET trait_parts 0 pipeline)
    list(GET trait_parts 1 epilogue)
    list(GET trait_parts 2 scheduler)

    add_dependencies(benchmark_flatmm_${pipeline}_pipeline ${target_name})
    add_dependencies(benchmark_flatmm_${epilogue}_epilogue ${target_name})
    add_dependencies(benchmark_flatmm_${scheduler}_scheduler ${target_name})
endfunction()

function(build_individual_flatmm_targets datatype layout)
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")

    if(DEFINED ENV{FLATMM_CONFIG_FILE} AND NOT "$ENV{FLATMM_CONFIG_FILE}" STREQUAL "")
        set(config_filename "$ENV{FLATMM_CONFIG_FILE}")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/${config_filename}")
        message(VERBOSE "  Using config from environment variable: ${config_filename}")
    elseif(NOT "${FLATMM_CONFIG_FILE}" STREQUAL "")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/${FLATMM_CONFIG_FILE}")
        message(VERBOSE "  Using custom config: ${FLATMM_CONFIG_FILE}")
    else()
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/default_config.json")
        message(VERBOSE "  Using default config for layout ${layout}")
    endif()

    if(NOT EXISTS ${json_blob})
        message(FATAL_ERROR "Config file not found: ${json_blob}")
    endif()

    file(MAKE_DIRECTORY ${working_path})

    message(VERBOSE "Generating FLATMM kernels for ${datatype} ${layout}")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -u ${CMAKE_CURRENT_LIST_DIR}/flatmm_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --config_json ${json_blob}
                --gpu_target "${FLATMM_GPU_TARGETS_CSV}"
                --list_kernels
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        RESULT_VARIABLE ret
        OUTPUT_VARIABLE list_output
        ERROR_VARIABLE list_error
    )

    if(NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to list FLATMM kernels for ${datatype} ${layout}: ${list_error}")
    endif()

    if(EXISTS ${working_path}/flatmm_kernel_count.txt)
        file(READ ${working_path}/flatmm_kernel_count.txt kernel_count)
        string(STRIP "${kernel_count}" kernel_count)
        message(VERBOSE "  Found ${kernel_count} FLATMM kernel configurations")
    else()
        message(FATAL_ERROR "Kernel count file not found")
    endif()

    if(EXISTS ${working_path}/flatmm_kernel_list.txt)
        file(STRINGS ${working_path}/flatmm_kernel_list.txt kernel_lines)
        foreach(line IN LISTS kernel_lines)
            string(REPLACE "|" ";" parts "${line}")
            list(GET parts 0 kernel_name)
            list(GET parts 1 tile_config)
            list(GET parts 2 trait_combo)

            create_individual_flatmm_target(
                "${datatype}"
                "${layout}"
                "${trait_combo}"
                "${tile_config}"
                "${json_blob}")
        endforeach()
    else()
        message(FATAL_ERROR "Kernel list file not found")
    endif()
endfunction()

message(VERBOSE "=== Starting Tile Engine FLATMM Configuration ===")
message(VERBOSE "FLATMM_DATATYPE: ${FLATMM_DATATYPE}")
message(VERBOSE "FLATMM_LAYOUT: ${FLATMM_LAYOUT}")
message(VERBOSE "SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")

set(FLATMM_GPU_TARGETS_INDIVIDUAL "")
set(DESIRED_TARGETS "gfx90a;gfx942;gfx950")

foreach(target IN LISTS SUPPORTED_GPU_TARGETS)
    if(target IN_LIST DESIRED_TARGETS)
        list(APPEND FLATMM_GPU_TARGETS_INDIVIDUAL ${target})
        message(VERBOSE "  Adding GPU target: ${target}")
    endif()
endforeach()

if(NOT FLATMM_GPU_TARGETS_INDIVIDUAL)
    message(WARNING "Skipping Tile Engine FLATMM build: No supported GPU targets (gfx90a, gfx942, gfx950) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
else()
    string(REPLACE ";" "," FLATMM_GPU_TARGETS_CSV "${FLATMM_GPU_TARGETS_INDIVIDUAL}")
    message(VERBOSE "Building individual FLATMM targets for GPU targets: ${FLATMM_GPU_TARGETS_INDIVIDUAL}")

    set_property(GLOBAL PROPERTY JOB_POOLS
        compile_heavy=4
        compile_normal=16
    )

    if(ENABLE_CCACHE_FLATMM)
        find_program(CCACHE_PROGRAM ccache)
        if(CCACHE_PROGRAM)
            set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
            message(VERBOSE "Using ccache for FLATMM compilation")
        else()
            message(WARNING "ccache requested but not found")
        endif()
    else()
        message(VERBOSE "ccache disabled for FLATMM ops (use -DENABLE_CCACHE_FLATMM=ON to enable)")
    endif()

    add_custom_target(benchmark_flatmm_all)

    foreach(dt IN LISTS FLATMM_DATATYPE)
        add_custom_target(benchmark_flatmm_${dt})
    endforeach()

    foreach(l IN LISTS FLATMM_LAYOUT)
        add_custom_target(benchmark_flatmm_${l})
    endforeach()

    foreach(dt IN LISTS FLATMM_DATATYPE)
        foreach(l IN LISTS FLATMM_LAYOUT)
            add_custom_target(benchmark_flatmm_${dt}_${l})
        endforeach()
    endforeach()

    set(FLATMM_PIPELINES "regv1")
    set(FLATMM_EPILOGUES "cshuffle")
    set(FLATMM_SCHEDULERS "default")

    foreach(pipeline IN LISTS FLATMM_PIPELINES)
        add_custom_target(benchmark_flatmm_${pipeline}_pipeline)
    endforeach()

    foreach(epilogue IN LISTS FLATMM_EPILOGUES)
        add_custom_target(benchmark_flatmm_${epilogue}_epilogue)
    endforeach()

    foreach(scheduler IN LISTS FLATMM_SCHEDULERS)
        add_custom_target(benchmark_flatmm_${scheduler}_scheduler)
    endforeach()

    foreach(dt IN LISTS FLATMM_DATATYPE)
        foreach(l IN LISTS FLATMM_LAYOUT)
            build_individual_flatmm_targets(${dt} ${l})
        endforeach()
    endforeach()
endif()
