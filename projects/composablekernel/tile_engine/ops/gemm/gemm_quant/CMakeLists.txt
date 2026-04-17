# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(GEMM_QUANT_DATATYPE "fp8;bf8" CACHE STRING "List of datatypes for GEMM Quant (semicolon-separated)")
set(GEMM_QUANT_LAYOUT "rcr" CACHE STRING "List of layouts for GEMM Quant (semicolon-separated)")
set(GEMM_QUANT_CONFIG_FILE "" CACHE STRING "Custom config file name (without path, must be in configs/ folder)")

set(GEMM_QUANT_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

function(create_individual_gemm_quant_target datatype layout kernel_name trait tile_config config_json)
    set(target_name "benchmark_${kernel_name}")
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")
    string(REPLACE "gemm_quant_" "" simplified_name ${kernel_name})
    set(instance_header "${working_path}/gemm_quant_single_${simplified_name}.hpp")

    add_custom_command(
        OUTPUT ${instance_header}
        COMMAND ${Python3_EXECUTABLE} ${GEMM_QUANT_SOURCE_DIR}/gemm_quant_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --config_json ${config_json}
                --gen_single
                --kernel_name "${kernel_name}"
                --tile_config "${tile_config}"
                --trait_combo "${trait}"
                --gpu_target "${GEMM_QUANT_GPU_TARGETS_INDIVIDUAL}"
        DEPENDS ${GEMM_QUANT_SOURCE_DIR}/gemm_quant_instance_builder.py ${config_json}
        COMMENT "Generating ${instance_header}"
    )

    add_executable(${target_name}
        EXCLUDE_FROM_ALL
        ${GEMM_QUANT_SOURCE_DIR}/gemm_quant_benchmark_single.cpp
        ${instance_header}
    )

    set_property(TARGET ${target_name} PROPERTY HIP_ARCHITECTURES ${GEMM_QUANT_GPU_TARGETS_INDIVIDUAL})

    target_include_directories(${target_name} PRIVATE
        ${GEMM_QUANT_SOURCE_DIR}
        ${working_path}
    )

    target_compile_options(${target_name} PRIVATE
        -Wno-undefined-func-template
        -Wno-float-equal
        --offload-compress
        -include ${instance_header}
    )

    add_dependencies(benchmark_gemm_quant_all ${target_name})
    add_dependencies(benchmark_gemm_quant_${datatype} ${target_name})
    add_dependencies(benchmark_gemm_quant_${layout} ${target_name})
    add_dependencies(benchmark_gemm_quant_${datatype}_${layout} ${target_name})
endfunction()

function(build_individual_gemm_quant_targets datatype layout)
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")

    if(DEFINED ENV{GEMM_QUANT_CONFIG_FILE} AND NOT "$ENV{GEMM_QUANT_CONFIG_FILE}" STREQUAL "")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/$ENV{GEMM_QUANT_CONFIG_FILE}")
    elseif(NOT "${GEMM_QUANT_CONFIG_FILE}" STREQUAL "")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/${GEMM_QUANT_CONFIG_FILE}")
    else()
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/default_config.json")
    endif()

    if(NOT EXISTS ${json_blob})
        message(FATAL_ERROR "Config file not found: ${json_blob}")
    endif()

    file(MAKE_DIRECTORY ${working_path})

    execute_process(
        COMMAND ${Python3_EXECUTABLE} -u ${CMAKE_CURRENT_LIST_DIR}/gemm_quant_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --config_json ${json_blob}
                --gpu_target ${GEMM_QUANT_GPU_TARGETS_INDIVIDUAL}
                --list_kernels
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        RESULT_VARIABLE ret
        ERROR_VARIABLE list_error
    )

    if(NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to list kernels for ${datatype} ${layout}: ${list_error}")
    endif()

    if(EXISTS ${working_path}/gemm_quant_kernel_list.txt)
        file(STRINGS ${working_path}/gemm_quant_kernel_list.txt kernel_lines)
        foreach(line IN LISTS kernel_lines)
            string(REPLACE "|" ";" parts "${line}")
            list(GET parts 0 kernel_name)
            list(GET parts 1 tile_config)
            list(GET parts 2 trait_combo)

            create_individual_gemm_quant_target(
                "${datatype}" "${layout}" "${kernel_name}" "${trait_combo}" "${tile_config}" "${json_blob}")
        endforeach()
    else()
        message(FATAL_ERROR "Kernel list file not found for GEMM Quant")
    endif()
endfunction()

set(GEMM_QUANT_GPU_TARGETS_INDIVIDUAL "")
set(DESIRED_TARGETS "gfx90a;gfx942;gfx950;gfx1201;gfx12-generic")

foreach(target IN LISTS SUPPORTED_GPU_TARGETS)
    if(target IN_LIST DESIRED_TARGETS)
        list(APPEND GEMM_QUANT_GPU_TARGETS_INDIVIDUAL ${target})
    endif()
endforeach()

if(NOT GEMM_QUANT_GPU_TARGETS_INDIVIDUAL)
    message(WARNING "Skipping Tile Engine GEMM Quant build: No supported GPU targets found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
else()
    add_custom_target(benchmark_gemm_quant_all)

    foreach(datatype ${GEMM_QUANT_DATATYPE})
        add_custom_target(benchmark_gemm_quant_${datatype})
    endforeach()

    foreach(layout ${GEMM_QUANT_LAYOUT})
        add_custom_target(benchmark_gemm_quant_${layout})
    endforeach()

    foreach(datatype ${GEMM_QUANT_DATATYPE})
        foreach(layout ${GEMM_QUANT_LAYOUT})
            add_custom_target(benchmark_gemm_quant_${datatype}_${layout})
            build_individual_gemm_quant_targets(${datatype} ${layout})
        endforeach()
    endforeach()
endif()
