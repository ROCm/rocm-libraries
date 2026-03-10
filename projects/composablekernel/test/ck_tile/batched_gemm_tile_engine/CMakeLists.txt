# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(TILE_ENGINE_BATCHED_GEMM_DIR "${PROJECT_SOURCE_DIR}/tile_engine/ops/gemm/batched_gemm")

if(NOT EXISTS ${TILE_ENGINE_BATCHED_GEMM_DIR})
    message(WARNING "Tile engine batched GEMM directory not found: ${TILE_ENGINE_BATCHED_GEMM_DIR}")
    return()
endif()

set(BATCHED_GEMM_TEST_GPU_TARGETS "")
set(DESIRED_TARGETS "gfx90a;gfx942")

foreach(target IN LISTS SUPPORTED_GPU_TARGETS)
    if(target IN_LIST DESIRED_TARGETS)
        list(APPEND BATCHED_GEMM_TEST_GPU_TARGETS ${target})
    endif()
endforeach()

if(NOT BATCHED_GEMM_TEST_GPU_TARGETS)
    message(WARNING "Skipping batched GEMM tile engine tests: No supported GPU targets (gfx90a, gfx942) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
    return()
endif()

set(TEST_DATATYPE "fp16")
set(TEST_LAYOUT "rcr")
set(TEST_CONFIG_JSON "${CMAKE_CURRENT_SOURCE_DIR}/configs/simple_test_config.json")

if(NOT EXISTS ${TEST_CONFIG_JSON})
    message(WARNING "Missing batched GEMM tile engine test config: ${TEST_CONFIG_JSON}")
    return()
endif()

set(WORKING_PATH "${CMAKE_CURRENT_BINARY_DIR}/${TEST_DATATYPE}/${TEST_LAYOUT}")
file(MAKE_DIRECTORY ${WORKING_PATH})

execute_process(
    COMMAND ${Python3_EXECUTABLE} -u ${TILE_ENGINE_BATCHED_GEMM_DIR}/batched_gemm_instance_builder.py
            --working_path ${WORKING_PATH}
            --gpu_target "${BATCHED_GEMM_TEST_GPU_TARGETS}"
            --datatype ${TEST_DATATYPE}
            --layout ${TEST_LAYOUT}
            --config_json ${TEST_CONFIG_JSON}
            --list_kernels
    WORKING_DIRECTORY ${TILE_ENGINE_BATCHED_GEMM_DIR}
    RESULT_VARIABLE ret
    ERROR_VARIABLE list_error
)

if(NOT ret EQUAL 0)
    message(WARNING "Failed to list batched GEMM kernels: ${list_error}")
    return()
endif()

if(NOT EXISTS ${WORKING_PATH}/batched_gemm_kernel_list.txt)
    message(WARNING "No batched GEMM kernels generated for test configuration")
    return()
endif()

file(STRINGS ${WORKING_PATH}/batched_gemm_kernel_list.txt KERNEL_LINES)

foreach(line IN LISTS KERNEL_LINES)
    string(REPLACE "|" ";" parts "${line}")
    list(LENGTH parts parts_len)
    if(NOT parts_len EQUAL 3)
        continue()
    endif()

    list(GET parts 0 kernel_name)
    list(GET parts 1 tile_config)
    list(GET parts 2 trait_combo)

    execute_process(
        COMMAND ${Python3_EXECUTABLE} -u ${TILE_ENGINE_BATCHED_GEMM_DIR}/batched_gemm_instance_builder.py
                --working_path ${WORKING_PATH}
                --gpu_target "${BATCHED_GEMM_TEST_GPU_TARGETS}"
                --datatype ${TEST_DATATYPE}
                --layout ${TEST_LAYOUT}
                --config_json ${TEST_CONFIG_JSON}
                --gen_single
                --kernel_name "${kernel_name}"
                --tile_config "${tile_config}"
                --trait_combo "${trait_combo}"
        WORKING_DIRECTORY ${TILE_ENGINE_BATCHED_GEMM_DIR}
        RESULT_VARIABLE gen_ret
        ERROR_VARIABLE gen_error
    )

    if(NOT gen_ret EQUAL 0)
        message(WARNING "Failed to generate batched GEMM header for ${kernel_name}: ${gen_error}")
        continue()
    endif()

    set(TEST_HEADER "${WORKING_PATH}/batched_gemm_single_${TEST_DATATYPE}_${TEST_LAYOUT}_${trait_combo}_${tile_config}.hpp")
    set(TARGET_NAME "test_batched_gemm_tile_engine_${TEST_DATATYPE}_${TEST_LAYOUT}_${trait_combo}_${tile_config}")

    add_gtest_executable(${TARGET_NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_batched_gemm_simple.cpp
    )

    set_property(TARGET ${TARGET_NAME} PROPERTY HIP_ARCHITECTURES ${BATCHED_GEMM_TEST_GPU_TARGETS})

    target_compile_definitions(${TARGET_NAME} PRIVATE
        BATCHED_GEMM_SINGLE_INSTANCE_HPP="${TEST_HEADER}"
    )

    target_include_directories(${TARGET_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/include
        ${PROJECT_SOURCE_DIR}
        ${WORKING_PATH}
        ${GTEST_INCLUDE_DIRS}
    )

    target_compile_options(${TARGET_NAME} PRIVATE
        -Wno-undefined-func-template
        -Wno-float-equal
        --offload-compress
        -include ${TEST_HEADER}
    )

    if(CK_USE_OCP_FP8)
        target_compile_options(${TARGET_NAME} PRIVATE -DCK_TILE_USE_OCP_FP8)
    endif()
endforeach()
