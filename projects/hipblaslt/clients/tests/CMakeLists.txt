# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_subdirectory(src)

set(HIPBLASLT_TEST_DATA "${CMAKE_CURRENT_BINARY_DIR}/../hipblaslt_gtest.data")

add_custom_command(
    OUTPUT "${HIPBLASLT_TEST_DATA}"
    COMMAND ${HIPBLASLT_PYTHON_COMMAND} "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_gentest.py"
                                        -I "${CMAKE_CURRENT_SOURCE_DIR}/../common/include"
                                        "${CMAKE_CURRENT_SOURCE_DIR}/data/hipblaslt_gtest.yaml"
                                        -o ${HIPBLASLT_TEST_DATA}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_gentest.py
                                        ${CMAKE_CURRENT_SOURCE_DIR}/data/hipblaslt_common.yaml
                                        ${CMAKE_CURRENT_SOURCE_DIR}/data/matmul_common.yaml
                                        ${CMAKE_CURRENT_SOURCE_DIR}/data/matmul_gtest.yaml
                                        ${CMAKE_CURRENT_SOURCE_DIR}/data/auxiliary_gtest.yaml
                                        ${CMAKE_CURRENT_SOURCE_DIR}/data/smoke_gtest.yaml
                                        ${CMAKE_CURRENT_SOURCE_DIR}/data/rocroller_gtest.yaml

)

add_custom_target(hipblaslt-test-data DEPENDS ${HIPBLASLT_TEST_DATA})
add_dependencies(hipblaslt-test hipblaslt-test-data)
if (HIPBLASLT_ENABLE_COVERAGE)
    target_include_directories(hipblaslt-test
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../common/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../tensilelite/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../tensilelite>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/amd_detail/include>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/amd_detail>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/include>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/include>
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/../../../tensilelite"
    )
endif()
