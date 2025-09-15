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
)

add_custom_target(hipblaslt-test-data DEPENDS ${HIPBLASLT_TEST_DATA})
add_dependencies(hipblaslt-test hipblaslt-test-data)
