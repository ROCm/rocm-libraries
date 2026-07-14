# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

get_filename_component(_rocm_libraries_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

add_test(
    NAME shared_ctest_optional_args_scope
    COMMAND
        ${CMAKE_COMMAND} -S "${CMAKE_CURRENT_LIST_DIR}/optional_args_scope" -B
        "${CMAKE_CURRENT_BINARY_DIR}/optional_args_scope"
        -DROCM_LIBRARIES_ROOT=${_rocm_libraries_root}
)
