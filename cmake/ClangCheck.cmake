# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

set(CLANG_FORMAT_PRUNE -path "./build" -prune -o -path "./sdk/include/hipdnn_sdk/data_objects" -prune -o)

find_program(CLANG_FORMAT_BINARY NAMES clang-format ${ROCM_PATH}/llvm/bin/clang-format)

if(NOT CLANG_FORMAT_BINARY)
    message(FATAL_ERROR "clang-format not found in PATH or ${ROCM_PATH}/llvm/bin")
endif()

add_custom_target(
    check_format
    COMMAND  find . ${CLANG_FORMAT_PRUNE} -regex ".*\\.\\(cpp\\|hpp\\|c\\|h\\)" -exec ${CLANG_FORMAT_BINARY} --dry-run --Werror {} +
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    VERBATIM
)

add_custom_target(
    format
    COMMAND  find . ${CLANG_FORMAT_PRUNE} -regex ".*\\.\\(cpp\\|hpp\\|c\\|h\\)" -exec ${CLANG_FORMAT_BINARY} --verbose -i {} +
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    VERBATIM
)
