# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(ENABLE_CLANG_FORMAT)
    include(${CMAKE_CURRENT_LIST_DIR}/CheckToolVersion.cmake)

    set(CLANG_FORMAT_PRUNE -path "./build" -prune -o -path "./sdk/include/hipdnn_sdk/data_objects" -prune -o)

    # Find and check clang-format version using unified function
    findAndCheckClangFormat()

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
endif()