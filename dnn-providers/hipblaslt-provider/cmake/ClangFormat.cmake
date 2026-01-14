# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# clang-format tool for code style

if(HIPBLASLT_ENABLE_CLANG_FORMAT)
    find_program(
        HIPBLASLT_CLANG_FORMAT_BIN
        NAMES clang-format-20
        HINTS ${ROCM_PATH}/llvm/bin /opt/rocm/llvm/bin
        DOC "clang-format executable used for formatting"
    )
    if(HIPBLASLT_CLANG_FORMAT_BIN)
        message(STATUS "clang-format-20 has been found; format targets will be built")

        set(HIPBLASLT_CLANG_FORMAT_PRUNE
            -path
            "./build"
            -prune
            -o
            -path
            "./install"
            -prune
            -o
        )
        set(HIPBLASLT_CLANG_FORMAT_REGEX ".*\\.\\(cpp\\|hpp\\|h\\)")
        add_custom_target(
            hipblaslt_check_format
            COMMAND find ${CMAKE_CURRENT_SOURCE_DIR} ${HIPBLASLT_CLANG_FORMAT_PRUNE} -regex
                    "${HIPBLASLT_CLANG_FORMAT_REGEX}" -exec ${HIPBLASLT_CLANG_FORMAT_BIN}
                    -style=file --dry-run --Werror {} +
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Checking code style with clang-format"
            VERBATIM
        )
        add_custom_target(
            hipblaslt_format
            COMMAND find ${CMAKE_CURRENT_SOURCE_DIR} ${HIPBLASLT_CLANG_FORMAT_PRUNE} -regex
                    "${HIPBLASLT_CLANG_FORMAT_REGEX}" -exec ${HIPBLASLT_CLANG_FORMAT_BIN}
                    -style=file -i {} +
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Applying clang-format to hipblaslt-provider sources"
            VERBATIM
        )
    else()
        message(
            WARNING
                "HIPBLASLT_ENABLE_CLANG_FORMAT=ON but clang-format-20 not found; skipping format targets"
        )
    endif()
endif()