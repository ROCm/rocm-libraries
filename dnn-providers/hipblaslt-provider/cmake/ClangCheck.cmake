# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# clang-format tool for code style
set(EXPECTED_CLANG_FORMAT_VERSION "20")

if(ENABLE_CLANG_FORMAT)
    find_program(
        CLANG_FORMAT_BINARY
        NAMES "clang-format-${EXPECTED_CLANG_FORMAT_VERSION}"
        HINTS ${ROCM_PATH}/llvm/bin /opt/rocm/llvm/bin
        DOC "clang-format executable used for formatting"
    )
    if(CLANG_FORMAT_BINARY)
        message(
            STATUS
                "clang-format-${EXPECTED_CLANG_FORMAT_VERSION} has been found; format targets will be built"
        )

        set(CLANG_FORMAT_PRUNE
            -path
            "./build"
            -prune
            -o
            -path
            "./install"
            -prune
            -o
        )
        set(CLANG_FORMAT_REGEX ".*\\.\\(cpp\\|hpp\\|c\\|h\\)")

        # Use prefixed target names in superbuild to avoid collisions
        if(ROCM_LIBS_SUPERBUILD)
            set(_CHECK_FORMAT_TARGET ${PROJECT_NAME}_check_format)
            set(_FORMAT_TARGET ${PROJECT_NAME}_format)
        else()
            set(_CHECK_FORMAT_TARGET check_format)
            set(_FORMAT_TARGET format)
        endif()

        add_custom_target(
            ${_CHECK_FORMAT_TARGET}
            COMMAND find ${CMAKE_CURRENT_SOURCE_DIR} ${CLANG_FORMAT_PRUNE} -regex
                    "${CLANG_FORMAT_REGEX}" -exec ${CLANG_FORMAT_BINARY}
                    -style=file --dry-run --Werror {} +
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Checking code style with clang-format (${PROJECT_NAME})"
            VERBATIM
        )
        add_custom_target(
            ${_FORMAT_TARGET}
            COMMAND find ${CMAKE_CURRENT_SOURCE_DIR} ${CLANG_FORMAT_PRUNE} -regex
                    "${CLANG_FORMAT_REGEX}" -exec ${CLANG_FORMAT_BINARY}
                    -style=file -i {} +
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Applying clang-format to ${PROJECT_NAME} sources"
            VERBATIM
        )

        # Alias targets with consistent hyphenated naming
        add_custom_target(
            ${PROJECT_NAME}-check-format
            DEPENDS ${_CHECK_FORMAT_TARGET}
            COMMENT "Alias for ${_CHECK_FORMAT_TARGET}"
        )
        add_custom_target(
            ${PROJECT_NAME}-format
            DEPENDS ${_FORMAT_TARGET}
            COMMENT "Alias for ${_FORMAT_TARGET}"
        )
    else()
        message(
            WARNING
                "ENABLE_CLANG_FORMAT=ON but clang-format-${EXPECTED_CLANG_FORMAT_VERSION} not found; skipping format targets"
        )
    endif()
endif()
