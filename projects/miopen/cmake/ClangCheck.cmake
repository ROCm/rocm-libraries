# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

set(CLANG_FORMAT_PRUNE -path "./build" -prune -o -path "./install" -prune -o -path "./fin" -prune -o)

# Note: The clang-format in /opt/rocm produces different results than the one in /usr/bin.  MIOpen
# formatting is based on the one in /usr/bin so we use that one
# set(CLANG_FORMAT_BINARY /opt/rocm/llvm/bin/clang-format)
set(CLANG_FORMAT_BINARY /usr/bin/clang-format-18)

find_program(PRE_COMMIT_BINARY pre-commit)
get_filename_component(REPO_ROOT "${CMAKE_SOURCE_DIR}/../.." ABSOLUTE)

if(NOT EXISTS "${REPO_ROOT}/.pre-commit-config.yaml")
    message(WARNING "Expected .pre-commit-config.yaml not found at ${REPO_ROOT}; trailing whitespace removal via pre-commit will be skipped")
    set(PRE_COMMIT_BINARY "")
endif()

add_custom_target(
    check_format
    COMMAND  find . ${CLANG_FORMAT_PRUNE} -regex ".*\\.\\(cpp\\|hpp\\|h.in\\|hpp.in\\|cpp.in\\|cl\\)" -exec ${CLANG_FORMAT_BINARY} --dry-run --Werror --verbose {} +
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    VERBATIM
)

if(PRE_COMMIT_BINARY)
    # Write a script so CMake VERBATIM can be used while still allowing shell $(git ls-files ...) expansion.
    file(WRITE "${CMAKE_BINARY_DIR}/trim_whitespace.sh"
        "#!/bin/sh\n"
        "set -e\n"
        "cd \"${REPO_ROOT}\"\n"
        "${PRE_COMMIT_BINARY} run trailing-whitespace --files $(git ls-files projects/miopen)\n"
    )

    add_custom_target(
        trim_whitespace
        COMMAND sh "${CMAKE_BINARY_DIR}/trim_whitespace.sh"
        VERBATIM
    )
else()
    message(WARNING "pre-commit not found; trim_whitespace target will not be available and trailing whitespace removal is skipped in format")

    add_custom_target(
        trim_whitespace
        COMMAND ${CMAKE_COMMAND} -E echo "pre-commit not found, skipping trailing whitespace removal"
    )
endif()

add_custom_target(
    format
    COMMAND find . ${CLANG_FORMAT_PRUNE} -regex ".*\\.\\(cpp\\|hpp\\|h.in\\|hpp.in\\|cpp.in\\|cl\\)" -exec ${CLANG_FORMAT_BINARY} --verbose -i {} +
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    VERBATIM
)
add_dependencies(format trim_whitespace)
