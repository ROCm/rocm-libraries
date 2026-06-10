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
    message(WARNING "Expected .pre-commit-config.yaml not found at ${REPO_ROOT}; pre-commit checks will be skipped")
    set(PRE_COMMIT_BINARY "")
endif()

add_custom_target(
    check_format
    COMMAND  find . ${CLANG_FORMAT_PRUNE} -regex ".*\\.\\(cpp\\|hpp\\|h.in\\|hpp.in\\|cpp.in\\|cl\\)" -exec ${CLANG_FORMAT_BINARY} --dry-run --Werror --verbose {} +
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    VERBATIM
)

if(PRE_COMMIT_BINARY)
    # pre-commit exits 1 when it modifies files (expected); only fail on higher codes.
    add_custom_target(
        pre_commit_checks
        COMMAND sh -c "${PRE_COMMIT_BINARY} run --files $(git ls-files projects/miopen); rc=$?; [ $rc -le 1 ] || exit $rc"
        WORKING_DIRECTORY ${REPO_ROOT}
        VERBATIM
    )
else()
    message(WARNING "pre-commit not found; pre_commit_checks target will not be available and pre-commit checks are skipped in format")

    add_custom_target(
        pre_commit_checks
        COMMAND ${CMAKE_COMMAND} -E echo "pre-commit not found, skipping pre-commit checks"
    )
endif()

add_custom_target(
    format
    COMMAND find . ${CLANG_FORMAT_PRUNE} -regex ".*\\.\\(cpp\\|hpp\\|h.in\\|hpp.in\\|cpp.in\\|cl\\)" -exec ${CLANG_FORMAT_BINARY} --verbose -i {} +
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    VERBATIM
)
add_dependencies(format pre_commit_checks)
