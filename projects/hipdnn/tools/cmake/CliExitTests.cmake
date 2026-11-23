# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# hipdnn_add_cli_exit_tests(<prefix> <tool target> <case>...)
#
# One test per case, checking the exit status a command line gets: what scripts branch on.
# A case is "name|comma-separated arguments|expected exit|expected output", and becomes test
# <prefix>_<name>, run by CliExitStatus.cmake. Build-tree only: the driver is not installed.
function(hipdnn_add_cli_exit_tests prefix tool)
    foreach(_case IN LISTS ARGN)
        string(REPLACE "|" ";" _fields "${_case}")
        list(GET _fields 0 _name)
        list(GET _fields 1 _args)
        list(GET _fields 2 _exit)
        list(GET _fields 3 _output)
        set(_test ${prefix}_${_name})
        add_test(NAME ${_test}
            COMMAND ${CMAKE_COMMAND} "-DTOOL=$<TARGET_FILE:${tool}>"
                    "-DCLI_ARGS=${_args}" "-DEXPECTED_EXIT=${_exit}"
                    "-DEXPECTED_OUTPUT=${_output}"
                    -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/CliExitStatus.cmake")
        set_tests_properties(${_test} PROPERTIES TIMEOUT 60)
        if(DEFINED TEST_ENVIRONMENT)
            set_tests_properties(${_test} PROPERTIES ENVIRONMENT "${TEST_ENVIRONMENT}")
        endif()
        # The tool loads the backend DLL from the build tree's bin directory on Windows.
        if(DEFINED TEST_ENVIRONMENT_MODIFICATION)
            set_tests_properties(${_test} PROPERTIES
                ENVIRONMENT_MODIFICATION "${TEST_ENVIRONMENT_MODIFICATION}")
        endif()
        _apply_hipdnn_test_category_labels(${_test})
    endforeach()
endfunction()
