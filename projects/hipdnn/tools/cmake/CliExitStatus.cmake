# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Runs TOOL with CLI_ARGS and checks its exit status, and that EXPECTED_OUTPUT appears in what
# it printed. CLI_ARGS is comma-separated: a semicolon would split the add_test() command line.
# `{empty}` is an empty argument, which a CMake list cannot carry.
if(NOT DEFINED TOOL OR NOT DEFINED CLI_ARGS OR NOT DEFINED EXPECTED_EXIT
   OR NOT DEFINED EXPECTED_OUTPUT)
    message(FATAL_ERROR "TOOL, CLI_ARGS, EXPECTED_EXIT and EXPECTED_OUTPUT are required")
endif()

string(REPLACE "," ";" cliArgs "${CLI_ARGS}")
set(quotedArgs "")
foreach(cliArg IN LISTS cliArgs)
    if(cliArg STREQUAL "{empty}")
        set(cliArg "")
    endif()
    string(APPEND quotedArgs " [==[${cliArg}]==]")
endforeach()
cmake_language(EVAL CODE "
execute_process(
    COMMAND [==[${TOOL}]==]${quotedArgs}
    OUTPUT_VARIABLE toolStdout
    ERROR_VARIABLE toolStderr
    RESULT_VARIABLE toolResult
)")

if(NOT "${toolResult}" STREQUAL "${EXPECTED_EXIT}")
    message(FATAL_ERROR
        "${TOOL} ${cliArgs} exited with '${toolResult}', expected ${EXPECTED_EXIT}\n"
        "stdout:\n${toolStdout}\nstderr:\n${toolStderr}")
endif()

string(FIND "${toolStdout}${toolStderr}" "${EXPECTED_OUTPUT}" outputPosition)
if(outputPosition EQUAL -1)
    message(FATAL_ERROR
        "${TOOL} ${cliArgs} did not print '${EXPECTED_OUTPUT}'\n"
        "stdout:\n${toolStdout}\nstderr:\n${toolStderr}")
endif()
