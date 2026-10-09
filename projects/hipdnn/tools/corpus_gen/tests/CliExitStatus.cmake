# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Runs hipdnn_corpus_gen with CLI_ARGS and checks its exit status, and that EXPECTED_OUTPUT
# appears in what it printed. CLI_ARGS is comma-separated: a semicolon would split the
# add_test() command line.
if(NOT DEFINED CORPUS_GEN_TOOL OR NOT DEFINED CLI_ARGS OR NOT DEFINED EXPECTED_EXIT
   OR NOT DEFINED EXPECTED_OUTPUT)
    message(FATAL_ERROR
        "CORPUS_GEN_TOOL, CLI_ARGS, EXPECTED_EXIT and EXPECTED_OUTPUT are required")
endif()

string(REPLACE "," ";" cliArgs "${CLI_ARGS}")
execute_process(
    COMMAND "${CORPUS_GEN_TOOL}" ${cliArgs}
    OUTPUT_VARIABLE toolStdout
    ERROR_VARIABLE toolStderr
    RESULT_VARIABLE toolResult
)

if(NOT "${toolResult}" STREQUAL "${EXPECTED_EXIT}")
    message(FATAL_ERROR
        "hipdnn_corpus_gen ${cliArgs} exited with '${toolResult}', expected ${EXPECTED_EXIT}\n"
        "stdout:\n${toolStdout}\nstderr:\n${toolStderr}")
endif()

string(FIND "${toolStdout}${toolStderr}" "${EXPECTED_OUTPUT}" outputPosition)
if(outputPosition EQUAL -1)
    message(FATAL_ERROR
        "hipdnn_corpus_gen ${cliArgs} did not print '${EXPECTED_OUTPUT}'\n"
        "stdout:\n${toolStdout}\nstderr:\n${toolStderr}")
endif()
