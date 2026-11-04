# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Runs hipdnn_corpus_gen into OUTPUT_DIR and checks what a run leaves: its exit status and,
# for a corpus of EXPECTED_GRAPHS graphs, a manifest that says whether an engine was asked
# (--without-engine: no, and warns so) and a commands.txt with one bench invocation per graph
# file it wrote.
# CLI_ARGS is comma-separated, as in CliExitStatus.cmake.
if(NOT DEFINED TOOL OR NOT DEFINED CLI_ARGS OR NOT DEFINED OUTPUT_DIR
   OR NOT DEFINED EXPECTED_EXIT OR NOT DEFINED EXPECTED_GRAPHS)
    message(FATAL_ERROR "TOOL, CLI_ARGS, OUTPUT_DIR, EXPECTED_EXIT and EXPECTED_GRAPHS are required")
endif()

file(REMOVE_RECURSE "${OUTPUT_DIR}")
string(REPLACE "," ";" cliArgs "${CLI_ARGS}")
execute_process(
    COMMAND "${TOOL}" ${cliArgs} --output "${OUTPUT_DIR}"
    OUTPUT_VARIABLE toolStdout
    ERROR_VARIABLE toolStderr
    RESULT_VARIABLE toolResult
)
if(NOT "${toolResult}" STREQUAL "${EXPECTED_EXIT}")
    message(FATAL_ERROR "${TOOL} ${cliArgs} exited with '${toolResult}', expected ${EXPECTED_EXIT}\n"
                        "stdout:\n${toolStdout}\nstderr:\n${toolStderr}")
endif()
if(EXPECTED_GRAPHS EQUAL 0)
    return()
endif()

file(READ "${OUTPUT_DIR}/manifest.json" manifest)
string(JSON verified GET "${manifest}" reports engine_verified)
if(CLI_ARGS MATCHES "--without-engine")
    if(verified)
        message(FATAL_ERROR "a --without-engine manifest says engine_verified: ${verified}")
    endif()
    string(JSON warning GET "${manifest}" reports warning)
    if(NOT warning MATCHES "--without-engine")
        message(FATAL_ERROR "the manifest does not warn that no engine was asked: '${warning}'")
    endif()
elseif(NOT verified)
    message(FATAL_ERROR "a run that named an engine says engine_verified: ${verified}")
endif()
string(JSON graphs LENGTH "${manifest}" graphs)
if(NOT graphs EQUAL EXPECTED_GRAPHS)
    message(FATAL_ERROR "the manifest lists ${graphs} graphs, expected ${EXPECTED_GRAPHS}")
endif()

file(STRINGS "${OUTPUT_DIR}/commands.txt" lines)
set(commands 0)
foreach(line IN LISTS lines)
    if(line MATCHES "^#" OR line STREQUAL "")
        continue()
    endif()
    math(EXPR commands "${commands} + 1")
    if(NOT line MATCHES "--graph '([^']+)'")
        message(FATAL_ERROR "a command names no graph: ${line}")
    endif()
    if(NOT EXISTS "${CMAKE_MATCH_1}")
        message(FATAL_ERROR "a command names a graph that was not written: ${CMAKE_MATCH_1}")
    endif()
endforeach()
if(NOT commands EQUAL EXPECTED_GRAPHS)
    message(FATAL_ERROR "commands.txt has ${commands} commands, expected ${EXPECTED_GRAPHS}")
endif()
