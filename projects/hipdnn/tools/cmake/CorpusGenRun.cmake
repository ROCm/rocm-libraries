# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Runs hipdnn_corpus_gen into OUTPUT_DIR and checks what a run leaves: its exit status and,
# for a corpus of EXPECTED_GRAPHS graphs, a manifest that says whether an engine was asked
# (--without-engine: no, and warns so) and a commands.txt with one bench invocation per graph
# file it wrote. Given EXPECTED_FORMS, it also checks the forms the run reports (see the end).
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

# EXPECTED_FORMS (optional, comma-separated labels): the corpus carries a `form` facet column of
# its own right after `grouping` in manifest.csv, holding exactly these labels, and the run
# reports each as a form quota (`form:<label>`, in reports.regime_quota of the --operation)
# that delivered at least one graph.
if(NOT EXPECTED_FORMS)
    return()
endif()
string(REPLACE "," ";" expectedForms "${EXPECTED_FORMS}")

# The columns up to `form` are generated labels, never quoted, so a plain split finds it.
file(STRINGS "${OUTPUT_DIR}/manifest.csv" rows)
list(POP_FRONT rows header)
string(REPLACE "," ";" columns "${header}")
list(FIND columns "grouping" groupingColumn)
list(FIND columns "form" formColumn)
if(groupingColumn LESS 0 OR formColumn LESS 0)
    message(FATAL_ERROR "manifest.csv lacks a grouping or form column: ${header}")
endif()
math(EXPR afterGrouping "${groupingColumn} + 1")
if(NOT formColumn EQUAL afterGrouping)
    message(FATAL_ERROR "the form column is not right after grouping in manifest.csv: ${header}")
endif()
set(forms "")
foreach(row IN LISTS rows)
    string(REPLACE "," ";" cells "${row}")
    list(GET cells ${formColumn} form)
    list(APPEND forms "${form}")
endforeach()
list(REMOVE_DUPLICATES forms)
list(SORT forms)
list(SORT expectedForms)
if(NOT "${forms}" STREQUAL "${expectedForms}")
    message(FATAL_ERROR "manifest.csv holds the forms '${forms}', expected '${expectedForms}'")
endif()

if(NOT CLI_ARGS MATCHES "--operation,([^,]+)")
    message(FATAL_ERROR "EXPECTED_FORMS needs an --operation to look its form quotas up under")
endif()
set(operation "${CMAKE_MATCH_1}")
foreach(form IN LISTS expectedForms)
    string(JSON delivered ERROR_VARIABLE quotaError
           GET "${manifest}" reports regime_quota "${operation}" "form:${form}" delivered)
    if(quotaError OR delivered LESS 1)
        message(FATAL_ERROR "reports.regime_quota.${operation} does not show form:${form} "
                            "delivering a graph: '${delivered}' ${quotaError}")
    endif()
endforeach()
