# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Runs hipdnn_corpus_gen --without-engine more than once and compares the corpora by their
# graphs' content keys (manifest `benchmark`). MODE is
#   exclude  a second corpus excluding the first shares none of its graphs;
#   seed     the same seed gives the same graphs, and another seed other graphs.
# CLI_ARGS is comma-separated, as in CliExitStatus.cmake.
cmake_policy(SET CMP0057 NEW) # if(IN_LIST)
if(NOT DEFINED TOOL OR NOT DEFINED CLI_ARGS OR NOT DEFINED OUTPUT_DIR OR NOT DEFINED MODE)
    message(FATAL_ERROR "TOOL, CLI_ARGS, OUTPUT_DIR and MODE are required")
endif()

string(REPLACE "," ";" cliArgs "${CLI_ARGS}")
file(REMOVE_RECURSE "${OUTPUT_DIR}")

# Runs the tool into OUTPUT_DIR/<name> with the extra arguments, and sets <name>_keys to the
# corpus's content keys, in manifest order.
function(corpus name)
    execute_process(
        COMMAND "${TOOL}" ${cliArgs} ${ARGN} --output "${OUTPUT_DIR}/${name}"
        OUTPUT_VARIABLE toolStdout
        ERROR_VARIABLE toolStderr
        RESULT_VARIABLE toolResult
    )
    if(NOT toolResult EQUAL 0)
        message(FATAL_ERROR "${TOOL} ${cliArgs} ${ARGN} exited with '${toolResult}'\n"
                            "stdout:\n${toolStdout}\nstderr:\n${toolStderr}")
    endif()
    file(READ "${OUTPUT_DIR}/${name}/manifest.json" manifest)
    string(JSON count LENGTH "${manifest}" graphs)
    if(count EQUAL 0)
        message(FATAL_ERROR "corpus ${name} holds no graphs")
    endif()
    set(keys "")
    math(EXPR last "${count} - 1")
    foreach(index RANGE 0 ${last} 1)
        string(JSON key GET "${manifest}" graphs ${index} benchmark)
        list(APPEND keys "${key}")
    endforeach()
    set(${name}_keys "${keys}" PARENT_SCOPE)
endfunction()

if(MODE STREQUAL "exclude")
    corpus(first --seed 1)
    corpus(second --seed 1 --exclude-corpus "${OUTPUT_DIR}/first/manifest.json")
    foreach(key IN LISTS second_keys)
        if(key IN_LIST first_keys)
            message(FATAL_ERROR "graph ${key} is in both corpora though the second excluded "
                                "the first")
        endif()
    endforeach()
elseif(MODE STREQUAL "seed")
    corpus(first --seed 5)
    corpus(again --seed 5)
    corpus(other --seed 6)
    if(NOT first_keys STREQUAL again_keys)
        message(FATAL_ERROR "one seed gave two corpora:\n${first_keys}\n${again_keys}")
    endif()
    if(first_keys STREQUAL other_keys)
        message(FATAL_ERROR "two seeds gave one corpus: ${first_keys}")
    endif()
else()
    message(FATAL_ERROR "unknown MODE '${MODE}'")
endif()
