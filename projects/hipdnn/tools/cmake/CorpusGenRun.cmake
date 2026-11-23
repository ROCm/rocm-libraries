# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Runs hipdnn_corpus_gen into OUTPUT_DIR and checks what a run leaves: its exit status and,
# for a corpus of EXPECTED_GRAPHS graphs, a manifest that says whether an engine was asked
# (--without-engine: no, and warns so) and a commands.txt with one bench invocation per graph
# file it wrote. Given EXPECTED_COMBINATIONS, EXPECTED_MANIFEST, EXPECTED_ANCHORED_PERCENT,
# EXPECTED_FORM_MIX or EXPECTED_FORMS, it also checks what each of them describes (see below).
# CLI_ARGS is comma-separated, as in CliExitStatus.cmake.
if(NOT DEFINED TOOL OR NOT DEFINED CLI_ARGS OR NOT DEFINED OUTPUT_DIR
   OR NOT DEFINED EXPECTED_EXIT OR NOT DEFINED EXPECTED_GRAPHS)
    message(FATAL_ERROR "TOOL, CLI_ARGS, OUTPUT_DIR, EXPECTED_EXIT and EXPECTED_GRAPHS are required")
endif()

file(REMOVE_RECURSE "${OUTPUT_DIR}")
string(REPLACE "," ";" cliArgs "${CLI_ARGS}")
# The operation a report is looked up under: the run's --operation, if it names one.
set(operation "")
if(CLI_ARGS MATCHES "--operation,([^,]+)")
    set(operation "${CMAKE_MATCH_1}")
endif()
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

# With quotas or floors and no --count the corpus is the quotas alone, so `requested` is the
# graphs written plus the fewest more that could fill what is short: a graph that fills a
# regime's floor and a form's (`<axis>:<label>`) is asked once, so the sum of the quotas would
# overstate it. A graph has one regime, so the regimes' shortfalls need graphs of their own, and
# so do the forms'.
if(NOT CLI_ARGS MATCHES "--count," AND NOT operation STREQUAL "")
    string(JSON quotas ERROR_VARIABLE quotaError
           GET "${manifest}" reports regime_quota "${operation}")
    if(NOT quotaError)
        set(regimesShort 0)
        set(formsShort 0)
        string(JSON quotaCount LENGTH "${quotas}")
        if(quotaCount GREATER 0)
            math(EXPR lastQuota "${quotaCount} - 1")
            foreach(index RANGE 0 ${lastQuota} 1)
                string(JSON key MEMBER "${quotas}" ${index})
                string(JSON asked GET "${quotas}" "${key}" asked)
                string(JSON delivered GET "${quotas}" "${key}" delivered)
                math(EXPR short "${asked} - ${delivered}")
                if("${key}" MATCHES ":")
                    math(EXPR formsShort "${formsShort} + ${short}")
                else()
                    math(EXPR regimesShort "${regimesShort} + ${short}")
                endif()
            endforeach()
        endif()
        if(regimesShort GREATER formsShort)
            set(stillShort ${regimesShort})
        else()
            set(stillShort ${formsShort})
        endif()
        math(EXPR expectedRequested "${graphs} + ${stillShort}")
        string(JSON requested GET "${manifest}" requested)
        if(NOT requested EQUAL expectedRequested)
            message(FATAL_ERROR "the manifest requests ${requested}, expected ${expectedRequested}: "
                                "${graphs} graphs and at least ${stillShort} more for the quotas "
                                "still short")
        endif()
    endif()
endif()

# EXPECTED_COMBINATIONS (optional, `<total>:<constraint empty>:<not explored>`): the manifest
# reports the categorical combinations the --operation's search held, how many of them the
# declared constraints leave empty and how many the search's bound left unexplored, under
# reports.categorical_combinations.
if(EXPECTED_COMBINATIONS)
    if(operation STREQUAL "")
        message(FATAL_ERROR "EXPECTED_COMBINATIONS needs an --operation to look its report up under")
    endif()
    string(REPLACE ":" ";" expectedCombinations "${EXPECTED_COMBINATIONS}")
    list(GET expectedCombinations 0 expectedTotal)
    list(GET expectedCombinations 1 expectedConstraintEmpty)
    list(GET expectedCombinations 2 expectedNotExplored)
    foreach(key IN ITEMS total constraint_empty not_explored)
        string(JSON reported ERROR_VARIABLE combinationError
               GET "${manifest}" reports categorical_combinations "${operation}" "${key}")
        if(combinationError)
            message(FATAL_ERROR "reports.categorical_combinations.${operation}.${key} is "
                                "missing: ${combinationError}")
        endif()
        set(reportedFor_${key} "${reported}")
    endforeach()
    if(NOT reportedFor_total EQUAL expectedTotal
       OR NOT reportedFor_constraint_empty EQUAL expectedConstraintEmpty
       OR NOT reportedFor_not_explored EQUAL expectedNotExplored)
        message(FATAL_ERROR "the manifest reports ${reportedFor_total} categorical combinations, "
                            "${reportedFor_constraint_empty} empty under the constraints and "
                            "${reportedFor_not_explored} not explored, expected ${expectedTotal}, "
                            "${expectedConstraintEmpty} and ${expectedNotExplored}")
    endif()
endif()

# EXPECTED_MANIFEST (optional, comma-separated `<path><op><value>`): each dotted path into
# manifest.json (an array index is a number) holds the value: `=` for equality, as CMake
# prints it (a JSON true is ON), `>=` and `<=` for integers.
if(EXPECTED_MANIFEST)
    string(REPLACE "," ";" expectedManifest "${EXPECTED_MANIFEST}")
    foreach(check IN LISTS expectedManifest)
        if(NOT check MATCHES "^([^=<>]+)(=|>=|<=)(.*)$")
            message(FATAL_ERROR "EXPECTED_MANIFEST check '${check}' is not <path>(=|>=|<=)<value>")
        endif()
        string(REPLACE "." ";" checkPath "${CMAKE_MATCH_1}")
        set(checkOperator "${CMAKE_MATCH_2}")
        set(checkValue "${CMAKE_MATCH_3}")
        string(JSON reported ERROR_VARIABLE checkError GET "${manifest}" ${checkPath})
        if(checkError)
            message(FATAL_ERROR "manifest.json has no ${CMAKE_MATCH_1}: ${checkError}")
        endif()
        if(checkOperator STREQUAL "=" AND NOT "${reported}" STREQUAL "${checkValue}")
            message(FATAL_ERROR "manifest.json has ${reported} at ${CMAKE_MATCH_1}, expected "
                                "${checkValue}")
        elseif(checkOperator STREQUAL ">=" AND NOT reported GREATER_EQUAL checkValue)
            message(FATAL_ERROR "manifest.json has ${reported} at ${CMAKE_MATCH_1}, expected at "
                                "least ${checkValue}")
        elseif(checkOperator STREQUAL "<=" AND NOT reported LESS_EQUAL checkValue)
            message(FATAL_ERROR "manifest.json has ${reported} at ${CMAKE_MATCH_1}, expected at "
                                "most ${checkValue}")
        endif()
    endforeach()
endif()

# EXPECTED_ANCHORED_PERCENT (optional, `>=<n>` or `<=<n>`): the share of the --operation's sweep
# graphs in the corpus that came from archetypes or their neighbourhood, as
# reports.anchored_share gives it in whole counts, in percent.
if(EXPECTED_ANCHORED_PERCENT)
    if(operation STREQUAL "")
        message(FATAL_ERROR "EXPECTED_ANCHORED_PERCENT needs an --operation to look its report up under")
    endif()
    if(NOT EXPECTED_ANCHORED_PERCENT MATCHES "^(>=|<=)([0-9]+)$")
        message(FATAL_ERROR "EXPECTED_ANCHORED_PERCENT '${EXPECTED_ANCHORED_PERCENT}' is not >=<n> or <=<n>")
    endif()
    set(anchoredOperator "${CMAKE_MATCH_1}")
    set(anchoredPercent "${CMAKE_MATCH_2}")
    string(JSON anchored GET "${manifest}" reports anchored_share "${operation}" anchored)
    string(JSON sweepDelivered GET "${manifest}" reports anchored_share "${operation}" delivered)
    if(sweepDelivered GREATER graphs OR anchored GREATER sweepDelivered)
        message(FATAL_ERROR "anchored_share counts ${anchored} anchored of ${sweepDelivered} "
                            "delivered, of ${graphs} graphs")
    endif()
    math(EXPR anchoredScaled "${anchored} * 100")
    math(EXPR wantedScaled "${anchoredPercent} * ${sweepDelivered}")
    if(anchoredOperator STREQUAL ">=" AND anchoredScaled LESS wantedScaled)
        message(FATAL_ERROR "${anchored} of ${sweepDelivered} delivered sweep graphs are anchored, "
                            "below ${anchoredPercent}%")
    elseif(anchoredOperator STREQUAL "<=" AND anchoredScaled GREATER wantedScaled)
        message(FATAL_ERROR "${anchored} of ${sweepDelivered} delivered sweep graphs are anchored, "
                            "above ${anchoredPercent}%")
    endif()
endif()

# The corpus's `form` column of manifest.csv, for the form checks below. The columns up to `form`
# are generated labels, never quoted, so a plain split finds it: it is right after `grouping`.
if(EXPECTED_FORMS OR EXPECTED_FORM_MIX)
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
endif()

# EXPECTED_FORM_MIX (optional, `<percent>:<label>:<label>...`): each label holds at least that
# percent of the corpus's graphs, so that a cut that leaves a form short, because fewer
# combinations can hold it, is caught: an even mix of n forms is 100 / n percent each.
if(EXPECTED_FORM_MIX)
    string(REPLACE ":" ";" formMix "${EXPECTED_FORM_MIX}")
    list(POP_FRONT formMix mixPercent)
    list(LENGTH forms formRows)
    foreach(form IN LISTS formMix)
        set(held ${forms})
        list(FILTER held INCLUDE REGEX "^${form}$")
        list(LENGTH held heldRows)
        math(EXPR heldScaled "${heldRows} * 100")
        math(EXPR neededScaled "${mixPercent} * ${formRows}")
        if(heldScaled LESS neededScaled)
            message(FATAL_ERROR "the form ${form} holds ${heldRows} of ${formRows} graphs, below "
                                "${mixPercent}%")
        endif()
    endforeach()
endif()

# EXPECTED_FORMS (optional, comma-separated labels): the corpus carries a `form` facet column of
# its own right after `grouping` in manifest.csv, holding exactly these labels, and the run
# reports each as a form quota (`form:<label>`, in reports.regime_quota of the --operation)
# that delivered at least one graph.
if(NOT EXPECTED_FORMS)
    return()
endif()
string(REPLACE "," ";" expectedForms "${EXPECTED_FORMS}")

list(REMOVE_DUPLICATES forms)
list(SORT forms)
list(SORT expectedForms)
if(NOT "${forms}" STREQUAL "${expectedForms}")
    message(FATAL_ERROR "manifest.csv holds the forms '${forms}', expected '${expectedForms}'")
endif()

if(operation STREQUAL "")
    message(FATAL_ERROR "EXPECTED_FORMS needs an --operation to look its form quotas up under")
endif()
foreach(form IN LISTS expectedForms)
    string(JSON delivered ERROR_VARIABLE quotaError
           GET "${manifest}" reports regime_quota "${operation}" "form:${form}" delivered)
    if(quotaError OR delivered LESS 1)
        message(FATAL_ERROR "reports.regime_quota.${operation} does not show form:${form} "
                            "delivering a graph: '${delivered}' ${quotaError}")
    endif()
endforeach()
