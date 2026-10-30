# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# The ctest entry tilewright-levers-parity (tests/CMakeLists.txt). rerank_check
# and memo_check (training_pipeline/scripts/experiments), built against the
# engine with each combination of the selection-time options, rank every
# shipped model under WEIGHTS with the sample kernel pool (KERNELS) on the
# benches' default shapes (SHAPES), as bf16 problems with A transposed on the
# hardware the real-model tests use. Every build must print the hashes the
# first build of BUILDS prints, and in each build of MEMO_BUILDS the re-rank
# memo must answer TensileLite's re-rank (rerank_check) and scenario S06 of
# memo_check, the same pair of rankings, for some shape of some model, so that
# the comparison reaches the memo. With DRIFT_FLAG (tests/drift_flag.cpp), a
# copy of every model the drift fallback can serve, served with it at the
# largest share, is ranked as well, except the models DRIFT_SKIP names (file
# names without .tilewright.bin); the sample pool degrades most of their
# cells. With VETO_MODELS (tests/veto_models.cpp), copies of the models the
# alignment veto can serve, the drift fallback's copies among them, each
# without the veto and with the veto of the gfx950 recipes (K above 512,
# 128-byte lines) whether or not it ships with one, rank bf16 problems of the
# two layouts the veto applies to (A and B transposed, and neither) with the
# sample kernels and their siblings whose cache hint on A or on B is the
# non-temporal one, and the same holds of the copies with the veto: equal
# hashes in every build, memo answers in each memo build, and, so that the
# comparison reaches the veto, hashes of some copy that differ from those of
# the copy without it in the first build.
# Without a materialized shipped model the entry is skipped. Each run's output
# is kept in OUT.
#
#   cmake -DWEIGHTS=DIR -DKERNELS=FILE -DSHAPES=FILE -DOUT=DIR -DBUILDS=a,b,...
#         -DMEMO_BUILDS=b,... -Drerank_check_<build>=EXE -Dmemo_check_<build>=EXE ...
#         [-DDRIFT_FLAG=EXE [-DDRIFT_SKIP=name,...]] [-DVETO_MODELS=EXE]
#         -P levers_parity.cmake

cmake_minimum_required(VERSION 3.25.2)

foreach(_var WEIGHTS KERNELS SHAPES OUT BUILDS MEMO_BUILDS)
    if(NOT DEFINED ${_var})
        message(FATAL_ERROR "levers_parity.cmake: -D${_var}=... is missing")
    endif()
endforeach()
string(REPLACE "," ";" _builds "${BUILDS}")
string(REPLACE "," ";" _memo_builds "${MEMO_BUILDS}")
list(GET _builds 0 _reference)
foreach(_build IN LISTS _builds)
    foreach(_check rerank_check memo_check)
        if(NOT EXISTS "${${_check}_${_build}}")
            message(FATAL_ERROR "levers_parity.cmake: -D${_check}_${_build}=... names no file")
        endif()
    endforeach()
endforeach()
foreach(_build IN LISTS _memo_builds)
    if(NOT _build IN_LIST _builds)
        message(FATAL_ERROR "levers_parity.cmake: memo build ${_build} is not in BUILDS")
    endif()
endforeach()

# Every *.tilewright.bin under WEIGHTS but Git LFS pointers.
string(HEX "version https://git-lfs.github.com/spec/" _lfs)
set(_models)
if(IS_DIRECTORY "${WEIGHTS}")
    file(GLOB_RECURSE _found LIST_DIRECTORIES false "${WEIGHTS}/*.tilewright.bin")
    list(SORT _found)
    foreach(_path IN LISTS _found)
        file(READ "${_path}" _head LIMIT 40 HEX)
        if(_head STREQUAL _lfs)
            message(WARNING "${_path} is a Git LFS pointer; skipped")
        else()
            list(APPEND _models "${_path}")
        endif()
    endforeach()
endif()
if(NOT _models)
    message(STATUS "no materialized shipped models under ${WEIGHTS}")
    return()
endif()

set(_n_flagged 0)
if(DEFINED DRIFT_FLAG)
    string(REPLACE "," ";" _drift_skip "${DRIFT_SKIP}")
    file(MAKE_DIRECTORY "${OUT}/drift")
    set(_flagged)
    foreach(_model IN LISTS _models)
        get_filename_component(_name "${_model}" NAME)
        string(REPLACE ".tilewright.bin" "" _stem "${_name}")
        if(_stem IN_LIST _drift_skip)
            continue()
        endif()
        set(_copy "${OUT}/drift/${_stem}.drift.tilewright.bin")
        execute_process(
            COMMAND "${DRIFT_FLAG}" "${_model}" "${_copy}" 1000000
            ERROR_VARIABLE _err
            RESULT_VARIABLE _rc
        )
        if(_rc EQUAL 0)
            list(APPEND _flagged "${_copy}")
        elseif(NOT _rc EQUAL 3)
            message(FATAL_ERROR "drift_flag on ${_name}: exit ${_rc}\n${_err}")
        endif()
    endforeach()
    list(LENGTH _flagged _n_flagged)
    list(APPEND _models ${_flagged})
endif()

# The inputs (bench_inputs.hpp): DataType::BFloat16 (7) for every type,
# Transpose::T (0) for A and Transpose::N (1) for B, as the sample kernels
# are; the hardware of test_real_models.cpp.
file(STRINGS "${KERNELS}" _kernels REGEX "^[0-9]")
file(STRINGS "${SHAPES}" _shapes REGEX "^[0-9]")
list(LENGTH _kernels _n_kernels)
list(LENGTH _shapes _n_shapes)
string(JOIN "\n" _kernel_lines ${_kernels})
string(JOIN "\n" _shape_lines ${_shapes})
file(MAKE_DIRECTORY "${OUT}")
set(_inputs "${OUT}/inputs.txt")
file(WRITE "${_inputs}"
     "PROBLEM 7 7 7 7 7 0 1\nHARDWARE 128 65536 4194304\nSHAPES ${_n_shapes}\n${_shape_lines}\n"
     "CONFIGS ${_n_kernels}\n${_kernel_lines}\n"
)

set(_failures 0)
foreach(_build IN LISTS _memo_builds)
    foreach(_check rerank_check memo_check)
        set(_answered_${_check}_${_build} 0)
    endforeach()
endforeach()
foreach(_model IN LISTS _models)
    get_filename_component(_name "${_model}" NAME)
    string(REPLACE ".tilewright.bin" "" _stem "${_name}")
    set(_summary "${_stem}:")
    foreach(_check rerank_check memo_check)
        string(APPEND _summary " ${_check}")
        foreach(_build IN LISTS _builds)
            execute_process(
                COMMAND "${${_check}_${_build}}" "${_model}" "${_inputs}"
                OUTPUT_VARIABLE _out
                ERROR_VARIABLE _err
                RESULT_VARIABLE _rc
            )
            file(WRITE "${OUT}/${_stem}.${_check}.${_build}.txt" "${_out}")
            if(NOT _rc EQUAL 0)
                message(FATAL_ERROR "${_check} (${_build}) on ${_name}: exit ${_rc}\n${_err}")
            endif()
            # The hashes: the output without its timings and the memo's counts.
            string(REGEX REPLACE "SECOND_P50_US [^\n]*\n" "" _hashes "${_out}")
            string(REGEX REPLACE " second_p50_us [^ \n]*" "" _hashes "${_hashes}")
            string(REGEX MATCH "# memo replays[^\n]*" _replays "${_hashes}")
            string(REGEX REPLACE "# memo replays[^\n]*\n" "" _hashes "${_hashes}")
            if(_build STREQUAL _reference)
                set(_want "${_hashes}")
            elseif(NOT _hashes STREQUAL _want)
                message(
                    SEND_ERROR
                        "${_check} on ${_name}: the ${_build} build's hashes differ from the "
                        "${_reference} build's\n${_reference}:\n${_want}\n${_build}:\n${_hashes}"
                )
                math(EXPR _failures "${_failures} + 1")
            endif()
            if(NOT _build IN_LIST _memo_builds)
                if(_replays)
                    message(SEND_ERROR "${_check} (${_build}) counts memo answers without the memo")
                    math(EXPR _failures "${_failures} + 1")
                endif()
                continue()
            endif()
            if(_check STREQUAL "rerank_check")
                string(REGEX MATCH "^# memo replays ([0-9]+)" _match "${_replays}")
            else()
                string(REGEX MATCH " S06 ([0-9]+)" _match "${_replays}")
            endif()
            if(NOT _match)
                message(FATAL_ERROR "${_check} (${_build}) on ${_name} printed no memo count:\n${_out}")
            endif()
            math(EXPR _answered_${_check}_${_build}
                 "${_answered_${_check}_${_build}} + ${CMAKE_MATCH_1}"
            )
            string(APPEND _summary " ${_build} ${CMAKE_MATCH_1}")
        endforeach()
    endforeach()
    message(STATUS "${_summary} (re-ranks the memo answered, of ${_n_shapes})")
endforeach()

foreach(_build IN LISTS _memo_builds)
    foreach(_check rerank_check memo_check)
        if(_answered_${_check}_${_build} EQUAL 0)
            message(
                SEND_ERROR
                    "${_check} (${_build}): the memo answered no re-rank of any model, so the "
                    "comparison does not reach it"
            )
            math(EXPR _failures "${_failures} + 1")
        endif()
    endforeach()
endforeach()
# The hashes of one tool's output: without its timings and the memo's counts.
function(_tw_hashes out text)
    string(REGEX REPLACE "SECOND_P50_US [^\n]*\n" "" _h "${text}")
    string(REGEX REPLACE " second_p50_us [^ \n]*" "" _h "${_h}")
    string(REGEX REPLACE "# memo replays[^\n]*\n" "" _h "${_h}")
    set(${out} "${_h}" PARENT_SCOPE)
endfunction()

set(_veto_summary "")
if(DEFINED VETO_MODELS)
    execute_process(
        COMMAND "${VETO_MODELS}" "${OUT}/veto" 512 1024 ${_models}
        OUTPUT_VARIABLE _copies
        ERROR_VARIABLE _err
        RESULT_VARIABLE _rc
    )
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "the alignment veto's copies: exit ${_rc}\n${_err}")
    endif()
    # The sample kernels, then each with the non-temporal hint on A, then on B.
    set(_hinted_lines)
    foreach(_hints "a" "b")
        foreach(_line IN LISTS _kernels)
            separate_arguments(_fields UNIX_COMMAND "${_line}")
            list(REMOVE_AT _fields 7 8)
            if(_hints STREQUAL "a")
                list(INSERT _fields 7 4 0)
            else()
                list(INSERT _fields 7 0 4)
            endif()
            string(JOIN " " _joined ${_fields})
            list(APPEND _hinted_lines "${_joined}")
        endforeach()
    endforeach()
    list(LENGTH _hinted_lines _n_hinted)
    math(EXPR _n_veto_kernels "${_n_kernels} + ${_n_hinted}")
    string(JOIN "\n" _hinted_text ${_hinted_lines})
    foreach(_layout tt nn)
        if(_layout STREQUAL "tt")
            set(_transposes "0 0")
        else()
            set(_transposes "1 1")
        endif()
        file(WRITE "${OUT}/inputs_veto_${_layout}.txt"
             "PROBLEM 7 7 7 7 7 ${_transposes}\nHARDWARE 128 65536 4194304\n"
             "SHAPES ${_n_shapes}\n${_shape_lines}\nCONFIGS ${_n_veto_kernels}\n"
             "${_kernel_lines}\n${_hinted_text}\n"
        )
    endforeach()
    foreach(_build IN LISTS _memo_builds)
        foreach(_check rerank_check memo_check)
            set(_veto_answered_${_check}_${_build} 0)
        endforeach()
    endforeach()
    string(REGEX MATCHALL "copy [^\n]+" _pairs "${_copies}")
    list(LENGTH _pairs _n_copies)
    set(_veto_changed 0)
    foreach(_pair IN LISTS _pairs)
        separate_arguments(_pp UNIX_COMMAND "${_pair}")
        list(GET _pp 1 _original)
        list(GET _pp 2 _copy)
        get_filename_component(_name "${_copy}" NAME)
        string(REPLACE ".tilewright.bin" "" _stem "${_name}")
        set(_summary "${_stem} with the veto:")
        foreach(_layout tt nn)
            set(_veto_inputs "${OUT}/inputs_veto_${_layout}.txt")
            foreach(_check rerank_check memo_check)
                string(APPEND _summary " ${_layout} ${_check}")
                foreach(_build IN LISTS _builds)
                    execute_process(
                        COMMAND "${${_check}_${_build}}" "${_copy}" "${_veto_inputs}"
                        OUTPUT_VARIABLE _out
                        ERROR_VARIABLE _err
                        RESULT_VARIABLE _rc
                    )
                    file(WRITE "${OUT}/${_stem}.veto.${_layout}.${_check}.${_build}.txt" "${_out}")
                    if(NOT _rc EQUAL 0)
                        message(FATAL_ERROR "${_check} (${_build}) on ${_name} with the veto, "
                                            "${_layout}: exit ${_rc}\n${_err}"
                        )
                    endif()
                    _tw_hashes(_hashes "${_out}")
                    string(REGEX MATCH "# memo replays[^\n]*" _replays "${_out}")
                    if(_build STREQUAL _reference)
                        set(_want "${_hashes}")
                        # The copy without the veto, in the first build.
                        execute_process(
                            COMMAND "${${_check}_${_build}}" "${_original}" "${_veto_inputs}"
                            OUTPUT_VARIABLE _plain
                            RESULT_VARIABLE _rc
                        )
                        if(NOT _rc EQUAL 0)
                            message(FATAL_ERROR "${_check} (${_build}) on ${_original}, ${_layout}: "
                                                "exit ${_rc}"
                            )
                        endif()
                        _tw_hashes(_plain_hashes "${_plain}")
                        if(NOT _plain_hashes STREQUAL _want)
                            math(EXPR _veto_changed "${_veto_changed} + 1")
                        endif()
                    elseif(NOT _hashes STREQUAL _want)
                        message(
                            SEND_ERROR
                                "${_check} on ${_name} with the veto (${_layout}): the ${_build} "
                                "build's hashes differ from the ${_reference} build's\n"
                                "${_reference}:\n${_want}\n${_build}:\n${_hashes}"
                        )
                        math(EXPR _failures "${_failures} + 1")
                    endif()
                    if(NOT _build IN_LIST _memo_builds)
                        if(_replays)
                            message(SEND_ERROR "${_check} (${_build}) counts memo answers without the memo")
                            math(EXPR _failures "${_failures} + 1")
                        endif()
                        continue()
                    endif()
                    if(_check STREQUAL "rerank_check")
                        string(REGEX MATCH "^# memo replays ([0-9]+)" _match "${_replays}")
                    else()
                        string(REGEX MATCH " S06 ([0-9]+)" _match "${_replays}")
                    endif()
                    if(NOT _match)
                        message(FATAL_ERROR "${_check} (${_build}) on ${_name} printed no memo count")
                    endif()
                    math(EXPR _veto_answered_${_check}_${_build}
                         "${_veto_answered_${_check}_${_build}} + ${CMAKE_MATCH_1}"
                    )
                    string(APPEND _summary " ${_build} ${CMAKE_MATCH_1}")
                endforeach()
            endforeach()
        endforeach()
        message(STATUS "${_summary} (re-ranks the memo answered, of ${_n_shapes})")
    endforeach()
    if(_n_copies EQUAL 0)
        message(SEND_ERROR "no shipped model can be served with the alignment veto")
        math(EXPR _failures "${_failures} + 1")
    endif()
    if(_veto_changed EQUAL 0)
        message(
            SEND_ERROR
                "the alignment veto changed no hash of any copy, so the comparison does not reach it"
        )
        math(EXPR _failures "${_failures} + 1")
    endif()
    foreach(_build IN LISTS _memo_builds)
        foreach(_check rerank_check memo_check)
            if(_veto_answered_${_check}_${_build} EQUAL 0)
                message(
                    SEND_ERROR
                        "${_check} (${_build}): the memo answered no re-rank of any copy with the "
                        "alignment veto, so the comparison does not reach it"
                )
                math(EXPR _failures "${_failures} + 1")
            endif()
        endforeach()
    endforeach()
    string(CONCAT _veto_summary
                  ", and so do ${_n_copies} copies with the alignment veto on TT and NN "
                  "problems (the veto changed the hashes of ${_veto_changed} runs)"
    )
endif()

list(LENGTH _models _n_models)
list(LENGTH _builds _n_builds)
if(_failures GREATER 0)
    message(FATAL_ERROR "levers parity: ${_failures} failure(s)")
endif()
message(
    STATUS
        "levers parity: ${_n_models} models (${_n_flagged} served with the drift fallback), "
        "${_n_shapes} shapes, ${_n_kernels} kernels: rerank_check and memo_check hash alike in "
        "${_n_builds} builds (${BUILDS})${_veto_summary}"
)
