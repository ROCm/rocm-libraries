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
# the comparison reaches the memo. Without a materialized shipped model the
# entry is skipped. Each run's output is kept in OUT.
#
#   cmake -DWEIGHTS=DIR -DKERNELS=FILE -DSHAPES=FILE -DOUT=DIR -DBUILDS=a,b,...
#         -DMEMO_BUILDS=b,... -Drerank_check_<build>=EXE -Dmemo_check_<build>=EXE ...
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
list(LENGTH _models _n_models)
list(LENGTH _builds _n_builds)
if(_failures GREATER 0)
    message(FATAL_ERROR "levers parity: ${_failures} failure(s)")
endif()
message(
    STATUS
        "levers parity: ${_n_models} models, ${_n_shapes} shapes, ${_n_kernels} kernels: "
        "rerank_check and memo_check hash alike in ${_n_builds} builds (${BUILDS})"
)
