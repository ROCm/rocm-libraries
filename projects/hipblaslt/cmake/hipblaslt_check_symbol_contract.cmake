# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)

foreach(_required_var LLVM_IFS CONSUMER PROVIDER REQUIRED_IMPORTS NAMESPACE_PATTERN WORK_DIR)
    if(NOT DEFINED ${_required_var})
        message(FATAL_ERROR "hipblaslt_check_symbol_contract: ${_required_var} is not set")
    endif()
endforeach()

file(MAKE_DIRECTORY "${WORK_DIR}")

function(_read_interface library out_defined out_undefined)
    get_filename_component(_name "${library}" NAME)
    set(_ifs "${WORK_DIR}/${_name}.ifs")
    execute_process(
        COMMAND "${LLVM_IFS}" --input-format=ELF "--output-ifs=${_ifs}" "${library}"
        RESULT_VARIABLE _rc
        ERROR_VARIABLE _err
    )
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "llvm-ifs failed on ${library}: ${_err}")
    endif()

    file(STRINGS "${_ifs}" _lines REGEX "^  - \\{ Name: ")
    set(_defined "")
    set(_undefined "")
    foreach(_line IN LISTS _lines)
        string(REGEX MATCH "Name: ([^, }]+)" _ "${_line}")
        set(_symbol "${CMAKE_MATCH_1}")
        if(_line MATCHES "Undefined: true")
            if(NOT _line MATCHES "Weak: true")
                list(APPEND _undefined "${_symbol}")
            endif()
        else()
            list(APPEND _defined "${_symbol}")
        endif()
    endforeach()
    set(${out_defined} "${_defined}" PARENT_SCOPE)
    set(${out_undefined} "${_undefined}" PARENT_SCOPE)
endfunction()

_read_interface("${CONSUMER}" _consumer_defined _consumer_imports)
_read_interface("${PROVIDER}" _provider_exports _provider_imports)

file(STRINGS "${REQUIRED_IMPORTS}" _required_imports REGEX "[^ \t]")

set(_violations "")

foreach(_symbol IN LISTS _required_imports)
    string(STRIP "${_symbol}" _symbol)
    if(NOT _symbol IN_LIST _consumer_imports)
        list(APPEND _violations "required import missing from ${CONSUMER}: ${_symbol}")
    endif()
    if(NOT _symbol IN_LIST _provider_exports)
        list(APPEND _violations "required import not exported by ${PROVIDER}: ${_symbol}")
    endif()
endforeach()

foreach(_symbol IN LISTS _consumer_imports)
    if(_symbol MATCHES "${NAMESPACE_PATTERN}" AND NOT _symbol IN_LIST _provider_exports)
        list(APPEND _violations "import not exported by ${PROVIDER}: ${_symbol}")
    endif()
endforeach()

if(_violations)
    list(JOIN _violations "\n  " _report)
    message(FATAL_ERROR "Symbol contract violated:\n  ${_report}")
endif()

list(LENGTH _required_imports _required_count)
message(STATUS "Symbol contract satisfied (${_required_count} required imports)")
