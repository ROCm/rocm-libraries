# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)

foreach(_required_var NM CONSUMER PROVIDER REQUIRED_IMPORTS NAMESPACE_PATTERN WORK_DIR)
    if(NOT DEFINED ${_required_var})
        message(FATAL_ERROR "hipblaslt_check_symbol_contract: ${_required_var} is not set")
    endif()
endforeach()

file(MAKE_DIRECTORY "${WORK_DIR}")

function(_read_interface library out_defined out_undefined)
    get_filename_component(_name "${library}" NAME)
    set(_symbols "${WORK_DIR}/${_name}.symbols")
    execute_process(
        COMMAND "${NM}" --dynamic --extern-only --format=posix --no-demangle "${library}"
        RESULT_VARIABLE _rc
        OUTPUT_FILE "${_symbols}"
        ERROR_VARIABLE _err
    )
    if(NOT _rc STREQUAL "0")
        message(FATAL_ERROR "nm failed on ${library}: ${_err}")
    endif()

    file(STRINGS "${_symbols}" _lines)
    if(NOT _lines)
        message(FATAL_ERROR "nm returned no dynamic symbols for ${library}")
    endif()
    set(_defined "")
    set(_undefined "")
    foreach(_line IN LISTS _lines)
        # GNU and LLVM nm share the POSIX name/type/value/size format.
        if(NOT _line MATCHES "^([^ \t]+)[ \t]+([A-Za-z])([ \t]+[0-9a-fA-F]+)?([ \t]+[0-9a-fA-F]+)?[ \t]*$")
            message(FATAL_ERROR "Unexpected nm output for ${library}: ${_line}")
        endif()
        set(_symbol "${CMAKE_MATCH_1}")
        set(_type "${CMAKE_MATCH_2}")
        # Compare linker symbol names, allowing GNU symbol-version suffixes.
        string(REGEX REPLACE "@.*$" "" _symbol "${_symbol}")
        if(_type STREQUAL "U")
            list(APPEND _undefined "${_symbol}")
        elseif(NOT _type MATCHES "^[wv]$")
            # Lowercase w/v are optional weak imports; uppercase W/V are definitions.
            list(APPEND _defined "${_symbol}")
        endif()
    endforeach()
    set(${out_defined} "${_defined}" PARENT_SCOPE)
    set(${out_undefined} "${_undefined}" PARENT_SCOPE)
endfunction()

_read_interface("${CONSUMER}" _consumer_defined _consumer_imports)
_read_interface("${PROVIDER}" _provider_exports _provider_imports)

file(STRINGS "${REQUIRED_IMPORTS}" _required_imports REGEX "[^ \t]")
if(NOT _required_imports)
    message(FATAL_ERROR "Symbol contract has no required imports: ${REQUIRED_IMPORTS}")
endif()

set(_violations "")

foreach(_symbol IN LISTS _required_imports)
    string(STRIP "${_symbol}" _symbol)
    if(_symbol IN_LIST _consumer_defined)
        list(APPEND _violations "required import also defined by ${CONSUMER}: ${_symbol}")
    endif()
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
