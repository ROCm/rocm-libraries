# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Reports the functions that host_selection_order.txt lists and the host
# library does not define. The link places the listed functions together and
# passes over a name the library does not define, which leaves the code that
# replaced it elsewhere: the selection still runs as before, from more pages of
# code. The report is advisory. Lambda and internal names change with
# unrelated edits and with the toolchain, so a missing name makes the
# TensileLite.HostSelectionOrder test report itself skipped, with the names,
# rather than fail.
#   cmake -DNM=<nm> -DLIBRARY=<host library> -DORDER=<list> -P check_host_selection_order.cmake
foreach(_var NM LIBRARY ORDER)
    if(NOT DEFINED ${_var})
        message(FATAL_ERROR "check_host_selection_order: ${_var} is not set")
    endif()
endforeach()

execute_process(
    COMMAND "${NM}" --defined-only "${LIBRARY}"
    OUTPUT_VARIABLE _symbols
    ERROR_VARIABLE _error
    RESULT_VARIABLE _result)
if(NOT _result EQUAL 0)
    message(STATUS "check_host_selection_order: skipped: ${NM} failed on ${LIBRARY}: ${_error}")
    return()
endif()
# Each line of nm ends with the name after a blank: a name is defined when
# " <name>\n" occurs.
string(APPEND _symbols "\n")

file(STRINGS "${ORDER}" _lines)
set(_listed 0)
set(_missing "")
foreach(_line IN LISTS _lines)
    string(STRIP "${_line}" _name)
    if(_name STREQUAL "" OR _name MATCHES "^#")
        continue()
    endif()
    math(EXPR _listed "${_listed} + 1")
    string(FIND "${_symbols}" " ${_name}\n" _at)
    if(_at EQUAL -1)
        string(APPEND _missing "\n  ${_name}")
    endif()
endforeach()

if(_listed EQUAL 0)
    message(FATAL_ERROR "check_host_selection_order: ${ORDER} lists no function")
endif()
if(NOT _missing STREQUAL "")
    message(STATUS "check_host_selection_order: skipped: ${LIBRARY} does not define these "
                   "functions of ${ORDER}, which the link then leaves where they are; list the "
                   "functions a first selection runs again:${_missing}")
    return()
endif()
message(STATUS "check_host_selection_order: the ${_listed} functions listed are defined")
