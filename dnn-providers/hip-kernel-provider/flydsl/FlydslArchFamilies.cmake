# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# The members of FlyDSL's LLVM generic targets, read from generators/arch_families.json,
# the table the descriptor generator reads, so CMake and the generator cannot disagree.
# Included by the provider's top-level CMakeLists.txt and by flydsl/CMakeLists.txt.

include_guard(GLOBAL)

set(_flydsl_families_json "${CMAKE_CURRENT_LIST_DIR}/generators/arch_families.json")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_flydsl_families_json}")
file(READ "${_flydsl_families_json}" _flydsl_families)

# The architectures a content target stands for: itself, or every member of an
# LLVM generic target.
function(_flydsl_target_members target entry out)
    if(NOT target MATCHES "-generic$")
        set(${out} "${target}" PARENT_SCOPE)
        return()
    endif()
    string(JSON _count ERROR_VARIABLE _flydsl_json_error
           LENGTH "${_flydsl_families}" "${target}" members)
    if(_flydsl_json_error)
        message(FATAL_ERROR
            "flydsl: content '${entry}' names generic target '${target}', which "
            "generators/arch_families.json does not list")
    endif()
    set(_members "")
    math(EXPR _last "${_count} - 1")
    # cmake-lint: disable=E1120
    foreach(_index RANGE ${_last})
        string(JSON _member GET "${_flydsl_families}" "${target}" members ${_index})
        list(APPEND _members "${_member}")
    endforeach()
    set(${out} "${_members}" PARENT_SCOPE)
endfunction()
