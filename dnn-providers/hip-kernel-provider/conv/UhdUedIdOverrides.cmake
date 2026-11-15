# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# UHD training branch only. The UHD pipeline keys an engine's L2 dataset by its UED id, and
# refuses new measurements of an engine whose descriptors changed under an unchanged id
# (dataset_binding_mismatch). uhd_ued_id_overrides.json gives such engines a new id here.
#
# The id is rewritten in a production source root that is a copy (the pipeline packs one,
# HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT), in the UED and in every descriptor naming it (the
# KDPs' engine field). The KDPs are DVC outputs, so the in-tree descriptors are never edited:
# a build packing the in-tree root keeps the original ids. A UUID replaces a UUID, so every
# catalog keeps the size its DVC pointer records (on Linux: file(WRITE) on Windows writes
# CRLF line endings).

if(NOT HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT)
    return()
endif()
file(REAL_PATH "${HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT}" _uhd_ids_root)
file(REAL_PATH "${HIPKERNELPROVIDER_PRODUCTION_DESCRIPTOR_SOURCE_ROOT}" _uhd_ids_in_tree)
if(_uhd_ids_root STREQUAL _uhd_ids_in_tree OR NOT IS_DIRECTORY "${_uhd_ids_root}/conv")
    message(STATUS "conv: UED id overrides not applied (the production root is the source tree)")
    return()
endif()

file(READ "${CMAKE_CURRENT_LIST_DIR}/uhd_ued_id_overrides.json" _uhd_ids_json)
string(JSON _uhd_ids_overrides GET "${_uhd_ids_json}" overrides)
string(JSON _uhd_ids_count LENGTH "${_uhd_ids_overrides}")
set(_uhd_ids_old "")
set(_uhd_ids_new "")
if(_uhd_ids_count GREATER 0)
    math(EXPR _uhd_ids_last "${_uhd_ids_count} - 1")
    foreach(_index RANGE ${_uhd_ids_last})
        string(JSON _old MEMBER "${_uhd_ids_overrides}" ${_index})
        string(JSON _new GET "${_uhd_ids_overrides}" "${_old}" id)
        list(APPEND _uhd_ids_old "${_old}")
        list(APPEND _uhd_ids_new "${_new}")
    endforeach()
endif()

file(GLOB_RECURSE _uhd_ids_docs "${_uhd_ids_root}/conv/*.ued.json"
     "${_uhd_ids_root}/conv/*.kdp.json")
set(_uhd_ids_rewritten 0)
foreach(_doc IN LISTS _uhd_ids_docs)
    file(READ "${_doc}" _text)
    set(_before "${_text}")
    foreach(_old _new IN ZIP_LISTS _uhd_ids_old _uhd_ids_new)
        string(REPLACE "${_old}" "${_new}" _text "${_text}")
    endforeach()
    if(NOT _text STREQUAL _before)
        file(WRITE "${_doc}" "${_text}")
        math(EXPR _uhd_ids_rewritten "${_uhd_ids_rewritten} + 1")
    endif()
endforeach()
message(STATUS "conv: UED id overrides: ${_uhd_ids_count} engine(s), "
               "${_uhd_ids_rewritten} descriptor(s) rewritten in ${_uhd_ids_root}/conv")
