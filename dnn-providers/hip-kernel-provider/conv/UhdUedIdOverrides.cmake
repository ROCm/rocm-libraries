# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# UHD training branch only. The UHD pipeline keys an engine's L2 dataset by its UED id, and
# refuses new measurements of an engine whose descriptors changed under an unchanged id
# (dataset_binding_mismatch). uhd_ued_id_overrides.json lists the engines given a new id.
#
# Their UEDs in git already carry the new id. Their KDPs are DVC outputs that still name the
# old one in their engine field, so this rewrites old to new in the catalogs of the root the
# product pack reads (a copy in the pipeline) and of the source tree (the pulled DVC outputs,
# which the pipeline's packet step promotes into); no git-tracked file is touched. A UUID
# replaces a UUID, so every catalog keeps the size its DVC pointer records (on Linux:
# file(WRITE) on Windows writes CRLF line endings).

set(_uhd_ids_roots "${HIPKERNELPROVIDER_PRODUCTION_DESCRIPTOR_SOURCE_ROOT}")
if(HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT)
    list(APPEND _uhd_ids_roots "${HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT}")
endif()
list(REMOVE_DUPLICATES _uhd_ids_roots)

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

foreach(_uhd_ids_root IN LISTS _uhd_ids_roots)
    if(NOT IS_DIRECTORY "${_uhd_ids_root}/conv")
        continue()
    endif()
    file(GLOB_RECURSE _uhd_ids_docs "${_uhd_ids_root}/conv/*.kdp.json")
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
endforeach()
