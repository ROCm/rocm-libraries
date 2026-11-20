# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

# TheRock's release, rc, nightly and dev identities, for example 10.3.0,
# 10.3.0rc2, 10.3.0a20261103, 10.3.0a20261103+bkc.20261110 and
# 10.3.0.dev0+<sha>. Keep in sync with tensilelite/tensilelite/_build_rocm.py.
set(HIPBLASLT_ROCM_IDENTITY_REGEX
    "^([0-9]+\\.[0-9]+\\.[0-9]+)((a|b|rc)[0-9]+)?(\\.dev[0-9]+)?(\\+[a-z0-9]+(\\.[a-z0-9]+)*)?$")

# hipblasLtGetBuildVersion callers may size their buffer to this.
set(HIPBLASLT_BUILD_VERSION_MAX_LENGTH 256)

# Sets <output> to the X.Y.Z of the ROCm installed at <root>, or to "" when the
# root records none or its two records disagree.
function(hipblaslt_rocm_root_version root output)
    set(_manifest "")
    set(_manifest_file "${root}/share/therock/therock_manifest.json")
    if(EXISTS "${_manifest_file}")
        file(READ "${_manifest_file}" _json)
        string(JSON _identity ERROR_VARIABLE _error GET "${_json}" rocm_package_version)
        if(NOT _error AND _identity MATCHES "${HIPBLASLT_ROCM_IDENTITY_REGEX}")
            set(_manifest "${CMAKE_MATCH_1}")
        endif()
    endif()

    set(_info "")
    set(_info_file "${root}/.info/version")
    if(EXISTS "${_info_file}")
        file(STRINGS "${_info_file}" _line LIMIT_COUNT 1)
        if(_line MATCHES "^([0-9]+\\.[0-9]+\\.[0-9]+)")
            set(_info "${CMAKE_MATCH_1}")
        endif()
    endif()

    if(_manifest AND _info AND NOT _manifest STREQUAL _info)
        message(WARNING "Inconsistent ROCm installation at ${root}: ${_manifest_file} says "
            "${_manifest} but ${_info_file} says ${_info}; not using either.")
        set(${output} "" PARENT_SCOPE)
    elseif(_manifest)
        set(${output} "${_manifest}" PARENT_SCOPE)
    else()
        set(${output} "${_info}" PARENT_SCOPE)
    endif()
endfunction()

# Sets <output> to the hipBLASLt product version, X.Y.Z, taken from the first of:
# <explicit> (HIPBLASLT_PROJECT_VERSION), <rocm_version> (ROCM_VERSION from a
# superproject such as TheRock), the ROCm installed at <rocm_root>, or 0.0.0.
# Consumers pack each part into three decimal digits (MIOpen's
# MIOPEN_HIPBLASLT_VERSION_FLAT), so no part may exceed 999.
function(hipblaslt_resolve_project_version explicit rocm_version rocm_root output)
    if(NOT explicit STREQUAL "")
        set(_version "${explicit}")
        set(_source "HIPBLASLT_PROJECT_VERSION")
    elseif(NOT rocm_version STREQUAL "")
        set(_version "${rocm_version}")
        set(_source "ROCM_VERSION")
    else()
        hipblaslt_rocm_root_version("${rocm_root}" _version)
        set(_source "the ROCm installation at ${rocm_root}")
        if(_version STREQUAL "")
            message(WARNING "No ROCm version found (set ROCM_VERSION or "
                "HIPBLASLT_PROJECT_VERSION, or build against a ROCm installation with "
                ".info/version); hipBLASLt reports version 0.0.0.")
            set(_version "0.0.0")
            set(_source "the fallback")
        endif()
    endif()

    if(NOT _version MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$")
        message(FATAL_ERROR "hipBLASLt version '${_version}' from ${_source} is not X.Y.Z.")
    endif()
    foreach(_part IN ITEMS "${CMAKE_MATCH_1}" "${CMAKE_MATCH_2}" "${CMAKE_MATCH_3}")
        if(_part GREATER 999)
            message(FATAL_ERROR
                "hipBLASLt version '${_version}' from ${_source} has a part above 999.")
        endif()
    endforeach()
    message(STATUS "hipBLASLt version ${_version} (from ${_source})")
    set(${output} "${_version}" PARENT_SCOPE)
endfunction()

# Sets <output> to "<identity> (git <revision>)". The identity is
# <package_version> (THEROCK_PACKAGE_VERSION) when it is a TheRock release, rc,
# nightly or dev identity, otherwise <project_version>.dev0, which never names a
# release.
function(hipblaslt_build_version project_version package_version revision output)
    if(package_version MATCHES "${HIPBLASLT_ROCM_IDENTITY_REGEX}")
        set(_identity "${package_version}")
        if(NOT CMAKE_MATCH_1 STREQUAL project_version)
            message(WARNING "THEROCK_PACKAGE_VERSION ${package_version} does not match "
                "hipBLASLt version ${project_version}.")
        endif()
    else()
        set(_identity "${project_version}.dev0")
    endif()
    if(revision STREQUAL "")
        set(revision "unknown")
    endif()
    set(_value "${_identity} (git ${revision})")
    string(LENGTH "${_value}" _length)
    if(NOT _length LESS HIPBLASLT_BUILD_VERSION_MAX_LENGTH)
        message(FATAL_ERROR "hipBLASLt build version '${_value}' is ${_length} characters; "
            "it must be shorter than ${HIPBLASLT_BUILD_VERSION_MAX_LENGTH}.")
    endif()
    set(${output} "${_value}" PARENT_SCOPE)
endfunction()
