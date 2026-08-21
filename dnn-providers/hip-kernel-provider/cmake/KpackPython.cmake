# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Resolves the rocm_kpack Python source for this provider's kpack consumers.
# The build never reaches the network unless HIPKERNELPROVIDER_KPACK_ALLOW_FETCH
# is set, and that fetch is pinned.
#
# Order: HIPKERNELPROVIDER_KPACK_PYTHON_DIR, the deprecated
# ROCKE_KPACK_PYTHON_DIR, a pinned fetch when allowed, else empty. Callers
# decide whether empty is fatal.

include_guard(GLOBAL)

set(HIPKERNELPROVIDER_KPACK_PYTHON_DIR "" CACHE PATH
    "Path to the rocm_kpack Python source (the parent of rocm_kpack/): \
<rocm-systems>/shared/kpack/python or <rocm-kpack>/python.")

# Superprojects still pass ROCKE_KPACK_PYTHON_DIR; accept it until they migrate.
# Absolutized here because the alias is untyped and the consumers include this
# module from different directories. The canonical flag wins: the alias writes
# the cache entry only when it is empty or still holds a previously seeded
# value, so a warm build dir tracks a changed alias without overwriting an
# explicit path.
if(ROCKE_KPACK_PYTHON_DIR AND
   (NOT HIPKERNELPROVIDER_KPACK_PYTHON_DIR OR
    "${HIPKERNELPROVIDER_KPACK_PYTHON_DIR}" STREQUAL "${_KPACK_SEEDED_KPACK_DIR}"))
    get_filename_component(_rocke_abs "${ROCKE_KPACK_PYTHON_DIR}"
                           ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
    if(NOT "${_rocke_abs}" STREQUAL "${HIPKERNELPROVIDER_KPACK_PYTHON_DIR}")
        set(HIPKERNELPROVIDER_KPACK_PYTHON_DIR "${_rocke_abs}" CACHE PATH
            "Path to the rocm_kpack Python source (the parent of rocm_kpack/): \
<rocm-systems>/shared/kpack/python or <rocm-kpack>/python." FORCE)
    endif()
    set(_KPACK_SEEDED_KPACK_DIR "${_rocke_abs}" CACHE INTERNAL
        "Canonical kpack path this module last seeded from the deprecated alias.")
    message(STATUS "kpack: ROCKE_KPACK_PYTHON_DIR is deprecated; "
        "pass HIPKERNELPROVIDER_KPACK_PYTHON_DIR instead.")
endif()

option(HIPKERNELPROVIDER_KPACK_ALLOW_FETCH
    "Fetch rocm_kpack when no local source is configured." OFF)

# FetchContent re-fetches a moving ref only on a clean populate (a wiped
# _deps/rocm_kpack-*), so prefer a SHA over a branch.
set(HIPKERNELPROVIDER_KPACK_GIT_REPO "https://github.com/ROCm/rocm-kpack.git"
    CACHE STRING "rocm-kpack git repository to fetch (override for a fork).")
set(HIPKERNELPROVIDER_KPACK_GIT_REF "e3483286e751060b3a70b792792cc122632c66e8"
    CACHE STRING "rocm-kpack git ref (SHA, tag, or branch) to fetch.")

# kpack_resolve_python_dir(<out_var>)
#   Sets <out_var> to a directory containing rocm_kpack/kpack.py, or empty when
#   kpack is unavailable and fetching is not permitted. A configured-but-wrong
#   path is fatal rather than a fallback to fetching.
function(kpack_resolve_python_dir out_var)
    if(HIPKERNELPROVIDER_KPACK_PYTHON_DIR)
        get_filename_component(_dir "${HIPKERNELPROVIDER_KPACK_PYTHON_DIR}" ABSOLUTE)
        if(NOT EXISTS "${_dir}/rocm_kpack/kpack.py")
            message(FATAL_ERROR
                "kpack: HIPKERNELPROVIDER_KPACK_PYTHON_DIR is '${_dir}' but "
                "'${_dir}/rocm_kpack/kpack.py' does not exist. Point it at the "
                "directory containing rocm_kpack/.")
        endif()
        set(${out_var} "${_dir}" PARENT_SCOPE)
        message(STATUS "kpack: using rocm_kpack from ${_dir}")
        return()
    endif()

    if(NOT HIPKERNELPROVIDER_KPACK_ALLOW_FETCH)
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()

    message(STATUS "kpack: fetching "
        "${HIPKERNELPROVIDER_KPACK_GIT_REPO}@${HIPKERNELPROVIDER_KPACK_GIT_REF}")
    include(FetchContent)
    FetchContent_Declare(
        rocm_kpack
        GIT_REPOSITORY "${HIPKERNELPROVIDER_KPACK_GIT_REPO}"
        GIT_TAG "${HIPKERNELPROVIDER_KPACK_GIT_REF}"
    )
    # Only the source tree is needed, so a bare populate is used. CMP0169
    # (CMake >= 3.30) deprecates the single-argument form. PUSH/POP keeps the
    # OLD setting from leaking into the including directory scope.
    cmake_policy(PUSH)
    if(POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif()
    FetchContent_GetProperties(rocm_kpack)
    if(NOT rocm_kpack_POPULATED)
        FetchContent_Populate(rocm_kpack)
    endif()
    cmake_policy(POP)
    if(NOT EXISTS "${rocm_kpack_SOURCE_DIR}/python/rocm_kpack/kpack.py")
        message(FATAL_ERROR "kpack: fetched "
            "${HIPKERNELPROVIDER_KPACK_GIT_REPO}@${HIPKERNELPROVIDER_KPACK_GIT_REF} "
            "but it has no python/rocm_kpack/kpack.py.")
    endif()
    set(${out_var} "${rocm_kpack_SOURCE_DIR}/python" PARENT_SCOPE)
    message(STATUS "kpack: fetched into ${rocm_kpack_SOURCE_DIR}/python")
endfunction()

# kpack_unset_reason(<out_var>)
#   The remediation message callers print when resolution returns empty.
function(kpack_unset_reason out_var)
    # One argument: multiple set() values build a ;-joined list.
    set(${out_var}
        "no rocm_kpack source configured. Pass -DHIPKERNELPROVIDER_KPACK_PYTHON_DIR=<rocm-systems>/shared/kpack/python (or <rocm-kpack>/python), or set -DHIPKERNELPROVIDER_KPACK_ALLOW_FETCH=ON to fetch the pinned commit"
        PARENT_SCOPE)
endfunction()

# kpack_check_python_deps(<python_exe> <pythonpath> <out_missing>)
#   Reports which of pack.py's imports are unavailable, under the same
#   interpreter and PYTHONPATH the pack command uses.
#
#   Importing the rocm_kpack modules covers their third-party dependencies
#   transitively (kpack imports msgpack, compression imports zstandard) and
#   catches a rocm_kpack that resolves on disk but fails to import.
#
#   Never installs: doing so at configure time would mutate the host
#   environment from an unpinned index.
function(kpack_check_python_deps python_exe pythonpath out_missing)
    set(_missing "")
    foreach(_mod rocm_kpack.compression rocm_kpack.kpack)
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${pythonpath}" --
                    "${python_exe}" -c "import ${_mod}"
            RESULT_VARIABLE _rc
            OUTPUT_QUIET ERROR_QUIET)
        if(NOT _rc EQUAL 0)
            list(APPEND _missing "${_mod}")
        endif()
    endforeach()
    set(${out_missing} "${_missing}" PARENT_SCOPE)
endfunction()
