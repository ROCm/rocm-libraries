# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Resolves the rocm_kpack Python source used to pack GPU kernels into .kpack
# archives. Reaches the network only when HIPKERNELPROVIDER_KPACK_ALLOW_FETCH
# is ON.
#
# Resolution order:
#   1. HIPKERNELPROVIDER_KPACK_PYTHON_DIR       fatal if it holds no rocm_kpack/
#   2. ROCKE_KPACK_PYTHON_DIR (deprecated)      seeds 1
#   3. HIPKERNELPROVIDER_KPACK_DEFAULT_DIRS     skipped when absent
#   4. pinned fetch, if allowed
#   5. empty -- callers decide whether that is fatal

include_guard(GLOBAL)

set(HIPKERNELPROVIDER_KPACK_PYTHON_DIR "" CACHE PATH
    "Path to the rocm_kpack Python source (the parent of rocm_kpack/): \
<rocm-systems>/shared/kpack/python, or a virtual environment's site-packages.")

# Superprojects pass ROCKE_KPACK_PYTHON_DIR. Absolutized because the alias is
# untyped and this module is included from several directories. The canonical
# flag wins: the alias writes the cache entry only when it is empty or still
# holds a previously seeded value, so a warm build directory tracks a changed
# alias without overwriting an explicit path.
if(NOT "${ROCKE_KPACK_PYTHON_DIR}" STREQUAL "" AND
   ("${HIPKERNELPROVIDER_KPACK_PYTHON_DIR}" STREQUAL "" OR
    "${HIPKERNELPROVIDER_KPACK_PYTHON_DIR}" STREQUAL "${_KPACK_SEEDED_KPACK_DIR}"))
    get_filename_component(_rocke_abs "${ROCKE_KPACK_PYTHON_DIR}"
                           ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
    if(NOT "${_rocke_abs}" STREQUAL "${HIPKERNELPROVIDER_KPACK_PYTHON_DIR}")
        set(HIPKERNELPROVIDER_KPACK_PYTHON_DIR "${_rocke_abs}" CACHE PATH
            "Path to the rocm_kpack Python source (the parent of rocm_kpack/): \
<rocm-systems>/shared/kpack/python, or a virtual environment's site-packages." FORCE)
    endif()
    set(_KPACK_SEEDED_KPACK_DIR "${_rocke_abs}" CACHE INTERNAL
        "Canonical kpack path this module last seeded from the deprecated alias.")
    message(STATUS "kpack: ROCKE_KPACK_PYTHON_DIR is deprecated; "
        "pass HIPKERNELPROVIDER_KPACK_PYTHON_DIR instead.")
endif()

option(HIPKERNELPROVIDER_KPACK_ALLOW_FETCH
    "Fetch rocm_kpack when no local source is configured." OFF)

# The dev container stages rocm_kpack plus msgpack/zstandard in the first entry,
# so a container build needs no flags. Absent or malformed entries are skipped,
# not fatal: they describe the environment rather than an explicit request.
set(HIPKERNELPROVIDER_KPACK_DEFAULT_DIRS "/opt/rocm-kpack/python" CACHE STRING
    "Directories searched for rocm_kpack when no path is configured.")

# Pin copied from the kpack-ref output of .github/actions/ci-env/action.yml;
# CMake cannot read that action. The kpack-pin-sync pre-commit hook holds them
# equal. Must stay a full SHA: the git wire protocol cannot fetch an
# abbreviated one. Source is rocm-systems; the standalone ROCm/rocm-kpack repo
# is unmaintained.
set(HIPKERNELPROVIDER_KPACK_GIT_REPO "https://github.com/ROCm/rocm-systems.git"
    CACHE STRING "Repository to fetch rocm_kpack from (override for a fork).")
set(HIPKERNELPROVIDER_KPACK_GIT_REF "a022846cf553c2b135410a5168f97705f1b9c6ac"
    CACHE STRING "rocm-systems git ref (SHA, tag, or branch) to fetch.")
# Directory within that repository holding the kpack project, and the path from
# there to the parent of rocm_kpack/.
set(HIPKERNELPROVIDER_KPACK_GIT_SUBDIR "shared/kpack/python" CACHE STRING
    "Subdirectory of the fetched repository holding the rocm_kpack package.")

# _kpack_is_python_dir(<dir> <out_var>)
#   TRUE when <dir> is the parent of an importable-looking rocm_kpack package.
#   Tests emptiness directly rather than relying on if(<value>): a path is data,
#   and CMake's falsey constants would reject a real directory named e.g.
#   ".../tree-NOTFOUND".
function(_kpack_is_python_dir dir out_var)
    if(NOT "${dir}" STREQUAL "" AND EXISTS "${dir}/rocm_kpack/kpack.py")
        set(${out_var} TRUE PARENT_SCOPE)
    else()
        set(${out_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

# _kpack_fetch(<out_dir>)
#   Sparse, blobless, depth-1 checkout of the pinned commit's kpack directory.
#   The pin lives in a monorepo, where a full clone costs ~1.6 GB to obtain a
#   directory of Python files. This is the same recipe the dev container and the
#   superbuild CI jobs use.
#
#   Re-entrant: a checkout already at the pinned commit is reused, so a warm
#   configure touches no network. Reuse is gated on the checked-out commit
#   rather than on the package being present, so a bumped pin refetches and a
#   checkout interrupted partway through is not mistaken for a complete one.
function(_kpack_fetch out_dir)
    set(_src "${CMAKE_BINARY_DIR}/_kpack-fetch")
    set(_result "${_src}/${HIPKERNELPROVIDER_KPACK_GIT_SUBDIR}")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
        WORKING_DIRECTORY "${_src}"
        RESULT_VARIABLE _rev_rc
        OUTPUT_VARIABLE _head
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    _kpack_is_python_dir("${_result}" _have)
    if(_have AND _rev_rc EQUAL 0 AND
       "${_head}" STREQUAL "${HIPKERNELPROVIDER_KPACK_GIT_REF}")
        set(${out_dir} "${_result}" PARENT_SCOPE)
        return()
    endif()

    message(STATUS "kpack: fetching "
        "${HIPKERNELPROVIDER_KPACK_GIT_REPO}@${HIPKERNELPROVIDER_KPACK_GIT_REF}")
    file(REMOVE_RECURSE "${_src}")
    file(MAKE_DIRECTORY "${_src}")
    # The cone holds the project directory, one level above the python/ dir the
    # caller receives, so pyproject.toml comes along with the package. A
    # single-component subdir has no parent; cone on it directly, since an empty
    # cone checks out nothing and would fail as if the pin were wrong.
    get_filename_component(_cone "${HIPKERNELPROVIDER_KPACK_GIT_SUBDIR}" DIRECTORY)
    if("${_cone}" STREQUAL "")
        set(_cone "${HIPKERNELPROVIDER_KPACK_GIT_SUBDIR}")
    endif()
    foreach(_step
            "init;-q;."
            "remote;add;origin;${HIPKERNELPROVIDER_KPACK_GIT_REPO}"
            "sparse-checkout;init;--cone"
            "sparse-checkout;set;${_cone}"
            "fetch;--depth;1;--filter=blob:none;origin;${HIPKERNELPROVIDER_KPACK_GIT_REF}"
            "checkout;-q;FETCH_HEAD")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" ${_step}
            WORKING_DIRECTORY "${_src}"
            RESULT_VARIABLE _rc
            OUTPUT_QUIET
            ERROR_VARIABLE _err)
        if(NOT _rc EQUAL 0)
            string(REPLACE ";" " " _cmd "${_step}")
            message(FATAL_ERROR "kpack: git ${_cmd} failed: ${_err}")
        endif()
    endforeach()

    _kpack_is_python_dir("${_result}" _ok)
    if(NOT _ok)
        message(FATAL_ERROR "kpack: fetched "
            "${HIPKERNELPROVIDER_KPACK_GIT_REPO}@${HIPKERNELPROVIDER_KPACK_GIT_REF} "
            "but it has no ${HIPKERNELPROVIDER_KPACK_GIT_SUBDIR}/rocm_kpack/kpack.py.")
    endif()
    set(${out_dir} "${_result}" PARENT_SCOPE)
    message(STATUS "kpack: fetched into ${_result}")
endfunction()

# kpack_resolve_python_dir(<out_dir>)
#   A directory to prepend to PYTHONPATH so rocm_kpack imports, or empty when
#   nothing resolved. A configured-but-wrong path is fatal rather than a
#   fallback to fetching: an explicit value states intent.
#
#   Every tier absolutizes against CMAKE_SOURCE_DIR. A bare ABSOLUTE would use
#   the including listfile's directory, and this module is included from more
#   than one, so a relative value would name different trees in one build.
function(kpack_resolve_python_dir out_dir)
    if(NOT "${HIPKERNELPROVIDER_KPACK_PYTHON_DIR}" STREQUAL "")
        get_filename_component(_dir "${HIPKERNELPROVIDER_KPACK_PYTHON_DIR}"
                               ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
        _kpack_is_python_dir("${_dir}" _ok)
        if(NOT _ok)
            message(FATAL_ERROR
                "kpack: HIPKERNELPROVIDER_KPACK_PYTHON_DIR is '${_dir}' but "
                "'${_dir}/rocm_kpack/kpack.py' does not exist. Point it at the "
                "directory containing rocm_kpack/.")
        endif()
        set(${out_dir} "${_dir}" PARENT_SCOPE)
        message(STATUS "kpack: using rocm_kpack from ${_dir}")
        return()
    endif()

    # Absolutized so the value handed to PYTHONPATH is valid from any cwd.
    foreach(_candidate IN LISTS HIPKERNELPROVIDER_KPACK_DEFAULT_DIRS)
        get_filename_component(_abs "${_candidate}" ABSOLUTE
                               BASE_DIR "${CMAKE_SOURCE_DIR}")
        _kpack_is_python_dir("${_abs}" _ok)
        if(_ok)
            set(${out_dir} "${_abs}" PARENT_SCOPE)
            message(STATUS "kpack: using rocm_kpack from ${_abs} "
                "(shipped by the build environment)")
            return()
        endif()
    endforeach()

    if(NOT HIPKERNELPROVIDER_KPACK_ALLOW_FETCH)
        set(${out_dir} "" PARENT_SCOPE)
        return()
    endif()

    find_package(Git REQUIRED)
    _kpack_fetch(_fetched)
    set(${out_dir} "${_fetched}" PARENT_SCOPE)
endfunction()

# kpack_unset_reason(<out_var>)
#   The remediation message callers print when resolution returns empty.
function(kpack_unset_reason out_var)
    # Re-joined with commas; a raw ; would read as sentence punctuation.
    string(REPLACE ";" ", " _dirs_csv "${HIPKERNELPROVIDER_KPACK_DEFAULT_DIRS}")
    set(_searched "")
    if(NOT "${_dirs_csv}" STREQUAL "")
        set(_searched " Searched ${_dirs_csv}, which the hipDNN dev container ships.")
    endif()
    # One argument: multiple set() values build a ;-joined list.
    set(${out_var}
        "no rocm_kpack source found.${_searched} Pass -DHIPKERNELPROVIDER_KPACK_PYTHON_DIR=<rocm-systems>/shared/kpack/python, or set -DHIPKERNELPROVIDER_KPACK_ALLOW_FETCH=ON to fetch the pinned commit"
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
#   Runs from an empty directory: `python -c` puts the working directory on
#   sys.path ahead of PYTHONPATH, so a stray rocm_kpack/ in the build tree would
#   otherwise answer for the resolved one.
#
#   Never installs: doing so at configure time would mutate the host
#   environment from an unpinned index.
function(kpack_check_python_deps python_exe pythonpath out_missing)
    set(_missing "")
    set(_probe_dir "${CMAKE_CURRENT_BINARY_DIR}/kpack-probe")
    file(MAKE_DIRECTORY "${_probe_dir}")
    foreach(_mod rocm_kpack.compression rocm_kpack.kpack)
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${pythonpath}"
                    "PYTHONDONTWRITEBYTECODE=1" --
                    "${python_exe}" -c "import ${_mod}"
            WORKING_DIRECTORY "${_probe_dir}"
            RESULT_VARIABLE _rc
            OUTPUT_QUIET ERROR_QUIET)
        if(NOT _rc EQUAL 0)
            list(APPEND _missing "${_mod}")
        endif()
    endforeach()
    set(${out_missing} "${_missing}" PARENT_SCOPE)
endfunction()
