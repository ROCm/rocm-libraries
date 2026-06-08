# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Provision a self-contained CPython for the embedded interpreter from
# astral's python-build-standalone (PBS), replacing any dependency on
# the host system Python.
#
# The PBS `install_only` distribution ships a relocatable prefix with a
# shared libpython, the full stdlib, and headers. Critically, it links
# the runtime C-extensions the CK DSL compile path needs
# (`_ctypes` -> comgr, `_hashlib`, `_bz2`, `_lzma`) statically into
# libpython itself, so the hardened plugin (hidden visibility +
# --exclude-libs=ALL) can load them through the single libpython
# DT_NEEDED dependency -- no separate extension .so resolution.
#
# Provisioning honors a caller-supplied distribution first
# (CKDSL_PYTHON_DIST_DIR), so air-gapped CI / TheRock can pre-stage the
# prefix; otherwise the pinned tarball is downloaded and checksum-
# verified via FetchContent.
#
# Outputs (cache):
#   CKDSL_PYTHON_PREFIX    absolute path of the CPython prefix
#                          (the dir containing bin/ lib/ include/)
#   CKDSL_PYTHON_VERSION   the CPython MAJOR.MINOR (e.g. "3.12")
#
# Side effects: sets Python3_ROOT_DIR + discovery hints in the parent
# scope so a subsequent find_package(Python3 ... Development) resolves
# the provisioned prefix rather than any host interpreter.

# --- Pinned distribution -----------------------------------------------------
# python-build-standalone release tag and the install_only assets. Bump
# the tag + all three checksums together; verify against the release's
# SHA256SUMS (PBS does not publish per-asset .sha256 sidecars).
set(_CKDSL_PYTHON_PBS_TAG "20260602")
set(_CKDSL_PYTHON_PBS_VERSION "3.12.13")
set(_CKDSL_PYTHON_PBS_BASE
    "https://github.com/astral-sh/python-build-standalone/releases/download/${_CKDSL_PYTHON_PBS_TAG}"
)

# Per-platform install_only asset + sha256 (Linux x86_64 verified; the
# others are wired for parity and selected by host, but only the Linux
# x86_64 path is validated today).
set(_CKDSL_PYTHON_ASSET_linux_x86_64
    "cpython-${_CKDSL_PYTHON_PBS_VERSION}+${_CKDSL_PYTHON_PBS_TAG}-x86_64-unknown-linux-gnu-install_only.tar.gz")
set(_CKDSL_PYTHON_SHA_linux_x86_64
    "9be5c21b78dbc371e739bc7faf3b007b8e607335f780bdd2e0dd44a6e3580d76")

set(_CKDSL_PYTHON_ASSET_linux_aarch64
    "cpython-${_CKDSL_PYTHON_PBS_VERSION}+${_CKDSL_PYTHON_PBS_TAG}-aarch64-unknown-linux-gnu-install_only.tar.gz")
set(_CKDSL_PYTHON_SHA_linux_aarch64
    "f0c9ea0022b2dfdf0a4733e962ba8cc883c45d26df26116b9802b658240a25d7")

set(_CKDSL_PYTHON_ASSET_windows_x86_64
    "cpython-${_CKDSL_PYTHON_PBS_VERSION}+${_CKDSL_PYTHON_PBS_TAG}-x86_64-pc-windows-msvc-install_only.tar.gz")
set(_CKDSL_PYTHON_SHA_windows_x86_64
    "f89539b0a6f1d48c655f97b7e77f3f8738bbe9d7a32c9306d0d20335dc8ae0fb")

# Snapshot this .cmake file's dir at include time (CMAKE_CURRENT_LIST_DIR
# inside a function body reflects the caller, not the definition site).
set(_ckDslPythonCmakeDir "${CMAKE_CURRENT_LIST_DIR}")

# Validate that a directory looks like a usable CPython prefix: the
# shared library, the public header, and the interpreter binary must all
# exist. The python-build-standalone layout differs by platform:
#   Linux:   bin/python<ver>, lib/libpython<ver>.so, include/python<ver>/
#   Windows: python.exe, python3.dll (+ python<ver>.dll), include/ (flat)
# Sets ${outVar} TRUE/FALSE in the caller's scope.
function(_ck_dsl_python_validate_prefix prefix version outVar)
    set(_ok TRUE)
    if(WIN32)
        set(_lib "${prefix}/python3.dll")
        set(_bin "${prefix}/python.exe")
        set(_inc "${prefix}/include/Python.h")
    else()
        set(_lib "${prefix}/lib/libpython${version}.so")
        set(_bin "${prefix}/bin/python${version}")
        set(_inc "${prefix}/include/python${version}/Python.h")
    endif()
    foreach(_p "${_lib}" "${_bin}" "${_inc}")
        if(NOT EXISTS "${_p}")
            message(STATUS "CK DSL provider: python prefix missing ${_p}")
            set(_ok FALSE)
        endif()
    endforeach()
    set(${outVar} ${_ok} PARENT_SCOPE)
endfunction()

# Resolve the pinned asset + sha for the host platform.
function(_ck_dsl_python_select_asset assetOut shaOut)
    if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(_key "windows_x86_64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(_key "linux_aarch64")
    else()
        set(_key "linux_x86_64")
    endif()
    if(NOT DEFINED _CKDSL_PYTHON_ASSET_${_key})
        message(FATAL_ERROR
            "CK DSL provider: no pinned python-build-standalone asset for this "
            "platform (key '${_key}'). Provide one via -DCKDSL_PYTHON_DIST_DIR.")
    endif()
    set(${assetOut} "${_CKDSL_PYTHON_ASSET_${_key}}" PARENT_SCOPE)
    set(${shaOut} "${_CKDSL_PYTHON_SHA_${_key}}" PARENT_SCOPE)
endfunction()

# Provision CPython and steer find_package(Python3) at it.
function(ck_dsl_provider_provision_python)
    set(_version "${_CKDSL_PYTHON_PBS_VERSION}")
    string(REGEX MATCH "^([0-9]+\\.[0-9]+)" _shortVersion "${_version}")

    # 1. Caller-supplied distribution wins (air-gap / pre-stage).
    if(DEFINED CKDSL_PYTHON_DIST_DIR AND CKDSL_PYTHON_DIST_DIR)
        get_filename_component(_prefix "${CKDSL_PYTHON_DIST_DIR}" ABSOLUTE)
        _ck_dsl_python_validate_prefix("${_prefix}" "${_shortVersion}" _valid)
        if(NOT _valid)
            message(FATAL_ERROR
                "CK DSL provider: CKDSL_PYTHON_DIST_DIR='${_prefix}' is not a "
                "usable CPython ${_shortVersion} prefix (expected bin/, "
                "lib/libpython${_shortVersion}.so, include/python${_shortVersion}).")
        endif()
        message(STATUS "CK DSL provider: using supplied Python prefix ${_prefix}")
    else()
        # 2. Download + checksum-verify the pinned PBS tarball.
        _ck_dsl_python_select_asset(_asset _sha)
        include(FetchContent)
        message(STATUS
            "CK DSL provider: fetching python-build-standalone ${_version} "
            "(${_asset})")
        fetchcontent_declare(
            ckdsl_cpython
            URL "${_CKDSL_PYTHON_PBS_BASE}/${_asset}"
            URL_HASH SHA256=${_sha}
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        # The archive has no CMakeLists, so MakeAvailable only populates
        # (download + extract); it never tries to add_subdirectory.
        fetchcontent_makeavailable(ckdsl_cpython)
        # PBS archives have a single top-level "python/" dir, which
        # FetchContent strips on extraction -- so bin/ lib/ include/ land
        # directly in SOURCE_DIR, which is therefore the prefix.
        set(_prefix "${ckdsl_cpython_SOURCE_DIR}")
        _ck_dsl_python_validate_prefix("${_prefix}" "${_shortVersion}" _valid)
        if(NOT _valid)
            message(FATAL_ERROR
                "CK DSL provider: fetched python-build-standalone archive did not "
                "unpack to the expected prefix at ${_prefix}.")
        endif()
    endif()

    set(CKDSL_PYTHON_PREFIX "${_prefix}" CACHE PATH
        "CPython prefix used by the ck-dsl-provider embedded interpreter" FORCE)
    set(CKDSL_PYTHON_VERSION "${_shortVersion}" CACHE STRING
        "CPython MAJOR.MINOR for the ck-dsl-provider embedded interpreter" FORCE)

    # Steer find_package(Python3 ... Development) at the provisioned
    # prefix. LOCATION strategy + NEVER virtualenv stop a host
    # interpreter (incl. a uv-managed one in ~/.local) from hijacking
    # discovery -- the original failure the system-Python pin guarded.
    set(Python3_ROOT_DIR "${_prefix}" PARENT_SCOPE)
    set(Python3_FIND_STRATEGY "LOCATION" PARENT_SCOPE)
    set(Python3_FIND_VIRTUALENV "NEVER" PARENT_SCOPE)

    message(STATUS
        "CK DSL provider: provisioned CPython ${_version} at ${_prefix}")
endfunction()
