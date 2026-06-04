# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Build the embedded-MicroPython static library the provider links instead of
# CPython/pybind11. The heavy lifting (MicroPython checkout, mpy-cross, the
# ck_dsl bundle transform, module freezing, and compiling the embed C sources)
# lives in micropython/build_embed.py; this module wraps it in a CMake custom
# command so the .a is (re)built as part of the normal build graph.
#
# Sets in the caller's scope:
#   CKDSL_MPY_BUILD_TARGET  custom target to depend on (orders the .a build)
#   CKDSL_MPY_STATIC_LIB    absolute path to libckdsl_micropython.a
#   CKDSL_MPY_INCLUDE_DIRS  include dirs for TUs that touch the MicroPython API
#   CKDSL_MPY_COMGR_LIB     resolved libamd_comgr to link
#   CKDSL_MPY_COMPILE_DEFS  compile definitions every MicroPython-API TU must see
#                           (must match what build_embed.py compiles the .a with)
#
# Distribution toggle (plan §module-loading): CKDSL_MICROPYTHON_MODE selects how
# ck_dsl ships:
#   frozen  ck_dsl/shims baked into the plugin as frozen bytecode (no filesystem;
#           default; smallest footprint, slowest to iterate -- editing a .py means
#           rebuilding + relinking the static lib).
#   py      ck_dsl/shims shipped as .py files beside the plugin, loaded from the
#           filesystem at runtime (fast iteration: edit the on-disk .py and rerun).
#   mpy     same, but pre-compiled to .mpy bytecode (smaller / faster load than .py).
# py and mpy use MICROPY_READER_POSIX + a real mp_import_stat in the embed port and
# put the on-disk bundle dir on sys.path (baked CKDSL_BUNDLE_DIR; a production
# install would relocate the bundle + override that path).
set(CKDSL_MICROPYTHON_MODE "frozen" CACHE STRING
    "How ck_dsl ships into the plugin: frozen | py | mpy")
set_property(CACHE CKDSL_MICROPYTHON_MODE PROPERTY STRINGS frozen py mpy)

# Optional pre-existing MicroPython checkout (offline / reproducible builds).
# When unset, build_embed.py clones the pinned commit under the build dir.
set(CKDSL_MICROPYTHON_DIR "" CACHE PATH
    "Existing MicroPython source checkout (empty = clone pinned commit)")

# Define the embed static-lib custom build and export the consumer variables
# (CKDSL_MPY_*) into the calling scope. Call once from the top-level CMakeLists
# after the ck_dsl Python paths are resolved.
function(ck_dsl_provider_configure_micropython)
    set(_validModes frozen py mpy)
    if(NOT CKDSL_MICROPYTHON_MODE IN_LIST _validModes)
        message(FATAL_ERROR
            "CK DSL provider: CKDSL_MICROPYTHON_MODE must be one of ${_validModes} "
            "(got '${CKDSL_MICROPYTHON_MODE}').")
    endif()
    message(STATUS "CK DSL provider MicroPython mode: ${CKDSL_MICROPYTHON_MODE}")

    # The embed build is a cross-platform Python driver (no bash, no GNU make),
    # so it works on Windows. Python is already a build-time requirement.
    if(NOT Python3_EXECUTABLE)
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
    endif()
    find_library(CKDSL_AMD_COMGR_LIBRARY amd_comgr REQUIRED
                 HINTS ${ROCM_PATH}/lib ${ROCM_PATH}/lib64)
    message(STATUS "CK DSL provider amd_comgr: ${CKDSL_AMD_COMGR_LIBRARY}")

    set(_root "${CMAKE_CURRENT_SOURCE_DIR}")
    set(_out "${CMAKE_CURRENT_BINARY_DIR}/micropython-embed")
    set(_lib "${_out}/libckdsl_micropython.a")
    set(_pkg "${_out}/micropython_embed")
    set(_buildEmbed "${_out}/build-embed")

    # Source-of-truth inputs: changing any of these rebuilds the lib. CONFIGURE_DEPENDS
    # re-globs the ck_dsl / ck_dsl_provider trees at build time, so editing a frozen
    # module's .py retriggers the embed build (no reconfigure needed).
    file(GLOB _shims "${_root}/micropython/shims/*.py")
    file(GLOB_RECURSE _ckPy CONFIGURE_DEPENDS
        "${CK_DSL_PYTHON_PACKAGE_PATH}/ck_dsl/*.py"
        "${CK_DSL_PROVIDER_PYTHON_PACKAGE_PATH}/ck_dsl_provider/*.py")
    set(_deps
        "${_root}/micropython/build_embed.py"
        "${_root}/micropython/build_bundle.py"
        "${_root}/micropython/build_frozen.py"
        "${_root}/micropython/manifest.py"
        "${_root}/src/micropython/comgr_compile.c"
        "${_root}/src/micropython/comgr_compile.h"
        "${_root}/src/micropython/modcomgr.c"
        "${_root}/src/micropython/embed_port.c"
        "${_root}/src/micropython/mpconfigport.h"
        ${_shims}
        ${_ckPy}
    )

    set(_env
        OUT_DIR=${_out}
        CK_DSL_SRC=${CK_DSL_PYTHON_PACKAGE_PATH}/ck_dsl
        CK_DSL_PROVIDER_SRC=${CK_DSL_PROVIDER_PYTHON_PACKAGE_PATH}/ck_dsl_provider
        ROCM_PATH=${ROCM_PATH}
        CKDSL_MODE=${CKDSL_MICROPYTHON_MODE}
    )
    if(CKDSL_MICROPYTHON_DIR)
        list(APPEND _env MPY_DIR=${CKDSL_MICROPYTHON_DIR})
    endif()

    # Compile definitions every MicroPython-API TU must agree on (the C++ bridge /
    # interpreter and the .a). Frozen bakes modules in; py/mpy load from disk and
    # bake the on-disk bundle dir for sys.path (the .a build sets CKDSL_ON_DISK
    # itself from CKDSL_MODE; here we mirror it for the C++ side + add the path).
    # Subdir name the on-disk bundle installs into beside the plugin .so, and
    # that EmbeddedInterpreter looks for via dladdr at runtime -- the two must
    # agree, so both come from this one variable.
    set(_bundleInstallDirname "ck_dsl_micropython")
    if(CKDSL_MICROPYTHON_MODE STREQUAL "frozen")
        set(_compileDefs MICROPY_MODULE_FROZEN_MPY=1 MICROPY_MODULE_FROZEN_STR=1)
        set(_bundleDir "")
    else()
        if(CKDSL_MICROPYTHON_MODE STREQUAL "mpy")
            set(_bundleDir "${_out}/frozen_src_mpy")
        else()
            set(_bundleDir "${_out}/frozen_src")
        endif()
        set(_compileDefs CKDSL_ON_DISK=1
            "CKDSL_BUNDLE_DIR=\"${_bundleDir}\""
            "CKDSL_BUNDLE_INSTALL_DIRNAME=\"${_bundleInstallDirname}\"")
    endif()

    add_custom_command(
        OUTPUT "${_lib}"
        COMMAND ${CMAKE_COMMAND} -E env ${_env}
                ${Python3_EXECUTABLE} "${_root}/micropython/build_embed.py"
        DEPENDS ${_deps}
        COMMENT "Building embedded MicroPython static library (${CKDSL_MICROPYTHON_MODE})"
        VERBATIM
        USES_TERMINAL
    )
    add_custom_target(ck_dsl_micropython_build
        DEPENDS "${_lib}"
        COMMENT "Embedded MicroPython static library up to date"
    )

    set(CKDSL_MPY_BUILD_TARGET ck_dsl_micropython_build PARENT_SCOPE)
    set(CKDSL_MPY_STATIC_LIB "${_lib}" PARENT_SCOPE)
    set(CKDSL_MPY_INCLUDE_DIRS
        "${_root}/src/micropython" "${_pkg}" "${_pkg}/port" "${_buildEmbed}"
        PARENT_SCOPE)
    set(CKDSL_MPY_COMGR_LIB "${CKDSL_AMD_COMGR_LIBRARY}" PARENT_SCOPE)
    set(CKDSL_MPY_COMPILE_DEFS "${_compileDefs}" PARENT_SCOPE)
    # On-disk modes only (empty in frozen mode): the bundle dir to install beside
    # the plugin, and the subdir name to install it under. The top-level
    # CMakeLists adds the install(DIRECTORY) rule from these.
    set(CKDSL_MPY_BUNDLE_DIR "${_bundleDir}" PARENT_SCOPE)
    set(CKDSL_MPY_BUNDLE_INSTALL_DIRNAME "${_bundleInstallDirname}" PARENT_SCOPE)

    # mpy-cross built from source by build_embed.py (always under the build dir, the
    # same way on every platform), unless a prebuilt one was supplied. The compat
    # lint uses it to compile-check every module with the real MicroPython compiler.
    if(DEFINED ENV{CKDSL_MPY_CROSS_BIN})
        set(CKDSL_MPY_CROSS "$ENV{CKDSL_MPY_CROSS_BIN}" PARENT_SCOPE)
    else()
        set(CKDSL_MPY_CROSS "${_out}/mpy-cross-build/mpy-cross${CMAKE_EXECUTABLE_SUFFIX}"
            PARENT_SCOPE)
    endif()
endfunction()
