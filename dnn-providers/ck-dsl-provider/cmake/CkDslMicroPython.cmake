# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Build the embedded-MicroPython static library the provider links instead of
# CPython/pybind11. The heavy lifting (MicroPython checkout, mpy-cross, the
# ck_dsl bundle transform, module freezing, and compiling the embed C sources)
# lives in micropython/build_embed.sh; this module wraps it in a CMake custom
# command so the .a is (re)built as part of the normal build graph.
#
# Sets in the caller's scope:
#   CKDSL_MPY_BUILD_TARGET  custom target to depend on (orders the .a build)
#   CKDSL_MPY_STATIC_LIB    absolute path to libckdsl_micropython.a
#   CKDSL_MPY_INCLUDE_DIRS  include dirs for TUs that touch the MicroPython API
#   CKDSL_MPY_COMGR_LIB     resolved libamd_comgr to link
#
# Distribution toggle (plan §module-loading): CKDSL_MICROPYTHON_FROZEN selects
# how ck_dsl ships. Only the frozen mode (modules baked into the .a) is wired
# today; the on-disk .py / .mpy modes are future work and currently assert.
option(CKDSL_MICROPYTHON_FROZEN "Freeze ck_dsl/ck_dsl_provider into the plugin" ON)

# Optional pre-existing MicroPython checkout (offline / reproducible builds).
# When unset, build_embed.sh clones the pinned commit under the build dir.
set(CKDSL_MICROPYTHON_DIR "" CACHE PATH
    "Existing MicroPython source checkout (empty = clone pinned commit)")

# Define the embed static-lib custom build and export the consumer variables
# (CKDSL_MPY_*) into the calling scope. Call once from the top-level CMakeLists
# after the ck_dsl Python paths are resolved.
function(ck_dsl_provider_configure_micropython)
    if(NOT CKDSL_MICROPYTHON_FROZEN)
        message(FATAL_ERROR
            "CK DSL provider: only the frozen MicroPython mode is implemented; "
            "set CKDSL_MICROPYTHON_FROZEN=ON.")
    endif()

    find_program(BASH_PROGRAM bash REQUIRED)
    find_library(CKDSL_AMD_COMGR_LIBRARY amd_comgr REQUIRED
                 HINTS ${ROCM_PATH}/lib ${ROCM_PATH}/lib64)
    message(STATUS "CK DSL provider amd_comgr: ${CKDSL_AMD_COMGR_LIBRARY}")

    set(_root "${CMAKE_CURRENT_SOURCE_DIR}")
    set(_out "${CMAKE_CURRENT_BINARY_DIR}/micropython-embed")
    set(_lib "${_out}/libckdsl_micropython.a")
    set(_pkg "${_out}/micropython_embed")
    set(_buildEmbed "${_out}/build-embed")

    # Source-of-truth inputs: changing any of these rebuilds the lib. (Changes
    # to the ck_dsl/ck_dsl_provider .py trees are not individually tracked here;
    # touch build_embed.sh or reconfigure to force a refresh.)
    file(GLOB _shims "${_root}/micropython/shims/*.py")
    set(_deps
        "${_root}/micropython/build_embed.sh"
        "${_root}/micropython/build_bundle.py"
        "${_root}/micropython/build_frozen.py"
        "${_root}/micropython/gen.mk"
        "${_root}/micropython/manifest.py"
        "${_root}/src/micropython/comgr_compile.c"
        "${_root}/src/micropython/comgr_compile.h"
        "${_root}/src/micropython/modcomgr.c"
        "${_root}/src/micropython/embed_port.c"
        "${_root}/src/micropython/mpconfigport.h"
        ${_shims}
    )

    set(_env
        OUT_DIR=${_out}
        CK_DSL_SRC=${CK_DSL_PYTHON_PACKAGE_PATH}/ck_dsl
        CK_DSL_PROVIDER_SRC=${CK_DSL_PROVIDER_PYTHON_PACKAGE_PATH}/ck_dsl_provider
        ROCM_PATH=${ROCM_PATH}
    )
    if(CKDSL_MICROPYTHON_DIR)
        list(APPEND _env MPY_DIR=${CKDSL_MICROPYTHON_DIR})
    endif()

    add_custom_command(
        OUTPUT "${_lib}"
        COMMAND ${CMAKE_COMMAND} -E env ${_env}
                ${BASH_PROGRAM} "${_root}/micropython/build_embed.sh"
        DEPENDS ${_deps}
        COMMENT "Building embedded MicroPython static library (ck_dsl frozen)"
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
endfunction()
