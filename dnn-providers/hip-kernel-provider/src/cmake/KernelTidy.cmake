# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# - This module makes the kernel sources that KernelEmbedding.cmake inlines into a
# generated translation unit visible to clang-tidy.
#
# An embedded kernel is never compiled by the build: embed_kernel_sources() reads it at
# configure time and pastes its text into a raw string literal, so it has no entry in
# compile_commands.json. clang_tidy_check() only sets <LANG>_CLANG_TIDY properties, which
# fire when a source is compiled, and the `tidy` targets drive run-clang-tidy off the
# compile database, so neither gate can ever see a kernel.
#
# add_kernel_tidy_target() closes that gap without compiling anything. It creates a custom
# target that invokes clang-tidy on the kernel files directly, passing the compiler flags
# they need after `--`. Nothing is added to the build graph that produces object code, so
# a normal build is unaffected in size and time; the target is not part of ALL and only
# runs when asked for.
#
# Three properties of the kernels shape the flags this module has to supply, because
# without a compile_commands.json entry there is nothing to inherit them from:
#
# * They are HIP sources, so they need -x hip. --offload-host-only keeps this to the host
#   pass, which still parses and type-checks device code but needs neither an
#   --offload-arch nor the ROCm device libraries.
# * They are hipRTC sources whose configuration arrives as -D options built by the host
#   plan (HIP_PLUGIN_*). Without those macros they do not parse, so the caller supplies a
#   PRELUDE header that defines a representative set, force-included with -include.
# * Their #includes are flat ("VectorTypes.hpp"), because at runtime they resolve out of
#   the embedded include map instead of the filesystem, so every directory holding a
#   registered kernel file goes on the include path.
#
# Headers are checked as well as sources. A kernel header that no kernel source includes
# would otherwise never be analysed, and .clang-tidy's HeaderFilterRegex only surfaces
# diagnostics from headers some translation unit actually reaches.
#
# Usage:
#   add_kernel_tidy_target(
#       NAME    hip_mlops_kernel_tidy
#       PRELUDE ${CMAKE_CURRENT_SOURCE_DIR}/tidy/KernelTidyPrelude.hpp
#       FILES   <the same list passed to add_kernels_for_embedding>)

# Tidy targets created so far, so the aggregate `tidy` targets can depend on them without
# each engine having to know what those targets are called.
define_property(GLOBAL PROPERTY KERNELTIDY_TARGETS)

# add_kernel_tidy_target(NAME <target> PRELUDE <header> FILES <kernel>...)
#
#   Create <target>, a custom target that runs clang-tidy over the embedded kernels in
#   FILES. Build it explicitly, or through the `tidy` targets that
#   hip_kernel_provider_tidy_dependencies() wires up.
#
#   Each file gets its own command, so the generator runs them in parallel and re-checks
#   only what changed: a per-file stamp under the build tree records the last successful
#   run, and a file is re-checked when it, the prelude or the .clang-tidy config is newer.
#
#   Does nothing when no clang-tidy run is configured (ENABLE_CLANG_TIDY off), so a plain
#   build never pays for it.
function(add_kernel_tidy_target)
    set(options "")
    set(oneValueArgs NAME PRELUDE)
    set(multiValueArgs FILES)
    cmake_parse_arguments(PARSE_ARGV 0 KERNEL_TIDY "${options}" "${oneValueArgs}"
                          "${multiValueArgs}")

    if(NOT KERNEL_TIDY_NAME)
        message(FATAL_ERROR "add_kernel_tidy_target called without a NAME!")
    endif()
    if(NOT KERNEL_TIDY_PRELUDE)
        message(FATAL_ERROR "add_kernel_tidy_target called without a PRELUDE!")
    endif()
    if(NOT EXISTS "${KERNEL_TIDY_PRELUDE}")
        message(FATAL_ERROR "add_kernel_tidy_target: PRELUDE ${KERNEL_TIDY_PRELUDE} does not exist.")
    endif()
    if(NOT KERNEL_TIDY_FILES)
        message(FATAL_ERROR "add_kernel_tidy_target called without any FILES!")
    endif()

    # The target only ever exists to run clang-tidy, and Windows has no `tidy` target at
    # all (see add_clang_tidy_custom_target), so in both cases it would be dead weight.
    if(NOT ENABLE_CLANG_TIDY)
        message(STATUS
                "ENABLE_CLANG_TIDY is off; skipping kernel tidy target ${KERNEL_TIDY_NAME}")
        return()
    endif()
    if(WIN32)
        message(STATUS
                "clang-tidy is not run on Windows; skipping kernel tidy target ${KERNEL_TIDY_NAME}")
        return()
    endif()

    # setClangTidyVars() owns the HIP flags (the ROCm include directory in particular), so
    # take them from there rather than working them out again here.
    setclangtidyvars()
    if(NOT CLANG_TIDY_EXE)
        message(WARNING
                "clang-tidy not found. The '${KERNEL_TIDY_NAME}' target will not be available.")
        return()
    endif()

    get_filename_component(KERNEL_TIDY_PRELUDE "${KERNEL_TIDY_PRELUDE}" ABSOLUTE)
    set(_tidy_config "${PROJECT_SOURCE_DIR}/.clang-tidy")

    # Absolute, de-duplicated list of the directories the kernels include each other from.
    set(_kernel_include_flags "")
    set(_kernel_files "")
    foreach(_kernel_file IN LISTS KERNEL_TIDY_FILES)
        get_filename_component(_kernel_file "${_kernel_file}" ABSOLUTE)
        if(NOT EXISTS "${_kernel_file}")
            message(FATAL_ERROR "add_kernel_tidy_target: ${_kernel_file} does not exist.")
        endif()
        list(APPEND _kernel_files "${_kernel_file}")
        get_filename_component(_kernel_dir "${_kernel_file}" DIRECTORY)
        list(APPEND _kernel_include_flags "-I${_kernel_dir}")
    endforeach()
    list(REMOVE_DUPLICATES _kernel_files)
    list(REMOVE_DUPLICATES _kernel_include_flags)

    # Flags after `--` replace the compile database, which has no entry for these files.
    set(_kernel_tidy_compiler_flags
        -x hip
        --offload-host-only
        -std=c++${CMAKE_CXX_STANDARD}
        -include "${KERNEL_TIDY_PRELUDE}"
        ${_kernel_include_flags})

    set(_stamp_dir "${CMAKE_CURRENT_BINARY_DIR}/kernel_tidy")
    set(_stamps "")
    foreach(_kernel_file IN LISTS _kernel_files)
        # Stamp names follow the kernel's path relative to the engine, not its bare
        # filename: two kernels in different subdirectories may share a name, and a shared
        # stamp would report one of them as checked when only the other ran.
        file(RELATIVE_PATH _kernel_relative "${CMAKE_CURRENT_SOURCE_DIR}" "${_kernel_file}")
        string(REGEX REPLACE "[^A-Za-z0-9]" "_" _stamp_token "${_kernel_relative}")
        set(_stamp "${_stamp_dir}/${_stamp_token}.stamp")

        add_custom_command(
            OUTPUT ${_stamp}
            COMMAND ${CLANG_TIDY_EXE} -config-file=${_tidy_config} --quiet
                    ${CLANG_TIDY_HIP_ARGS} ${_kernel_file} -- ${_kernel_tidy_compiler_flags}
            COMMAND ${CMAKE_COMMAND} -E touch ${_stamp}
            DEPENDS ${_kernel_file} ${KERNEL_TIDY_PRELUDE} ${_tidy_config}
            COMMENT "Running clang-tidy on embedded kernel ${_kernel_relative}"
            VERBATIM
        )
        list(APPEND _stamps ${_stamp})
    endforeach()

    add_custom_target(${KERNEL_TIDY_NAME} DEPENDS ${_stamps})

    set_property(GLOBAL APPEND PROPERTY KERNELTIDY_TARGETS ${KERNEL_TIDY_NAME})
endfunction()

# hip_kernel_provider_tidy_dependencies()
#   Make every `tidy` target built so far also run the kernel tidy targets.
#
#   run-clang-tidy cannot reach the kernels on its own: they are not in
#   compile_commands.json, and adding them to it is exactly the build-graph weight this
#   module avoids. Depending on the targets is what puts them in a `tidy` run. Call this
#   after add_clang_tidy_custom_target(), once the `tidy` targets exist.
function(hip_kernel_provider_tidy_dependencies)
    get_property(_kernel_tidy_targets GLOBAL PROPERTY KERNELTIDY_TARGETS)
    if(NOT _kernel_tidy_targets)
        return()
    endif()

    foreach(_tidy_target IN ITEMS tidy tidy-cxx ${PROJECT_NAME}_tidy ${PROJECT_NAME}_tidy-cxx)
        if(TARGET ${_tidy_target})
            add_dependencies(${_tidy_target} ${_kernel_tidy_targets})
        endif()
    endforeach()
endfunction()
