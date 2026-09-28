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
# add_kernel_tidy_target() closes that gap. It generates one wrapper translation unit per
# kernel file, each of which includes a prelude header and then the kernel itself, and
# compiles them in an EXCLUDE_FROM_ALL object library. That library is a real target, so
# its sources land in compile_commands.json and clang_tidy_check() applies to them; the
# kernel text is analysed through the #include, and diagnostics are reported against the
# kernel file rather than the wrapper.
#
# Two properties of the kernels make the wrapper necessary rather than optional:
#
# * They are hipRTC sources whose configuration arrives as -D options built by the host
#   plan (HIP_PLUGIN_*). Without those macros they do not parse, so the caller supplies a
#   PRELUDE header that defines a representative set.
# * Their #includes are flat ("VectorTypes.hpp"), because at runtime they resolve out of
#   the embedded include map instead of the filesystem. The target therefore gets every
#   directory that holds a registered kernel file on its include path.
#
# Headers are wrapped as well as sources. A kernel header that no kernel source includes
# would otherwise never be analysed, and .clang-tidy's HeaderFilterRegex only surfaces
# diagnostics from headers some translation unit actually reaches.
#
# Usage:
#   add_kernel_tidy_target(
#       NAME    hip_mlops_kernel_tidy
#       PRELUDE ${CMAKE_CURRENT_SOURCE_DIR}/tidy/KernelTidyPrelude.hpp
#       FILES   <the same list passed to add_kernels_for_embedding>)

# Tidy shim targets created so far, so the aggregate `tidy` targets can depend on them
# without each engine having to know what those targets are called.
define_property(GLOBAL PROPERTY KERNELTIDY_TARGETS)

# add_kernel_tidy_target(NAME <target> PRELUDE <header> FILES <kernel>...)
#
#   Create <target>, an EXCLUDE_FROM_ALL object library that compiles one wrapper per
#   entry of FILES so that clang-tidy analyses the embedded kernels.
#
#   The target is excluded from the default build because the wrappers exist for analysis
#   only: they duplicate no shipped code path, and nothing links their objects. Build it
#   explicitly, or through the `tidy` targets that hip_kernel_provider_tidy_dependencies()
#   wires up.
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

    # The shim only ever exists to run clang-tidy. Skipping it when clang-tidy is off
    # keeps the target list, and compile_commands.json, free of entries nothing consumes.
    if(NOT ENABLE_CLANG_TIDY)
        message(STATUS
                "ENABLE_CLANG_TIDY is off; skipping kernel tidy target ${KERNEL_TIDY_NAME}")
        return()
    endif()

    # Windows has no `tidy` target (see add_clang_tidy_custom_target) and the provider is
    # not analysed there, so the shim would be dead weight.
    if(WIN32)
        message(STATUS
                "clang-tidy is not run on Windows; skipping kernel tidy target ${KERNEL_TIDY_NAME}")
        return()
    endif()

    set(_wrapper_dir "${CMAKE_CURRENT_BINARY_DIR}/kernel_tidy")
    set(_wrapper_sources "")
    set(_kernel_include_dirs "")
    set(_seen_wrappers "")

    get_filename_component(KERNEL_TIDY_PRELUDE "${KERNEL_TIDY_PRELUDE}" ABSOLUTE)

    foreach(_kernel_file IN LISTS KERNEL_TIDY_FILES)
        get_filename_component(_kernel_file "${_kernel_file}" ABSOLUTE)
        if(NOT EXISTS "${_kernel_file}")
            message(FATAL_ERROR "add_kernel_tidy_target: ${_kernel_file} does not exist.")
        endif()

        get_filename_component(_kernel_dir "${_kernel_file}" DIRECTORY)
        list(APPEND _kernel_include_dirs "${_kernel_dir}")

        # Name the wrapper after the kernel's path relative to its engine, not after its
        # bare filename: two kernels in different subdirectories may share a name, and one
        # wrapper silently overwriting the other would drop a file from the analysis.
        file(RELATIVE_PATH _kernel_relative "${CMAKE_CURRENT_SOURCE_DIR}" "${_kernel_file}")
        string(REGEX REPLACE "[^A-Za-z0-9]" "_" _wrapper_token "${_kernel_relative}")
        list(FIND _seen_wrappers "${_wrapper_token}" _wrapper_seen_at)
        if(NOT _wrapper_seen_at EQUAL -1)
            message(FATAL_ERROR
                    "add_kernel_tidy_target: ${KERNEL_TIDY_NAME} was given ${_kernel_file} twice.")
        endif()
        list(APPEND _seen_wrappers "${_wrapper_token}")

        set(KERNEL_TIDY_WRAPPED_SOURCE "${_kernel_file}")
        set(KERNEL_TIDY_WRAPPED_PRELUDE "${KERNEL_TIDY_PRELUDE}")
        set(_wrapper_source "${_wrapper_dir}/${_wrapper_token}.cpp")
        configure_file(${PROJECT_SOURCE_DIR}/src/cmake/templates/kernel_tidy_wrapper.cpp.in
                       ${_wrapper_source} @ONLY)
        list(APPEND _wrapper_sources "${_wrapper_source}")
    endforeach()

    list(REMOVE_DUPLICATES _kernel_include_dirs)

    add_library(${KERNEL_TIDY_NAME} OBJECT EXCLUDE_FROM_ALL ${_wrapper_sources})

    # -x hip is what makes __global__, __shared__ and the block/thread builtins parse; the
    # wrappers are named .cpp so the rest of the build treats them as ordinary C++.
    # --offload-host-only keeps this to the host pass, which still parses and type-checks
    # device code but needs neither an --offload-arch nor the ROCm device libraries, so the
    # shim stays as cheap and as portable as the rest of this CXX-only project.
    target_compile_options(${KERNEL_TIDY_NAME} PRIVATE -x hip --offload-host-only)
    target_include_directories(${KERNEL_TIDY_NAME} PRIVATE ${_kernel_include_dirs})
    target_link_libraries(${KERNEL_TIDY_NAME} PRIVATE hip::host)
    set_target_properties(${KERNEL_TIDY_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    clang_tidy_check(${KERNEL_TIDY_NAME})

    set_property(GLOBAL APPEND PROPERTY KERNELTIDY_TARGETS ${KERNEL_TIDY_NAME})
endfunction()

# hip_kernel_provider_tidy_dependencies()
#   Make every `tidy` target built so far also build the kernel tidy shims.
#
#   run-clang-tidy walks compile_commands.json, which already lists the wrappers, but it
#   only reaches them if the shim was configured; depending on the targets also means a
#   `tidy` run fails loudly when a wrapper stops compiling instead of quietly analysing
#   less code. Call this after add_clang_tidy_custom_target(), once the `tidy` targets
#   exist.
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
