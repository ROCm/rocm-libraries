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

# add_kernel_tidy_target() closes the gap by createing a custom
# target that invokes clang-tidy on the kernel files directly, passing the compiler flags
# they need

# TODO the target is not part of ALL and only uns when asked for.

# Usage:
#   add_kernel_tidy_target(
#       NAME    hip_mlops_kernel_tidy
#       FILES   <the same list passed to add_kernels_for_embedding>)

# List of kernel tidy targets created
define_property(GLOBAL PROPERTY KERNELTIDY_TARGETS)

# hiprtc_runtime_header(<out-var>)
#
#   Extract the hipRTC pre-include header from libhiprtc-builtins.so into the build tree
#   and set <out-var> to its path. hipRTC embeds the header its online compiler prepends
#   to every program in an ELF section; dumping it is how clang-tidy sees the same
#   declarations the real compile sees, without checking in a copy that goes stale on
#   every ROCm bump.
#
#   Linux only: the header lives in an ELF section, so there is nothing to dump elsewhere.
#   <out-var> is left empty when the header cannot be produced; callers must handle that.
function(hiprtc_runtime_header out_var)
    set(${out_var} "" PARENT_SCOPE)

    if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
        message(STATUS "hipRTC runtime header can only be extracted on Linux; skipping")
        return()
    endif()

    find_library(HIPRTC_BUILTINS_LIBRARY
                 NAMES hiprtc-builtins
                 HINTS "${ROCM_PATH}/lib" /opt/rocm/lib
                 DOC "libhiprtc-builtins.so, which carries the .hipRTC_header section")

    # The ROCm llvm-objcopy is the one matched to the .so; CMAKE_OBJCOPY (GNU binutils)
    # understands --dump-section too and is a fine fallback.
    find_program(HIPRTC_OBJCOPY_EXE
                 NAMES llvm-objcopy objcopy
                 HINTS "${ROCM_PATH}/llvm/bin" /opt/rocm/llvm/bin)
    if(NOT HIPRTC_OBJCOPY_EXE)
        set(HIPRTC_OBJCOPY_EXE "${CMAKE_OBJCOPY}")
    endif()

    if(NOT HIPRTC_BUILTINS_LIBRARY OR NOT HIPRTC_OBJCOPY_EXE)
        message(WARNING "libhiprtc-builtins.so or objcopy not found; "
                        "the hipRTC runtime header will not be generated.")
        return()
    endif()

    set(_header "${CMAKE_BINARY_DIR}/hiprtc/hiprtc_runtime.h")

    # One generator for the whole build: the header is identical for every consumer, and a
    # second OUTPUT rule for the same file is an error.
    if(NOT TARGET hiprtc_runtime_header)
        add_custom_command(
            OUTPUT ${_header}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/hiprtc"
            # objcopy always wants an output object; /dev/null discards it, we only want
            # the dumped section.
            COMMAND ${HIPRTC_OBJCOPY_EXE}
                    --dump-section .hipRTC_header=${_header}
                    ${HIPRTC_BUILTINS_LIBRARY} /dev/null
            DEPENDS ${HIPRTC_BUILTINS_LIBRARY}
            COMMENT "Extracting hipRTC runtime header from ${HIPRTC_BUILTINS_LIBRARY}"
            VERBATIM)
        add_custom_target(hiprtc_runtime_header DEPENDS ${_header})
    endif()

    set(${out_var} "${_header}" PARENT_SCOPE)
endfunction()


# add_kernel_tidy_target(NAME <target>  PRELUDE <header> FILES <kernel>...)
#
#   Create <target>, a custom target that runs clang-tidy over the embedded kernels in
#   FILES. Builds explicitly, or through the `tidy` targets that  hip_kernel_provider_tidy_dependencies()
#   wires up.
#
#   Each file gets its own command, so the generator runs them in parallel and re-checks
#   only what changed: a per-file stamp under the build tree records the last successful
#   run, and a file is re-checked when it, the prelude or the .clang-tidy config is newer.
#
#   Requires HIPRTC_RUNTIME_HEADER (see hiprtc_runtime_header()); no target is created
#   when it is unset.
function(add_kernel_tidy_target)
    set(options "")
    set(oneValueArgs NAME PRELUDE)
    set(multiValueArgs FILES)
    cmake_parse_arguments(PARSE_ARGV 0 KERNEL_TIDY "${options}" "${oneValueArgs}"
                          "${multiValueArgs}")

    if(NOT KERNEL_TIDY_NAME)
        message(FATAL_ERROR "add_kernel_tidy_target called without a NAME!")
    endif()

    if(NOT KERNEL_TIDY_FILES)
        message(FATAL_ERROR "add_kernel_tidy_target called without any FILES!")
    endif()
        if(NOT KERNEL_TIDY_PRELUDE)
        message(FATAL_ERROR "add_kernel_tidy_target called without a PRELUDE!")
    endif()
    if(NOT EXISTS "${KERNEL_TIDY_PRELUDE}")
        message(FATAL_ERROR "add_kernel_tidy_target: PRELUDE ${KERNEL_TIDY_PRELUDE} does not exist.")
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

    # Without the hipRTC pre-include header the kernels are missing the declarations the
    # real hipRTC compile gives them, so clang-tidy would drown in bogus diagnostics
    # instead of reporting anything useful. Skip the check rather than report noise.
    if(NOT HIPRTC_RUNTIME_HEADER)
        message(WARNING
                "hipRTC runtime header not available; skipping kernel tidy target "
                "${KERNEL_TIDY_NAME}")
        return()
    endif()

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
        --offload-device-only
        -std=c++${CMAKE_CXX_STANDARD}
        -nogpulib
        -nogpuinc
        -D__HIPCC_RTC__
        -include "${KERNEL_TIDY_PRELUDE}"
        -include "${HIPRTC_RUNTIME_HEADER}"
        ${_kernel_include_flags}
    )
    
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
            COMMAND
                    /opt/rocm/llvm/bin/clang-tidy
                    -config-file=${_tidy_config} --quiet
                    --exclude-header-filter=hiprtc_runtime* 
                    ${CLANG_TIDY_HIP_ARGS}  # TODO replace with exported header
                    ${_kernel_file} 
                    -- ${_kernel_tidy_compiler_flags}
            COMMAND ${CMAKE_COMMAND} -E touch ${_stamp}
            DEPENDS ${_kernel_file}  ${KERNEL_TIDY_PRELUDE} ${_tidy_config}
                    ${HIPRTC_RUNTIME_HEADER}
            COMMENT "Running clang-tidy on embedded kernel ${_kernel_relative}"
            VERBATIM
        )
        list(APPEND _stamps ${_stamp})
    endforeach()

    add_custom_target(${KERNEL_TIDY_NAME} DEPENDS ${_stamps})

    # The header is generated in another directory scope, so the dependency has to be
    # spelled out for generators that do not infer it from the stamp's DEPENDS.
    if(TARGET hiprtc_runtime_header)
        add_dependencies(${KERNEL_TIDY_NAME} hiprtc_runtime_header)
    endif()

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
