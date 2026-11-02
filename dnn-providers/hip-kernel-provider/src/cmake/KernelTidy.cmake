# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(${PROJECT_SOURCE_DIR}/../cmake/CheckToolVersion.cmake)

# - This module compiles the kernel sources that KernelEmbedding.cmake inlines into a
# generated translation unit, so clang-tidy checks them as part of a normal build.
#
# An embedded kernel is otherwise never compiled by the build: embed_kernel_sources()
# reads it at configure time and pastes its text into a raw string literal. The object
# library created below is intentionally not linked; its compile commands give
# clang-tidy the hipRTC-compatible context and let the compiler track included headers.

# List of kernel tidy targets created
define_property(GLOBAL PROPERTY KERNELTIDY_TARGETS)


# hiprtc_runtime_header(<out-var>)
#
#   Extract the hipRTC pre-include header from libhiprtc-builtins into the build tree
#   and set <out-var> to its path. hipRTC embeds the header its online compiler prepends
#   to every program and exports it, with its length, as data symbols. Dumping it is how
#   clang-tidy sees the same declarations the real compile sees, without checking in a
#   copy that goes stale on every ROCm bump.
#
#   Linux only for now: kernel clang-tidy does not run on Windows, and whether the Windows
#   hipRTC builtins library exports the same symbols is unverified.
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
                 DOC "hipRTC builtins library, which exports the __hipRTC_header text")
    find_package(Python3 QUIET COMPONENTS Interpreter)

    if(NOT HIPRTC_BUILTINS_LIBRARY OR NOT Python3_Interpreter_FOUND)
        message(WARNING "hipRTC builtins library or Python 3 interpreter not found; "
                        "the hipRTC runtime header will not be generated.")
        return()
    endif()

    set(_header "${CMAKE_BINARY_DIR}/hiprtc/hiprtc_runtime.h")
    set(_dump_script "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/dump_hiprtc_header.py")

    # One generator for the whole build: the header is identical for every consumer, and a
    # second OUTPUT rule for the same file is an error.
    if(NOT TARGET hiprtc_runtime_header)
        add_custom_command(
            OUTPUT ${_header}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/hiprtc"
            COMMAND ${Python3_EXECUTABLE} ${_dump_script} ${HIPRTC_BUILTINS_LIBRARY} ${_header}
            DEPENDS ${HIPRTC_BUILTINS_LIBRARY} ${_dump_script}
            COMMENT "Extracting hipRTC runtime header from ${HIPRTC_BUILTINS_LIBRARY}"
            VERBATIM)
        add_custom_target(hiprtc_runtime_header
                          DEPENDS ${_header}
                          COMMENT "Preparing the hipRTC runtime header for kernel clang-tidy")
    endif()

    set(${out_var} "${_header}" PARENT_SCOPE)
endfunction()

# add_kernel_tidy_target(NAME <target> PRELUDE <header> FILES <kernel>...)
#
#   Create an unlinked OBJECT library that compiles the embedded kernel translation units
#   and runs clang-tidy on each one. The library is part of the default build, and every
#   source depends on the headers listed in FILES so that header changes re-check kernels.
#   Requires HIPRTC_RUNTIME_HEADER (see hiprtc_runtime_header()); no target is created
#   when it is unset.
#
# Keep validation and compiler flags together.
# cmake-lint: disable=R0912,R0915
function(add_kernel_tidy_target)
    set(options "")
    set(oneValueArgs NAME PRELUDE)
    set(multiValueArgs FILES COMPILE_DEFINITIONS)
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

    # Without the hipRTC pre-include header the kernels are missing the declarations the
    # real hipRTC compile gives them, so clang-tidy would drown in bogus diagnostics
    # instead of reporting anything useful. Skip the check rather than report noise.
    if(NOT HIPRTC_RUNTIME_HEADER)
        message(WARNING
                "hipRTC runtime header not available; skipping kernel tidy target "
                "${KERNEL_TIDY_NAME}")
        return()
    endif()

    # The kernels are device code built at runtime by hipRTC, so they are checked with the
    # clang-tidy from the ROCm toolchain rather than the image's: the hipRTC pre-include
    # header below comes from the same ROCm install, and only a matching clang-tidy can be
    # relied on to parse it. CLANG_TIDY_EXE deliberately is not reused here -- it is the
    # host C++ one, an older LLVM.
    findAndCheckRocmClangTidy()
    if(NOT ROCM_CLANG_TIDY_EXE)
        message(WARNING
                "ROCm clang-tidy not found. The '${KERNEL_TIDY_NAME}' target will not be available.")
        return()
    endif()
    get_filename_component(KERNEL_TIDY_PRELUDE "${KERNEL_TIDY_PRELUDE}" ABSOLUTE)
    get_filename_component(_tidy_config "${PROJECT_SOURCE_DIR}/.clang-tidy" ABSOLUTE)
    if(NOT EXISTS "${_tidy_config}")
        message(FATAL_ERROR
                "add_kernel_tidy_target: clang-tidy config ${_tidy_config} does not exist.")
    endif()

    # Absolute, de-duplicated list of sources and directories they include each other from.
    set(_kernel_include_dirs "")
    set(_kernel_sources "")
    set(_kernel_headers "")
    foreach(_kernel_file IN LISTS KERNEL_TIDY_FILES)
        get_filename_component(_kernel_file "${_kernel_file}" ABSOLUTE)
        if(NOT EXISTS "${_kernel_file}")
            message(FATAL_ERROR "add_kernel_tidy_target: ${_kernel_file} does not exist.")
        endif()
        get_filename_component(_kernel_dir "${_kernel_file}" DIRECTORY)
        list(APPEND _kernel_include_dirs "${_kernel_dir}")
        get_filename_component(_kernel_extension "${_kernel_file}" EXT)
        if(_kernel_extension MATCHES "^\\.(cpp|cc|cxx)$")
            list(APPEND _kernel_sources "${_kernel_file}")
        else()
            list(APPEND _kernel_headers "${_kernel_file}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _kernel_include_dirs)
    list(REMOVE_DUPLICATES _kernel_sources)
    list(REMOVE_DUPLICATES _kernel_headers)
    if(NOT _kernel_sources)
        message(FATAL_ERROR
                "add_kernel_tidy_target: FILES contains no compilable kernel sources.")
    endif()

    # These are the options needed to compile the sources outside of hipRTC.
    set(_kernel_tidy_compiler_flags
        -x hip
        --offload-device-only
        # Emit relocatable device code: a full device link needs the device libs that
        # -nogpulib omits (hipRTC links them at runtime).
        -fgpu-rdc
        -nogpulib
        -nogpuinc
        -D__HIPCC_RTC__
        # SHELL: keeps each -include paired with its path; CMake de-duplicates bare options.
        "SHELL:-include \"${KERNEL_TIDY_PRELUDE}\""
        "SHELL:-include \"${HIPRTC_RUNTIME_HEADER}\""
    )
    set(_gpu_targets "${GPU_TARGETS}")
    if(NOT _gpu_targets)
        set(_gpu_targets "${CMAKE_HIP_ARCHITECTURES}")
    endif()
    if(_gpu_targets)
        foreach(_gpu_target IN LISTS _gpu_targets)
            list(APPEND _kernel_tidy_compiler_flags "--offload-arch=${_gpu_target}")
        endforeach()
    endif()

    # The driver writes no depfile for --offload-device-only compiles, so depend on the
    # embedded headers explicitly to re-check kernels when they change.
    set_source_files_properties(${_kernel_sources} PROPERTIES
        OBJECT_DEPENDS
            "${KERNEL_TIDY_PRELUDE};${HIPRTC_RUNTIME_HEADER};${_tidy_config};${_kernel_headers}"
    )
    add_library(${KERNEL_TIDY_NAME} OBJECT ${_kernel_sources})
    set_target_properties(${KERNEL_TIDY_NAME} PROPERTIES
        CXX_STANDARD "${CMAKE_CXX_STANDARD}"
        CXX_STANDARD_REQUIRED ON
        CXX_CLANG_TIDY
            "${ROCM_CLANG_TIDY_EXE};-config-file=${_tidy_config};--quiet;--exclude-header-filter=hiprtc_runtime.h"
    )
    target_compile_options(${KERNEL_TIDY_NAME} PRIVATE ${_kernel_tidy_compiler_flags})
    target_compile_definitions(${KERNEL_TIDY_NAME} PRIVATE ${KERNEL_TIDY_COMPILE_DEFINITIONS})
    target_include_directories(${KERNEL_TIDY_NAME} PRIVATE ${_kernel_include_dirs})

    # Keep compilation objects separate from the provider; this library only exists to
    # expose normal compile commands and dependency tracking for the embedded sources.
    if(TARGET hiprtc_runtime_header)
        add_dependencies(${KERNEL_TIDY_NAME} hiprtc_runtime_header)
    endif()

    set_property(GLOBAL APPEND PROPERTY KERNELTIDY_TARGETS ${KERNEL_TIDY_NAME})
endfunction()

# hip_kernel_provider_tidy_dependencies()
#   Make every `tidy` target built so far also build the kernel-check object libraries.
#
#   The object libraries are part of the default build; depending on them here also makes
#   `tidy` run their clang-tidy checks. Call this after add_clang_tidy_custom_target(),
#   once the `tidy` targets exist.
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
