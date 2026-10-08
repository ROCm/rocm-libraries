# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(${PROJECT_SOURCE_DIR}/../cmake/CheckToolVersion.cmake)

# - This module makes the kernel sources that KernelEmbedding.cmake inlines into a
# generated translation unit visible to the compiler, and so to clang-tidy.
#
# An embedded kernel is never compiled by the build: embed_kernel_sources() reads it at
# configure time and pastes its text into a raw string literal, so it has no entry in
# compile_commands.json. add_kernel_tidy_library() closes the gap by compiling the kernels
# into an object library that nothing links, using the options hipRTC compiles them with at
# runtime. The library is part of `all`, so every build checks the kernels, and because a
# real compile drives the check the generated depfiles re-check a kernel when a header it
# includes changes.

# List of kernel tidy libraries created
define_property(GLOBAL PROPERTY KERNELTIDY_TARGETS)


# hiprtc_runtime_header(<out-var>)
#
#   Extract the hipRTC pre-include header from libhiprtc-builtins.so into the build tree
#   and set <out-var> to its path. hipRTC embeds the header its online compiler prepends
#   to every program in an ELF section. Dumping it is how clang-tidy sees the same
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

    # The dumped section ends in a line of raw bytes (the section's terminator, which is
    # not part of the header text) that clang rejects, so it is stripped below. sed edits
    # the file in place, which keeps this to one extra command and no temporary file.
    find_program(HIPRTC_SED_EXE NAMES sed)

    if(NOT HIPRTC_BUILTINS_LIBRARY OR NOT HIPRTC_OBJCOPY_EXE OR NOT HIPRTC_SED_EXE)
        message(WARNING "libhiprtc-builtins.so, sed, or objcopy not found; "
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
            COMMAND ${HIPRTC_SED_EXE} -i "$d" ${_header}
            DEPENDS ${HIPRTC_BUILTINS_LIBRARY}
            COMMENT "Extracting hipRTC runtime header from ${HIPRTC_BUILTINS_LIBRARY}"
            VERBATIM)
        add_custom_target(hiprtc_runtime_header
                          DEPENDS ${_header}
                          COMMENT "Preparing the hipRTC runtime header for kernel clang-tidy")
    endif()

    set(${out_var} "${_header}" PARENT_SCOPE)
endfunction()


# add_kernel_tidy_library(NAME <target> PRELUDE <header> FILES <kernel>...)
#
#   Create <target>, an OBJECT library that compiles the embedded kernels in FILES with the
#   same options hipRTC compiles them with at runtime, and runs clang-tidy on them through
#   the target's CXX_CLANG_TIDY launcher.
#
#   Nothing links the library: its only purposes are to put the kernels into
#   compile_commands.json and to give clang-tidy a translation unit per kernel. It is part
#   of `all` on purpose, so a normal build checks the kernels instead of relying on
#   somebody remembering to build `tidy`. The compiler writes a depfile for every kernel,
#   so editing a header a kernel includes re-checks the kernel.
#
#   FILES may list headers as well as kernels. Headers are attached to the target for IDEs
#   but not compiled on their own; they are checked as part of the kernels that include
#   them, which is also where their templates are instantiated.
#
#   Requires HIPRTC_RUNTIME_HEADER (see hiprtc_runtime_header()); no target is created
#   when it is unset.
#
# Keep validation, compiler flags, and target setup together.
# cmake-lint: disable=R0915
function(add_kernel_tidy_library)
    set(options "")
    set(oneValueArgs NAME PRELUDE)
    set(multiValueArgs FILES)
    cmake_parse_arguments(PARSE_ARGV 0 KERNEL_TIDY "${options}" "${oneValueArgs}"
                          "${multiValueArgs}")

    if(NOT KERNEL_TIDY_NAME)
        message(FATAL_ERROR "add_kernel_tidy_library called without a NAME!")
    endif()
    if(NOT KERNEL_TIDY_FILES)
        message(FATAL_ERROR "add_kernel_tidy_library called without any FILES!")
    endif()
    if(NOT KERNEL_TIDY_PRELUDE)
        message(FATAL_ERROR "add_kernel_tidy_library called without a PRELUDE!")
    endif()
    if(NOT EXISTS "${KERNEL_TIDY_PRELUDE}")
        message(FATAL_ERROR
                "add_kernel_tidy_library: PRELUDE ${KERNEL_TIDY_PRELUDE} does not exist.")
    endif()

    # The library exists only to be clang-tidied, and Windows has no `tidy` target at all
    # (see add_clang_tidy_custom_target), so in both cases it would be dead weight.
    if(NOT ENABLE_CLANG_TIDY)
        message(STATUS
                "ENABLE_CLANG_TIDY is off; skipping kernel tidy library ${KERNEL_TIDY_NAME}")
        return()
    endif()
    if(WIN32)
        message(STATUS
                "clang-tidy is not run on Windows; skipping kernel tidy library ${KERNEL_TIDY_NAME}")
        return()
    endif()

    # The kernels are device code, so the C++ compiler has to be the ROCm clang++ the
    # toolchain file selects (see cmake/ClangToolChain.cmake). A non-clang compiler cannot
    # be handed `-x hip`, and there is no per-target compiler in CMake, so there is nothing
    # to fall back to.
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(WARNING
                "CXX compiler is ${CMAKE_CXX_COMPILER_ID}, not Clang; skipping kernel tidy "
                "library ${KERNEL_TIDY_NAME}")
        return()
    endif()

    # Without the hipRTC pre-include header the kernels are missing the declarations the
    # real hipRTC compile gives them, so the compile would fail and clang-tidy would drown
    # in bogus diagnostics. Skip the check rather than report noise.
    if(NOT HIPRTC_RUNTIME_HEADER)
        message(WARNING
                "hipRTC runtime header not available; skipping kernel tidy library "
                "${KERNEL_TIDY_NAME}")
        return()
    endif()

    # The kernels are checked with the clang-tidy from the ROCm toolchain rather than the
    # image's: the hipRTC pre-include header comes from the same ROCm install, and only a
    # matching clang-tidy can be relied on to parse it. CLANG_TIDY_EXE deliberately is not
    # reused here -- it is the host C++ one, an older LLVM.
    findAndCheckRocmClangTidy()
    if(NOT ROCM_CLANG_TIDY_EXE)
        message(WARNING
                "ROCm clang-tidy not found. The '${KERNEL_TIDY_NAME}' library will not be built.")
        return()
    endif()

    get_filename_component(KERNEL_TIDY_PRELUDE "${KERNEL_TIDY_PRELUDE}" ABSOLUTE)
    set(_tidy_config "${PROJECT_SOURCE_DIR}/.clang-tidy")
    if(NOT EXISTS "${_tidy_config}")
        message(FATAL_ERROR "add_kernel_tidy_library: ${_tidy_config} does not exist.")
    endif()

    # Kernels become translation units; headers ride along as non-compiled sources. The
    # include directories are the directories the kernels include each other from.
    set(_kernel_sources "")
    set(_kernel_headers "")
    set(_kernel_include_dirs "")
    foreach(_kernel_file IN LISTS KERNEL_TIDY_FILES)
        get_filename_component(_kernel_file "${_kernel_file}" ABSOLUTE)
        if(NOT EXISTS "${_kernel_file}")
            message(FATAL_ERROR "add_kernel_tidy_library: ${_kernel_file} does not exist.")
        endif()
        get_filename_component(_kernel_extension "${_kernel_file}" LAST_EXT)
        if(_kernel_extension STREQUAL ".cpp")
            list(APPEND _kernel_sources "${_kernel_file}")
        else()
            list(APPEND _kernel_headers "${_kernel_file}")
        endif()
        get_filename_component(_kernel_dir "${_kernel_file}" DIRECTORY)
        list(APPEND _kernel_include_dirs "${_kernel_dir}")
    endforeach()
    list(REMOVE_DUPLICATES _kernel_sources)
    list(REMOVE_DUPLICATES _kernel_headers)
    list(REMOVE_DUPLICATES _kernel_include_dirs)

    if(NOT _kernel_sources)
        message(FATAL_ERROR "add_kernel_tidy_library: FILES contains no .cpp kernels.")
    endif()

    # One arch is enough: the point is a parse, not a fat binary, and every extra arch is
    # another device compile of the same source.
    if(GPU_TARGETS)
        list(GET GPU_TARGETS 0 _kernel_tidy_arch)
    else()
        set(_kernel_tidy_arch "gfx942")
    endif()

    add_library(${KERNEL_TIDY_NAME} OBJECT ${_kernel_sources} ${_kernel_headers})
    set_source_files_properties(${_kernel_headers} PROPERTIES HEADER_FILE_ONLY ON)
    target_include_directories(${KERNEL_TIDY_NAME} PRIVATE ${_kernel_include_dirs})
    target_compile_options(
        ${KERNEL_TIDY_NAME}
        PRIVATE
            # Match what hipRTC does at runtime: device-only compilation of a source that
            # sees neither the HIP headers nor the device library, with the pre-include
            # header supplying the declarations and the prelude the kernel macros that
            # the plans would otherwise define.
            -x hip
            --offload-device-only
            --offload-arch=${_kernel_tidy_arch}
            -nogpulib
            -nogpuinc
            -D__HIPCC_RTC__
            # SHELL: keeps CMake from de-duplicating the second -include away: it
            # collapses repeated options, and the two headers are not interchangeable.
            "SHELL:-include ${KERNEL_TIDY_PRELUDE}"
            "SHELL:-include ${HIPRTC_RUNTIME_HEADER}")

    # The per-target property overrides CMAKE_CXX_CLANG_TIDY, which CI sets globally to the
    # host clang-tidy; that binary is an older LLVM and cannot parse the hipRTC header.
    set_target_properties(
        ${KERNEL_TIDY_NAME}
        PROPERTIES CXX_CLANG_TIDY
                   "${ROCM_CLANG_TIDY_EXE};-config-file=${_tidy_config};--quiet;--exclude-header-filter=hiprtc_runtime.h"
                   # One kernel per translation unit: a unity build would hide the
                   # per-kernel diagnostics behind whichever kernel it merged them into.
                   UNITY_BUILD OFF
                   # Spelled out because being in `all` is the point of the library.
                   EXCLUDE_FROM_ALL OFF)

    # The pre-include header is generated in another directory scope, so the dependency has
    # to be spelled out for generators that do not infer it from the compile line.
    if(TARGET hiprtc_runtime_header)
        add_dependencies(${KERNEL_TIDY_NAME} hiprtc_runtime_header)
    endif()

    # The kernels are now in compile_commands.json, which is what makes the compile-driven
    # check work -- but it also puts them in reach of run-clang-tidy, and that uses the host
    # clang-tidy, an older LLVM that cannot parse the hipRTC header. Keep the `tidy` targets
    # off them; the CXX_CLANG_TIDY launcher above is what checks them.
    foreach(_kernel_dir IN LISTS _kernel_include_dirs)
        string(REGEX REPLACE "([][.^$+*?()|{}\\])" "\\\\\\1" _kernel_dir "${_kernel_dir}")
        set_property(GLOBAL APPEND PROPERTY CLANG_TIDY_SOURCE_EXCLUDE_PATTERNS "${_kernel_dir}")
    endforeach()

    set_property(GLOBAL APPEND PROPERTY KERNELTIDY_TARGETS ${KERNEL_TIDY_NAME})
endfunction()

# hip_kernel_provider_tidy_dependencies()
#   Make every `tidy` target built so far also build the kernel tidy libraries.
#
#   The libraries are in `all`, so a normal build already checks the kernels; this only
#   makes `ninja tidy` on a clean tree check them too. run-clang-tidy cannot do it instead:
#   it uses the host clang-tidy, which is an older LLVM than the hipRTC header needs, which
#   is why add_kernel_tidy_library() excludes the kernels from its source filter. Call this
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
