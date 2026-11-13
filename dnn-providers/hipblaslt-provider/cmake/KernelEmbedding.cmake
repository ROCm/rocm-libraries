# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Embeds HIP kernel sources in a target as string literals, for run-time compilation
# with HIPRTC (compilation/Program.cpp). Ported from hip-kernel-provider's
# src/cmake/KernelEmbedding.cmake, under a provider-specific name: CMake functions are
# global, and a superbuild configures both providers in one CMake run.

# hipblaslt_plugin_embed_kernels(TARGET <target> OUTPUT_DIR <dir> FILES <file>...)
# Writes <dir>/kernel_sources.hpp and <dir>/kernel_sources.cpp, which hold the kernel
# sources and the headers (.h, .hpp) they include, and adds them to <target>. Program
# looks a source up by its file name and offers every header to it under its file
# name. Editing one of the files reconfigures the build.
function(hipblaslt_plugin_embed_kernels)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "TARGET;OUTPUT_DIR" "FILES")
    if(NOT ARG_TARGET OR NOT ARG_OUTPUT_DIR OR NOT ARG_FILES)
        message(FATAL_ERROR "hipblaslt_plugin_embed_kernels needs a TARGET, an OUTPUT_DIR and FILES")
    endif()

    set(KERNEL_DEFINITIONS "")
    set(KERNEL_MAP_ENTRIES "")
    set(HEADER_MAP_ENTRIES "")
    foreach(_file IN LISTS ARG_FILES)
        file(READ "${_file}" _content)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_file}")
        get_filename_component(_name "${_file}" NAME)
        get_filename_component(_ext "${_file}" LAST_EXT)
        string(MAKE_C_IDENTIFIER "${_name}_SOURCE" _id)
        string(TOUPPER "${_id}" _id)
        string(APPEND KERNEL_DEFINITIONS
               "const char* const ${_id} = R\"KERNEL_SOURCE(\n" "${_content}" "\n)KERNEL_SOURCE\";\n\n")
        if(_ext STREQUAL ".h" OR _ext STREQUAL ".hpp")
            string(APPEND HEADER_MAP_ENTRIES "        {\"${_name}\", ${_id}},\n")
        else()
            string(APPEND KERNEL_MAP_ENTRIES "        {\"${_name}\", ${_id}},\n")
        endif()
    endforeach()

    set(_templates "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../templates")
    configure_file("${_templates}/kernel_sources.hpp.in" "${ARG_OUTPUT_DIR}/kernel_sources.hpp" @ONLY)
    configure_file("${_templates}/kernel_sources.cpp.in" "${ARG_OUTPUT_DIR}/kernel_sources.cpp" @ONLY)
    # Kernel sources exceed the length -Wpedantic allows for a string literal.
    set_source_files_properties("${ARG_OUTPUT_DIR}/kernel_sources.cpp"
                                PROPERTIES COMPILE_OPTIONS "-Wno-overlength-strings")
    target_sources(${ARG_TARGET} PRIVATE "${ARG_OUTPUT_DIR}/kernel_sources.cpp")
    target_include_directories(${ARG_TARGET} PRIVATE "${ARG_OUTPUT_DIR}")
endfunction()
