# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# - This module is responsible for inlining the kernels into a single source file,
# from which kernels are retrieved when running any of them from HipProgram.
# It defines a global property that contains a list of all kernel source files
# from any of the available engines.
#
# Usage:
# within the engine that contains HIP kernel kernels,
# call the function add_kernels_for_embedding with arguments being the paths
# to the kernels within the directory (see hip_mlops_engine for an example)
#

# Keep a list of kernel files to be inlined
define_property(GLOBAL PROPERTY KERNELEMBEDDING_KERNEL_FILES)
# The lookup key each file is registered under, appended in lockstep with
# KERNELEMBEDDING_KERNEL_FILES so index N of one names index N of the other.
define_property(GLOBAL PROPERTY KERNELEMBEDDING_KERNEL_KEYS)
# CMake ignores attempts to redefine an already defined property,
# so the value in this property won't be overwritten when including
# this module anywhere else in the repo

# Function to mark kernel source files for embedding in the single source file
#
# KEYS gives the lookup key of each file, in the order FILES names them. A
# kernel source resolves through getKernelSrc() under its key, so a descriptor
# that names a set-root-relative path passes that path here. Without KEYS each
# file takes its bare filename as its key.
function(add_kernels_for_embedding)
    set(options "")
    set(oneValueArgs TARGET)
    set(multiValueArgs FILES KEYS)
    cmake_parse_arguments(PARSE_ARGV 0 ADD_KERNELS "${options}" "${oneValueArgs}" "${multiValueArgs}")

    # Validation checks
    if(NOT ADD_KERNELS_TARGET)
        message(FATAL_ERROR "add_kernels_for_embedding called without a TARGET!")
    endif()
    if(NOT TARGET ${ADD_KERNELS_TARGET})
        message(FATAL_ERROR "add_kernels_for_embedding: The target ${ADD_KERNELS_TARGET} does not exist yet.
                Please make sure to call add_kernels_for_embedding after the target is created.")
    endif()
    if(NOT ADD_KERNELS_FILES)
        message(FATAL_ERROR "add_kernels_for_embedding called without any FILES!")
    endif()

    list(LENGTH ADD_KERNELS_FILES _kernel_file_count)
    list(LENGTH ADD_KERNELS_KEYS _kernel_key_count)
    if(ADD_KERNELS_KEYS AND NOT _kernel_key_count EQUAL _kernel_file_count)
        message(FATAL_ERROR "add_kernels_for_embedding: ${_kernel_key_count} KEYS for "
                "${_kernel_file_count} FILES on target ${ADD_KERNELS_TARGET}. "
                "Give one key per file, in the same order.")
    endif()

    # Add kernel file paths and their keys to the target properties
    set(_kernel_index 0)
    foreach(KERNEL_FILE IN LISTS ADD_KERNELS_FILES)
        if(ADD_KERNELS_KEYS)
            list(GET ADD_KERNELS_KEYS ${_kernel_index} KERNEL_KEY)
        else()
            get_filename_component(KERNEL_KEY "${KERNEL_FILE}" NAME)
        endif()
        math(EXPR _kernel_index "${_kernel_index} + 1")
        set_property(TARGET ${ADD_KERNELS_TARGET} APPEND PROPERTY KERNELEMBEDDING_KERNEL_FILES ${KERNEL_FILE})
        set_property(TARGET ${ADD_KERNELS_TARGET} APPEND PROPERTY KERNELEMBEDDING_KERNEL_KEYS ${KERNEL_KEY})
    endforeach()
endfunction()

# _embed_one_kernel_source(<target> <file> <key> <content_variable>)
#   Add one kernel source to the generated translation unit of <target>.
#
#   <key> names the source in the map getKernelSrc() reads, and the C++ object name comes
#   from it. Two sources under one key insert once and drop the rest with no diagnostic,
#   so a lookup would return another file's source. A repeat is fatal here instead.
#
#   Reads and returns the caller's KERNEL_DECLARATIONS, KERNEL_DEFINITIONS,
#   KERNEL_MAP_ENTRIES, SEEN_KERNEL_KEYS and SEEN_KERNEL_KEY_FILES. Pass the content as a
#   variable name, so a source that holds CMake syntax reaches the output unevaluated.
function(_embed_one_kernel_source EMBED_ONE_TARGET EMBED_ONE_FILE EMBED_ONE_KEY
         EMBED_ONE_CONTENT_VAR)
    list(FIND SEEN_KERNEL_KEYS "${EMBED_ONE_KEY}" _key_seen_at)
    if(NOT _key_seen_at EQUAL -1)
        list(GET SEEN_KERNEL_KEY_FILES ${_key_seen_at} _key_first_file)
        message(FATAL_ERROR
                "embed_kernel_sources: target ${EMBED_ONE_TARGET} registers two "
                "sources under the key '${EMBED_ONE_KEY}':\n"
                "  ${_key_first_file}\n"
                "  ${EMBED_ONE_FILE}\n"
                "Give each source its own key.")
    endif()
    list(APPEND SEEN_KERNEL_KEYS "${EMBED_ONE_KEY}")
    list(APPEND SEEN_KERNEL_KEY_FILES "${EMBED_ONE_FILE}")

    string(REGEX REPLACE "[^A-Za-z0-9]" "_" _key_token "${EMBED_ONE_KEY}")
    string(TOUPPER "${EMBED_ONE_TARGET}_${_key_token}" _source_var)

    string(APPEND KERNEL_DECLARATIONS "extern const char* const ${_source_var}_SOURCE;\n")
    string(APPEND KERNEL_DEFINITIONS "const char* const ${_source_var}_SOURCE = R\"KERNEL_SOURCE(\n")
    string(APPEND KERNEL_DEFINITIONS "${${EMBED_ONE_CONTENT_VAR}}")
    string(APPEND KERNEL_DEFINITIONS "\n)KERNEL_SOURCE\";\n\n")
    string(APPEND KERNEL_MAP_ENTRIES "        {\"${EMBED_ONE_KEY}\", ${_source_var}_SOURCE},\n")

    set(SEEN_KERNEL_KEYS "${SEEN_KERNEL_KEYS}" PARENT_SCOPE)
    set(SEEN_KERNEL_KEY_FILES "${SEEN_KERNEL_KEY_FILES}" PARENT_SCOPE)
    set(KERNEL_DECLARATIONS "${KERNEL_DECLARATIONS}" PARENT_SCOPE)
    set(KERNEL_DEFINITIONS "${KERNEL_DEFINITIONS}" PARENT_SCOPE)
    set(KERNEL_MAP_ENTRIES "${KERNEL_MAP_ENTRIES}" PARENT_SCOPE)
endfunction()

# _embed_one_kernel_header(<target> <file> <key> <object_name> <content_variable>)
#   Add one kernel header to the generated include table of <target>.
#
#   <key> names the header in the map GetKernelInc() reads. The map holds one entry per
#   key. A second header under one key drops, and a lookup returns the first file's text.
#   A repeat is fatal here instead.
#
#   Reads and returns the caller's HEADER_DECLARATIONS, HEADER_DEFINITIONS,
#   HEADER_MAP_ENTRIES, HEADER_FILENAMES, SEEN_HEADER_KEYS and SEEN_HEADER_KEY_FILES.
#   Pass the content as a variable name, so a header that holds CMake syntax reaches the
#   output unevaluated.
function(_embed_one_kernel_header EMBED_ONE_TARGET EMBED_ONE_FILE EMBED_ONE_KEY
         EMBED_ONE_OBJECT_NAME EMBED_ONE_CONTENT_VAR)
    list(FIND SEEN_HEADER_KEYS "${EMBED_ONE_KEY}" _header_seen_at)
    if(NOT _header_seen_at EQUAL -1)
        list(GET SEEN_HEADER_KEY_FILES ${_header_seen_at} _header_first_file)
        message(FATAL_ERROR
                "embed_kernel_sources: target ${EMBED_ONE_TARGET} registers two "
                "headers under the key '${EMBED_ONE_KEY}':\n"
                "  ${_header_first_file}\n"
                "  ${EMBED_ONE_FILE}\n"
                "Give each header its own filename.")
    endif()
    list(APPEND SEEN_HEADER_KEYS "${EMBED_ONE_KEY}")
    list(APPEND SEEN_HEADER_KEY_FILES "${EMBED_ONE_FILE}")

    string(APPEND HEADER_DECLARATIONS
           "extern const char* const ${EMBED_ONE_OBJECT_NAME}_SOURCE;\n")
    string(APPEND HEADER_DEFINITIONS
           "const char* const ${EMBED_ONE_OBJECT_NAME}_SOURCE = R\"KERNEL_SOURCE(\n")
    string(APPEND HEADER_DEFINITIONS "${${EMBED_ONE_CONTENT_VAR}}")
    string(APPEND HEADER_DEFINITIONS "\n)KERNEL_SOURCE\";\n\n")
    string(APPEND HEADER_MAP_ENTRIES
           "        {\"${EMBED_ONE_KEY}\", ${EMBED_ONE_OBJECT_NAME}_SOURCE},\n")
    string(APPEND HEADER_FILENAMES "        \"${EMBED_ONE_KEY}\",\n")

    set(SEEN_HEADER_KEYS "${SEEN_HEADER_KEYS}" PARENT_SCOPE)
    set(SEEN_HEADER_KEY_FILES "${SEEN_HEADER_KEY_FILES}" PARENT_SCOPE)
    set(HEADER_DECLARATIONS "${HEADER_DECLARATIONS}" PARENT_SCOPE)
    set(HEADER_DEFINITIONS "${HEADER_DEFINITIONS}" PARENT_SCOPE)
    set(HEADER_MAP_ENTRIES "${HEADER_MAP_ENTRIES}" PARENT_SCOPE)
    set(HEADER_FILENAMES "${HEADER_FILENAMES}" PARENT_SCOPE)
endfunction()

# _write_kernel_key_manifest(<file>)
#   Record what the enclosing embed_kernel_sources() call embeds, as one
#   'key<TAB>absolute file' pair per source, sorted by key.
#
#   A staged descriptor tree carries no kernel source, so this table is the only
#   evidence that a descriptor's source_file resolves at runtime. Reads the
#   caller's SEEN_KERNEL_KEYS and SEEN_KERNEL_KEY_FILES, which hold the sources
#   alone: getKernelSrc() never reaches a header.
#
#   Rewrites <file> on a change alone. The verification step depends on this file,
#   and an unconditional write reruns that step on every configure.
function(_write_kernel_key_manifest KEY_MANIFEST_FILE)
    set(_manifest_lines "")
    set(_manifest_index 0)
    foreach(_manifest_key IN LISTS SEEN_KERNEL_KEYS)
        list(GET SEEN_KERNEL_KEY_FILES ${_manifest_index} _manifest_file)
        math(EXPR _manifest_index "${_manifest_index} + 1")
        get_filename_component(_manifest_file "${_manifest_file}" ABSOLUTE)
        list(APPEND _manifest_lines "${_manifest_key}\t${_manifest_file}")
    endforeach()
    list(SORT _manifest_lines)

    set(_manifest_text "")
    foreach(_manifest_line IN LISTS _manifest_lines)
        string(APPEND _manifest_text "${_manifest_line}\n")
    endforeach()

    set(_manifest_stale TRUE)
    if(EXISTS "${KEY_MANIFEST_FILE}")
        file(READ "${KEY_MANIFEST_FILE}" _manifest_current)
        if("${_manifest_current}" STREQUAL "${_manifest_text}")
            set(_manifest_stale FALSE)
        endif()
    endif()
    if(_manifest_stale)
        file(WRITE "${KEY_MANIFEST_FILE}" "${_manifest_text}")
    endif()
endfunction()

# Function to embed kernel sources as C++ strings at configure time
function(embed_kernel_sources)
    set(options "")
    set(oneValueArgs TARGET OUTPUT_SRCS_CPP OUTPUT_SRCS_HPP OUTPUT_INCS_CPP OUTPUT_INCS_HPP)
    set(multiValueArgs "")
    cmake_parse_arguments(PARSE_ARGV 0 EMBED_KERNELS "${options}" "${oneValueArgs}" "${multiValueArgs}")

    # Validation checks
    if(NOT EMBED_KERNELS_TARGET)
        message(FATAL_ERROR "embed_kernel_sources called without a TARGET!")
    endif()
    if(NOT TARGET ${EMBED_KERNELS_TARGET})
        message(FATAL_ERROR "embed_kernel_sources: The target ${EMBED_KERNELS_TARGET} does not exist yet.
                Please make sure to call embed_kernel_sources after the target is created.")
    endif()
    if(NOT EMBED_KERNELS_OUTPUT_SRCS_CPP OR NOT EMBED_KERNELS_OUTPUT_SRCS_HPP OR NOT EMBED_KERNELS_OUTPUT_INCS_CPP OR NOT EMBED_KERNELS_OUTPUT_INCS_HPP)
        message(FATAL_ERROR "embed_kernel_sources called without all output file arguments specified.")
    endif()

    get_target_property(KERNEL_FILES ${EMBED_KERNELS_TARGET} KERNELEMBEDDING_KERNEL_FILES)
    get_target_property(KERNEL_KEYS ${EMBED_KERNELS_TARGET} KERNELEMBEDDING_KERNEL_KEYS)
    set(KERNEL_DECLARATIONS "")
    set(KERNEL_DEFINITIONS "")
    set(KERNEL_MAP_ENTRIES "")
    set(HEADER_DECLARATIONS "")
    set(HEADER_DEFINITIONS "")
    set(HEADER_MAP_ENTRIES "")
    set(HEADER_FILENAMES "")

    set(SEEN_KERNEL_KEYS "")
    set(SEEN_KERNEL_KEY_FILES "")
    set(SEEN_HEADER_KEYS "")
    set(SEEN_HEADER_KEY_FILES "")

    set(_embed_index 0)
    foreach(KERNEL_FILE IN LISTS KERNEL_FILES)
        list(GET KERNEL_KEYS ${_embed_index} KERNEL_KEY)
        math(EXPR _embed_index "${_embed_index} + 1")
        file(READ ${KERNEL_FILE} KERNEL_CONTENT)
        # Setting this property ensures that the kernels will get properly re-inlined as they are modified.
        # The drawback is that CMake is going to reconfigure every time a kernel is modified.
        # Yet, that will improve developer experience, because it will lead to shorter build times -- only
        # the build of kernel_sources and kernel_includes will be affected.
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${KERNEL_FILE})
        get_filename_component(KERNEL_NAME ${KERNEL_FILE} NAME_WE)
        get_filename_component(KERNEL_FILENAME ${KERNEL_FILE} NAME)
        get_filename_component(KERNEL_EXT ${KERNEL_FILE} EXT)
        string(TOUPPER "${EMBED_KERNELS_TARGET}_${KERNEL_NAME}" KERNEL_VAR_NAME)

        # Check if this is a header file
        if(KERNEL_EXT STREQUAL ".h" OR KERNEL_EXT STREQUAL ".hpp")
            _embed_one_kernel_header("${EMBED_KERNELS_TARGET}" "${KERNEL_FILE}"
                                     "${KERNEL_FILENAME}" "${KERNEL_VAR_NAME}" KERNEL_CONTENT)
        else()
            _embed_one_kernel_source("${EMBED_KERNELS_TARGET}" "${KERNEL_FILE}" "${KERNEL_KEY}"
                                     KERNEL_CONTENT)
        endif()
    endforeach()

    # Published on the target, because the manifest belongs to the target and not to the
    # directory that happened to declare it. A consumer in another directory scope reads
    # the path from here instead of spelling the same rule again, so it cannot end up
    # naming a file nothing wrote.
    set(_key_manifest "${CMAKE_CURRENT_BINARY_DIR}/${EMBED_KERNELS_TARGET}_kernel_keys.txt")
    _write_kernel_key_manifest("${_key_manifest}")
    set_target_properties(${EMBED_KERNELS_TARGET}
                          PROPERTIES KERNELEMBEDDING_KEY_MANIFEST "${_key_manifest}")

    # Generate kernel source files
    configure_file(
        ${PROJECT_SOURCE_DIR}/src/cmake/templates/kernel_sources.hpp.in
        ${EMBED_KERNELS_OUTPUT_SRCS_HPP}
        @ONLY
    )
    configure_file(
        ${PROJECT_SOURCE_DIR}/src/cmake/templates/kernel_sources.cpp.in
        ${EMBED_KERNELS_OUTPUT_SRCS_CPP}
        @ONLY
    )

    # Generate kernel include files
    configure_file(
        ${PROJECT_SOURCE_DIR}/src/cmake/templates/kernel_includes.hpp.in
        ${EMBED_KERNELS_OUTPUT_INCS_HPP}
        @ONLY
    )
    configure_file(
        ${PROJECT_SOURCE_DIR}/src/cmake/templates/kernel_includes.cpp.in
        ${EMBED_KERNELS_OUTPUT_INCS_CPP}
        @ONLY
    )

endfunction()
