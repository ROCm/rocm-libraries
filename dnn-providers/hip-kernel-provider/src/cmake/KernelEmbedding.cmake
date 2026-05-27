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
# CMake ignores attempts to redefine an already defined property,
# so the value in this property won't be overwritten when including
# this module anywhere else in the repo

# Function to mark kernel source files for embedding in the single source file
function(add_kernels_for_embedding)
    foreach(KERNEL_FILE IN LISTS ARGV)
        set_property(GLOBAL APPEND PROPERTY KERNELEMBEDDING_KERNEL_FILES ${KERNEL_FILE})
    endforeach()
endfunction()

# Function to embed kernel sources as C++ strings at configure time
function(embed_kernel_sources OUTPUT_SRCS_CPP OUTPUT_SRCS_HPP OUTPUT_INCS_CPP OUTPUT_INCS_HPP)
    get_property(KERNEL_FILES GLOBAL PROPERTY KERNELEMBEDDING_KERNEL_FILES)
    set(KERNEL_DECLARATIONS "")
    set(KERNEL_DEFINITIONS "")
    set(KERNEL_MAP_ENTRIES "")
    set(HEADER_DECLARATIONS "")
    set(HEADER_DEFINITIONS "")
    set(HEADER_MAP_ENTRIES "")
    set(HEADER_FILENAMES "")

    foreach(KERNEL_FILE ${KERNEL_FILES})
        file(READ ${KERNEL_FILE} KERNEL_CONTENT)
        get_filename_component(KERNEL_NAME ${KERNEL_FILE} NAME_WE)
        get_filename_component(KERNEL_FILENAME ${KERNEL_FILE} NAME)
        get_filename_component(KERNEL_EXT ${KERNEL_FILE} EXT)
        string(TOUPPER ${KERNEL_NAME} KERNEL_VAR_NAME)

        # Check if this is a header file
        if(KERNEL_EXT STREQUAL ".h" OR KERNEL_EXT STREQUAL ".hpp")
            # Build declarations for header includes
            string(APPEND HEADER_DECLARATIONS "extern const char* const ${KERNEL_VAR_NAME}_SOURCE;\n")

            # Build definitions for header includes
            string(APPEND HEADER_DEFINITIONS "const char* const ${KERNEL_VAR_NAME}_SOURCE = R\"KERNEL_SOURCE(\n")
            string(APPEND HEADER_DEFINITIONS "${KERNEL_CONTENT}")
            string(APPEND HEADER_DEFINITIONS "\n)KERNEL_SOURCE\";\n\n")

            # Build map entry for GetKernelInc lookup
            string(APPEND HEADER_MAP_ENTRIES "        {\"${KERNEL_FILENAME}\", ${KERNEL_VAR_NAME}_SOURCE},\n")

            # Build list of header filenames
            string(APPEND HEADER_FILENAMES "        \"${KERNEL_FILENAME}\",\n")
        else()
            # Build declarations for kernel sources
            string(APPEND KERNEL_DECLARATIONS "extern const char* const ${KERNEL_VAR_NAME}_SOURCE;\n")

            # Build definitions for kernel sources
            string(APPEND KERNEL_DEFINITIONS "const char* const ${KERNEL_VAR_NAME}_SOURCE = R\"KERNEL_SOURCE(\n")
            string(APPEND KERNEL_DEFINITIONS "${KERNEL_CONTENT}")
            string(APPEND KERNEL_DEFINITIONS "\n)KERNEL_SOURCE\";\n\n")

            # Build map entry for getKernelSrc lookup
            string(APPEND KERNEL_MAP_ENTRIES "        {\"${KERNEL_FILENAME}\", ${KERNEL_VAR_NAME}_SOURCE},\n")
        endif()
    endforeach()

    # Generate kernel source files
    configure_file(
        ${PROJECT_SOURCE_DIR}/src/cmake/templates/kernel_sources.hpp.in
        ${OUTPUT_SRCS_HPP}
        @ONLY
    )
    configure_file(
        ${PROJECT_SOURCE_DIR}/src/cmake/templates/kernel_sources.cpp.in
        ${OUTPUT_SRCS_CPP}
        @ONLY
    )

    # Generate kernel include files
    configure_file(
        ${PROJECT_SOURCE_DIR}/src/cmake/templates/kernel_includes.hpp.in
        ${OUTPUT_INCS_HPP}
        @ONLY
    )
    configure_file(
        ${PROJECT_SOURCE_DIR}/src/cmake/templates/kernel_includes.cpp.in
        ${OUTPUT_INCS_CPP}
        @ONLY
    )

   endfunction()
