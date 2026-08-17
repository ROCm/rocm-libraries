# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Standalone CMake script that embeds kernel source files as C++ string literals.
#
# Invoked at build time via:
#   cmake -DKERNEL_FILES="file1.cpp|file2.cpp"
#         -DTEMPLATE_DIR=/path/to/templates
#         -DOUTPUT_DIR=/path/to/output
#         -P EmbedKernelSources.cmake
#
# KERNEL_FILES uses pipe (|) as the separator instead of semicolons to avoid
# escaping issues when VERBATIM passes the argument through the shell.
#
# Reads each file in KERNEL_FILES, classifies it as a kernel source (.cpp) or
# a kernel header (.h/.hpp), and populates template variables used by
# configure_file() to generate four output files:
#   kernel_sources.hpp / kernel_sources.cpp   - kernel .cpp source registry
#   kernel_includes.hpp / kernel_includes.cpp - kernel .h/.hpp header registry

# Convert pipe-separated list back to CMake list (semicolons)
string(REPLACE "|" ";" KERNEL_FILES "${KERNEL_FILES}")

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

    if(KERNEL_EXT STREQUAL ".h" OR KERNEL_EXT STREQUAL ".hpp")
        string(APPEND HEADER_DECLARATIONS "extern const char* const ${KERNEL_VAR_NAME}_SOURCE;\n")
        string(APPEND HEADER_DEFINITIONS "const char* const ${KERNEL_VAR_NAME}_SOURCE = R\"KERNEL_SOURCE(\n")
        string(APPEND HEADER_DEFINITIONS "${KERNEL_CONTENT}")
        string(APPEND HEADER_DEFINITIONS "\n)KERNEL_SOURCE\";\n\n")
        string(APPEND HEADER_MAP_ENTRIES "        {\"${KERNEL_FILENAME}\", ${KERNEL_VAR_NAME}_SOURCE},\n")
        string(APPEND HEADER_FILENAMES "        \"${KERNEL_FILENAME}\",\n")
    else()
        string(APPEND KERNEL_DECLARATIONS "extern const char* const ${KERNEL_VAR_NAME}_SOURCE;\n")
        string(APPEND KERNEL_DEFINITIONS "const char* const ${KERNEL_VAR_NAME}_SOURCE = R\"KERNEL_SOURCE(\n")
        string(APPEND KERNEL_DEFINITIONS "${KERNEL_CONTENT}")
        string(APPEND KERNEL_DEFINITIONS "\n)KERNEL_SOURCE\";\n\n")
        string(APPEND KERNEL_MAP_ENTRIES "        {\"${KERNEL_FILENAME}\", ${KERNEL_VAR_NAME}_SOURCE},\n")
    endif()
endforeach()

configure_file(${TEMPLATE_DIR}/kernel_sources.hpp.in ${OUTPUT_DIR}/kernel_sources.hpp @ONLY)
configure_file(${TEMPLATE_DIR}/kernel_sources.cpp.in ${OUTPUT_DIR}/kernel_sources.cpp @ONLY)
configure_file(${TEMPLATE_DIR}/kernel_includes.hpp.in ${OUTPUT_DIR}/kernel_includes.hpp @ONLY)
configure_file(${TEMPLATE_DIR}/kernel_includes.cpp.in ${OUTPUT_DIR}/kernel_includes.cpp @ONLY)
