# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Function to embed kernel sources as C++ strings at configure time
function(embed_kernel_sources OUTPUT_SRCS_CPP OUTPUT_SRCS_HPP OUTPUT_INCS_CPP OUTPUT_INCS_HPP KERNEL_FILES)
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

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/templates/kernel_sources.hpp.in ${OUTPUT_SRCS_HPP} @ONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/templates/kernel_sources.cpp.in ${OUTPUT_SRCS_CPP} @ONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/templates/kernel_includes.hpp.in ${OUTPUT_INCS_HPP} @ONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/templates/kernel_includes.cpp.in ${OUTPUT_INCS_CPP} @ONLY)
endfunction()

# List kernel files to embed
set(KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/relu/ReluForward.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/conv/ConvForwardNaive.cpp
)

# Generate embedded kernel sources at configure time
embed_kernel_sources(
    ${CMAKE_CURRENT_BINARY_DIR}/kernel_sources.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/kernel_sources.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/kernel_includes.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/kernel_includes.hpp
    "${KERNEL_FILES}"
)

# Export generated sources for src/ to compile
set(KERNEL_GENERATED_SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/kernel_sources.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/kernel_includes.cpp
    PARENT_SCOPE
)
