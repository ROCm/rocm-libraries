# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# This module provides a CMake function to copy external header files with nested
# headers inlined, using the copy_external_header.py script to recursively
# resolve #include directives.

# Find Python3 interpreter once at module scope
find_package(Python3 REQUIRED COMPONENTS Interpreter)

function(copy_external_header)
    set(options "")
    set(oneValueArgs TARGET SOURCE_HEADER ROOT_INCLUDE_DIR OUTPUT_PATH COMMENT)
    set(multiValueArgs "")

    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Validate required arguments
    if(NOT ARG_TARGET)
	message(FATAL_ERROR "copy_external_header: TARGET argument is required")
    endif()

    if(NOT ARG_SOURCE_HEADER)
	message(FATAL_ERROR "copy_external_header: SOURCE_HEADER argument is required")
    endif()

    if(NOT ARG_ROOT_INCLUDE_DIR)
	message(FATAL_ERROR "copy_external_header: ROOT_INCLUDE_DIR argument is required")
    endif()

    if(NOT ARG_OUTPUT_PATH)
	message(FATAL_ERROR "copy_external_header: OUTPUT_PATH argument is required")
    endif()

    # Check if source header exists
    if(NOT EXISTS ${ARG_SOURCE_HEADER})
	message(FATAL_ERROR "copy_external_header: Source header not found: ${ARG_SOURCE_HEADER}")
    endif()

    # Check if root include directory exists
    if(NOT EXISTS ${ARG_ROOT_INCLUDE_DIR})
	message(FATAL_ERROR "miopen_inline_header: Root include directory not found: ${ARG_ROOT_INCLUDE_DIR}")
    endif()

    # Set default comment if not provided
    if(NOT ARG_COMMENT)
	get_filename_component(HEADER_NAME ${ARG_SOURCE_HEADER} NAME)
	set(ARG_COMMENT "Inlining ${HEADER_NAME} with all dependencies as C++ source")
    endif()

    # Get the directory of the output path to create it if needed
    get_filename_component(OUTPUT_DIR ${ARG_OUTPUT_PATH} DIRECTORY)

    # Create custom command to inline the header
    add_custom_command(
	OUTPUT ${ARG_OUTPUT_PATH}
	COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
	COMMAND ${Python3_EXECUTABLE}
	    ${PROJECT_SOURCE_DIR}/cmake/copy_external_header.py
	    ${ARG_SOURCE_HEADER}
	    ${ARG_ROOT_INCLUDE_DIR}
	    ${ARG_OUTPUT_PATH}
	DEPENDS
	    ${PROJECT_SOURCE_DIR}/cmake/copy_external_header.py
	    ${ARG_SOURCE_HEADER}
	COMMENT "${ARG_COMMENT}"
	VERBATIM
    )

    # Create a custom target to ensure it's generated
    add_custom_target(${ARG_TARGET}
	DEPENDS ${ARG_OUTPUT_PATH}
    )
endfunction()
