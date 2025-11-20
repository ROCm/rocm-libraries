# Function to copy multiple folders with path transformation
function(copy_emulation_config_folders)
    # Function arguments
    set(options "")                            # Optional arguments
    set(oneValueArgs DESTINATION_PREFIX)       # Single-value arguments
    set(multiValueArgs SOURCE_FOLDERS SOURCE_SCRIPTS)         # Multi-value arguments
    cmake_parse_arguments(COPY "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Check if required parameters are provided
    if(NOT COPY_SOURCE_FOLDERS)
        message(FATAL_ERROR "SOURCE_FOLDERS must be specified")
    endif()

    if(NOT COPY_SOURCE_SCRIPTS)
        message(FATAL_ERROR "SOURCE_SCRIPTS must be specified")
    endif()

    if(NOT COPY_DESTINATION_PREFIX)
        message(FATAL_ERROR "DESTINATION_PREFIX must be specified")
    endif()

    # Process each source folder
    foreach(source_path ${COPY_SOURCE_FOLDERS})
        # Ensure source directory exists
        if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source_path})
            message(WARNING "Source directory '${CMAKE_CURRENT_SOURCE_DIR}/${source_path}' does not exist, skipping...")
            continue()
        endif()

        # Extract the folder name (e.g., "folder01" from "folder01/configs/emulation")
        string(REGEX MATCH "^[^/]+" folder_name ${source_path})

        # Create the destination path
        set(dest_path "${COPY_DESTINATION_PREFIX}/${folder_name}")

        # Find all files recursively in the source folder
        file(GLOB_RECURSE source_files
            LIST_DIRECTORIES false
            "${CMAKE_CURRENT_SOURCE_DIR}/${source_path}/*.yaml"
        )

        # Create the destination directory
        file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/${dest_path})

        # Copy each file preserving directory structure
        foreach(source_file ${source_files})
            # Get the relative path from the source directory
            file(RELATIVE_PATH rel_path
                "${CMAKE_CURRENT_SOURCE_DIR}/${source_path}"
                "${source_file}"
            )

            # Get the destination directory for this file
            get_filename_component(dest_dir
                "${CMAKE_BINARY_DIR}/${dest_path}/${rel_path}"
                DIRECTORY
            )

            # Create destination directory if it doesn't exist
            file(MAKE_DIRECTORY ${dest_dir})

            # Copy the file
            configure_file(
                ${source_file}
                "${CMAKE_BINARY_DIR}/${dest_path}/${rel_path}"
                COPYONLY
            )

        endforeach()
    endforeach()

    foreach(source_script_path ${COPY_SOURCE_SCRIPTS})
        # Ensure source directory exists
        if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source_script_path})
            message(WARNING "Source directory '${CMAKE_CURRENT_SOURCE_DIR}/${source_script_path}' does not exist, skipping...")
            continue()
        endif()

        get_filename_component(source_script_name ${source_script_path} NAME)

        # Copy the file
        file(COPY_FILE
            ${source_script_path}
            "${CMAKE_BINARY_DIR}/${source_script_name}"
        )
    endforeach()

    message(STATUS "Copying emulation test config files")

    # Create a custom target to track the copied files
    add_custom_target(copy_emulation_config_files DEPENDS ${source_files} ${COPY_SOURCE_SCRIPTS})
endfunction()

function(copy_code_coverage_config_files SOURCE_FOLDER DEST_FOLDER)
    # Define the patterns to process
    set(PATTERNS "01_contraction" "02_elementwise" "03_reduction")
    # Initialize variable to track copied files
    set(COPIED_FILES_LIST "")

    # Process each pattern
    foreach(PATTERN ${PATTERNS})
        # Create the source path
        if(PATTERN STREQUAL "01_contraction")
            set(SOURCE_PATHS "${SOURCE_FOLDER}/${PATTERN}/configs/code_coverage/")
        elseif(PATTERN STREQUAL "02_elementwise")
            # The config files of smoke and code coverage are the same for permution and reduction
            # reuse the smoke config files to avoid duplicated files
            set(SOURCE_PATHS
                "${SOURCE_FOLDER}/${PATTERN}/configs/emulation/smoke/permutation"
                "${SOURCE_FOLDER}/${PATTERN}/configs/emulation/smoke/binary_op"
                "${SOURCE_FOLDER}/${PATTERN}/configs/emulation/smoke/trinary_op"
                )
        elseif(PATTERN STREQUAL "03_reduction")
            set(SOURCE_PATHS "${SOURCE_FOLDER}/${PATTERN}/configs/code_coverage/")
        endif()

        foreach(SOURCE_PATH ${SOURCE_PATHS})
            # Check if the source directory exists
            if(EXISTS "${SOURCE_PATH}")
                # Create destination directory if it doesn't exist
                file(MAKE_DIRECTORY "${DEST_FOLDER}/${PATTERN}")

                # Find all yaml files in the source directory
                file(GLOB YAML_FILES "${SOURCE_PATH}/*.yaml")

                # Copy each yaml file to the destination
                foreach(YAML_FILE ${YAML_FILES})
                    get_filename_component(FILE_NAME ${YAML_FILE} NAME)
                    file(COPY "${YAML_FILE}" DESTINATION "${DEST_FOLDER}/${PATTERN}")
                    message(VERBOSE "Copied ${FILE_NAME} to ${DEST_FOLDER}/${PATTERN}/")
                    list(APPEND COPIED_FILES_LIST "${SOURCE_PATH}/${FILE_NAME}")
                endforeach()
            else()
                message(WARNING "Source directory ${SOURCE_PATH} does not exist.")
            endif()
        endforeach()
    endforeach()

    message(STATUS "Copying code coverage config files")

    # Create a custom target to track the copied files
    add_custom_target(copy_code_coverage_config_files DEPENDS ${COPIED_FILES_LIST})
endfunction()

