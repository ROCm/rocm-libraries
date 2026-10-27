# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# hipdnn_tools_stage_files(<target> <directory> FILES <file>...
#                          [INSTALL_DESTINATION <dir>])
#
# Copies each file into <directory> in the build tree whenever it changes, behind <target>,
# and installs it to INSTALL_DESTINATION when given. A binary that finds its data at a fixed
# offset from itself then reads the copy beside whichever tree it runs from.
function(hipdnn_tools_stage_files target directory)
    cmake_parse_arguments(ARG "" "INSTALL_DESTINATION" "FILES" ${ARGN})
    set(staged_files "")
    foreach(source IN LISTS ARG_FILES)
        get_filename_component(name "${source}" NAME)
        set(staged "${directory}/${name}")
        add_custom_command(
            OUTPUT "${staged}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${directory}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${source}" "${staged}"
            DEPENDS "${source}"
            COMMENT "Staging ${name}"
            VERBATIM
        )
        list(APPEND staged_files "${staged}")
    endforeach()
    add_custom_target(${target} DEPENDS ${staged_files} COMMENT "Staging files for ${target}")
    if(ARG_INSTALL_DESTINATION)
        install(FILES ${ARG_FILES} DESTINATION "${ARG_INSTALL_DESTINATION}")
    endif()
endfunction()
