# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Path of the long-path-aware application manifest that sits next to this module.
set(_HIPDNN_LONG_PATH_MANIFEST
    "${CMAKE_CURRENT_LIST_DIR}/hipdnn-long-path-aware.manifest"
)

# Collects the build system targets of `dir` and of every directory below it into `out_var`.
function(_hipdnn_collect_targets_recursive dir out_var)
    get_directory_property(dir_targets DIRECTORY "${dir}" BUILDSYSTEM_TARGETS)
    get_directory_property(subdirs DIRECTORY "${dir}" SUBDIRECTORIES)
    set(all_targets ${dir_targets})
    foreach(subdir IN LISTS subdirs)
        _hipdnn_collect_targets_recursive("${subdir}" sub_targets)
        list(APPEND all_targets ${sub_targets})
    endforeach()
    set(${out_var} ${all_targets} PARENT_SCOPE)
endfunction()

# Adds the long-path-aware manifest to every executable defined in the current source directory
# or below it. Windows only; a no-op elsewhere. Call it as the last statement of the root
# CMakeLists.txt so that every executable already exists. Safe to call more than once: each
# executable receives the manifest a single time.
function(hipdnn_apply_long_path_manifest)
    if(NOT WIN32)
        return()
    endif()

    _hipdnn_collect_targets_recursive("${CMAKE_CURRENT_SOURCE_DIR}" targets)
    foreach(target IN LISTS targets)
        get_target_property(target_type ${target} TYPE)
        get_target_property(is_imported ${target} IMPORTED)
        get_target_property(already_applied ${target} HIPDNN_LONG_PATH_MANIFEST_APPLIED)
        if(target_type STREQUAL "EXECUTABLE" AND NOT is_imported AND NOT already_applied)
            target_sources(${target} PRIVATE "${_HIPDNN_LONG_PATH_MANIFEST}")
            set_target_properties(${target} PROPERTIES HIPDNN_LONG_PATH_MANIFEST_APPLIED TRUE)
            message(VERBOSE "hipdnn long-path manifest: ${target}")
        endif()
    endforeach()
endfunction()
