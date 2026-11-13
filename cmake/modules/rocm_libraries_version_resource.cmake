# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Embeds a Windows PE VERSIONINFO resource in a DLL or EXE target, so the
# binary's File Explorer "Details" tab shows its version, description and the
# build it came from. Does nothing on non-Windows platforms.
#
#   rocm_libraries_add_version_resource(<target> <name> <description>
#       [PRODUCT_NAME <product_name>]
#       [COMMENTS <comments>])
#
# FILEVERSION and PRODUCTVERSION come from PROJECT_VERSION_{MAJOR,MINOR,PATCH}
# in the calling scope. <name> sets InternalName and <description> sets
# FileDescription. OriginalFilename is derived from the target's OUTPUT_NAME,
# PREFIX and SUFFIX. PRODUCT_NAME overrides ProductName (default
# "<name> <filename>"); COMMENTS adds a Comments value, for example the git
# commit the binary was built from.
#
# The .rc file is only rewritten when its content changes, so reconfiguring
# does not relink the target. When no resource compiler is available
# (CMAKE_RC_COMPILER unset, NOTFOUND or NOTREQUIRED, or RC not enabled) a
# warning is printed and the target is built without a version resource.
#
#   rocm_libraries_check_version_resources(DIRECTORY <dir>
#       [EXCLUDE <target>...]
#       [EXCLUDE_DIRECTORIES <dir>...])
#
# Fails configuration, on every platform, if a DLL, EXE or module target
# defined under <dir> was never passed to rocm_libraries_add_version_resource.
# Call it once at the end of a project's top-level CMakeLists.txt. Targets
# from other source trees, from the build tree (e.g. FetchContent
# dependencies), under EXCLUDE_DIRECTORIES or listed in EXCLUDE are skipped.

include_guard(GLOBAL)

# Escape VAR in place for an RC string literal: backslash is an escape there, and a doubled quote embeds a quote.
macro(_rocm_libraries_version_resource_escape VAR)
    string(REPLACE "\\" "\\\\" ${VAR} "${${VAR}}")
    string(REPLACE "\"" "\"\"" ${VAR} "${${VAR}}")
endmacro()

# Set OUT_FILENAME to the file TARGET builds and OUT_FILETYPE to VFT_APP or VFT_DLL.
function(_rocm_libraries_version_resource_file TARGET OUT_FILENAME OUT_FILETYPE)
    get_target_property(type ${TARGET} TYPE)
    if(type STREQUAL "EXECUTABLE")
        set(filetype VFT_APP)
        set(prefix "")
        set(suffix "${CMAKE_EXECUTABLE_SUFFIX}")
    elseif(type STREQUAL "SHARED_LIBRARY")
        set(filetype VFT_DLL)
        set(prefix "${CMAKE_SHARED_LIBRARY_PREFIX}")
        set(suffix "${CMAKE_SHARED_LIBRARY_SUFFIX}")
    elseif(type STREQUAL "MODULE_LIBRARY")
        set(filetype VFT_DLL)
        set(prefix "${CMAKE_SHARED_MODULE_PREFIX}")
        set(suffix "${CMAKE_SHARED_MODULE_SUFFIX}")
    else()
        message(FATAL_ERROR "rocm_libraries_add_version_resource: ${TARGET} is a ${type}; "
                            "only EXECUTABLE, SHARED_LIBRARY and MODULE_LIBRARY targets are supported")
    endif()
    get_target_property(basename ${TARGET} OUTPUT_NAME)
    if(NOT basename)
        set(basename ${TARGET})
    endif()
    get_target_property(target_prefix ${TARGET} PREFIX)
    if(NOT target_prefix MATCHES "-NOTFOUND$")
        set(prefix "${target_prefix}")
    endif()
    get_target_property(target_suffix ${TARGET} SUFFIX)
    if(NOT target_suffix MATCHES "-NOTFOUND$")
        set(suffix "${target_suffix}")
    endif()
    set(${OUT_FILENAME} "${prefix}${basename}${suffix}" PARENT_SCOPE)
    set(${OUT_FILETYPE} ${filetype} PARENT_SCOPE)
endfunction()

# Write CONTENT to PATH only when it differs, so an unchanged resource keeps its timestamp.
function(_rocm_libraries_version_resource_write PATH CONTENT)
    set(existing "")
    if(EXISTS "${PATH}")
        file(READ "${PATH}" existing)
    endif()
    if(NOT existing STREQUAL CONTENT)
        file(WRITE "${PATH}" "${CONTENT}")
    endif()
endfunction()

# Embed VERSIONINFO in TARGET; see the top of this file for the arguments.
function(rocm_libraries_add_version_resource TARGET NAME DESCRIPTION)
    # PARSE_ARGV keeps a quoted value containing ';' (e.g. a git subject) as one argument.
    cmake_parse_arguments(PARSE_ARGV 3 ARG "" "PRODUCT_NAME;COMMENTS" "")
    if(ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "rocm_libraries_add_version_resource: unexpected arguments: ${ARG_UNPARSED_ARGUMENTS}")
    endif()
    # Marked on every platform so rocm_libraries_check_version_resources works in Linux builds too.
    set_property(TARGET ${TARGET} PROPERTY ROCM_LIBRARIES_VERSION_RESOURCE TRUE)
    if(NOT WIN32)
        return()
    endif()
    _rocm_libraries_version_resource_file(${TARGET} filename filetype)

    get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
    list(FIND languages RC rc_index)
    if(rc_index EQUAL -1
       OR NOT CMAKE_RC_COMPILER
       OR CMAKE_RC_COMPILER MATCHES "NOTFOUND|NOTREQUIRED")
        message(WARNING "rocm_libraries_add_version_resource: no Windows resource compiler is available; "
                        "${TARGET} will be built without a version resource")
        return()
    endif()

    set(product_name "${NAME} ${filename}")
    if(DEFINED ARG_PRODUCT_NAME)
        set(product_name "${ARG_PRODUCT_NAME}")
    endif()
    _rocm_libraries_version_resource_escape(NAME)
    _rocm_libraries_version_resource_escape(DESCRIPTION)
    _rocm_libraries_version_resource_escape(product_name)
    set(comments_value "")
    if(DEFINED ARG_COMMENTS)
        set(comments "${ARG_COMMENTS}")
        _rocm_libraries_version_resource_escape(comments)
        set(comments_value "      VALUE \"Comments\", \"${comments}\\0\"\n")
    endif()

    set(version "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")
    set(version_csv "${PROJECT_VERSION_MAJOR},${PROJECT_VERSION_MINOR},${PROJECT_VERSION_PATCH},0")
    string(TIMESTAMP year "%Y")
    set(rc_path "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_version.rc")
    _rocm_libraries_version_resource_write("${rc_path}" "#include <winver.h>
VS_VERSION_INFO VERSIONINFO
FILEVERSION     ${version_csv}
PRODUCTVERSION  ${version_csv}
FILEFLAGSMASK   VS_FFI_FILEFLAGSMASK
#ifdef _DEBUG
FILEFLAGS       VS_FF_DEBUG
#else
FILEFLAGS       0
#endif
FILEOS          VOS_NT_WINDOWS32
FILETYPE        ${filetype}
FILESUBTYPE     VFT2_UNKNOWN
BEGIN
  BLOCK \"StringFileInfo\"
  BEGIN
    BLOCK \"040904B0\"
    BEGIN
      VALUE \"CompanyName\", \"Advanced Micro Devices, Inc.\\0\"
      VALUE \"FileDescription\", \"${DESCRIPTION}\\0\"
      VALUE \"FileVersion\", \"${version}.0\\0\"
      VALUE \"InternalName\", \"${NAME}\\0\"
      VALUE \"LegalCopyright\", \"Copyright (c) ${year} Advanced Micro Devices, Inc. All rights reserved.\\0\"
      VALUE \"OriginalFilename\", \"${filename}\\0\"
      VALUE \"ProductName\", \"${product_name}\\0\"
      VALUE \"ProductVersion\", \"${version}\\0\"
${comments_value}    END
  END
  BLOCK \"VarFileInfo\"
  BEGIN
    VALUE \"Translation\", 0x409, 1200
  END
END
")
    target_sources(${TARGET} PRIVATE "${rc_path}")
endfunction()

# Set OUT_VAR to every buildsystem target defined in DIR or its subdirectories.
function(_rocm_libraries_version_resource_targets DIR OUT_VAR)
    get_property(targets DIRECTORY "${DIR}" PROPERTY BUILDSYSTEM_TARGETS)
    get_property(subdirs DIRECTORY "${DIR}" PROPERTY SUBDIRECTORIES)
    foreach(subdir IN LISTS subdirs)
        _rocm_libraries_version_resource_targets("${subdir}" subdir_targets)
        list(APPEND targets ${subdir_targets})
    endforeach()
    set(${OUT_VAR} ${targets} PARENT_SCOPE)
endfunction()

# Set OUT_VAR to TRUE when TARGET must carry a version resource under the given project and exclusions.
function(_rocm_libraries_version_resource_required TARGET DIRECTORY EXCLUDE EXCLUDE_DIRECTORIES OUT_VAR)
    set(${OUT_VAR} FALSE PARENT_SCOPE)
    get_target_property(type ${TARGET} TYPE)
    list(FIND EXCLUDE ${TARGET} excluded_index)
    if(NOT type MATCHES "^(EXECUTABLE|SHARED_LIBRARY|MODULE_LIBRARY)$" OR NOT excluded_index EQUAL -1)
        return()
    endif()
    get_target_property(source_dir ${TARGET} SOURCE_DIR)
    cmake_path(IS_PREFIX DIRECTORY "${source_dir}" NORMALIZE in_project)
    if(NOT in_project)
        return()
    endif()
    set(fetchcontent_dir "${CMAKE_BINARY_DIR}/_deps")
    if(FETCHCONTENT_BASE_DIR)
        set(fetchcontent_dir "${FETCHCONTENT_BASE_DIR}")
    endif()
    foreach(skipped_dir IN LISTS EXCLUDE_DIRECTORIES fetchcontent_dir)
        cmake_path(IS_PREFIX skipped_dir "${source_dir}" NORMALIZE in_skipped)
        if(in_skipped)
            return()
        endif()
    endforeach()
    set(${OUT_VAR} TRUE PARENT_SCOPE)
endfunction()

# Fail configuration if a DLL, EXE or module target under DIRECTORY lacks a version resource; see the top of this file.
function(rocm_libraries_check_version_resources)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "DIRECTORY" "EXCLUDE;EXCLUDE_DIRECTORIES")
    if(ARG_UNPARSED_ARGUMENTS OR NOT ARG_DIRECTORY)
        message(FATAL_ERROR "rocm_libraries_check_version_resources: usage: "
                            "DIRECTORY <dir> [EXCLUDE <target>...] [EXCLUDE_DIRECTORIES <dir>...]")
    endif()
    _rocm_libraries_version_resource_targets("${ARG_DIRECTORY}" targets)
    set(missing "")
    foreach(target IN LISTS targets)
        get_target_property(marked ${target} ROCM_LIBRARIES_VERSION_RESOURCE)
        if(NOT marked)
            _rocm_libraries_version_resource_required(
                ${target} "${ARG_DIRECTORY}" "${ARG_EXCLUDE}" "${ARG_EXCLUDE_DIRECTORIES}" required)
            if(required)
                list(APPEND missing ${target})
            endif()
        endif()
    endforeach()
    if(missing)
        list(JOIN missing ", " missing_list)
        message(FATAL_ERROR
            "These DLL/EXE targets under ${ARG_DIRECTORY} have no Windows version resource: ${missing_list}. "
            "Add a rocm_libraries_add_version_resource call (or the project's wrapper) for each shipped "
            "binary, or list a binary that is not shipped in the project's EXCLUDE list.")
    endif()
endfunction()
