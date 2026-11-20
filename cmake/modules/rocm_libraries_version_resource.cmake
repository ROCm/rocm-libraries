# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Embeds a Windows PE VERSIONINFO resource in a DLL or EXE target, so the
# binary's File Explorer "Details" tab shows its version, description and the
# build it came from. Does nothing on non-Windows platforms.
#
#   rocm_libraries_add_version_resource(<target> <name> <description>
#       VERSION <major>.<minor>.<patch>
#       [PRODUCT_NAME <product_name>]
#       [COMMENTS <comments>])
#
# VERSION sets FILEVERSION and PRODUCTVERSION and is checked on every platform.
# <name> sets InternalName and <description> sets
# FileDescription. OriginalFilename is derived from the target's OUTPUT_NAME,
# PREFIX and SUFFIX. PRODUCT_NAME overrides ProductName (default
# "<name> <filename>"); COMMENTS adds a Comments value, for example the git
# commit the binary was built from.
#
# The .rc file is only rewritten when its content changes, so reconfiguring
# does not relink the target. When no resource compiler is available
# (CMAKE_RC_COMPILER unset, NOTFOUND or NOTREQUIRED, or RC not enabled),
# configuration fails, so a Windows build cannot ship a binary without the
# resource. Set ROCM_LIBRARIES_REQUIRE_VERSION_RESOURCE=OFF to warn instead
# and build the target without one.
#
#   rocm_libraries_check_version_resources(DIRECTORY <dir>
#       [EXCLUDE <target>...]
#       [EXCLUDE_DIRECTORIES <dir>...])
#
# Fails configuration, on every platform, if a DLL, EXE or module target
# defined under <dir> was never passed to rocm_libraries_add_version_resource.
# Call it once at the end of a project's top-level CMakeLists.txt. Targets
# from other source trees, FetchContent dependencies (under
# FETCHCONTENT_BASE_DIR, by default <build>/_deps), targets under
# EXCLUDE_DIRECTORIES and targets listed in EXCLUDE are skipped.

include_guard(GLOBAL)

option(ROCM_LIBRARIES_REQUIRE_VERSION_RESOURCE
       "Fail configuration on Windows when a DLL or EXE version resource cannot be compiled" ON)

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

# Embed VERSIONINFO in TARGET; see the top of this file for the arguments.
function(rocm_libraries_add_version_resource TARGET NAME DESCRIPTION)
    # PARSE_ARGV keeps a quoted value containing ';' (e.g. a git subject) as one argument.
    cmake_parse_arguments(PARSE_ARGV 3 ARG "" "VERSION;PRODUCT_NAME;COMMENTS" "")
    if(ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "rocm_libraries_add_version_resource: unexpected arguments: ${ARG_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT ARG_VERSION MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$")
        message(FATAL_ERROR "rocm_libraries_add_version_resource: ${TARGET} needs VERSION <major>.<minor>.<patch>, "
                            "got '${ARG_VERSION}'")
    endif()
    set(version "${ARG_VERSION}")
    set(version_csv "${CMAKE_MATCH_1},${CMAKE_MATCH_2},${CMAKE_MATCH_3},0")
    # Marked on every platform so rocm_libraries_check_version_resources works in Linux builds too.
    # On Windows a marked target has its resource attached, unless the build opted out below.
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
        if(ROCM_LIBRARIES_REQUIRE_VERSION_RESOURCE)
            message(FATAL_ERROR "rocm_libraries_add_version_resource: no Windows resource compiler is available, "
                                "so ${TARGET} would be built without a version resource. Install the Windows SDK "
                                "or put llvm-rc on PATH, or configure with "
                                "-DROCM_LIBRARIES_REQUIRE_VERSION_RESOURCE=OFF to build without one.")
        endif()
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
        set(comments_value "      VALUE \"Comments\", \"${comments}\\0\"")
    endif()

    string(TIMESTAMP year "%Y")
    set(rc_path "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_version.rc")
    configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/rocm_libraries_version_resource.rc.in" "${rc_path}" @ONLY)
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
