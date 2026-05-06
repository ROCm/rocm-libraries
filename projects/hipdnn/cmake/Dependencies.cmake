# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)

include(FetchContent)

option(HIPDNN_NO_DOWNLOAD "Disables downloading of any external dependencies" OFF)

if(HIPDNN_NO_DOWNLOAD)
    set(FETCHCONTENT_FULLY_DISCONNECTED OFF
        CACHE BOOL "Don't attempt to download or update anything" FORCE
    )
endif()

# Dependencies where the local version should be used, if available
set(_hipdnn_all_local_deps GTest flatbuffers spdlog nlohmann_json)
# Dependencies where we never look for a local version
set(_hipdnn_all_remote_deps)

# hipdnn_add_dependency( dep_name [NO_LOCAL] [VERSION version] [FIND_PACKAGE_ARGS args...]
# [COMPONENT component] [PACKAGE_NAME [package_name] [DEB deb_package_name] [RPM rpm_package_name]]
# )
#
# Adds a dependency to the project.
function(hipdnn_add_dependency dep_name)
    set(options NO_LOCAL)
    set(oneValueArgs VERSION HASH)
    set(multiValueArgs FIND_PACKAGE_ARGS PACKAGE_NAME COMPONENTS)
    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if(dep_name IN_LIST _hipdnn_all_local_deps)
        message(VERBOSE "----------- Finding ${dep_name} -----------")
        if(NOT PARSE_NO_LOCAL)
            find_package(${dep_name} ${PARSE_VERSION} QUIET ${PARSE_FIND_PACKAGE_ARGS})
        endif()
        if(NOT ${dep_name}_FOUND)
            message(STATUS "Did not find ${dep_name}, it will be built locally")
            _build_local()
        else()
            message(
                STATUS
                    "Found ${dep_name}: ${${dep_name}_DIR} (found version \"${${dep_name}_VERSION}\")"
            )
            set(${dep_name}_FOUND ${${dep_name}_FOUND} PARENT_SCOPE)
            # Only export ${dep_name}_VERSION when the package config actually
            # populated it. Some vendored or repackaged distributions skip
            # setting it; exporting an empty string would mask the unset state
            # in callers that branch on `if(DEFINED dep_VERSION)`.
            if(DEFINED ${dep_name}_VERSION)
                set(${dep_name}_VERSION "${${dep_name}_VERSION}" PARENT_SCOPE)
            endif()
            foreach(VAR IN LISTS ${dep_name}_EXPORT_VARS)
                set(${VAR} ${${VAR}} PARENT_SCOPE)
            endforeach()
        endif()
    elseif(dep_name IN_LIST _hipdnn_all_remote_deps)
        message(TRACE "Will build ${dep_name} locally")
        _build_local()
    else()
        message(WARNING "Unknown dependency: ${dep_name}")
        return()
    endif()
endfunction()

# Extract and use include directories from dependency targets instead of linking
# Function to add include directories and optional compile definitions from dependency targets
# Automatically detects target type and uses appropriate visibility (INTERFACE for interface libs, PUBLIC for others)
function(hipdnn_add_dependency_includes TARGET_NAME HEADER_LIB_TARGET_NAME)
    # Parse optional arguments
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs COMPILE_DEFINITIONS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Validate required parameters
    if(NOT TARGET ${TARGET_NAME})
        message(FATAL_ERROR "hipdnn_add_dependency_includes: Target '${TARGET_NAME}' does not exist")
        return()
    endif()

    if(NOT TARGET ${HEADER_LIB_TARGET_NAME})
        message(FATAL_ERROR "hipdnn_add_dependency_includes: Header library target '${HEADER_LIB_TARGET_NAME}' does not exist")
        return()
    endif()

    # Determine visibility based on target type
    get_target_property(_target_type ${TARGET_NAME} TYPE)
    if(_target_type STREQUAL "INTERFACE_LIBRARY")
        set(_visibility INTERFACE)
    else()
        set(_visibility PUBLIC)
    endif()

    get_target_property(_dep_includes ${HEADER_LIB_TARGET_NAME} INTERFACE_INCLUDE_DIRECTORIES)
    if(_dep_includes)
        foreach(_include IN LISTS _dep_includes)
            message(VERBOSE "${TARGET_NAME} adding include from ${HEADER_LIB_TARGET_NAME}: ${_include}")
            target_include_directories(${TARGET_NAME} SYSTEM ${_visibility} $<BUILD_INTERFACE:${_include}>)
        endforeach()
    endif()

    if(ARG_COMPILE_DEFINITIONS)
        target_compile_definitions(${TARGET_NAME} ${_visibility} ${ARG_COMPILE_DEFINITIONS})
    endif()
endfunction()


# Builds a dependency locally
macro(_build_local)
    cmake_policy(PUSH)
    message(VERBOSE "=========== Adding ${dep_name} ===========")
    _pushstate()
    set(CMAKE_MESSAGE_INDENT "${CMAKE_MESSAGE_INDENT}[${dep_name}] ")
    cmake_language(CALL _fetch_${dep_name} "${PARSE_VERSION}" "${PARSE_HASH}")
    _popstate()
    message(VERBOSE "=========== Added ${dep_name} ===========")
    cmake_policy(POP)
    foreach(VAR IN LISTS ${dep_name}_EXPORT_VARS)
        set(${VAR} ${${VAR}} PARENT_SCOPE)
    endforeach()
endmacro()

# Fetches GoogleTest
function(_fetch_gtest VERSION HASH)
    if(VERSION AND VERSION STREQUAL 1.16.0)
        set(GIT_TAG v1.16.0)
    else()
        _determine_git_tag(v v1.16.0)
    endif()
    if(HASH)
        set(HASH_ARG HASH ${HASH})
    endif()
    fetchcontent_declare(
        googletest URL https://github.com/google/googletest/archive/refs/tags/${GIT_TAG}.zip
                       ${HASH_ARG} DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    option(HIPDNN_GTEST_SHARED "Build GTest as a shared library." OFF)
    _save_var(BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ${HIPDNN_GTEST_SHARED} CACHE INTERNAL "")
    set(INSTALL_GTEST OFF)
    # Suppress ROCMChecks warnings from GTest's internal cmake modifying
    # CMAKE_C_FLAGS/CMAKE_CXX_FLAGS. ROCMChecks uses variable_watch() and always
    # prints regardless of ROCM_WARN_TOOLCHAIN_VAR. Override the callback with a
    # flag-gated macro wrapper (same pattern as composablekernel/cmake/gtest.cmake).
    # Must be a macro (not function) so the flag is read in the caller's scope.
    if(ROCM_LIBS_SUPERBUILD AND COMMAND rocm_check_toolchain_var
            AND NOT COMMAND _rocm_check_toolchain_var)
        # Overrides ROCMChecks callback to suppress warnings from third-party code
        macro(rocm_check_toolchain_var var access value list_file)
            if(NOT _HIPDNN_DISABLE_ROCM_CHECKS)
                _rocm_check_toolchain_var("${var}" "${access}" "${value}" "${list_file}")
            endif()
        endmacro()
    endif()
    set(_HIPDNN_DISABLE_ROCM_CHECKS TRUE)
    fetchcontent_makeavailable(googletest)
    set(_HIPDNN_DISABLE_ROCM_CHECKS FALSE)
    _restore_var(BUILD_SHARED_LIBS)

    _exclude_from_all(${googletest_SOURCE_DIR})
    _mark_targets_as_system(${googletest_SOURCE_DIR})
    target_compile_options(gtest PUBLIC ${EXTRA_COMPILE_OPTIONS})
    target_compile_options(gmock PUBLIC ${EXTRA_COMPILE_OPTIONS})
endfunction()

# Fetches FlatBuffers
function(_fetch_flatbuffers VERSION HASH)
    _determine_git_tag(v 25.9.23)

    _save_var(FLATBUFFERS_BUILD_FLATC)
    _save_var(FLATBUFFERS_INSTALL)
    _save_var(FLATBUFFERS_BUILD_FLATLIB)
    _save_var(FLATBUFFERS_BUILD_TESTS)
    _save_var(FLATBUFFERS_BUILD_FLATHASH)
    _save_var(FLATBUFFERS_ENABLE_PCH)

    set(FLATBUFFERS_BUILD_FLATC ON)
    set(FLATBUFFERS_INSTALL ON)
    set(FLATBUFFERS_BUILD_FLATLIB OFF)
    set(FLATBUFFERS_BUILD_TESTS OFF)
    set(FLATBUFFERS_BUILD_FLATHASH OFF)
    set(FLATBUFFERS_ENABLE_PCH ON)

    fetchcontent_declare(
        flatbuffers
        GIT_REPOSITORY https://github.com/google/flatbuffers.git
        GIT_TAG ${GIT_TAG}
        DOWNLOAD_EXTRACT_TIMESTAMP
        TRUE
    )

    fetchcontent_makeavailable(flatbuffers)

    _restore_var(FLATBUFFERS_BUILD_FLATC)
    _restore_var(FLATBUFFERS_INSTALL)
    _restore_var(FLATBUFFERS_BUILD_FLATLIB)
    _restore_var(FLATBUFFERS_BUILD_TESTS)
    _restore_var(FLATBUFFERS_BUILD_FLATHASH)
    _restore_var(FLATBUFFERS_ENABLE_PCH)

    set(HIP_DNN_FLATBUFFERS_INCLUDE_DIR ${flatbuffers_SOURCE_DIR}/include
        CACHE PATH "Path to flatbuffers include"
    )

    _exclude_from_all(${flatbuffers_SOURCE_DIR})
    _mark_targets_as_system(${flatbuffers_SOURCE_DIR})
endfunction()

# Fetches spdlog
function(_fetch_spdlog VERSION HASH)
    _determine_git_tag(v v1.15.3)

    fetchcontent_declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG ${GIT_TAG}
        DOWNLOAD_EXTRACT_TIMESTAMP
        TRUE
    )

    set(_HIPDNN_DISABLE_ROCM_CHECKS TRUE)
    fetchcontent_makeavailable(spdlog)
    set(_HIPDNN_DISABLE_ROCM_CHECKS FALSE)

    set(HIP_DNN_SPDLOG_INCLUDE_DIR ${spdlog_SOURCE_DIR}/include CACHE PATH "Path to spdlog include")

    _exclude_from_all(${spdlog_SOURCE_DIR})
    _mark_targets_as_system(${spdlog_SOURCE_DIR})
endfunction()

# Doesn't conform with the others and ignores the VERSION and HASH arguments, but this will change
# very soon
#
# Fetches nlohmann_json
function(_fetch_nlohmann_json VERSION HASH)
    fetchcontent_declare(
        json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    )

    fetchcontent_makeavailable(json)

    set(HIP_DNN_NLOHMANN_JSON_INCLUDE_DIR ${json_SOURCE_DIR}/include
        CACHE PATH "Path to nlohmann::json include"
    )

    _exclude_from_all(${json_SOURCE_DIR})
    _mark_targets_as_system(${json_SOURCE_DIR})

endfunction()

# Utility functions, pulled from rocroller repo
#
# Determines the git tag to use
macro(_determine_git_tag PREFIX DEFAULT)
    if(HASH)
        set(GIT_TAG ${HASH})
    elseif(VERSION AND NOT "${PREFIX}" STREQUAL "FALSE")
        set(GIT_TAG ${PREFIX}${VERSION})
    else()
        set(GIT_TAG ${DEFAULT})
    endif()
endmacro()

# Saves a variable
macro(_save_var _name)
    if(DEFINED CACHE{${_name}})
        set(_old_cache_${_name} $CACHE{${_name}})
        unset(${_name} CACHE)
    endif()
    # We can't tell if a variable is referring to a cache or a regular variable. To ensure this gets
    # the value of the regular, variable, temporarily unset the cache variable if it was set before
    # checking for the regular variable.
    if(DEFINED ${_name})
        set(_old_${_name} ${${_name}})
    endif()
    if(DEFINED _old_cache_${_name})
        set(${_name} ${_old_cache_${_name}} CACHE INTERNAL "")
    endif()
    if(DEFINED ENV{${_name}})
        set(_old_env_${_name} $ENV{${_name}})
    endif()
endmacro()

# Restores a variable
macro(_restore_var _name)
    if(DEFINED _old_${_name})
        set(${_name} ${_old_${_name}})
        unset(_old_${_name})
    else()
        unset(${_name})
    endif()
    if(DEFINED _old_cache_${_name})
        set(${_name} ${_old_cache_${_name}} CACHE INTERNAL "")
        unset(_old_cache_${_name})
    else()
        unset(${_name} CACHE)
    endif()
    if(DEFINED _old_env_${_name})
        set(ENV{${_name}} ${_old_env_${_name}})
        unset(_old_env_${_name})
    else()
        unset(ENV{${_name}})
    endif()
endmacro()

# not actually a stack, but that shouldn't be relevant
#
# Pushes the current state
macro(_pushstate)
    _save_var(CMAKE_CXX_CPPCHECK)
    unset(CMAKE_CXX_CPPCHECK)
    unset(CMAKE_CXX_CPPCHECK CACHE)
    _save_var(CMAKE_MESSAGE_INDENT)
    _save_var(CPACK_GENERATOR)
endmacro()

# Pops the previous state
macro(_popstate)
    _restore_var(CPACK_GENERATOR)
    _restore_var(CMAKE_MESSAGE_INDENT)
    _restore_var(CMAKE_CXX_CPPCHECK)
endmacro()

# Excludes a directory from all
macro(_exclude_from_all _dir)
    set_property(DIRECTORY ${_dir} PROPERTY EXCLUDE_FROM_ALL ON)
endmacro()

# Marks targets as system
macro(_mark_targets_as_system _dirs)
    foreach(_dir ${_dirs})
        get_directory_property(_targets DIRECTORY ${_dir} BUILDSYSTEM_TARGETS)
        foreach(_target IN LISTS _targets)
            get_target_property(_includes ${_target} INTERFACE_INCLUDE_DIRECTORIES)
            if(_includes)
                target_include_directories(${_target} SYSTEM INTERFACE ${_includes})
            endif()
        endforeach()
    endforeach()
endmacro()
