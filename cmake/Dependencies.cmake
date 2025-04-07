
################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

cmake_minimum_required(VERSION 3.21...3.22)

include(FetchContent)

option(ROCROLLER_NO_DOWNLOAD
       "Disables downloading of any external dependencies" OFF
)

if(ROCROLLER_NO_DOWNLOAD)
    set(FETCHCONTENT_FULLY_DISCONNECTED
        OFF
        CACHE BOOL "Don't attempt to download or update anything" FORCE
    )
endif()

# Dependencies where the local version should be used, if available
set(_rocroller_all_local_deps
    spdlog
    msgpack
    Catch2
    CLI11
    libdivide
    boost
    ROCmCMakeBuildTools
)
# Dependencies where we never look for a local version
set(_rocroller_all_remote_deps
    fmt
    yaml-cpp
    isa_spec_manager
    mrisa_xml
    googletest
    mxDataGenerator
    cmrc
)

# rocroller_add_dependency(
#   dep_name
#   [NO_LOCAL]
#   [VERSION version]
#   [FIND_PACKAGE_ARGS args...]
#   [COMPONENT component]
#   [PACKAGE_NAME
#      [package_name]
#      [DEB deb_package_name]
#      [RPM rpm_package_name]]
# )
function(rocroller_add_dependency dep_name)
    set(options NO_LOCAL)
    set(oneValueArgs VERSION HASH)
    set(multiValueArgs FIND_PACKAGE_ARGS PACKAGE_NAME COMPONENTS)
    cmake_parse_arguments(
        PARSE
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )

    if(dep_name IN_LIST _rocroller_all_local_deps)
        if(NOT PARSE_NO_LOCAL)
            find_package(
                ${dep_name} ${PARSE_VERSION} QUIET ${PARSE_FIND_PACKAGE_ARGS}
            )
        endif()
        if(NOT ${dep_name}_FOUND)
            message(STATUS "Did not find ${dep_name}, it will be built locally")
            _build_local()
        else()
            message(
                STATUS
                    "Found ${dep_name}: ${${dep_name}_DIR} (found version \"${${dependency_name}_VERSION}\")"
            )
            foreach(VAR IN LISTS ${dep_name}_EXPORT_VARS)
                set(${VAR}
                    ${${VAR}}
                    PARENT_SCOPE
                )
            endforeach()
        endif()
    elseif(dep_name IN_LIST _rocroller_all_remote_deps)
        message(TRACE "Will build ${dep_name} locally")
        _build_local()
    else()
        message(WARNING "Unknown dependency: ${dep_name}")
        return()
    endif()

    if(PARSE_COMPONENTS AND PROJECT_IS_TOP_LEVEL)
        if(COMMAND rocm_package_add_dependencies)
            unset(name_deb)
            unset(name_rpm)
            if(PARSE_PACKAGE_NAME)
                cmake_parse_arguments(
                    PKG_NAME
                    ""
                    "DEB;RPM"
                    ""
                    ${PARSE_PACKAGE_NAME}
                )
                if(PKG_NAME_DEB)
                    set(name_deb "${PKG_NAME_DEB}")
                elseif(PKG_NAME_UNPARSED_ARGUMENTS)
                    set(name_deb "${PKG_NAME_UNPARSED_ARGUMENTS}")
                endif()
                if(PKG_NAME_RPM)
                    set(name_rpm "${PKG_NAME_RPM}")
                elseif(PKG_NAME_UNPARSED_ARGUMENTS)
                    set(name_rpm "${PKG_NAME_UNPARSED_ARGUMENTS}")
                endif()
            endif()
            foreach(COMPONENT IN LISTS PARSE_COMPONENTS)
                if("${name_deb}" STREQUAL "")
                    set(name_deb "lib${dep_name}-dev")
                endif()
                if("${name_rpm}" STREQUAL "")
                    set(name_rpm "lib${dep_name}-devel")
                endif()
                if(PARSE_VERSION)
                    set(VERSION_STR " >= ${PARSE_VERSION}")
                endif()
                rocm_package_add_deb_dependencies(
                    QUIET
                    COMPONENT ${COMPONENT}
                    DEPENDS "${name_deb}${VERSION_STR}"
                )
                rocm_package_add_rpm_dependencies(
                    QUIET
                    COMPONENT ${COMPONENT}
                    DEPENDS "${name_rpm}${VERSION_STR}"
                )
                string(TOUPPER "${COMPONENT}" COMPONENT_VAR)
                set(CPACK_DEBIAN_${COMPONENT_VAR}_PACKAGE_DEPENDS
                    "${CPACK_DEBIAN_${COMPONENT_VAR}_PACKAGE_DEPENDS}"
                    PARENT_SCOPE
                )
                set(CPACK_RPM_${COMPONENT_VAR}_PACKAGE_REQUIRES
                    "${CPACK_RPM_${COMPONENT_VAR}_PACKAGE_REQUIRES}"
                    PARENT_SCOPE
                )
            endforeach()
        else()
            message(
                ERROR
                "ROCmCMakeBuildTools is required to add dependencies to a package"
            )
        endif()
    endif()
endfunction()

macro(_build_local)
    cmake_policy(PUSH)
    if(BUILD_VERBOSE)
        message(STATUS "=========== Adding ${dep_name} ===========")
    endif()
    _pushstate()
    set(CMAKE_MESSAGE_INDENT "[${dep_name}] ")
    cmake_language(
        CALL _fetch_${dep_name} "${PARSE_VERSION}" "${PARSE_HASH}"
    )
    _popstate()
    if(BUILD_VERBOSE)
        message(STATUS "=========== Added ${dep_name} ===========")
    endif()
    cmake_policy(POP)
    foreach(VAR IN LISTS ${dep_name}_EXPORT_VARS)
        set(${VAR}
            ${${VAR}}
            PARENT_SCOPE
        )
    endforeach()
endmacro()

# Functions to fetch individual components
function(_fetch_fmt VERSION HASH)
    _determine_git_tag("" master)

    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(FMT_SYSTEM_HEADERS ON)

    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG ${GIT_TAG}
    )
    FetchContent_MakeAvailable(fmt)
    _exclude_from_all(${fmt_SOURCE_DIR})
endfunction()

function(_fetch_spdlog VERSION HASH)
    _determine_git_tag(v v1.x)

    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(SPDLOG_USE_STD_FORMAT OFF)
    set(SPDLOG_FMT_EXTERNAL_HO ON)
    set(SPDLOG_BUILD_PIC ON)
    set(SPDLOG_INSTALL ON)
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG ${GIT_TAG}
    )
    FetchContent_MakeAvailable(spdlog)
    set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)
    _exclude_from_all(${spdlog_SOURCE_DIR})
    _mark_targets_as_system(${spdlog_SOURCE_DIR})
endfunction()

function(_fetch_msgpack VERSION HASH)
    _determine_git_tag(cpp- cpp-master)

    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)
    set(MSGPACK_USE_BOOST OFF)
    set(MSGPACK_BOOST OFF) # for earlier versions of msgpack
    set(MSGPACK_BUILD_EXAMPLES OFF) # for earlier versions of msgpack
    FetchContent_Declare(
        msgpack
        GIT_REPOSITORY https://github.com/msgpack/msgpack-c.git
        GIT_TAG ${GIT_TAG}
    )
    FetchContent_MakeAvailable(msgpack)
    set_property(
        TARGET msgpackc-cxx
        APPEND
        PROPERTY COMPILE_DEFINITIONS MSGPACK_NO_BOOST
    )
    set_property(
        TARGET msgpackc-cxx
        APPEND
        PROPERTY INTERFACE_COMPILE_DEFINITIONS MSGPACK_NO_BOOST
    )

    _exclude_from_all(${msgpack_SOURCE_DIR})
    _mark_targets_as_system(${msgpack_SOURCE_DIR})
endfunction()

function(_fetch_yaml-cpp VERSION HASH)
    if(VERSION VERSION_LESS 0.8.0)
        _determine_git_tag(yaml-cpp- master)
    else()
        _determine_git_tag("" master)
    endif()

    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(YAML_CPP_BUILD_TOOLS OFF)
    set(YAML_CPP_INSTALL ON)
    set(YAML_BUILD_SHARED_LIBS OFF)
    FetchContent_Declare(
        yaml_cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG ${GIT_TAG}
    )
    FetchContent_MakeAvailable(yaml_cpp)
    install(
        TARGETS yaml-cpp
        EXPORT yaml-cpp-targets
        COMPONENT dummy
    )
    export(EXPORT yaml-cpp-targets
           FILE "${yaml_cpp_BINARY_DIR}/yaml-cpp-targets.cmake"
    )
    install(
        EXPORT yaml-cpp-targets
        FILE yaml-cpp-targets.cmake
        DESTINATION "${CMAKE_INSTALL_DATADIR}"
        COMPONENT dummy
    )
    _exclude_from_all(${yaml_cpp_SOURCE_DIR})
    _mark_targets_as_system(${yaml_cpp_SOURCE_DIR})
    target_compile_options(yaml-cpp PUBLIC ${EXTRA_COMPILE_OPTIONS})
    target_link_options(yaml-cpp PUBLIC ${EXTRA_LINK_OPTIONS})
    set_target_properties(yaml-cpp PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_property(
        TARGET yaml-cpp
        APPEND
        PROPERTY COMPILE_OPTIONS "-Wno-shadow"
    )
endfunction()

function(_fetch_isa_spec_manager VERSION HASH)
    _determine_git_tag(v main)

    FetchContent_Declare(
        isa_spec_manager
        GIT_REPOSITORY https://github.com/GPUOpen-Tools/isa_spec_manager.git
        GIT_TAG ${GIT_TAG}
    )
    FetchContent_MakeAvailable(isa_spec_manager)
    _exclude_from_all(${isa_spec_manager_SOURCE_DIR})
    add_library(
        isa_spec_manager OBJECT
        ${isa_spec_manager_SOURCE_DIR}/source/common/isa_xml_reader.cpp
        ${isa_spec_manager_SOURCE_DIR}/source/third_party/tinyxml2/tinyxml2.cpp
        ${isa_spec_manager_SOURCE_DIR}/source/common/amdisa_utility.cpp
    )
    target_include_directories(
        isa_spec_manager SYSTEM
        PUBLIC ${isa_spec_manager_SOURCE_DIR}/include
               ${isa_spec_manager_SOURCE_DIR}/source/common
               ${isa_spec_manager_SOURCE_DIR}/source/third_party/tinyxml2
    )
endfunction()

function(_fetch_mrisa_xml VERSION HASH)
    if(VERSION)
        set(MRISA_VERSION AMD_GPU_MR_ISA_XML-${VERSION}.zip)
    else()
        set(MRISA_VERSION latest)
    endif()
    if(HASH)
        set(HASH_ARG HASH ${HASH})
    endif()
    FetchContent_Declare(
        mrisa_xml
        URL https://gpuopen.com/download/machine-readable-isa/${MRISA_VERSION}
            ${HASH_ARG}
    )
    FetchContent_MakeAvailable(mrisa_xml)
    set(mrisa_xml_SOURCE_DIR
        ${mrisa_xml_SOURCE_DIR}
        PARENT_SCOPE
    )
    set(EXPORT_VARS
        mrisa_xml_SOURCE_DIR
        PARENT_SCOPE
    )
endfunction()
set(mrisa_xml_EXPORT_VARS mrisa_xml_SOURCE_DIR)

function(_fetch_boost VERSION HASH)
    _determine_git_tag("boost-" "boost-1.81.0")
    FetchContent_Declare(
        boost
        URL https://github.com/boostorg/boost/releases/download/${GIT_TAG}/${GIT_TAG}.tar.gz
    )
    _save_var(BUILD_TESTING)
    set(BUILD_TESTING OFF)
    _save_var(BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS OFF)
    set(Boost_USE_STATIC_LIBS ON)
    FetchContent_MakeAvailable(boost)
    _restore_var(BUILD_SHARED_LIBS)
    _restore_var(BUILD_TESTING)
    _exclude_from_all(${boost_SOURCE_DIR})
    _mark_targets_as_system(${boost_SOURCE_DIR})
endfunction()

function(_fetch_cmrc VERSION HASH)
    _determine_git_tag("" master)
    FetchContent_Declare(
        cmrc
        GIT_REPOSITORY https://github.com/vector-of-bool/cmrc.git
        GIT_TAG ${GIT_TAG}
    )
    FetchContent_MakeAvailable(cmrc)
endfunction()

function(_fetch_libdivide VERSION HASH)
    _determine_git_tag(v master)

    # We want to include the header as system, and don't want all of libdivide's
    # configuration. Therefore we define our own libraries with the same names
    # as those in libdivide for compatibility purposes
    FetchContent_Declare(
        libdivide
        GIT_REPOSITORY https://github.com/ridiculousfish/libdivide.git
        GIT_TAG ${GIT_TAG}
    )
    if(NOT libdivide_POPULATED)
        FetchContent_Populate(libdivide)
    endif()
    add_library(libdivide INTERFACE)
    add_library(libdivide::libdivide ALIAS libdivide)
    target_include_directories(
        libdivide SYSTEM INTERFACE ${libdivide_SOURCE_DIR}
    )
endfunction()

function(_fetch_googletest VERSION HASH)
    if(VERSION AND VERSION STREQUAL 1.12.1)
        set(GIT_TAG release-1.12.1)
    else()
        _determine_git_tag(v release-1.12.1)
    endif()
    if(HASH)
        set(HASH_ARG HASH ${HASH})
    endif()
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/${GIT_TAG}.zip
        ${HASH_ARG}
    )

    option(ROCROLLER_GTEST_SHARED "Build GTest as a shared library." OFF)
    _save_var(BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS
        ${ROCROLLER_GTEST_SHARED}
        CACHE INTERNAL ""
    )
    set(INSTALL_GTEST OFF)
    FetchContent_MakeAvailable(googletest)
    _restore_var(BUILD_SHARED_LIBS)

    _exclude_from_all(${googletest_SOURCE_DIR})
    _mark_targets_as_system(${googletest_SOURCE_DIR})
    target_compile_options(gtest PUBLIC ${EXTRA_COMPILE_OPTIONS})
    target_compile_options(gmock PUBLIC ${EXTRA_COMPILE_OPTIONS})
endfunction()

function(_fetch_Catch2 VERSION HASH)
    _determine_git_tag(v devel)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG ${GIT_TAG}
    )
    option(ROCROLLER_CATCH_SHARED "Build Catch2 as a shared library." OFF)
    _save_var(BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS
        ${ROCROLLER_CATCH_SHARED}
        CACHE INTERNAL ""
    )
    FetchContent_MakeAvailable(Catch2)
    _restore_var(BUILD_SHARED_LIBS)

    target_compile_options(Catch2 PRIVATE ${EXTRA_COMPILE_OPTIONS})
    target_compile_options(Catch2WithMain PRIVATE ${EXTRA_COMPILE_OPTIONS})
    target_link_options(Catch2 PRIVATE ${EXTRA_LINK_OPTIONS})
    target_link_options(Catch2WithMain PRIVATE ${EXTRA_LINK_OPTIONS})
    list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
    set(CMAKE_MODULE_PATH
        ${CMAKE_MODULE_PATH}
        PARENT_SCOPE
    )

    _exclude_from_all(${Catch2_SOURCE_DIR})
    _mark_targets_as_system(${Catch2_SOURCE_DIR})
endfunction()
set(Catch2_EXPORT_VARS CMAKE_MODULE_PATH)

function(_fetch_CLI11 VERSION HASH)
    _determine_git_tag(v main)
    FetchContent_Declare(
        CLI11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG ${GIT_TAG}
    )
    FetchContent_MakeAvailable(CLI11)
endfunction()

function(_fetch_mxDataGenerator VERSION HASH)
    _determine_git_tag(v main)
    if(MXDATAGENERATOR_SSH)
        set(mxDataGenerator_url "git@${MXDATAGENERATOR_GIT_URL}:ROCm/mxDataGenerator.git")
    else()
        set(mxDataGenerator_url "https://${MXDATAGENERATOR_GIT_URL}/ROCm/mxDataGenerator.git")
    endif()
    FetchContent_Declare(
        mxDataGenerator
        GIT_REPOSITORY ${mxDataGenerator_url}
        GIT_TAG ${GIT_TAG}
    )
    FetchContent_MakeAvailable(mxDataGenerator)
    _mark_targets_as_system(${FPDataGeneration_SOURCE_DIR})
endfunction()

function(_fetch_ROCmCMakeBuildTools VERSION HASH)
    _determine_git_tag(FALSE rocm-6.3.0)
    FetchContent_Declare(
        ROCmCMakeBuildTools
        GIT_REPOSITORY https://github.com/ROCm/rocm-cmake.git
        GIT_TAG ${GIT_TAG}
        SOURCE_SUBDIR "DISABLE ADDING TO BUILD"
        # Don't consume the build/test targets of ROCmCMakeBuildTools
    )
    FetchContent_MakeAvailable(ROCmCMakeBuildTools)
    list(
        APPEND CMAKE_MODULE_PATH
        ${rocmcmakebuildtools_SOURCE_DIR}/share/rocmcmakebuildtools/cmake
    )
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} PARENT_SCOPE)
endfunction()
set(ROCmCMakeBuildTools_EXPORT_VARS CMAKE_MODULE_PATH)

# Utility functions
macro(_determine_git_tag PREFIX DEFAULT)
    if(HASH)
        set(GIT_TAG ${HASH})
    elseif(VERSION AND NOT "${PREFIX}" STREQUAL "FALSE")
        set(GIT_TAG ${PREFIX}${VERSION})
    else()
        set(GIT_TAG ${DEFAULT})
    endif()
endmacro()

macro(_save_var _name)
    if(DEFINED CACHE{${_name}})
        set(_old_cache_${_name} $CACHE{${_name}})
        unset(${_name} CACHE)
    endif()
    # We can't tell if a variable is referring to a cache or a regular variable.
    # To ensure this gets the value of the regular, variable, temporarily unset
    # the cache variable if it was set before checking for the regular variable.
    if(DEFINED ${_name})
        set(_old_${_name} ${${_name}})
    endif()
    if(DEFINED _old_cache_${_name})
        set(${_name}
            ${_old_cache_${_name}}
            CACHE INTERNAL ""
        )
    endif()
    if(DEFINED ENV{${_name}})
        set(_old_env_${_name} $ENV{${_name}})
    endif()
endmacro()

macro(_restore_var _name)
    if(DEFINED _old_${_name})
        set(${_name} ${_old_${_name}})
        unset(_old_${_name})
    else()
        unset(${_name})
    endif()
    if(DEFINED _old_cache_${_name})
        set(${_name}
            ${_old_cache_${_name}}
            CACHE INTERNAL ""
        )
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

## not actually a stack, but that shouldn't be relevant
macro(_pushstate)
    _save_var(CMAKE_CXX_CPPCHECK)
    unset(CMAKE_CXX_CPPCHECK)
    unset(CMAKE_CXX_CPPCHECK CACHE)
    _save_var(CMAKE_MESSAGE_INDENT)
    _save_var(CPACK_GENERATOR)
endmacro()

macro(_popstate)
    _restore_var(CPACK_GENERATOR)
    _restore_var(CMAKE_MESSAGE_INDENT)
    _restore_var(CMAKE_CXX_CPPCHECK)
endmacro()

macro(_exclude_from_all _dir)
    set_property(DIRECTORY ${_dir} PROPERTY EXCLUDE_FROM_ALL ON)
endmacro()

macro(_mark_targets_as_system _dirs)
    foreach(_dir ${_dirs})
        get_directory_property(_targets DIRECTORY ${_dir} BUILDSYSTEM_TARGETS)
        foreach(_target IN LISTS _targets)
            get_target_property(
                _includes ${_target} INTERFACE_INCLUDE_DIRECTORIES
            )
            if(_includes)
                target_include_directories(
                    ${_target} SYSTEM INTERFACE ${_includes}
                )
            endif()
        endforeach()
    endforeach()
endmacro()
