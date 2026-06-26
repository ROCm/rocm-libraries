# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Shared CMake support for provider-owned rocKE ahead-of-time artifacts.
#
# This module is intentionally source-tree oriented: rocKE itself is consumed
# from ROCKE_CLIENT_ROCKE_SOURCE_DIR/Python, while provider AOT helpers come from
# rocKE-client/aot/python. Build rules pass that import path explicitly to every
# Python invocation instead of requiring the developer shell to be preconfigured.
find_package(Python3 COMPONENTS Interpreter REQUIRED)

get_filename_component(_ROCKE_CLIENT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
if(NOT DEFINED ROCKE_CLIENT_ROCKE_SOURCE_DIR)
    set(ROCKE_CLIENT_ROCKE_SOURCE_DIR "${_ROCKE_CLIENT_ROOT}/../rocKE")
endif()
get_filename_component(_ROCKE_CLIENT_ROCKE_SOURCE_DIR
    "${ROCKE_CLIENT_ROCKE_SOURCE_DIR}" ABSOLUTE
)
set(_ROCKE_CLIENT_ROCKE_PYTHON_ROOT
    "${_ROCKE_CLIENT_ROCKE_SOURCE_DIR}/Python"
)
if(NOT EXISTS "${_ROCKE_CLIENT_ROCKE_PYTHON_ROOT}/rocke")
    message(FATAL_ERROR
        "ROCKE_CLIENT_ROCKE_SOURCE_DIR must point to a rocKE source tree with Python/rocke: "
        "${_ROCKE_CLIENT_ROCKE_SOURCE_DIR}"
    )
endif()

set(ROCKE_CLIENT_ROCKE_SOURCE_DIR "${_ROCKE_CLIENT_ROCKE_SOURCE_DIR}"
    CACHE PATH "Path to the rocKE source tree used by rocKE-client AOT builds" FORCE
)
set(_ROCKE_CLIENT_AOT_BUILD_TOOL "${_ROCKE_CLIENT_ROOT}/aot/tools/rocke_aot_build.py")

# Reconfigure when common AOT Python helpers, JSON Schemas, or rocKE Python
# sources change. Individual kernel families add their handler, schemas, and
# instance JSON files inside rocke_client_add_aot_instances().
file(GLOB_RECURSE _ROCKE_CLIENT_AOT_PACKAGE_MODULES CONFIGURE_DEPENDS
    "${_ROCKE_CLIENT_ROOT}/aot/python/rocke_client_aot/*.py"
)
file(GLOB _ROCKE_CLIENT_AOT_COMMON_SCHEMA_DEPENDS CONFIGURE_DEPENDS
    "${_ROCKE_CLIENT_ROOT}/aot/schemas/*.schema.json"
)

file(GLOB_RECURSE _ROCKE_CLIENT_AOT_COMMON_ROCKE_PYTHON_DEPENDS CONFIGURE_DEPENDS
    "${_ROCKE_CLIENT_ROCKE_PYTHON_ROOT}/rocke/*.py"
)

# Return the PYTHONPATH used by rocKE client AOT tooling in OUT_VAR.
#
# Local source roots are prepended so checked-out rocKE and rocKE-client code
# always win. An incoming developer PYTHONPATH is preserved last for optional
# dependencies supplied outside this tree.
function(rocke_client_aot_pythonpath OUT_VAR)
    set(_ROCKE_CLIENT_AOT_PYTHONPATH
        "${_ROCKE_CLIENT_ROCKE_PYTHON_ROOT}"
        "${_ROCKE_CLIENT_ROOT}/aot/python"
    )
    if(DEFINED ENV{PYTHONPATH} AND NOT "$ENV{PYTHONPATH}" STREQUAL "")
        cmake_path(CONVERT "$ENV{PYTHONPATH}" TO_CMAKE_PATH_LIST
                   _ROCKE_CLIENT_AOT_INCOMING_PYTHONPATH)
        list(APPEND _ROCKE_CLIENT_AOT_PYTHONPATH
             ${_ROCKE_CLIENT_AOT_INCOMING_PYTHONPATH})
    endif()
    cmake_path(CONVERT "${_ROCKE_CLIENT_AOT_PYTHONPATH}" TO_NATIVE_PATH_LIST
               _ROCKE_CLIENT_AOT_PYTHONPATH_NATIVE)
    set(${OUT_VAR} "${_ROCKE_CLIENT_AOT_PYTHONPATH_NATIVE}" PARENT_SCOPE)
endfunction()

# Return the CMake -E env / CTest ENVIRONMENT entries for AOT Python commands.
#
# PYTHONDONTWRITEBYTECODE keeps configure/build/test runs from writing __pycache__
# into source trees, which matters because generated artifacts live in the build
# tree and the source tree should stay reviewable.
function(rocke_client_aot_pythonpath_environment OUT_VAR)
    rocke_client_aot_pythonpath(_ROCKE_CLIENT_AOT_PYTHONPATH_NATIVE)
    string(REPLACE ";" "\\;" _ROCKE_CLIENT_AOT_PYTHONPATH_ESCAPED
           "${_ROCKE_CLIENT_AOT_PYTHONPATH_NATIVE}")
    set(${OUT_VAR}
        "PYTHONPATH=${_ROCKE_CLIENT_AOT_PYTHONPATH_ESCAPED}"
        "PYTHONDONTWRITEBYTECODE=1"
        PARENT_SCOPE
    )
endfunction()

# Register one AOT kernel instance set.
#
# Required arguments:
#   NAME         Target name and final artifact directory component.
#   ARCH         rocKE architecture component, e.g. gfx942 or gfx1151.
#   INSTANCE_DIR Directory containing checked-in *.instance.json files.
#
# Optional arguments:
#   PYTHON_DEPENDS Extra Python files whose edits should rebuild the artifacts.
function(rocke_client_add_aot_instances)
    cmake_parse_arguments(ARG "" "NAME;ARCH;INSTANCE_DIR" "PYTHON_DEPENDS" ${ARGN})
    if(NOT ARG_NAME OR NOT ARG_ARCH OR NOT ARG_INSTANCE_DIR)
        message(FATAL_ERROR
            "rocke_client_add_aot_instances requires NAME, ARCH, and INSTANCE_DIR"
        )
    endif()

    if(NOT TARGET rocke_client_aot_artifacts)
        message(FATAL_ERROR
            "Create rocke_client_aot_artifacts before calling rocke_client_add_aot_instances"
        )
    endif()

    get_filename_component(_ROCKE_CLIENT_AOT_INSTANCE_DIR
        "${ARG_INSTANCE_DIR}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    if(NOT IS_DIRECTORY "${_ROCKE_CLIENT_AOT_INSTANCE_DIR}")
        message(FATAL_ERROR
            "rocKE client AOT instance directory does not exist: ${_ROCKE_CLIENT_AOT_INSTANCE_DIR}"
        )
    endif()

    # Treat the checked-in instance JSON files as source inputs. CONFIGURE_DEPENDS
    # keeps target membership current when instances are added or removed.
    file(GLOB _ROCKE_CLIENT_AOT_INSTANCE_SOURCES CONFIGURE_DEPENDS
        "${_ROCKE_CLIENT_AOT_INSTANCE_DIR}/*.instance.json"
    )
    if(NOT _ROCKE_CLIENT_AOT_INSTANCE_SOURCES)
        message(FATAL_ERROR
            "No rocKE client AOT instances found in ${_ROCKE_CLIENT_AOT_INSTANCE_DIR}"
        )
    endif()
    # A kernel-family directory owns its operation-specific handler and optional
    # JSON Schema overlays. The common build tool imports the handler at runtime.
    set(_ROCKE_CLIENT_AOT_KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    file(GLOB _ROCKE_CLIENT_AOT_KERNEL_SCHEMA_DEPENDS CONFIGURE_DEPENDS
        "${_ROCKE_CLIENT_AOT_KERNEL_DIR}/schemas/*.schema.json"
    )
    set(_ROCKE_CLIENT_AOT_KERNEL_HANDLER
        "${_ROCKE_CLIENT_AOT_KERNEL_DIR}/aot_instance.py"
    )
    if(NOT EXISTS "${_ROCKE_CLIENT_AOT_KERNEL_HANDLER}")
        message(FATAL_ERROR
            "rocKE client AOT kernel directory is missing aot_instance.py: ${_ROCKE_CLIENT_AOT_KERNEL_DIR}"
        )
    endif()

    rocke_client_aot_pythonpath_environment(_ROCKE_CLIENT_AOT_BUILD_ENVIRONMENT)

    set(_ROCKE_CLIENT_AOT_OUTPUT_ROOT "${PROJECT_BINARY_DIR}/rocKE-client/aot")
    set(_ROCKE_CLIENT_AOT_ARCH_OUTPUT_DIR
        "${_ROCKE_CLIENT_AOT_OUTPUT_ROOT}/${ARG_ARCH}/${ARG_NAME}"
    )
    set(_ROCKE_CLIENT_AOT_BUILD_STAMP
        "${_ROCKE_CLIENT_AOT_ARCH_OUTPUT_DIR}/build.stamp"
    )
    # Keep a manifest file as a stable dependency for the set of instance files.
    # It gives the custom command one dependency that changes when membership or
    # absolute source paths change, while each JSON file remains a direct input.
    set(_ROCKE_CLIENT_AOT_INSTANCE_MANIFEST
        "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${ARG_NAME}.instances.manifest"
    )
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles")
    file(WRITE "${_ROCKE_CLIENT_AOT_INSTANCE_MANIFEST}" "")
    set(_ROCKE_CLIENT_AOT_GENERATED_OUTPUTS)
    foreach(_ROCKE_CLIENT_AOT_INSTANCE_SOURCE IN LISTS _ROCKE_CLIENT_AOT_INSTANCE_SOURCES)
        file(APPEND "${_ROCKE_CLIENT_AOT_INSTANCE_MANIFEST}"
             "${_ROCKE_CLIENT_AOT_INSTANCE_SOURCE}\n")
        get_filename_component(_ROCKE_CLIENT_AOT_INSTANCE_FILE
            "${_ROCKE_CLIENT_AOT_INSTANCE_SOURCE}" NAME
        )
        string(REGEX REPLACE "\\.instance\\.json$" ""
               _ROCKE_CLIENT_AOT_INSTANCE_BASENAME
               "${_ROCKE_CLIENT_AOT_INSTANCE_FILE}")
        list(APPEND _ROCKE_CLIENT_AOT_GENERATED_OUTPUTS
             "${_ROCKE_CLIENT_AOT_ARCH_OUTPUT_DIR}/${_ROCKE_CLIENT_AOT_INSTANCE_BASENAME}.hsaco"
             "${_ROCKE_CLIENT_AOT_ARCH_OUTPUT_DIR}/${_ROCKE_CLIENT_AOT_INSTANCE_BASENAME}.sidecar.json")
    endforeach()

    # Recreate the artifact directory on every rebuild so removed/renamed
    # instances cannot leave stale HSACO or sidecar files behind.
    add_custom_command(
        OUTPUT "${_ROCKE_CLIENT_AOT_BUILD_STAMP}"
               ${_ROCKE_CLIENT_AOT_GENERATED_OUTPUTS}
        COMMAND "${CMAKE_COMMAND}" -E remove_directory "${_ROCKE_CLIENT_AOT_ARCH_OUTPUT_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_ROCKE_CLIENT_AOT_ARCH_OUTPUT_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                ${_ROCKE_CLIENT_AOT_INSTANCE_SOURCES}
                "${_ROCKE_CLIENT_AOT_ARCH_OUTPUT_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E env
                ${_ROCKE_CLIENT_AOT_BUILD_ENVIRONMENT}
                "${Python3_EXECUTABLE}"
                "${_ROCKE_CLIENT_AOT_BUILD_TOOL}"
                --artifact-dir "${_ROCKE_CLIENT_AOT_ARCH_OUTPUT_DIR}"
                --kernel-dir "${_ROCKE_CLIENT_AOT_KERNEL_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${_ROCKE_CLIENT_AOT_BUILD_STAMP}"
        DEPENDS
                "${_ROCKE_CLIENT_AOT_KERNEL_HANDLER}"
                ${_ROCKE_CLIENT_AOT_INSTANCE_SOURCES}
                "${_ROCKE_CLIENT_AOT_INSTANCE_MANIFEST}"
                "${_ROCKE_CLIENT_AOT_BUILD_TOOL}"
                ${_ROCKE_CLIENT_AOT_PACKAGE_MODULES}
                ${_ROCKE_CLIENT_AOT_COMMON_SCHEMA_DEPENDS}
                ${_ROCKE_CLIENT_AOT_KERNEL_SCHEMA_DEPENDS}
                ${_ROCKE_CLIENT_AOT_COMMON_ROCKE_PYTHON_DEPENDS}
                ${ARG_PYTHON_DEPENDS}
        VERBATIM
        COMMENT "Build rocKE client ${ARG_ARCH} ${ARG_NAME} AOT artifacts"
    )

    add_custom_target("${ARG_NAME}"
        DEPENDS "${_ROCKE_CLIENT_AOT_BUILD_STAMP}"
                ${_ROCKE_CLIENT_AOT_GENERATED_OUTPUTS}
        COMMENT "Build ${ARG_NAME} AOT artifacts"
    )
    add_dependencies(rocke_client_aot_artifacts "${ARG_NAME}")
endfunction()
