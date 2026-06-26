# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(Python3 COMPONENTS Interpreter REQUIRED)

get_filename_component(_ROCKE_CLIENT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
get_filename_component(_HIP_KERNEL_PROVIDER_ROOT "${_ROCKE_CLIENT_ROOT}/.." ABSOLUTE)

set(_ROCKE_CLIENT_AOT_BUILD_TOOL "${_ROCKE_CLIENT_ROOT}/aot/tools/rocke_aot_build.py")

file(GLOB_RECURSE _ROCKE_CLIENT_AOT_PACKAGE_MODULES CONFIGURE_DEPENDS
    "${_ROCKE_CLIENT_ROOT}/aot/python/rocke_client_aot/*.py"
)

set(_ROCKE_CLIENT_AOT_ROCKE_PYTHON_DEPENDS
    "${_HIP_KERNEL_PROVIDER_ROOT}/rocKE/Python/rocke/instances/common/fmha_mfma.py"
    "${_HIP_KERNEL_PROVIDER_ROOT}/rocKE/Python/rocke/instances/common/_fmha_common.py"
    "${_HIP_KERNEL_PROVIDER_ROOT}/rocKE/Python/rocke/instances/__init__.py"
    "${_HIP_KERNEL_PROVIDER_ROOT}/rocKE/Python/rocke/helpers/compile.py"
    "${_HIP_KERNEL_PROVIDER_ROOT}/rocKE/Python/rocke/core/arch/__init__.py"
    "${_HIP_KERNEL_PROVIDER_ROOT}/rocKE/Python/rocke/core/arch/target.py"
)

# Build the Python search path used by rocKE client AOT tooling.
function(rocke_client_aot_pythonpath OUT_VAR)
    set(_ROCKE_CLIENT_AOT_PYTHONPATH
        "${_HIP_KERNEL_PROVIDER_ROOT}/rocKE/Python"
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

# Build the test and command environment for rocKE client AOT Python invocations.
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

# Add a custom target that generates AOT artifacts for a kernel instance set.
function(rocke_client_add_aot_instances)
    cmake_parse_arguments(ARG "" "NAME;ARCH;INSTANCE_DIR" "" ${ARGN})
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

    file(GLOB _ROCKE_CLIENT_AOT_INSTANCE_SOURCES CONFIGURE_DEPENDS
        "${_ROCKE_CLIENT_AOT_INSTANCE_DIR}/*.instance.json"
    )
    if(NOT _ROCKE_CLIENT_AOT_INSTANCE_SOURCES)
        message(FATAL_ERROR
            "No rocKE client AOT instances found in ${_ROCKE_CLIENT_AOT_INSTANCE_DIR}"
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
    set(_ROCKE_CLIENT_AOT_INSTANCE_MANIFEST
        "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${ARG_NAME}.instances.manifest"
    )
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles")
    file(WRITE "${_ROCKE_CLIENT_AOT_INSTANCE_MANIFEST}" "")
    foreach(_ROCKE_CLIENT_AOT_INSTANCE_SOURCE IN LISTS _ROCKE_CLIENT_AOT_INSTANCE_SOURCES)
        file(APPEND "${_ROCKE_CLIENT_AOT_INSTANCE_MANIFEST}"
             "${_ROCKE_CLIENT_AOT_INSTANCE_SOURCE}\n")
    endforeach()

    add_custom_command(
        OUTPUT "${_ROCKE_CLIENT_AOT_BUILD_STAMP}"
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
        COMMAND "${CMAKE_COMMAND}" -E touch "${_ROCKE_CLIENT_AOT_BUILD_STAMP}"
        DEPENDS
                ${_ROCKE_CLIENT_AOT_INSTANCE_SOURCES}
                "${_ROCKE_CLIENT_AOT_INSTANCE_MANIFEST}"
                "${_ROCKE_CLIENT_AOT_BUILD_TOOL}"
                ${_ROCKE_CLIENT_AOT_PACKAGE_MODULES}
                ${_ROCKE_CLIENT_AOT_ROCKE_PYTHON_DEPENDS}
        VERBATIM
        COMMENT "Build rocKE client ${ARG_ARCH} ${ARG_NAME} AOT artifacts"
    )

    add_custom_target("${ARG_NAME}"
        DEPENDS "${_ROCKE_CLIENT_AOT_BUILD_STAMP}"
        COMMENT "Build ${ARG_NAME} AOT artifacts"
    )
    add_dependencies(rocke_client_aot_artifacts "${ARG_NAME}")
endfunction()
