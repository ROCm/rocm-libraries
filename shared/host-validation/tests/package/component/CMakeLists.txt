# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(ROCHostValidationComponentPackageTest LANGUAGES CXX)

foreach(required_variable
        ROCHostValidation_TEST_COMPONENT
        ROCHostValidation_TEST_EXPECTED_COMPONENTS)
    if(NOT DEFINED ${required_variable})
        message(FATAL_ERROR "${required_variable} is required.")
    endif()
endforeach()

find_package(
    ROCHostValidation
    CONFIG
    REQUIRED
    COMPONENTS "${ROCHostValidation_TEST_COMPONENT}"
)

set(_expected_Core_target roc::host-validation-core)
set(_expected_Core_dependencies)
set(_expected_Operations_target roc::host-validation)
set(_expected_Operations_dependencies Core)
set(_expected_Blocked_target roc::host-validation-blocked)
set(_expected_Blocked_dependencies Operations)
set(_expected_BLAS_target roc::host-validation-blas)
set(_expected_BLAS_dependencies Operations)
set(_expected_MX_target roc::host-validation-mx)
set(_expected_MX_dependencies Core)
set(
    _expected_AMDGPULayout_target
    roc::host-validation-amd-gpu-layout
)
set(_expected_AMDGPULayout_dependencies)

if(NOT "${ROCHostValidation_SUPPORTED_COMPONENTS}"
   STREQUAL "Core;Operations;Blocked;BLAS;MX;AMDGPULayout")
    message(FATAL_ERROR
        "Unexpected supported-component list: "
        "${ROCHostValidation_SUPPORTED_COMPONENTS}"
    )
endif()

string(
    REPLACE
    ","
    ";"
    _expected_components
    "${ROCHostValidation_TEST_EXPECTED_COMPONENTS}"
)
foreach(component IN LISTS ROCHostValidation_SUPPORTED_COMPONENTS)
    set(_target_variable "ROCHostValidation_${component}_TARGET")
    set(
        _dependencies_variable
        "ROCHostValidation_${component}_COMPONENT_DEPENDENCIES"
    )
    if(NOT "${${_target_variable}}"
       STREQUAL "${_expected_${component}_target}")
        message(FATAL_ERROR
            "Component ${component} maps to ${${_target_variable}}; expected "
            "${_expected_${component}_target}."
        )
    endif()
    if(NOT "${${_dependencies_variable}}"
       STREQUAL "${_expected_${component}_dependencies}")
        message(FATAL_ERROR
            "Component ${component} requires ${${_dependencies_variable}}; "
            "expected ${_expected_${component}_dependencies}."
        )
    endif()

    if(component IN_LIST _expected_components)
        if(NOT ROCHostValidation_${component}_FOUND)
            message(FATAL_ERROR
                "Expected transitive component ${component} was not found."
            )
        endif()
        if(NOT TARGET "${${_target_variable}}")
            message(FATAL_ERROR
                "Expected target ${${_target_variable}} was not loaded."
            )
        endif()
    else()
        if(ROCHostValidation_${component}_FOUND)
            message(FATAL_ERROR
                "Unrequested component ${component} was marked found."
            )
        endif()
        if(TARGET "${${_target_variable}}")
            message(FATAL_ERROR
                "Unrequested target ${${_target_variable}} was loaded."
            )
        endif()
    endif()
endforeach()

if(ROCHostValidation_TEST_COMPONENT STREQUAL "Core")
    set(_test_source "${CMAKE_CURRENT_LIST_DIR}/../core_only/main.cpp")
elseif(ROCHostValidation_TEST_COMPONENT STREQUAL "Operations")
    set(_test_source "${CMAKE_CURRENT_LIST_DIR}/../validation.cpp")
elseif(ROCHostValidation_TEST_COMPONENT STREQUAL "Blocked")
    set(_test_source "${CMAKE_CURRENT_LIST_DIR}/../blocked.cpp")
elseif(ROCHostValidation_TEST_COMPONENT STREQUAL "BLAS")
    set(_test_source "${CMAKE_CURRENT_LIST_DIR}/../blas.cpp")
    if(NOT "${ROCHostValidation_BLAS_INTEGER_SIZE}" STREQUAL "4")
        message(FATAL_ERROR
            "The installed BLAS component did not report its LP64 integer ABI."
        )
    endif()
    if(NOT ROCHostValidation_BLAS_BUILD_PROVIDER)
        message(FATAL_ERROR
            "The installed BLAS component did not report its build provider."
        )
    endif()
elseif(ROCHostValidation_TEST_COMPONENT STREQUAL "MX")
    set(_test_source "${CMAKE_CURRENT_LIST_DIR}/../mx.cpp")
elseif(ROCHostValidation_TEST_COMPONENT STREQUAL "AMDGPULayout")
    set(_test_source "${CMAKE_CURRENT_LIST_DIR}/../amd_gpu_layout.cpp")
else()
    message(FATAL_ERROR
        "No package test source is defined for "
        "${ROCHostValidation_TEST_COMPONENT}."
    )
endif()

set(
    _requested_target_variable
    "ROCHostValidation_${ROCHostValidation_TEST_COMPONENT}_TARGET"
)
string(TOLOWER "${ROCHostValidation_TEST_COMPONENT}" _component_suffix)
add_executable(
    "host-validation-${_component_suffix}-component-package-test"
    "${_test_source}"
)
target_link_libraries(
    "host-validation-${_component_suffix}-component-package-test"
    PRIVATE
        "${${_requested_target_variable}}"
)

enable_testing()
add_test(
    NAME "host-validation-${_component_suffix}-component-package-consumer"
    COMMAND "host-validation-${_component_suffix}-component-package-test"
)
