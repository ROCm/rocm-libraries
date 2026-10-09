# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(HostNumericsComponentPackageTest LANGUAGES CXX)

foreach(required_variable
        HostNumerics_TEST_COMPONENT
        HostNumerics_TEST_EXPECTED_COMPONENTS)
    if(NOT DEFINED ${required_variable})
        message(FATAL_ERROR "${required_variable} is required.")
    endif()
endforeach()

find_package(
    HostNumerics
    CONFIG
    REQUIRED
    COMPONENTS "${HostNumerics_TEST_COMPONENT}"
)

set(_expected_Core_target roc::hostnumerics-core)
set(_expected_Core_dependencies)
set(_expected_Operations_target roc::hostnumerics)
set(_expected_Operations_dependencies Core)
set(_expected_BLAS_target roc::hostnumerics-blas)
set(_expected_BLAS_dependencies Operations)
set(
    _expected_AMDGPULayout_target
    roc::hostnumerics-amd-gpu-layout
)
set(_expected_AMDGPULayout_dependencies)

if(NOT "${HostNumerics_SUPPORTED_COMPONENTS}"
   STREQUAL "Core;Operations;BLAS;AMDGPULayout")
    message(FATAL_ERROR
        "Unexpected supported-component list: "
        "${HostNumerics_SUPPORTED_COMPONENTS}"
    )
endif()

string(
    REPLACE
    ","
    ";"
    _expected_components
    "${HostNumerics_TEST_EXPECTED_COMPONENTS}"
)
foreach(component IN LISTS HostNumerics_SUPPORTED_COMPONENTS)
    set(_target_variable "HostNumerics_${component}_TARGET")
    set(
        _dependencies_variable
        "HostNumerics_${component}_COMPONENT_DEPENDENCIES"
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
        if(NOT HostNumerics_${component}_FOUND)
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
        if(HostNumerics_${component}_FOUND)
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

if(HostNumerics_TEST_COMPONENT STREQUAL "Core")
    set(_test_source "${CMAKE_CURRENT_LIST_DIR}/../core_only/main.cpp")
elseif(HostNumerics_TEST_COMPONENT STREQUAL "Operations")
    set(_test_source "${CMAKE_CURRENT_LIST_DIR}/../validation.cpp")
elseif(HostNumerics_TEST_COMPONENT STREQUAL "BLAS")
    set(_test_source "${CMAKE_CURRENT_LIST_DIR}/../blas.cpp")
elseif(HostNumerics_TEST_COMPONENT STREQUAL "AMDGPULayout")
    set(_test_source "${CMAKE_CURRENT_LIST_DIR}/../amd_gpu_layout.cpp")
else()
    message(FATAL_ERROR
        "No package test source is defined for "
        "${HostNumerics_TEST_COMPONENT}."
    )
endif()

set(
    _requested_target_variable
    "HostNumerics_${HostNumerics_TEST_COMPONENT}_TARGET"
)
string(TOLOWER "${HostNumerics_TEST_COMPONENT}" _component_suffix)
add_executable(
    "hostnumerics-${_component_suffix}-component-package-test"
    "${_test_source}"
)
target_link_libraries(
    "hostnumerics-${_component_suffix}-component-package-test"
    PRIVATE
        "${${_requested_target_variable}}"
)

enable_testing()
add_test(
    NAME "hostnumerics-${_component_suffix}-component-package-consumer"
    COMMAND "hostnumerics-${_component_suffix}-component-package-test"
)
