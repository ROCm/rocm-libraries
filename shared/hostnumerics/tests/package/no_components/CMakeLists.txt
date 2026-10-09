# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(HostNumericsNoComponents LANGUAGES CXX)

find_package(HostNumerics CONFIG REQUIRED)

if(NOT HostNumerics_Core_FOUND OR NOT HostNumerics_Operations_FOUND)
    message(FATAL_ERROR
        "Component-less lookup did not load the default Operations and Core components."
    )
endif()
foreach(unexpected_component BLAS AMDGPULayout)
    if(HostNumerics_${unexpected_component}_FOUND)
        message(FATAL_ERROR
            "Component-less lookup unexpectedly loaded ${unexpected_component}."
        )
    endif()
endforeach()

add_executable(hostnumerics-default-package-test ../mx.cpp)
target_link_libraries(
    hostnumerics-default-package-test
    PRIVATE
        roc::hostnumerics
)

enable_testing()
add_test(
    NAME hostnumerics-default-package-consumer
    COMMAND hostnumerics-default-package-test
)
