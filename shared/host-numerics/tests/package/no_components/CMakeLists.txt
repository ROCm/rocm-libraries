# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(ROCHostNumericsNoComponents LANGUAGES CXX)

find_package(ROCHostNumerics CONFIG REQUIRED)

if(NOT ROCHostNumerics_Core_FOUND OR NOT ROCHostNumerics_Operations_FOUND)
    message(FATAL_ERROR
        "Component-less lookup did not load the default Operations and Core components."
    )
endif()
foreach(unexpected_component BLAS AMDGPULayout)
    if(ROCHostNumerics_${unexpected_component}_FOUND)
        message(FATAL_ERROR
            "Component-less lookup unexpectedly loaded ${unexpected_component}."
        )
    endif()
endforeach()

add_executable(host-numerics-default-package-test ../mx.cpp)
target_link_libraries(
    host-numerics-default-package-test
    PRIVATE
        roc::host-numerics
)

enable_testing()
add_test(
    NAME host-numerics-default-package-consumer
    COMMAND host-numerics-default-package-test
)
