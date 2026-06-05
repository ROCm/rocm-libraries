# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# CMake script to run cmake --install quietly, only showing output on error.
# Usage: cmake -DBUILD_DIR=<dir> -DPREFIX=<prefix> -DCOMPONENT=<component> -P quiet_install.cmake

if(NOT DEFINED BUILD_DIR)
    message(FATAL_ERROR "BUILD_DIR must be defined")
endif()
if(NOT DEFINED PREFIX)
    message(FATAL_ERROR "PREFIX must be defined")
endif()
if(NOT DEFINED COMPONENT)
    message(FATAL_ERROR "COMPONENT must be defined")
endif()

execute_process(
    COMMAND ${CMAKE_COMMAND} --install "${BUILD_DIR}" --prefix "${PREFIX}" --component "${COMPONENT}"
    OUTPUT_VARIABLE _install_output
    ERROR_VARIABLE _install_error
    RESULT_VARIABLE _install_result
)

if(NOT _install_result EQUAL 0)
    message(STATUS "Install output:\n${_install_output}")
    message(STATUS "Install errors:\n${_install_error}")
    message(FATAL_ERROR "cmake --install failed for component ${COMPONENT} with exit code ${_install_result}")
endif()
