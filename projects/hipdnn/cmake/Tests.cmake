# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(HIP_DNN_SKIP_TESTS)
    return()
endif()

hipdnn_add_dependency(GTest VERSION ${HIPDNN_GTEST_VERSION})
include(GoogleTest)

find_package(Python3 COMPONENTS Interpreter)

# Set executable prefix based on platform
if(WIN32)
    set(EXEC_PREFIX "")
else()
    set(EXEC_PREFIX "./")
endif()

# Function to make test executables relocatable by setting proper RPATH
# This allows test binaries to find libraries when the build directory is moved
# Windows does not use RPATH, so this function will have no effect there
function(MakeExecutableRelocatable target)
    set(RPATH_LIST "$ORIGIN/../lib")
    
    # Add any additional RPATH entries passed as extra arguments
    foreach(extra_rpath ${ARGN})
        list(APPEND RPATH_LIST ${extra_rpath})
    endforeach()
    
    set_target_properties(${target} PROPERTIES
        BUILD_WITH_INSTALL_RPATH OFF
        INSTALL_RPATH_USE_LINK_PATH FALSE
        BUILD_RPATH_USE_ORIGIN TRUE
        BUILD_RPATH "${RPATH_LIST}"
    )
endfunction()

# TODO: Consider adding test project run instructions to docs when we finalize it
set(INSTALL_TEST_PROJECT "${CMAKE_CURRENT_BINARY_DIR}/test_project/CMakeLists.txt")
file(WRITE "${INSTALL_TEST_PROJECT}"
[=[
# CMake test project to aid with running the tests easilly via ctest

cmake_minimum_required(VERSION 3.25.2)

project(
    hipDNN_tests
    LANGUAGES CXX)

enable_testing()

]=]
)

set(CHECK_COMMAND_GLOBAL "" CACHE INTERNAL "Accumulated check commands" FORCE)
set(CHECK_DEPENDS_GLOBAL "" CACHE INTERNAL "Accumulated check depends" FORCE)
set(CHECK_EXECUTABLE_PATHS_GLOBAL "" CACHE INTERNAL "Accumulated check executable paths" FORCE)

# Global collections for unit tests
set(UNIT_CHECK_COMMAND_GLOBAL "" CACHE INTERNAL "Accumulated unit check commands" FORCE)
set(UNIT_CHECK_DEPENDS_GLOBAL "" CACHE INTERNAL "Accumulated unit check depends" FORCE)

# Global collections for integration tests
set(INTEGRATION_CHECK_COMMAND_GLOBAL "" CACHE INTERNAL "Accumulated integration check commands" FORCE)
set(INTEGRATION_CHECK_DEPENDS_GLOBAL "" CACHE INTERNAL "Accumulated integration check depends" FORCE)

function(add_test_to_test_project test_target)
    get_target_property(EXE_PATH ${test_target} RUNTIME_OUTPUT_DIRECTORY)
    if(EXE_PATH STREQUAL "EXE_PATH-NOTFOUND")
        set(EXE_PATH ".")
    endif()
    get_filename_component(EXE_PATH "${EXE_PATH}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    get_target_property(EXE_NAME ${test_target} RUNTIME_OUTPUT_NAME)
    if(EXE_NAME STREQUAL "EXE_NAME-NOTFOUND")
        get_target_property(EXE_NAME ${test_target} OUTPUT_NAME)
        if(EXE_NAME STREQUAL "EXE_NAME-NOTFOUND")
            set(EXE_NAME "${test_target}")
        endif()
    endif()
    file(RELATIVE_PATH rel_path "${CMAKE_CURRENT_BINARY_DIR}" "${EXE_PATH}/${EXE_NAME}")


    file(APPEND "${INSTALL_TEST_PROJECT}" "add_test(${test_target} \"../bin/${EXE_NAME}\")\n")
endfunction()

function(create_test_name_validation_target)
    if(Python3_FOUND)
        # Write list of test executables with their paths to a file
        set(TEST_EXECUTABLES_FILE ${CMAKE_BINARY_DIR}/test_executables.txt)
        list(REMOVE_DUPLICATES CHECK_EXECUTABLE_PATHS_GLOBAL)
        file(WRITE ${TEST_EXECUTABLES_FILE} "")
        foreach(test_executable ${CHECK_EXECUTABLE_PATHS_GLOBAL})
            file(APPEND ${TEST_EXECUTABLES_FILE} "${test_executable}\n")
        endforeach()
        
        add_custom_command(
            OUTPUT ${CMAKE_BINARY_DIR}/test_names_validated
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/scripts/test_name_validator.py 
                    --test-executables ${TEST_EXECUTABLES_FILE}
                    --build-dir ${CMAKE_BINARY_DIR}
                    --strict
            COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/test_names_validated
            DEPENDS 
                ${CMAKE_SOURCE_DIR}/cmake/scripts/test_name_validator.py
                ${CHECK_DEPENDS_GLOBAL}
            COMMENT "Validating test names with --gtest_list_tests test collection"
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            VERBATIM
        )
        
        add_custom_target(validate_test_names 
            DEPENDS ${CMAKE_BINARY_DIR}/test_names_validated)
    else()
        message(WARNING "Python3 not found. Test name validation will be skipped.")
        add_custom_target(validate_test_names
            COMMAND ${CMAKE_COMMAND} -E echo "Test name validation skipped - Python3 not found"
        )
    endif()
endfunction()

# Generic internal function to append tests to check targets
function(_append_test_to_check_target_internal TARGET WORKING_DIR TEST_TYPE STATUS_MESSAGE)
    if(STATUS_MESSAGE)
        message(STATUS "${STATUS_MESSAGE}: ${TARGET} in working directory: ${WORKING_DIR}")
    endif()
    
    if("${TEST_TYPE}" STREQUAL "UNIT")
        set(COMMAND_VAR "UNIT_CHECK_COMMAND_GLOBAL")
        set(DEPENDS_VAR "UNIT_CHECK_DEPENDS_GLOBAL")
        set(CACHE_DESC "Accumulated unit check targets")
    elseif("${TEST_TYPE}" STREQUAL "INTEGRATION")
        set(COMMAND_VAR "INTEGRATION_CHECK_COMMAND_GLOBAL")
        set(DEPENDS_VAR "INTEGRATION_CHECK_DEPENDS_GLOBAL")
        set(CACHE_DESC "Accumulated integration check targets")
    else()
        set(COMMAND_VAR "CHECK_COMMAND_GLOBAL")
        set(DEPENDS_VAR "CHECK_DEPENDS_GLOBAL")
        set(CACHE_DESC "Accumulated check targets")
    endif()

    # Build environment list properly
    set(ENVIRONMENT_LIST "")
    if(DEFINED TEST_ENVIRONMENT)
        set(ENVIRONMENT_LIST ${TEST_ENVIRONMENT})
    endif()
    
    if(CODE_COVERAGE)
        # For code coverage builds, we want each profraw file to have a unique name.  The %m
        # in the LLVM_PROFILE_FILE environment variable will auto generate a unique id.
        list(APPEND ENVIRONMENT_LIST "LLVM_PROFILE_FILE=./bin/%m.profraw")
    endif()
    
    set(NEW_COMMAND "")
    if("${${COMMAND_VAR}}" STREQUAL "")
        set(NEW_COMMAND ${CMAKE_COMMAND} -E env ${ENVIRONMENT_LIST} ${CMAKE_BINARY_DIR}/bin/${TARGET})
    else()
        set(NEW_COMMAND && ${CMAKE_COMMAND} -E env ${ENVIRONMENT_LIST} ${CMAKE_BINARY_DIR}/bin/${TARGET})
    endif()
    
    set(${COMMAND_VAR} ${${COMMAND_VAR}} ${NEW_COMMAND} CACHE INTERNAL "${CACHE_DESC}" FORCE)
    set(${DEPENDS_VAR} ${${DEPENDS_VAR}} ${TARGET} CACHE INTERNAL "Accumulated ${TEST_TYPE} check depends" FORCE)
    
    # Track the binary paths for test name validation
    set(EXECUTABLE_PATH "bin/${TARGET}")
    set(CHECK_EXECUTABLE_PATHS_GLOBAL ${CHECK_EXECUTABLE_PATHS_GLOBAL} ${EXECUTABLE_PATH} CACHE INTERNAL "Accumulated check executable paths" FORCE)
endfunction()

# Generic internal function to finalize check targets
function(_finalize_check_target_internal TARGET_NAME COMMAND_VAR DEPENDS_VAR)
    add_custom_target(
        ${TARGET_NAME}
        COMMAND ${${COMMAND_VAR}}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS ${${DEPENDS_VAR}}
        VERBATIM)
    message(STATUS "Created ${TARGET_NAME} target")
endfunction()

function(finalize_custom_check_target)
    _finalize_check_target_internal("check" "CHECK_COMMAND_GLOBAL" "CHECK_DEPENDS_GLOBAL")
endfunction()

function(finalize_unit_check_target)
    _finalize_check_target_internal("unit-check" "UNIT_CHECK_COMMAND_GLOBAL" "UNIT_CHECK_DEPENDS_GLOBAL")
endfunction()

function(finalize_integration_check_target)
    _finalize_check_target_internal("integration-check" "INTEGRATION_CHECK_COMMAND_GLOBAL" "INTEGRATION_CHECK_DEPENDS_GLOBAL")
endfunction()

enable_testing() # Cmake wont discover or run tests without this line

# Add a check_ctest target which will run all tests discovered by gtest_discover_tests via ctest. 
add_custom_target(check_ctest 
    COMMAND ${CMAKE_COMMAND} -E env ${TEST_ENVIRONMENT} ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})

function(_add_gtest_target_internal APPEND_FUNCTION_SUFFIX TARGET WORKING_DIR)
    if("${APPEND_FUNCTION_SUFFIX}" STREQUAL "test")
        _append_test_to_check_target_internal(${TARGET} ${WORKING_DIR} "" "Appending check target")
    elseif("${APPEND_FUNCTION_SUFFIX}" STREQUAL "unit_test")
        _append_test_to_check_target_internal(${TARGET} ${WORKING_DIR} "UNIT" "Appending unit check target")
        _append_test_to_check_target_internal(${TARGET} ${WORKING_DIR} "" "")
    elseif("${APPEND_FUNCTION_SUFFIX}" STREQUAL "integration_test")
        _append_test_to_check_target_internal(${TARGET} ${WORKING_DIR} "INTEGRATION" "Appending integration check target")
        _append_test_to_check_target_internal(${TARGET} ${WORKING_DIR} "" "")
    else()
        message(FATAL_ERROR "Unknown test type suffix: ${APPEND_FUNCTION_SUFFIX}")
    endif()
    
    set_target_properties(${TARGET} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    
    # Make test executables relocatable so they can find libraries when build directory is moved
    MakeExecutableRelocatable(${TARGET})

    add_dependencies(check_ctest ${TARGET})
    add_test(
        NAME ${TARGET}
        COMMAND ${TARGET} 
        WORKING_DIRECTORY ${WORKING_DIR}
    )

    add_test_to_test_project(${TARGET})
endfunction()

# Adds a generic test target
function(add_target_to_check_targets TARGET WORKING_DIR)
    _add_gtest_target_internal(test ${TARGET} ${WORKING_DIR})
endfunction()

# Adds a unit test target
function(add_unit_test_target TARGET WORKING_DIR)
    _add_gtest_target_internal(unit_test ${TARGET} ${WORKING_DIR})
endfunction()

# Adds an integration test target
function(add_integration_test_target TARGET WORKING_DIR)
    _add_gtest_target_internal(integration_test ${TARGET} ${WORKING_DIR})
endfunction()
