# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

find_package(nanobind CONFIG QUIET)
if(NOT nanobind_FOUND)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        RESULT_VARIABLE _host_validation_nanobind_result
        OUTPUT_VARIABLE _host_validation_nanobind_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_host_validation_nanobind_result EQUAL 0)
        list(APPEND CMAKE_PREFIX_PATH "${_host_validation_nanobind_dir}")
    endif()
    find_package(nanobind CONFIG REQUIRED)
endif()

nanobind_add_module(
    host-validation-python
    NB_STATIC
    src/bindings.cpp
    src/comparison_bindings.cpp
    src/generation_bindings.cpp
    src/gemm_bindings.cpp
    src/mx_bindings.cpp
    src/operation_bindings.cpp
)
target_link_libraries(
    host-validation-python
    PRIVATE
        roc::host-validation
        roc::host-validation-blocked
        roc::host-validation-mx
)
set_target_properties(
    host-validation-python
    PROPERTIES
        OUTPUT_NAME "_roc_host_validation"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/roc_host_validation"
)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/roc_host_validation")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/roc_host_validation/__init__.py"
    "${CMAKE_CURRENT_BINARY_DIR}/roc_host_validation/__init__.py"
    COPYONLY
)

if(HOST_VALIDATION_BUILD_TESTING)
    add_test(
        NAME host-validation-python-tests
        COMMAND
            "${Python_EXECUTABLE}" -m unittest discover
            -s "${CMAKE_CURRENT_SOURCE_DIR}/tests"
            -p "test_*.py"
    )
    set_tests_properties(
        host-validation-python-tests
        PROPERTIES
            ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()

set(
    HOST_VALIDATION_PYTHON_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/roc_host_validation"
)
install(
    TARGETS host-validation-python
    LIBRARY DESTINATION "${HOST_VALIDATION_PYTHON_INSTALL_DIR}"
    COMPONENT python
)
install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/roc_host_validation/__init__.py"
    DESTINATION "${HOST_VALIDATION_PYTHON_INSTALL_DIR}"
    COMPONENT python
)
