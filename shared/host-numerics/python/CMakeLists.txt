# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

find_package(nanobind CONFIG QUIET)
if(NOT nanobind_FOUND)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        RESULT_VARIABLE _host_numerics_nanobind_result
        OUTPUT_VARIABLE _host_numerics_nanobind_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_host_numerics_nanobind_result EQUAL 0)
        list(APPEND CMAKE_PREFIX_PATH "${_host_numerics_nanobind_dir}")
    endif()
    find_package(nanobind CONFIG REQUIRED)
endif()

nanobind_add_module(
    host-numerics-python
    NB_STATIC
    src/bindings.cpp
    src/comparison_bindings.cpp
    src/generation_bindings.cpp
    src/gemm_bindings.cpp
    src/mx_bindings.cpp
    src/operation_bindings.cpp
)
target_link_libraries(
    host-numerics-python
    PRIVATE
        roc::host-numerics
)
set_target_properties(
    host-numerics-python
    PROPERTIES
        OUTPUT_NAME "_roc_host_numerics"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/roc_host_numerics"
)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/roc_host_numerics")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/roc_host_numerics/__init__.py"
    "${CMAKE_CURRENT_BINARY_DIR}/roc_host_numerics/__init__.py"
    COPYONLY
)

if(HOST_NUMERICS_BUILD_TESTING)
    add_test(
        NAME host-numerics-python-tests
        COMMAND
            "${Python_EXECUTABLE}" -m unittest discover
            -s "${CMAKE_CURRENT_SOURCE_DIR}/tests"
            -p "test_*.py"
    )
    set_tests_properties(
        host-numerics-python-tests
        PROPERTIES
            ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()

set(
    HOST_NUMERICS_PYTHON_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/roc_host_numerics"
)
install(
    TARGETS host-numerics-python
    LIBRARY DESTINATION "${HOST_NUMERICS_PYTHON_INSTALL_DIR}"
    COMPONENT python
)
install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/roc_host_numerics/__init__.py"
    DESTINATION "${HOST_NUMERICS_PYTHON_INSTALL_DIR}"
    COMPONENT python
)
