# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# nanobind 3.0.1, used in CI, requires Python 3.10 or newer.
# This version is a lower bound; newer Python versions are accepted.
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

find_package(nanobind CONFIG QUIET)
if(NOT nanobind_FOUND)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        RESULT_VARIABLE _hostnumerics_nanobind_result
        OUTPUT_VARIABLE _hostnumerics_nanobind_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_hostnumerics_nanobind_result EQUAL 0)
        list(APPEND CMAKE_PREFIX_PATH "${_hostnumerics_nanobind_dir}")
    endif()
    find_package(nanobind CONFIG REQUIRED)
endif()

nanobind_add_module(hostnumerics-python NB_STATIC src/bindings.cpp)
target_link_libraries(hostnumerics-python PRIVATE roc::hostnumerics)
set_target_properties(hostnumerics-python
    PROPERTIES
        OUTPUT_NAME "_hostnumerics"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/hostnumerics"
)

file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/hostnumerics/__init__.py"
    INPUT "${CMAKE_CURRENT_SOURCE_DIR}/hostnumerics/__init__.py"
)

if(HOSTNUMERICS_BUILD_TESTING)
    add_test(NAME hostnumerics-python-import
        COMMAND "${Python_EXECUTABLE}" -m unittest discover
                -s "${CMAKE_CURRENT_SOURCE_DIR}/tests" -p "test_*.py"
    )
    set_tests_properties(hostnumerics-python-import
        PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>"
    )
endif()

set(HOSTNUMERICS_PYTHON_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/hostnumerics"
)
install(TARGETS hostnumerics-python
    LIBRARY DESTINATION "${HOSTNUMERICS_PYTHON_INSTALL_DIR}"
)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/hostnumerics/__init__.py"
    DESTINATION "${HOSTNUMERICS_PYTHON_INSTALL_DIR}"
)
