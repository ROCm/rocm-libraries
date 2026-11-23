# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)

project(tilewright_python LANGUAGES CXX)

# Built as its own project (pip / scikit-build-core): compile the engine from
# the parent directory without its tests, bindings or install rules.
if(NOT TARGET roc::tilewright)
    set(TILEWRIGHT_BUILD_TESTING OFF)
    set(TILEWRIGHT_ENABLE_PYTHON OFF)
    set(TILEWRIGHT_ENABLE_INSTALL OFF)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/.." tilewright EXCLUDE_FROM_ALL)
endif()

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)

# Prefer the nanobind installed in the build Python (pip build isolation
# provides it); fetch it only for CMake builds that allow fetching.
if(NOT nanobind_ROOT)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_VARIABLE _tw_nanobind_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _tw_nanobind_status
        ERROR_QUIET
    )
    if(_tw_nanobind_status EQUAL 0)
        set(nanobind_ROOT "${_tw_nanobind_dir}")
    endif()
endif()
find_package(nanobind CONFIG QUIET)
if(NOT nanobind_FOUND)
    if(TILEWRIGHT_ENABLE_FETCH AND NOT SKBUILD)
        include(FetchContent)
        FetchContent_Declare(
            nanobind
            GIT_REPOSITORY https://github.com/wjakob/nanobind.git
            GIT_TAG v3.1.0
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(nanobind)
    else()
        message(FATAL_ERROR "nanobind not found for ${Python_EXECUTABLE} (pip install nanobind)")
    endif()
endif()

nanobind_add_module(_tilewright "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp")
target_link_libraries(_tilewright PRIVATE roc::tilewright)

# Importable build-tree package: <build>/package/tilewright/{__init__.py,_tilewright.*}
set(_tw_package_dir "${CMAKE_CURRENT_BINARY_DIR}/package/tilewright")
set_target_properties(_tilewright PROPERTIES LIBRARY_OUTPUT_DIRECTORY "$<1:${_tw_package_dir}>")
add_custom_command(
    TARGET _tilewright
    POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/src/tilewright/__init__.py" "${_tw_package_dir}/__init__.py"
    COMMENT "Assembling the build-tree tilewright Python package"
)

if(SKBUILD)
    install(TARGETS _tilewright LIBRARY DESTINATION tilewright)
elseif(TILEWRIGHT_ENABLE_INSTALL)
    set(_tw_site_dir
        "${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/tilewright"
    )
    install(TARGETS _tilewright LIBRARY DESTINATION "${_tw_site_dir}" COMPONENT python)
    install(
        FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/tilewright/__init__.py"
        DESTINATION "${_tw_site_dir}"
        COMPONENT python
    )
endif()

if(TILEWRIGHT_BUILD_TESTING)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import pytest"
        RESULT_VARIABLE _tw_pytest_status
        OUTPUT_QUIET ERROR_QUIET
    )
    if(_tw_pytest_status EQUAL 0)
        add_test(
            NAME tilewright-python
            COMMAND "${Python_EXECUTABLE}" -m pytest -q -p no:cacheprovider
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests"
        )
        set_tests_properties(
            tilewright-python
            PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/package;PYTHONDONTWRITEBYTECODE=1"
        )
    else()
        message(STATUS "pytest is not available for ${Python_EXECUTABLE}; Python tests not registered")
    endif()
endif()
