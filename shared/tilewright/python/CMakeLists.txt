# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Python bindings for the standalone tilewright kernel-recommender library.
# tilewright has NO HIP / GEMM-framework dependency, so the extension links ONLY
# against roc::tilewright (no hip::host, no __HIP_PLATFORM_AMD__).

cmake_minimum_required(VERSION 3.25.2)

project(tilewright-python VERSION 0.1.0 LANGUAGES CXX)

# Find Python
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

option(TILEWRIGHT_BUILD_FROM_SOURCE "Build tilewright from source instead of finding installed version" ON)
option(TILEWRIGHT_PYTHON_ENABLE_WARNINGS "Enable compiler warnings for Python extension build" OFF)

# Resolve the tilewright library target:
#   1. reuse roc::tilewright / tilewright if already present in this CMake tree
#   2. else add_subdirectory("..") with python bindings OFF
#   3. else find an installed package
set(TILEWRIGHT_TARGET "")
if(TARGET roc::tilewright)
    set(TILEWRIGHT_TARGET roc::tilewright)
elseif(TARGET tilewright)
    set(TILEWRIGHT_TARGET tilewright)
endif()

if(TILEWRIGHT_TARGET)
    # tilewright already exists in this CMake tree; reuse it to avoid duplicate targets.
elseif(TILEWRIGHT_BUILD_FROM_SOURCE)
    set(TILEWRIGHT_ENABLE_PYTHON OFF CACHE BOOL "Build tilewright python bindings" FORCE)
    add_subdirectory("${PROJECT_SOURCE_DIR}/.." tilewright_build EXCLUDE_FROM_ALL)
    set(TILEWRIGHT_TARGET roc::tilewright)
else()
    find_package(tilewright REQUIRED)
    set(TILEWRIGHT_TARGET roc::tilewright)
endif()

# nanobind: fetch v2.0.0 when TILEWRIGHT_ENABLE_FETCH is ON, else require an
# installed copy.
if(NOT DEFINED TILEWRIGHT_ENABLE_FETCH)
    set(TILEWRIGHT_ENABLE_FETCH ON)
endif()

if(TILEWRIGHT_ENABLE_FETCH)
    include(FetchContent)
    FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG        v2.0.0
    )
    FetchContent_MakeAvailable(nanobind)
endif()

find_package(nanobind CONFIG REQUIRED)

if(SKBUILD)
    # When building a wheel via pip/scikit-build-core
    set(PYTHON_INSTALL_DIR "${SKBUILD_PLATLIB_DIR}/tilewright")
else()
    # When building standalone with CMake, install under the CMake prefix.
    include(GNUInstallDirs)
    set(PYTHON_INSTALL_DIR
        "${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/tilewright"
    )
endif()

nanobind_add_module(_pytilewright "${CMAKE_CURRENT_SOURCE_DIR}/src/tilewright/bindings.cpp")

# The compiled module imports as `tilewright.tilewright` (see src/tilewright/__init__.py).
set_target_properties(_pytilewright PROPERTIES OUTPUT_NAME "tilewright")

# Place the compiled module in the source package directory for development/testing.
set_target_properties(_pytilewright PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/tilewright"
)

# Link ONLY against the tilewright static library (no HIP).
target_link_libraries(_pytilewright PRIVATE ${TILEWRIGHT_TARGET})

set_target_properties(_pytilewright PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

if(TILEWRIGHT_PYTHON_ENABLE_WARNINGS)
    target_compile_options(_pytilewright PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
    )
endif()

# Set RPATH so the Python extension can find libtilewright at runtime.
if(NOT WIN32)
    set_target_properties(_pytilewright PROPERTIES
        INSTALL_RPATH "$ORIGIN/../../..:$ORIGIN/../../lib"
    )
endif()

# Install the compiled module to the package directory.
install(TARGETS _pytilewright LIBRARY DESTINATION "${PYTHON_INSTALL_DIR}")

# Install Python package files.
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tilewright/__init__.py"
    DESTINATION "${PYTHON_INSTALL_DIR}"
)

if(TILEWRIGHT_BUILD_TESTING OR BUILD_TESTING)
    # Enable CTest so add_test() registers tests for ctest.
    enable_testing()

    add_test(
        NAME "Python: test_tilewright_bindings"
        COMMAND ${Python_EXECUTABLE} -m pytest tests/test_tilewright.py -v
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    set_tests_properties(
        "Python: test_tilewright_bindings"
        PROPERTIES
            ENVIRONMENT
            "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src:$ENV{PYTHONPATH}"
            DEPENDS _pytilewright
    )

    # Install Python test files for downstream CI testing.
    install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tests/"
        DESTINATION "${CMAKE_INSTALL_BINDIR}/tilewright/tests"
        COMPONENT tests
        FILES_MATCHING
        PATTERN "*.py"
    )
endif()
