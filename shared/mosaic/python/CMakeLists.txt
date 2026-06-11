# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Python bindings for the standalone mosaic kernel-recommender library.
# mosaic has NO HIP / GEMM-framework dependency, so the extension links ONLY
# against roc::mosaic (no hip::host, no __HIP_PLATFORM_AMD__).

cmake_minimum_required(VERSION 3.25.2)

project(mosaic-python VERSION 0.1.0 LANGUAGES CXX)

# Find Python
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

option(MOSAIC_BUILD_FROM_SOURCE "Build mosaic from source instead of finding installed version" ON)
option(MOSAIC_PYTHON_ENABLE_WARNINGS "Enable compiler warnings for Python extension build" OFF)

# Resolve the mosaic library target:
#   1. reuse roc::mosaic / mosaic if already present in this CMake tree
#   2. else add_subdirectory("..") with python bindings OFF
#   3. else find an installed package
set(MOSAIC_TARGET "")
if(TARGET roc::mosaic)
    set(MOSAIC_TARGET roc::mosaic)
elseif(TARGET mosaic)
    set(MOSAIC_TARGET mosaic)
endif()

if(MOSAIC_TARGET)
    # mosaic already exists in this CMake tree; reuse it to avoid duplicate targets.
elseif(MOSAIC_BUILD_FROM_SOURCE)
    set(MOSAIC_ENABLE_PYTHON OFF CACHE BOOL "Build mosaic python bindings" FORCE)
    add_subdirectory("${PROJECT_SOURCE_DIR}/.." mosaic_build EXCLUDE_FROM_ALL)
    set(MOSAIC_TARGET roc::mosaic)
else()
    find_package(mosaic REQUIRED)
    set(MOSAIC_TARGET roc::mosaic)
endif()

# nanobind: fetch v2.0.0 when MOSAIC_ENABLE_FETCH is ON, else require an
# installed copy.
if(NOT DEFINED MOSAIC_ENABLE_FETCH)
    set(MOSAIC_ENABLE_FETCH ON)
endif()

if(MOSAIC_ENABLE_FETCH)
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
    set(PYTHON_INSTALL_DIR "${SKBUILD_PLATLIB_DIR}/mosaic")
else()
    # When building standalone with CMake, install under the CMake prefix.
    include(GNUInstallDirs)
    set(PYTHON_INSTALL_DIR
        "${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/mosaic"
    )
endif()

nanobind_add_module(_pymosaic "${CMAKE_CURRENT_SOURCE_DIR}/src/mosaic/bindings.cpp")

# The compiled module imports as `mosaic.mosaic` (see src/mosaic/__init__.py).
set_target_properties(_pymosaic PROPERTIES OUTPUT_NAME "mosaic")

# Place the compiled module in the source package directory for development/testing.
set_target_properties(_pymosaic PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/mosaic"
)

# Link ONLY against the mosaic static library (no HIP).
target_link_libraries(_pymosaic PRIVATE ${MOSAIC_TARGET})

set_target_properties(_pymosaic PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

if(MOSAIC_PYTHON_ENABLE_WARNINGS)
    target_compile_options(_pymosaic PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
    )
endif()

# Set RPATH so the Python extension can find libmosaic at runtime.
if(NOT WIN32)
    set_target_properties(_pymosaic PROPERTIES
        INSTALL_RPATH "$ORIGIN/../../..:$ORIGIN/../../lib"
    )
endif()

# Install the compiled module to the package directory.
install(TARGETS _pymosaic LIBRARY DESTINATION "${PYTHON_INSTALL_DIR}")

# Install Python package files.
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/mosaic/__init__.py"
    DESTINATION "${PYTHON_INSTALL_DIR}"
)

if(MOSAIC_BUILD_TESTING OR BUILD_TESTING)
    # Enable CTest so add_test() registers tests for ctest.
    enable_testing()

    add_test(
        NAME "Python: test_mosaic_bindings"
        COMMAND ${Python_EXECUTABLE} -m pytest tests/test_mosaic.py -v
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    set_tests_properties(
        "Python: test_mosaic_bindings"
        PROPERTIES
            ENVIRONMENT
            "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src:$ENV{PYTHONPATH}"
            DEPENDS _pymosaic
    )

    # Install Python test files for downstream CI testing.
    install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tests/"
        DESTINATION "${CMAKE_INSTALL_BINDIR}/mosaic/tests"
        COMPONENT tests
        FILES_MATCHING
        PATTERN "*.py"
    )
endif()
