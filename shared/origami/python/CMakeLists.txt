# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)

project(origami-python VERSION 0.1.0 LANGUAGES CXX)

# Find Python
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

option(ORIGAMI_BUILD_FROM_SOURCE "Build origami from source instead of finding installed version" ON)
option(ORIGAMI_PYTHON_ENABLE_WARNINGS "Enable compiler warnings for Python extension build" OFF)

set(ORIGAMI_TARGET "")
if(TARGET roc::origami)
    set(ORIGAMI_TARGET roc::origami)
elseif(TARGET origami)
    set(ORIGAMI_TARGET origami)
endif()

if(ORIGAMI_TARGET)
    # Origami already exists in this CMake tree; reuse it to avoid duplicate targets.
elseif(ORIGAMI_BUILD_FROM_SOURCE)
    set(ORIGAMI_ENABLE_PYTHON OFF CACHE BOOL "Build origami python bindings" FORCE)
    add_subdirectory("${PROJECT_SOURCE_DIR}/.." origami_build EXCLUDE_FROM_ALL)
    set(ORIGAMI_TARGET roc::origami)
else()
    find_package(origami REQUIRED)
    set(ORIGAMI_TARGET roc::origami)
endif()

find_package(hip REQUIRED)

include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG        v2.0.0
)
FetchContent_MakeAvailable(nanobind)
find_package(nanobind CONFIG REQUIRED)

if(SKBUILD)
  # When building a wheel via pip/scikit-build-core
  set(PYTHON_INSTALL_DIR "${SKBUILD_PLATLIB_DIR}/origami")
else()
  # When building standalone with CMake, install under the CMake prefix.
  include(GNUInstallDirs)
  set(PYTHON_INSTALL_DIR
      "${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/origami"
  )
endif()

nanobind_add_module(_pyorigami "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/bindings.cpp")

set_target_properties(_pyorigami PROPERTIES OUTPUT_NAME "origami")

# Link against origami library and HIP
target_link_libraries(_pyorigami PRIVATE ${ORIGAMI_TARGET} hip::host)

set_target_properties(_pyorigami PROPERTIES
    CXX_STANDARD 17                    # We can make this optional too if we want to support newer standards
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
target_compile_definitions(_pyorigami PRIVATE
    __HIP_PLATFORM_AMD__
)

if(ORIGAMI_PYTHON_ENABLE_WARNINGS)
    target_compile_options(_pyorigami PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
    )
endif()

# Install the compiled module to the package directory
install(TARGETS _pyorigami LIBRARY DESTINATION "${PYTHON_INSTALL_DIR}")

# Install Python package files
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/__init__.py"
    DESTINATION "${PYTHON_INSTALL_DIR}"
)

if(ORIGAMI_BUILD_TESTING OR BUILD_TESTING)
    # Enable CTest so add_test() registers tests for ctest
    enable_testing()

    # Run pytest tests
    add_test(
        NAME "Python: test_bindings"
        COMMAND ${Python_EXECUTABLE} -m pytest tests/ -v
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    set_tests_properties(
        "Python: test_bindings"
        PROPERTIES
            ENVIRONMENT
            "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}:${CMAKE_CURRENT_SOURCE_DIR}/src:$ENV{PYTHONPATH}"
            DEPENDS _pyorigami
    )

    # Install Python module to bin/origami for TheRock CI testing
    install(TARGETS _pyorigami
        LIBRARY DESTINATION "${CMAKE_INSTALL_BINDIR}/origami"
        COMPONENT tests
    )

    # Install Python test files to bin/origami/tests for TheRock CI testing  
    install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tests/"
        DESTINATION "${CMAKE_INSTALL_BINDIR}/origami/tests"
        COMPONENT tests
        FILES_MATCHING 
        PATTERN "*.py"
    )
endif()
