# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

find_package(hip REQUIRED)

find_package(nanobind CONFIG QUIET)
if(NOT nanobind_FOUND AND ORIGAMI_ENABLE_FETCH)
    include(FetchContent)
    FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG        v2.0.0
    )
    FetchContent_MakeAvailable(nanobind)
    find_package(nanobind CONFIG REQUIRED)
endif()

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

nanobind_add_module(_pyorigami "${CMAKE_CURRENT_SOURCE_DIR}/csrc/bindings.cpp")

# Place the compiled module in the source package directory for development/testing
set_target_properties(_pyorigami PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/origami"
)

target_link_libraries(_pyorigami PRIVATE roc::origami hip::host)

set_target_properties(_pyorigami PROPERTIES
    CXX_STANDARD 17
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

# Set RPATH so the Python extension can find liborigami.so at runtime.
if(NOT WIN32)
    set_target_properties(_pyorigami PROPERTIES
        INSTALL_RPATH "$ORIGIN/../../..:$ORIGIN/../../lib"
    )
endif()

# Install the compiled module to the package directory
install(TARGETS _pyorigami LIBRARY DESTINATION "${PYTHON_INSTALL_DIR}")

# Install Python package files
install(FILES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/__init__.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/selector.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/origami.py"
    DESTINATION "${PYTHON_INSTALL_DIR}"
)

if(ORIGAMI_BUILD_TESTING OR BUILD_TESTING)
    # Run core origami binding tests (no torch required)
    add_test(
        NAME "Python: test_origami_bindings"
        COMMAND ${Python_EXECUTABLE} -m pytest tests/test_origami.py -v
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    set_tests_properties(
        "Python: test_origami_bindings"
        PROPERTIES
            ENVIRONMENT
            "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src:$ENV{PYTHONPATH}"
            DEPENDS _pyorigami
    )

    # Run selector tests (requires torch)
    add_test(
        NAME "Python: test_selector"
        COMMAND ${Python_EXECUTABLE} -m pytest tests/test_selector.py -v
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    set_tests_properties(
        "Python: test_selector"
        PROPERTIES
            ENVIRONMENT
            "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src:$ENV{PYTHONPATH}"
            DEPENDS _pyorigami
            SKIP_RETURN_CODE 5  # Allow skipping tests without emitting an error
    )

    # Run core origami binding tests (no torch required)
    add_test(
        NAME "Python: test_attention_bindings"
        COMMAND ${Python_EXECUTABLE} -m pytest tests/test_attention.py -v
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    set_tests_properties(
        "Python: test_attention_bindings"
        PROPERTIES
            ENVIRONMENT
            "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src:$ENV{PYTHONPATH}"
            DEPENDS _pyorigami
    )

    # Run selector tests (requires torch)
    add_test(
        NAME "Python: test_attention_selector"
        COMMAND ${Python_EXECUTABLE} -m pytest tests/test_attention_selector.py -v
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    set_tests_properties(
        "Python: test_attention_selector"
        PROPERTIES
            ENVIRONMENT
            "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src:$ENV{PYTHONPATH}"
            DEPENDS _pyorigami
            SKIP_RETURN_CODE 5  # Allow skipping tests without emitting an error
    )

    # Run hardware tests
    add_test(
        NAME "Python: test_hardware"
        COMMAND ${Python_EXECUTABLE} -m pytest tests/test_hardware.py -v
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    set_tests_properties(
        "Python: test_hardware"
        PROPERTIES
            ENVIRONMENT
            "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src:$ENV{PYTHONPATH}"
            DEPENDS _pyorigami
    )

    # Run ranking regression tests
    add_test(
        NAME "Python: test_ranking_regression"
        COMMAND ${Python_EXECUTABLE} -m pytest tests/test_ranking_regression.py -v
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    set_tests_properties(
        "Python: test_ranking_regression"
        PROPERTIES
            ENVIRONMENT
            "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src:$ENV{PYTHONPATH}"
            DEPENDS _pyorigami
    )

    # Install Python test files to bin/origami/tests for TheRock CI testing.
    install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tests/"
        DESTINATION "${CMAKE_INSTALL_BINDIR}/origami/tests"
        COMPONENT tests
        FILES_MATCHING
        PATTERN "*.py"
        PATTERN "*.csv"
        PATTERN "*.yaml"
        PATTERN "*.md"
    )
endif()
