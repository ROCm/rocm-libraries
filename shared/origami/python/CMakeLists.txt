# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(GNUInstallDirs)

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

set(PYTHON_INSTALL_DIR "origami")

nanobind_add_module(_pyorigami "${CMAKE_CURRENT_SOURCE_DIR}/csrc/bindings.cpp")

set_target_properties(_pyorigami PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/origami"
)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/__init__.py"
    "${CMAKE_CURRENT_BINARY_DIR}/origami/__init__.py"
    COPYONLY
)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/selector.py"
    "${CMAKE_CURRENT_BINARY_DIR}/origami/selector.py"
    COPYONLY
)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/origami.py"
    "${CMAKE_CURRENT_BINARY_DIR}/origami/origami.py"
    COPYONLY
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

if(ORIGAMI_BUILD_FROM_SOURCE AND SKBUILD_STATE STREQUAL "editable")
    set_target_properties(_pyorigami PROPERTIES
        INSTALL_RPATH "$<TARGET_FILE_DIR:roc::origami>"
        INSTALL_RPATH_USE_LINK_PATH ON
    )
else()
    set_target_properties(_pyorigami PROPERTIES
        INSTALL_RPATH "$ORIGIN/../../..:$ORIGIN/../../lib:$ORIGIN/../lib"
    )
endif()

install(TARGETS _pyorigami
    LIBRARY DESTINATION "${PYTHON_INSTALL_DIR}"
    RUNTIME DESTINATION "${PYTHON_INSTALL_DIR}"
)

if(NOT SKBUILD)
    install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/__init__.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/selector.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/origami.py"
        DESTINATION "${PYTHON_INSTALL_DIR}"
    )
endif()

if(ORIGAMI_BUILD_TESTING OR BUILD_TESTING)
    add_test(
        NAME "Python: pytest"
        COMMAND ${Python_EXECUTABLE} -m pytest -v
        WORKING_DIRECTORY ${Origami_SOURCE_DIR}
    )
    set_tests_properties("Python: pytest" PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}:$ENV{PYTHONPATH}"
        SKIP_RETURN_CODE 5
    )
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
