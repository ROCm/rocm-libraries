# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.26)

project(OrigamiPython LANGUAGES CXX)

include(GNUInstallDirs)
find_package(origami CONFIG REQUIRED)
option(ORIGAMI_BUILD_TESTING "Enable Python tests." OFF)
option(ORIGAMI_PYTHON_ENABLE_WARNINGS "Enable compiler warnings." OFF)

find_package(Python 3.12 COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)

find_package(hip REQUIRED)

find_package(nanobind CONFIG QUIET)
if(NOT nanobind_FOUND)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_VARIABLE nanobind_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )
    find_package(nanobind CONFIG REQUIRED)
endif()

if(SKBUILD)
    set(PYTHON_INSTALL_DIR "origami")
else()
    set(PYTHON_INSTALL_DIR
        "${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/origami"
    )
endif()

nanobind_add_module(_pyorigami STABLE_ABI "${CMAKE_CURRENT_SOURCE_DIR}/csrc/bindings.cpp")

set_target_properties(_pyorigami PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/origami"
)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/origami/__init__.py"
    "${CMAKE_CURRENT_BINARY_DIR}/origami/__init__.py"
    COPYONLY
)
execute_process(
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/build_metadata.py"
        "${CMAKE_CURRENT_BINARY_DIR}/origami/_rocm_init.py"
    COMMAND_ERROR_IS_FATAL ANY
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

set_target_properties(_pyorigami PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../..:$ORIGIN/../../lib:$ORIGIN/../lib"
)

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

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/origami/_rocm_init.py"
    DESTINATION "${PYTHON_INSTALL_DIR}"
)

if(ORIGAMI_BUILD_TESTING OR BUILD_TESTING)
    add_test(
        NAME "Python: pytest"
        COMMAND ${Python_EXECUTABLE} -m pytest -v
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
    set_tests_properties("Python: pytest" PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}:$ENV{PYTHONPATH}"
        SKIP_RETURN_CODE 5
    )
endif()
