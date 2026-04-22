# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Support standalone builds (e.g. pip install / tox)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.21)
    project(rocisa LANGUAGES CXX)
    set(ROCISA_STANDALONE ON)
    set(HIPBLASLT_BUNDLE_PYTHON_DEPS ON)

    # nanobind requires the Python::Module target (FindPython, not FindPython3)
    find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

    # Find HIP (needed by nanobind target and origami)
    find_package(hip REQUIRED)

    # origami: prefer a sibling monorepo checkout, fall back to find_package
    set(_origami_sibling "${CMAKE_CURRENT_SOURCE_DIR}/../../../../shared/origami")
    if(EXISTS "${_origami_sibling}/CMakeLists.txt")
        add_subdirectory("${_origami_sibling}" "${CMAKE_CURRENT_BINARY_DIR}/origami")
    else()
        find_package(origami REQUIRED)
    endif()
else()
    set(ROCISA_STANDALONE OFF)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(rocisa-cpp INTERFACE)
add_library(rocisa::rocisa-cpp ALIAS rocisa-cpp)
target_sources(rocisa-cpp
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/rocisa/include/enum.hpp>
)
target_include_directories(rocisa-cpp
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

if(HIPBLASLT_BUNDLE_PYTHON_DEPS)
    include(FetchContent)
    FetchContent_Declare(
      nanobind
      GIT_REPOSITORY https://github.com/wjakob/nanobind.git
      GIT_TAG        9b3afa9dbdc23641daf26fadef7743e7127ff92f # v2.6.1
    )
    FetchContent_MakeAvailable(nanobind)

    set(ROCISAINST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/instruction.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/common.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/branch.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/cmp.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/cvt.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/mem.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/mfma.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/extension.cpp")
    set(ROCISAFUNC_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/functions/functions.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/functions/argument.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/functions/f_math.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/functions/f_branch.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/functions/f_cast.cpp")
    set(ROCISAPASS_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/pass.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/graph.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/composite.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/cycle.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/remove.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/insert_delay_alu.cpp")

    nanobind_add_module(_rocisa NOMINSIZE NB_SUPPRESS_WARNINGS
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/main.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/base.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/code.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/container.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/count.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/enum.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/helper.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/label.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/macro.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/register.cpp"
                            ${ROCISAINST_SOURCE}
                            ${ROCISAPASS_SOURCE}
                            ${ROCISAFUNC_SOURCE})
    target_include_directories(_rocisa PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../../shared/origami/include"
    )

    # Optional inclusion of external shared/stinkytofu CMake project
    set(STINKYTOFU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../shared/stinkytofu")
    if(EXISTS "${STINKYTOFU_DIR}/CMakeLists.txt")
        # Add the external directory; build output goes into a local binary dir 'stinkytofu'
        add_subdirectory("${STINKYTOFU_DIR}" stinkytofu)
    else()
        message(FATAL_ERROR "External directory not found: ${STINKYTOFU_DIR} (skipping)")
    endif()

    target_link_libraries(_rocisa PUBLIC roc::origami)
    target_link_libraries(_rocisa PRIVATE hip::host stinkytofu)
    # rocisa conversion sources include stinkytofu public headers that
    # transitively include generated files (hardware/gfxIsa.inc) from
    # stinkytofu's build directory.
    target_include_directories(_rocisa PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/stinkytofu")
    set_target_properties(_rocisa PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/rocisa")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/__init__.py"
        "${CMAKE_CURRENT_BINARY_DIR}/rocisa/__init__.py"
        COPYONLY
    )
endif()
