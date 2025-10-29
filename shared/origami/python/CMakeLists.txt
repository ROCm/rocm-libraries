# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG        v1.9.2  # or latest stable version
)
FetchContent_MakeAvailable(nanobind)

find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(origami_python 
    "${CMAKE_CURRENT_SOURCE_DIR}/origami_module.cpp"
)

set_target_properties(origami_python PROPERTIES
    OUTPUT_NAME "origami"
)

target_link_libraries(origami_python PRIVATE roc::origami)

