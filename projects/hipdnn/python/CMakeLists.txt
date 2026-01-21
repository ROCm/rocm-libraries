# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.18)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(hipdnn_python_bindings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and nanobind
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# FetchContent to get nanobind if not found
include(FetchContent)
fetchcontent_declare(nanobind GIT_REPOSITORY https://github.com/wjakob/nanobind GIT_TAG v1.8.0)
fetchcontent_makeavailable(nanobind)

# Find hipDNN frontend headers
set(HIPDNN_FRONTEND_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../frontend/include"
    CACHE PATH "Path to hipDNN frontend includes"
)

# Find HIP (if needed)
find_package(hip REQUIRED)
find_package(hipdnn_frontend CONFIG REQUIRED)
find_package(hipdnn_backend REQUIRED)

# Create the Python module
nanobind_add_module(
    hipdnn_frontend_python
    src/module.cpp
    src/graph_bindings.cpp
    src/tensor_bindings.cpp
    src/attributes_bindings.cpp
    src/types_bindings.cpp
    src/handle_bindings.cpp
    src/memory_bindings.cpp
)

# Include directories
target_include_directories(
    hipdnn_frontend_python PRIVATE ${HIPDNN_FRONTEND_INCLUDE_DIR} ${HIP_INCLUDE_DIRS}
)

set_target_properties(hipdnn_frontend_python PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

# Link against hipDNN frontend library if it exists as a separate library Adjust this based on your
# actual library structure
target_link_libraries(hipdnn_frontend_python PRIVATE hipdnn_frontend hipdnn_backend hip::host)

# Install the module - scikit-build-core handles the destination
install(TARGETS hipdnn_frontend_python LIBRARY DESTINATION .)
