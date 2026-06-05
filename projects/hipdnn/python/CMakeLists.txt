# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(DEFINED SKBUILD)
    cmake_minimum_required(VERSION 3.18)
    project(hipdnn_python_bindings LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

if(DEFINED SKBUILD)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "Path to the ROCm installation.")
    endif()
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to the ROCm installation.")
    list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
    find_package(nanobind REQUIRED CONFIG)
    find_package(hipdnn_frontend REQUIRED CONFIG)
    find_package(hipdnn_backend REQUIRED CONFIG)
else()
    hipdnn_add_dependency(tsl-robin-map VERSION ${HIPDNN_TSL_ROBIN_MAP_VERSION})
    hipdnn_add_dependency(nanobind VERSION ${HIPDNN_NANOBIND_VERSION})
endif()

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

target_link_libraries(
    hipdnn_frontend_python
    PRIVATE
        hipdnn_frontend
        hipdnn_backend
)

set_target_properties(
    hipdnn_frontend_python
    PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH TRUE
)

if(DEFINED SKBUILD)
    install(TARGETS hipdnn_frontend_python DESTINATION hipdnn_frontend)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/hipdnn_frontend/__init__.py"
            DESTINATION hipdnn_frontend)
else()
    set(_staging "share/hipdnn/python/hipdnn_frontend")
    install(TARGETS hipdnn_frontend_python DESTINATION "${_staging}")
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/hipdnn_frontend/__init__.py"
            DESTINATION "${_staging}")
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/hipdnn_frontend/test/"
            DESTINATION "share/hipdnn/tests/python"
            FILES_MATCHING PATTERN "*.py")
endif()
