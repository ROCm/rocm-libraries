# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.18)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(hipdnn_python_bindings LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

hipdnn_add_dependency(tsl-robin-map VERSION ${HIPDNN_TSL_ROBIN_MAP_VERSION})
hipdnn_add_dependency(nanobind VERSION ${HIPDNN_NANOBIND_VERSION})

set(HIPDNN_FRONTEND_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../frontend/include"
    CACHE PATH "Path to hipDNN frontend includes"
)

find_package(hip REQUIRED)

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

target_include_directories(
    hipdnn_frontend_python
    PRIVATE
        ${HIPDNN_FRONTEND_INCLUDE_DIR}
        ${HIP_INCLUDE_DIRS}
)

target_link_libraries(
    hipdnn_frontend_python
    PRIVATE
        hipdnn_frontend
        hipdnn_backend
        hip::host
)

set_target_properties(
    hipdnn_frontend_python
    PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Stage the extension where the wheel build hook can find it.
install(TARGETS hipdnn_frontend_python DESTINATION lib/hipdnn/python)

install(CODE "
    set(_wheel_dir \"\${CMAKE_INSTALL_PREFIX}/share/hipdnn/wheels\")
    file(MAKE_DIRECTORY \"\${_wheel_dir}\")

    message(STATUS \"Building hipdnn-frontend wheel\")
    execute_process(
        COMMAND \"${Python_EXECUTABLE}\"
            \"${CMAKE_CURRENT_SOURCE_DIR}/build_python_wheel.py\"
            --source-dir \"${CMAKE_CURRENT_SOURCE_DIR}\"
            --ext-dir \"\${CMAKE_INSTALL_PREFIX}/lib/hipdnn/python\"
            --wheel-dir \"\${_wheel_dir}\"
        RESULT_VARIABLE _result
    )
    if(_result)
        message(FATAL_ERROR \"Failed to build hipdnn-frontend wheel\")
    endif()
")
