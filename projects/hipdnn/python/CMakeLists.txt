# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.18)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(hipdnn_python_bindings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and nanobind
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

if(NOT COMMAND hipdnn_add_dependency)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Dependencies.cmake")
endif()

hipdnn_add_dependency(tsl-robin-map VERSION ${HIPDNN_TSL_ROBIN_MAP_VERSION})
hipdnn_add_dependency(nanobind VERSION ${HIPDNN_NANOBIND_VERSION})

# Find hipDNN frontend headers
set(HIPDNN_FRONTEND_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../frontend/include"
    CACHE PATH "Path to hipDNN frontend includes"
)

# Find HIP (if needed)
find_package(hip REQUIRED)

# When built as a subdirectory of hipdnn, these targets already exist.
# When built standalone (pip wheel / pip install), find the installed libraries.
if(NOT TARGET hipdnn_frontend)
    find_package(hipdnn_frontend CONFIG REQUIRED)
endif()
if(NOT TARGET hipdnn_backend)
    find_package(hipdnn_backend REQUIRED)
endif()

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

# ============================================================================
# Installation
# ============================================================================

# SKBUILD is set by scikit-build-core when building a wheel.
if(DEFINED SKBUILD)
    # pip install . / pip wheel . → scikit-build-core packages the extension into a wheel
    install(TARGETS hipdnn_frontend_python DESTINATION .)
else()
    # cmake --install → calls pip wheel to build .whl, stages it in share/hipdnn/wheels/
    install(CODE "
        set(_wheel_dir \"\${CMAKE_INSTALL_PREFIX}/share/hipdnn/wheels\")
        file(MAKE_DIRECTORY \"\${_wheel_dir}\")

        message(STATUS \"Building hipdnn-frontend wheel\")
        execute_process(
            COMMAND \"${Python_EXECUTABLE}\" -m pip wheel --no-deps
                -w \"\${_wheel_dir}\"
                \"${CMAKE_CURRENT_SOURCE_DIR}\"
                -C \"cmake.define.CMAKE_PREFIX_PATH=\${CMAKE_INSTALL_PREFIX}\"
                -C \"cmake.define.nanobind_DIR=${nanobind_DIR}\"
                -C \"cmake.define.tsl-robin-map_DIR=${tsl-robin-map_DIR}\"
            RESULT_VARIABLE _result
        )
        if(_result)
            message(FATAL_ERROR \"Failed to build hipdnn-frontend wheel\")
        endif()
    ")
endif()
