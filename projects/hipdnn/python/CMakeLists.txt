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

install(TARGETS hipdnn_frontend_python
    LIBRARY DESTINATION lib/hipdnn_python_staging
)

install(CODE "
    set(_staging \"\${CMAKE_INSTALL_PREFIX}/lib/hipdnn_python_staging\")
    set(_wheel_dir \"\${CMAKE_INSTALL_PREFIX}/share/hipdnn/wheels\")
    file(MAKE_DIRECTORY \"\${_wheel_dir}\")

    file(GLOB _so_files \"\${_staging}/hipdnn_frontend_python*.so\")
    list(LENGTH _so_files _so_count)
    if(_so_count EQUAL 0)
        message(FATAL_ERROR \"No hipdnn_frontend_python .so found in \${_staging}\")
    endif()
    list(GET _so_files 0 _so_path)

    message(STATUS \"Assembling hipdnn-frontend wheel from \${_so_path}\")
    execute_process(
        COMMAND \"${Python_EXECUTABLE}\"
            \"${CMAKE_CURRENT_SOURCE_DIR}/scripts/assemble_wheel.py\"
            --so-path \"\${_so_path}\"
            --package-dir \"${CMAKE_CURRENT_SOURCE_DIR}/hipdnn_frontend\"
            --output-dir \"\${_wheel_dir}\"
        RESULT_VARIABLE _result
    )
    if(_result)
        message(FATAL_ERROR \"Failed to assemble hipdnn-frontend wheel\")
    endif()

    file(REMOVE_RECURSE \"\${_staging}\")
")
