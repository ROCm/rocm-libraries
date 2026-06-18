# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(DEFINED SKBUILD)
    cmake_minimum_required(VERSION 3.26)
    project(hipdnn_python_bindings LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python 3.12
    REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)

if(DEFINED SKBUILD)
    if(DEFINED ENV{ROCM_PATH} AND NOT DEFINED ROCM_PATH)
        set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "Path to the ROCm installation.")
    endif()
    if(DEFINED ENV{CMAKE_PREFIX_PATH} AND NOT "$ENV{CMAKE_PREFIX_PATH}" STREQUAL "")
        list(PREPEND CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")
    elseif(DEFINED ROCM_PATH)
        list(PREPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
    elseif(NOT CMAKE_PREFIX_PATH)
        set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to the ROCm installation.")
        list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
    endif()
    message(STATUS "hipDNN Python bindings CMAKE_PREFIX_PATH='${CMAKE_PREFIX_PATH}'")
    find_package(nanobind REQUIRED CONFIG)
    find_package(hipdnn_frontend CONFIG)
    if(NOT hipdnn_frontend_FOUND)
        message(FATAL_ERROR "hipdnn_frontendConfig.cmake not found. Set CMAKE_PREFIX_PATH to the ROCm/hipDNN prefix (for ROCm torch wheels, the _rocm_sdk_libraries_* package directory). Searched CMAKE_PREFIX_PATH='${CMAKE_PREFIX_PATH}'")
    endif()
    find_package(hipdnn_backend CONFIG)
    if(NOT hipdnn_backend_FOUND)
        message(FATAL_ERROR "hipdnn_backendConfig.cmake not found. Set CMAKE_PREFIX_PATH to the ROCm/hipDNN prefix (for ROCm torch wheels, the _rocm_sdk_libraries_* package directory). Searched CMAKE_PREFIX_PATH='${CMAKE_PREFIX_PATH}'")
    endif()
else()
    hipdnn_add_dependency(tsl-robin-map VERSION ${HIPDNN_TSL_ROBIN_MAP_VERSION})
    hipdnn_add_dependency(nanobind VERSION ${HIPDNN_NANOBIND_VERSION})
endif()

find_package(hip CONFIG)
if(NOT hip_FOUND)
    message(FATAL_ERROR "hipConfig.cmake not found. Set CMAKE_PREFIX_PATH to include the ROCm SDK devel prefix (for ROCm torch wheels, the _rocm_sdk_devel package directory). Searched CMAKE_PREFIX_PATH='${CMAKE_PREFIX_PATH}'")
endif()

nanobind_add_module(
    hipdnn_frontend_python
    STABLE_ABI
    src/module.cpp
    src/graph_bindings.cpp
    src/tensor_bindings.cpp
    src/attributes_bindings.cpp
    src/types_bindings.cpp
    src/handle_bindings.cpp
    src/memory_bindings.cpp
    src/hip_bindings.cpp
)

target_link_libraries(
    hipdnn_frontend_python
    PRIVATE
        hipdnn_frontend
        hipdnn_backend
        hip::host
)

# Co-locate the extension with libhipdnn_backend so a bare dlopen of the module
# (e.g. rocm-sdk's testSharedLibrariesLoad) resolves the backend as a sibling:
# $ORIGIN on Linux, the module's own directory on Windows. The backend installs
# to lib/ (Linux) / bin/ (Windows); CMake routes the MODULE artifact to the same
# place (LIBRARY on Linux, RUNTIME on Windows).
set_target_properties(
    hipdnn_frontend_python
    PROPERTIES
        INSTALL_RPATH "$ORIGIN"
)

if(DEFINED SKBUILD)
    install(TARGETS hipdnn_frontend_python DESTINATION hipdnn_frontend)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/hipdnn_frontend/__init__.py"
            DESTINATION hipdnn_frontend)
else()
    # The ext stays bare in LIBDIR/BINDIR as a libhipdnn_backend sibling so a
    # standalone dlopen resolves the backend via $ORIGIN. The importable package
    # (__init__.py) and its tests live under LIBDIR/python; `import
    # hipdnn_frontend` then loads the ext via its top-level name with LIBDIR on
    # PYTHONPATH.
    install(TARGETS hipdnn_frontend_python
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/hipdnn_frontend/__init__.py"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/python/hipdnn_frontend")
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/hipdnn_frontend/test/"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/python/tests"
            FILES_MATCHING PATTERN "*.py")
endif()
