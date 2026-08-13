# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.26)
project(hipdnn_frontend_bindings LANGUAGES CXX)
include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(NB_USE_SUBMODULE_DEPS OFF)

find_package(Python 3.12
    REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)

find_package(tsl-robin-map 1.4.1 CONFIG QUIET)
if(NOT TARGET tsl::robin_map)
    FetchContent_Declare(
        tsl_robin_map
        URL https://github.com/Tessil/robin-map/archive/refs/tags/v1.4.1.tar.gz
        URL_HASH SHA256=0e3f53a377fdcdc5f9fed7a4c0d4f99e82bbb64175233bd13427fef9a771f4a1
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(TSL_ROBIN_MAP_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(tsl_robin_map)
endif()

find_package(nanobind 2.12.0 CONFIG QUIET)
if(NOT COMMAND nanobind_add_module)
    FetchContent_Declare(
        nanobind
        URL https://github.com/wjakob/nanobind/archive/refs/tags/v2.12.0.tar.gz
        URL_HASH SHA256=01f1f0cd0398743c18f33d07ae36ad410bd7f4a1e90683b508504de897d6e629
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(NB_TEST OFF CACHE BOOL "" FORCE)
    set(NB_CREATE_INSTALL_RULES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(nanobind)
endif()

# When add_subdirectory'd from the rocm-libraries superbuild (the "hipdnn-python"
# component), hipdnn_frontend/hipdnn_backend already exist as in-tree targets;
# only fall back to find_package for a standalone build against an installed
# hipDNN prefix (e.g. a ROCm/hipDNN nightly artifact).
if(NOT TARGET hipdnn_frontend)
    find_package(hipdnn_frontend CONFIG)
    if(NOT hipdnn_frontend_FOUND)
        message(FATAL_ERROR "hipdnn_frontendConfig.cmake not found. Set CMAKE_PREFIX_PATH to the ROCm/hipDNN nightly artifact prefix. Searched CMAKE_PREFIX_PATH='${CMAKE_PREFIX_PATH}'")
    endif()
endif()

if(NOT TARGET hipdnn_backend)
    find_package(hipdnn_backend CONFIG)
    if(NOT hipdnn_backend_FOUND)
        message(FATAL_ERROR "hipdnn_backendConfig.cmake not found. Set CMAKE_PREFIX_PATH to the ROCm/hipDNN nightly artifact prefix. Searched CMAKE_PREFIX_PATH='${CMAKE_PREFIX_PATH}'")
    endif()
endif()

find_package(hip CONFIG)
if(NOT hip_FOUND)
    message(FATAL_ERROR "hipConfig.cmake not found. Set CMAKE_PREFIX_PATH to include the ROCm SDK devel prefix. Searched CMAKE_PREFIX_PATH='${CMAKE_PREFIX_PATH}'")
endif()

if(NOT TARGET hipdnn_frontend)
    message(FATAL_ERROR "hipdnn_frontend target is required for hipDNN Python bindings; update the nightly frontend artifact.")
endif()
if(NOT TARGET hipdnn_backend)
    message(FATAL_ERROR "hipdnn_backend target is required for hipDNN Python bindings; update the nightly backend artifact.")
endif()

nanobind_add_module(
    ${PROJECT_NAME}
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
    ${PROJECT_NAME}
    PRIVATE
        hipdnn_frontend
        hipdnn_backend
        hip::host
)

# Keep the compiled extension filename aligned with NB_MODULE(hipdnn_frontend_python).
# The public import remains `import hipdnn_frontend as hipdnn`; the extension is
# loaded by hipdnn_frontend/__init__.py as a package-private implementation detail.
# When installed into the same prefix as the prebuilt backend, $ORIGIN resolves
# libhipdnn_backend as a sibling; otherwise runtime loading relies on rocm_sdk,
# ROCM_PATH/HIP_PATH, or the platform loader search path.
set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
        OUTPUT_NAME "hipdnn_frontend_python"
        INSTALL_RPATH "$ORIGIN"
)
