# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# bindings/CMakeLists.txt -- builds the pybind11 module `rocke_engine` that wraps
# the prebuilt C++ rocke engine archive (librocke_core.a). This is additive and
# self-contained: it does NOT add the engine sources, it links the archive.
#
# Configure (example):
#   cmake -S bindings -B /tmp/rocke_pybind/build \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DROCKE_ENGINE_ARCHIVE=/tmp/rocke_pybind/engine/librocke_core.a \
#         -DROCKE_INCLUDE_DIR=<repo>/rocke/include \
#         -Dpybind11_DIR=$(python -m pybind11 --cmakedir) \
#         -DPYTHON_EXECUTABLE=/path/to/python
#   cmake --build /tmp/rocke_pybind/build -j
#
# Output: rocke_engine<EXT>.so under the build dir.

cmake_minimum_required(VERSION 3.18)
project(rocke_engine_binding LANGUAGES CXX)

include(GNUInstallDirs)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- Locate the prebuilt engine archive (read-only; built separately) --------
set(ROCKE_ENGINE_ARCHIVE "" CACHE FILEPATH "Path to prebuilt librocke_core.a")
if(TARGET rocke_core)
  set(_ROCKE_BINDING_LINK_TARGET rocke_core)
elseif(ROCKE_ENGINE_ARCHIVE AND EXISTS "${ROCKE_ENGINE_ARCHIVE}")
  set(_ROCKE_BINDING_LINK_TARGET "${ROCKE_ENGINE_ARCHIVE}")
else()
  message(FATAL_ERROR
    "ROCKE_ENGINE_ARCHIVE must point to a prebuilt librocke_core.a "
    "when the binding is configured outside the rocKE engine build "
    "(got: '${ROCKE_ENGINE_ARCHIVE}'). Build the engine first:\n"
    "  cmake -S <rocke/platform> -B <bld>/engine -DCMAKE_BUILD_TYPE=Release\n"
    "  cmake --build <bld>/engine -j")
endif()

# --- Engine public headers (include/rocke/*.h) ---------------------------------
set(ROCKE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    CACHE PATH "Path to rocke/include")
if(NOT EXISTS "${ROCKE_INCLUDE_DIR}/rocke/ir.h")
  message(FATAL_ERROR "ROCKE_INCLUDE_DIR='${ROCKE_INCLUDE_DIR}' has no rocke/ir.h")
endif()

# --- pybind11 ----------------------------------------------------------------
include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/rocke_pybind11.cmake")
rocke_locate_pybind11(_ROCKE_HAVE_PYBIND11)
if(NOT _ROCKE_HAVE_PYBIND11)
  message(FATAL_ERROR
    "rocke_engine needs pybind11, which ${Python3_EXECUTABLE} does not have: "
    "install it there (python -m pip install 'pybind11>=3.0') or set pybind11_DIR.")
endif()
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(rocke_engine rocke_engine.cpp
                    rocke_engine_attention.cpp rocke_engine_moe_gemm_fused.cpp
                    rocke_engine_fmha_extra.cpp rocke_engine_fused_moe_e2e.cpp)
target_include_directories(rocke_engine PRIVATE "${ROCKE_INCLUDE_DIR}")

# Link the engine archive whole, plus libm. The engine is C++; its symbols are
# exposed via extern "C" headers, so the archive links cleanly here.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(rocke_engine PRIVATE "${_ROCKE_BINDING_LINK_TARGET}"
                      ${CMAKE_DL_LIBS} Threads::Threads)
if(UNIX)
  target_link_libraries(rocke_engine PRIVATE m)
endif()

if(NOT DEFINED ROCKE_PYTHON_INSTALL_DIR)
  set(ROCKE_PYTHON_INSTALL_DIR
      "${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
      CACHE STRING "Relative install destination for rocke_engine")
endif()
# Inside the rocKE build, platform/CMakeLists.txt sets where the rocke package
# installs; the module goes beside it so whatever imports rocke imports it too.
if(NOT DEFINED ROCKE_PYTHON_PACKAGE_INSTALL_DIR)
  set(ROCKE_PYTHON_PACKAGE_INSTALL_DIR "${ROCKE_PYTHON_INSTALL_DIR}")
endif()

install(TARGETS rocke_engine
        LIBRARY DESTINATION "${ROCKE_PYTHON_PACKAGE_INSTALL_DIR}"
        RUNTIME DESTINATION "${ROCKE_PYTHON_PACKAGE_INSTALL_DIR}")
# The installed test tree records which engine modules the build makes
# (platform/CMakeLists.txt writes the marker its guard reads).
set_property(GLOBAL APPEND PROPERTY ROCKE_ENGINE_PYTHON_MODULES rocke_engine)
