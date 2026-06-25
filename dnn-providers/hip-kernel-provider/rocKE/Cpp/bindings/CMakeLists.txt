# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# bindings/CMakeLists.txt -- builds the pybind11 module `ckc_engine` that wraps
# the prebuilt C++ ck_dsl_c engine archive (libckc_core.a). This is additive and
# self-contained: it does NOT add the engine sources, it links the archive.
#
# Configure (example):
#   cmake -S bindings -B /tmp/ckc_pybind/build \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DCKC_ENGINE_ARCHIVE=/tmp/ckc_pybind/engine/libckc_core.a \
#         -DCKC_INCLUDE_DIR=<repo>/ck_dsl_c/include \
#         -Dpybind11_DIR=$(python -m pybind11 --cmakedir) \
#         -DPYTHON_EXECUTABLE=/path/to/python
#   cmake --build /tmp/ckc_pybind/build -j
#
# Output: ckc_engine<EXT>.so under the build dir.

cmake_minimum_required(VERSION 3.18)
project(ckc_engine_binding LANGUAGES CXX)

include(GNUInstallDirs)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- Locate the prebuilt engine archive (read-only; built separately) --------
set(CKC_ENGINE_ARCHIVE "" CACHE FILEPATH "Path to prebuilt libckc_core.a")
if(TARGET ckc_core)
  set(_CKC_BINDING_LINK_TARGET ckc_core)
elseif(CKC_ENGINE_ARCHIVE AND EXISTS "${CKC_ENGINE_ARCHIVE}")
  set(_CKC_BINDING_LINK_TARGET "${CKC_ENGINE_ARCHIVE}")
else()
  message(FATAL_ERROR
    "CKC_ENGINE_ARCHIVE must point to a prebuilt libckc_core.a "
    "when the binding is configured outside the rocKE engine build "
    "(got: '${CKC_ENGINE_ARCHIVE}'). Build the engine first:\n"
    "  cmake -S <rocKE> -B <bld>/engine -DCMAKE_BUILD_TYPE=Release\n"
    "  cmake --build <bld>/engine -j")
endif()

# --- Engine public headers (include/ckc/*.h) ---------------------------------
set(CKC_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    CACHE PATH "Path to ck_dsl_c/include")
if(NOT EXISTS "${CKC_INCLUDE_DIR}/ckc/ir.h")
  message(FATAL_ERROR "CKC_INCLUDE_DIR='${CKC_INCLUDE_DIR}' has no ckc/ir.h")
endif()

# --- pybind11 ----------------------------------------------------------------
if(NOT pybind11_DIR)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE _CKC_PYBIND11_CMAKEDIR
    RESULT_VARIABLE _CKC_PYBIND11_CMAKEDIR_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(_CKC_PYBIND11_CMAKEDIR_RESULT EQUAL 0 AND EXISTS "${_CKC_PYBIND11_CMAKEDIR}")
    set(pybind11_DIR "${_CKC_PYBIND11_CMAKEDIR}" CACHE PATH "pybind11 CMake package directory")
  endif()
endif()
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(ckc_engine ckc_engine.cpp ckc_engine_img2col.cpp
                    ckc_engine_attention.cpp ckc_engine_moe_gemm_fused.cpp
                    ckc_engine_fmha_extra.cpp ckc_engine_fused_moe_e2e.cpp)
target_include_directories(ckc_engine PRIVATE "${CKC_INCLUDE_DIR}")

# Link the engine archive whole, plus libm. The engine is C++; its symbols are
# exposed via extern "C" headers, so the archive links cleanly here.
target_link_libraries(ckc_engine PRIVATE "${_CKC_BINDING_LINK_TARGET}")
if(UNIX)
  target_link_libraries(ckc_engine PRIVATE m)
endif()

if(NOT DEFINED CKC_PYTHON_INSTALL_DIR)
  set(CKC_PYTHON_INSTALL_DIR
      "${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
      CACHE STRING "Relative install destination for ckc_engine")
endif()

install(TARGETS ckc_engine
        LIBRARY DESTINATION "${CKC_PYTHON_INSTALL_DIR}"
        RUNTIME DESTINATION "${CKC_PYTHON_INSTALL_DIR}")
