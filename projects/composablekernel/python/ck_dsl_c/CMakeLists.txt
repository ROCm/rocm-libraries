# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# ck_dsl_c -- pure C99 port of the ck_dsl Python IR -> LLVM codegen engine.
# Builds a static library `ckc_core` from every src/*.c. New port modules
# (lower_llvm, lower_hip, ...) drop their .c files into src/ and are picked up
# automatically by the glob below; re-run cmake after adding files.

cmake_minimum_required(VERSION 3.16)
project(ck_dsl_c LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# -Wno-comment: doc-comments faithfully quote LLVM IR / Python source containing
#   '/*' sequences (e.g. pointer math, packed-lane notes) -- intentional, not a bug.
# -Wno-unused-function: a few static parity-anchor / arch-gated helpers are kept
#   for documentation/symmetry even when a given config doesn't reference them.
# -Wno-format-truncation: error-message snprintf calls write into bounded
#   CKC_ERR_MSG_CAP buffers; truncating a long diagnostic string is intentional.
add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-comment
                    -Wno-unused-function -Wno-format-truncation)

file(GLOB CKC_CORE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")

add_library(ckc_core STATIC ${CKC_CORE_SOURCES})
target_include_directories(ckc_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
# Required so the .a can be linked into the shared ck_dsl_provider plugin
# (--whole-archive into a .so): some const data tables otherwise emit
# R_X86_64_PC32 relocations the linker rejects for a PIC shared object.
set_target_properties(ckc_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

include(CTest)

# Optional smoke test: cmake --build build --target ckc_smoke
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/smoke.c")
  add_executable(ckc_smoke tests/smoke.c)
  target_link_libraries(ckc_smoke PRIVATE ckc_core)
endif()

if(BUILD_TESTING)
  add_executable(ckc_tiled_attention_2d_reentrancy
                 tests/tiled_attention_2d_reentrancy.c)
  target_link_libraries(ckc_tiled_attention_2d_reentrancy PRIVATE ckc_core)
  if(UNIX)
    target_link_libraries(ckc_tiled_attention_2d_reentrancy PRIVATE m)
  endif()
  add_test(NAME ckc_tiled_attention_2d_reentrancy
           COMMAND ckc_tiled_attention_2d_reentrancy)
  set_tests_properties(ckc_tiled_attention_2d_reentrancy
                       PROPERTIES LABELS "ckc;unit;host")
endif()
