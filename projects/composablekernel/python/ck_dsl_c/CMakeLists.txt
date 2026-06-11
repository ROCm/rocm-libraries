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

add_compile_options(-Wall -Wextra -Wno-unused-parameter)

file(GLOB CKC_CORE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")

add_library(ckc_core STATIC ${CKC_CORE_SOURCES})
target_include_directories(ckc_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Optional smoke test: cmake --build build --target ckc_smoke
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/smoke.c")
  add_executable(ckc_smoke tests/smoke.c)
  target_link_libraries(ckc_smoke PRIVATE ckc_core)
endif()
