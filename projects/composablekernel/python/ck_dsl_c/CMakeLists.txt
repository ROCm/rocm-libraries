# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# ck_dsl_c -- pure C99 port of the ck_dsl Python IR -> LLVM codegen engine.
# Builds a static library `ckc_core` from every src/**/*.c. Sources live in
# subfolders mirroring the Python package; new modules drop into the matching
# src/ subfolder and are picked up automatically by the recursive glob below;
# re-run cmake after adding files.

cmake_minimum_required(VERSION 3.16)
# WS3 Phase 1: the C99 sources are compiled as C++20 (no file renames). C is kept
# enabled so the .c extension still configures; LANGUAGE CXX is forced per-source
# below so every translation unit goes through the C++ frontend.
project(ck_dsl_c LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# WS3 Phase 2 hardening: warnings are errors. Every -Wall/-Wextra diagnostic is
# fixed at the root (unused vars removed; unused params/funcs given (void)/
# __attribute__((unused)) with intent comments). Two diagnostic classes are
# retained as -Wno-* because they are provably cosmetic and cannot mask a bug:
#
# -Wno-comment: doc-comments faithfully quote LLVM IR / Python source containing
#   '/*' sequences (e.g. pointer math, packed-lane notes) inside // line comments.
#   A '/*' inside a // comment cannot start a block comment, so it is impossible
#   for this to hide code -- purely a quoting artifact, intentional.
# -Wno-missing-field-initializers: every occurrence is the C-idiomatic `= {0}` /
#   `(T){0}` zero-initializer, which value-initializes ALL members per the C and
#   C++ standards. GCC still warns that only one member is named in the braces;
#   this is a well-known false positive for `{0}` and cannot leave a field
#   uninitialized. Naming every field (or switching to `{}`, a C++-only form)
#   would add noise without removing any bug class. (Part of -Wextra.)
add_compile_options(-Wall -Wextra -Werror
                    -Wno-comment -Wno-missing-field-initializers)

# WS3 Phase 2 hardening: optional Address+UndefinedBehavior sanitizer build.
#   cmake -S . -B build_asan -DCKC_SANITIZE=ON && cmake --build build_asan -j
# Applies to the engine, the unit tests, and (via run_diff --archive) the parity
# emitters that link libckc_core.a. Not for Release artifacts -- diagnostics only.
option(CKC_SANITIZE "Build with -fsanitize=address,undefined" OFF)
if(CKC_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
endif()

# Sources are organized into subfolders mirroring the ck_dsl Python package
# (src/core/{ir,arch,isa,lower_llvm,lower_hip}, src/helpers, src/instances/{common,gfxNNNN},
# src/support). GLOB_RECURSE picks up every .c regardless of depth.
file(GLOB_RECURSE CKC_CORE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")

# WS3 Phase 1: compile the .c sources through the C++20 frontend without renaming
# them (RFC two-step rule: get it building as C++ identically first).
set_source_files_properties(${CKC_CORE_SOURCES} PROPERTIES LANGUAGE CXX)

add_library(ckc_core STATIC ${CKC_CORE_SOURCES})
target_include_directories(ckc_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
# Required so the .a can be linked into the shared ck_dsl_provider plugin
# (--whole-archive into a .so): some const data tables otherwise emit
# R_X86_64_PC32 relocations the linker rejects for a PIC shared object.
set_target_properties(ckc_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

include(CTest)

# Optional smoke test: cmake --build build --target ckc_smoke
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/smoke.c")
  set_source_files_properties(tests/smoke.c PROPERTIES LANGUAGE CXX)
  add_executable(ckc_smoke tests/smoke.c)
  target_link_libraries(ckc_smoke PRIVATE ckc_core)
endif()

if(BUILD_TESTING)
  set_source_files_properties(tests/tiled_attention_2d_reentrancy.c
                              PROPERTIES LANGUAGE CXX)
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

  # WS1: ck.dsl.ir/v1 serializer round-trip + verifier test. Proves
  # lower(parse(serialize(k))) == lower(k) byte-for-byte, serialization
  # idempotence, and a clean verifier pass for the buildable GEMM configs.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/ir_serialize_roundtrip.c")
    set_source_files_properties(tests/ir_serialize_roundtrip.c
                                PROPERTIES LANGUAGE CXX)
    add_executable(ckc_ir_serialize_roundtrip tests/ir_serialize_roundtrip.c)
    target_link_libraries(ckc_ir_serialize_roundtrip PRIVATE ckc_core)
    if(UNIX)
      target_link_libraries(ckc_ir_serialize_roundtrip PRIVATE m)
    endif()
    add_test(NAME ckc_ir_serialize_roundtrip
             COMMAND ckc_ir_serialize_roundtrip)
    set_tests_properties(ckc_ir_serialize_roundtrip
                         PROPERTIES LABELS "ckc;unit;host")
  endif()
endif()
