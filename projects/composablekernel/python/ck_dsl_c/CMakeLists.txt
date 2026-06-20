# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# ck_dsl_c -- C++20 port of the ck_dsl Python IR -> LLVM codegen engine.
# Builds a static library `ckc_core` from every src/**/*.cpp. Sources live in
# subfolders mirroring the Python package; new modules drop into the matching
# src/ subfolder and are picked up automatically by the recursive glob below;
# re-run cmake after adding files.

cmake_minimum_required(VERSION 3.16)
# The engine translation units are C++20 (.cpp). The public ABI surface in
# include/ stays extern "C" so emitters, the provider plugin, and the language
# bindings link unchanged.
project(ck_dsl_c LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Build hardening: warnings are errors. Every -Wall/-Wextra diagnostic is
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

# Optional Address+UndefinedBehavior sanitizer build (diagnostics only, not for
# Release artifacts):
#   cmake -S . -B build_asan -DCKC_SANITIZE=ON && cmake --build build_asan -j
# Applies to the engine, the unit tests, and (via run_diff --archive) the parity
# emitters that link libckc_core.a.
# _FORTIFY_SOURCE is disabled here because it conflicts with AddressSanitizer
# (officially discouraged). -Wformat-truncation is also suppressed in this build
# ONLY: the sanitizer instrumentation perturbs inlining enough that the -O
# interprocedural analysis raises a false-positive "null format string" on the
# variadic vsnprintf emit helpers (ckc_ll_emitf / ckc_ll_block_emitf, whose
# forwarded format is always a literal). The normal/Release build keeps
# -Werror=format-truncation, where it does not false-positive. This build is for
# memory-safety diagnostics, not format-truncation.
option(CKC_SANITIZE "Build with -fsanitize=address,undefined" OFF)
if(CKC_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g
                      -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -Wno-format-truncation)
  add_link_options(-fsanitize=address,undefined)
endif()

# Sources are organized into subfolders mirroring the ck_dsl Python package
# (src/core/{ir,arch,isa,lower_llvm,lower_hip}, src/helpers, src/instances/{common,gfxNNNN},
# src/support). GLOB_RECURSE picks up every .cpp regardless of depth.
file(GLOB_RECURSE CKC_CORE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(ckc_core STATIC ${CKC_CORE_SOURCES})
target_include_directories(ckc_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

# --- Engine freshness / provenance stamp -------------------------------------
# Compute a deterministic, git-independent content hash of the engine source
# tree and inject it (plus a human version) ONLY into src/core/build_id.cpp via
# compile definitions. Scoping the -D to that single TU keeps every other object
# untouched: only build_id.o recompiles when the hash changes, and -- crucially
# -- no lowering/emission TU ever sees these values, so the emitted LLVM-IR is
# byte-identical. The build_id TU exposes ckc_build_id()/ckc_engine_version()
# and is intentionally NOT referenced by any emission path; consumers (harnesses,
# the language bindings, the provider) read it purely as an artifact stamp.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ckc_build_id.cmake")
ckc_compute_build_id(CKC_BUILD_ID "${CMAKE_CURRENT_SOURCE_DIR}")
# Human-readable version. Bump the leading semver when the IR contract changes;
# the date keeps successive builds from the same version distinguishable.
string(TIMESTAMP CKC_BUILD_DATE "%Y%m%d" UTC)
set(CKC_ENGINE_VERSION "1.0.0+${CKC_BUILD_DATE}")
message(STATUS "ckc_core build-id: ${CKC_BUILD_ID}  version: ${CKC_ENGINE_VERSION}")
set_source_files_properties(
  "${CMAKE_CURRENT_SOURCE_DIR}/src/core/build_id.cpp"
  PROPERTIES COMPILE_DEFINITIONS
  "CKC_BUILD_ID=${CKC_BUILD_ID};CKC_ENGINE_VERSION=${CKC_ENGINE_VERSION}")
# Required so the .a can be linked into the shared ck_dsl_provider plugin
# (--whole-archive into a .so): some const data tables otherwise emit
# R_X86_64_PC32 relocations the linker rejects for a PIC shared object.
set_target_properties(ckc_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

include(CTest)

# Optional smoke test: cmake --build build --target ckc_smoke
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/smoke.cpp")
  add_executable(ckc_smoke tests/smoke.cpp)
  target_link_libraries(ckc_smoke PRIVATE ckc_core)
endif()

if(BUILD_TESTING)
  add_executable(ckc_tiled_attention_2d_reentrancy
                 tests/tiled_attention_2d_reentrancy.cpp)
  target_link_libraries(ckc_tiled_attention_2d_reentrancy PRIVATE ckc_core)
  if(UNIX)
    target_link_libraries(ckc_tiled_attention_2d_reentrancy PRIVATE m)
  endif()
  add_test(NAME ckc_tiled_attention_2d_reentrancy
           COMMAND ckc_tiled_attention_2d_reentrancy)
  set_tests_properties(ckc_tiled_attention_2d_reentrancy
                       PROPERTIES LABELS "ckc;unit;host")

  # ck.dsl.ir/v1 serializer round-trip + verifier test. Proves
  # lower(parse(serialize(k))) == lower(k) byte-for-byte, serialization
  # idempotence, and a clean verifier pass for the buildable GEMM configs.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/ir_serialize_roundtrip.cpp")
    add_executable(ckc_ir_serialize_roundtrip tests/ir_serialize_roundtrip.cpp)
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
