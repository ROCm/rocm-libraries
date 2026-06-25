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

include(GNUInstallDirs)
find_package(Python3 COMPONENTS Interpreter QUIET)

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

option(CKC_BUILD_PYBIND "Build the ckc_engine Python extension module" OFF)
option(CKC_INSTALL_PYTHON_PACKAGE "Install the ck_dsl Python frontend" ON)
option(CKC_INSTALL_TESTS "Install rocKE smoke test artifacts and CTest metadata" ON)
option(CKC_INSTALL_STANDALONE_CTEST "Install standalone rocKE CTest metadata" ON)
if(NOT DEFINED CKC_INSTALL_PROVIDER_CTEST_SUBDIR)
  set(CKC_INSTALL_PROVIDER_CTEST_SUBDIR "" CACHE STRING
      "Provider installed CTest subdir to append rocKE tests to, e.g. hip_kernel_provider")
endif()

if(Python3_Interpreter_FOUND)
  set(_CKC_DEFAULT_PYTHON_INSTALL_DIR
      "${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")
else()
  set(_CKC_DEFAULT_PYTHON_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/python/site-packages")
endif()
set(CKC_PYTHON_INSTALL_DIR "${_CKC_DEFAULT_PYTHON_INSTALL_DIR}"
    CACHE STRING "Relative install destination for ck_dsl and ckc_engine")

# Sources live under Cpp/, in subfolders mirroring the ck_dsl Python package
# (Cpp/core/{ir,arch,isa,lower_llvm,lower_hip}, Cpp/helpers, Cpp/instances/{common,gfxNNNN},
# Cpp/support). GLOB_RECURSE picks up every .cpp regardless of depth; the
# pybind bindings under Cpp/bindings are a separate target and are excluded.
file(GLOB_RECURSE CKC_CORE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/Cpp/*.cpp")
list(FILTER CKC_CORE_SOURCES EXCLUDE REGEX "/Cpp/bindings/")

add_library(ckc_core STATIC ${CKC_CORE_SOURCES})
target_include_directories(ckc_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Cpp/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# --- Engine freshness / provenance stamp -------------------------------------
# Compute a deterministic, git-independent content hash of the engine source
# tree and inject it (plus a human version) ONLY into src/core/ckc_build_id.cpp via
# compile definitions. Scoping the -D to that single TU keeps every other object
# untouched: only ckc_build_id.o recompiles when the hash changes, and -- crucially
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
  "${CMAKE_CURRENT_SOURCE_DIR}/Cpp/core/ckc_build_id.cpp"
  PROPERTIES COMPILE_DEFINITIONS
  "CKC_BUILD_ID=${CKC_BUILD_ID};CKC_ENGINE_VERSION=${CKC_ENGINE_VERSION}")
# Required so the .a can be linked into the shared ck_dsl_provider plugin
# (--whole-archive into a .so): some const data tables otherwise emit
# R_X86_64_PC32 relocations the linker rejects for a PIC shared object.
set_target_properties(ckc_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

include(CTest)

# C++ test targets are registered in the by-layer tests/ tree (tests/CMakeLists.txt).
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  add_subdirectory(tests)
endif()

if(CKC_BUILD_PYBIND)
  add_subdirectory(Cpp/bindings)
endif()

# --- Installable engine + test artifacts -------------------------------------
install(TARGETS ckc_core
        EXPORT ckcTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Cpp/include/"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT ckcTargets
        FILE ckcTargets.cmake
        NAMESPACE ckc::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ckc)

if(CKC_INSTALL_PYTHON_PACKAGE)
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Python/ck_dsl"
          DESTINATION "${CKC_PYTHON_INSTALL_DIR}"
          FILES_MATCHING
          PATTERN "*.py")
endif()

if(CKC_INSTALL_TESTS)
  install(PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/tools/rocke_installed_smoke.py"
          DESTINATION ${CMAKE_INSTALL_BINDIR})

  set(_CKC_INSTALLED_CTEST_ENTRIES "")
  get_property(_CKC_INSTALL_TEST_TARGETS GLOBAL PROPERTY CKC_INSTALL_TEST_TARGETS)
  foreach(_test_target ${_CKC_INSTALL_TEST_TARGETS})
    string(APPEND _CKC_INSTALLED_CTEST_ENTRIES
           "add_test(${_test_target} \"../${_test_target}\")\n")
    string(APPEND _CKC_INSTALLED_CTEST_ENTRIES
           "set_tests_properties(${_test_target} PROPERTIES LABELS \"ckc;unit;host;quick\")\n")
  endforeach()

  string(APPEND _CKC_INSTALLED_CTEST_ENTRIES
         "add_test(rocke_python_import_smoke \"../rocke_installed_smoke.py\")\n")
  string(APPEND _CKC_INSTALLED_CTEST_ENTRIES
         "set_tests_properties(rocke_python_import_smoke PROPERTIES LABELS \"ckc;python;quick\")\n")

  if(CKC_INSTALL_PROVIDER_CTEST_SUBDIR)
    set_property(GLOBAL APPEND_STRING PROPERTY
                 "EXTERNAL_TEST_INSTALL_STAGING_${CKC_INSTALL_PROVIDER_CTEST_SUBDIR}"
                 "\n# rocKE installed smoke tests\n${_CKC_INSTALLED_CTEST_ENTRIES}")
  endif()

  if(CKC_INSTALL_STANDALONE_CTEST)
    set(_CKC_INSTALLED_CTEST "${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake.install")
    file(WRITE "${_CKC_INSTALLED_CTEST}"
         "# Autogenerated CTestTestfile for installed rocKE tests\n")
    file(APPEND "${_CKC_INSTALLED_CTEST}"
         "# Generated by the rocKE build system\n\n")
    file(APPEND "${_CKC_INSTALLED_CTEST}" "${_CKC_INSTALLED_CTEST_ENTRIES}")

    install(FILES "${_CKC_INSTALLED_CTEST}"
            DESTINATION ${CMAKE_INSTALL_BINDIR}/rocke
            RENAME CTestTestfile.cmake)
  endif()
endif()
