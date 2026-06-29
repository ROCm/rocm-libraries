# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# rocke -- C++20 port of the rocke Python IR -> LLVM codegen engine.
# Builds a static library `rocke_core` from every src/**/*.cpp. Sources live in
# subfolders mirroring the Python package; new modules drop into the matching
# src/ subfolder and are picked up automatically by the recursive glob below;
# re-run cmake after adding files.

cmake_minimum_required(VERSION 3.16)
# The engine translation units are C++20 (.cpp). The public ABI surface in
# include/ stays extern "C" so emitters, the provider plugin, and the language
# bindings link unchanged.
project(rocke LANGUAGES CXX)

include(GNUInstallDirs)
find_package(Python3 COMPONENTS Interpreter QUIET)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Do not run clang-tidy on rocKE. rocKE is a vendored dual-engine stack whose
# public surface is an intentionally flat, snake_case `extern "C"` ABI
# (`rocke_*`, `rocke_strbuf`, ...), constrained by the Python<->C++ byte-identity
# contract; it cannot satisfy the inherited hipDNN clang-tidy naming rules.
# RockCI configures a global -DCMAKE_CXX_CLANG_TIDY, so every C++ target would
# otherwise run that gate (and a rocKE-local `.clang-tidy` with `Checks: -*`
# fails outright with "no checks enabled"). Clearing the variable in this
# subtree (and child scopes: tests/, bindings/) leaves the rocKE targets'
# <LANG>_CLANG_TIDY properties empty so clang-tidy never runs on them. rocKE
# keeps its own -Wall -Wextra -Werror build plus tools/check_byte_identity.py
# as its quality bar.
set(CMAKE_CXX_CLANG_TIDY "")
set(CMAKE_C_CLANG_TIDY "")

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

# On Windows the MSVC CRT marks standard C functions (getenv/strcpy/fopen/
# fscanf/...) as "deprecated" via __declspec(deprecated); rocKE's -Werror would
# turn those -Wdeprecated-declarations diagnostics into hard build errors. rocKE
# uses the portable standard names deliberately (shared verbatim with the Linux
# build and bound by the Python<->C++ byte-identity contract), so opt out of the
# MSVC-only deprecation rather than fork the sources to the non-portable *_s
# variants. Defined for the whole rocKE subtree (engine, tests, bindings) so it
# also covers call sites not reached before ninja stops on the first failure.
if(WIN32)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_WARNINGS)
endif()

# Optional Address+UndefinedBehavior sanitizer build (diagnostics only, not for
# Release artifacts):
#   cmake -S . -B build_asan -DROCKE_SANITIZE=ON && cmake --build build_asan -j
# Applies to the engine, the unit tests, and (via run_diff --archive) the parity
# emitters that link librocke_core.a.
# _FORTIFY_SOURCE is disabled here because it conflicts with AddressSanitizer
# (officially discouraged). -Wformat-truncation is also suppressed in this build
# ONLY: the sanitizer instrumentation perturbs inlining enough that the -O
# interprocedural analysis raises a false-positive "null format string" on the
# variadic vsnprintf emit helpers (rocke_ll_emitf / rocke_ll_block_emitf, whose
# forwarded format is always a literal). The normal/Release build keeps
# -Werror=format-truncation, where it does not false-positive. This build is for
# memory-safety diagnostics, not format-truncation.
option(ROCKE_SANITIZE "Build with -fsanitize=address,undefined" OFF)
if(ROCKE_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g
                      -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -Wno-format-truncation)
  add_link_options(-fsanitize=address,undefined)
endif()

option(ROCKE_BUILD_PYBIND "Build the rocke_engine Python extension module" OFF)
option(ROCKE_INSTALL_PYTHON_PACKAGE "Install the rocke Python frontend" ON)
option(ROCKE_INSTALL_TESTS "Install rocKE smoke test artifacts and CTest metadata" ON)
option(ROCKE_INSTALL_STANDALONE_CTEST "Install standalone rocKE CTest metadata" ON)
if(NOT DEFINED ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR)
  set(ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR "" CACHE STRING
      "Provider installed CTest subdir to append rocKE tests to, e.g. hip_kernel_provider")
endif()

# Where rocKE's installed test artifacts (the C++ smoke binaries + the Python
# import-smoke script) land, and how the generated add_test() commands reach
# them from the CTestTestfile directory:
#
#  * Provider mode (ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR set, e.g.
#    "hip_kernel_provider"): install the artifacts INTO that subdir, next to the
#    provider's installed CTestTestfile.cmake, and run them via "./<name>".
#    TheRock packages its test artifact by the provider's "<subdir>" prefix (the
#    CTestTestfile and the bin/<subdir>* test binaries); a bare bin/ executable
#    with an unrelated name (rocke_*, rocke_*) is routed to a different artifact
#    and is therefore missing from THEROCK_BIN_DIR at test time -> "Not Run".
#    Co-locating ours under bin/<subdir>/ gets them into the same test bucket.
#  * Standalone mode: keep the artifacts in bin/ with the CTestTestfile under
#    bin/rocke/, reached via "../<name>".
if(ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR)
  set(ROCKE_TEST_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}/${ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR}")
  set(ROCKE_TEST_CTEST_REL "./")
else()
  set(ROCKE_TEST_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}")
  set(ROCKE_TEST_CTEST_REL "../")
endif()

if(Python3_Interpreter_FOUND)
  set(_ROCKE_DEFAULT_PYTHON_INSTALL_DIR
      "${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")
else()
  set(_ROCKE_DEFAULT_PYTHON_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/python/site-packages")
endif()
set(ROCKE_PYTHON_INSTALL_DIR "${_ROCKE_DEFAULT_PYTHON_INSTALL_DIR}"
    CACHE STRING "Relative install destination for rocke and rocke_engine")

# Sources live under Cpp/, in subfolders mirroring the rocke Python package
# (Cpp/core/{ir,arch,isa,lower_llvm,lower_hip}, Cpp/helpers, Cpp/instances/{common,gfxNNNN},
# Cpp/support). GLOB_RECURSE picks up every .cpp regardless of depth; the
# pybind bindings under Cpp/bindings are a separate target and are excluded.
file(GLOB_RECURSE ROCKE_CORE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/Cpp/*.cpp")
list(FILTER ROCKE_CORE_SOURCES EXCLUDE REGEX "/Cpp/bindings/")

add_library(rocke_core STATIC ${ROCKE_CORE_SOURCES})
target_include_directories(rocke_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Cpp/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# --- Engine freshness / provenance stamp -------------------------------------
# Compute a deterministic, git-independent content hash of the engine source
# tree and inject it (plus a human version) ONLY into src/core/rocke_build_id.cpp via
# compile definitions. Scoping the -D to that single TU keeps every other object
# untouched: only rocke_build_id.o recompiles when the hash changes, and -- crucially
# -- no lowering/emission TU ever sees these values, so the emitted LLVM-IR is
# byte-identical. The build_id TU exposes rocke_build_id()/rocke_engine_version()
# and is intentionally NOT referenced by any emission path; consumers (harnesses,
# the language bindings, the provider) read it purely as an artifact stamp.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/rocke_build_id.cmake")
rocke_compute_build_id(ROCKE_BUILD_ID "${CMAKE_CURRENT_SOURCE_DIR}")
# Human-readable version. Bump the leading semver when the IR contract changes;
# the date keeps successive builds from the same version distinguishable.
string(TIMESTAMP ROCKE_BUILD_DATE "%Y%m%d" UTC)
set(ROCKE_ENGINE_VERSION "1.0.0+${ROCKE_BUILD_DATE}")
message(STATUS "rocke_core build-id: ${ROCKE_BUILD_ID}  version: ${ROCKE_ENGINE_VERSION}")
set_source_files_properties(
  "${CMAKE_CURRENT_SOURCE_DIR}/Cpp/core/rocke_build_id.cpp"
  PROPERTIES COMPILE_DEFINITIONS
  "ROCKE_BUILD_ID=${ROCKE_BUILD_ID};ROCKE_ENGINE_VERSION=${ROCKE_ENGINE_VERSION}")
# Required so the .a can be linked into the shared rocke_provider plugin
# (--whole-archive into a .so): some const data tables otherwise emit
# R_X86_64_PC32 relocations the linker rejects for a PIC shared object.
set_target_properties(rocke_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

include(CTest)

# C++ test targets are registered in the by-layer tests/ tree (tests/CMakeLists.txt).
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  add_subdirectory(tests)
endif()

if(ROCKE_BUILD_PYBIND)
  add_subdirectory(Cpp/bindings)
endif()

# --- Installable engine + test artifacts -------------------------------------
install(TARGETS rocke_core
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

if(ROCKE_INSTALL_PYTHON_PACKAGE)
  # In provider mode, co-locate the rocke package under the captured test subdir
  # (bin/<subdir>/rocke) next to rocke_installed_smoke.py so the Python smoke is
  # fully self-contained inside TheRock's test artifact. rocke normally installs
  # to lib/pythonX.Y/site-packages, which is NOT under the <subdir> prefix and so
  # would not be fetched for the test job. Standalone keeps the site-packages
  # layout.
  if(ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR)
    set(_ROCKE_PKG_DEST "${ROCKE_TEST_INSTALL_DIR}")
  else()
    set(_ROCKE_PKG_DEST "${ROCKE_PYTHON_INSTALL_DIR}")
  endif()
  # Install the whole package (not just *.py): rocke ships runtime data such as
  # core/arch/data/arch_specs.json that the engine loads at import/build time.
  # A *.py-only filter silently drops it and breaks ArchTarget.from_gfx(). Use
  # PATTERN + REGEX (distinct keywords) to drop only build cruft and keep
  # cmake-lint happy (a second PATTERN would trip its duplicate-keyword check).
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Python/rocke"
          DESTINATION "${_ROCKE_PKG_DEST}"
          PATTERN "__pycache__" EXCLUDE
          REGEX "\\.pyc$" EXCLUDE)
endif()

if(ROCKE_INSTALL_TESTS)
  install(PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/tools/rocke_installed_smoke.py"
          DESTINATION ${ROCKE_TEST_INSTALL_DIR})

  set(_ROCKE_INSTALLED_CTEST_ENTRIES "")
  get_property(_ROCKE_INSTALL_TEST_TARGETS GLOBAL PROPERTY ROCKE_INSTALL_TEST_TARGETS)
  foreach(_test_target ${_ROCKE_INSTALL_TEST_TARGETS})
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "add_test(${_test_target} \"${ROCKE_TEST_CTEST_REL}${_test_target}\")\n")
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "set_tests_properties(${_test_target} PROPERTIES LABELS \"ckc;unit;host;quick\")\n")
  endforeach()

  string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
         "add_test(rocke_python_import_smoke \"${ROCKE_TEST_CTEST_REL}rocke_installed_smoke.py\")\n")
  string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
         "set_tests_properties(rocke_python_import_smoke PROPERTIES LABELS \"ckc;python;quick\")\n")

  if(ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR)
    set_property(GLOBAL APPEND_STRING PROPERTY
                 "EXTERNAL_TEST_INSTALL_STAGING_${ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR}"
                 "\n# rocKE installed smoke tests\n${_ROCKE_INSTALLED_CTEST_ENTRIES}")
  endif()

  if(ROCKE_INSTALL_STANDALONE_CTEST)
    set(_ROCKE_INSTALLED_CTEST "${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake.install")
    file(WRITE "${_ROCKE_INSTALLED_CTEST}"
         "# Autogenerated CTestTestfile for installed rocKE tests\n")
    file(APPEND "${_ROCKE_INSTALLED_CTEST}"
         "# Generated by the rocKE build system\n\n")
    file(APPEND "${_ROCKE_INSTALLED_CTEST}" "${_ROCKE_INSTALLED_CTEST_ENTRIES}")

    install(FILES "${_ROCKE_INSTALLED_CTEST}"
            DESTINATION ${CMAKE_INSTALL_BINDIR}/rocke
            RENAME CTestTestfile.cmake)
  endif()
endif()
