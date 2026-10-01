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
option(ROCKE_INSTALL_PYTHON_TESTS "Install rocKE Python smoke/golden CI tests" ON)
option(ROCKE_INSTALL_PLATFORM_PYTEST "Install the full rocKE platform pytest CTest entry" ON)
option(ROCKE_INSTALL_STANDALONE_CTEST "Install standalone rocKE CTest metadata" ON)
set(ROCKE_SDPA_REFERENCE_BUNDLE "" CACHE PATH
    "Qualified gfx942 SDPA reference bundle to install with the GPU tests")
if(NOT DEFINED ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR)
  set(ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR "" CACHE STRING
      "Provider installed CTest subdir to append rocKE tests to, e.g. hip_kernel_provider")
endif()

set(_ROCKE_SDPA_DEFAULT OFF)
# Package references by default for provider artifact builds. The
# rocm-libraries superbuild runs build-tree tests and does not fetch the bundle.
if(ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR AND NOT ROCM_LIBS_SUPERBUILD)
  set(_ROCKE_SDPA_DEFAULT ON)
endif()
option(ROCKE_INSTALL_SDPA_REFERENCE
       "Install pinned SDPA GPU tests from DVC data" ${_ROCKE_SDPA_DEFAULT})

# Where rocKE's installed CTest metadata and standalone Python smoke script land:
#  * Provider mode: append CTest entries to the provider's installed
#    bin/<subdir>/CTestTestfile.cmake. Provider-mode C++ smoke binaries use the
#    bare-bin naming scheme below so TheRock's existing test artifact patterns
#    capture them.
#  * Standalone mode: keep artifacts in bin/ with the CTestTestfile under
#    bin/rocke/, reached via "../<name>".
if(ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR)
  set(ROCKE_TEST_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}/${ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR}")
  set(ROCKE_TEST_CTEST_REL "./")
else()
  set(ROCKE_TEST_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}")
  set(ROCKE_TEST_CTEST_REL "../")
endif()

# C++ smoke binaries use a separate scheme so TheRock's test artifact captures
# them with NO descriptor edit. The hip-kernel-provider test artifact includes
# "bin/<subdir>*_test*", so in provider mode install the C++ tests at bare bin/
# under a "<subdir>_<target>_test" name (e.g.
# hip_kernel_provider_rocke_ir_serialize_roundtrip_test) and reference them from
# the provider CTestTestfile via "../". Standalone keeps the plain target name
# in bin/. (The Python import smoke and the rocke package, which cannot match
# that name pattern, are co-located under bin/<subdir>/ and captured by explicit
# includes in ml-libs/artifact-hipkernelprovider.toml.)
set(ROCKE_CXX_TEST_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}")
set(ROCKE_CXX_TEST_CTEST_REL "../")

# Map a C++ test target to its installed file name for the active mode.
function(rocke_cxx_test_installed_name out_var target)
  if(ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR)
    set(${out_var} "${ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR}_${target}_test"
        PARENT_SCOPE)
  else()
    set(${out_var} "${target}" PARENT_SCOPE)
  endif()
endfunction()

if(Python3_Interpreter_FOUND)
  set(_ROCKE_DEFAULT_PYTHON_INSTALL_DIR
      "${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")
else()
  set(_ROCKE_DEFAULT_PYTHON_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/python/site-packages")
endif()
set(ROCKE_PYTHON_INSTALL_DIR "${_ROCKE_DEFAULT_PYTHON_INSTALL_DIR}"
    CACHE STRING "Relative install destination for rocke and rocke_engine")

# Sources live under cpp/, in subfolders mirroring the rocke Python package
# (cpp/core/{ir,arch,isa,lower_llvm,lower_hip}, cpp/helpers, cpp/instances/{common,gfxNNNN},
# cpp/support). GLOB_RECURSE picks up every .cpp regardless of depth; the
# pybind bindings under cpp/bindings are a separate target and are excluded.
file(GLOB_RECURSE ROCKE_CORE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.cpp")
list(FILTER ROCKE_CORE_SOURCES EXCLUDE REGEX "/cpp/bindings/")

add_library(rocke_core STATIC ${ROCKE_CORE_SOURCES})
target_include_directories(rocke_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>
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
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/rocke_build_id.cpp"
  PROPERTIES COMPILE_DEFINITIONS
  "ROCKE_BUILD_ID=${ROCKE_BUILD_ID};ROCKE_ENGINE_VERSION=${ROCKE_ENGINE_VERSION}")
# Required so the .a can be linked into the shared rocke_provider plugin
# (--whole-archive into a .so): some const data tables otherwise emit
# R_X86_64_PC32 relocations the linker rejects for a PIC shared object.
set_target_properties(rocke_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- Shared engine for the portable-IR replay path ---------------------------
# The Python-side portable-IR tooling (rocke.portable_ir.src.online) reaches the
# C++ importer, recipe VM and lowerer through ctypes, which needs a .so rather
# than the archive. online.build_lib() can produce one on demand, but a CI gate
# wants a named target it can build and point ROCKE_ONLINE_LIB at, with the link
# flags living here rather than duplicated in workflow YAML.
#
# --whole-archive is required: nothing in the .so references the engine's
# entry points, so without it the linker drops every object as unused and
# produces a well-formed library with no symbols in it.
option(ROCKE_BUILD_SHARED_ENGINE
       "Also link a shared librocke.so (needed by the portable-IR ctypes path)" OFF)
if(ROCKE_BUILD_SHARED_ENGINE)
  # A SHARED target needs a source of its own. It has to be one that is NOT
  # already in rocke_core: reusing an engine TU would compile it here and pull
  # it in again through --whole-archive, and the duplicate definition is a link
  # error. A generated stub keeps the anchor out of the source tree.
  set(_rocke_anchor "${CMAKE_CURRENT_BINARY_DIR}/rocke_shared_anchor.cpp")
  file(WRITE "${_rocke_anchor}"
       "// Generated by CMake. Anchors the rocke_shared target; the engine\n"
       "// itself arrives via --whole-archive rocke_core.\n")
  add_library(rocke_shared SHARED "${_rocke_anchor}")
  set_target_properties(rocke_shared PROPERTIES OUTPUT_NAME rocke)
  target_link_libraries(rocke_shared PRIVATE
    -Wl,--whole-archive rocke_core -Wl,--no-whole-archive m)
  install(TARGETS rocke_shared LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

include(CTest)

# C++ test targets are registered in the by-layer tests/ tree (tests/CMakeLists.txt).
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  add_subdirectory(tests)
endif()

if(ROCKE_BUILD_PYBIND)
  add_subdirectory(cpp/bindings)
endif()

# --- Installable engine + test artifacts -------------------------------------
install(TARGETS rocke_core
        EXPORT ckcTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include/"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT ckcTargets
        FILE ckcTargets.cmake
        NAMESPACE ckc::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ckc)

if(ROCKE_INSTALL_PYTHON_PACKAGE)
  # In provider mode, co-locate the rocke package under the test subdir
  # (bin/<subdir>/rocke) next to rocke_installed_smoke.py so the Python smoke is
  # self-contained inside TheRock's test artifact (paired with a matching include
  # in ml-libs/artifact-hipkernelprovider.toml for bin/<subdir>/rocke/**).
  # Standalone keeps the lib/pythonX.Y/site-packages layout.
  if(ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR)
    set(_ROCKE_PKG_DEST "${ROCKE_TEST_INSTALL_DIR}")
  else()
    set(_ROCKE_PKG_DEST "${ROCKE_PYTHON_INSTALL_DIR}")
  endif()
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/rocke"
          DESTINATION "${_ROCKE_PKG_DEST}"
          PATTERN "__pycache__" EXCLUDE
          REGEX "\\.pyc$" EXCLUDE)

  # Also install pre-built wheels when available. These are not yet consumed
  # by CI (the existing PYTHONPATH=. approach is still used), but shipping
  # them in the artifact prepares for a future migration to pip-installed
  # wheels and enables the library tests to run.
  if(ROCKE_BUILD_PYENV)
    install(FILES
            "${ROCKE_WHEEL_DIR}/rocke-${ROCKE_WHEEL_VERSION}-py3-none-any.whl"
            "${ROCKE_WHEEL_DIR}/rocke_library-${ROCKE_WHEEL_VERSION}-py3-none-any.whl"
            DESTINATION "${ROCKE_TEST_INSTALL_DIR}/wheels")
  endif()
endif()

if(ROCKE_INSTALL_TESTS)
  if(ROCKE_INSTALL_PYTHON_TESTS)
    # The Python import smoke (placeholder for the rocke Python CI test) installs
    # next to the CTestTestfile in both modes: provider mode -> bin/<subdir>/,
    # standalone -> bin/. In provider mode it imports the co-located rocke package
    # installed above; TheRock's test artifact captures both via explicit includes.
    install(PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/tools/rocke_installed_smoke.py"
            DESTINATION ${ROCKE_TEST_INSTALL_DIR})

    # Host-only golden IR byte-stability gate. The relocatable runner rebuilds the
    # representative IR from the co-located rocke package and compares digests to
    # the committed golden. It needs the parity harness module (imported by name)
    # and the golden JSON co-located next to it; no GPU and no examples required.
    install(PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/tests/rocke_installed_golden_test.py"
            DESTINATION ${ROCKE_TEST_INSTALL_DIR})
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/tests/instances/rocke_ir_parity_harness.py"
            DESTINATION ${ROCKE_TEST_INSTALL_DIR})
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/tests/golden/rocke_representative_ir_sha256.json"
            DESTINATION ${ROCKE_TEST_INSTALL_DIR}/golden)

    # The harness above builds LIBRARY kernels for its attention, attention_dense,
    # attention_d256 and kda_chunkwise families (the one platform -> library
    # exception), so the library packages must be importable by both runners.
    # Without them those cases degrade to expected failures and
    # rocke_golden_static reports drift instead of passing.
    #
    # Destination matters: TheRock's artifact-hipkernelprovider.toml captures the
    # test component through a fixed glob list whose only recursive entry for this
    # subtree is bin/hip_kernel_provider/tests/**. Staging the packages one level
    # down under tests/ therefore gets them into the CI test artifact with no
    # change to that manifest; installing them at the test root would match no
    # glob and they would be silently dropped from the artifact.
    #
    # Guarded on the sibling library tree so a platform-only copy of this directory
    # still configures; keep the guard, it is the copy-able contract.
    set(_ROCKE_LIBRARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../library")
    if(EXISTS "${_ROCKE_LIBRARY_DIR}/kernels")
      foreach(_rocke_lib_pkg kernels builders dispatch benchmarks)
        install(DIRECTORY "${_ROCKE_LIBRARY_DIR}/${_rocke_lib_pkg}"
                DESTINATION "${ROCKE_TEST_INSTALL_DIR}/tests/library"
                PATTERN "__pycache__" EXCLUDE
                REGEX "\\.pyc$" EXCLUDE)
      endforeach()
      # Use the same explicit enrollment list as the Python harness.
      file(READ "${_ROCKE_LIBRARY_DIR}/tests/sdpa_reference/architectures/registry.json"
           _ROCKE_SDPA_REGISTRY)
      string(JSON _ROCKE_SDPA_COUNT LENGTH "${_ROCKE_SDPA_REGISTRY}")
      math(EXPR _ROCKE_SDPA_LAST "${_ROCKE_SDPA_COUNT} - 1")
      set(_ROCKE_SDPA_ARCHITECTURES "")
      foreach(_index RANGE 0 ${_ROCKE_SDPA_LAST} 1)
        string(JSON _arch GET "${_ROCKE_SDPA_REGISTRY}" ${_index})
        list(APPEND _ROCKE_SDPA_ARCHITECTURES "${_arch}")
      endforeach()
      set(ROCKE_SDPA_REFERENCE_ARCHITECTURES "${_ROCKE_SDPA_ARCHITECTURES}" CACHE STRING
          "Enrolled SDPA reference architectures to install")
      if(ROCKE_INSTALL_SDPA_REFERENCE AND NOT ROCKE_SDPA_REFERENCE_ARCHITECTURES)
        message(FATAL_ERROR "Select an enrolled SDPA architecture or disable ROCKE_INSTALL_SDPA_REFERENCE")
      endif()
      foreach(_arch IN LISTS ROCKE_SDPA_REFERENCE_ARCHITECTURES)
        if(NOT _arch IN_LIST _ROCKE_SDPA_ARCHITECTURES)
          message(FATAL_ERROR "SDPA reference architecture is not enrolled: ${_arch}")
        endif()
        set(ROCKE_SDPA_REFERENCE_BUNDLE_${_arch} "" CACHE PATH
            "Optional qualified SDPA bundle directory for ${_arch}")
        set(_bundle "${ROCKE_SDPA_REFERENCE_BUNDLE_${_arch}}")
        # Preserve the original local override for its original target.
        if(_arch STREQUAL "gfx942" AND ROCKE_SDPA_REFERENCE_BUNDLE AND NOT _bundle)
          set(_bundle "${ROCKE_SDPA_REFERENCE_BUNDLE}")
        endif()
        set(_lock "${_ROCKE_LIBRARY_DIR}/tests/sdpa_reference/architectures/${_arch}/baseline_lock.json")
        if(ROCKE_INSTALL_SDPA_REFERENCE AND NOT _bundle)
          set(_archive "${_ROCKE_LIBRARY_DIR}/tests/reference_bundles/${_arch}.tar.gz")
          if(NOT EXISTS "${_archive}")
            message(FATAL_ERROR
                    "SDPA reference archive missing. Run dvc pull for "
                    "dnn-providers/hip-kernel-provider/rocke/library/tests/reference_bundles/${_arch}.tar.gz.dvc "
                    "or configure with ROCKE_INSTALL_SDPA_REFERENCE=OFF.")
          endif()
          find_package(Python3 COMPONENTS Interpreter REQUIRED)
          file(SHA256 "${_archive}" _archive_sha256)
          set(_bundle "${CMAKE_CURRENT_BINARY_DIR}/sdpa-reference/${_arch}/${_archive_sha256}")
          execute_process(
            COMMAND "${Python3_EXECUTABLE}" -I
                    "${_ROCKE_LIBRARY_DIR}/tests/sdpa_reference/artifact.py" unpack
                    --archive "${_archive}" --bundle "${_bundle}" --lock "${_lock}"
            RESULT_VARIABLE _result ERROR_VARIABLE _error)
          if(NOT _result EQUAL 0)
            message(FATAL_ERROR "SDPA reference staging failed for ${_arch}: ${_error}")
          endif()
        endif()
        if(_bundle)
          find_package(Python3 COMPONENTS Interpreter REQUIRED)
          execute_process(
            COMMAND "${Python3_EXECUTABLE}" -I
                    "${_ROCKE_LIBRARY_DIR}/tests/sdpa_reference/artifact.py" validate
                    --bundle "${_bundle}" --lock "${_lock}"
            RESULT_VARIABLE _result ERROR_VARIABLE _error)
          if(NOT _result EQUAL 0)
            message(FATAL_ERROR "SDPA reference validation failed for ${_arch}: ${_error}")
          endif()
          install(DIRECTORY "${_bundle}/"
                  DESTINATION "${ROCKE_TEST_INSTALL_DIR}/tests/library/tests/reference_bundles/${_arch}"
                  PATTERN "__pycache__" EXCLUDE REGEX "\\.pyc$" EXCLUDE)
          set(_ROCKE_HAS_SDPA_REFERENCE ON)
        endif()
      endforeach()

      # Install the library test tree for the library pytest entries below.
      # cmake-lint: disable=E1122
      install(DIRECTORY "${_ROCKE_LIBRARY_DIR}/tests"
              DESTINATION "${ROCKE_TEST_INSTALL_DIR}/tests/library"
              PATTERN "__pycache__" EXCLUDE
              REGEX "\\.pyc$" EXCLUDE
              PATTERN "*.c" EXCLUDE
              PATTERN "reference_bundles" EXCLUDE
              PATTERN "run_all.py" EXCLUDE)
    else()
      message(WARNING
              "rocKE: library tree not found at ${_ROCKE_LIBRARY_DIR}; the installed "
              "golden gate will fail on its library-kernel cases")
    endif()

    # Keep the LDS skill CLI under tests/ so TheRock's test artifact includes it.
    install(PROGRAMS
            "${CMAKE_CURRENT_SOURCE_DIR}/dsl_docs/optimization/utilities/skills/lds-bank-conflict-expert/scripts/predict.py"
            DESTINATION "${ROCKE_TEST_INSTALL_DIR}/tests/analysis/lds-bank-conflict-expert")

    # Install the full pytest suite for installed/artifact test execution.
    # Excludes: C/C++ parity emitters (compiled against rocke_core at build time),
    # the differential harness (needs compiler + source), the WaveScope sidecar
    # tests (load tools from the adjacent dsl_docs source tree).
    # Include run_all.py: installed runner unit tests import it as a dependency.
    # cmake-lint: disable=E1122
    block()
      install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tests/"
              DESTINATION "${ROCKE_TEST_INSTALL_DIR}/tests"
              PATTERN "__pycache__" EXCLUDE
              PATTERN "*.pyc" EXCLUDE
              PATTERN "*.c" EXCLUDE
              PATTERN "*.cpp" EXCLUDE
              PATTERN "differential" EXCLUDE
              PATTERN "test_wavescope_sidecar.py" EXCLUDE)

    endblock()

    # This GPU test lives alongside its benchmark helpers in the Python package.
    # Stage it under tests/ so rocke_pytest collects it in both install layouts.
    install(FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/python/rocke/benchmark/gemm/tests/test_fp16_rcr_multigpu.py"
            DESTINATION "${ROCKE_TEST_INSTALL_DIR}/tests/benchmark")
  endif()

  set(_ROCKE_INSTALLED_CTEST_ENTRIES "")
  get_property(_ROCKE_INSTALL_TEST_TARGETS GLOBAL PROPERTY ROCKE_INSTALL_TEST_TARGETS)
  foreach(_test_target ${_ROCKE_INSTALL_TEST_TARGETS})
    rocke_cxx_test_installed_name(_rocke_installed_name ${_test_target})
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "add_test(${_test_target} \"${ROCKE_CXX_TEST_CTEST_REL}${_rocke_installed_name}\")\n")
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "set_tests_properties(${_test_target} PROPERTIES LABELS \"ckc;unit;host;quick\")\n")
  endforeach()

  if(ROCKE_INSTALL_PYTHON_TESTS)
    # Python test entries invoke via "python" (resolved from PATH at test time)
    # rather than executing the .py directly, so they work on Windows where
    # shebangs are not supported. The CI test environment activates a venv with
    # requirements-test.txt before running ctest.
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "add_test(rocke_python_import_smoke python \"${ROCKE_TEST_CTEST_REL}rocke_installed_smoke.py\")\n")
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "set_tests_properties(rocke_python_import_smoke PROPERTIES LABELS \"ckc;python;quick\")\n")

    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "add_test(rocke_golden_static python \"${ROCKE_TEST_CTEST_REL}rocke_installed_golden_test.py\")\n")
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "set_tests_properties(rocke_golden_static PROPERTIES LABELS \"ckc;python;golden;host;quick\")\n")

    if(ROCKE_INSTALL_PLATFORM_PYTEST)
      # Full pytest suite. PYTHONPATH points at the CTestTestfile's working
      # directory (which contains the installed rocke package) so `import rocke`
      # resolves without the source-tree sys.path shims. The library tests have
      # their own CTest entry below and must not be collected twice.
      string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
             "add_test(rocke_pytest python -m pytest \"${ROCKE_TEST_CTEST_REL}tests\" --ignore \"${ROCKE_TEST_CTEST_REL}tests/library\" -v -rs --timeout=60)\n")
      string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
             "set_tests_properties(rocke_pytest PROPERTIES\n"
             "  LABELS \"ckc\\;python\\;quick\"\n"
             "  ENVIRONMENT \"PYTHONPATH=.\"\n"
             "  TIMEOUT 300)\n")
    endif()

    # Library conv CPU tests (no GPU required; dispatch selection + signature guards).
    # The test environment installs the platform wheel. PYTHONPATH puts the staged
    # library tree first so kernels.*, builders.*, dispatch.*, and benchmarks.* do
    # not collide with rocke's platform subpackages. Both the test paths and that
    # tree are relative to the CTestTestfile's own directory, which sits one level
    # below the install root in standalone mode -- hence the ROCKE_TEST_CTEST_REL
    # prefix on each, matching the other installed tests.
    set(_ROCKE_LIBRARY_CONV_TEST_RELPATHS
        "tests/library/tests/dispatch/test_grouped_conv_wgrad_dispatch.py"
        "tests/library/tests/test_conv_build_signatures.py"
        "tests/library/tests/test_builder_signature_contract.py"
        "tests/library/tests/test_library_layering.py")
    set(_ROCKE_LIBRARY_CONV_TESTS "")
    foreach(_rocke_conv_test ${_ROCKE_LIBRARY_CONV_TEST_RELPATHS})
      string(APPEND _ROCKE_LIBRARY_CONV_TESTS
             " \"${ROCKE_TEST_CTEST_REL}${_rocke_conv_test}\"")
    endforeach()
    # pytest-timeout applies per test item and is intentionally well below the
    # CTest limit so a hung import or structural check produces pytest diagnostics.
    # CTest remains the hard backstop for collection, cumulative runtime, and
    # failures outside an individual test item.
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "add_test(rocke_library_conv_pytest python -m pytest${_ROCKE_LIBRARY_CONV_TESTS} -v -rs --timeout=30)\n")
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "set_tests_properties(rocke_library_conv_pytest PROPERTIES\n"
           "  LABELS \"ckc\\;python\\;conv\\;host\\;quick\"\n"
           "  ENVIRONMENT \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library\"\n"
            "  TIMEOUT 180)\n")

    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "add_test(rocke_sdpa_reference_unit_pytest python -m pytest \"${ROCKE_TEST_CTEST_REL}tests/library/tests/test_sdpa_reference_contract.py\" \"${ROCKE_TEST_CTEST_REL}tests/library/tests/test_sdpa_reference_artifact.py\" -v -rs --timeout=30)\n")
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "set_tests_properties(rocke_sdpa_reference_unit_pytest PROPERTIES\n"
           "  LABELS \"ckc\\;python\\;sdpa\\;host\\;quick\"\n"
           "  ENVIRONMENT \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library\"\n"
           "  TIMEOUT 180)\n")

    # Enroll the SDPA lane only with a qualified bundle. Its files live beneath
    # tests/ so the provider's existing artifact manifest captures them. Once
    # enrolled, missing gfx942 hardware or a failed qualification check is an
    # error; other GPU architectures remain explicitly outside this pilot.
    if(_ROCKE_HAS_SDPA_REFERENCE)
      string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
             "add_test(rocke_sdpa_gpu_pytest python -m pytest \"${ROCKE_TEST_CTEST_REL}tests/library/tests/test_sdpa_pinned_reference.py\" -v -rs)\n")
      string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
             "set_tests_properties(rocke_sdpa_gpu_pytest PROPERTIES\n"
             "  LABELS \"ckc\\;python\\;sdpa\\;gpu\\;quick\"\n"
             "  ENVIRONMENT \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library;ROCKE_REQUIRE_SDPA_GPU=1\"\n"
             "  TIMEOUT 1800)\n")
    endif()

    # Library GPU numeric suites: convolution, attention, KDA, and GDN.
    # Select files rather than the "gpu" pytest marker because several runtime
    # tests lack that marker. Their dependency and architecture checks report
    # skips when the required torch installation or GPU is unavailable.
    set(_ROCKE_LIBRARY_GPU_TEST_RELPATHS
        "tests/library/tests/test_conv_fwd_correctness.py"
        "tests/library/tests/test_conv_wgrad_correctness.py"
        "tests/library/tests/test_conv_dgrad_correctness.py"
        "tests/library/tests/test_direct_conv_correctness.py"
        "tests/library/tests/test_attention_dense_gfx942_numeric.py"
        "tests/library/tests/test_attention_dense_gfx950_numeric.py"
        "tests/library/tests/test_attn_bf16_d128_ring.py"
        "tests/library/tests/test_dtype_dispatch.py"
        "tests/library/tests/test_extended_parity_attention.py"
        "tests/library/tests/test_gdn_decode_gfx950_numeric.py"
        "tests/library/tests/test_gfx1250_attention.py"
        "tests/library/tests/test_gfx942_gqa_head_fold_numeric.py"
        "tests/library/tests/test_kda_chunkwise_gfx942_numeric.py"
        "tests/library/tests/test_kda_chunkwise_gfx950_numeric.py"
        "tests/library/tests/test_kda_gdn_gfx950_numeric.py")
    set(_ROCKE_LIBRARY_GPU_TESTS "")
    foreach(_rocke_gpu_test ${_ROCKE_LIBRARY_GPU_TEST_RELPATHS})
      string(APPEND _ROCKE_LIBRARY_GPU_TESTS
             " \"${ROCKE_TEST_CTEST_REL}${_rocke_gpu_test}\"")
    endforeach()
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "add_test(rocke_library_gpu_pytest python -m pytest${_ROCKE_LIBRARY_GPU_TESTS} -v -rs)\n")
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "set_tests_properties(rocke_library_gpu_pytest PROPERTIES\n"
           "  LABELS \"ckc\\;python\\;gpu\"\n"
           "  ENVIRONMENT \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library\"\n"
           "  TIMEOUT 1800)\n")
  endif()

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
