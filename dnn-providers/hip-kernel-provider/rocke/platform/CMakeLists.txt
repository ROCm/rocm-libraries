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
# Required: the engine's copy of the intrinsic arch-domain table is generated
# at build time from the committed JSON columns (see below).
find_package(Python3 COMPONENTS Interpreter REQUIRED)

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
if(NOT DEFINED ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR)
  set(ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR "" CACHE STRING
      "Provider installed CTest subdir to append rocKE tests to, e.g. hip_kernel_provider")
endif()

set(_ROCKE_GPU_REFERENCES_DEFAULT OFF)
# Provider artifacts include all published references. The rocm-libraries
# superbuild does not fetch these test bundles.
if(ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR AND NOT ROCM_LIBS_SUPERBUILD)
  set(_ROCKE_GPU_REFERENCES_DEFAULT ON)
endif()
option(ROCKE_INSTALL_TEST_GPU_REFERENCES
       "Install enrolled GPU test references and register their GPU tests"
       ${_ROCKE_GPU_REFERENCES_DEFAULT})

# Where rocKE's installed CTest metadata and standalone Python smoke script land:
#  * Provider mode: append CTest entries to the provider's installed
#    bin/<subdir>/CTestTestfile.cmake. Provider-mode C++ smoke binaries use the
#    bare-bin naming scheme below so TheRock's existing test artifact patterns
#    capture them.
#  * Standalone mode: keep artifacts in bin/ with the CTestTestfile under
#    bin/rocke/, reached via "../<name>".
if(NOT DEFINED ROCKE_TEST_INSTALL_DIR)
  if(ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR)
    set(ROCKE_TEST_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}/${ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR}")
  else()
    set(ROCKE_TEST_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}")
  endif()
endif()
if(ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR)
  set(ROCKE_TEST_CTEST_REL "./")
else()
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

# Return the installed-CTestTestfile text that registers rocke_pytest_<part>,
# one part of the platform pytest suite. The remaining arguments are the part's
# pytest path and option arguments. _ROCKE_PLATFORM_PYTEST_IGNORE_<part>, when
# set, lists files under tests/ that the part must not collect (for example a
# file that runs in its own CTest entry); each becomes an --ignore. Every
# argument is quoted. Every part shares the marker filter, the per-test limit,
# the labels, the CTest limit and the environment list that the installed
# CTestTestfile defines as _rocke_platform_pytest_environment.
function(rocke_platform_pytest_entry out_var part)
  set(_test_name "rocke_pytest_${part}")
  set(_pytest_argument_list ${ARGN})
  foreach(_ignored_file IN LISTS _ROCKE_PLATFORM_PYTEST_IGNORE_${part})
    list(APPEND _pytest_argument_list --ignore "${ROCKE_TEST_CTEST_REL}tests/${_ignored_file}")
  endforeach()
  set(_pytest_arguments "")
  foreach(_pytest_argument IN LISTS _pytest_argument_list)
    string(APPEND _pytest_arguments " \"${_pytest_argument}\"")
  endforeach()
  string(CONCAT _entry
         "add_test(${_test_name} python -m pytest${_pytest_arguments} -m \"not integration\" -v -rs --timeout=60)\n"
         "set_tests_properties(${_test_name} PROPERTIES\n"
         "  LABELS \"ckc\\;python\\;quick\"\n"
         "  ENVIRONMENT \"\${_rocke_platform_pytest_environment}\"\n"
         "  TIMEOUT 300)\n")
  set(${out_var} "${_entry}" PARENT_SCOPE)
endfunction()

set(_ROCKE_DEFAULT_PYTHON_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")
set(ROCKE_PYTHON_INSTALL_DIR "${_ROCKE_DEFAULT_PYTHON_INSTALL_DIR}"
    CACHE STRING "Relative install destination for rocke and rocke_engine")

# Sources live under cpp/, in subfolders mirroring the rocke Python package
# (cpp/core/{ir,arch,isa,lower_llvm,lower_hip}, cpp/helpers, cpp/instances/{common,gfxNNNN},
# cpp/support). GLOB_RECURSE picks up every .cpp regardless of depth; the
# pybind bindings under cpp/bindings are a separate target and are excluded.
file(GLOB_RECURSE ROCKE_CORE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.cpp")
list(FILTER ROCKE_CORE_SOURCES EXCLUDE REGEX "/cpp/bindings/")

# Compile the engine once, then reuse those objects in the static archive and
# optional shared replay engine. Direct objects also let CMake discover and
# export the flat C ABI when producing a Windows DLL.
add_library(rocke_core_objects OBJECT ${ROCKE_CORE_SOURCES})
add_library(rocke_core STATIC $<TARGET_OBJECTS:rocke_core_objects>)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(rocke_core_objects PRIVATE Threads::Threads)
target_link_libraries(rocke_core PUBLIC ${CMAKE_DL_LIBS} Threads::Threads)
target_include_directories(rocke_core_objects PRIVATE
                           "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include")
target_include_directories(rocke_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# --- Intrinsic arch-domain table ---------------------------------------------
# Which intrinsic declarations each gfx target can lower is measured by
# tools/gen_arch_domain.py and committed as one JSON column per LLVM flavor,
# beside the reviewed waivers in arch_domain_waivers.json. Those files are the
# one source of truth. The Python engine reads them when it lowers; this engine
# gets a copy generated here, through the same Python reader, so there is no
# hand-maintained mirror to drift. Regenerating a column or editing a waiver
# needs no other step: the next build rewrites this file. It lives in the
# build tree and is never committed.
set(_rocke_arch_domain_inc
    "${CMAKE_CURRENT_BINARY_DIR}/generated/rocke/arch_domain_table.inc")
file(GLOB _rocke_arch_domain_columns CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/python/rocke/core/arch/data/intrinsic_arch_domain.*.json")
add_custom_command(
  OUTPUT "${_rocke_arch_domain_inc}"
  COMMAND "${Python3_EXECUTABLE}" -I
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_arch_domain_cpp.py"
          --out "${_rocke_arch_domain_inc}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_arch_domain_cpp.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/rocke/core/arch/domain.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/rocke/core/arch/data/arch_domain_waivers.json"
          ${_rocke_arch_domain_columns}
  COMMENT "Generating the C++ arch-domain table from the committed columns"
  VERBATIM)
# The objects are compiled once in rocke_core_objects and shared by the static
# and shared libraries, so the table belongs to that target.
target_sources(rocke_core_objects PRIVATE "${_rocke_arch_domain_inc}")
target_include_directories(rocke_core_objects PRIVATE
                           "${CMAKE_CURRENT_BINARY_DIR}/generated")

# --- Portable-IR bundle freshness / provenance stamp -------------------------
# Generate a deterministic, git-independent provenance header from the engine
# and replay CLI sources plus their CMake build recipes. This common bundle ID is
# a build dependency, not merely a configure-time value: an incremental edit
# regenerates the stamp before rocke_build_id.o is rebuilt. No lowering/emission
# TU includes it, so LLVM-IR stays byte-identical.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/rocke_build_id.cmake")
rocke_compute_build_id(ROCKE_CONFIGURE_BUILD_ID "${CMAKE_CURRENT_SOURCE_DIR}")
# Human-readable version. Bump the leading semver when the IR contract changes;
# the date keeps successive builds from the same version distinguishable.
string(TIMESTAMP ROCKE_BUILD_DATE "%Y%m%d" UTC)
set(ROCKE_ENGINE_VERSION "1.0.0+${ROCKE_BUILD_DATE}")
message(STATUS
        "portable-IR build-id: ${ROCKE_CONFIGURE_BUILD_ID}  version: ${ROCKE_ENGINE_VERSION}")
set(ROCKE_BUILD_ID_HEADER
    "${CMAKE_CURRENT_BINARY_DIR}/generated/rocke_build_id_values.h")
set(ROCKE_BUILD_ID_MANIFEST
    "${CMAKE_CURRENT_BINARY_DIR}/generated/rocke_portable_ir_provenance.txt")
# Run on every build so a newly added private header is covered even before a
# reconfigure. copy_if_different keeps rocke_build_id.o untouched when the hash
# is unchanged.
add_custom_target(
  rocke_build_id_header
  COMMAND "${CMAKE_COMMAND}"
          "-DROCKE_SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}"
          "-DROCKE_BUILD_ID_HEADER=${ROCKE_BUILD_ID_HEADER}"
          "-DROCKE_BUILD_ID_MANIFEST=${ROCKE_BUILD_ID_MANIFEST}"
          "-DROCKE_ENGINE_VERSION=${ROCKE_ENGINE_VERSION}"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_rocke_build_id.cmake"
  BYPRODUCTS "${ROCKE_BUILD_ID_HEADER}" "${ROCKE_BUILD_ID_MANIFEST}"
  COMMENT "Refreshing rocKE portable-IR bundle build-id"
  VERBATIM)
add_dependencies(rocke_core_objects rocke_build_id_header)
target_include_directories(rocke_core_objects PRIVATE
                           "${CMAKE_CURRENT_BINARY_DIR}/generated")
set_source_files_properties(
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp/core/rocke_build_id.cpp"
  PROPERTIES OBJECT_DEPENDS "${ROCKE_BUILD_ID_HEADER}")
# These objects feed both static and shared libraries. PIC is required on ELF
# hosts; enabling it uniformly keeps one object set portable across consumers.
set_target_properties(rocke_core_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- Shared engine for the portable-IR replay path ---------------------------
# The Python-side portable-IR tooling (rocke.portable_ir.src.online) reaches the
# C++ importer, recipe VM and lowerer through ctypes, which needs a .so rather
# than the archive. online.build_lib() can produce one on demand, but a CI gate
# wants a named target it can build and point ROCKE_ONLINE_LIB at, with the link
# flags living here rather than duplicated in workflow YAML. Supplying the
# object library directly retains every C ABI entry point without linker-specific
# whole-archive flags. On Windows, it also gives CMake the .obj files needed to
# generate the DLL export definition.
option(ROCKE_BUILD_SHARED_ENGINE
       "Also link a shared librocke.so (needed by the portable-IR ctypes path)" OFF)
if(ROCKE_BUILD_SHARED_ENGINE)
  add_library(rocke_shared SHARED $<TARGET_OBJECTS:rocke_core_objects>)
  set_target_properties(rocke_shared PROPERTIES OUTPUT_NAME rocke)
  target_link_libraries(rocke_shared PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
  if(MSVC)
    set_target_properties(rocke_shared PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
  elseif(UNIX)
    target_link_libraries(rocke_shared PRIVATE m)
  endif()
  install(TARGETS rocke_shared
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

include(CTest)

# C++ test targets are orchestrated from tests/CMakeLists.txt and registered in
# concern-owned tests/cmake/*.cmake fragments.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  add_subdirectory(tests)
endif()

if(ROCKE_BUILD_PYBIND)
  add_subdirectory(cpp/bindings)
endif()

# --- Library-owned C++ mirrors ------------------------------------------------
# A kernel family whose Python lives in ../library keeps its C++ byte-identity
# mirror there too, in library/cpp, so one team owns both halves of the
# contract. It builds its own archive (rocke_library_core) and pybind module
# (rocke_library_engine) that link rocke_core; nothing here links back, so
# rocke_core and rocke_engine never carry library code. The installed ckc
# package exports the archive beside rocke_core as ckc::rocke_library, so its C
# API stays installed. Added from this directory so it inherits the engine's
# warning, sanitizer and clang-tidy settings above. Guarded on the sibling tree
# so a platform-only copy of this directory still configures; keep the guard, it
# is the copy-able contract.
option(ROCKE_BUILD_LIBRARY_CPP
       "Build the library-owned C++ engine mirrors in ../library/cpp" ON)
set(_ROCKE_LIBRARY_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../library/cpp")
if(ROCKE_BUILD_LIBRARY_CPP AND EXISTS "${_ROCKE_LIBRARY_CPP_DIR}/CMakeLists.txt")
  add_subdirectory("${_ROCKE_LIBRARY_CPP_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/library_cpp")
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
include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ckcConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/ckcConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ckc)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ckcConfig.cmake"
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
    # and the golden's per-family JSON shards co-located next to it; no GPU and no
    # examples required.
    install(PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/tests/rocke_installed_golden_test.py"
            DESTINATION ${ROCKE_TEST_INSTALL_DIR})
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/tests/instances/rocke_ir_parity_harness.py"
            DESTINATION ${ROCKE_TEST_INSTALL_DIR})
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tests/golden/rocke_representative_ir_sha256"
            DESTINATION ${ROCKE_TEST_INSTALL_DIR}/golden
            FILES_MATCHING PATTERN "*.json")

    # Host-only differential golden check (Python reference emitters only; no
    # compiler). The rest of tests/instances/differential needs the C++ build and
    # stays out of the installed tree (see the tests/ install below); these files
    # are the Python-only check: the runner, its batch worker, the golden store
    # and file, and the CPU-count helper the runner imports from tools/.
    set(_ROCKE_DIFFERENTIAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/instances/differential")
    install(FILES "${_ROCKE_DIFFERENTIAL_DIR}/run_diff.py"
                  "${_ROCKE_DIFFERENTIAL_DIR}/python_batch_worker.py"
                  "${_ROCKE_DIFFERENTIAL_DIR}/golden_store.py"
                  "${CMAKE_CURRENT_SOURCE_DIR}/tools/_hostcaps.py"
            DESTINATION ${ROCKE_TEST_INSTALL_DIR}/tests/instances/differential)
    install(FILES "${_ROCKE_DIFFERENTIAL_DIR}/golden/llvm_gfx_all.json"
            DESTINATION ${ROCKE_TEST_INSTALL_DIR}/tests/instances/differential/golden)

    # The harness above builds LIBRARY kernels for its attention, attention_dense,
    # attention_d256, kda_chunkwise and lightning_indexer families (the one platform -> library
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
      # Only published bundles participate in installation and CTest enrollment.
      include(cmake/InstallGpuReferences.cmake)

      # Install the library test tree for the library pytest entries below.
      # cmake-lint: disable=E1122
      install(DIRECTORY "${_ROCKE_LIBRARY_DIR}/tests"
              DESTINATION "${ROCKE_TEST_INSTALL_DIR}/tests/library"
              PATTERN "__pycache__" EXCLUDE
              REGEX "\\.pyc$" EXCLUDE
              PATTERN "*.c" EXCLUDE
              PATTERN "reference_bundles" EXCLUDE
              PATTERN "check_torch_reference.py" EXCLUDE
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
    # the differential harness (needs compiler + source; its Python-only golden
    # check is installed separately above), the WaveScope sidecar,
    # LDS analyzer and dsl_probes tests (load tools from the adjacent dsl_doc
    # source tree), the arch-table parity test (reads cpp/core/arch/data.cpp),
    # the MMA catalog cross-check unit tests (load tools/ from the source tree),
    # and the arch-domain artifact unit tests. The latter remain
    # source-only units, while the full regeneration sweep is installed below as
    # its own CTest with an independent timeout and resource policy.
    # Include run_all.py: installed runner unit tests import it as a dependency.
    # Linux detector fixtures link a stdin wrapper against the production archive.
    # Keep it with the pytest assets so TheRock's test artifact captures it even
    # though the library artifact excludes lib/*.a.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      install(FILES "$<TARGET_FILE:rocke_core>"
              DESTINATION "${ROCKE_TEST_INSTALL_DIR}/tests")
    endif()
    # cmake-lint: disable=E1122
    block()
      install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tests/"
              DESTINATION "${ROCKE_TEST_INSTALL_DIR}/tests"
              PATTERN "__pycache__" EXCLUDE
              PATTERN "*.pyc" EXCLUDE
              PATTERN "*.c" EXCLUDE
              PATTERN "*.cpp" EXCLUDE
              PATTERN "differential" EXCLUDE
              PATTERN "test_engine_linkage.py" EXCLUDE
              PATTERN "test_wavescope_sidecar.py" EXCLUDE
              PATTERN "test_lds_analyzer.py" EXCLUDE
              PATTERN "test_dsl_probes.py" EXCLUDE
              PATTERN "test_arch_domain_artifact.py" EXCLUDE
              PATTERN "test_arch_table_engine_parity.py" EXCLUDE
              PATTERN "test_gemm_spec_dict_binding_parity.py" EXCLUDE
              PATTERN "test_check_mma_catalog.py" EXCLUDE
              PATTERN "test_gpu_references.py" EXCLUDE)

    endblock()

    # The top-level developer runner is not part of the standalone platform
    # component, but its command-plan regression tests run in the relocated
    # full-repository pytest suite. Stage a test-only copy when the sibling
    # exists; a verbatim platform-only copy must still configure and install.
    set(_ROCKE_RUN_CHECKS "${CMAKE_CURRENT_SOURCE_DIR}/../tools/run_checks.py")
    if(EXISTS "${_ROCKE_RUN_CHECKS}")
      install(FILES "${_ROCKE_RUN_CHECKS}"
              DESTINATION "${ROCKE_TEST_INSTALL_DIR}/tests/tools")
    else()
      message(STATUS
              "rocKE: top-level run_checks.py not found; skipping its test-only install")
    endif()

    # Test-only implementation for the installed arch-domain reproduction gate.
    # Keep it outside the production package: CTest owns the external compiler
    # sweep, while package resources provide the committed columns being checked.
    # tests/conftest.py also points rocke.tests.ir_validity at this copy, so the
    # installed pytest runs compile lowered IR with the same tool.
    install(FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_arch_domain.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_ir_validity.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/_corpus.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/_bounded_run.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/_flavor_gate.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/_hostcaps.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/_llvm_identity.py"
            DESTINATION "${ROCKE_TEST_INSTALL_DIR}/tests/arch_domain")

    # Test-only copy of the compiler-option measurement tool for the installed
    # reproduction gate below; the table it checks is package data.
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_compiler_options.py"
            DESTINATION "${ROCKE_TEST_INSTALL_DIR}/tests/compiler_options")

    # This GPU test lives alongside its benchmark helpers in the Python package.
    # Stage it under tests/ so the rocke_pytest_* parts collect it in both install
    # layouts.
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
    # Provider artifacts co-locate the package with their CTest manifest. A
    # standalone install keeps it in the configured site-packages directory;
    # derive that relative path so the installed tree remains relocatable.
    if(ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR)
      set(_ROCKE_INSTALLED_PYTHONPATH ".")
    else()
      file(RELATIVE_PATH _ROCKE_INSTALLED_PYTHONPATH
           "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/rocke"
           "${CMAKE_INSTALL_PREFIX}/${ROCKE_PYTHON_INSTALL_DIR}")
    endif()

    # Python test entries invoke via "python" (resolved from PATH at test time)
    # rather than executing the .py directly, so they work on Windows where
    # shebangs are not supported. The CI test environment activates a venv with
    # requirements-test.txt before running ctest.
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "add_test(rocke_python_import_smoke python \"${ROCKE_TEST_CTEST_REL}rocke_installed_smoke.py\")\n")
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "set_tests_properties(rocke_python_import_smoke PROPERTIES LABELS \"ckc;python;quick\")\n")

    set(_rocke_barrier_parity_entries "")
    if(TARGET rocke_optimization_barrier_emit)
      rocke_cxx_test_installed_name(_rocke_installed_name
                                  rocke_optimization_barrier_emit)
      string(APPEND _rocke_barrier_parity_entries
             "add_test(rocke_optimization_barrier_hip_pair python "
             "\"${ROCKE_TEST_CTEST_REL}tests/core/"
             "test_optimization_barrier_cli.py\" --native "
             "\"${ROCKE_CXX_TEST_CTEST_REL}${_rocke_installed_name}${CMAKE_EXECUTABLE_SUFFIX}\" "
             "--launcher \"${_ROCKE_INSTALLED_PYTHONPATH}/rocke/tests/"
             "subprocess_launcher.py\" --emitter "
             "\"${ROCKE_TEST_CTEST_REL}tests/instances/parity/"
             "optimization_barrier_emit.py\" --count 17)\n"
             "set_tests_properties(rocke_optimization_barrier_hip_pair "
             "PROPERTIES LABELS \"ckc;python;host;quick\" TIMEOUT 120)\n")
    endif()
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "add_test(rocke_golden_static python \"${ROCKE_TEST_CTEST_REL}rocke_installed_golden_test.py\")\n")
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "set_tests_properties(rocke_golden_static PROPERTIES LABELS \"ckc;python;golden;host;quick\")\n"
           "${_rocke_barrier_parity_entries}")

    # Compiler tool locations for Python entries that compile and disassemble
    # kernels: hipcc in ROCm's bin/, llvm-objdump and llvm-readelf in its
    # lib/llvm/bin/ (neither is on the CI image's PATH), and the device-library
    # bitcode. CTest resolves these lists at test time; they stay empty off Linux
    # or without ROCM_PATH. Tool discovery is separate from enforcement policy:
    # only entries whose compiler checks must run append the requirement list.
    # Contract: tests/test_engine_linkage.py::
    # test_installed_ctest_entries_get_scoped_environment_contracts.
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "set(_rocke_compiler_tool_environment)\n"
           "set(_rocke_compiler_requirement_environment)\n"
           "if(CMAKE_HOST_SYSTEM_NAME STREQUAL \"Linux\" AND NOT \"\$ENV{ROCM_PATH}\" STREQUAL \"\")\n"
           "  list(APPEND _rocke_compiler_tool_environment\n"
           "       \"PATH=\$ENV{ROCM_PATH}/bin:\$ENV{ROCM_PATH}/lib/llvm/bin:\$ENV{PATH}\"\n"
           "       \"HIP_DEVICE_LIB_PATH=\$ENV{ROCM_PATH}/lib/llvm/amdgcn/bitcode\")\n"
           "  list(APPEND _rocke_compiler_requirement_environment\n"
           "       \"ROCKE_REQUIRE_COMPILER_TOOLS=1\")\n"
           "endif()\n")

    # Reproduce the committed host-flavor column from the relocated artifact.
    # This mirrors the build-tree CTest in tests/cmake/CoreTests.cmake, but deliberately
    # uses the installed package data that downstream/provider CI consumes.
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "add_test(rocke_arch_domain_reproducibility python -u \"${ROCKE_TEST_CTEST_REL}tests/arch_domain/gen_arch_domain.py\" --check --unvalidated-exit-code 77)\n")
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "set_tests_properties(rocke_arch_domain_reproducibility PROPERTIES\n"
           "  LABELS \"ckc;python;host;integration;toolchain;generated-artifact;standard\"\n"
           "  ENVIRONMENT \"PYTHONPATH=${_ROCKE_INSTALLED_PYTHONPATH}\"\n"
           "  RUN_SERIAL TRUE\n"
           "  SKIP_RETURN_CODE 77\n"
           "  TIMEOUT 1800)\n")

    # Re-measure which -mllvm options the loaded compiler accepts and fail when
    # its committed column in compiler_options.json differs or is missing. The
    # build-tree twin is in tests/cmake/CoreTests.cmake. It fails rather than
    # skips without a queryable COMGR: every lane that runs it compiles. Like
    # the arch-domain entry above it only loads COMGR, so it needs neither the
    # compiler tools nor the compiler-check requirement.
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "add_test(rocke_compiler_options_reproducibility python -u \"${ROCKE_TEST_CTEST_REL}tests/compiler_options/gen_compiler_options.py\" --check)\n")
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "set_tests_properties(rocke_compiler_options_reproducibility PROPERTIES\n"
           "  LABELS \"ckc;python;host;toolchain;generated-artifact;standard\"\n"
           "  ENVIRONMENT \"PYTHONPATH=${_ROCKE_INSTALLED_PYTHONPATH}\"\n"
           "  TIMEOUT 600)\n")

    # The differential golden: every parity family's Python reference output,
    # LLVM IR at each LLVM flavor (the flavor is a lowering argument, so one host
    # checks all of them) and canonical IR. tests/instances/differential/golden/README.md.
    set(_ROCKE_RUN_DIFF "${ROCKE_TEST_CTEST_REL}tests/instances/differential/run_diff.py")
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "add_test(rocke_differential_golden_ll python \"${_ROCKE_RUN_DIFF}\" --mode ll --python-only --check-golden --flavor llvm20,llvm22,llvm23,llvm24)\n")
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "set_tests_properties(rocke_differential_golden_ll PROPERTIES LABELS \"ckc;python;golden;host;quick\" ENVIRONMENT \"PYTHONPATH=${_ROCKE_INSTALLED_PYTHONPATH}\" TIMEOUT 300)\n")
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "add_test(rocke_differential_golden_canonical python \"${_ROCKE_RUN_DIFF}\" --mode ir --canonical --python-only --check-golden)\n")
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "set_tests_properties(rocke_differential_golden_canonical PROPERTIES LABELS \"ckc;python;golden;host;quick\" ENVIRONMENT \"PYTHONPATH=${_ROCKE_INSTALLED_PYTHONPATH}\" TIMEOUT 300)\n")

    if(ROCKE_INSTALL_PLATFORM_PYTEST)
      # Full pytest suite. PYTHONPATH points at the CTestTestfile's working
      # directory (which contains the installed rocke package) so `import rocke`
      # resolves without the source-tree sys.path shims. The library tests have
      # their own CTest entry below and must not be collected twice.
      #
      # Some pytest files compare the Python engine with a native C++ test
      # binary named by an environment variable, and skip without it. Point
      # each variable at its installed binary, as tests/run_all.py does for a
      # source build (its NATIVE_PARITY_TESTS), so those cases run here too.
      # Paths are relative to the CTestTestfile. ROCKE_REPLAY_CLI is the
      # portable-IR replay CLI and ROCKE_PROVENANCE_MANIFEST its provenance
      # manifest; ROCKE_UNSIGNED_INT_OPS_CLI is the C API program whose
      # --print-llvm output test_unsigned_int_ops.py compares with the Python
      # builder's text; ROCKE_INTEXPR_CLI is the C++ integer-expression
      # evaluator whose --eval output test_intexpr.py compares with Python's;
      # ROCKE_PARAM_FACTS_CLI prints the C++ print_ir, verifier messages and
      # canonicalize that test_param_facts.py compares with Python's;
      # ROCKE_GPU_CHECKED_VALUE_CLI prints the C++ builder's GPU-check cases
      # and marker verifier messages that test_gpu_checked_value.py compares.
      set(_ROCKE_PYTEST_ENVIRONMENT
          "PYTHONPATH=."
          "ROCKE_REPLAY_CLI=${ROCKE_TEST_CTEST_REL}tests/portable_ir/rocke_portable_ir_replay_cli${CMAKE_EXECUTABLE_SUFFIX}"
          "ROCKE_PROVENANCE_MANIFEST=${ROCKE_TEST_CTEST_REL}tests/portable_ir/rocke_portable_ir_provenance.txt")
      foreach(_rocke_native_parity
              "ROCKE_STORAGE_TEST=rocke_storage"
              "ROCKE_IR_TYPES_TEST=rocke_ir_types"
              "ROCKE_NONTEMPORAL_HIP_TEST=rocke_nontemporal_hip"
              "ROCKE_UNSIGNED_INT_OPS_CLI=rocke_unsigned_int_ops"
              "ROCKE_INTEXPR_CLI=rocke_intexpr"
              "ROCKE_PARAM_FACTS_CLI=rocke_param_facts"
              "ROCKE_GPU_CHECKED_VALUE_CLI=rocke_gpu_checked_value"
              "ROCKE_ARCH_TARGET_TEST_EXE=rocke_arch_target_identity"
              "ROCKE_BUFFER_RSRC_HIP_TEST=rocke_buffer_rsrc_hip")
        string(REPLACE "=" ";" _rocke_native_parity "${_rocke_native_parity}")
        list(GET _rocke_native_parity 0 _rocke_parity_var)
        list(GET _rocke_native_parity 1 _rocke_parity_target)
        if(_rocke_parity_target IN_LIST _ROCKE_INSTALL_TEST_TARGETS)
          rocke_cxx_test_installed_name(_rocke_installed_name ${_rocke_parity_target})
          list(APPEND _ROCKE_PYTEST_ENVIRONMENT
               "${_rocke_parity_var}=${ROCKE_CXX_TEST_CTEST_REL}${_rocke_installed_name}${CMAKE_EXECUTABLE_SUFFIX}")
        endif()
      endforeach()
      string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
             "set(_rocke_platform_pytest_environment \"${_ROCKE_PYTEST_ENVIRONMENT}\")\n"
             "list(APPEND _rocke_platform_pytest_environment \${_rocke_compiler_tool_environment})\n")

      # The suite runs as several entries, each a group of directories under
      # tests/: as one entry it reached its 300 s limit on the slowest CI
      # runners with every test passing. The groups are sized from
      # per-directory times in CI logs; _ROCKE_PLATFORM_PYTEST_GROUP_<group>
      # lists the directories of rocke_pytest_<group>. rocke_pytest_other runs
      # the rest of tests/: the files at its top level and every directory no
      # group names, so a new directory runs with no edit here. Together the
      # entries collect what one entry over tests/ would. To keep a file out of
      # every part, list(APPEND) it (a path under tests/) to
      # _ROCKE_PLATFORM_PYTEST_IGNORE_<part> for the part whose directories hold
      # it, or for "other"; APPEND keeps any value given on the configure line,
      # which tests/test_engine_linkage.py uses. The rocke_pytest_ name prefix
      # is reserved for these entries; the provider categories tier them by
      # that prefix.
      set(_ROCKE_PLATFORM_PYTEST_GROUPS core instances_dispatch runtime_tools)
      set(_ROCKE_PLATFORM_PYTEST_GROUP_core core)
      set(_ROCKE_PLATFORM_PYTEST_GROUP_instances_dispatch instances dispatch)
      set(_ROCKE_PLATFORM_PYTEST_GROUP_runtime_tools
          runtime portable_ir tools analysis in_tree)
      set(_rocke_pytest_other_arguments
          "${ROCKE_TEST_CTEST_REL}tests" --ignore "${ROCKE_TEST_CTEST_REL}tests/library")
      foreach(_rocke_pytest_group IN LISTS _ROCKE_PLATFORM_PYTEST_GROUPS)
        set(_rocke_pytest_group_dirs ${_ROCKE_PLATFORM_PYTEST_GROUP_${_rocke_pytest_group}})
        if(NOT _rocke_pytest_group_dirs)
          message(FATAL_ERROR
                  "platform pytest group ${_rocke_pytest_group} lists no directory: "
                  "set _ROCKE_PLATFORM_PYTEST_GROUP_${_rocke_pytest_group}")
        endif()
        set(_rocke_pytest_group_paths "")
        foreach(_rocke_pytest_dir IN LISTS _rocke_pytest_group_dirs)
          set(_rocke_pytest_dir_path "${ROCKE_TEST_CTEST_REL}tests/${_rocke_pytest_dir}")
          list(APPEND _rocke_pytest_group_paths "${_rocke_pytest_dir_path}")
          list(APPEND _rocke_pytest_other_arguments --ignore "${_rocke_pytest_dir_path}")
        endforeach()
        rocke_platform_pytest_entry(_rocke_pytest_entry ${_rocke_pytest_group}
                                    ${_rocke_pytest_group_paths})
        string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES "${_rocke_pytest_entry}")
      endforeach()
      rocke_platform_pytest_entry(_rocke_pytest_entry other ${_rocke_pytest_other_arguments})
      string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES "${_rocke_pytest_entry}")

      # The tiling package's tests live inside the package
      # (rocke/tests/helpers/tiling), so the rocke_pytest_* parts above, which
      # collect the installed tests/ tree, never see them. Three entries run
      # them from the installed package. The gpu marker, which the tree's
      # conftest.py puts on every test asking for a GPU fixture, splits the
      # tests that launch kernels, in rocke_tiling_gpu_pytest, from the rest.
      # The rest are split again by directory: the figure-drawing tests of
      # visualization/ run in rocke_tiling_visualization_pytest, which the
      # provider categories tier for scheduled CI only (TESTING.md, section
      # 5.1.2), and every other directory runs in rocke_tiling_pytest. None of
      # the three is one of the parts, so none takes their name prefix. Each
      # budget keeps the entry's time on the slowest runners under half its
      # CTest limit. The CPU tests build and check many kernels, so
      # rocke_tiling_pytest has a longer limit than a part; the visualization
      # and GPU tests are fewer. Four kinds of test skip here and pass: the GPU
      # tests, without a usable GPU; the C++-engine cases of the IR tests, as
      # the installed tree has no rocke_engine binding (their Python-engine
      # cases run); the source scans of the dsl_docs and library trees, which
      # the installed tree does not hold (rocke.tests.source_tree); and every
      # test that loads platform/tools/, which it does not hold either. The last
      # kind runs only from a source checkout: the backend coverage
      # re-measurement, the compile of every issued MMA instruction and the
      # gfx950 fp8 x fp4 hardware run (traits/README.md, Backend coverage). Like
      # the rocke_pytest_* parts, the entries get the compiler tool locations
      # but not the compiler requirement policy.
      #
      # No directory above the installed package holds a rocke pytest
      # configuration, so without -c pytest would take one from any pytest.ini or
      # pyproject.toml above the install (a CI workspace has one whose
      # python_files collects nothing here). -c names the installed
      # tests/pytest.ini, the configuration the rocke_pytest_* parts run under.
      set(_ROCKE_TILING_TESTS "${_ROCKE_INSTALLED_PYTHONPATH}/rocke/tests/helpers/tiling")
      set(_ROCKE_TILING_VISUALIZATION_TESTS "${_ROCKE_TILING_TESTS}/visualization")
      set(_rocke_tiling_pytest_launcher "python -m pytest -c \"${ROCKE_TEST_CTEST_REL}tests/pytest.ini\"")
      set(_rocke_tiling_pytest_options "-v -rs --timeout=60")
      string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
             "set(_rocke_tiling_pytest_environment \"PYTHONPATH=${_ROCKE_INSTALLED_PYTHONPATH}\")\n"
             "list(APPEND _rocke_tiling_pytest_environment \${_rocke_compiler_tool_environment})\n"
             "add_test(rocke_tiling_pytest ${_rocke_tiling_pytest_launcher} \"${_ROCKE_TILING_TESTS}\" --ignore \"${_ROCKE_TILING_VISUALIZATION_TESTS}\" ${_rocke_tiling_pytest_options} -m \"not gpu\")\n"
             "set_tests_properties(rocke_tiling_pytest PROPERTIES\n"
             "  LABELS \"ckc\\;python\\;quick\"\n"
             "  ENVIRONMENT \"\${_rocke_tiling_pytest_environment}\"\n"
             "  TIMEOUT 600)\n"
             "add_test(rocke_tiling_visualization_pytest ${_rocke_tiling_pytest_launcher} \"${_ROCKE_TILING_VISUALIZATION_TESTS}\" ${_rocke_tiling_pytest_options} -m \"not gpu\")\n"
             "set_tests_properties(rocke_tiling_visualization_pytest PROPERTIES\n"
             "  LABELS \"ckc\\;python\"\n"
             "  ENVIRONMENT \"\${_rocke_tiling_pytest_environment}\"\n"
             "  TIMEOUT 300)\n"
             "add_test(rocke_tiling_gpu_pytest ${_rocke_tiling_pytest_launcher} \"${_ROCKE_TILING_TESTS}\" ${_rocke_tiling_pytest_options} -m gpu)\n"
             "set_tests_properties(rocke_tiling_gpu_pytest PROPERTIES\n"
             "  LABELS \"ckc\\;python\\;gpu\"\n"
             "  ENVIRONMENT \"\${_rocke_tiling_pytest_environment}\"\n"
             "  TIMEOUT 300)\n")
    endif()

    # Library suites enrol themselves: library/tests/ctest_lists.cmake appends
    # their entries (CPU and GPU lanes) to _ROCKE_INSTALLED_CTEST_ENTRIES. The
    # pinned GPU reference lanes below stay here, driven by
    # cmake/InstallGpuReferences.cmake.
    set(_ROCKE_LIBRARY_CTEST_LISTS "${CMAKE_CURRENT_SOURCE_DIR}/../library/tests/ctest_lists.cmake")
    if(EXISTS "${_ROCKE_LIBRARY_CTEST_LISTS}")
      include("${_ROCKE_LIBRARY_CTEST_LISTS}")
    endif()

    # cmake/InstallGpuReferences.cmake discovers the operations from library/tests.
    foreach(_operation IN LISTS _ROCKE_REFERENCE_OPERATIONS)
      string(TOUPPER "${_operation}" _operation_upper)
      string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
             "add_test(rocke_${_operation}_reference_unit_pytest python -m pytest \"${ROCKE_TEST_CTEST_REL}tests/library/tests/test_${_operation}_reference_contract.py\" -v -rs --timeout=30)\n")
      string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
             "set_tests_properties(rocke_${_operation}_reference_unit_pytest PROPERTIES\n"
             "  LABELS \"ckc\\;python\\;${_operation}\\;host\\;quick\"\n"
             "  ENVIRONMENT \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library\"\n"
             "  TIMEOUT 180)\n")

      # Each installed architecture retains its own GPU CTest result, which
      # covers every variant installed for it, so its timeout scales with them.
      # library/tests/reference_common/docs/gpu-reference-workflow.md records
      # this trade-off and when to register one CTest per variant instead.
      foreach(_arch IN LISTS _ROCKE_${_operation_upper}_INSTALLED_ARCHITECTURES)
        list(LENGTH _ROCKE_${_operation_upper}_${_arch}_INSTALLED_VARIANTS _variant_count)
        math(EXPR _timeout "300 * ${_variant_count}")
        string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
               "add_test(rocke_${_operation}_gpu_${_arch}_pytest python -m pytest \"${ROCKE_TEST_CTEST_REL}tests/library/tests/test_${_operation}_pinned_reference.py\" --rocke-reference-operation ${_operation} --rocke-reference-arch ${_arch} -v -rs)\n")
        string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
               "set_tests_properties(rocke_${_operation}_gpu_${_arch}_pytest PROPERTIES\n"
               "  LABELS \"ckc\\;python\\;${_operation}\\;gpu\\;quick\"\n"
               "  ENVIRONMENT \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library\"\n"
               "  SKIP_RETURN_CODE 77\n"
               "  TIMEOUT ${_timeout})\n")
      endforeach()
    endforeach()

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
