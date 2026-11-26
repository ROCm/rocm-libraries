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
option(ROCKE_INSTALL_DEVELOPMENT
       "Install the rocke_core and rocke_library archives, the public headers and the ckc CMake package"
       ON)
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

# Record where `target`, a program the build makes, lands: its path is written to
# <target>.path at the top of the build tree, under the configuration's
# directory in a multi-config build, as tests/run_all.py looks for it. The name
# and directory vary with the mode (the provider renames C++ test programs and
# puts them in bin/), so the runner reads the configured path instead of
# guessing it. The top is CMAKE_BINARY_DIR, the directory holding CMakeCache.txt,
# which is the --build-root run_all expects in every mode: the platform or rocke
# tree configured alone, and the hip-kernel-provider build that adds rocke as a
# subdirectory (there CMAKE_CURRENT_BINARY_DIR is <build-root>/rocke/platform).
function(rocke_record_program_path target)
  get_property(_rocke_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
  if(_rocke_multi_config)
    set(_rocke_path_file "${CMAKE_BINARY_DIR}/$<CONFIG>/${target}.path")
  else()
    set(_rocke_path_file "${CMAKE_BINARY_DIR}/${target}.path")
  endif()
  file(GENERATE OUTPUT "${_rocke_path_file}" CONTENT "$<TARGET_FILE:${target}>\n")
endfunction()

# Every installed rocKE pytest entry except the explicit gates starts its
# command with this text. The rocke.tests.ctest_shard plugin and its flag make
# the entry run only the CI shard's share of its tests. The gates stop on a
# partial run, so they keep their own command. A build that installs the
# shared C++ engine beside the package appends the
# rocke.tests.require_cpp_engine plugin (see ROCKE_BUILD_SHARED_ENGINE below).
set(_ROCKE_INSTALLED_PYTEST
    "python -m pytest -p rocke.tests.ctest_shard --rocke-shard-from-gtest-env")

# A pytest entry whose share of a CI shard takes more than a few seconds and
# whose tests launch kernels in few or none of them puts this text right after
# ${_ROCKE_INSTALLED_PYTEST}: CTest runs a shard's entries one at a time, and
# the rocke.tests.ctest_workers plugin runs the entry's share in worker
# processes, each on its own block of the CPUs; the few GPU tests of such an
# entry share the shard's GPU among the workers. rocke_library_gpu_pytest and
# rocke_platform_gpu_pytest take it too, their workers sharing the shard's GPU:
# their tests compile their kernels on the CPU before they launch them.
set(_ROCKE_INSTALLED_PYTEST_WORKERS "-p rocke.tests.ctest_workers --rocke-workers 4")

# The marker filter and per-test limit of every rocke_pytest_<part>. The
# filter leaves out the tests marked gpu, which launch GPU work;
# rocke_platform_gpu_pytest runs them, with the same per-test limit.
set(_ROCKE_PLATFORM_PYTEST_MARKERS "not integration and not gpu")
set(_ROCKE_PLATFORM_PYTEST_GPU_MARKERS "gpu and not integration")
set(_ROCKE_PLATFORM_PYTEST_ITEM_TIMEOUT "--timeout=60")

# Return the installed-CTestTestfile text that registers rocke_pytest_<part>,
# one part of the platform pytest suite. The remaining arguments are the part's
# pytest path and option arguments. _ROCKE_PLATFORM_PYTEST_IGNORE_<part>, when
# set, lists files under tests/ that the part must not collect (for example a
# file that runs in its own CTest entry); each becomes an --ignore. Every
# argument is quoted. Every part shares the installed pytest command, the
# marker filter, the per-test limit, the labels, the CTest limit and the
# environment list that the installed CTestTestfile defines as
# _rocke_platform_pytest_environment.
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
         "add_test(${_test_name} ${_ROCKE_INSTALLED_PYTEST} ${_ROCKE_INSTALLED_PYTEST_WORKERS}${_pytest_arguments} -m \"${_ROCKE_PLATFORM_PYTEST_MARKERS}\" -v -rs ${_ROCKE_PLATFORM_PYTEST_ITEM_TIMEOUT})\n"
         "set_tests_properties(${_test_name} PROPERTIES\n"
         "  LABELS \"ckc;python;standard\"\n"
         "  ENVIRONMENT \"\${_rocke_platform_pytest_environment}\"\n"
         "  TIMEOUT ${ROCKE_TEST_TIMEOUT_TOOLCHAIN})\n")
  set(${out_var} "${_entry}" PARENT_SCOPE)
endfunction()

set(_ROCKE_DEFAULT_PYTHON_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")
set(ROCKE_PYTHON_INSTALL_DIR "${_ROCKE_DEFAULT_PYTHON_INSTALL_DIR}"
    CACHE STRING "Relative install destination for rocke and rocke_engine")

# Where the rocke package and the engine's Python modules (rocke_engine,
# rocke_library_engine) install, side by side so one sys.path entry finds all
# of them. Provider mode co-locates them with the installed tests
# (bin/<subdir>), whose CTest entries put that directory on PYTHONPATH and
# whose files TheRock's test artifact captures (paired with a matching include
# in ml-libs/artifact-hipkernelprovider.toml for bin/<subdir>/rocke/**). The
# modules and the shared engine are .so files, which TheRock's lib component
# claims wherever they are (its default patterns match every *.so and *.so.*,
# and it comes first in the component chain), so until that manifest excludes
# them from lib and includes them in test, they ship in the provider's lib
# artifact rather than with the tests. A standalone install keeps the
# site-packages layout.
if(ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR)
  set(ROCKE_PYTHON_PACKAGE_INSTALL_DIR "${ROCKE_TEST_INSTALL_DIR}")
else()
  set(ROCKE_PYTHON_PACKAGE_INSTALL_DIR "${ROCKE_PYTHON_INSTALL_DIR}")
endif()

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

# --- C++ keyword table ---------------------------------------------------------
# Both engines refuse a value name an author picks when it is a C++ keyword (the
# HIP lowering prints such names unchanged). python/rocke/core/cxx_keywords.py is
# the one list; this engine gets a copy generated from it at build time, in the
# build tree and never committed. tests/core/test_value_names.cpp includes the
# same copy to check this engine refuses every word of it.
set(_rocke_cxx_keywords_include_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(_rocke_cxx_keywords_inc "${_rocke_cxx_keywords_include_dir}/rocke/cxx_keywords.inc")
add_custom_command(
  OUTPUT "${_rocke_cxx_keywords_inc}"
  COMMAND "${Python3_EXECUTABLE}" -I
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_cxx_keywords_cpp.py"
          --out "${_rocke_cxx_keywords_inc}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_cxx_keywords_cpp.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/rocke/core/cxx_keywords.py"
  COMMENT "Generating the C++ keyword table from python/rocke/core/cxx_keywords.py"
  VERBATIM)
target_sources(rocke_core_objects PRIVATE "${_rocke_cxx_keywords_inc}")

# --- Portable-IR bundle freshness / provenance stamp -------------------------
# Generate a deterministic, git-independent provenance header from the engine
# and replay CLI sources plus their CMake build recipes. This common bundle ID is
# a build dependency, not merely a configure-time value: an incremental edit
# regenerates the stamp before rocke_build_id.o is rebuilt. No lowering/emission
# TU includes it, so LLVM-IR stays byte-identical.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/rocke_build_id.cmake")
rocke_compute_build_id(ROCKE_CONFIGURE_BUILD_ID "${CMAKE_CURRENT_SOURCE_DIR}")
# Human-readable version. Bump the leading semver when the IR contract changes.
# The date is taken from SOURCE_DATE_EPOCH (https://reproducible-builds.org/specs/
# source-date-epoch/) so the same commit built with the same value reports the
# same version; without it the date is omitted rather than stamped with the
# configure time, which would make two builds of one commit differ. The
# content-hash build id (ROCKE_CONFIGURE_BUILD_ID) is what tells builds apart.
set(ROCKE_ENGINE_VERSION "1.0.0")
if(DEFINED ENV{SOURCE_DATE_EPOCH} AND NOT "$ENV{SOURCE_DATE_EPOCH}" STREQUAL "")
  string(TIMESTAMP ROCKE_BUILD_DATE "%Y%m%d" UTC)
  set(ROCKE_ENGINE_VERSION "${ROCKE_ENGINE_VERSION}+${ROCKE_BUILD_DATE}")
endif()
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

# --- Shared engine for the ctypes callers -------------------------------------
# Python reaches the engine's C ABI through ctypes, which needs a .so rather than
# the archive: rocke.core.native_engine lowers serialized IR through it (the cpp
# backend and the both-engine tests), and the portable-IR tooling
# (rocke.portable_ir.src.online) reaches the importer, recipe VM and lowerer the
# same way. online.build_lib() can produce one on demand, but a CI gate wants a
# named target it can build and point ROCKE_ONLINE_LIB at, with the link flags
# living here rather than duplicated in workflow YAML. Supplying the object
# library directly retains every C ABI entry point without linker-specific
# whole-archive flags. On Windows, it also gives CMake the .obj files needed to
# generate the DLL export definition.
option(ROCKE_BUILD_SHARED_ENGINE
       "Also link a shared librocke.so (needed by the Python ctypes callers)" OFF)
set(_ROCKE_SHARED_ENGINE_BESIDE_PACKAGE OFF)
if(ROCKE_BUILD_SHARED_ENGINE)
  add_library(rocke_shared SHARED $<TARGET_OBJECTS:rocke_core_objects>)
  set_target_properties(rocke_shared PROPERTIES OUTPUT_NAME rocke)
  target_link_libraries(rocke_shared PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
  if(MSVC)
    set_target_properties(rocke_shared PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
  elseif(UNIX)
    target_link_libraries(rocke_shared PRIVATE m)
  endif()
  # A provider build installs it beside the rocke package, where
  # rocke.core.native_engine finds it, as the engine's Python modules do, and
  # only with its tests: there it is test content, and lib/ would make it a
  # public library of the ROCm install (TheRock's lib component still claims
  # it, as it does every .so; see ROCKE_PYTHON_PACKAGE_INSTALL_DIR). A
  # standalone install keeps lib/.
  if(ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR)
    if(ROCKE_INSTALL_TESTS)
      set(_ROCKE_SHARED_ENGINE_BESIDE_PACKAGE ON)
      install(TARGETS rocke_shared
              LIBRARY DESTINATION "${ROCKE_PYTHON_PACKAGE_INSTALL_DIR}"
              RUNTIME DESTINATION "${ROCKE_PYTHON_PACKAGE_INSTALL_DIR}")
    endif()
  else()
    set(ROCKE_SHARED_ENGINE_ABSENT_REASON
        "a standalone install puts it in the library directory, not beside the rocke package")
    install(TARGETS rocke_shared
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endif()

# Where the build installs the shared engine beside the rocke package, its
# installed pytest entries count as covering the C++ engine: they load
# rocke.tests.require_cpp_engine, which sets ROCKE_REQUIRE_CPP_ENGINE=1 and
# ROCKE_CPP_STRICT=1 for the run, so a missing or broken engine fails them
# instead of skipping its cases or falling back to the Python engine.
if(_ROCKE_SHARED_ENGINE_BESIDE_PACKAGE)
  string(APPEND _ROCKE_INSTALLED_PYTEST " -p rocke.tests.require_cpp_engine")
endif()

include(CTest)

# The CTest TIMEOUT of every test entry, one limit per resource class. Outside
# BUILD_TESTING: the installed test text below uses the limits too.
include("${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/TestTimeouts.cmake")

# Every installed GPU pytest entry limits each test, so a test that stalls
# fails under its own name instead of running the entry into its CTest limit
# unnamed. It uses pytest-timeout's thread method: a stall there is mostly
# inside a HIP or COMGR call, which the signal method cannot interrupt. On a
# stall the method prints the test's stack under its name (the -v line) and
# ends the pytest process; rocke.tests.ctest_workers names a worker that ended
# so, and rocke.tests.run_records records the tests it did not run as
# unfinished. Two limits of the method remain: the rest of that process's
# tests do not run, and a stall inside a native call that holds the Python GIL
# stops the method's timer thread too, so only the entry's CTest limit ends it,
# without the test's name.
#
# The rocke.tests.item_time_limit plugin keeps one per-test limit for every
# test of the entry but one that declares its slowest measured CI time (the
# rocke_measured_seconds mark), which gets twice that time; a pytest.mark.timeout
# above the entry's limit is a usage error. An entry with no per-test limit of
# its own puts _ROCKE_INSTALLED_PYTEST_GPU_TEST_LIMIT at the end of its
# command; one with a lower limit adds only _ROCKE_INSTALLED_PYTEST_GPU_TEST_POLICY,
# the plugins and the method. The policy also loads rocke.tests.require_gpu
# (its docstring says why). Every GPU entry also takes the CTest resource lock _ROCKE_INSTALLED_GPU_RESOURCE_LOCK, so a
# parallel ctest -j never runs two of them at once on the one GPU.
set(_ROCKE_INSTALLED_PYTEST_GPU_TEST_POLICY
    "-p rocke.tests.item_time_limit -p rocke.tests.require_gpu --timeout-method=thread")
# Where the shared engine is beside the package, the policy also loads
# rocke.tests.differential_engines (ROCKE_BACKEND=both): the GPU checks run the
# Python engine's kernels, which the provider ships, and compare the C++
# engine's output with them on every kernel they build.
if(_ROCKE_SHARED_ENGINE_BESIDE_PACKAGE)
  string(APPEND _ROCKE_INSTALLED_PYTEST_GPU_TEST_POLICY
         " -p rocke.tests.differential_engines")
endif()
set(_ROCKE_INSTALLED_PYTEST_GPU_TEST_LIMIT
    "--timeout=${ROCKE_TEST_ITEM_TIMEOUT_GPU} ${_ROCKE_INSTALLED_PYTEST_GPU_TEST_POLICY}")
set(_ROCKE_INSTALLED_GPU_RESOURCE_LOCK "rocke_gpu")

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
# The rocke_core and rocke_library targets share the ckcTargets export set, so
# every install rule that names that set sits under ROCKE_INSTALL_DEVELOPMENT:
# library/cpp/CMakeLists.txt gates its half on the same option, and exporting a
# target that is not installed is a generate-time error.
if(ROCKE_INSTALL_DEVELOPMENT)
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
endif()

if(ROCKE_INSTALL_PYTHON_PACKAGE)
  # In provider mode the package lands next to rocke_installed_smoke.py, so the
  # Python smoke is self-contained inside TheRock's test artifact.
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/rocke"
          DESTINATION "${ROCKE_PYTHON_PACKAGE_INSTALL_DIR}"
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

    # Which of the C++ engine's Python modules this build makes, or why it
    # makes none, and whether it makes the shared engine, or why not, for the
    # installed guard rocke_engine_modules_check (rocke.tests.engine_modules),
    # which fails when a module or the shared engine the build made does not
    # load. Under tests/, which TheRock's test artifact captures. A build that
    # decides against the modules (the provider build on Windows) says why in
    # ROCKE_ENGINE_MODULES_ABSENT_REASON, one with ROCKE_BUILD_PYBIND off gets
    # the default reason below, and one that does not install the shared engine
    # beside the package says why in ROCKE_SHARED_ENGINE_ABSENT_REASON.

    # `value` as a JSON string literal: its backslashes, then its quotes, escaped.
    function(_rocke_json_string out_var value)
      string(REPLACE "\\" "\\\\" value "${value}")
      string(REPLACE "\"" "\\\"" value "${value}")
      set(${out_var} "\"${value}\"" PARENT_SCOPE)
    endfunction()
    get_property(_ROCKE_ENGINE_PYTHON_MODULES GLOBAL PROPERTY ROCKE_ENGINE_PYTHON_MODULES)
    if(_ROCKE_ENGINE_PYTHON_MODULES)
      list(TRANSFORM _ROCKE_ENGINE_PYTHON_MODULES REPLACE "(.+)" "\"\\1\"")
      list(JOIN _ROCKE_ENGINE_PYTHON_MODULES ", " _rocke_modules)
      set(_rocke_modules_fields "\"modules\": [${_rocke_modules}]")
    else()
      if(NOT DEFINED ROCKE_ENGINE_MODULES_ABSENT_REASON)
        set(ROCKE_ENGINE_MODULES_ABSENT_REASON "the build had ROCKE_BUILD_PYBIND off")
      endif()
      _rocke_json_string(_rocke_absent_reason "${ROCKE_ENGINE_MODULES_ABSENT_REASON}")
      set(_rocke_modules_fields "\"modules\": [], \"absent_reason\": ${_rocke_absent_reason}")
    endif()
    if(_ROCKE_SHARED_ENGINE_BESIDE_PACKAGE)
      set(_rocke_shared_engine_fields
          "\"shared_engine\": \"$<TARGET_FILE_NAME:rocke_shared>\"")
    else()
      if(NOT DEFINED ROCKE_SHARED_ENGINE_ABSENT_REASON)
        set(ROCKE_SHARED_ENGINE_ABSENT_REASON "the build had ROCKE_BUILD_SHARED_ENGINE off")
      endif()
      _rocke_json_string(_rocke_absent_reason "${ROCKE_SHARED_ENGINE_ABSENT_REASON}")
      set(_rocke_shared_engine_fields
          "\"shared_engine\": \"\", \"shared_engine_absent_reason\": ${_rocke_absent_reason}")
    endif()
    set(_rocke_engine_modules_json "{${_rocke_modules_fields}, ${_rocke_shared_engine_fields}}")
    set(_ROCKE_ENGINE_MODULES_MARKER "${CMAKE_CURRENT_BINARY_DIR}/rocke_engine_modules.json")
    file(GENERATE OUTPUT "${_ROCKE_ENGINE_MODULES_MARKER}"
         CONTENT "${_rocke_engine_modules_json}\n")
    install(FILES "${_ROCKE_ENGINE_MODULES_MARKER}"
            DESTINATION "${ROCKE_TEST_INSTALL_DIR}/tests")

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
    # and file. The HIP source identity gate reuses the runner and worker, and
    # lowers through the installed rocke_ir_lower_cli (rocke_hip_source_identity
    # below).
    set(_ROCKE_DIFFERENTIAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/instances/differential")
    install(FILES "${_ROCKE_DIFFERENTIAL_DIR}/run_diff.py"
                  "${_ROCKE_DIFFERENTIAL_DIR}/python_batch_worker.py"
                  "${_ROCKE_DIFFERENTIAL_DIR}/golden_store.py"
                  "${_ROCKE_DIFFERENTIAL_DIR}/test_hip_source_identity.py"
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
              PATTERN "reference_bundles" EXCLUDE)

    else()
      message(WARNING
              "rocKE: library tree not found at ${_ROCKE_LIBRARY_DIR}; the installed "
              "golden gate will fail on its library-kernel cases")
    endif()

    # Keep the LDS skill CLI under tests/ so TheRock's test artifact includes it.
    install(PROGRAMS
            "${CMAKE_CURRENT_SOURCE_DIR}/dsl_docs/optimization/utilities/skills/lds-bank-conflict-expert/scripts/predict.py"
            DESTINATION "${ROCKE_TEST_INSTALL_DIR}/tests/analysis/lds-bank-conflict-expert")
    # The counted ISA gate uses the shared disassembler without installing the
    # source dsl_docs tree. Stage that one support module at an explicit path;
    # CTest supplies it through ROCKE_ISA_INSPECT_PROBE below.
    install(FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/dsl_docs/optimization/utilities/tools/dsl_probes/probe_isa_inspect.py"
            DESTINATION "${ROCKE_TEST_INSTALL_DIR}/tests/support")

    # Install the full pytest suite for installed/artifact test execution.
    # Excludes: C/C++ parity emitters (compiled against rocke_core at build time),
    # the differential harness (needs compiler + source; its Python-only golden
    # check is installed separately above), the WaveScope sidecar,
    # LDS analyzer and dsl_probes tests (load tools from the adjacent dsl_doc
    # source tree), the arch-table parity test (reads cpp/core/arch/data.cpp),
    # the MMA catalog cross-check unit tests (load tools/ from the source tree),
    # the tree-wide checks of rocke.tests.torch_uses and
    # rocke.tests.toolchain_pins and their allowlists (they list the source
    # checkout), the compliance-pattern test (reads rocke/tools/ and lists
    # the source checkout), the registration checks of rocke_pending_lanes_check
    # (they configure the source tree), and the arch-domain artifact/probe unit
    # tests. The host-only support-policy
    # units are installed with the rest of tests/tools; the full regeneration
    # sweep has its own CTest timeout and resource policy.
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
              PATTERN "test_gpu_references.py" EXCLUDE
              PATTERN "test_pending_lanes.py" EXCLUDE
              PATTERN "test_limbo_audit.py" EXCLUDE
              PATTERN "test_gpu_work_placement.py" EXCLUDE
              PATTERN "test_quick_tier.py" EXCLUDE
              PATTERN "test_no_torch.py" EXCLUDE
              PATTERN "no_torch_allowlist.txt" EXCLUDE
              PATTERN "test_toolchain_pins.py" EXCLUDE
              PATTERN "toolchain_pins_allowlist.txt" EXCLUDE
              PATTERN "test_compliance_patterns.py" EXCLUDE)

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

    # The golden tool and its index: tests/tools/test_golden_tool.py checks the
    # installed tree's goldens with them.
    set(_ROCKE_GOLDEN_TOOL "${CMAKE_CURRENT_SOURCE_DIR}/../tools/golden.py")
    if(EXISTS "${_ROCKE_GOLDEN_TOOL}")
      install(FILES "${_ROCKE_GOLDEN_TOOL}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/../tools/golden_index.py"
              DESTINATION "${ROCKE_TEST_INSTALL_DIR}/tests/tools")
    endif()

    # The per-entry time check judges the installed registration against the CI
    # shard budgets, from the entry times recorded beside it. Staged with the
    # other test tools when the top-level tools/ sibling exists; the entry that
    # runs it is registered below on the same condition.
    set(_ROCKE_ENTRY_TIME_CHECK "${CMAKE_CURRENT_SOURCE_DIR}/../tools/check_entry_times.py")
    if(EXISTS "${_ROCKE_ENTRY_TIME_CHECK}")
      install(FILES "${_ROCKE_ENTRY_TIME_CHECK}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/../tools/entry_times.json"
              DESTINATION "${ROCKE_TEST_INSTALL_DIR}/tests/tools")
    endif()

    # Test-only implementation shared by the installed support-policy units and
    # arch-domain reproduction gate. Keep it outside the production package:
    # CTest owns the compiler sweep; pytest checks the policy without one.
    # tests/conftest.py also points rocke.tests.ir_validity at this copy, so the
    # installed pytest runs compile lowered IR with the same tool.
    install(FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_arch_domain.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_ir_validity.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/_corpus.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/_bounded_run.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/_flavor_gate.py"
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
           "set_tests_properties(${_test_target} PROPERTIES LABELS \"ckc;unit;host;quick\" TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})\n")
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

    # Library tests need both the staged library packages and the exact installed
    # rocke package. Escape Windows' path separator because CTest ENVIRONMENT is
    # itself a semicolon-separated list.
    if(WIN32)
      set(_ROCKE_LIBRARY_PYTHONPATH
          "${ROCKE_TEST_CTEST_REL}tests/library\\;${_ROCKE_INSTALLED_PYTHONPATH}")
    else()
      set(_ROCKE_LIBRARY_PYTHONPATH
          "${ROCKE_TEST_CTEST_REL}tests/library:${_ROCKE_INSTALLED_PYTHONPATH}")
    endif()

    # Python test entries invoke via "python" (resolved from PATH at test time)
    # rather than executing the .py directly, so they work on Windows where
    # shebangs are not supported. The CI test environment activates a venv with
    # requirements-test.txt before running ctest.
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "add_test(rocke_python_import_smoke python \"${ROCKE_TEST_CTEST_REL}rocke_installed_smoke.py\")\n")
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "set_tests_properties(rocke_python_import_smoke PROPERTIES LABELS \"ckc;python;quick\" TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})\n")

    # The engine modules and the shared engine the build made load from the
    # installed tree, and lower_kernel_to_llvm lowers through the C++ engine:
    # without this entry, losing them turns thousands of installed tests into
    # skips and the run stays green. It skips (exit 77) with the marker's
    # reasons when the build made neither. Not a pytest entry, so it runs whole
    # on every CI shard; it costs about a second. With the import smoke it is
    # the quick tier's packaging check (TESTING.md, section 5.0.3).
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "add_test(rocke_engine_modules_check python -m rocke.tests.engine_modules \"${ROCKE_TEST_CTEST_REL}tests/rocke_engine_modules.json\")\n"
           "set_tests_properties(rocke_engine_modules_check PROPERTIES\n"
           "  LABELS \"ckc;python;host;quick\"\n"
           "  ENVIRONMENT \"PYTHONPATH=${_ROCKE_INSTALLED_PYTHONPATH}\"\n"
           "  SKIP_RETURN_CODE 77\n"
           "  TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})\n")

    # Each CI shard compares the HIP emitter configurations it owns
    # (rocke.tests.ci_shard, imported from the installed package); without the
    # shard variables it compares them all.
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
             "optimization_barrier_emit.py\" --count 17 "
             "--rocke-shard-from-gtest-env)\n"
             "set_tests_properties(rocke_optimization_barrier_hip_pair "
             "PROPERTIES LABELS \"ckc;python;host;standard\"\n"
             "  ENVIRONMENT \"PYTHONPATH=${_ROCKE_INSTALLED_PYTHONPATH}\"\n"
             "  TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})\n")
    endif()
    # The golden entries below check only their CI shard's share (the
    # rocke.tests.ci_shard rule, the same as the pytest entries'), so each
    # standard shard runs a quarter of them; without the shard variables they
    # check everything.
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "add_test(rocke_golden_static python \"${ROCKE_TEST_CTEST_REL}rocke_installed_golden_test.py\" --rocke-shard-from-gtest-env)\n")
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "set_tests_properties(rocke_golden_static PROPERTIES LABELS \"ckc;python;golden;host;standard\" TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})\n"
           "${_rocke_barrier_parity_entries}")

    # Would every entry of this registration fit its CI shard budget? Run
    # from the CTest directory, the check lists the registration there.
    # It is one case of its own, so a sharded tier runs it on one shard only.
    if(EXISTS "${_ROCKE_ENTRY_TIME_CHECK}")
      string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
             "add_test(rocke_entry_time_check python \"${ROCKE_TEST_CTEST_REL}tests/tools/check_entry_times.py\" check --ctest-dir . --rocke-shard-from-gtest-env)\n"
             "set_tests_properties(rocke_entry_time_check PROPERTIES\n"
             "  LABELS \"ckc;python;host;standard\"\n"
             "  ENVIRONMENT \"PYTHONPATH=${_ROCKE_INSTALLED_PYTHONPATH}\"\n"
             "  TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})\n")
    endif()

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
    # uses the installed package data that downstream/provider CI consumes. Each
    # CI shard checks the declaration keys it owns (rocke.tests.ci_shard).
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "add_test(rocke_arch_domain_reproducibility python -u \"${ROCKE_TEST_CTEST_REL}tests/arch_domain/gen_arch_domain.py\" --check --unvalidated-exit-code 77 --rocke-shard-from-gtest-env)\n")
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "set_tests_properties(rocke_arch_domain_reproducibility PROPERTIES\n"
           "  LABELS \"ckc;python;host;integration;toolchain;generated-artifact;standard\"\n"
           "  ENVIRONMENT \"PYTHONPATH=${_ROCKE_INSTALLED_PYTHONPATH}\"\n"
           "  RUN_SERIAL TRUE\n"
           "  SKIP_RETURN_CODE 77\n"
           "  TIMEOUT ${ROCKE_TEST_TIMEOUT_TOOLCHAIN})\n")

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
           "  TIMEOUT ${ROCKE_TEST_TIMEOUT_TOOLCHAIN})\n")

    # The differential golden: every parity family's Python reference output,
    # LLVM IR at each LLVM flavor (the flavor is a lowering argument, so one host
    # checks all of them) and canonical IR. tests/instances/differential/golden/README.md.
    set(_ROCKE_RUN_DIFF "${ROCKE_TEST_CTEST_REL}tests/instances/differential/run_diff.py")
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "add_test(rocke_differential_golden_ll python \"${_ROCKE_RUN_DIFF}\" --mode ll --python-only --check-golden --flavor all --rocke-shard-from-gtest-env)\n")
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "set_tests_properties(rocke_differential_golden_ll PROPERTIES LABELS \"ckc;python;golden;host;standard\" ENVIRONMENT \"PYTHONPATH=${_ROCKE_INSTALLED_PYTHONPATH}\" TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})\n")
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "add_test(rocke_differential_golden_canonical python \"${_ROCKE_RUN_DIFF}\" --mode ir --canonical --python-only --check-golden --rocke-shard-from-gtest-env)\n")
    string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
           "set_tests_properties(rocke_differential_golden_canonical PROPERTIES LABELS \"ckc;python;golden;host;standard\" ENVIRONMENT \"PYTHONPATH=${_ROCKE_INSTALLED_PYTHONPATH}\" TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})\n")

    # The installed tests compare the C++ HIP lowering with the Python one
    # through rocke_ir_lower_cli, as the wheel has no rocke_engine binding:
    # rocke_hip_source_identity and rocke_hip_lowering_parity below.
    #
    # Neither is registered on Windows: the Windows CI shards run close to their
    # time limit. The cost is that the MSVC-built engine's HIP output is not
    # compared with Python's in CI; it matched on every parity kernel on the
    # Windows lanes when this gate was added, and float formatting, the most
    # host-sensitive piece, keeps its own rocke_float_format test there.
    # Register them on Windows once those shards have room. They are registered
    # on every other platform.
    set(_ROCKE_IR_LOWER_CLI_ENVIRONMENT "")
    if(TARGET rocke_ir_lower_cli AND NOT WIN32)
      rocke_cxx_test_installed_name(_rocke_installed_name rocke_ir_lower_cli)
      set(_ROCKE_IR_LOWER_CLI_ENVIRONMENT
          "ROCKE_IR_LOWER_CLI=${ROCKE_CXX_TEST_CTEST_REL}${_rocke_installed_name}${CMAKE_EXECUTABLE_SUFFIX}")
    elseif(WIN32)
      message(STATUS "rocKE: the installed tests do not compare the C++ HIP lowering on Windows (CI shard time; see comment above)")
    endif()

    # Both engines lower every parity kernel to the same HIP source. One family
    # is one test, and the test bounds its emitter run below the 120 s
    # per-test limit (EMITTER_TIMEOUT_S). The path is under tests/, in
    # rocke_pytest_instances_dispatch's directory.
    set(_ROCKE_HIP_SOURCE_IDENTITY_TEST "instances/differential/test_hip_source_identity.py")
    if(_ROCKE_IR_LOWER_CLI_ENVIRONMENT)
      string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
             "add_test(rocke_hip_source_identity ${_ROCKE_INSTALLED_PYTEST} ${_ROCKE_INSTALLED_PYTEST_WORKERS} \"${ROCKE_TEST_CTEST_REL}tests/${_ROCKE_HIP_SOURCE_IDENTITY_TEST}\" -v -rs --timeout=120)\n")
      string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
             "set_tests_properties(rocke_hip_source_identity PROPERTIES LABELS \"ckc;python;host;standard\" ENVIRONMENT \"PYTHONPATH=${_ROCKE_INSTALLED_PYTHONPATH};${_ROCKE_IR_LOWER_CLI_ENVIRONMENT}\" TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})\n")
    endif()

    # Both engines lower each MMA atom and operand register type to the same HIP
    # source, or refuse it alike, and resolve a serialized kernel's reused SSA
    # names alike. These files hold the both-engine checks and the checks of
    # the helpers they use, so they run here, with the program, rather than in
    # rocke_pytest_core, whose time limit has no room for them and which has no
    # C++ HIP lowering; hipcc checks the source where ROCm is found. The paths
    # are under tests/, all in rocke_pytest_core's directory.
    set(_ROCKE_HIP_LOWERING_PARITY_TESTS
        "core/test_hip_lowering_parity.py"
        "core/test_mma_operand_register_type.py"
        "core/test_serialized_ir_name_scopes.py")
    if(_ROCKE_IR_LOWER_CLI_ENVIRONMENT)
      list(TRANSFORM _ROCKE_HIP_LOWERING_PARITY_TESTS
           PREPEND "\"${ROCKE_TEST_CTEST_REL}tests/" OUTPUT_VARIABLE _rocke_parity_paths)
      list(TRANSFORM _rocke_parity_paths APPEND "\"")
      list(JOIN _rocke_parity_paths " " _rocke_parity_paths)
      string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
             "set(_rocke_hip_lowering_parity_environment \"PYTHONPATH=${_ROCKE_INSTALLED_PYTHONPATH};${_ROCKE_IR_LOWER_CLI_ENVIRONMENT}\")\n"
             "list(APPEND _rocke_hip_lowering_parity_environment \${_rocke_compiler_tool_environment})\n"
             "add_test(rocke_hip_lowering_parity ${_ROCKE_INSTALLED_PYTEST} ${_ROCKE_INSTALLED_PYTEST_WORKERS} ${_rocke_parity_paths} -v -rs --timeout=60)\n"
             "set_tests_properties(rocke_hip_lowering_parity PROPERTIES\n"
             "  LABELS \"ckc;python;host;standard\"\n"
             "  ENVIRONMENT \"\${_rocke_hip_lowering_parity_environment}\"\n"
             "  TIMEOUT ${ROCKE_TEST_TIMEOUT_TOOLCHAIN})\n")
    endif()

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
      # Paths are relative to the CTestTestfile, except ROCKE_REPLAY_CLI, the
      # portable-IR replay CLI, and ROCKE_PROVENANCE_MANIFEST, its provenance
      # manifest: their reader (rocke.assets.beside_package) takes a relative
      # value beside the installed rocke package, wherever a test runs, so
      # they are relative to ROCKE_PYTHON_PACKAGE_INSTALL_DIR (the same as the
      # CTestTestfile's directory in provider mode, not in a standalone
      # install). ROCKE_UNSIGNED_INT_OPS_CLI is the C API program whose
      # --print-llvm output test_unsigned_int_ops.py compares with the Python
      # builder's text; ROCKE_INTEXPR_CLI is the C++ integer-expression
      # evaluator whose --eval output test_intexpr.py compares with Python's;
      # ROCKE_PARAM_FACTS_CLI prints the C++ print_ir, verifier messages and
      # canonicalize that test_param_facts.py compares with Python's;
      # ROCKE_GPU_CHECKED_VALUE_CLI prints the C++ builder's GPU-check cases
      # and marker verifier messages that test_gpu_checked_value.py compares.
      file(RELATIVE_PATH _rocke_portable_ir_beside_package
           "${CMAKE_INSTALL_PREFIX}/${ROCKE_PYTHON_PACKAGE_INSTALL_DIR}"
           "${CMAKE_INSTALL_PREFIX}/${ROCKE_TEST_INSTALL_DIR}/tests/portable_ir")
      set(_ROCKE_PYTEST_ENVIRONMENT
          "PYTHONPATH=."
          "ROCKE_REPLAY_CLI=${_rocke_portable_ir_beside_package}/rocke_portable_ir_replay_cli${CMAKE_EXECUTABLE_SUFFIX}"
          "ROCKE_PROVENANCE_MANIFEST=${_rocke_portable_ir_beside_package}/rocke_portable_ir_provenance.txt")
      foreach(_rocke_native_parity
              "ROCKE_STORAGE_TEST=rocke_storage"
              "ROCKE_IR_TYPES_TEST=rocke_ir_types"
              "ROCKE_FLOAT_FORMAT_TEST=rocke_float_format"
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
      # group names, so a new directory runs with no edit here. Together with
      # rocke_ctest_shard_pytest below, the entries collect what one entry over
      # tests/ would. To keep a file out of every part, list(APPEND) it (a path
      # under tests/) to _ROCKE_PLATFORM_PYTEST_IGNORE_<part> for the part whose
      # directories hold it, or for "other"; APPEND keeps any value given on the
      # configure line, which tests/test_engine_linkage.py uses. The
      # rocke_pytest_ name prefix is reserved for these entries; the provider
      # categories tier them by that prefix.
      set(_ROCKE_PLATFORM_PYTEST_GROUPS core instances_dispatch runtime_tools)
      # The HIP source identity test and the HIP lowering parity files run in
      # their own entries (above): their both-engine checks need the C++ HIP
      # lowering, which the installed tree reaches only through
      # rocke_ir_lower_cli. Where those entries are not registered (Windows),
      # the files stay in their directory's part, so their Python-engine
      # checks still run there. (A Windows build installs no shared engine,
      # so their C++ cases have no engine to compare there.)
      if(_ROCKE_IR_LOWER_CLI_ENVIRONMENT)
        list(APPEND _ROCKE_PLATFORM_PYTEST_IGNORE_instances_dispatch
             ${_ROCKE_HIP_SOURCE_IDENTITY_TEST})
        list(APPEND _ROCKE_PLATFORM_PYTEST_IGNORE_core ${_ROCKE_HIP_LOWERING_PARITY_TESTS})
      endif()
      set(_ROCKE_PLATFORM_PYTEST_GROUP_core core)
      set(_ROCKE_PLATFORM_PYTEST_GROUP_instances_dispatch instances dispatch)
      set(_ROCKE_PLATFORM_PYTEST_GROUP_runtime_tools
          runtime portable_ir tools analysis in_tree)
      # The tests of the rocke.tests.ctest_shard plugin start many child pytest
      # and CTest runs. They run in rocke_ctest_shard_pytest, which the provider
      # categories tier as standard, so the quick tier does not run them.
      list(APPEND _ROCKE_PLATFORM_PYTEST_IGNORE_other test_ctest_shard.py)
      string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
             "add_test(rocke_ctest_shard_pytest ${_ROCKE_INSTALLED_PYTEST} ${_ROCKE_INSTALLED_PYTEST_WORKERS} \"${ROCKE_TEST_CTEST_REL}tests/test_ctest_shard.py\" -m \"${_ROCKE_PLATFORM_PYTEST_MARKERS}\" -v -rs ${_ROCKE_PLATFORM_PYTEST_ITEM_TIMEOUT})\n"
             "set_tests_properties(rocke_ctest_shard_pytest PROPERTIES\n"
             "  LABELS \"ckc;python;standard\"\n"
             "  ENVIRONMENT \"\${_rocke_platform_pytest_environment}\"\n"
             "  TIMEOUT ${ROCKE_TEST_TIMEOUT_TOOLCHAIN})\n")
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

      # The platform tests that launch GPU work, the ones marked gpu
      # (tests/pytest.ini), run in this GPU entry instead of their parts: it
      # collects tests/ less the library tests and keeps only those tests,
      # with the parts' per-test limit and the GPU policy, lock and CTest
      # limit. Not a part: it selects by mark, not by directory.
      string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
             "add_test(rocke_platform_gpu_pytest ${_ROCKE_INSTALLED_PYTEST} ${_ROCKE_INSTALLED_PYTEST_WORKERS} \"${ROCKE_TEST_CTEST_REL}tests\" --ignore \"${ROCKE_TEST_CTEST_REL}tests/library\" -m \"${_ROCKE_PLATFORM_PYTEST_GPU_MARKERS}\" -v -rs ${_ROCKE_PLATFORM_PYTEST_ITEM_TIMEOUT} ${_ROCKE_INSTALLED_PYTEST_GPU_TEST_POLICY})\n"
             "set_tests_properties(rocke_platform_gpu_pytest PROPERTIES\n"
             "  LABELS \"ckc;python;gpu;standard\"\n"
             "  ENVIRONMENT \"\${_rocke_platform_pytest_environment}\"\n"
             "  RESOURCE_LOCK ${_ROCKE_INSTALLED_GPU_RESOURCE_LOCK}\n"
             "  TIMEOUT ${ROCKE_TEST_TIMEOUT_GPU})\n")

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
      # the three is one of the parts, so none takes their name prefix. Four
      # kinds of test skip here and pass: the GPU tests, without a usable GPU;
      # the C++-engine cases of the IR tests, where the build made no engine
      # modules (see rocke_engine_modules_check; their Python-engine cases
      # run); the source scans of the dsl_docs and library trees, which the
      # installed tree does not hold (rocke.tests.source_tree); and every
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
      # The launcher and options serve every entry that runs a test directory
      # of the installed package (the tiling and heuristics entries below).
      set(_ROCKE_TILING_TESTS "${_ROCKE_INSTALLED_PYTHONPATH}/rocke/tests/helpers/tiling")
      set(_ROCKE_TILING_VISUALIZATION_TESTS "${_ROCKE_TILING_TESTS}/visualization")
      set(_rocke_package_pytest_launcher "${_ROCKE_INSTALLED_PYTEST} -c \"${ROCKE_TEST_CTEST_REL}tests/pytest.ini\"")
      set(_rocke_package_pytest_options "-v -rs --timeout=60")
      string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
             "set(_rocke_tiling_pytest_environment \"PYTHONPATH=${_ROCKE_INSTALLED_PYTHONPATH}\")\n"
             "list(APPEND _rocke_tiling_pytest_environment \${_rocke_compiler_tool_environment})\n"
             "add_test(rocke_tiling_pytest ${_rocke_package_pytest_launcher} ${_ROCKE_INSTALLED_PYTEST_WORKERS} \"${_ROCKE_TILING_TESTS}\" --ignore \"${_ROCKE_TILING_VISUALIZATION_TESTS}\" ${_rocke_package_pytest_options} -m \"not gpu\")\n"
             "set_tests_properties(rocke_tiling_pytest PROPERTIES\n"
             "  LABELS \"ckc;python;standard\"\n"
             "  ENVIRONMENT \"\${_rocke_tiling_pytest_environment}\"\n"
             "  TIMEOUT ${ROCKE_TEST_TIMEOUT_TOOLCHAIN})\n"
             "add_test(rocke_tiling_visualization_pytest ${_rocke_package_pytest_launcher} \"${_ROCKE_TILING_VISUALIZATION_TESTS}\" ${_rocke_package_pytest_options} -m \"not gpu\")\n"
             "set_tests_properties(rocke_tiling_visualization_pytest PROPERTIES\n"
             "  LABELS \"ckc;python\"\n"
             "  ENVIRONMENT \"\${_rocke_tiling_pytest_environment}\"\n"
             "  TIMEOUT ${ROCKE_TEST_TIMEOUT_TOOLCHAIN})\n"
             "add_test(rocke_tiling_gpu_pytest ${_rocke_package_pytest_launcher} \"${_ROCKE_TILING_TESTS}\" ${_rocke_package_pytest_options} ${_ROCKE_INSTALLED_PYTEST_GPU_TEST_POLICY} -m gpu)\n"
             "set_tests_properties(rocke_tiling_gpu_pytest PROPERTIES\n"
             "  LABELS \"ckc;python;gpu\"\n"
             "  ENVIRONMENT \"\${_rocke_tiling_pytest_environment}\"\n"
             "  RESOURCE_LOCK ${_ROCKE_INSTALLED_GPU_RESOURCE_LOCK}\n"
             "  TIMEOUT ${ROCKE_TEST_TIMEOUT_GPU})\n")
      # The kernel-selection heuristics tests live inside the package
      # (rocke/heuristics/tests) and need packages that are not rocKE
      # dependencies: rocke/heuristics/tests/requirements.txt, installed beside
      # the tests, lists them, and each test module reads it, skipping where one
      # is missing (waiting for the build-variant lane). A job that runs this
      # entry installs them with `pip install -r <that requirements.txt>`.
      # Scheduled CI only (the comprehensive tier).
      set(_ROCKE_HEURISTICS_TESTS "${_ROCKE_INSTALLED_PYTHONPATH}/rocke/heuristics/tests")
      string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
             "add_test(rocke_heuristics_pytest ${_rocke_package_pytest_launcher} \"${_ROCKE_HEURISTICS_TESTS}\" ${_rocke_package_pytest_options})\n"
             "set_tests_properties(rocke_heuristics_pytest PROPERTIES\n"
             "  LABELS \"ckc;python;host;comprehensive;full\"\n"
             "  ENVIRONMENT \"PYTHONPATH=${_ROCKE_INSTALLED_PYTHONPATH}\"\n"
             "  TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})\n")
      # check_backend_access_width.py compiles a few copy kernels through COMGR
      # and reads their disassembly. A host module that does that is kept out of
      # ordinary test discovery (TESTING.md, section 5.0.1), so this entry names
      # the file; tests/run_all.py runs the same file as a stage of its own. The
      # provider categories tier it as standard: it guards performance, not
      # correctness. Unlike the entries above it takes the compiler requirement
      # policy: it checks nothing without the tools, so where ROCM_PATH supplies
      # them a missing tool fails it instead of skipping every case.
      string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
             "set(_rocke_tiling_backend_width_pytest_environment \${_rocke_tiling_pytest_environment})\n"
             "list(APPEND _rocke_tiling_backend_width_pytest_environment \${_rocke_compiler_requirement_environment})\n"
             "add_test(rocke_tiling_backend_width_pytest ${_rocke_package_pytest_launcher} \"${_ROCKE_TILING_TESTS}/check_backend_access_width.py\" ${_rocke_package_pytest_options})\n"
             "set_tests_properties(rocke_tiling_backend_width_pytest PROPERTIES\n"
             "  LABELS \"ckc;python;standard\"\n"
             "  ENVIRONMENT \"\${_rocke_tiling_backend_width_pytest_environment}\"\n"
             "  TIMEOUT ${ROCKE_TEST_TIMEOUT_TOOLCHAIN})\n")
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
             "add_test(rocke_${_operation}_reference_unit_pytest ${_ROCKE_INSTALLED_PYTEST} \"${ROCKE_TEST_CTEST_REL}tests/library/tests/test_${_operation}_reference_contract.py\" -v -rs --timeout=30)\n")
      string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
             "set_tests_properties(rocke_${_operation}_reference_unit_pytest PROPERTIES\n"
             "  LABELS \"ckc;python;${_operation};host;standard\"\n"
             "  ENVIRONMENT \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library\"\n"
             "  TIMEOUT ${ROCKE_TEST_TIMEOUT_CPU})\n")

      # Each installed architecture retains its own GPU CTest result, which
      # covers every variant installed for it; the gpu limit covers the at most
      # two variants one entry runs, each request of a case stopping at the
      # worker's 100 s request timeout (PINNED_REQUEST_TIMEOUT_SECONDS), below
      # the 120 s per-test limit of the GPU pytest entries.
      # library/tests/reference_common/docs/gpu-reference-workflow.md records
      # this trade-off and when to register one CTest per variant instead.
      foreach(_arch IN LISTS _ROCKE_${_operation_upper}_INSTALLED_ARCHITECTURES)
        string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
               "add_test(rocke_${_operation}_gpu_${_arch}_pytest ${_ROCKE_INSTALLED_PYTEST} \"${ROCKE_TEST_CTEST_REL}tests/library/tests/test_${_operation}_pinned_reference.py\" --rocke-reference-operation ${_operation} --rocke-reference-arch ${_arch} -v -rs ${_ROCKE_INSTALLED_PYTEST_GPU_TEST_LIMIT})\n")
        string(APPEND _ROCKE_INSTALLED_CTEST_ENTRIES
               "set_tests_properties(rocke_${_operation}_gpu_${_arch}_pytest PROPERTIES\n"
               "  LABELS \"ckc;python;${_operation};gpu;standard\"\n"
               "  ENVIRONMENT \"PYTHONPATH=${ROCKE_TEST_CTEST_REL}tests/library\"\n"
               "  SKIP_RETURN_CODE 77\n"
               "  RESOURCE_LOCK ${_ROCKE_INSTALLED_GPU_RESOURCE_LOCK}\n"
               "  TIMEOUT ${ROCKE_TEST_TIMEOUT_GPU})\n")
      endforeach()
    endforeach()

  endif()

  # TheRock starts every installed entry on every hipkernelprovider CI shard. A
  # pytest entry runs only its shard's share when its command starts with
  # ${_ROCKE_INSTALLED_PYTEST}; otherwise it runs whole on every shard. Stop the
  # configure for such an entry. An entry runs pytest when its command runs
  # "-m pytest" or its program is pytest (or py.test). The explicit gates,
  # _ROCKE_EXPLICIT_PYTEST_GATES in library/tests/ctest_lists.cmake, stop on a
  # partial run, so they must not load the shard plugin.
  string(REGEX MATCHALL "add_test\\([^\n]*" _rocke_adds "${_ROCKE_INSTALLED_CTEST_ENTRIES}")
  set(_rocke_bad_pytest_entries "")
  foreach(_rocke_add IN LISTS _rocke_adds)
    if(NOT _rocke_add MATCHES
       "^add_test\\([A-Za-z0-9_]+ (\"?([^ \"]*[/\\\\])?(pytest|py\\.test)(\\.exe)?\"?[ )]|[^\n]* -m pytest[ )])")
      continue()
    endif()
    string(REGEX REPLACE "^add_test\\(([A-Za-z0-9_]+) .*" "\\1" _rocke_name "${_rocke_add}")
    if(_rocke_name IN_LIST _ROCKE_EXPLICIT_PYTEST_GATES)
      if(_rocke_add MATCHES "rocke\\.tests\\.ctest_shard")
        list(APPEND _rocke_bad_pytest_entries
             "${_rocke_name} (an explicit gate must not load the shard plugin)")
      endif()
    else()
      string(FIND "${_rocke_add}" "add_test(${_rocke_name} ${_ROCKE_INSTALLED_PYTEST}"
             _rocke_prefix_pos)
      if(NOT _rocke_prefix_pos EQUAL 0)
        list(APPEND _rocke_bad_pytest_entries "${_rocke_name}")
      endif()
    endif()
  endforeach()
  if(_rocke_bad_pytest_entries)
    list(JOIN _rocke_bad_pytest_entries "\n  " _rocke_bad_pytest_entries)
    message(FATAL_ERROR
            "Each installed rocKE pytest entry must start its command with "
            "\${_ROCKE_INSTALLED_PYTEST}, so that each hipkernelprovider CI shard "
            "runs only its share of the entry's tests (see rocke/TESTING.md):\n"
            "  ${_rocke_bad_pytest_entries}")
  endif()

  # CI runs ctest with a default timeout far above its step limit, so an
  # installed entry without a TIMEOUT that hangs ends the whole CI step instead
  # of failing alone. Stop the configure for such an entry.
  set(_rocke_untimed_entries "")
  foreach(_rocke_add IN LISTS _rocke_adds)
    string(REGEX REPLACE "^add_test\\(([A-Za-z0-9_]+) .*" "\\1" _rocke_name "${_rocke_add}")
    if(NOT _ROCKE_INSTALLED_CTEST_ENTRIES MATCHES
       "set_tests_properties\\(${_rocke_name} PROPERTIES[^)]*TIMEOUT [0-9]+")
      list(APPEND _rocke_untimed_entries "${_rocke_name}")
    endif()
  endforeach()
  if(_rocke_untimed_entries)
    list(JOIN _rocke_untimed_entries "\n  " _rocke_untimed_entries)
    message(FATAL_ERROR
            "Each installed rocKE test entry must set a CTest TIMEOUT, the limit "
            "of its resource class in tests/cmake/TestTimeouts.cmake:\n"
            "  ${_rocke_untimed_entries}")
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
