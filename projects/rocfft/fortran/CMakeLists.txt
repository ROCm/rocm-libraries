# Fortran bindings for rocFFT.
#
# rocfft.F90 is GENERATED from the rocFFT C headers by the rocm-fortran generator
# and committed here, next to the library it binds, so that a header change and
# its binding travel in the same pull request. Do not edit it by hand: regenerate
# it instead.
#
# The bindings are a single self-contained Fortran module, `rocfft`, carrying the
# interfaces, the enum constants, the opaque handle types and the rocfftCheck
# status helper. A consumer writes one `use rocfft`.
#
# This directory is self-contained: it builds with no dependency on the rocFFT C
# target, because a Fortran binding compiles against nothing (it only declares
# bind(C) interfaces). librocfft is required to LINK a final executable, and is
# carried as the target's link interface for that reason.

cmake_minimum_required(VERSION 3.19)

# Configured directly (not as a subdirectory of the rocFFT build), this directory
# is its own project, so that the bindings can be built and packaged on their
# own. In-tree, rocFFT's own project() is already in effect.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(rocfft-fortran LANGUAGES NONE)
endif()

# ON by default but GUARDED: a site with no Fortran compiler must still configure
# and build the C library. The monorepo root already probes once and publishes
# ROCM_HAVE_FORTRAN; a standalone rocFFT build falls back to its own probe.
option(BUILD_FORTRAN_BINDINGS "Build the rocFFT Fortran bindings" ON)

# The bindings' own test suite. OFF by default and guarded the same way: asking
# for the tests when the bindings were not built is a skip, not an error.
option(BUILD_FORTRAN_TESTS "Build the rocFFT Fortran binding tests" OFF)

# BUILD_FORTRAN_BINDINGS and BUILD_FORTRAN_TESTS are ONE cache entry shared by
# every library in the monorepo, which is what you want for the common case:
# -DBUILD_FORTRAN_BINDINGS=OFF from the top turns the whole Fortran track off in
# a single flag. It is not what you want when one library has to be singled out,
# so each library also honours a name of its own, which wins when it is set.
if(DEFINED ROCFFT_BUILD_FORTRAN_BINDINGS)
  set(BUILD_FORTRAN_BINDINGS ${ROCFFT_BUILD_FORTRAN_BINDINGS})
endif()
if(DEFINED ROCFFT_BUILD_FORTRAN_TESTS)
  set(BUILD_FORTRAN_TESTS ${ROCFFT_BUILD_FORTRAN_TESTS})
endif()

# Which array-argument overloads the module exposes. One tri-state replaces the
# two booleans hipFORT used (USE_FPOINTER_INTERFACES / ASSUMED_RANK), because the
# two upper tiers cannot legally coexist: an assumed-rank dummy is not
# distinguishable by rank from the per-rank specifics, so assumed-rank REPLACES
# the per-rank variants rather than adding to them.
#
#   none           the raw type(c_ptr) surface only (Fortran 2003)
#   assumed-shape  + the per-rank array overloads (Fortran 2008) -- the default,
#                  and what the bindings shipped in ROCm are built with
#   assumed-rank   + the single dimension(..) overload per routine (Fortran 2018)
set(FORTRAN_ARRAY_INTERFACES "assumed-shape" CACHE STRING
    "Array-argument overloads to expose: none, assumed-shape, or assumed-rank")
set_property(CACHE FORTRAN_ARRAY_INTERFACES PROPERTY STRINGS
             none assumed-shape assumed-rank)

# Report to the parent scope whether the bindings were really built. Distinct
# from ROCM_HAVE_FORTRAN, which reports only that a compiler exists: the
# compiler can be present while this directory is switched off.
# Only when there IS a parent: standalone, this directory is the top-level scope
# and PARENT_SCOPE warns under -Wdev rather than being a silent no-op.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(ROCFFT_HAVE_FORTRAN_BINDINGS FALSE PARENT_SCOPE)
endif()

if(NOT BUILD_FORTRAN_BINDINGS)
  return()
endif()

# Probing for a Fortran compiler is not free, and every library in a monorepo
# build would otherwise repeat it, so the answer is computed ONCE and published
# as ROCM_HAVE_FORTRAN. That name is deliberately neither rocm-libraries' nor
# rocm-systems': a library from one monorepo is regularly built inside the other,
# and a repo-specific name would make it read the wrong variable, or none. Here
# it is this repo's root that publishes it on the way in; rocm-systems, which has
# no root CMakeLists, publishes the same name from shared/cmake/ROCmFortran.cmake.
#
# ROCM_LIBS_HAVE_FORTRAN is what this repo's root publishes TODAY
# (CMakeLists.txt:11), so it is honoured as a legacy spelling -- second, never
# over the new name -- until the root is switched over. It is a bridge, not an
# alias: once the root sets ROCM_HAVE_FORTRAN this branch is dead and should go.
#
# The local probe is the standalone path, where there is no root to publish
# anything, and it is also the honest fallback: a directory that can be
# configured on its own has to be able to answer this question on its own.
if(DEFINED ROCM_HAVE_FORTRAN)
  set(_have_fortran ${ROCM_HAVE_FORTRAN})
elseif(DEFINED ROCM_LIBS_HAVE_FORTRAN)
  set(_have_fortran ${ROCM_LIBS_HAVE_FORTRAN})
else()
  include(CheckLanguage)
  check_language(Fortran)
  if(CMAKE_Fortran_COMPILER)
    set(_have_fortran TRUE)
  else()
    set(_have_fortran FALSE)
  endif()
endif()

if(NOT _have_fortran)
  message(STATUS "rocFFT: no Fortran compiler found, skipping the Fortran bindings")
  return()
endif()

enable_language(Fortran)

if(NOT FORTRAN_ARRAY_INTERFACES MATCHES "^(none|assumed-shape|assumed-rank)$")
  message(FATAL_ERROR
    "FORTRAN_ARRAY_INTERFACES must be none, assumed-shape or assumed-rank, "
    "not '${FORTRAN_ARRAY_INTERFACES}'.")
endif()

add_library(rocfft_fortran STATIC rocfft.F90)
# roc:: is the namespace rocFFT already exports roc::rocfft under, so the
# binding is reachable the same way the C library is.
add_library(roc::rocfft_fortran ALIAS rocfft_fortran)

# A .mod is not portable across compilers, so the artifacts are installed under a
# per-compiler directory rather than a shared one. Two compilers can then be
# installed side by side from the same package.
#
# The directory is named after the compiler a user actually types (amdflang,
# gfortran), not after CMake's compiler ID, so that the -I/-L paths in the
# documentation are the ones a raw (non-CMake) link line needs. Deriving it from
# the ID and mapping back, rather than taking the basename of
# CMAKE_Fortran_COMPILER, is what makes that survive the cases where the two
# differ: an MPI wrapper (mpif90 around gfortran), a versioned binary
# (gfortran-13), an absolute path, or the Cray wrapper (ftn). All four would give
# the wrong directory under a plain basename, and the .mod would then be filed
# under a name no consumer looks up.
#
# This mapping MUST mirror the one in rocfft-fortran-config.cmake.in exactly.
# That file recomputes the same name for the CONSUMER's compiler and compares
# the two, and the two cannot share code -- the config runs in a consumer's
# build, long after this one is gone. Let them drift and the config rejects
# every consumer with a compiler mismatch that does not exist.
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  set(_fc_name "gfortran")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Flang")     # Flang, LLVMFlang
  set(_fc_name "amdflang")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "Cray")
  set(_fc_name "ftn")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "IntelLLVM")
  set(_fc_name "ifx")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
  set(_fc_name "ifort")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
  set(_fc_name "nvfortran")
else()
  # Unknown compiler: fall back to the basename, lowercased, which is the best
  # guess available and still keeps two compilers from colliding.
  get_filename_component(_fc_name "${CMAKE_Fortran_COMPILER}" NAME_WE)
  string(TOLOWER "${_fc_name}" _fc_name)
endif()

# Strictly a .mod is tied to the compiler VERSION too, not just the compiler, so
# this name is deliberately lossy: gfortran 13 and gfortran 14 install to the
# same directory and the second overwrites the first. That is the documented
# layout (include/fortran/gfortran), and it is the right default because the
# overwhelmingly common case is one Fortran compiler per prefix. A site that
# really does install several versions of the SAME compiler side by side sets
# ROCFFT_FORTRAN_COMPILER_DIR itself, for example to gfortran-13.3.0.
set(ROCFFT_FORTRAN_COMPILER_DIR "${_fc_name}" CACHE STRING
    "Per-compiler subdirectory the .mod and the archive install under")

set(_moddir "${CMAKE_CURRENT_BINARY_DIR}/include")
set_target_properties(rocfft_fortran PROPERTIES
  Fortran_MODULE_DIRECTORY "${_moddir}"
  # rocfft.F90 is uppercase: the array overloads sit behind preprocessor guards,
  # so the source must go through the C preprocessor.
  Fortran_PREPROCESS ON
  # Free form is the .F90 default, but stating it lets CMake emit whichever flag
  # the compiler in use expects rather than relying on the extension.
  Fortran_FORMAT FREE
  POSITION_INDEPENDENT_CODE ON)

# One section per procedure, so that a consumer linking this archive can discard
# the wrappers it does not call.
#
# This is not a size optimisation, it is what keeps the archive usable against an
# OLDER librocfft than the one these bindings were generated from. The module is
# built from a single source into a single object, and a static archive links at
# object granularity, so pulling in one wrapper pulls in every undefined C symbol
# the object references -- including any rocFFT entry point added since the
# installed library was built. A consumer who never touches the new API would
# still fail to link, with undefined references to functions they have never
# heard of. rocFFT's C API is small and stable enough that every symbol this
# module wraps is present in a current librocfft, so nothing is broken without
# the flags today; the point is that it stays unbroken the first time an entry
# point is added, without anyone having to remember why.
#
# With per-function sections plus --gc-sections below, the unreferenced wrappers
# are dropped and the link succeeds. Guarded on the compilers known to accept the
# spelling; others simply keep the current behaviour, which is correct whenever
# the installed library matches.
if(CMAKE_Fortran_COMPILER_ID MATCHES "^(GNU|Flang|LLVMFlang|Intel|IntelLLVM)$")
  target_compile_options(rocfft_fortran PRIVATE -ffunction-sections -fdata-sections)
  target_link_options(rocfft_fortran INTERFACE
    "$<$<LINK_LANG_AND_ID:Fortran,GNU,Flang,LLVMFlang,Intel,IntelLLVM>:LINKER:--gc-sections>")
endif()

# Translate the tri-state into the two macros the generated source is guarded
# with. assumed-rank implies the F2008 tier: in the source the assumed-rank
# specific is nested inside USE_FPOINTER_INTERFACES, and selects itself over the
# per-rank ones.
if(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-shape")
  target_compile_definitions(rocfft_fortran PRIVATE USE_FPOINTER_INTERFACES=1)
elseif(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-rank")
  target_compile_definitions(rocfft_fortran PRIVATE
    USE_FPOINTER_INTERFACES=1 USE_ASSUMED_RANK_INTERFACES=1)
endif()

# No line-length flag: the generator wraps the emitted source at 112 columns,
# inside the 132 that free-form Fortran guarantees, so every compiler reads it
# without a vendor-specific option.

target_include_directories(rocfft_fortran PUBLIC
  "$<BUILD_INTERFACE:${_moddir}>"
  "$<INSTALL_INTERFACE:include/fortran/${ROCFFT_FORTRAN_COMPILER_DIR}>")

# Link interface only: needed to link an executable, not to build this archive.
# In-tree roc::rocfft comes from rocFFT's own library/src. Standalone it has to
# be looked for, and testing for the target without ever calling find_package
# would mean it is never found: the archive would then export no link interface
# at all, and every consumer would hit undefined references to rocfft_setup and
# friends despite linking roc::rocfft_fortran exactly as documented.
#
# rocfft-config.cmake does find_dependency(hip), and hip-config.cmake reaches
# FindThreads, which hard-errors unless C or CXX is enabled. This directory has
# only Fortran, so C is enabled first -- it compiles none of the sources here.
# QUIET, because a site with no installed ROCm can still legitimately build the
# bindings: they declare bind(C) interfaces and compile against nothing.
if(NOT TARGET roc::rocfft)
  if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
    enable_language(C)
  endif()
  find_package(rocfft QUIET)
endif()

if(TARGET roc::rocfft)
  target_link_libraries(rocfft_fortran INTERFACE roc::rocfft)
else()
  message(STATUS
    "rocFFT: librocfft was not found, so the Fortran bindings carry no link "
    "interface; a consumer will have to link it (-lrocfft) itself.")
endif()

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(ROCFFT_HAVE_FORTRAN_BINDINGS TRUE PARENT_SCOPE)
endif()

if(BUILD_FORTRAN_TESTS)
  # enable_testing() here is enough for a STANDALONE build of this directory, but
  # NOT for an in-tree one: CMake only writes the chain of CTestTestfile.cmake
  # that ctest walks down from a build root when testing was enabled at the TOP
  # level. Enabled only here, the tests are built and registered, and `ctest` run
  # from the top of the parent build reports "Total Tests: 0" -- a silent
  # nothing, which is the worst way to find out. So say it.
  enable_testing()
  if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    get_property(_top_testing GLOBAL PROPERTY CTEST_TARGETS_ADDED)
    if(NOT _top_testing)
      message(STATUS
        "rocFFT: the Fortran binding tests are registered, but ctest discovers "
        "tests only from a build root where testing was enabled at the top "
        "level. If `ctest` there reports no tests, add include(CTest) or "
        "enable_testing() to the top-level CMakeLists, or run ctest from "
        "${CMAKE_CURRENT_BINARY_DIR}/test.")
    endif()
  endif()
  add_subdirectory(test)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS rocfft_fortran
        EXPORT rocfft-fortran-targets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/fortran/${ROCFFT_FORTRAN_COMPILER_DIR}")
install(DIRECTORY "${_moddir}/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fortran/${ROCFFT_FORTRAN_COMPILER_DIR}"
        FILES_MATCHING PATTERN "*.mod")

# The generated source, so that a user on a compiler ROCm ships no .mod for can
# rebuild the binding without fetching a monorepo to get one file. Source
# obtained separately can also be generated from a different ROCm than the
# installed librocfft, which surfaces as an undefined symbol at link time or as
# an interface that silently differs from the one the caller meant; source that
# ships with the library cannot drift from it.
#
# Under share/, not include/. The .mod is compiler-specific and this is not, and
# every entry under include/fortran/ is partitioned by compiler, so a source file
# there would be read as naming one: the config below enumerates that directory
# to report which compilers are installed, and would list rocfft.F90 among them.
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/rocfft.F90"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/rocfft/fortran")
install(EXPORT rocfft-fortran-targets
        FILE rocfft-fortran-targets.cmake
        NAMESPACE roc::
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/rocfft")

# lib/cmake/rocfft is where rocFFT's own rocm_export_targets() puts
# rocfft-config.cmake, so the Fortran package lands beside the C one and a single
# CMAKE_PREFIX_PATH entry finds both.
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/rocfft-fortran-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/rocfft-fortran-config.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/rocfft")
# No PATH_VARS: it only creates PACKAGE_<var>, and the template interpolates
# @ROCFFT_FORTRAN_COMPILER_DIR@ directly. Passing it was a dead argument, and
# worse than dead if anyone later switched the template to the PACKAGE_ form,
# since PATH_VARS would then resolve this relative value as a path under the
# install prefix.
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rocfft-fortran-config.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/rocfft")

# That placement is what lets rocfft-config.cmake pick the bindings up with a
# bare include(rocfft-fortran-config.cmake OPTIONAL): a Fortran binding installed
# after the C library is then found at find_package(rocfft) time, and ignored
# when absent.
#
# But find_package(rocfft-fortran) -- the other documented spelling -- searches
# lib/cmake/rocfft-fortran*, which lib/cmake/rocfft does not match, so on its own
# that placement makes the direct spelling fail with "could not find a package
# configuration file". A forwarder gives the search somewhere to land without
# duplicating the real config, which would also need its own copy of the targets
# file and would then be a second thing to keep in step.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/forward/rocfft-fortran-config.cmake"
"# Generated forwarder. The real config lives beside the rocFFT C package, in
# lib/cmake/rocfft, so that rocfft-config.cmake can include it as a sibling. This
# file only exists so that find_package(rocfft-fortran) resolves to the same
# place.
include(\"\${CMAKE_CURRENT_LIST_DIR}/../rocfft/rocfft-fortran-config.cmake\")
")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/forward/rocfft-fortran-config.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/rocfft-fortran")

# A version file, so that find_package(rocfft-fortran 1.0) resolves rather than
# failing with "could not find a configuration file compatible with requested
# version". SameMajorVersion because the module's bind(C) surface tracks the C
# API's compatibility promise.
#
# The version is read from THIS tree's top-level CMakeLists, not from
# rocfft_VERSION. That variable looks like the obvious source and is a trap: in a
# standalone build it is set by the find_package(rocfft) above, so it reports
# whatever rocFFT happens to be INSTALLED rather than the version of the headers
# these bindings were generated from. The bindings ship with their source tree,
# so the source tree is the truth.
set(_fortran_version "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt")
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt" _rocfft_root)
  # rocFFT does NOT spell the version as a literal anywhere: it sets the three
  # components and composes VERSION_STRING from them, so a regex looking for
  # "x.y.z" matches nothing. Read the components and recompose. Note the spaces
  # inside the parentheses -- rocFFT writes set ( ROCFFT_VERSION_MAJOR "1" ).
  #
  # Success is judged on the whole match being non-empty, NOT on the captured
  # group being truthy: the minor component is the string "0", one of CMake's
  # false constants, so if(NOT CMAKE_MATCH_1) treats a perfectly good match as a
  # failure and no version file is installed at all.
  set(_rocfft_parts "")
  foreach(_part MAJOR MINOR PATCH)
    string(REGEX MATCH "set[ \t]*\\([ \t]*ROCFFT_VERSION_${_part}[ \t]+\"?([0-9]+)"
           _m "${_rocfft_root}")
    if(_m STREQUAL "")
      set(_rocfft_parts "")
      break()
    endif()
    list(APPEND _rocfft_parts "${CMAKE_MATCH_1}")
  endforeach()
  if(_rocfft_parts)
    list(JOIN _rocfft_parts "." _fortran_version)
  endif()
  unset(_rocfft_parts)
  unset(_rocfft_root)
endif()

if(_fortran_version)
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/rocfft-fortran-config-version.cmake"
    VERSION "${_fortran_version}"
    COMPATIBILITY SameMajorVersion)
  # Installed in BOTH package directories, and it has to be. CMake looks for
  # <pkg>-config-version.cmake NEXT TO the <pkg>-config.cmake it actually
  # resolved, not wherever the real one lives. find_package(rocfft-fortran 1.0)
  # resolves the forwarder in lib/cmake/rocfft-fortran, so a version file only
  # beside the real config in lib/cmake/rocfft is never consulted, and the call
  # fails with "could not find a configuration file compatible with requested
  # version" even though the config is right there and the version matches.
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rocfft-fortran-config-version.cmake"
          DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/rocfft")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rocfft-fortran-config-version.cmake"
          DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/rocfft-fortran")
else()
  # Better no version file than a wrong one: without it an unversioned
  # find_package still succeeds, whereas a version file claiming the wrong number
  # makes a versioned one succeed and link the wrong .mod.
  message(STATUS
    "rocFFT: could not determine the version for the Fortran package, so no "
    "rocfft-fortran-config-version.cmake is installed; "
    "find_package(rocfft-fortran <version>) will not match.")
endif()
