# Fortran bindings for hipFFT and hipFFTW.
#
# hipfft.F90 and hipfftw.F90 are GENERATED from the hipFFT and hipFFTW C headers
# by the rocm-fortran generator and committed here, next to the libraries they
# bind, so that a header change and its binding travel in the same pull request.
# Do not edit them by hand: regenerate them instead.
#
# Each binding is a single self-contained Fortran module. `hipfft` carries the
# interfaces, the enum constants (HIPFFT_Z2Z, HIPFFT_FORWARD, ...) and the
# hipfftCheck status helper; `hipfftw` carries the FFTW compatibility surface,
# the FFTW_* flag constants and the fftw_iodim/fftw_iodim64 derived types. A
# consumer writes one `use hipfft` or one `use hipfftw`.
#
# TWO targets in ONE directory, because that is how the libraries themselves are
# built: projects/hipfft/library compiles both libhipfft and libhipfftw from one
# source tree and ships both in the hipfft package. The bindings follow the
# libraries -- they are packaged, installed and found together -- so there is one
# CMakeLists here and not two.
#
# The two modules stay independent all the same: hipfftw.F90 contains no
# `use hipfft`, so the two archives have no compile-time relationship and no
# ordering constraint, and a program on the FFTW compatibility API links
# libhipfftw alone without ever touching libhipfft.
#
# This directory is self-contained: it builds with no dependency on the hipFFT C
# targets, because a Fortran binding compiles against nothing (it only declares
# bind(C) interfaces). libhipfft/libhipfftw are required to LINK a final
# executable, and are carried as the targets' link interfaces for that reason.

cmake_minimum_required(VERSION 3.19)

# Configured directly (not as a subdirectory of the hipFFT build), this
# directory is its own project, so that the bindings can be built and packaged
# on their own. In-tree, hipFFT's own project() is already in effect.
#
# LANGUAGES NONE, not C: nothing here is written in C, and Fortran is enabled
# further down, only once the build is known to be wanted. Locating the
# installed hipFFT does need a C (or CXX) compiler -- find_package(hipfft)
# find_dependency(hip)s, and hip-config-amd.cmake reaches FindThreads, which
# hard-errors with neither language enabled -- but that is a property of the
# lookup, not of this project, so C is enabled at each lookup site instead,
# guarded on neither language being loaded already. Enabling it here would make
# a C compiler a hard requirement of a build that does not compile a line of C.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(hipfft-fortran LANGUAGES NONE)
endif()

# ON by default but GUARDED: a site with no Fortran compiler must still
# configure and build the C libraries. The monorepo root already probes once and
# publishes ROCM_HAVE_FORTRAN; a standalone hipFFT build falls back to its own
# probe.
option(BUILD_FORTRAN_BINDINGS "Build the hipFFT and hipFFTW Fortran bindings" ON)

# The bindings' own test suites. OFF by default and guarded the same way: asking
# for the tests when the bindings were not built is a skip, not an error.
option(BUILD_FORTRAN_TESTS "Build the hipFFT and hipFFTW Fortran binding tests" OFF)

# BUILD_FORTRAN_BINDINGS and BUILD_FORTRAN_TESTS are ONE cache entry shared by
# every library in the monorepo, which is what you want for the common case:
# -DBUILD_FORTRAN_BINDINGS=OFF from the top turns the whole Fortran track off in
# a single flag. It is not what you want when one library has to be singled out,
# so each library also honours a name of its own, which wins when it is set.
#
# This directory builds TWO libraries' bindings, so it honours two pairs:
# HIPFFT_* selects the hipfft target, HIPFFTW_* the hipfftw one. Either can be
# switched off without the other -- they are separate targets binding separate
# shared libraries, and a site that wants only the FFTW compatibility surface
# should not have to build both.
set(_hipfft_bindings  ${BUILD_FORTRAN_BINDINGS})
set(_hipfftw_bindings ${BUILD_FORTRAN_BINDINGS})
set(_hipfft_tests     ${BUILD_FORTRAN_TESTS})
set(_hipfftw_tests    ${BUILD_FORTRAN_TESTS})
if(DEFINED HIPFFT_BUILD_FORTRAN_BINDINGS)
  set(_hipfft_bindings ${HIPFFT_BUILD_FORTRAN_BINDINGS})
endif()
if(DEFINED HIPFFTW_BUILD_FORTRAN_BINDINGS)
  set(_hipfftw_bindings ${HIPFFTW_BUILD_FORTRAN_BINDINGS})
endif()
if(DEFINED HIPFFT_BUILD_FORTRAN_TESTS)
  set(_hipfft_tests ${HIPFFT_BUILD_FORTRAN_TESTS})
endif()
if(DEFINED HIPFFTW_BUILD_FORTRAN_TESTS)
  set(_hipfftw_tests ${HIPFFTW_BUILD_FORTRAN_TESTS})
endif()

# Which array-argument overloads the modules expose. One tri-state replaces the
# two booleans hipFORT used (USE_FPOINTER_INTERFACES / ASSUMED_RANK), because the
# two upper tiers cannot legally coexist: an assumed-rank dummy is not
# distinguishable by rank from the per-rank specifics, so assumed-rank REPLACES
# the per-rank variants rather than adding to them.
#
#   none           the raw type(c_ptr) surface only (Fortran 2003)
#   assumed-shape  + the per-rank array overloads (Fortran 2008) -- the default,
#                  and what the bindings shipped in ROCm are built with
#   assumed-rank   + the single dimension(..) overload per routine (Fortran 2018)
#
# hipFFTW is the one binding where the three tiers currently coincide: the FFTW
# compatibility API takes bare pointers to already-allocated buffers, so the
# generator emitted no array overloads and no preprocessor guards in
# hipfftw.F90. The option is still applied to both targets, so that a
# regenerated, guarded hipfftw.F90 needs no CMake change.
set(FORTRAN_ARRAY_INTERFACES "assumed-shape" CACHE STRING
    "Array-argument overloads to expose: none, assumed-shape, or assumed-rank")
set_property(CACHE FORTRAN_ARRAY_INTERFACES PROPERTY STRINGS
             none assumed-shape assumed-rank)

# Report to the parent scope whether each binding was really built. Distinct
# from ROCM_HAVE_FORTRAN, which reports only that a compiler exists: the
# compiler can be present while either half is switched off.
# Only when there IS a parent: standalone, this directory is the top-level scope
# and PARENT_SCOPE warns under -Wdev rather than being a silent no-op.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(HIPFFT_HAVE_FORTRAN_BINDINGS  FALSE PARENT_SCOPE)
  set(HIPFFTW_HAVE_FORTRAN_BINDINGS FALSE PARENT_SCOPE)
endif()

if(NOT _hipfft_bindings AND NOT _hipfftw_bindings)
  return()
endif()

# Probing for a Fortran compiler is not free, and every library in a monorepo
# build would otherwise repeat it, so the monorepo root probes ONCE and publishes
# the answer as ROCM_HAVE_FORTRAN. That name is deliberately neither
# rocm-libraries' nor rocm-systems': a library from one monorepo is regularly
# built inside the other, and a repo-specific name would have each side reading a
# variable the other never sets. ROCM_LIBS_HAVE_FORTRAN is what rocm-libraries
# publishes today and is honoured as a legacy spelling until its root adopts the
# new name.
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
  message(STATUS "hipFFT: no Fortran compiler found, skipping the Fortran bindings")
  return()
endif()

enable_language(Fortran)

if(NOT FORTRAN_ARRAY_INTERFACES MATCHES "^(none|assumed-shape|assumed-rank)$")
  message(FATAL_ERROR
    "FORTRAN_ARRAY_INTERFACES must be none, assumed-shape or assumed-rank, "
    "not '${FORTRAN_ARRAY_INTERFACES}'.")
endif()

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
# THIS MAPPING MUST MIRROR the one in hipfft-fortran-config.cmake.in and the one
# in hipfftw-fortran-config.cmake.in exactly. The three cannot share code -- the
# configs run in a consumer's build, long after this one is gone -- so they are
# kept in step by hand. Get it wrong and the check in the configs never matches:
# the packages then reject every consumer with a message claiming a compiler
# mismatch that does not exist.
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
# HIPFFT_FORTRAN_COMPILER_DIR itself, for example to gfortran-13.3.0.
set(HIPFFT_FORTRAN_COMPILER_DIR "${_fc_name}" CACHE STRING
    "Per-compiler subdirectory the .mod files and archives install under")

# hipFFTW installs into the same layout, out of the same package, for the same
# compiler, so it does not get a cache entry of its own: two overridable names
# for one directory could only ever disagree, and disagreeing would scatter one
# compiler's .mod files across two directories. The mirror exists only so that
# hipfftw-fortran-config.cmake.in can substitute a name of its own.
set(HIPFFTW_FORTRAN_COMPILER_DIR "${HIPFFT_FORTRAN_COMPILER_DIR}")

# One module directory for both targets. hipfft.mod and hipfftw.mod are distinct
# names and neither module `use`s the other, so there is no clash and no ordering
# constraint, and the single install(DIRECTORY) below is correct whichever halves
# were built: only the .mod files of the targets that really compiled exist here.
set(_moddir "${CMAKE_CURRENT_BINARY_DIR}/include")

# Locate the C libraries ONCE, for both halves: the installed
# hipfft-targets.cmake defines hip::hipfft and hip::hipfftw together, so a single
# lookup settles both. In-tree they come from this project's library/ directory
# and the lookup is skipped entirely.
#
# Testing for the targets without ever calling find_package -- which is what this
# used to do -- means they are never found in a standalone build: the archives
# would then export no link interface at all, and every consumer would hit
# undefined references to hipfftPlan1d and friends despite linking
# hip::hipfft_fortran exactly as documented.
#
# hipfft-config.cmake find_dependency(hip)s, and hip-config-amd.cmake reaches
# FindThreads, which hard-errors unless C or CXX is enabled -- and QUIET does not
# suppress a FATAL_ERROR. A Fortran-only configure of this directory has neither,
# so C is enabled first. QUIET, because a site with no installed ROCm can still
# legitimately build the bindings: they compile against nothing.
if(NOT TARGET hip::hipfft AND NOT TARGET hip::hipfftw)
  if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
    enable_language(C)
  endif()
  find_package(hipfft QUIET)
endif()

if(NOT TARGET hip::hipfft AND NOT TARGET hip::hipfftw)
  message(STATUS
    "hipFFT: neither hip::hipfft nor hip::hipfftw was found, so the Fortran "
    "bindings carry no link interface; a consumer will have to link the C "
    "library (-lhipfft / -lhipfftw) itself. Point CMAKE_PREFIX_PATH at an "
    "installed ROCm to have it recorded.")
endif()

# --------------------------------------------------------------------------
# The hipFFT binding: module `hipfft`, binding libhipfft.
# --------------------------------------------------------------------------
if(_hipfft_bindings)
  add_library(hipfft_fortran STATIC hipfft.F90)
  # hipFFT exports its C targets under the hip:: namespace (hip::hipfft,
  # hip::hipfftw), not roc::, so the bindings follow the namespace this project
  # already publishes rather than inventing a second one.
  add_library(hip::hipfft_fortran ALIAS hipfft_fortran)

  set_target_properties(hipfft_fortran PROPERTIES
    Fortran_MODULE_DIRECTORY "${_moddir}"
    # hipfft.F90 is uppercase: the array overloads sit behind preprocessor
    # guards, so the source must go through the C preprocessor.
    Fortran_PREPROCESS ON
    # Free form is the .F90 default, but stating it lets CMake emit whichever
    # flag the compiler in use expects rather than relying on the extension.
    Fortran_FORMAT FREE
    POSITION_INDEPENDENT_CODE ON)

  # One section per procedure, so that a consumer linking this archive can
  # discard the wrappers it does not call.
  #
  # This is not a size optimisation, it is what makes the archive usable against
  # an older libhipfft at all. The module is generated from THIS tree's headers,
  # so it wraps whatever API this tree has and a released ROCm may not yet. The
  # archive holds a single object, and a static archive links at object
  # granularity, so pulling in one wrapper pulls in every undefined C symbol the
  # object references -- including the ones the installed libhipfft does not
  # export. A consumer who never touches the new API still fails to link, with
  # undefined references to functions they have never heard of.
  #
  # With per-function sections plus --gc-sections below, the unreferenced
  # wrappers are dropped and the link succeeds.
  #
  # Guarded on the compilers known to accept the spelling; others simply keep the
  # current behaviour, which is correct whenever the installed library matches.
  if(CMAKE_Fortran_COMPILER_ID MATCHES "^(GNU|Flang|LLVMFlang|Intel|IntelLLVM)$")
    target_compile_options(hipfft_fortran PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(hipfft_fortran INTERFACE
      "$<$<LINK_LANG_AND_ID:Fortran,GNU,Flang,LLVMFlang,Intel,IntelLLVM>:LINKER:--gc-sections>")
  endif()

  # Translate the tri-state into the two macros the generated source is guarded
  # with. assumed-rank implies the F2008 tier: in the source the assumed-rank
  # specific is nested inside USE_FPOINTER_INTERFACES, and selects itself over
  # the per-rank ones.
  if(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-shape")
    target_compile_definitions(hipfft_fortran PRIVATE USE_FPOINTER_INTERFACES=1)
  elseif(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-rank")
    target_compile_definitions(hipfft_fortran PRIVATE
      USE_FPOINTER_INTERFACES=1 USE_ASSUMED_RANK_INTERFACES=1)
  endif()

  # No line-length flag: the generator wraps the emitted source at 112 columns,
  # inside the 132 that free-form Fortran guarantees, so every compiler reads it
  # without a vendor-specific option.

  target_include_directories(hipfft_fortran PUBLIC
    "$<BUILD_INTERFACE:${_moddir}>"
    "$<INSTALL_INTERFACE:include/fortran/${HIPFFT_FORTRAN_COMPILER_DIR}>")

  # Link interface only: needed to link an executable, not to build this archive.
  if(TARGET hip::hipfft)
    target_link_libraries(hipfft_fortran INTERFACE hip::hipfft)
    # Recorded so the config package knows whether the exported target actually
    # names hip::hipfft in its link interface, and therefore whether it has to
    # find_dependency(hipfft) before including the targets file. A bindings-only
    # install made with no hipFFT in sight must not demand one at consume time.
    set(HIPFFT_FORTRAN_NEEDS_HIPFFT_PACKAGE TRUE)
  else()
    set(HIPFFT_FORTRAN_NEEDS_HIPFFT_PACKAGE FALSE)
  endif()

  if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(HIPFFT_HAVE_FORTRAN_BINDINGS TRUE PARENT_SCOPE)
  endif()
endif()

# --------------------------------------------------------------------------
# The hipFFTW binding: module `hipfftw`, binding libhipfftw.
# --------------------------------------------------------------------------
if(_hipfftw_bindings)
  add_library(hipfftw_fortran STATIC hipfftw.F90)
  # hip:: and not roc::, because that is the namespace this project already
  # exports hipfftw itself under (hip::hipfftw). A consumer should not have to
  # remember that the binding and the library it binds sit in different
  # namespaces.
  add_library(hip::hipfftw_fortran ALIAS hipfftw_fortran)

  set_target_properties(hipfftw_fortran PROPERTIES
    Fortran_MODULE_DIRECTORY "${_moddir}"
    # hipfftw.F90 is uppercase for consistency with the other generated
    # bindings, whose array overloads sit behind preprocessor guards. This one
    # currently has no guards, but preprocessing it costs nothing and keeps the
    # file regenerable into a guarded form without a CMake change.
    Fortran_PREPROCESS ON
    Fortran_FORMAT FREE
    POSITION_INDEPENDENT_CODE ON)

  # Per-procedure sections and --gc-sections for the consumer, for the same
  # reason as hipfft_fortran above: this archive is one object, so without them a
  # consumer who links it drags in every C symbol the whole module references,
  # and fails on the ones their installed libhipfftw does not export.
  if(CMAKE_Fortran_COMPILER_ID MATCHES "^(GNU|Flang|LLVMFlang|Intel|IntelLLVM)$")
    target_compile_options(hipfftw_fortran PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(hipfftw_fortran INTERFACE
      "$<$<LINK_LANG_AND_ID:Fortran,GNU,Flang,LLVMFlang,Intel,IntelLLVM>:LINKER:--gc-sections>")
  endif()

  # Defined even though hipfftw.F90 uses neither macro today, so that a
  # regenerated, guarded source needs no change here.
  if(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-shape")
    target_compile_definitions(hipfftw_fortran PRIVATE USE_FPOINTER_INTERFACES=1)
  elseif(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-rank")
    target_compile_definitions(hipfftw_fortran PRIVATE
      USE_FPOINTER_INTERFACES=1 USE_ASSUMED_RANK_INTERFACES=1)
  endif()

  target_include_directories(hipfftw_fortran PUBLIC
    "$<BUILD_INTERFACE:${_moddir}>"
    "$<INSTALL_INTERFACE:include/fortran/${HIPFFTW_FORTRAN_COMPILER_DIR}>")

  # hip::hipfftw and never hip::hipfft: the FFTW compatibility surface is its own
  # shared library, and a program on it links libhipfftw alone.
  if(TARGET hip::hipfftw)
    target_link_libraries(hipfftw_fortran INTERFACE hip::hipfftw)
    set(HIPFFTW_FORTRAN_NEEDS_HIPFFT_PACKAGE TRUE)
  else()
    set(HIPFFTW_FORTRAN_NEEDS_HIPFFT_PACKAGE FALSE)
  endif()

  if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(HIPFFTW_HAVE_FORTRAN_BINDINGS TRUE PARENT_SCOPE)
  endif()
endif()

if(_hipfft_tests OR _hipfftw_tests)
  # enable_testing() here is enough for a STANDALONE build of this directory, but
  # NOT for an in-tree one: CMake only writes the chain of CTestTestfile.cmake
  # that ctest walks down from a build root when testing was enabled at the TOP
  # level. hipFFT's own enable_testing() sits inside its BUILD_CLIENTS_* guard,
  # so a build that asks for the Fortran bindings and nothing else does not get
  # it. Enabled only here, the tests are built and registered, and `ctest` run
  # from the top of the parent build reports "Total Tests: 0" -- a silent
  # nothing, which is the worst way to find out. So say it.
  enable_testing()
  if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    get_property(_top_testing GLOBAL PROPERTY CTEST_TARGETS_ADDED)
    if(NOT _top_testing)
      message(STATUS
        "hipFFT: the Fortran binding tests are registered, but ctest discovers "
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

# The version both Fortran packages are stamped with, so that
# find_package(hipfft-fortran 1.0) and find_package(hipfftw-fortran 1.0) resolve
# rather than failing with "could not find a configuration file compatible with
# requested version".
#
# ONE version for both: hipFFTW has no version of its own, it is built from the
# hipFFT source tree and shipped in the hipFFT package, so its binding carries
# hipFFT's number.
#
# Read from THIS tree's top-level CMakeLists, not from hipfft_VERSION. That
# variable looks like the obvious source and is a trap: it is set by the
# find_package(hipfft) above, so it reports whatever hipFFT happens to be
# INSTALLED rather than the version of the headers these bindings were generated
# from. The bindings ship with their source tree, so the source tree is the
# truth.
#
# hipFFT does not spell the version as a literal: VERSION_STRING is assembled
# from three components, so the three are matched and joined here rather than
# matching the assignment that references them.
set(_fortran_version "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt")
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt" _hipfft_root)
  set(_v_parts "")
  foreach(_v_comp MAJOR MINOR PATCH)
    # The whole match, not CMAKE_MATCH_1, decides whether this component was
    # found: a failed string(REGEX MATCH) leaves CMAKE_MATCH_1 holding whatever
    # the last SUCCESSFUL match put there, so reading it unguarded would silently
    # reuse the previous component's number and stamp the package with a version
    # that appears nowhere in the tree.
    string(REGEX MATCH "set[ \t]*\\([ \t]*HIPFFT_VERSION_${_v_comp}[ \t]+\"([0-9]+)\""
           _v_whole "${_hipfft_root}")
    if(_v_whole)
      list(APPEND _v_parts "${CMAKE_MATCH_1}")
    endif()
  endforeach()
  list(LENGTH _v_parts _v_n)
  if(_v_n EQUAL 3)
    list(JOIN _v_parts "." _fortran_version)
  endif()
  unset(_hipfft_root)
endif()

if(NOT _fortran_version)
  # Better no version file than a wrong one: without it an unversioned
  # find_package still succeeds, whereas a version file claiming the wrong number
  # makes a versioned one succeed and link the wrong .mod.
  message(STATUS
    "hipFFT: could not determine the version for the Fortran packages, so no "
    "hipfft-fortran-config-version.cmake or hipfftw-fortran-config-version.cmake "
    "is installed; find_package(hipfft-fortran <version>) and "
    "find_package(hipfftw-fortran <version>) will not match.")
endif()

# One install(DIRECTORY) covers both modules: only the .mod files of the targets
# that were really compiled exist in _moddir.
install(DIRECTORY "${_moddir}/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fortran/${HIPFFT_FORTRAN_COMPILER_DIR}"
        FILES_MATCHING PATTERN "*.mod")

# Two export sets and two config packages, not one of each. The two bindings are
# found independently (find_package(hipfft-fortran) vs
# find_package(hipfftw-fortran)), each config carries its own compiler check and
# its own hip*_FORTRAN_FOUND flag, and a consumer of the FFTW API must not be
# made to import a hipfft binding it never asked for.
#
# Both land in lib/cmake/hipfft: hipFFTW has no config-package directory of its
# own -- it ships inside the hipfft package -- and putting the hipFFT one there
# too is what lets hipfft-config.cmake pick either up with a bare
# include(<name> OPTIONAL) as a sibling.
if(_hipfft_bindings)
  install(TARGETS hipfft_fortran
          EXPORT hipfft-fortran-targets
          ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/fortran/${HIPFFT_FORTRAN_COMPILER_DIR}")
  install(EXPORT hipfft-fortran-targets
          FILE hipfft-fortran-targets.cmake
          NAMESPACE hip::
          DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hipfft")

  # The generated source, so that a user on a compiler ROCm ships no .mod for can
  # rebuild the binding without fetching a monorepo to get one file. Source
  # obtained separately can also be generated from a different ROCm than the
  # installed libhipfft, which surfaces as an undefined symbol at link time or as
  # an interface that silently differs from the one the caller meant; source that
  # ships with the library cannot drift from it.
  #
  # Under share/, not include/. The .mod is compiler-specific and this is not, and
  # every entry under include/fortran/ is partitioned by compiler, so a source
  # file there would be read as naming one: the config below enumerates that
  # directory to report which compilers are installed, and would list hipfft.F90
  # among them.
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/hipfft.F90"
          DESTINATION "${CMAKE_INSTALL_DATADIR}/hipfft/fortran")

  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/hipfft-fortran-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/hipfft-fortran-config.cmake"
    # No PATH_VARS: it only creates PACKAGE_<var>, and the template interpolates
    # @HIPFFT_FORTRAN_COMPILER_DIR@ directly. Passing it was a dead argument, and
    # worse than dead if anyone later switched the template to the PACKAGE_ form,
    # since PATH_VARS would then resolve this relative value as a path under the
    # install prefix.
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hipfft")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hipfft-fortran-config.cmake"
          DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hipfft")

  # find_package(hipfft-fortran) -- the documented spelling -- searches
  # lib/cmake/hipfft-fortran*, which lib/cmake/hipfft does not match, so on its
  # own that placement makes the direct spelling fail with "could not find a
  # package configuration file". A forwarder gives the search somewhere to land
  # without duplicating the real config, which would also need its own copy of
  # the targets file and would then be a second thing to keep in step.
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/forward/hipfft-fortran-config.cmake"
"# Generated forwarder. The real config lives beside the hipFFT C package, in
# lib/cmake/hipfft, so that hipfft-config.cmake can include it as a sibling.
# This file only exists so that find_package(hipfft-fortran) resolves to the
# same place.
include(\"\${CMAKE_CURRENT_LIST_DIR}/../hipfft/hipfft-fortran-config.cmake\")
")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/forward/hipfft-fortran-config.cmake"
          DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hipfft-fortran")

  # SameMajorVersion because the module's bind(C) surface tracks the C API's
  # compatibility promise.
  if(_fortran_version)
    write_basic_package_version_file(
      "${CMAKE_CURRENT_BINARY_DIR}/hipfft-fortran-config-version.cmake"
      VERSION "${_fortran_version}"
      COMPATIBILITY SameMajorVersion)
    # Installed in BOTH package directories, and it has to be. CMake looks for
    # <pkg>-config-version.cmake NEXT TO the <pkg>-config.cmake it actually
    # resolved, not wherever the real one lives. find_package(hipfft-fortran 1.0)
    # resolves the forwarder in lib/cmake/hipfft-fortran, so a version file only
    # beside the real config in lib/cmake/hipfft is never consulted, and the call
    # fails with "could not find a configuration file compatible with requested
    # version" even though the config is right there and the version matches.
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hipfft-fortran-config-version.cmake"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hipfft")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hipfft-fortran-config-version.cmake"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hipfft-fortran")
  endif()
endif()

if(_hipfftw_bindings)
  install(TARGETS hipfftw_fortran
          EXPORT hipfftw-fortran-targets
          ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/fortran/${HIPFFTW_FORTRAN_COMPILER_DIR}")
  install(EXPORT hipfftw-fortran-targets
          FILE hipfftw-fortran-targets.cmake
          NAMESPACE hip::
          DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hipfft")

  # The generated source, for the same reason as hipfft.F90 above, and under the
  # same share/hipfft/fortran: hipFFTW is packaged with hipFFT and has no
  # directory of its own anywhere else in the layout, so it has none here either.
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/hipfftw.F90"
          DESTINATION "${CMAKE_INSTALL_DATADIR}/hipfft/fortran")

  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/hipfftw-fortran-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/hipfftw-fortran-config.cmake"
    # No PATH_VARS, for the same reason as hipfft above.
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hipfft")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hipfftw-fortran-config.cmake"
          DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hipfft")

  # Same forwarder, for the same reason: lib/cmake/hipfft matches neither
  # hipfft-fortran* nor hipfftw-fortran*.
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/forward/hipfftw-fortran-config.cmake"
"# Generated forwarder. hipFFTW ships inside the hipFFT package, so the real
# config lives in lib/cmake/hipfft beside hipfft-config.cmake. This file only
# exists so that find_package(hipfftw-fortran) resolves to the same place.
include(\"\${CMAKE_CURRENT_LIST_DIR}/../hipfft/hipfftw-fortran-config.cmake\")
")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/forward/hipfftw-fortran-config.cmake"
          DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hipfftw-fortran")

  # A version file of its own, carrying hipFFT's number, and installed in both
  # of ITS package directories for the same reason as hipfft's above: the
  # forwarder in lib/cmake/hipfftw-fortran is what find_package resolves, and a
  # version file is only ever looked for beside the config that was resolved.
  if(_fortran_version)
    write_basic_package_version_file(
      "${CMAKE_CURRENT_BINARY_DIR}/hipfftw-fortran-config-version.cmake"
      VERSION "${_fortran_version}"
      COMPATIBILITY SameMajorVersion)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hipfftw-fortran-config-version.cmake"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hipfft")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hipfftw-fortran-config-version.cmake"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hipfftw-fortran")
  endif()
endif()
