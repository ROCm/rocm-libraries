# Fortran bindings for rocSOLVER.
#
# rocsolver.F90 is GENERATED from the rocSOLVER C headers by the rocm-fortran
# generator and committed here, next to the library it binds, so that a header
# change and its binding travel in the same pull request. Do not edit it by
# hand: regenerate it instead.
#
# The bindings are a single Fortran module, `rocsolver`, carrying the interfaces,
# the enum constants, the derived types, the opaque handle types and the
# rocsolverCheck status helper. A consumer writes one `use rocsolver`.
#
# UNLIKE every other library's binding in this monorepo, this one is not
# self-contained at COMPILE time. rocsolver-functions.h includes rocblas.h and the
# rocSOLVER API is spelled in rocBLAS types, so the generated module opens with
#
#   use rocblas, only: rocblas_handle_t, rocblas_fill_upper, ...
#
# and rocblas.mod, built by the SAME compiler, must exist before this archive can
# be compiled. That dependency is resolved below, and it is PUBLIC: a consumer
# that says `use rocsolver` needs rocblas.mod too, because the compiler follows
# the module's own use-association to read it.
#
# librocsolver, in contrast, is required only to LINK a final executable, and is
# carried as the target's link interface for that reason.

cmake_minimum_required(VERSION 3.19)

# Configured directly (not as a subdirectory of the rocSOLVER build), this
# directory is its own project, so that the bindings can be built and packaged
# on their own, which is the supported path for anyone whose Fortran compiler
# is not the amdflang the shipped bindings were built with. In-tree, rocSOLVER's
# own project() is already in effect.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(rocsolver-fortran LANGUAGES NONE)
endif()

# ON by default but GUARDED: a site with no Fortran compiler must still
# configure and build the C library. The monorepo root already probes once and
# publishes ROCM_HAVE_FORTRAN; a standalone rocSOLVER build falls back to its own
# probe.
option(BUILD_FORTRAN_BINDINGS "Build the rocSOLVER Fortran bindings" ON)

# The bindings' own test suite. OFF by default and guarded the same way: asking
# for the tests when the bindings were not built is a skip, not an error.
option(BUILD_FORTRAN_TESTS "Build the rocSOLVER Fortran binding tests" OFF)

# BUILD_FORTRAN_BINDINGS and BUILD_FORTRAN_TESTS are ONE cache entry shared by
# every library in the monorepo, which is what you want for the common case:
# -DBUILD_FORTRAN_BINDINGS=OFF from the top turns the whole Fortran track off in
# a single flag. It is not what you want when one library has to be singled out,
# so each library also honours a name of its own, which wins when it is set.
if(DEFINED ROCSOLVER_BUILD_FORTRAN_BINDINGS)
  set(BUILD_FORTRAN_BINDINGS ${ROCSOLVER_BUILD_FORTRAN_BINDINGS})
endif()
if(DEFINED ROCSOLVER_BUILD_FORTRAN_TESTS)
  set(BUILD_FORTRAN_TESTS ${ROCSOLVER_BUILD_FORTRAN_TESTS})
endif()

# Which array-argument overloads the module exposes. One tri-state replaces the
# two booleans hipFORT used (USE_FPOINTER_INTERFACES / ASSUMED_RANK), because the
# two upper tiers cannot legally coexist: an assumed-rank dummy is not
# distinguishable by rank from the per-rank specifics, so assumed-rank REPLACES
# the per-rank variants rather than adding to them.
#
#   none           the raw type(c_ptr) surface only (Fortran 2003)
#   assumed-shape  + the per-rank array overloads (Fortran 2008): the default,
#                  and what the bindings shipped in ROCm are built with
#   assumed-rank   + the single dimension(..) overload per routine (Fortran 2018)
set(FORTRAN_ARRAY_INTERFACES "assumed-shape" CACHE STRING
    "Array-argument overloads to expose: none, assumed-shape, or assumed-rank")
set_property(CACHE FORTRAN_ARRAY_INTERFACES PROPERTY STRINGS
             none assumed-shape assumed-rank)

# Report to the parent scope whether the bindings were really built. Distinct
# from ROCM_HAVE_FORTRAN, which reports only that a compiler exists: the
# compiler can be present while this directory is switched off.
#
# Only when there IS a parent: standalone, this directory is the top-level scope
# and PARENT_SCOPE warns under -Wdev rather than being a silent no-op, which
# turns into a hard failure under -Werror=dev. Standalone is the documented path
# for anyone whose Fortran compiler is not the one ROCm ships the bindings for,
# so it has to configure cleanly.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(ROCSOLVER_HAVE_FORTRAN_BINDINGS FALSE PARENT_SCOPE)
endif()

if(NOT BUILD_FORTRAN_BINDINGS)
  return()
endif()

# Probing for a Fortran compiler is not free, and every library in a monorepo
# build would otherwise repeat it, so the monorepo root probes ONCE and publishes
# the answer as ROCM_HAVE_FORTRAN. That name is deliberately neither
# rocm-libraries' nor rocm-systems': rocSOLVER is regularly built inside a
# rocm-systems tree (and HIP inside this one), and a repo-specific name would
# make it read the wrong variable, or none. ROCM_LIBS_HAVE_FORTRAN is what the
# rocm-libraries root publishes today and is honoured as a legacy spelling until
# that root adopts the new name.
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
  message(STATUS "rocSOLVER: no Fortran compiler found, skipping the Fortran bindings")
  return()
endif()

enable_language(Fortran)

if(NOT FORTRAN_ARRAY_INTERFACES MATCHES "^(none|assumed-shape|assumed-rank)$")
  message(FATAL_ERROR
    "FORTRAN_ARRAY_INTERFACES must be none, assumed-shape or assumed-rank, "
    "not '${FORTRAN_ARRAY_INTERFACES}'.")
endif()

# The rocBLAS binding, and the one dependency here that is not optional: the
# module says `use rocblas`, so rocblas.mod has to exist before rocsolver.F90 can
# be compiled at all. In-tree the rocblas_fortran target is already defined,
# because projects/rocblas is configured before projects/rocsolver; standalone
# the Fortran package has to be found.
#
# PATH_SUFFIXES lib/cmake/rocblas because that is where the real config lives,
# beside the rocBLAS C package. The forwarder in lib/cmake/rocblas-fortran makes
# the plain spelling work as well, and CMake searches the default suffixes in
# addition to these, so this call finds either layout, including a rocBLAS
# installed before the forwarder existed.
#
# Resolved BEFORE the target is created, and a skip rather than a hard error.
# BUILD_FORTRAN_BINDINGS is ON by default, so a missing optional binding must not
# take the rocSOLVER C library down with it: a tree whose rocBLAS ships no
# Fortran binding still has to configure and build.
if(NOT TARGET roc::rocblas_fortran)
  find_package(rocblas-fortran QUIET CONFIG PATH_SUFFIXES lib/cmake/rocblas)
endif()
if(NOT TARGET roc::rocblas_fortran)
  message(STATUS
    "rocSOLVER: the rocBLAS Fortran binding was not found, skipping the "
    "rocSOLVER Fortran bindings. They are written against it (`use rocblas`), "
    "so rocblas.mod must be built with ${CMAKE_Fortran_COMPILER} and reachable "
    "through CMAKE_PREFIX_PATH.")
  return()
endif()

add_library(rocsolver_fortran STATIC rocsolver.F90)

# roc::, matching the namespace rocSOLVER exports its C library under (see
# library/src/CMakeLists.txt), so that a consumer is not made to remember that
# two halves of one package sit in different namespaces.
add_library(roc::rocsolver_fortran ALIAS rocsolver_fortran)

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
# ROCSOLVER_FORTRAN_COMPILER_DIR itself, for example to gfortran-13.3.0.
set(ROCSOLVER_FORTRAN_COMPILER_DIR "${_fc_name}" CACHE STRING
    "Per-compiler subdirectory the .mod and the archive install under")

set(_moddir "${CMAKE_CURRENT_BINARY_DIR}/include")
set_target_properties(rocsolver_fortran PROPERTIES
  Fortran_MODULE_DIRECTORY "${_moddir}"
  # rocsolver.F90 is uppercase: the array overloads sit behind preprocessor
  # guards, so the source must go through the C preprocessor.
  Fortran_PREPROCESS ON
  # Free form is the .F90 default, but stating it lets CMake emit whichever flag
  # the compiler in use expects rather than relying on the extension.
  Fortran_FORMAT FREE
  POSITION_INDEPENDENT_CODE ON)

# One section per procedure, so that a consumer linking this archive can discard
# the wrappers it does not call.
#
# This is not a size optimisation, it is what makes the archive usable against an
# older librocsolver at all. The module is generated from THIS tree's headers, so
# it wraps API that a released ROCm need not have: against ROCm 10.0
# (librocsolver 3.36) 52 of the C symbols it references are missing, the cholqr,
# gehd2/gehrd and *_64 families that 3.38 adds. The archive holds a single
# object, and a static archive links at object granularity, so pulling in one
# wrapper pulls in every undefined C symbol the object references. A consumer who
# never touches the new API still fails to link, with undefined references to
# functions they have never heard of.
#
# With per-function sections plus --gc-sections below, the unreferenced wrappers
# are dropped and the link succeeds. Verified against that ROCm: a program that
# does nothing but call rocsolverCheck links with these flags and fails without
# them, with 156 undefined references to rocsolver_zcholqr_strided_batched_64
# and its neighbours.
#
# Only a consumer that touches a MODULE PROCEDURE (rocsolverCheck, or any of the
# array overloads) pulls the object in at all; one that calls the plain bind(C)
# interfaces references the C symbols directly and never needs the archive. That
# is why the failure looks arbitrary from the outside, and why the flags are not
# optional.
#
# Guarded on the compilers known to accept the spelling; others simply keep the
# current behaviour, which is correct whenever the installed library matches.
if(CMAKE_Fortran_COMPILER_ID MATCHES "^(GNU|Flang|LLVMFlang|Intel|IntelLLVM)$")
  target_compile_options(rocsolver_fortran PRIVATE -ffunction-sections -fdata-sections)
  target_link_options(rocsolver_fortran INTERFACE
    "$<$<LINK_LANG_AND_ID:Fortran,GNU,Flang,LLVMFlang,Intel,IntelLLVM>:LINKER:--gc-sections>")
endif()

# Translate the tri-state into the two macros the generated source is guarded
# with. assumed-rank implies the F2008 tier: in the source the assumed-rank
# specific is nested inside USE_FPOINTER_INTERFACES, and selects itself over the
# per-rank ones.
if(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-shape")
  target_compile_definitions(rocsolver_fortran PRIVATE USE_FPOINTER_INTERFACES=1)
elseif(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-rank")
  target_compile_definitions(rocsolver_fortran PRIVATE
    USE_FPOINTER_INTERFACES=1 USE_ASSUMED_RANK_INTERFACES=1)
endif()

# No line-length flag: the generator wraps the emitted source at 112 columns,
# inside the 132 that free-form Fortran guarantees, so every compiler reads it
# without a vendor-specific option.

target_include_directories(rocsolver_fortran PUBLIC
  "$<BUILD_INTERFACE:${_moddir}>"
  "$<INSTALL_INTERFACE:include/fortran/${ROCSOLVER_FORTRAN_COMPILER_DIR}>")

# PUBLIC, not INTERFACE: a consumer that says `use rocsolver` needs rocblas.mod on
# its include path too, since the compiler follows the module's use-association.
# The target was resolved above, before this one was created.
target_link_libraries(rocsolver_fortran PUBLIC roc::rocblas_fortran)

# Link interface only: needed to link an executable, not to build this archive.
# roc::rocsolver is defined by rocSOLVER's library/src, and brings the HIP runtime with
# it. Standalone it has to be looked for, and testing for the target without
# ever calling find_package would mean it is never found: the archive would then
# export no link interface at all, and every consumer would hit undefined
# references to rocsolver_dgetrf and friends despite linking
# roc::rocsolver_fortran exactly as documented.
#
# rocsolver-config.cmake does find_dependency(hip), and hip-config.cmake reaches
# FindThreads, which hard-errors unless C or CXX is enabled. This project
# declares LANGUAGES NONE and then enables only Fortran, so C is enabled first;
# it compiles none of the sources here. QUIET, because a site that has no
# installed rocSOLVER can still legitimately build the bindings: they compile
# against nothing.
if(NOT TARGET roc::rocsolver)
  if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
    enable_language(C)
  endif()
  find_package(rocsolver QUIET)
endif()

if(TARGET roc::rocsolver)
  target_link_libraries(rocsolver_fortran INTERFACE roc::rocsolver)
else()
  message(STATUS
    "rocSOLVER: the C library was not found, so the Fortran bindings carry no "
    "link interface; a consumer will have to link it (-lrocsolver) itself.")
endif()

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(ROCSOLVER_HAVE_FORTRAN_BINDINGS TRUE PARENT_SCOPE)
endif()

if(BUILD_FORTRAN_TESTS)
  # enable_testing() here is enough for a STANDALONE build of this directory, but
  # NOT for an in-tree one: CMake only writes the chain of CTestTestfile.cmake
  # that ctest walks down from a build root when testing was enabled at the TOP
  # level. Enabled only here, the tests are built and registered, and `ctest` run
  # from the top of the rocSOLVER build reports "Total Tests: 0": a silent
  # nothing, which is the worst way to find out. So say it.
  enable_testing()
  if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    get_property(_top_testing GLOBAL PROPERTY CTEST_TARGETS_ADDED)
    if(NOT _top_testing)
      message(STATUS
        "rocSOLVER: the Fortran binding tests are registered, but ctest discovers "
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

# lib/cmake/rocsolver is where rocSOLVER's own rocm_export_targets() puts
# rocsolver-config.cmake, so the Fortran package lands beside the C one and a
# single CMAKE_PREFIX_PATH entry finds both.
#
# rocRAND and hipRAND cannot do this and install into lib/cmake/<lib>-fortran
# alone, because their shipped <lib>-config.cmake carries a bare unguarded
# include() of a sibling <lib>-fortran-config.cmake, and putting a file there
# would recurse. rocSOLVER has no such include, checked against an installed
# ROCm, rocsolver-config.cmake mentions Fortran nowhere, so it uses the shared
# layout, like hipBLAS, rocBLAS and the FFT libraries.
set(_fortran_pkgdir "${CMAKE_INSTALL_LIBDIR}/cmake/rocsolver")

install(TARGETS rocsolver_fortran
        EXPORT rocsolver-fortran-targets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/fortran/${ROCSOLVER_FORTRAN_COMPILER_DIR}")
install(DIRECTORY "${_moddir}/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fortran/${ROCSOLVER_FORTRAN_COMPILER_DIR}"
        FILES_MATCHING PATTERN "*.mod")

# The generated source, so that a user on a compiler ROCm ships no .mod for can
# rebuild the binding without fetching a monorepo to get one file. Source
# obtained separately can also be generated from a different ROCm than the
# installed librocsolver, which surfaces as an undefined symbol at link time or as
# an interface that silently differs from the one the caller meant; source that
# ships with the library cannot drift from it.
#
# Under share/, not include/. The .mod is compiler-specific and this is not, and
# every entry under include/fortran/ is partitioned by compiler, so a source
# file there would be read as naming one: the config below enumerates that
# directory to report which compilers are installed, and would list rocsolver.F90
# among them.
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/rocsolver.F90"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/rocsolver/fortran")
install(EXPORT rocsolver-fortran-targets
        FILE rocsolver-fortran-targets.cmake
        NAMESPACE roc::
        DESTINATION "${_fortran_pkgdir}")

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/rocsolver-fortran-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/rocsolver-fortran-config.cmake"
  # No PATH_VARS: it only creates PACKAGE_<var>, and the template interpolates
  # @ROCSOLVER_FORTRAN_COMPILER_DIR@ directly. Passing it was a dead argument, and
  # worse than dead if anyone later switched the template to the PACKAGE_ form,
  # since PATH_VARS would then resolve this relative value as a path under the
  # install prefix.
  INSTALL_DESTINATION "${_fortran_pkgdir}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rocsolver-fortran-config.cmake"
        DESTINATION "${_fortran_pkgdir}")

# The config lands in lib/cmake/rocsolver, BESIDE the C package, because that is
# what would let rocsolver-config.cmake pick it up with a bare
# include(rocsolver-fortran-config.cmake OPTIONAL) once rocm_export_targets grows
# a hook for it: a Fortran binding installed after the C library would then be
# found at find_package(rocsolver) time, and ignored when absent.
#
# rocm_export_targets' existing INCLUDE argument is NOT that hook. It emits a
# bare, non-OPTIONAL include() and emits it BEFORE the targets file
# (ROCMInstallTargets.cmake writes INCLUDE entries first), so a Fortran config
# reached that way would run before roc::rocsolver exists. That change belongs in
# rocm-cmake.
#
# But find_package(rocsolver-fortran): the documented spelling, and the one the
# config's own header shows, searches lib/cmake/rocsolver-fortran*, which
# lib/cmake/rocsolver does not match, so on its own that placement makes the
# direct spelling fail with "could not find a package configuration file". A
# forwarder gives the search somewhere to land without duplicating the real
# config, which would also need its own copy of the targets file and would then
# be a second thing to keep in step.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/forward/rocsolver-fortran-config.cmake"
"# Generated forwarder. The real config lives beside the rocSOLVER C package, in
# lib/cmake/rocsolver, so that rocsolver-config.cmake can include it as a sibling.
# This file only exists so that find_package(rocsolver-fortran) resolves to the
# same place.
include(\"\${CMAKE_CURRENT_LIST_DIR}/../rocsolver/rocsolver-fortran-config.cmake\")
")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/forward/rocsolver-fortran-config.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/rocsolver-fortran")

# A version file, so that find_package(rocsolver-fortran 3.38) resolves rather
# than failing with "could not find a configuration file compatible with
# requested version". SameMajorVersion because the module's bind(C) surface
# tracks the C API's compatibility promise.
#
# The version is read from THIS tree's top-level CMakeLists, not from
# rocsolver_VERSION. That variable looks like the obvious source and is a trap: in
# a standalone build it is set by the find_package(rocsolver) above, so it reports
# whatever rocSOLVER happens to be INSTALLED (3.36.0 from a stock /opt/rocm)
# rather than the version of the headers these bindings were generated from. The
# bindings ship with their source tree, so the source tree is the truth.
set(_fortran_version "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt")
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt" _rocsolver_root)
  # rocSOLVER spells it set(VERSION_STRING "3.38.0") and then passes the variable
  # to rocm_setup_version, so match the literal assignment rather than the call.
  string(REGEX MATCH "set\\([ \t]*VERSION_STRING[ \t]+\"([0-9]+\\.[0-9]+\\.[0-9]+)\""
         _m "${_rocsolver_root}")
  if(CMAKE_MATCH_1)
    set(_fortran_version "${CMAKE_MATCH_1}")
  endif()
  unset(_rocsolver_root)
endif()

if(_fortran_version)
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/rocsolver-fortran-config-version.cmake"
    VERSION "${_fortran_version}"
    COMPATIBILITY SameMajorVersion)
  # Installed in BOTH package directories, and it has to be. CMake looks for
  # <pkg>-config-version.cmake NEXT TO the <pkg>-config.cmake it actually
  # resolved, not wherever the real one lives. find_package(rocsolver-fortran 3.38)
  # resolves the forwarder in lib/cmake/rocsolver-fortran, so a version file only
  # beside the real config in lib/cmake/rocsolver is never consulted, and the call
  # fails with "could not find a configuration file compatible with requested
  # version" even though the config is right there and the version matches.
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rocsolver-fortran-config-version.cmake"
          DESTINATION "${_fortran_pkgdir}")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rocsolver-fortran-config-version.cmake"
          DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/rocsolver-fortran")
else()
  # Better no version file than a wrong one: without it an unversioned
  # find_package still succeeds, whereas a version file claiming the wrong
  # number makes a versioned one succeed and link the wrong .mod.
  message(STATUS
    "rocSOLVER: could not determine the version for the Fortran package, so no "
    "rocsolver-fortran-config-version.cmake is installed; "
    "find_package(rocsolver-fortran <version>) will not match.")
endif()
