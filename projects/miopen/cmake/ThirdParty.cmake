# Fetch and build the third-party dependencies MIOpen requires when no
# system / superbuild copies are available. Activated by MIOPEN_STANDALONE_BUILD
# from projects/MIOpen/CMakeLists.txt.
#
# For each dependency we first try find_package(... QUIET) so that an
# already-installed system copy (matching the version constraint MIOpen's main
# CMakeLists imposes) is used as-is. Only deps that aren't found system-wide
# fall back to FetchContent. Upstream URLs and hashes follow the choices in
# D:/develop/src/TheRock/third-party (and .../sysdeps/common); upstream sources
# are preferred and AMD bucket URLs are noted in comments as fallback.
#
# Each FetchContent_Declare uses OVERRIDE_FIND_PACKAGE so the existing
# find_package(<name> [REQUIRED]) calls in projects/MIOpen/CMakeLists.txt
# resolve transparently against the populated content.
#
# Requires CMake 3.25 (OVERRIDE_FIND_PACKAGE from 3.24, plus the SYSTEM keyword
# on FetchContent_Declare / add_subdirectory from 3.25 so third-party headers
# reach consumers via -isystem rather than -I, suppressing their warnings).
# MIOpen's top-level cmake_minimum_required stays at 3.15; this floor only
# applies in standalone mode where we know we control the environment.

cmake_minimum_required(VERSION 3.25)
include(FetchContent)

message(STATUS "MIOpen: standalone build — preferring system third-party deps, fetching what's missing")

# -----------------------------------------------------------------------------
# Save state of variables we override below so we can restore them before
# returning to MIOpen's own CMakeLists. Without this:
#   - FORCE-setting a cache variable and then unset(... CACHE) silently wipes
#     the user's -D value. This broke BUILD_TESTING and skipped the test/
#     subdirectory, so miopen_gtest never got created.
#   - Setting a normal variable without restoring leaks into MIOpen's parent
#     scope (CMAKE_POLICY_VERSION_MINIMUM, FETCHCONTENT_QUIET).
# -----------------------------------------------------------------------------
macro(_miopen_save_var name)
    if(DEFINED ${name})
        set(_miopen_saved_${name} "${${name}}")
        set(_miopen_saved_${name}_was_defined TRUE)
    else()
        set(_miopen_saved_${name}_was_defined FALSE)
    endif()
endmacro()

# Restore a cache variable saved by _miopen_save_var, or unset it if it was
# undefined before we touched it.
macro(_miopen_restore_cache_var name docstring)
    if(_miopen_saved_${name}_was_defined)
        set(${name} "${_miopen_saved_${name}}" CACHE BOOL "${docstring}" FORCE)
    else()
        unset(${name} CACHE)
    endif()
    unset(_miopen_saved_${name})
    unset(_miopen_saved_${name}_was_defined)
endmacro()

# Restore a normal (non-cache) variable saved by _miopen_save_var, or unset it
# if it was undefined before we touched it.
macro(_miopen_restore_var name)
    if(_miopen_saved_${name}_was_defined)
        set(${name} "${_miopen_saved_${name}}")
    else()
        unset(${name})
    endif()
    unset(_miopen_saved_${name})
    unset(_miopen_saved_${name}_was_defined)
endmacro()

_miopen_save_var(BUILD_TESTING)
_miopen_save_var(CMAKE_WARN_DEPRECATED)
_miopen_save_var(FETCHCONTENT_QUIET)
_miopen_save_var(CMAKE_POLICY_VERSION_MINIMUM)

# Suppress sub-project noise: only CMake's own one-liner per dep.
set(FETCHCONTENT_QUIET ON)

# Per-dep build options must be set before MakeAvailable.
set(JSON_BuildTests       OFF CACHE BOOL "" FORCE)
set(JSON_MultipleHeaders  ON  CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK           ON  CACHE BOOL "" FORCE)
set(INSTALL_GTEST         OFF CACHE BOOL "" FORCE)

# Eigen3: disable testing/doc to avoid creating a 'check' target that
# collides with MIOpen's own 'check' target in test/CMakeLists.txt.
set(BUILD_TESTING         OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING   OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC       OFF CACHE BOOL "" FORCE)

# frugally-deep's CMake floor is older than 3.30's policy default.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
# Suppress deprecation warnings from third-party CMakeLists (frugally-deep).
# Set as a normal variable so save/restore via _miopen_save_var/_miopen_restore_var
# round-trips cleanly without leaving a cache entry behind.
set(CMAKE_WARN_DEPRECATED OFF)

# -----------------------------------------------------------------------------
# Helper: log the find_package outcome for a dep so the configure log makes the
# system-vs-fetched decision visible.
# -----------------------------------------------------------------------------
function(_miopen_report_dep name found version)
    if(found)
        if(version)
            message(STATUS "MIOpen: using system ${name} ${version}")
        else()
            message(STATUS "MIOpen: using system ${name}")
        endif()
    else()
        message(STATUS "MIOpen: ${name} not found system-wide, will FetchContent")
    endif()
endfunction()

# -----------------------------------------------------------------------------
# Helpers: silence ROCMChecks's variable_watch on CMAKE_*_FLAGS / CMAKE_*_COMPILER
# while a third-party subproject configures.
#
# Several upstream projects (googletest, eigen) set these toolchain variables
# directly — perfectly normal CMake usage that we can't fix upstream. ROCMChecks
# installs a variable_watch that fires on every assignment, producing pages of
# warnings during configure. Override the watcher to a no-op around the fetch,
# then re-include ROCMChecks (with the warn/error toggles off so it only
# redefines the function and doesn't reinstall a second watch) to restore the
# original handler for MIOpen's own code.
# -----------------------------------------------------------------------------
macro(_miopen_suppress_rocm_toolchain_checks)
    # Override ROCMChecks's watcher with a no-op for the duration of a fetch.
    function(rocm_check_toolchain_var)
    endfunction()
endmacro()

# Re-include ROCMChecks to restore the real rocm_check_toolchain_var handler
# for MIOpen's own code, with the warn/error toggles off so no second watch is
# installed.
macro(_miopen_restore_rocm_toolchain_checks)
    set(ROCM_WARN_TOOLCHAIN_VAR OFF)
    set(ROCM_ERROR_TOOLCHAIN_VAR OFF)
    include(ROCMChecks)
    unset(ROCM_WARN_TOOLCHAIN_VAR)
    unset(ROCM_ERROR_TOOLCHAIN_VAR)
endmacro()

# -----------------------------------------------------------------------------
# Eigen3 (header-only)
# -----------------------------------------------------------------------------
find_package(Eigen3 QUIET)
_miopen_report_dep(Eigen3 "${Eigen3_FOUND}" "${Eigen3_VERSION}")
if(NOT Eigen3_FOUND)
    # Originally mirrored from https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/eigen-3.4.0.tar.bz2
    FetchContent_Declare(Eigen3
        URL      https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2
        URL_HASH SHA256=b4c198460eba6f28d34894e3a5710998818515104d6e74e5cc331ce31e46e626
        SYSTEM
        OVERRIDE_FIND_PACKAGE)
    # Eigen's CMakeLists writes to CMAKE_CXX_FLAGS while probing compiler support
    # for many warning flags; without the guard ROCMChecks emits one warning per
    # probe (dozens during configure).
    _miopen_suppress_rocm_toolchain_checks()
    FetchContent_MakeAvailable(Eigen3)
    _miopen_restore_rocm_toolchain_checks()
endif()

# -----------------------------------------------------------------------------
# nlohmann_json (>= 3.9.1, used directly by MIOpen and as a frugally-deep dep)
# -----------------------------------------------------------------------------
find_package(nlohmann_json 3.9.1 QUIET)
_miopen_report_dep(nlohmann_json "${nlohmann_json_FOUND}" "${nlohmann_json_VERSION}")
if(NOT nlohmann_json_FOUND)
    # Note: TheRock pins the GitHub release asset (json-3.12.0.tar.gz, hash 42f6e9...);
    # we use the auto-generated tag archive instead (different SHA, identical content).
    FetchContent_Declare(nlohmann_json
        URL      https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.tar.gz
        URL_HASH SHA256=4b92eb0c06d10683f7447ce9406cb97cd4b453be18d7279320f7b2f025c10187
        SYSTEM
        OVERRIDE_FIND_PACKAGE)
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# Workaround: TheRock's prebuilt nlohmann_json artifact may be missing the
# natvis file (a Visual Studio debug visualizer) referenced by
# INTERFACE_SOURCES.  Drop it from the imported target only if the file
# doesn't exist on disk.  Once the artifact is fixed upstream, the file
# will be present and the property is left untouched.
if(TARGET nlohmann_json::nlohmann_json)
    get_target_property(_njson_srcs nlohmann_json::nlohmann_json INTERFACE_SOURCES)
    if(_njson_srcs)
        set(_njson_filtered "${_njson_srcs}")
        foreach(_src IN LISTS _njson_srcs)
            string(GENEX_STRIP "${_src}" _src_plain)
            if(_src_plain MATCHES "\\.natvis$" AND NOT EXISTS "${_src_plain}")
                message(STATUS "nlohmann_json: dropping missing natvis entry: ${_src_plain}")
                list(REMOVE_ITEM _njson_filtered "${_src}")
            endif()
        endforeach()
        set_property(TARGET nlohmann_json::nlohmann_json
                     PROPERTY INTERFACE_SOURCES "${_njson_filtered}")
    endif()
endif()

# -----------------------------------------------------------------------------
# FunctionalPlus (transitive dep of frugally-deep build; only needed if we have
# to build frugally-deep ourselves).
# -----------------------------------------------------------------------------
find_package(frugally-deep CONFIG QUIET)
_miopen_report_dep(frugally-deep "${frugally-deep_FOUND}" "${frugally-deep_VERSION}")
if(NOT frugally-deep_FOUND)
    find_package(FunctionalPlus QUIET)
    _miopen_report_dep(FunctionalPlus "${FunctionalPlus_FOUND}" "${FunctionalPlus_VERSION}")
    if(NOT FunctionalPlus_FOUND)
        # Originally mirrored from https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/FunctionalPlus-0.2.25.tar.gz
        FetchContent_Declare(FunctionalPlus
            URL      https://github.com/Dobiasd/FunctionalPlus/archive/refs/tags/v0.2.25.tar.gz
            URL_HASH SHA256=9b5e24bbc92f43b977dc83efbc173bcf07dbe07f8718fc2670093655b56fcee3
            SYSTEM
            OVERRIDE_FIND_PACKAGE)
        FetchContent_MakeAvailable(FunctionalPlus)
    endif()

    # frugally-deep: populate but stub upstream cmake/pkgconfig.cmake before
    # add_subdirectory() so its install(EXPORT) referencing un-exported nlohmann_json
    # doesn't trip CMake's generate-time export-set validation.
    # Originally mirrored from https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/frugally-deep-0.15.31.tar.gz
    FetchContent_Declare(frugally-deep
        URL      https://github.com/Dobiasd/frugally-deep/archive/refs/tags/v0.15.31.tar.gz
        URL_HASH SHA256=49bf5e30ad2d33e464433afbc8b6fe8536fc959474004a1ce2ac03d7c54bc8ba
        SYSTEM
        OVERRIDE_FIND_PACKAGE)
    FetchContent_GetProperties(frugally-deep)
    if(NOT frugally-deep_POPULATED)
        FetchContent_Populate(frugally-deep)
    endif()
    file(WRITE "${frugally-deep_SOURCE_DIR}/cmake/pkgconfig.cmake"
         "# install rules disabled by MIOpen standalone wrapper (cmake/ThirdParty.cmake)\n")
    # SYSTEM here forces -isystem for fdeep's include dirs in consumers; the
    # SYSTEM flag on FetchContent_Declare above does not propagate to a manual
    # add_subdirectory(), only to MakeAvailable's implicit one.
    add_subdirectory("${frugally-deep_SOURCE_DIR}" "${frugally-deep_BINARY_DIR}" SYSTEM EXCLUDE_FROM_ALL)

    # Mimic what FetchContent_MakeAvailable would have written for OVERRIDE_FIND_PACKAGE,
    # so the later find_package(frugally-deep CONFIG REQUIRED) call resolves to the
    # in-tree fdeep target created by the add_subdirectory() above.
    file(WRITE "${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/frugally-deep-config.cmake"
         "include(\"\${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/frugally-deep-extra.cmake\" OPTIONAL)\n")
    file(WRITE "${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/frugally-deep-config-version.cmake"
         "set(PACKAGE_VERSION \"0.15.31\")\n"
         "set(PACKAGE_VERSION_COMPATIBLE TRUE)\n"
         "set(PACKAGE_VERSION_EXACT TRUE)\n")
endif()

# -----------------------------------------------------------------------------
# BZip2
# -----------------------------------------------------------------------------
find_package(BZip2 QUIET)
_miopen_report_dep(BZip2 "${BZIP2_FOUND}" "${BZIP2_VERSION_STRING}")
if(NOT BZIP2_FOUND)
    FetchContent_Declare(BZip2
        URL      https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz
        URL_HASH SHA256=ab5a03176ee106d3f0fa90e381da478ddae405918153cca248e682cd0c4a2269
        SYSTEM
        OVERRIDE_FIND_PACKAGE)
    FetchContent_MakeAvailable(BZip2)
    # Upstream tarball has no CMakeLists.txt; build from the vendored wrapper,
    # which creates the BZip2::BZip2 alias and installs miopen_bz2 in the
    # miopen-targets export set. SYSTEM marks the wrapper's targets so their
    # PUBLIC include dirs (the bzip2 headers) reach consumers via -isystem.
    set(BZIP2_UPSTREAM_SOURCE_DIR "${bzip2_SOURCE_DIR}")
    add_subdirectory(
        "${CMAKE_CURRENT_LIST_DIR}/thirdparty/bzip2"
        "${bzip2_BINARY_DIR}-wrapper"
        SYSTEM
    )
    if(MSVC)
        target_compile_options(miopen_bz2 PRIVATE /W0)
    else()
        target_compile_options(miopen_bz2 PRIVATE -w)
    endif()
endif()

# -----------------------------------------------------------------------------
# SQLite3
# -----------------------------------------------------------------------------
find_package(SQLite3 QUIET)
_miopen_report_dep(SQLite3 "${SQLite3_FOUND}" "${SQLite3_VERSION}")
if(NOT SQLite3_FOUND)
    FetchContent_Declare(SQLite3
        URL      https://sqlite.org/2026/sqlite-amalgamation-3510300.zip
        URL_HASH SHA256=acb1e6f5d832484bf6d32b681e858c38add8b2acdfd42ac5df24b8afb46552b4
        SYSTEM
        OVERRIDE_FIND_PACKAGE)
    FetchContent_MakeAvailable(SQLite3)
    # Upstream tarball has no CMakeLists.txt; build from the vendored wrapper,
    # which creates the SQLite::SQLite3 alias and installs miopen_sqlite3 in the
    # miopen-targets export set. SYSTEM marks the wrapper's targets so their
    # PUBLIC include dirs (the sqlite3 amalgamation) reach consumers via -isystem.
    set(SQLITE3_UPSTREAM_SOURCE_DIR "${sqlite3_SOURCE_DIR}")
    add_subdirectory(
        "${CMAKE_CURRENT_LIST_DIR}/thirdparty/sqlite3"
        "${sqlite3_BINARY_DIR}-wrapper"
        SYSTEM
    )
    if(MSVC)
        target_compile_options(miopen_sqlite3 PRIVATE /W0)
    else()
        target_compile_options(miopen_sqlite3 PRIVATE -w)
    endif()
endif()

# -----------------------------------------------------------------------------
# GTest
# -----------------------------------------------------------------------------
find_package(GTest QUIET)
_miopen_report_dep(GTest "${GTest_FOUND}" "${GTest_VERSION}")
if(NOT GTest_FOUND)
    # GTest must be built static on Windows: gmock has class-static members
    # (e.g. testing::internal::g_gmock_mutex) that lack __declspec(dllexport),
    # so a shared gmock_main.dll fails to link. Save/restore BUILD_SHARED_LIBS
    # so only the gtest subdirectory sees the override.
    # Originally mirrored from https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/googletest-1.17.0.tar.gz
    FetchContent_Declare(GTest
        URL      https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
        URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
        SYSTEM
        OVERRIDE_FIND_PACKAGE)
    _miopen_save_var(BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS OFF)
    _miopen_suppress_rocm_toolchain_checks()
    FetchContent_MakeAvailable(GTest)
    _miopen_restore_rocm_toolchain_checks()
    _miopen_restore_var(BUILD_SHARED_LIBS)

    foreach(_tgt gtest gtest_main gmock gmock_main)
        if(TARGET ${_tgt})
            target_compile_options(${_tgt} PRIVATE -w)
        endif()
    endforeach()

    # Some of MIOpen's existing find_package consumers expect GTest::gtest /
    # GTest::gtest_main aliases (the names FindGTest creates). googletest's own
    # CMakeLists exports plain gtest / gtest_main targets, so add the aliases.
    if(TARGET gtest AND NOT TARGET GTest::gtest)
        add_library(GTest::gtest ALIAS gtest)
    endif()
    if(TARGET gtest_main AND NOT TARGET GTest::gtest_main)
        add_library(GTest::gtest_main ALIAS gtest_main)
    endif()
    if(TARGET gmock AND NOT TARGET GTest::gmock)
        add_library(GTest::gmock ALIAS gmock)
    endif()
    if(TARGET gmock_main AND NOT TARGET GTest::gmock_main)
        add_library(GTest::gmock_main ALIAS gmock_main)
    endif()
endif()

# Restore the variables we overrode at the top of this file so nothing leaks
# into MIOpen's own CMakeLists.
_miopen_restore_cache_var(BUILD_TESTING "Build the testing tree.")
_miopen_restore_var(CMAKE_WARN_DEPRECATED)
_miopen_restore_var(FETCHCONTENT_QUIET)
_miopen_restore_var(CMAKE_POLICY_VERSION_MINIMUM)
