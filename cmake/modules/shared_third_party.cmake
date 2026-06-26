# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Shared third-party dependency declarations for the rocm-libraries superbuild.
# Each subproject (hipdnn, miopen, etc.) may independently FetchContent the
# same libraries. To avoid duplicate downloads and the resulting risk of
# version skew or option conflicts, the superbuild may declare them once here,
# before any add_subdirectory() call, with OVERRIDE_FIND_PACKAGE. Subprojects
# then naturally pick up the shared content via find_package().

include_guard(GLOBAL)

include(FetchContent)

# _create_missing_gtest_alias(<name>)
#
# Create a `GTest::<name>` ALIAS for the plain `<name>` target, if the source
# target exists and the alias does not already exist.
#
# Arguments:
#   name  Plain target name exported by upstream googletest (one of
#         `gtest`, `gtest_main`, `gmock`, `gmock_main`).
#
# Rationale:
#   Upstream googletest defines plain targets, while consumers that previously
#   used CMake's FindGTest module expect the `GTest::` namespaced targets.
#   This helper bridges the two without erroring if the source target is
#   absent (e.g. when `BUILD_GMOCK` is OFF) or the alias is already present.
function(_create_missing_gtest_alias name)
    if(TARGET ${name} AND NOT TARGET GTest::${name})
        add_library(GTest::${name} ALIAS ${name})
    endif()
endfunction()

# rocm_libs_declare_shared_deps()
#
# Declare third-party dependencies that are shared across superbuild
# subprojects, so that all subprojects resolve to a single fetched copy
# instead of each populating their own.
#
# This function takes no arguments. It must be called from the top-level
# superbuild CMakeLists.txt *before* any `add_subdirectory()` call that
# brings in a subproject which would otherwise fetch the same dependency.
#
# Each dependency is declared with `FetchContent_Declare(... OVERRIDE_FIND_PACKAGE)`
# and immediately made available, so subsequent `find_package()` calls in
# subprojects resolve to the shared content and their own fetches are skipped.
#
# Currently declares:
#   - nlohmann_json (v3.12.0): shared by hipdnn and miopen.
#   - GTest         (v1.17.0): shared by hipdnn and miopen; forced static to
#                              work around missing dllexports on Windows.
#
# Side effects:
#   - Sets cache variables for the declared dependencies (e.g.
#     `JSON_BuildTests`, `gtest_force_shared_crt`, `BUILD_GMOCK`,
#     `INSTALL_GTEST`) with FORCE.
#   - Temporarily overrides `BUILD_SHARED_LIBS` while populating GTest and
#     restores the original value before returning.
#   - Creates `GTest::{gtest,gtest_main,gmock,gmock_main}` ALIAS targets.
function(rocm_libs_declare_shared_deps)
    # -------------------------------------------------------------------------
    # nlohmann_json
    #
    # Used by hipdnn (projects/hipdnn/cmake/Dependencies.cmake) and miopen
    # (projects/miopen/cmake/ThirdParty.cmake). hipdnn fetches the content
    # under the name `json` and uses the raw include dir; miopen fetches under
    # the name `nlohmann_json` with OVERRIDE_FIND_PACKAGE and uses the
    # standard `nlohmann_json::nlohmann_json` target. Declaring here (also
    # under the name `nlohmann_json` with OVERRIDE_FIND_PACKAGE) means both
    # subprojects' subsequent find_package(nlohmann_json) calls resolve to
    # this single copy.
    #
    # Version and URL+hash mirror projects/miopen/cmake/ThirdParty.cmake so
    # behavior is identical when miopen runs in MIOPEN_STANDALONE_BUILD mode
    # outside the superbuild.
    # -------------------------------------------------------------------------
    set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
    set(JSON_MultipleHeaders ON CACHE BOOL "" FORCE)
    set(JSON_Install ON CACHE BOOL "" FORCE)

    FetchContent_Declare(nlohmann_json
        URL      https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.tar.gz
        URL_HASH SHA256=4b92eb0c06d10683f7447ce9406cb97cd4b453be18d7279320f7b2f025c10187
        SYSTEM
        OVERRIDE_FIND_PACKAGE)
    FetchContent_MakeAvailable(nlohmann_json)

    # -------------------------------------------------------------------------
    # GTest
    #
    # Both hipdnn (projects/hipdnn/cmake/Dependencies.cmake `_fetch_gtest`,
    # content name `googletest`, pin v1.16.0) and miopen
    # (projects/miopen/cmake/ThirdParty.cmake, content name `GTest`,
    # pin v1.17.0) FetchContent gtest. Their content names differ, so both
    # populate. But both call add_library(gtest) / add_library(gmock) etc.,
    # which produces a hard "target already exists" error when both
    # subprojects are configured in the same superbuild.
    #
    # Declaring GTest here with OVERRIDE_FIND_PACKAGE causes both
    # subprojects' find_package(GTest) calls to succeed against this single
    # content, and they skip their own fetches. Pin matches miopen's choice
    # (v1.17.0).
    #
    # GTest must be built static on Windows: gmock has class-static members
    # (e.g. testing::internal::g_gmock_mutex) that lack __declspec(dllexport),
    # so a shared gmock_main.dll fails to link. Match the same options miopen
    # sets in projects/miopen/cmake/ThirdParty.cmake.
    # -------------------------------------------------------------------------
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(BUILD_GMOCK           ON  CACHE BOOL "" FORCE)
    set(INSTALL_GTEST         OFF CACHE BOOL "" FORCE)
    set(_rocm_libs_saved_BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS}")
    set(BUILD_SHARED_LIBS OFF)
    FetchContent_Declare(GTest
        URL      https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
        URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
        SYSTEM
        OVERRIDE_FIND_PACKAGE)
    FetchContent_MakeAvailable(GTest)
    set(BUILD_SHARED_LIBS "${_rocm_libs_saved_BUILD_SHARED_LIBS}")
    unset(_rocm_libs_saved_BUILD_SHARED_LIBS)

    foreach(_lib gtest gtest_main gmock gmock_main)
        _create_missing_gtest_alias(${_lib})
    endforeach()
endfunction()
