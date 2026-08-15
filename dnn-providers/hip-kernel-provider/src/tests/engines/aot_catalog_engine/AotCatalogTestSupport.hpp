// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Shared test-support for the AOT catalog engine's family + substrate tests.
// This is engine SUBSTRATE (not tied to any one kernel family), so per the family
// self-containment contract in library/CMakeLists.txt it lives here at the engine
// test level rather than in a family folder. It is header-only (a macro), so a
// family test that includes it gains no link dependency on another family.
#pragma once

#include <gtest/gtest.h>

#include "catalog/ModulePath.hpp"

#include <filesystem>
#include <string>

// Set by src/tests/CMakeLists.txt: 1 when this build configured rocKE-AOT kernel
// families for its target arch(es) -- i.e. the AOT_CATALOG_FAMILY_TARGETS global
// property is non-empty (rocKE available + arch requested) -- and 0 otherwise.
// Defaulted here so the header still compiles if included outside that build.
#ifndef AOT_ROCKE_FAMILIES_EXPECTED
#define AOT_ROCKE_FAMILIES_EXPECTED 0
#endif

// Decide what an EMPTY build-tree catalog means for a parity/selection test, and
// end the test accordingly:
//
//   * rocKE ON (AOT_ROCKE_FAMILIES_EXPECTED=1): the producers ran for an
//     arch-gated family, so an empty catalog means every kernel was silently
//     dropped -- a producer emitted nothing, or the loader rejected the family.
//     A GTEST_SKIP here would keep CI green over a total failure, so FAIL loudly.
//   * rocKE OFF (AOT_ROCKE_FAMILIES_EXPECTED=0, e.g. TheRock CI today): families
//     legitimately were not built, so an empty catalog is expected and we skip.
//
// Both FAIL() and GTEST_SKIP() return from the calling TEST body, so this ends
// the test exactly as the bare GTEST_SKIP it replaces did.
//
// The rocKE-ON vs rocKE-OFF choice is a compile-time constant
// (AOT_ROCKE_FAMILIES_EXPECTED is #defined to 0 or 1), so the branch is taken at
// the *preprocessor* level rather than with a runtime `if`. That keeps each
// expansion a single gtest statement -- avoiding both readability-else-after-return
// (FAIL() contains a return) and -Wunreachable-code (a constant-condition `if`),
// which a runtime if/else form trips under the Superbuild's warnings-as-errors.
#if AOT_ROCKE_FAMILIES_EXPECTED
#define AOT_SKIP_OR_FAIL_ON_EMPTY_CATALOG(catalog_dir)                      \
    FAIL() << "empty AOT catalog at " << (catalog_dir)                      \
           << ": rocKE-AOT families were configured for this build "        \
              "(AOT_ROCKE_FAMILIES_EXPECTED=1) but no kernels loaded -- a " \
              "producer or the catalog loader dropped the whole family; "   \
              "refusing to skip."
#else
#define AOT_SKIP_OR_FAIL_ON_EMPTY_CATALOG(catalog_dir)                     \
    GTEST_SKIP() << "empty AOT catalog at " << (catalog_dir)               \
                 << " (rocKE-AOT families not built for this arch; build " \
                    "with -DROCKE_PYTHON_DIR to populate it)"
#endif

// Catalog-path pieces baked by src/tests/CMakeLists.txt. RELDIR is the catalog's
// offset beneath the plugin-engine dir; ENGINE_SUBDIR is the plugin-engine dir's
// offset beneath the install "bin"/"lib" root. Defaulted here so the header still
// compiles if included outside that build.
#ifndef AOT_CATALOG_RELDIR
#define AOT_CATALOG_RELDIR "arch_content/aot_catalog"
#endif
#ifndef AOT_CATALOG_ENGINE_SUBDIR
#define AOT_CATALOG_ENGINE_SUBDIR "hipdnn_plugins/engines"
#endif
#ifndef AOT_CATALOG_TEST_DIR
#define AOT_CATALOG_TEST_DIR ""
#endif

// Resolve the AOT catalog directory for a test at RUNTIME, mirroring the engine's
// own self-location (catalog/ModulePath.cpp + Catalog.cpp::resolveCatalogDir), so a
// test finds the catalog whether it runs from the build tree or from a packaged
// install tree on a *separate* runner (TheRock's split build/test CI). The baked
// absolute AOT_CATALOG_TEST_DIR points at the BUILD machine's tree and is invalid
// on the test runner, so it is only a last-resort local-dev fallback.
//
// thisModuleDir() is anchored on engine code that is statically linked INTO the
// test executable, so here it returns the *test exe's* own directory. That dir sits
// at a different depth than the plugin between the two trees, so two offsets are
// tried in order:
//   * build tree   -- the test exe's RUNTIME_OUTPUT_DIRECTORY *is* the engine dir,
//                     so the catalog is at  <exe>/AOT_CATALOG_RELDIR.
//   * install tree -- the test exe installs to bin/, but the catalog installs to
//                     bin/AOT_CATALOG_ENGINE_SUBDIR/AOT_CATALOG_RELDIR.
// The first candidate that exists wins; emptiness of a located catalog is left for
// AOT_SKIP_OR_FAIL_ON_EMPTY_CATALOG to report against the real path.
inline std::string aotResolveTestCatalogDir()
{
    namespace fs = std::filesystem;

    const auto exists = [](const fs::path& dir) {
        std::error_code ec;
        return fs::is_directory(dir, ec);
    };

    const std::string moduleDir = aot_catalog_engine::catalog::thisModuleDir();
    if(!moduleDir.empty())
    {
        const fs::path buildTree = fs::path(moduleDir) / AOT_CATALOG_RELDIR;
        if(exists(buildTree))
        {
            return buildTree.string();
        }
        const fs::path installTree
            = fs::path(moduleDir) / AOT_CATALOG_ENGINE_SUBDIR / AOT_CATALOG_RELDIR;
        if(exists(installTree))
        {
            return installTree.string();
        }
    }

    return AOT_CATALOG_TEST_DIR;
}
