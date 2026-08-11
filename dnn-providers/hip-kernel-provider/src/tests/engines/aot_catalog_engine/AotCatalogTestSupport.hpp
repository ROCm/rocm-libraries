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
#define AOT_SKIP_OR_FAIL_ON_EMPTY_CATALOG(catalog_dir)                              \
    do                                                                              \
    {                                                                               \
        if(AOT_ROCKE_FAMILIES_EXPECTED)                                             \
        {                                                                           \
            FAIL() << "empty AOT catalog at " << (catalog_dir)                      \
                   << ": rocKE-AOT families were configured for this build "        \
                      "(AOT_ROCKE_FAMILIES_EXPECTED=1) but no kernels loaded -- a " \
                      "producer or the catalog loader dropped the whole family; "   \
                      "refusing to skip.";                                          \
        }                                                                           \
        else                                                                        \
        {                                                                           \
            GTEST_SKIP() << "empty AOT catalog at " << (catalog_dir)                \
                         << " (rocKE-AOT families not built for this arch; build "  \
                            "with -DROCKE_PYTHON_DIR to populate it)";              \
        }                                                                           \
    } while(0)
