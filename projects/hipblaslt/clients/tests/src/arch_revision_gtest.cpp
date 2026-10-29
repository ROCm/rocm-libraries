// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only smoke test for the ASIC-revision -> library-arch mapping, a pure
// function with no HIP dependency (see rocblaslt_arch_revision.hpp). Included by
// relative path so the white-box test stays self-contained without adding the
// internal rocblaslt include dir to the whole test target.

#include <gtest/gtest.h>

#include "../../../library/src/amd_detail/rocblaslt/src/include/rocblaslt_arch_revision.hpp"

namespace
{
    TEST(arch_revision_smoke, gfx1250_revision0_is_the_v0_subtree)
    {
        // An A0 part in non-strict mode reports gfx1250 and loads its own tree.
        EXPECT_EQ(rocblaslt_revisioned_arch_name("gfx1250", 0), "gfx1250v0");
    }

    TEST(arch_revision_smoke, gfx1250_strict_keeps_its_reported_name)
    {
        // An A0 part in strict mode reports gfx1250-strict (asicRevision is still 0)
        // and keeps loading library/gfx1250-strict/.
        EXPECT_EQ(rocblaslt_revisioned_arch_name("gfx1250-strict", 0), "gfx1250-strict");
    }

    TEST(arch_revision_smoke, gfx1250_nonzero_revision_keeps_the_base_name)
    {
        // Assumes B0 reports a non-zero revision; it maps to the plain gfx1250 tree.
        EXPECT_EQ(rocblaslt_revisioned_arch_name("gfx1250", 1), "gfx1250");
        EXPECT_EQ(rocblaslt_revisioned_arch_name("gfx1250", 2), "gfx1250");
    }

    TEST(arch_revision_smoke, gfx1250_unknown_revision_keeps_the_base_name)
    {
        // Only a revision known to be A0 selects the A0 tree.
        EXPECT_EQ(rocblaslt_revisioned_arch_name("gfx1250", -1), "gfx1250");
    }

    TEST(arch_revision_smoke, other_arches_are_unaffected_by_revision)
    {
        EXPECT_EQ(rocblaslt_revisioned_arch_name("gfx942", 0), "gfx942");
        EXPECT_EQ(rocblaslt_revisioned_arch_name("gfx950", 0), "gfx950");
        EXPECT_EQ(rocblaslt_revisioned_arch_name("gfx1250v0", 0), "gfx1250v0");
        EXPECT_EQ(rocblaslt_revisioned_arch_name("gfx1250-strict", 1), "gfx1250-strict");
    }
}
