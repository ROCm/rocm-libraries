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
    TEST(ArchRevisionSmoke, Gfx1250Revision0IsTheV0Subtree)
    {
        // The one case that diverges: pre-production v0 loads its own tree.
        EXPECT_EQ(rocblaslt_revisioned_arch_name("gfx1250", 0), "gfx1250v0");
    }

    TEST(ArchRevisionSmoke, Gfx1250V1RevisionKeepsTheBaseName)
    {
        // v1 is revision 1; it must map to the plain gfx1250 tree.
        EXPECT_EQ(rocblaslt_revisioned_arch_name("gfx1250", 1), "gfx1250");
    }

    TEST(ArchRevisionSmoke, Gfx1250UnknownOrFutureRevisionDefaultsToV1)
    {
        // -1 is what HIP reports when it is too old to expose the field; any
        // unseen value must default to the v1 tree rather than invent a
        // subtree that was never built.
        EXPECT_EQ(rocblaslt_revisioned_arch_name("gfx1250", -1), "gfx1250");
        EXPECT_EQ(rocblaslt_revisioned_arch_name("gfx1250", 2), "gfx1250");
    }

    TEST(ArchRevisionSmoke, OtherArchesAreUnaffectedByRevision)
    {
        // Only gfx1250 has a revision split; every other arch is returned
        // unchanged regardless of what asicRevision happens to be.
        EXPECT_EQ(rocblaslt_revisioned_arch_name("gfx942", 0), "gfx942");
        EXPECT_EQ(rocblaslt_revisioned_arch_name("gfx950", 0), "gfx950");
        EXPECT_EQ(rocblaslt_revisioned_arch_name("gfx1250v0", 0), "gfx1250v0");
    }
}
