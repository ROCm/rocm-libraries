// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Pins the rule that picks which gfx1250 stepping hipBLASLt serves a device with.
//
// hipBLASLt used to take that from the name the runtime reported, which for an A0
// part is decided by HSA_DISABLE_GFX12_STRICT: `"0"` reports gfx1250-strict,
// anything else -- including unset, today's default -- reports gfx1250. The
// library now derives it from hipDeviceProp_t::asicRevision instead, so the same
// binary behaves identically on the same silicon whichever way that variable is
// set, and so that a library never has to set a process-global ROCr variable on
// its own behalf.
//
// The property the tests below exist to protect is not "an A0 gets
// gfx1250-strict". It is narrower and easier to break:
//
//   * all three spellings of one A0 produce the SAME candidate list, and
//   * that list always ends in the base architecture,
//
// because the second is what keeps a gfx1250-only package -- which is what ROCm
// ships during the transition -- working on an A0 the runtime calls gfx1250. A
// rule that answered gfx1250-strict and stopped there would point the library
// path, the mapping filename and the code-object filter at a directory the build
// never produced.
//
// Host-only and deterministic: the rule under test is pure string and integer
// work, so every stepping can be exercised here, including ones no CI machine
// has. Which stepping the silicon in the room actually is gets decided elsewhere
// and cannot be unit-tested.
//
// Smoke tier: PR CI runs hipBLASLt with TEST_TYPE=quick, which selects
// `--gtest_filter=*smoke*` (test/therock/test_hipblaslt.py). The names below
// carry the `smoke` token so this guard runs on the PR gate.

#include <gtest/gtest.h>

// Included by relative path on purpose, for the same reason secure_env_gtest.cpp
// does it: rocblaslt_arch_candidates.hpp is dependency-free (only <string> and
// <vector>, no HIP types), and adding the internal rocblaslt include directory to
// the hipblaslt-test target shadows the clients' own "utility.hpp" with the
// library's internal one and breaks unrelated test sources.
#include "../../../library/src/amd_detail/rocblaslt/src/include/rocblaslt_arch_candidates.hpp"

namespace
{
    // The revision values as ROCr reports them, named rather than spelled, so the
    // cases below read as the hardware they stand for.
    constexpr int kRevisionA0 = 0;
    constexpr int kRevisionB0 = 1;

    using Names = std::vector<std::string>;
}

// The whole point of the change: whatever ROCr called it, an A0 is served
// kernels generated under the stepping's capabilities, so
// HSA_DISABLE_GFX12_STRICT cannot put multicast or the fp4 32x16 WMMA on
// silicon that cannot execute them. Which of the two subtrees answers does
// depend on the reported name, because that is what decides loadability.
TEST(arch_candidates, smoke_a0_never_leaves_the_stepping_kernels)
{
    // HSA_DISABLE_GFX12_STRICT unset or 1: only base-named objects load.
    const Names reportedBase{"gfx1250v0"};
    EXPECT_EQ(rocblaslt_arch_name_candidates("gfx1250", kRevisionA0), reportedBase);
    EXPECT_EQ(rocblaslt_arch_name_candidates("gfx1250:xnack-", kRevisionA0), reportedBase);

    // HSA_DISABLE_GFX12_STRICT=0: the stepping's own objects load, and nothing
    // else is offered -- a package that does not carry them fails rather than
    // substituting another subtree.
    const Names reportedStepping{"gfx1250-strict"};
    EXPECT_EQ(rocblaslt_arch_name_candidates("gfx1250-strict", kRevisionA0), reportedStepping);
    EXPECT_EQ(rocblaslt_arch_name_candidates("gfx1250-strict:xnack-", kRevisionA0),
              reportedStepping);
}

// The safety property, and the one worth a test of its own: an A0 must never be
// offered the base architecture. That subtree is the other revision's corpus,
// and running its kernels hangs the part -- failing to find a library is the
// better outcome.
TEST(arch_candidates, smoke_a0_is_never_offered_the_base_architecture)
{
    for(const char* reported : {"gfx1250", "gfx1250:xnack-", "gfx1250-strict",
                                "gfx1250-strict:xnack-"})
    {
        const Names candidates = rocblaslt_arch_name_candidates(reported, kRevisionA0);
        ASSERT_FALSE(candidates.empty()) << reported;
        for(const std::string& candidate : candidates)
            EXPECT_NE(candidate, "gfx1250") << reported;
    }
}

// No fallback: every case offers exactly one subtree, so a package that does not
// carry it fails instead of quietly answering with another revision's kernels.
TEST(arch_candidates, smoke_exactly_one_subtree_is_offered)
{
    for(const char* reported : {"gfx1250", "gfx1250:xnack-", "gfx1250-strict",
                                "gfx1250-strict:xnack-"})
    {
        for(int revision : {kRevisionA0, kRevisionB0, 2})
        {
            EXPECT_EQ(rocblaslt_arch_name_candidates(reported, revision).size(), 1u)
                << reported << " rev " << revision;
        }
    }
}

// A later revision is the base architecture and nothing else, so its list both
// starts and ends there.
TEST(arch_candidates, smoke_later_revisions_resolve_to_the_base_architecture)
{
    for(const char* reported : {"gfx1250", "gfx1250-strict", "gfx1250-strict:xnack-"})
    {
        for(int revision : {kRevisionB0, 2})
        {
            const Names candidates = rocblaslt_arch_name_candidates(reported, revision);
            ASSERT_FALSE(candidates.empty()) << reported << " rev " << revision;
            EXPECT_EQ(candidates.back(), "gfx1250") << reported << " rev " << revision;
        }
    }
}

// A non-zero revision has no strict variant, so nothing is offered but the base.
// Guards against a rule that keys off the name and would hand a B0 an A0 library.
TEST(arch_candidates, smoke_later_revisions_never_offer_strict)
{
    EXPECT_EQ(rocblaslt_arch_name_candidates("gfx1250", kRevisionB0), Names{"gfx1250"});
    EXPECT_EQ(rocblaslt_arch_name_candidates("gfx1250", 2), Names{"gfx1250"});

    // Even if a B0 were somehow reported under the strict name, the revision is
    // what decides -- the name is only a starting point.
    EXPECT_EQ(rocblaslt_arch_name_candidates("gfx1250-strict", kRevisionB0), Names{"gfx1250"});
}

// The stripping order: features first, stepping second. Reversed, the stepping
// test never matches a name carrying `:xnack-`, and the suffix is then appended
// to a name that already has it.
TEST(arch_candidates, smoke_no_doubled_stepping_suffix)
{
    for(const char* reported : {"gfx1250-strict", "gfx1250-strict:xnack-"})
    {
        for(const std::string& candidate : rocblaslt_arch_name_candidates(reported, kRevisionA0))
        {
            EXPECT_EQ(candidate.find("-strict-strict"), std::string::npos) << candidate;
            EXPECT_EQ(candidate.find(':'), std::string::npos) << candidate;
        }
    }
}

TEST(arch_candidates, smoke_base_name_strips_both_decorations)
{
    EXPECT_EQ(rocblaslt_arch_base_name("gfx1250"), "gfx1250");
    EXPECT_EQ(rocblaslt_arch_base_name("gfx1250-strict"), "gfx1250");
    EXPECT_EQ(rocblaslt_arch_base_name("gfx1250:xnack-"), "gfx1250");
    EXPECT_EQ(rocblaslt_arch_base_name("gfx1250-strict:sramecc+:xnack-"), "gfx1250");

    // Already-base names are unchanged, including ones with a hyphen that is part
    // of the architecture rather than a stepping.
    EXPECT_EQ(rocblaslt_arch_base_name("gfx942:sramecc+:xnack-"), "gfx942");
    EXPECT_EQ(rocblaslt_arch_base_name("gfx12-generic"), "gfx12-generic");
}

// Other architectures are not special-cased, mirroring ROCr, whose gate is on the
// revision alone. This is documentation as much as assertion: a rev-0 gfx942 does
// offer a strict candidate, which costs one failed lookup and never resolves.
TEST(arch_candidates, smoke_rule_is_architecture_agnostic)
{
    const Names expected{"gfx942-strict", "gfx942"};
    EXPECT_EQ(rocblaslt_arch_name_candidates("gfx942:sramecc+:xnack-", kRevisionA0), expected);
}
