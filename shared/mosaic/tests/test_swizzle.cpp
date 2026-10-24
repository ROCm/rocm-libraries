// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// test_swizzle.cpp
//
// Tests for the swizzle structs at include/mosaic/swizzle.hpp.
//
// Two flavors:
//   Swizzle<B, M, S>            — XOR bit-swizzle, CuTe-style
//   RotateSwizzle<BB, RS>       — additive rotation, AMD-validated
//
// Each is verified at three levels:
//   (1) Hand-computed offsets against the documented formula.
//   (2) Bijection check — the swizzle is invertible over the relevant
//       offset range (otherwise it'd lose data when used as an LDS
//       address permutation).
//   (3) Composition with a base `Layout` via `mosaic::compose`, since
//       the kernel uses `sA(m, k)` syntax through a ComposedLayout.

#include <gtest/gtest.h>

#include <mosaic/composed.hpp>
#include <mosaic/layout.hpp>
#include <mosaic/swizzle.hpp>
#include <set>

using namespace mosaic;

// =====================================================================
// Swizzle<B, M, S> — XOR pattern
// =====================================================================

TEST(Swizzle, MatchesXorFormula_3_3_3) {
    // Swizzle<3,3,3>: mask = ((off >> 6) & 7) << 3
    Swizzle<3, 3, 3> sw{};
    EXPECT_EQ(sw(0x000), 0x000u);              // no high bits → no mask
    EXPECT_EQ(sw(0x007), 0x007u);              // low 3 bits, no high bits
    EXPECT_EQ(sw(0x040), 0x040u ^ (1u << 3));  // src bit 6 set → mask 8
    EXPECT_EQ(sw(0x080), 0x080u ^ (2u << 3));  // src bit 7 set → mask 16
    EXPECT_EQ(sw(0x100), 0x100u ^ (4u << 3));  // src bit 8 set → mask 32
    EXPECT_EQ(sw(0x1C0), 0x1C0u ^ (7u << 3));  // src bits 6,7,8 set → mask 56
}

TEST(Swizzle, ComplementsItself) {
    // XOR is self-inverse: applying the same swizzle twice = identity.
    Swizzle<3, 3, 3> sw{};
    for (std::size_t offset : {0u, 7u, 0x40u, 0x100u, 0x1FFu, 0x2A5u}) {
        EXPECT_EQ(sw(sw(offset)), offset) << "self-inverse at " << offset;
    }
}

TEST(Swizzle, IsBijectionOverItsDomain) {
    // Swizzle<3,3,3> is a bijection on the 2^(B+M+S) = 2^9 = 512-offset range.
    Swizzle<3, 3, 3> sw{};
    std::set<std::size_t> seen;
    for (std::size_t i = 0; i < 512; ++i) {
        auto const out = sw(i);
        EXPECT_LT(out, 512u);
        seen.insert(out);
    }
    EXPECT_EQ(seen.size(), 512u);
}

TEST(Swizzle, ReportsSize) {
    EXPECT_EQ((Swizzle<3, 3, 3>{}.size()), std::size_t{1} << 9);
    EXPECT_EQ((Swizzle<2, 0, 4>{}.size()), std::size_t{1} << 6);
}

// =====================================================================
// RotateSwizzle<BlockBits, RowShift>
// =====================================================================

TEST(RotateSwizzle, MatchesRotationFormula_3_1) {
    // RotateSwizzle<3,1>: col = (col + (row >> 1) * 2) & 7
    // where col = offset & 7, row = offset >> 3.
    RotateSwizzle<3, 1> rs{};
    EXPECT_EQ(rs(0), 0u);    // col=0, row=0 → rot=0
    EXPECT_EQ(rs(5), 5u);    // col=5, row=0 → rot=5
    EXPECT_EQ(rs(8), 8u);    // col=0, row=1 → rot=0 (row>>1=0)
    EXPECT_EQ(rs(16), 18u);  // col=0, row=2 → rot=2
    EXPECT_EQ(rs(20), 22u);  // col=4, row=2 → rot=(4+2)&7=6
    EXPECT_EQ(rs(24), 26u);  // col=0, row=3 → rot=2 (row>>1=1)
    EXPECT_EQ(rs(32), 36u);  // col=0, row=4 → rot=4
    EXPECT_EQ(rs(40), 44u);  // col=0, row=5 → rot=4 (row>>1=2)
    EXPECT_EQ(rs(48), 54u);  // col=0, row=6 → rot=6
    EXPECT_EQ(rs(64), 64u);  // col=0, row=8 → rot=(0+8)&7=0 (cycle)
}

TEST(RotateSwizzle, PreservesHighBitsAboveBlock) {
    // The rotation only touches the low BlockBits=3 bits. Everything
    // above BlockBits should pass through unchanged.
    RotateSwizzle<3, 1> rs{};
    for (std::size_t offset : {0u, 17u, 64u, 0x200u, 0xABCDu}) {
        auto const out = rs(offset);
        EXPECT_EQ(out & ~std::size_t{7}, offset & ~std::size_t{7})
            << "high bits altered at " << offset;
    }
}

TEST(RotateSwizzle, IsBijectionWithin128ByteBlock) {
    // The rotation cycles every 16 rows (BlockBits=3 cols × RowShift=1 →
    // 8 rotation values × 2 rows each). Across 128 offsets (16 rows of 8
    // cols) the swizzle is a bijection.
    RotateSwizzle<3, 1> rs{};
    std::set<std::size_t> seen;
    for (std::size_t i = 0; i < 128; ++i) {
        seen.insert(rs(i));
    }
    EXPECT_EQ(seen.size(), 128u);
}

// =====================================================================
// Composition with a base Layout via mosaic::compose
//
// Real kernel usage: sA_layout() returns compose(swizzle, base_layout).
// Indexing sA(m, k) goes through ComposedLayout::operator()(m, k) which
// computes base_layout(m, k) (an offset) and then feeds that through the
// swizzle's operator()(size_t). End result: sA(m, k) returns a swizzled
// LDS offset.
// =====================================================================

TEST(SwizzleCompose, XorSwizzleOverRowMajor) {
    // Base: 8 × 8 row-major. compose(Swizzle<2,0,3>, base).
    auto base = make_layout(make_shape(Int<8>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{}));
    auto sw = Swizzle<2, 0, 3>{};
    auto s = compose(sw, base);

    // For a few coords, hand-verify base(m,k) → sw(offset) match.
    for (int m : {0, 1, 3, 7}) {
        for (int k : {0, 1, 4, 7}) {
            auto const raw_offset = static_cast<std::size_t>(m * 8 + k);
            auto const expected = sw(raw_offset);
            EXPECT_EQ(s(m, k), expected) << "compose mismatch at (" << m << ", " << k << ")";
        }
    }
}

TEST(SwizzleCompose, RotateSwizzleOverRowMajor) {
    // Base: 16 × 8 row-major. compose(RotateSwizzle<3,1>, base).
    // For row-major (M=16, K=8) with stride (8, 1), the flat offset is
    // m*8 + k. The swizzle's "col" is the low 3 bits = k, and "row" is
    // the high bits = m. So `s(m, k) = m*8 + ((k + (m>>1)*2) & 7)`.
    auto base = make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{}));
    auto rs = RotateSwizzle<3, 1>{};
    auto s = compose(rs, base);

    for (int m : {0, 1, 2, 3, 4, 7, 15}) {
        for (int k : {0, 1, 5, 7}) {
            std::size_t const raw = static_cast<std::size_t>(m * 8 + k);
            std::size_t const col = raw & 7;
            std::size_t const row = raw >> 3;
            std::size_t const rotated = (col + (row >> 1) * 2) & 7;
            std::size_t const expected = (raw & ~std::size_t{7}) | rotated;
            EXPECT_EQ(s(m, k), expected) << "compose mismatch at (" << m << ", " << k << ")";
        }
    }
}

TEST(SwizzleCompose, ComposedIsBijectionOverTileCoords) {
    // Compose RotateSwizzle<3,1> over a 16×8 row-major. The composed
    // layout should produce 128 distinct offsets (the natural domain
    // of the base layout) covering [0, 128).
    auto base = make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{}));
    auto rs = RotateSwizzle<3, 1>{};
    auto s = compose(rs, base);

    std::set<std::size_t> seen;
    for (int m = 0; m < 16; ++m) {
        for (int k = 0; k < 8; ++k) {
            seen.insert(s(m, k));
        }
    }
    EXPECT_EQ(seen.size(), 128u);
    // All offsets should fit in [0, 128).
    for (auto v : seen) EXPECT_LT(v, 128u);
}
