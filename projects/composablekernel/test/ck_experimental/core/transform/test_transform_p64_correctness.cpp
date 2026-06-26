// Positive P64 correctness tests for the V4 transform framework.
//
// Builds graphs with `make_transform_graph<Precision64>(...)` and drives them
// with coordinate / parameter values ABOVE 2^32. Each result is checked at
// runtime via gtest expectations.
//
// If any width were truncated to 32 bits (a field, State member, or cast left
// int32), the computed value would wrap below 2^32 and the EXPECT would fail.
//
// Covered: OFFSET, SLICE, FREEZE, EMBED, UNMERGE, MERGE (64-bit magic
// division), XOR, PAD. BROADCAST writes no output coordinate, so it has no
// >2^32 value to assert; its P64 path is covered structurally by the graphs
// below compiling and running under Precision64.

#include "ck_experimental/core/transform/v4_experimental.hpp"

#include <gtest/gtest.h>

#include <cstdint>

namespace {
namespace v4 = ck_tile::core::transform::v4;
using ck_tile::index_t;

/// 2^32 == 4'294'967'296. Every expected result below exceeds this, so a
/// 32-bit truncation anywhere on the path would change the value.
constexpr int64_t kTwoPow32 = int64_t{1} << 32;

// -- OFFSET: out = in + shift ------------------------------------------------
constexpr index_t kOffIn = 0, kOffOut = 1;
constexpr auto g_offset = v4::make_transform_graph<v4::Precision64>(
    v4::outputs(v4::read(kOffOut)),
    v4::make_offset(int64_t{5'000'000'000}, v4::read(kOffIn), v4::write(kOffOut)),
    v4::inputs(v4::dims(uint64_t{16}), v4::write(kOffIn)));

int64_t run_offset()
{
    auto    gb     = v4::make_graph_bindings<g_offset>();
    int64_t in[1]  = {int64_t{1'000'000'000}};
    int64_t out[1] = {0};
    v4::mapCoord<g_offset>(out, in, gb);
    return out[0];
}

// -- SLICE: out = in + begin -------------------------------------------------
constexpr index_t kSlIn = 0, kSlOut = 1;
constexpr auto g_slice = v4::make_transform_graph<v4::Precision64>(
    v4::outputs(v4::read(kSlOut)),
    v4::make_slice(int64_t{5'000'000'000}, int64_t{6'000'000'000},
                   v4::read(kSlIn), v4::write(kSlOut)),
    v4::inputs(v4::dims(uint64_t{7'000'000'000}), v4::write(kSlIn)));

int64_t run_slice()
{
    auto    gb     = v4::make_graph_bindings<g_slice>();
    int64_t in[1]  = {int64_t{1'000'000'000}};
    int64_t out[1] = {0};
    v4::mapCoord<g_slice>(out, in, gb);
    return out[0];
}

// -- FREEZE: out = frozen_idx (no input) -------------------------------------
constexpr index_t kFrOut = 0;
constexpr auto g_freeze = v4::make_transform_graph<v4::Precision64>(
    v4::outputs(v4::read(kFrOut)),
    v4::make_freeze(int64_t{7'000'000'000}, v4::read(), v4::write(kFrOut)),
    v4::inputs(v4::write()));

int64_t run_freeze()
{
    auto    gb     = v4::make_graph_bindings<g_freeze>();
    int64_t in[1]  = {0};   // unused: FREEZE reads no input
    int64_t out[1] = {0};
    v4::mapCoord<g_freeze>(out, in, gb);
    return out[0];
}

// -- EMBED: out = in * stride (single dim) -----------------------------------
constexpr index_t kEmIn = 0, kEmOut = 1;
constexpr auto g_embed = v4::make_transform_graph<v4::Precision64>(
    v4::outputs(v4::read(kEmOut)),
    v4::make_embed(v4::dims(uint64_t{8}), v4::strides(int64_t{3'000'000'000}),
                   v4::read(kEmIn), v4::write(kEmOut)),
    v4::inputs(v4::dims(uint64_t{8}), v4::write(kEmIn)));

int64_t run_embed()
{
    auto    gb     = v4::make_graph_bindings<g_embed>();
    int64_t in[1]  = {int64_t{2}};
    int64_t out[1] = {0};
    v4::mapCoord<g_embed>(out, in, gb);
    return out[0];
}

// -- UNMERGE: out = in0 * derived_stride0 + in1 (additive; no magic div) -----
// component_lengths = [4, 3e9] => derived_strides = [3e9, 1]. A >2^31 stride
// would have tripped the old INT32_MAX overflow guard; the precision-aware cap
// now admits it for P64.
constexpr index_t kUmIn0 = 0, kUmIn1 = 1, kUmOut = 2;
constexpr auto g_unmerge = v4::make_transform_graph<v4::Precision64>(
    v4::outputs(v4::read(kUmOut)),
    v4::make_unmerge(v4::dims(uint64_t{4}, uint64_t{3'000'000'000}),
                     v4::read(kUmIn0, kUmIn1), v4::write(kUmOut)),
    v4::inputs(v4::write(kUmIn0, kUmIn1)));

int64_t run_unmerge()
{
    auto    gb     = v4::make_graph_bindings<g_unmerge>();
    int64_t in[2]  = {int64_t{2}, int64_t{0}};
    int64_t out[1] = {0};
    v4::mapCoord<g_unmerge>(out, in, gb);
    return out[0];
}

// -- MERGE: decompose a >2^32 index via 64-bit magic division ----------------
// components [4, 3e9] => derived_strides [3e9, 1]. Decomposing idx = 6e9:
//   out0 = 6e9 / 3e9 = 2   (magic division by a >2^31 divisor on a >2^32 idx)
//   out1 = 6e9 - 2*3e9 = 0
constexpr index_t kMgIn = 0, kMgOut0 = 1, kMgOut1 = 2;
constexpr auto g_merge = v4::make_transform_graph<v4::Precision64>(
    v4::outputs(v4::read(kMgOut0, kMgOut1)),
    v4::make_merge(v4::dims(uint64_t{4}, uint64_t{3'000'000'000}),
                   v4::read(kMgIn), v4::write(kMgOut0, kMgOut1)),
    v4::inputs(v4::write(kMgIn)));

void run_merge(int64_t out[2])
{
    auto    gb    = v4::make_graph_bindings<g_merge>();
    int64_t in[1] = {int64_t{6'000'000'000}};
    out[0] = 0;
    out[1] = 0;
    v4::mapCoord<g_merge>(out, in, gb);
}

// -- XOR: out0 = in0; out1 = in1 ^ (in0 % length_1) --------------------------
// 2 dims -> 2 dims. Drives the 64-bit modulo (in0 % length_1 with a >2^32
// divisor) and a >2^32 XOR result. length_1 = 5e9, in0 = 7e9 => 7e9 % 5e9 =
// 2e9; in1 = 2^34. 2e9 (high bit 30) and 2^34 share no bits, so the XOR equals
// the sum: out1 = 2^34 + 2e9 = 19'179'869'184. out0 = in0 = 7e9.
constexpr index_t kXIn0 = 0, kXIn1 = 1, kXOut0 = 2, kXOut1 = 3;
constexpr auto g_xor = v4::make_transform_graph<v4::Precision64>(
    v4::outputs(v4::read(kXOut0, kXOut1)),
    v4::make_xor(v4::read(kXIn0, kXIn1), v4::write(kXOut0, kXOut1)),
    v4::inputs(v4::dims(uint64_t{8'000'000'000}, uint64_t{5'000'000'000}),
               v4::write(kXIn0, kXIn1)));

void run_xor(int64_t out[2])
{
    auto    gb    = v4::make_graph_bindings<g_xor>();
    int64_t in[2] = {int64_t{7'000'000'000}, int64_t{17'179'869'184}};  // in1 = 2^34
    out[0] = 0;
    out[1] = 0;
    v4::mapCoord<g_xor>(out, in, gb);
}

// -- PAD: out = in - left_pad ------------------------------------------------
// left_pad = 5e9 (a >2^32 length), in = 11e9 => out = 6e9. The output edge
// length is DERIVED (in_len + left_pad + right_pad), also exercised >2^32.
constexpr index_t kPIn = 0, kPOut = 1;
constexpr auto g_pad = v4::make_transform_graph<v4::Precision64>(
    v4::outputs(v4::read(kPOut)),
    v4::make_pad(uint64_t{5'000'000'000}, uint64_t{1'000'000'000},
                 v4::read(kPIn), v4::write(kPOut)),
    v4::inputs(v4::dims(uint64_t{12'000'000'000}), v4::write(kPIn)));

int64_t run_pad()
{
    auto    gb     = v4::make_graph_bindings<g_pad>();
    int64_t in[1]  = {int64_t{11'000'000'000}};
    int64_t out[1] = {0};
    v4::mapCoord<g_pad>(out, in, gb);
    return out[0];
}

} // namespace

TEST(V4Precision64, Offset)
{
    const int64_t got = run_offset();
    EXPECT_EQ(got, int64_t{6'000'000'000});
    EXPECT_GT(got, kTwoPow32) << "result <= 2^32: likely 32-bit truncation";
}

TEST(V4Precision64, Slice)
{
    const int64_t got = run_slice();
    EXPECT_EQ(got, int64_t{6'000'000'000});
    EXPECT_GT(got, kTwoPow32) << "result <= 2^32: likely 32-bit truncation";
}

TEST(V4Precision64, Freeze)
{
    const int64_t got = run_freeze();
    EXPECT_EQ(got, int64_t{7'000'000'000});
    EXPECT_GT(got, kTwoPow32) << "result <= 2^32: likely 32-bit truncation";
}

TEST(V4Precision64, Embed)
{
    const int64_t got = run_embed();
    EXPECT_EQ(got, int64_t{6'000'000'000});
    EXPECT_GT(got, kTwoPow32) << "result <= 2^32: likely 32-bit truncation";
}

TEST(V4Precision64, Unmerge)
{
    const int64_t got = run_unmerge();
    EXPECT_EQ(got, int64_t{6'000'000'000});
    EXPECT_GT(got, kTwoPow32) << "result <= 2^32: likely 32-bit truncation";
}

TEST(V4Precision64, Merge)
{
    int64_t out[2] = {0, 0};
    run_merge(out);
    EXPECT_EQ(out[0], int64_t{2});  // 6e9 / 3e9 via 64-bit magic division
    EXPECT_EQ(out[1], int64_t{0});
}

TEST(V4Precision64, Xor)
{
    int64_t out[2] = {0, 0};
    run_xor(out);
    EXPECT_EQ(out[0], int64_t{7'000'000'000});
    EXPECT_EQ(out[1], int64_t{19'179'869'184});
    EXPECT_GT(out[1], kTwoPow32) << "result <= 2^32: likely 32-bit truncation";
}

TEST(V4Precision64, Pad)
{
    const int64_t got = run_pad();
    EXPECT_EQ(got, int64_t{6'000'000'000});
    EXPECT_GT(got, kTwoPow32) << "result <= 2^32: likely 32-bit truncation";
}
