/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <cstddef>
#include <vector>

#include "framework/config_param.hpp"
#include "framework/intensity.hpp"
#include "framework/interpolation.hpp"
#include "framework/tensor_setup.hpp"

// Unit tests for framework/interpolation.hpp -- src_texel() and sample().
//
// Every geometric golden (resize, resize_crop_mirror, resize_mirror_normalize, rotate,
// warp_affine, warp_perspective, remap, fisheye, lens_correction, water, glitch, jitter,
// pixelate) inverse-maps an output coordinate and calls sample() to read the source there.
// sample() is therefore the single most load-bearing function in the reference tree, and it
// had no test of its own: its conventions were only ever exercised indirectly, through a
// whole op, against a kernel, with a tolerance in the way.
//
// Everything here is a literal expected value, hand-derived in the comment above it. No
// expectation is computed from a formula, because a formula in the test is just a second
// implementation of the thing under test.
//
// The three conventions this file pins, in decreasing order of how easy they are to get wrong:
//
//   1. BORDER RULE. The valid source region is the half-open rectangle [x0,x1) x [y0,y1), and a
//      sample outside it is resolved by the caller's BorderMode: Constant substitutes a value,
//      while Replicate / Wrap / Reflect / Reflect101 remap the coordinate back inside so every
//      tap reads real data. Constant is the default and is what every op uses today, so under
//      BILINEAR an edge sample still BLENDS WITH the border rather than replicating the edge
//      texel. Which mode each op should actually use is still open -- the RPP warp kernels
//      replicate (see docs/reference-audit/, F1/G4) -- but the mechanism to say so now exists.
//   2. NEAREST TIE-BREAKING. floor(x + 0.5), i.e. round-half-UP, so x = 0.5 selects texel 1.
//   3. COORDINATE FRAME. Coordinates are absolute within the plane; the rectangle is the ROI,
//      which may be a strict subset of the image. A texel that exists in the buffer but lies
//      outside the ROI is border, not data.

using namespace rpptest;

namespace {

constexpr Rpp32u kW = 4;
constexpr Rpp32u kH = 4;

// A sentinel that no plane below contains, so a border contribution is unmistakable in a
// failure message and shows up as a large, obviously-wrong number rather than plausible data.
constexpr double kBorder = 1000.0;

// v(x,y) = 16*y + x. Each texel names its own coordinate, so a NEAREST result reads back as
// "row 2, column 1" at a glance. Being affine in x and y, it is reproduced EXACTLY by bilinear
// interpolation, which is what makes the interior goldens below hand-checkable.
constexpr int kCoordPlane[kH][kW] = {
    {0, 1, 2, 3},
    {16, 17, 18, 19},
    {32, 33, 34, 35},
    {48, 49, 50, 51},
};

// v(x,y) = 16*x*y. A pure cross term, and the reason this second plane exists: bilinear
// reproduces the basis {1, x, y, xy}, kCoordPlane covers {1, x, y}, and only a plane with a
// non-zero xy component can catch a bug in the dx*dy weight. On an affine plane that weight's
// coefficient is zero and dropping the term entirely would go unnoticed.
constexpr int kProductPlane[kH][kW] = {
    {0, 0, 0, 0},
    {0, 16, 32, 48},
    {0, 32, 64, 96},
    {0, 48, 96, 144},
};

// A 4x4 plane built through make_descriptor(), so the row stride is padded (padded_width(4) is
// 8, not 4). Anything that walked the buffer flat instead of addressing through plane_index()
// would read the wrong element, so every case in this file also exercises that indirectly --
// and SrcTexelTest.ReadsThroughTheDescriptorStride checks it head-on.
template <typename T>
struct Plane {
    RpptDesc desc;
    std::vector<T> buf;

    const T* data() const {
        return buf.data();
    }
    std::size_t base(Rpp32u c = 0) const {
        return plane_base(desc, 0, c);
    }
};

// value(x, y, c) supplies the stored element; using from_double keeps the Rpp16f path valid.
template <typename T, typename Fn>
Plane<T> make_plane(DType dt, Layout layout, Fn value, bool pad = true) {
    Plane<T> p;
    const Rpp32u c = static_cast<Rpp32u>(channels_of(layout));
    p.desc = make_descriptor({1, c, kH, kW}, dt, layout, pad);
    p.buf.assign(element_count(p.desc), T{});
    for (Rpp32u ch = 0; ch < c; ++ch) {
        const std::size_t base = plane_base(p.desc, 0, ch);
        for (Rpp32u y = 0; y < kH; ++y)
            for (Rpp32u x = 0; x < kW; ++x)
                p.buf[plane_index(p.desc, base, y, x)] = from_double<T>(
                    value(static_cast<int>(x), static_cast<int>(y), static_cast<int>(ch)));
    }
    return p;
}

// The two planes above, as U8 PLN1, which is what most cases use.
Plane<Rpp8u> coord_plane() {
    return make_plane<Rpp8u>(DType::U8, Layout::PLN1,
                             [](int x, int y, int) { return kCoordPlane[y][x]; });
}

Plane<Rpp8u> product_plane() {
    return make_plane<Rpp8u>(DType::U8, Layout::PLN1,
                             [](int x, int y, int) { return kProductPlane[y][x]; });
}

// sample() over the whole 4x4 plane, which is the common case here.
template <typename T>
double sample_full(const Plane<T>& p, double x, double y, RpptInterpolationType interp,
                   Border border = kBorder) {
    return sample(p.data(), p.desc, p.base(), x, y, 0, 0, static_cast<int>(kW),
                  static_cast<int>(kH), interp, border);
}

}  // namespace

// ---- src_texel: the half-open rectangle and the stride ----------------------

// The rectangle is left/top INCLUSIVE and right/bottom EXCLUSIVE. Callers pass
// x1 = x0 + roiWidth, so the last valid column is x1 - 1.
TEST(SrcTexelTest, HalfOpenRectangle) {
    const Plane<Rpp8u> p = coord_plane();
    // Rectangle [1,3) x [1,3): the 2x2 block of texels (1,1) (2,1) (1,2) (2,2).
    auto at = [&](int x, int y) {
        return src_texel(p.data(), p.desc, p.base(), x, y, 1, 1, 3, 3, kBorder);
    };

    EXPECT_DOUBLE_EQ(at(1, 1), 17.0);  // v(1,1) = 16*1 + 1
    EXPECT_DOUBLE_EQ(at(2, 1), 18.0);
    EXPECT_DOUBLE_EQ(at(1, 2), 33.0);  // v(1,2) = 16*2 + 1
    EXPECT_DOUBLE_EQ(at(2, 2), 34.0);

    // Just outside on each of the four sides. Every one of these texels EXISTS in the buffer
    // (the plane is 4x4); they are border because they are outside the rectangle, which is the
    // ROI-versus-image distinction the whole geometric family depends on.
    EXPECT_DOUBLE_EQ(at(0, 1), kBorder);  // x < x0
    EXPECT_DOUBLE_EQ(at(3, 1), kBorder);  // x >= x1, so x1 is exclusive
    EXPECT_DOUBLE_EQ(at(1, 0), kBorder);  // y < y0
    EXPECT_DOUBLE_EQ(at(1, 3), kBorder);  // y >= y1
}

// The goldens address through the descriptor's strides, never by walking the buffer flat. A
// padded and a dense descriptor hold the same logical plane at different offsets, so reading
// both must give the same 16 values.
TEST(SrcTexelTest, ReadsThroughTheDescriptorStride) {
    const auto fill = [](int x, int y, int) { return kCoordPlane[y][x]; };
    const Plane<Rpp8u> padded = make_plane<Rpp8u>(DType::U8, Layout::PLN1, fill, /*pad=*/true);
    const Plane<Rpp8u> dense = make_plane<Rpp8u>(DType::U8, Layout::PLN1, fill, /*pad=*/false);

    ASSERT_EQ(padded.desc.strides.hStride, 8u);  // padded_width(4)
    ASSERT_EQ(dense.desc.strides.hStride, 4u);
    ASSERT_NE(padded.buf.size(), dense.buf.size());

    for (int y = 0; y < static_cast<int>(kH); ++y)
        for (int x = 0; x < static_cast<int>(kW); ++x)
            EXPECT_DOUBLE_EQ(
                src_texel(padded.data(), padded.desc, padded.base(), x, y, 0, 0, 4, 4, kBorder),
                src_texel(dense.data(), dense.desc, dense.base(), x, y, 0, 0, 4, 4, kBorder))
                << "at (" << x << ", " << y << ")";
}

// ---- NEAREST_NEIGHBOR ------------------------------------------------------

// At an integer coordinate the sampler must return that texel verbatim, for all 16.
TEST(SampleNearestTest, ExactTexelCentres) {
    const Plane<Rpp8u> p = coord_plane();
    EXPECT_DOUBLE_EQ(sample_full(p, 0.0, 0.0, NEAREST_NEIGHBOR), 0.0);
    EXPECT_DOUBLE_EQ(sample_full(p, 3.0, 0.0, NEAREST_NEIGHBOR), 3.0);
    EXPECT_DOUBLE_EQ(sample_full(p, 0.0, 3.0, NEAREST_NEIGHBOR), 48.0);
    EXPECT_DOUBLE_EQ(sample_full(p, 3.0, 3.0, NEAREST_NEIGHBOR), 51.0);
    EXPECT_DOUBLE_EQ(sample_full(p, 2.0, 1.0, NEAREST_NEIGHBOR), 18.0);  // v(2,1) = 16 + 2
}

// The tie rule is floor(x + 0.5): round-half-UP, so an exact .5 goes to the HIGHER texel.
// This is the convention that decides whether a resize is shifted by one pixel at particular
// size ratios, and it is invisible at any other coordinate.
TEST(SampleNearestTest, RoundsHalfUp) {
    const Plane<Rpp8u> p = coord_plane();

    // Just below the tie stays on texel 0; the tie itself moves to texel 1.
    EXPECT_DOUBLE_EQ(sample_full(p, 0.49, 0.0, NEAREST_NEIGHBOR), 0.0);  // floor(0.99) = 0
    EXPECT_DOUBLE_EQ(sample_full(p, 0.50, 0.0, NEAREST_NEIGHBOR), 1.0);  // floor(1.00) = 1
    EXPECT_DOUBLE_EQ(sample_full(p, 1.50, 0.0, NEAREST_NEIGHBOR), 2.0);  // floor(2.00) = 2
    EXPECT_DOUBLE_EQ(sample_full(p, 2.50, 0.0, NEAREST_NEIGHBOR), 3.0);  // floor(3.00) = 3

    // Same rule on the row axis: y = 0.5 selects row 1, so v(0,1) = 16.
    EXPECT_DOUBLE_EQ(sample_full(p, 0.0, 0.50, NEAREST_NEIGHBOR), 16.0);

    // Round-half-up is not symmetric about zero: x = -0.5 rounds UP to texel 0 and is therefore
    // still in range, while anything below it falls out. Worth pinning because a warp can
    // legitimately produce a small negative coordinate at the left edge.
    EXPECT_DOUBLE_EQ(sample_full(p, -0.50, 0.0, NEAREST_NEIGHBOR), 0.0);      // floor(0.0) = 0
    EXPECT_DOUBLE_EQ(sample_full(p, -0.51, 0.0, NEAREST_NEIGHBOR), kBorder);  // floor(-0.01) = -1
}

TEST(SampleNearestTest, OutOfRangeReturnsBorder) {
    const Plane<Rpp8u> p = coord_plane();

    // The last in-range coordinate on each axis, then the first out-of-range one. x = 3.5
    // rounds up to 4, which is x1, and x1 is exclusive.
    EXPECT_DOUBLE_EQ(sample_full(p, 2.99, 0.0, NEAREST_NEIGHBOR), 3.0);  // floor(3.49) = 3
    EXPECT_DOUBLE_EQ(sample_full(p, 3.50, 0.0, NEAREST_NEIGHBOR), kBorder);
    EXPECT_DOUBLE_EQ(sample_full(p, 0.0, 3.50, NEAREST_NEIGHBOR), kBorder);
    EXPECT_DOUBLE_EQ(sample_full(p, -1.0, 0.0, NEAREST_NEIGHBOR), kBorder);  // floor(-0.5) = -1
    EXPECT_DOUBLE_EQ(sample_full(p, 0.0, -1.0, NEAREST_NEIGHBOR), kBorder);
}

// ---- BILINEAR --------------------------------------------------------------

// dx = dy = 0, so the single in-range tap carries all the weight.
TEST(SampleBilinearTest, ExactTexelCentres) {
    const Plane<Rpp8u> p = coord_plane();
    EXPECT_DOUBLE_EQ(sample_full(p, 0.0, 0.0, BILINEAR), 0.0);
    EXPECT_DOUBLE_EQ(sample_full(p, 1.0, 2.0, BILINEAR), 33.0);  // v(1,2) = 32 + 1
    EXPECT_DOUBLE_EQ(sample_full(p, 2.0, 1.0, BILINEAR), 18.0);
}

// Interior weights on the affine plane, where bilinear reproduces v(x,y) = 16y + x exactly.
// Each expectation is the four-tap sum written out longhand.
TEST(SampleBilinearTest, InteriorWeights) {
    const Plane<Rpp8u> p = coord_plane();

    // (0.5, 0.0): dx = 0.5, dy = 0 -> only the top pair contributes.
    //   0*0.5 + 1*0.5 = 0.5
    EXPECT_DOUBLE_EQ(sample_full(p, 0.5, 0.0, BILINEAR), 0.5);

    // (1.5, 1.5): dx = dy = 0.5 -> the plain average of the 2x2 block at (1,1).
    //   (17 + 18 + 33 + 34) / 4 = 102 / 4 = 25.5
    EXPECT_DOUBLE_EQ(sample_full(p, 1.5, 1.5, BILINEAR), 25.5);

    // (0.25, 2.75): block at (0,2), dx = 0.25, dy = 0.75.
    //   32*0.75*0.25 + 33*0.25*0.25 + 48*0.75*0.75 + 49*0.25*0.75
    // =  6.0        +  2.0625      + 27.0         +  9.1875       = 44.25
    EXPECT_DOUBLE_EQ(sample_full(p, 0.25, 2.75, BILINEAR), 44.25);

    // (2.75, 0.25): block at (2,0), dx = 0.75, dy = 0.25.
    //   2*0.25*0.75 + 3*0.75*0.75 + 18*0.25*0.25 + 19*0.75*0.25
    // = 0.375       + 1.6875      + 1.125        + 3.5625        = 6.75
    EXPECT_DOUBLE_EQ(sample_full(p, 2.75, 0.25, BILINEAR), 6.75);
}

// The dx*dy weight, which an affine plane cannot see. On v(x,y) = 16*x*y the whole result at
// (0.5, 0.5) comes from that one term: drop it and the expectation collapses to 0.
TEST(SampleBilinearTest, CrossTermWeight) {
    const Plane<Rpp8u> p = product_plane();

    // (0.5, 0.5): block at (0,0) is {0, 0, 0, 16}; only v11 is non-zero.
    //   16 * 0.5 * 0.5 = 4
    EXPECT_DOUBLE_EQ(sample_full(p, 0.5, 0.5, BILINEAR), 4.0);

    // (0.25, 0.75): same block, asymmetric weights -- catches a dx/dy swap as well as a
    // dropped term.
    //   16 * 0.25 * 0.75 = 3
    EXPECT_DOUBLE_EQ(sample_full(p, 0.25, 0.75, BILINEAR), 3.0);

    // (1.5, 1.5): block at (1,1) = {16, 32, 32, 64}, plain average.
    //   (16 + 32 + 32 + 64) / 4 = 144 / 4 = 36
    EXPECT_DOUBLE_EQ(sample_full(p, 1.5, 1.5, BILINEAR), 36.0);

    // (2.5, 2.5): block at (2,2) = {64, 96, 96, 144}.
    //   (64 + 96 + 96 + 144) / 4 = 400 / 4 = 100
    EXPECT_DOUBLE_EQ(sample_full(p, 2.5, 2.5, BILINEAR), 100.0);
}

// At an integer coordinate on the last row/column the +1 neighbour is out of range, but its
// weight is exactly zero, so the result is still the exact texel. This is the case that makes
// "resize to the same size is a verbatim copy" hold at the far edge, and it is what
// distinguishes this from the genuinely contaminated samples in SampleBorderTest.
TEST(SampleBilinearTest, ZeroWeightNeighbourOutOfRangeIsHarmless) {
    const Plane<Rpp8u> p = coord_plane();
    EXPECT_DOUBLE_EQ(sample_full(p, 3.0, 0.0, BILINEAR), 3.0);   // (4,0) and (4,1) OOB, dx = 0
    EXPECT_DOUBLE_EQ(sample_full(p, 0.0, 3.0, BILINEAR), 48.0);  // (0,4) and (1,4) OOB, dy = 0
    EXPECT_DOUBLE_EQ(sample_full(p, 3.0, 3.0, BILINEAR), 51.0);  // three of four taps OOB
}

// ---- BorderMode::Constant (the default) ------------------------------------
//
// A bilinear tap outside [x0,x1) x [y0,y1) contributes the constant, so a fractional sample near
// the edge blends with it instead of replicating the edge texel. This is what every op does
// today, because none has opted into another mode yet. The RPP warp kernels replicate
// (rpp_cpu_interpolation.hpp clamps both taps into the ROI), which is the divergence recorded as
// F1 in docs/reference-audit/ -- SampleBorderModeTest below shows what each alternative gives for
// the same coordinates.
TEST(SampleBorderTest, ConstantBlendsWithTheBorderValue) {
    const Plane<Rpp8u> p = coord_plane();

    // (3.5, 0.0): right edge. v00 = v(3,0) = 3 at weight 0.5, v01 = (4,0) is OOB at weight 0.5.
    //   3*0.5 + 1000*0.5 = 1.5 + 500 = 501.5          (replicate would give 3)
    EXPECT_DOUBLE_EQ(sample_full(p, 3.5, 0.0, BILINEAR), 501.5);

    // (-0.5, 0.0): left edge. xa = -1, so v00 = (-1,0) is OOB at weight 0.5 and v01 = v(0,0) = 0.
    //   1000*0.5 + 0*0.5 = 500.0                      (replicate would give 0)
    EXPECT_DOUBLE_EQ(sample_full(p, -0.5, 0.0, BILINEAR), 500.0);

    // (0.0, 3.5): bottom edge. v00 = v(0,3) = 48 at weight 0.5, v10 = (0,4) is OOB.
    //   48*0.5 + 1000*0.5 = 24 + 500 = 524.0          (replicate would give 48)
    EXPECT_DOUBLE_EQ(sample_full(p, 0.0, 3.5, BILINEAR), 524.0);

    // (3.5, 3.5): the corner, where three of the four taps are out of range.
    //   51*0.25 + 1000*0.75 = 12.75 + 750 = 762.75    (replicate would give 51)
    EXPECT_DOUBLE_EQ(sample_full(p, 3.5, 3.5, BILINEAR), 762.75);
}

// A bare double still means Constant, which is what keeps every pre-existing call site --
// `sample(..., dtype_black(dt))` -- meaning exactly what it meant before BorderMode existed.
TEST(SampleBorderTest, BareDoubleMeansConstant) {
    const Plane<Rpp8u> p = coord_plane();
    EXPECT_DOUBLE_EQ(sample_full(p, 5.0, 5.0, NEAREST_NEIGHBOR, 42.0), 42.0);
    EXPECT_EQ(Border{7.5}.mode, BorderMode::Constant);
    EXPECT_DOUBLE_EQ(Border{7.5}.value, 7.5);
}

// ---- border_index: the remapping math on its own ---------------------------
//
// Testing the index map directly, rather than only through a sampled value, is what makes the
// multi-period cases legible: `border_index(-5, 4, Wrap) == 3` says what it means, where the
// equivalent sampled expectation would just be a number.
//
// Each mode is checked at least two periods out, because a rule written for a single step
// outside (a lone clamp, or one subtraction) passes a +/-1 test and is wrong everywhere else.

TEST(BorderIndexTest, Replicate) {
    // aaa|abcd|ddd -- everything left of the range is index 0, everything right is n-1.
    EXPECT_EQ(border_index(-1, 4, BorderMode::Replicate), 0);
    EXPECT_EQ(border_index(-7, 4, BorderMode::Replicate), 0);
    EXPECT_EQ(border_index(4, 4, BorderMode::Replicate), 3);
    EXPECT_EQ(border_index(99, 4, BorderMode::Replicate), 3);
}

TEST(BorderIndexTest, Wrap) {
    // bcd|abcd|abc -- period n, in both directions.
    EXPECT_EQ(border_index(-1, 4, BorderMode::Wrap), 3);
    EXPECT_EQ(border_index(-4, 4, BorderMode::Wrap), 0);
    EXPECT_EQ(border_index(-5, 4, BorderMode::Wrap), 3);  // C++ gives -5 % 4 == -1; must not leak
    EXPECT_EQ(border_index(4, 4, BorderMode::Wrap), 0);
    EXPECT_EQ(border_index(5, 4, BorderMode::Wrap), 1);
    EXPECT_EQ(border_index(8, 4, BorderMode::Wrap), 0);
}

TEST(BorderIndexTest, Reflect) {
    // cba|abcd|dcb -- period 2n, and the edge index appears twice at the fold (-1 -> 0, 4 -> 3).
    EXPECT_EQ(border_index(-1, 4, BorderMode::Reflect), 0);
    EXPECT_EQ(border_index(-2, 4, BorderMode::Reflect), 1);
    EXPECT_EQ(border_index(-3, 4, BorderMode::Reflect), 2);
    EXPECT_EQ(border_index(-4, 4, BorderMode::Reflect), 3);
    EXPECT_EQ(border_index(-5, 4, BorderMode::Reflect), 3);  // second fold, back outwards
    EXPECT_EQ(border_index(4, 4, BorderMode::Reflect), 3);
    EXPECT_EQ(border_index(5, 4, BorderMode::Reflect), 2);
    EXPECT_EQ(border_index(8, 4, BorderMode::Reflect), 0);  // one full period
}

TEST(BorderIndexTest, Reflect101) {
    // dcb|abcd|cba -- period 2n-2, and the edge index appears once (-1 -> 1, 4 -> 2).
    EXPECT_EQ(border_index(-1, 4, BorderMode::Reflect101), 1);
    EXPECT_EQ(border_index(-2, 4, BorderMode::Reflect101), 2);
    EXPECT_EQ(border_index(-3, 4, BorderMode::Reflect101), 3);
    EXPECT_EQ(border_index(-4, 4, BorderMode::Reflect101), 2);  // fold back
    EXPECT_EQ(border_index(-5, 4, BorderMode::Reflect101), 1);
    EXPECT_EQ(border_index(4, 4, BorderMode::Reflect101), 2);
    EXPECT_EQ(border_index(5, 4, BorderMode::Reflect101), 1);
    EXPECT_EQ(border_index(6, 4, BorderMode::Reflect101), 0);  // one full period
    EXPECT_EQ(border_index(7, 4, BorderMode::Reflect101), 1);
}

// Every mode leaves an already-valid index alone, which is what lets src_texel() remap both axes
// unconditionally once either one is out of range.
TEST(BorderIndexTest, IdentityInsideTheRange) {
    for (int r = 0; r < 4; ++r) {
        EXPECT_EQ(border_index(r, 4, BorderMode::Replicate), r);
        EXPECT_EQ(border_index(r, 4, BorderMode::Wrap), r);
        EXPECT_EQ(border_index(r, 4, BorderMode::Reflect), r);
        EXPECT_EQ(border_index(r, 4, BorderMode::Reflect101), r);
    }
}

// n == 1 has only one answer, and Reflect101's period (2n-2) would be zero there -- a modulo by
// zero rather than a wrong result. A 1-pixel ROI is reachable (make_roi halves the extents), so
// this is a real case and not a hypothetical.
TEST(BorderIndexTest, DegenerateSingleElementRange) {
    for (int r : {-3, -1, 0, 1, 5}) {
        EXPECT_EQ(border_index(r, 1, BorderMode::Replicate), 0) << "r = " << r;
        EXPECT_EQ(border_index(r, 1, BorderMode::Wrap), 0) << "r = " << r;
        EXPECT_EQ(border_index(r, 1, BorderMode::Reflect), 0) << "r = " << r;
        EXPECT_EQ(border_index(r, 1, BorderMode::Reflect101), 0) << "r = " << r;
    }
}

// ---- The modes, through sample() -------------------------------------------
//
// Row 0 of kCoordPlane is {0, 1, 2, 3}, so a NEAREST sample there returns the column index it
// landed on: the expectation and the remapped coordinate are the same number, and no derivation
// is needed to read these.

TEST(SampleBorderModeTest, NearestOutsideTheRectangleOnEachAxis) {
    const Plane<Rpp8u> p = coord_plane();
    auto col = [&](double x, BorderMode m) {
        return sample_full(p, x, 0.0, NEAREST_NEIGHBOR, Border{m});
    };
    // Column 0 holds {0, 16, 32, 48}, so a row index reads back as 16*row.
    auto row = [&](double y, BorderMode m) {
        return sample_full(p, 0.0, y, NEAREST_NEIGHBOR, Border{m});
    };

    EXPECT_DOUBLE_EQ(col(-1.0, BorderMode::Replicate), 0.0);
    EXPECT_DOUBLE_EQ(col(-3.0, BorderMode::Replicate), 0.0);
    EXPECT_DOUBLE_EQ(col(4.0, BorderMode::Replicate), 3.0);
    EXPECT_DOUBLE_EQ(col(6.0, BorderMode::Replicate), 3.0);

    EXPECT_DOUBLE_EQ(col(-1.0, BorderMode::Wrap), 3.0);
    EXPECT_DOUBLE_EQ(col(-5.0, BorderMode::Wrap), 3.0);
    EXPECT_DOUBLE_EQ(col(4.0, BorderMode::Wrap), 0.0);
    EXPECT_DOUBLE_EQ(col(5.0, BorderMode::Wrap), 1.0);

    EXPECT_DOUBLE_EQ(col(-1.0, BorderMode::Reflect), 0.0);
    EXPECT_DOUBLE_EQ(col(-2.0, BorderMode::Reflect), 1.0);
    EXPECT_DOUBLE_EQ(col(4.0, BorderMode::Reflect), 3.0);
    EXPECT_DOUBLE_EQ(col(5.0, BorderMode::Reflect), 2.0);

    EXPECT_DOUBLE_EQ(col(-1.0, BorderMode::Reflect101), 1.0);
    EXPECT_DOUBLE_EQ(col(-2.0, BorderMode::Reflect101), 2.0);
    EXPECT_DOUBLE_EQ(col(4.0, BorderMode::Reflect101), 2.0);
    EXPECT_DOUBLE_EQ(col(5.0, BorderMode::Reflect101), 1.0);

    // The row axis is resolved by the same rule, independently.
    EXPECT_DOUBLE_EQ(row(-1.0, BorderMode::Replicate), 0.0);   // row 0
    EXPECT_DOUBLE_EQ(row(4.0, BorderMode::Replicate), 48.0);   // row 3
    EXPECT_DOUBLE_EQ(row(-1.0, BorderMode::Wrap), 48.0);       // row 3
    EXPECT_DOUBLE_EQ(row(4.0, BorderMode::Wrap), 0.0);         // row 0
    EXPECT_DOUBLE_EQ(row(-1.0, BorderMode::Reflect), 0.0);     // row 0
    EXPECT_DOUBLE_EQ(row(5.0, BorderMode::Reflect), 32.0);     // row 2
    EXPECT_DOUBLE_EQ(row(-1.0, BorderMode::Reflect101), 16.0); // row 1
    EXPECT_DOUBLE_EQ(row(4.0, BorderMode::Reflect101), 32.0);  // row 2
}

// (3.5, 0.0): v00 = v(3,0) = 3 and v01 = texel (4,0), each at weight 0.5, dy = 0. Under a
// remapping mode no border colour enters at all -- the result is a blend of two real texels.
//
// Reflect and Replicate agree here, because one step outside is exactly where they agree: both
// fold back onto the edge texel. That is not a coincidence to be surprised by later, so the next
// test pushes one step further.
TEST(SampleBorderModeTest, BilinearOneStepOutside) {
    const Plane<Rpp8u> p = coord_plane();
    auto at = [&](BorderMode m) { return sample_full(p, 3.5, 0.0, BILINEAR, Border{m}); };

    EXPECT_DOUBLE_EQ(at(BorderMode::Replicate), 3.0);   // (3 + 3) / 2, tap folds to column 3
    EXPECT_DOUBLE_EQ(at(BorderMode::Wrap), 1.5);        // (3 + 0) / 2, tap wraps to column 0
    EXPECT_DOUBLE_EQ(at(BorderMode::Reflect), 3.0);     // (3 + 3) / 2, edge texel repeated
    EXPECT_DOUBLE_EQ(at(BorderMode::Reflect101), 2.5);  // (3 + 2) / 2, edge texel not repeated
    EXPECT_DOUBLE_EQ(at(BorderMode::Constant), 1.5);    // (3 + 0) / 2, dtype_black default of 0
}

// (4.5, 0.0) is a full texel further out, so BOTH taps are outside: v00 = texel (4,0) and
// v01 = texel (5,0), weights 0.5 each. This is where all four remapping modes separate, and it
// is the case that a single-step implementation of any of them would get wrong.
TEST(SampleBorderModeTest, BilinearTwoStepsOutsideSeparatesEveryMode) {
    const Plane<Rpp8u> p = coord_plane();
    auto at = [&](BorderMode m) { return sample_full(p, 4.5, 0.0, BILINEAR, Border{m}); };

    EXPECT_DOUBLE_EQ(at(BorderMode::Replicate), 3.0);   // cols 3, 3 -> (3 + 3) / 2
    EXPECT_DOUBLE_EQ(at(BorderMode::Wrap), 0.5);        // cols 0, 1 -> (0 + 1) / 2
    EXPECT_DOUBLE_EQ(at(BorderMode::Reflect), 2.5);     // cols 3, 2 -> (3 + 2) / 2
    EXPECT_DOUBLE_EQ(at(BorderMode::Reflect101), 1.5);  // cols 2, 1 -> (2 + 1) / 2
}

// The mode only governs what happens outside the rectangle; an interior sample is the same
// number under all five. Without this, a mode that quietly perturbed in-range taps would look
// like a border-handling difference.
TEST(SampleBorderModeTest, InRangeSamplesAreUnaffectedByMode) {
    const Plane<Rpp8u> p = coord_plane();
    for (BorderMode m : {BorderMode::Constant, BorderMode::Replicate, BorderMode::Wrap,
                         BorderMode::Reflect, BorderMode::Reflect101}) {
        EXPECT_DOUBLE_EQ(sample_full(p, 1.5, 1.5, BILINEAR, Border{m}), 25.5);
        EXPECT_DOUBLE_EQ(sample_full(p, 0.25, 2.75, BILINEAR, Border{m}), 44.25);
        EXPECT_DOUBLE_EQ(sample_full(p, 2.0, 1.0, NEAREST_NEIGHBOR, Border{m}), 18.0);
        // The last column with dx = 0: the out-of-range neighbour has zero weight, so the
        // result is the exact texel no matter how that neighbour would have been resolved.
        EXPECT_DOUBLE_EQ(sample_full(p, 3.0, 0.0, BILINEAR, Border{m}), 3.0);
    }
}

// The period is the ROI's extent, not the image's. With ROI [1,3) the range is two texels wide,
// so a coordinate one step outside resolves within {17, 18} -- never to texel (0,1) = 16 or
// (3,1) = 19, which exist in the buffer and are exactly what a remap written against the image
// bounds would return.
TEST(SampleBorderModeTest, RemappingIsRelativeToTheRoiNotTheImage) {
    const Plane<Rpp8u> p = coord_plane();
    auto at = [&](double x, BorderMode m) {
        return sample(p.data(), p.desc, p.base(), x, 1.0, 1, 1, 3, 3, NEAREST_NEIGHBOR, Border{m});
    };

    // One step left of the ROI (column 0). n = 2, so: replicate -> col 1, wrap -> col 2,
    // reflect -> col 1 (edge repeated), reflect101 -> col 2.
    EXPECT_DOUBLE_EQ(at(0.0, BorderMode::Replicate), 17.0);
    EXPECT_DOUBLE_EQ(at(0.0, BorderMode::Wrap), 18.0);
    EXPECT_DOUBLE_EQ(at(0.0, BorderMode::Reflect), 17.0);
    EXPECT_DOUBLE_EQ(at(0.0, BorderMode::Reflect101), 18.0);

    // One step right of the ROI (column 3), the mirror of the above.
    EXPECT_DOUBLE_EQ(at(3.0, BorderMode::Replicate), 18.0);
    EXPECT_DOUBLE_EQ(at(3.0, BorderMode::Wrap), 17.0);
    EXPECT_DOUBLE_EQ(at(3.0, BorderMode::Reflect), 18.0);
    EXPECT_DOUBLE_EQ(at(3.0, BorderMode::Reflect101), 17.0);
}

// A 1x1 ROI: there is exactly one texel, so every remapping mode must return it from anywhere.
// This is the sampler-level counterpart of BorderIndexTest.DegenerateSingleElementRange, and the
// case where Reflect101's period collapses to zero.
TEST(SampleBorderModeTest, SingleTexelRoi) {
    const Plane<Rpp8u> p = coord_plane();
    for (BorderMode m : {BorderMode::Replicate, BorderMode::Wrap, BorderMode::Reflect,
                         BorderMode::Reflect101}) {
        for (double x : {-2.0, 0.0, 1.0, 3.0}) {
            EXPECT_DOUBLE_EQ(sample(p.data(), p.desc, p.base(), x, 1.0, 1, 1, 2, 2,
                                    NEAREST_NEIGHBOR, Border{m}),
                             17.0)
                << "x = " << x;
        }
    }
}

// ---- The rectangle is the ROI, not the image -------------------------------

// With a sub-rectangle, texels that exist in the buffer but sit outside it are border. Getting
// this wrong is the "clamp to the image instead of the ROI" defect that shows up across the
// filter and warp families as a PartialRoi-only failure.
TEST(SampleRoiTest, TexelsOutsideTheRoiAreBorderEvenThoughTheyExist) {
    const Plane<Rpp8u> p = coord_plane();
    // ROI [1,3) x [1,3) -- the interior 2x2 block.
    auto at = [&](double x, double y, RpptInterpolationType interp) {
        return sample(p.data(), p.desc, p.base(), x, y, 1, 1, 3, 3, interp, kBorder);
    };

    EXPECT_DOUBLE_EQ(at(1.0, 1.0, NEAREST_NEIGHBOR), 17.0);
    EXPECT_DOUBLE_EQ(at(2.0, 2.0, NEAREST_NEIGHBOR), 34.0);

    // Texel (0,1) holds 16 and is present in the buffer, but the ROI starts at x0 = 1.
    EXPECT_DOUBLE_EQ(at(0.0, 1.0, NEAREST_NEIGHBOR), kBorder);
    // Texel (3,1) holds 19 and is present too, but x1 = 3 is exclusive.
    EXPECT_DOUBLE_EQ(at(3.0, 1.0, NEAREST_NEIGHBOR), kBorder);

    // The ROI interior interpolates exactly as it would on the full plane: the 2x2 block at
    // (1,1) is entirely inside, so this is the same 25.5 as SampleBilinearTest.InteriorWeights.
    EXPECT_DOUBLE_EQ(at(1.5, 1.5, BILINEAR), 25.5);

    // (2.5, 1.0) straddles the ROI's right edge: v00 = v(2,1) = 18 at weight 0.5, and v01 =
    // (3,1) is outside the ROI at weight 0.5 -- even though it holds real data.
    //   18*0.5 + 1000*0.5 = 9 + 500 = 509.0
    EXPECT_DOUBLE_EQ(at(2.5, 1.0, BILINEAR), 509.0);
}

// ---- Layout and dtype independence -----------------------------------------

// sample() reaches elements through plane_base/plane_index, so the packed and planar layouts
// must give identical results for identical logical content. Channel c holds v(x,y) + 100*c,
// which also proves the per-channel plane origin is right rather than always channel 0.
TEST(SampleTest, LayoutAgnostic) {
    const auto fill = [](int x, int y, int c) { return kCoordPlane[y][x] + 100 * c; };
    const Plane<Rpp8u> pkd = make_plane<Rpp8u>(DType::U8, Layout::PKD3, fill);
    const Plane<Rpp8u> pln = make_plane<Rpp8u>(DType::U8, Layout::PLN3, fill);

    ASSERT_EQ(pkd.desc.strides.wStride, 3u);  // interleaved
    ASSERT_EQ(pln.desc.strides.wStride, 1u);  // planar

    // The same interior golden as InteriorWeights, offset per channel: 44.25 + 100*c.
    const double expected[3] = {44.25, 144.25, 244.25};
    for (Rpp32u c = 0; c < 3; ++c) {
        EXPECT_DOUBLE_EQ(sample(pkd.data(), pkd.desc, pkd.base(c), 0.25, 2.75, 0, 0, 4, 4, BILINEAR,
                                kBorder),
                         expected[c])
            << "PKD3 channel " << c;
        EXPECT_DOUBLE_EQ(sample(pln.data(), pln.desc, pln.base(c), 0.25, 2.75, 0, 0, 4, 4, BILINEAR,
                                kBorder),
                         expected[c])
            << "PLN3 channel " << c;
    }
}

// interpolation.hpp claims that because interpolation is affine it commutes with the I8
// intensity offset, and therefore needs no unit conversion. That is a real claim about the
// sampler and it is cheap to check: the same plane stored as I8 (every value shifted by -128)
// must produce exactly the U8 result shifted by -128.
TEST(SampleTest, I8OffsetCommutesWithInterpolation) {
    const Plane<Rpp8s> p = make_plane<Rpp8s>(
        DType::I8, Layout::PLN1, [](int x, int y, int) { return kCoordPlane[y][x] - 128; });

    EXPECT_DOUBLE_EQ(sample_full(p, 1.5, 1.5, BILINEAR), 25.5 - 128.0);      // -102.5
    EXPECT_DOUBLE_EQ(sample_full(p, 0.25, 2.75, BILINEAR), 44.25 - 128.0);   // -83.75
    EXPECT_DOUBLE_EQ(sample_full(p, 2.0, 1.0, NEAREST_NEIGHBOR), 18.0 - 128.0);
}

// The sampler is dtype-generic and works in stored units, so an F32 plane interpolates the same
// way. Values are k/64, which is dyadic and therefore exact in float, so the result is exact
// too -- no tolerance is needed or wanted here.
TEST(SampleTest, F32StoredUnits) {
    const Plane<Rpp32f> p = make_plane<Rpp32f>(
        DType::F32, Layout::PLN1, [](int x, int y, int) { return kCoordPlane[y][x] / 64.0; });

    EXPECT_DOUBLE_EQ(sample_full(p, 1.5, 1.5, BILINEAR), 25.5 / 64.0);     // 0.3984375
    EXPECT_DOUBLE_EQ(sample_full(p, 0.25, 2.75, BILINEAR), 44.25 / 64.0);  // 0.69140625
}

// The geometric drivers pass dtype_black(dt) as the border, so the out-of-frame fill is 0 for
// U8/F16/F32 and -128 for I8 -- both of which are "black" in their own stored space. An op that
// filled I8 with 0 would be filling with mid-grey, which is a real defect seen in the kernels.
TEST(SampleTest, DtypeBlackIsTheGeometricBorderFill) {
    const Plane<Rpp8u> u8 = coord_plane();
    const Plane<Rpp8s> i8 = make_plane<Rpp8s>(
        DType::I8, Layout::PLN1, [](int x, int y, int) { return kCoordPlane[y][x] - 128; });

    EXPECT_DOUBLE_EQ(sample_full(u8, 5.0, 5.0, NEAREST_NEIGHBOR, dtype_black(DType::U8)), 0.0);
    EXPECT_DOUBLE_EQ(sample_full(i8, 5.0, 5.0, NEAREST_NEIGHBOR, dtype_black(DType::I8)), -128.0);
}
