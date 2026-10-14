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

#include "framework/interpolation.hpp"

#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <cstddef>
#include <vector>

#include "framework/config_param.hpp"
#include "framework/intensity.hpp"
#include "framework/tensor_setup.hpp"

// Unit tests for framework/interpolation.hpp: src_texel(), border_index() and sample(), shared by
// every geometric golden. Expected values are literals, hand-derived next to each check.
//
// Conventions pinned here:
//   1. The valid region is the half-open rectangle [x0,x1) x [y0,y1); outside it the BorderMode
//      applies. Under Constant (the default) a BILINEAR edge sample blends with the border value.
//   2. NEAREST rounds half up: floor(x + 0.5).
//   3. Coordinates are absolute within the plane and the rectangle is the ROI, so a texel outside
//      the ROI is border even though it exists in the buffer.

using namespace rpptest;

namespace {

constexpr Rpp32u kW = 4;
constexpr Rpp32u kH = 4;

// A sentinel no plane contains, so a border contribution is unmistakable.
constexpr double kBorder = 1000.0;

// v(x,y) = 16*y + x: each texel names its coordinate, and bilinear reproduces it exactly.
constexpr int kCoordPlane[kH][kW] = {
    {0, 1, 2, 3},
    {16, 17, 18, 19},
    {32, 33, 34, 35},
    {48, 49, 50, 51},
};

// v(x,y) = 16*x*y: a pure cross term, which an affine plane cannot use to check the dx*dy weight.
constexpr int kProductPlane[kH][kW] = {
    {0, 0, 0, 0},
    {0, 16, 32, 48},
    {0, 32, 64, 96},
    {0, 48, 96, 144},
};

// A 4x4 plane built through make_descriptor(), so the row stride is padded (8, not 4).
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

// The two planes above, as U8 PLN1.
Plane<Rpp8u> coord_plane() {
    return make_plane<Rpp8u>(DType::U8, Layout::PLN1,
                             [](int x, int y, int) { return kCoordPlane[y][x]; });
}

Plane<Rpp8u> product_plane() {
    return make_plane<Rpp8u>(DType::U8, Layout::PLN1,
                             [](int x, int y, int) { return kProductPlane[y][x]; });
}

// sample() over the whole 4x4 plane.
template <typename T>
double sample_full(const Plane<T>& p, double x, double y, RpptInterpolationType interp,
                   Border border = kBorder) {
    return sample(p.data(), p.desc, p.base(), x, y, 0, 0, static_cast<int>(kW),
                  static_cast<int>(kH), interp, border);
}

}  // namespace

// ---- src_texel: the half-open rectangle and the stride ----------------------

// The rectangle is left/top inclusive and right/bottom exclusive (x1 = x0 + roiWidth).
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

    // Just outside each side: these texels exist in the buffer but are outside the rectangle.
    EXPECT_DOUBLE_EQ(at(0, 1), kBorder);  // x < x0
    EXPECT_DOUBLE_EQ(at(3, 1), kBorder);  // x >= x1, so x1 is exclusive
    EXPECT_DOUBLE_EQ(at(1, 0), kBorder);  // y < y0
    EXPECT_DOUBLE_EQ(at(1, 3), kBorder);  // y >= y1
}

// A padded and a dense descriptor hold the same logical plane, so both must read the same values.
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

// At an integer coordinate the sampler returns that texel verbatim.
TEST(SampleNearestTest, ExactTexelCentres) {
    const Plane<Rpp8u> p = coord_plane();
    EXPECT_DOUBLE_EQ(sample_full(p, 0.0, 0.0, NEAREST_NEIGHBOR), 0.0);
    EXPECT_DOUBLE_EQ(sample_full(p, 3.0, 0.0, NEAREST_NEIGHBOR), 3.0);
    EXPECT_DOUBLE_EQ(sample_full(p, 0.0, 3.0, NEAREST_NEIGHBOR), 48.0);
    EXPECT_DOUBLE_EQ(sample_full(p, 3.0, 3.0, NEAREST_NEIGHBOR), 51.0);
    EXPECT_DOUBLE_EQ(sample_full(p, 2.0, 1.0, NEAREST_NEIGHBOR), 18.0);  // v(2,1) = 16 + 2
}

// The tie rule is floor(x + 0.5): an exact .5 goes to the higher texel.
TEST(SampleNearestTest, RoundsHalfUp) {
    const Plane<Rpp8u> p = coord_plane();

    EXPECT_DOUBLE_EQ(sample_full(p, 0.49, 0.0, NEAREST_NEIGHBOR), 0.0);  // floor(0.99) = 0
    EXPECT_DOUBLE_EQ(sample_full(p, 0.50, 0.0, NEAREST_NEIGHBOR), 1.0);  // floor(1.00) = 1
    EXPECT_DOUBLE_EQ(sample_full(p, 1.50, 0.0, NEAREST_NEIGHBOR), 2.0);  // floor(2.00) = 2
    EXPECT_DOUBLE_EQ(sample_full(p, 2.50, 0.0, NEAREST_NEIGHBOR), 3.0);  // floor(3.00) = 3

    // Same rule on the row axis: y = 0.5 selects row 1, so v(0,1) = 16.
    EXPECT_DOUBLE_EQ(sample_full(p, 0.0, 0.50, NEAREST_NEIGHBOR), 16.0);

    // Not symmetric about zero: x = -0.5 rounds up to texel 0 and is still in range.
    EXPECT_DOUBLE_EQ(sample_full(p, -0.50, 0.0, NEAREST_NEIGHBOR), 0.0);      // floor(0.0) = 0
    EXPECT_DOUBLE_EQ(sample_full(p, -0.51, 0.0, NEAREST_NEIGHBOR), kBorder);  // floor(-0.01) = -1
}

TEST(SampleNearestTest, OutOfRangeReturnsBorder) {
    const Plane<Rpp8u> p = coord_plane();

    // x = 3.5 rounds up to 4, which is x1 and exclusive.
    EXPECT_DOUBLE_EQ(sample_full(p, 2.99, 0.0, NEAREST_NEIGHBOR), 3.0);  // floor(3.49) = 3
    EXPECT_DOUBLE_EQ(sample_full(p, 3.50, 0.0, NEAREST_NEIGHBOR), kBorder);
    EXPECT_DOUBLE_EQ(sample_full(p, 0.0, 3.50, NEAREST_NEIGHBOR), kBorder);
    EXPECT_DOUBLE_EQ(sample_full(p, -1.0, 0.0, NEAREST_NEIGHBOR), kBorder);  // floor(-0.5) = -1
    EXPECT_DOUBLE_EQ(sample_full(p, 0.0, -1.0, NEAREST_NEIGHBOR), kBorder);
}

// ---- BILINEAR --------------------------------------------------------------

TEST(SampleBilinearTest, ExactTexelCentres) {
    const Plane<Rpp8u> p = coord_plane();
    EXPECT_DOUBLE_EQ(sample_full(p, 0.0, 0.0, BILINEAR), 0.0);
    EXPECT_DOUBLE_EQ(sample_full(p, 1.0, 2.0, BILINEAR), 33.0);  // v(1,2) = 32 + 1
    EXPECT_DOUBLE_EQ(sample_full(p, 2.0, 1.0, BILINEAR), 18.0);
}

// Interior weights on the affine plane.
TEST(SampleBilinearTest, InteriorWeights) {
    const Plane<Rpp8u> p = coord_plane();

    // 0*0.5 + 1*0.5
    EXPECT_DOUBLE_EQ(sample_full(p, 0.5, 0.0, BILINEAR), 0.5);

    // (17 + 18 + 33 + 34) / 4
    EXPECT_DOUBLE_EQ(sample_full(p, 1.5, 1.5, BILINEAR), 25.5);

    // 32*0.75*0.25 + 33*0.25*0.25 + 48*0.75*0.75 + 49*0.25*0.75
    EXPECT_DOUBLE_EQ(sample_full(p, 0.25, 2.75, BILINEAR), 44.25);

    // 2*0.25*0.75 + 3*0.75*0.75 + 18*0.25*0.25 + 19*0.75*0.25
    EXPECT_DOUBLE_EQ(sample_full(p, 2.75, 0.25, BILINEAR), 6.75);
}

// The dx*dy weight, checked on v(x,y) = 16*x*y.
TEST(SampleBilinearTest, CrossTermWeight) {
    const Plane<Rpp8u> p = product_plane();

    // 16 * 0.5 * 0.5; only v11 is non-zero
    EXPECT_DOUBLE_EQ(sample_full(p, 0.5, 0.5, BILINEAR), 4.0);

    // 16 * 0.25 * 0.75; asymmetric weights catch a dx/dy swap
    EXPECT_DOUBLE_EQ(sample_full(p, 0.25, 0.75, BILINEAR), 3.0);

    // (16 + 32 + 32 + 64) / 4
    EXPECT_DOUBLE_EQ(sample_full(p, 1.5, 1.5, BILINEAR), 36.0);

    // (64 + 96 + 96 + 144) / 4
    EXPECT_DOUBLE_EQ(sample_full(p, 2.5, 2.5, BILINEAR), 100.0);
}

// On the last row/column the +1 neighbour is out of range but has zero weight.
TEST(SampleBilinearTest, ZeroWeightNeighbourOutOfRangeIsHarmless) {
    const Plane<Rpp8u> p = coord_plane();
    EXPECT_DOUBLE_EQ(sample_full(p, 3.0, 0.0, BILINEAR), 3.0);   // (4,0) and (4,1) OOB, dx = 0
    EXPECT_DOUBLE_EQ(sample_full(p, 0.0, 3.0, BILINEAR), 48.0);  // (0,4) and (1,4) OOB, dy = 0
    EXPECT_DOUBLE_EQ(sample_full(p, 3.0, 3.0, BILINEAR), 51.0);  // three of four taps OOB
}

// ---- BorderMode::Constant (the default) ------------------------------------
//
// A bilinear tap outside the rectangle contributes the constant, so an edge sample blends with it.
TEST(SampleBorderTest, ConstantBlendsWithTheBorderValue) {
    const Plane<Rpp8u> p = coord_plane();

    // Right edge: 3*0.5 + 1000*0.5
    EXPECT_DOUBLE_EQ(sample_full(p, 3.5, 0.0, BILINEAR), 501.5);

    // Left edge: 1000*0.5 + 0*0.5
    EXPECT_DOUBLE_EQ(sample_full(p, -0.5, 0.0, BILINEAR), 500.0);

    // Bottom edge: 48*0.5 + 1000*0.5
    EXPECT_DOUBLE_EQ(sample_full(p, 0.0, 3.5, BILINEAR), 524.0);

    // Corner, three taps out of range: 51*0.25 + 1000*0.75
    EXPECT_DOUBLE_EQ(sample_full(p, 3.5, 3.5, BILINEAR), 762.75);
}

// A bare double means Constant.
TEST(SampleBorderTest, BareDoubleMeansConstant) {
    const Plane<Rpp8u> p = coord_plane();
    EXPECT_DOUBLE_EQ(sample_full(p, 5.0, 5.0, NEAREST_NEIGHBOR, 42.0), 42.0);
    EXPECT_EQ(Border{7.5}.mode, BorderMode::Constant);
    EXPECT_DOUBLE_EQ(Border{7.5}.value, 7.5);
}

// ---- border_index ------------------------------------------------------------
//
// Each mode is checked at least two periods out, since a rule written for a single step would
// pass a +/-1 test.

TEST(BorderIndexTest, Replicate) {
    // aaa|abcd|ddd
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

// src_texel() relies on this to remap both axes once either is out of range.
TEST(BorderIndexTest, IdentityInsideTheRange) {
    for (int r = 0; r < 4; ++r) {
        EXPECT_EQ(border_index(r, 4, BorderMode::Replicate), r);
        EXPECT_EQ(border_index(r, 4, BorderMode::Wrap), r);
        EXPECT_EQ(border_index(r, 4, BorderMode::Reflect), r);
        EXPECT_EQ(border_index(r, 4, BorderMode::Reflect101), r);
    }
}

// Reflect101's period (2n-2) is zero at n == 1. A 1-pixel ROI is reachable.
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
// Row 0 of kCoordPlane is {0, 1, 2, 3}, so a NEAREST sample there returns its column index.

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

    EXPECT_DOUBLE_EQ(row(-1.0, BorderMode::Replicate), 0.0);    // row 0
    EXPECT_DOUBLE_EQ(row(4.0, BorderMode::Replicate), 48.0);    // row 3
    EXPECT_DOUBLE_EQ(row(-1.0, BorderMode::Wrap), 48.0);        // row 3
    EXPECT_DOUBLE_EQ(row(4.0, BorderMode::Wrap), 0.0);          // row 0
    EXPECT_DOUBLE_EQ(row(-1.0, BorderMode::Reflect), 0.0);      // row 0
    EXPECT_DOUBLE_EQ(row(5.0, BorderMode::Reflect), 32.0);      // row 2
    EXPECT_DOUBLE_EQ(row(-1.0, BorderMode::Reflect101), 16.0);  // row 1
    EXPECT_DOUBLE_EQ(row(4.0, BorderMode::Reflect101), 32.0);   // row 2
}

// (3.5, 0.0): v(3,0) = 3 and texel (4,0), each at weight 0.5. Reflect and Replicate agree one
// step outside.
TEST(SampleBorderModeTest, BilinearOneStepOutside) {
    const Plane<Rpp8u> p = coord_plane();
    auto at = [&](BorderMode m) { return sample_full(p, 3.5, 0.0, BILINEAR, Border{m}); };

    EXPECT_DOUBLE_EQ(at(BorderMode::Replicate), 3.0);   // (3 + 3) / 2, tap folds to column 3
    EXPECT_DOUBLE_EQ(at(BorderMode::Wrap), 1.5);        // (3 + 0) / 2, tap wraps to column 0
    EXPECT_DOUBLE_EQ(at(BorderMode::Reflect), 3.0);     // (3 + 3) / 2, edge texel repeated
    EXPECT_DOUBLE_EQ(at(BorderMode::Reflect101), 2.5);  // (3 + 2) / 2, edge texel not repeated
    EXPECT_DOUBLE_EQ(at(BorderMode::Constant), 1.5);    // (3 + 0) / 2, dtype_black default of 0
}

// (4.5, 0.0): both taps outside, where all four remapping modes separate.
TEST(SampleBorderModeTest, BilinearTwoStepsOutsideSeparatesEveryMode) {
    const Plane<Rpp8u> p = coord_plane();
    auto at = [&](BorderMode m) { return sample_full(p, 4.5, 0.0, BILINEAR, Border{m}); };

    EXPECT_DOUBLE_EQ(at(BorderMode::Replicate), 3.0);   // cols 3, 3 -> (3 + 3) / 2
    EXPECT_DOUBLE_EQ(at(BorderMode::Wrap), 0.5);        // cols 0, 1 -> (0 + 1) / 2
    EXPECT_DOUBLE_EQ(at(BorderMode::Reflect), 2.5);     // cols 3, 2 -> (3 + 2) / 2
    EXPECT_DOUBLE_EQ(at(BorderMode::Reflect101), 1.5);  // cols 2, 1 -> (2 + 1) / 2
}

// The mode only governs what happens outside the rectangle.
TEST(SampleBorderModeTest, InRangeSamplesAreUnaffectedByMode) {
    const Plane<Rpp8u> p = coord_plane();
    for (BorderMode m : {BorderMode::Constant, BorderMode::Replicate, BorderMode::Wrap,
                         BorderMode::Reflect, BorderMode::Reflect101}) {
        EXPECT_DOUBLE_EQ(sample_full(p, 1.5, 1.5, BILINEAR, Border{m}), 25.5);
        EXPECT_DOUBLE_EQ(sample_full(p, 0.25, 2.75, BILINEAR, Border{m}), 44.25);
        EXPECT_DOUBLE_EQ(sample_full(p, 2.0, 1.0, NEAREST_NEIGHBOR, Border{m}), 18.0);
        // Last column with dx = 0: the out-of-range neighbour has zero weight.
        EXPECT_DOUBLE_EQ(sample_full(p, 3.0, 0.0, BILINEAR, Border{m}), 3.0);
    }
}

// Remapping uses the ROI's extent, not the image's: with ROI [1,3) the result is within {17, 18},
// never the image texels 16 or 19.
TEST(SampleBorderModeTest, RemappingIsRelativeToTheRoiNotTheImage) {
    const Plane<Rpp8u> p = coord_plane();
    auto at = [&](double x, BorderMode m) {
        return sample(p.data(), p.desc, p.base(), x, 1.0, 1, 1, 3, 3, NEAREST_NEIGHBOR, Border{m});
    };

    // One step left of the ROI (column 0).
    EXPECT_DOUBLE_EQ(at(0.0, BorderMode::Replicate), 17.0);
    EXPECT_DOUBLE_EQ(at(0.0, BorderMode::Wrap), 18.0);
    EXPECT_DOUBLE_EQ(at(0.0, BorderMode::Reflect), 17.0);
    EXPECT_DOUBLE_EQ(at(0.0, BorderMode::Reflect101), 18.0);

    // One step right of the ROI (column 3).
    EXPECT_DOUBLE_EQ(at(3.0, BorderMode::Replicate), 18.0);
    EXPECT_DOUBLE_EQ(at(3.0, BorderMode::Wrap), 17.0);
    EXPECT_DOUBLE_EQ(at(3.0, BorderMode::Reflect), 18.0);
    EXPECT_DOUBLE_EQ(at(3.0, BorderMode::Reflect101), 17.0);
}

// A 1x1 ROI: every remapping mode must return its one texel from anywhere.
TEST(SampleBorderModeTest, SingleTexelRoi) {
    const Plane<Rpp8u> p = coord_plane();
    for (BorderMode m :
         {BorderMode::Replicate, BorderMode::Wrap, BorderMode::Reflect, BorderMode::Reflect101}) {
        for (double x : {-2.0, 0.0, 1.0, 3.0}) {
            EXPECT_DOUBLE_EQ(
                sample(p.data(), p.desc, p.base(), x, 1.0, 1, 1, 2, 2, NEAREST_NEIGHBOR, Border{m}),
                17.0)
                << "x = " << x;
        }
    }
}

// ---- The rectangle is the ROI, not the image -------------------------------

// Texels that exist in the buffer but sit outside the ROI are border.
TEST(SampleRoiTest, TexelsOutsideTheRoiAreBorderEvenThoughTheyExist) {
    const Plane<Rpp8u> p = coord_plane();
    // ROI [1,3) x [1,3) -- the interior 2x2 block.
    auto at = [&](double x, double y, RpptInterpolationType interp) {
        return sample(p.data(), p.desc, p.base(), x, y, 1, 1, 3, 3, interp, kBorder);
    };

    EXPECT_DOUBLE_EQ(at(1.0, 1.0, NEAREST_NEIGHBOR), 17.0);
    EXPECT_DOUBLE_EQ(at(2.0, 2.0, NEAREST_NEIGHBOR), 34.0);

    EXPECT_DOUBLE_EQ(at(0.0, 1.0, NEAREST_NEIGHBOR), kBorder);  // x < x0
    EXPECT_DOUBLE_EQ(at(3.0, 1.0, NEAREST_NEIGHBOR), kBorder);  // x1 is exclusive

    EXPECT_DOUBLE_EQ(at(1.5, 1.5, BILINEAR), 25.5);  // fully inside, same as InteriorWeights

    // Straddles the ROI's right edge: 18*0.5 + 1000*0.5
    EXPECT_DOUBLE_EQ(at(2.5, 1.0, BILINEAR), 509.0);
}

// ---- Layout and dtype independence -----------------------------------------

// Packed and planar layouts must give identical results. Channel c holds v(x,y) + 100*c, which
// also checks the per-channel plane origin.
TEST(SampleTest, LayoutAgnostic) {
    const auto fill = [](int x, int y, int c) { return kCoordPlane[y][x] + 100 * c; };
    const Plane<Rpp8u> pkd = make_plane<Rpp8u>(DType::U8, Layout::PKD3, fill);
    const Plane<Rpp8u> pln = make_plane<Rpp8u>(DType::U8, Layout::PLN3, fill);

    ASSERT_EQ(pkd.desc.strides.wStride, 3u);  // interleaved
    ASSERT_EQ(pln.desc.strides.wStride, 1u);  // planar

    // 44.25 + 100*c
    const double expected[3] = {44.25, 144.25, 244.25};
    for (Rpp32u c = 0; c < 3; ++c) {
        EXPECT_DOUBLE_EQ(
            sample(pkd.data(), pkd.desc, pkd.base(c), 0.25, 2.75, 0, 0, 4, 4, BILINEAR, kBorder),
            expected[c])
            << "PKD3 channel " << c;
        EXPECT_DOUBLE_EQ(
            sample(pln.data(), pln.desc, pln.base(c), 0.25, 2.75, 0, 0, 4, 4, BILINEAR, kBorder),
            expected[c])
            << "PLN3 channel " << c;
    }
}

// Interpolation is affine, so it commutes with the I8 -128 offset.
TEST(SampleTest, I8OffsetCommutesWithInterpolation) {
    const Plane<Rpp8s> p = make_plane<Rpp8s>(
        DType::I8, Layout::PLN1, [](int x, int y, int) { return kCoordPlane[y][x] - 128; });

    EXPECT_DOUBLE_EQ(sample_full(p, 1.5, 1.5, BILINEAR), 25.5 - 128.0);     // -102.5
    EXPECT_DOUBLE_EQ(sample_full(p, 0.25, 2.75, BILINEAR), 44.25 - 128.0);  // -83.75
    EXPECT_DOUBLE_EQ(sample_full(p, 2.0, 1.0, NEAREST_NEIGHBOR), 18.0 - 128.0);
}

// Values are k/64, exact in float, so no tolerance is needed.
TEST(SampleTest, F32StoredUnits) {
    const Plane<Rpp32f> p = make_plane<Rpp32f>(
        DType::F32, Layout::PLN1, [](int x, int y, int) { return kCoordPlane[y][x] / 64.0; });

    EXPECT_DOUBLE_EQ(sample_full(p, 1.5, 1.5, BILINEAR), 25.5 / 64.0);     // 0.3984375
    EXPECT_DOUBLE_EQ(sample_full(p, 0.25, 2.75, BILINEAR), 44.25 / 64.0);  // 0.69140625
}

// The geometric drivers fill with dtype_black(dt): 0 for U8/F16/F32, -128 for I8 (0 would be
// mid-grey).
TEST(SampleTest, DtypeBlackIsTheGeometricBorderFill) {
    const Plane<Rpp8u> u8 = coord_plane();
    const Plane<Rpp8s> i8 = make_plane<Rpp8s>(
        DType::I8, Layout::PLN1, [](int x, int y, int) { return kCoordPlane[y][x] - 128; });

    EXPECT_DOUBLE_EQ(sample_full(u8, 5.0, 5.0, NEAREST_NEIGHBOR, dtype_black(DType::U8)), 0.0);
    EXPECT_DOUBLE_EQ(sample_full(i8, 5.0, 5.0, NEAREST_NEIGHBOR, dtype_black(DType::I8)), -128.0);
}
