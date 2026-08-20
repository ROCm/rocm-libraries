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

#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

using namespace rpptest;

TEST(RoiTest, XywhAliasesLtrb) {
    RpptROI roi;
    roi.xywhROI.xy.x = 5;
    roi.xywhROI.xy.y = 7;
    EXPECT_EQ(roi.ltrbROI.lt.x, 5);
    EXPECT_EQ(roi.ltrbROI.lt.y, 7);
}

// Every image test builds its tensors with make_descriptor(), so its stride convention is an
// assumption shared by the whole image domain: the row width is padded to (w/8)*8+8 elements for
// the kernels that store a full SIMD vector on a row's tail, while d.w stays the logical width.
// The width chosen here is not a multiple of 8, so the padding is visible in the strides.
TEST(DescriptorTest, PaddedStridesPacked) {
    const RpptDesc d = make_descriptor({2, 3, 4, 12}, DType::U8, Layout::PKD3);
    const Rpp32u pw = padded_width(12);  // 16
    EXPECT_EQ(pw, 16u);
    EXPECT_EQ(d.w, 12u);  // logical width is unpadded
    EXPECT_EQ(d.layout, RpptLayout::NHWC);
    EXPECT_EQ(d.strides.cStride, 1u);
    EXPECT_EQ(d.strides.wStride, 3u);
    EXPECT_EQ(d.strides.hStride, 3u * pw);
    EXPECT_EQ(d.strides.nStride, 3u * 4u * pw);
}

TEST(DescriptorTest, PaddedStridesPlanar) {
    const RpptDesc d = make_descriptor({2, 3, 4, 12}, DType::U8, Layout::PLN3);
    const Rpp32u pw = padded_width(12);
    EXPECT_EQ(d.layout, RpptLayout::NCHW);
    EXPECT_EQ(d.strides.wStride, 1u);
    EXPECT_EQ(d.strides.hStride, pw);
    EXPECT_EQ(d.strides.cStride, 4u * pw);
    EXPECT_EQ(d.strides.nStride, 3u * 4u * pw);
}

// pad=false is what golden_layout_test uses to prove the goldens address through the descriptor
// rather than walking the buffer, so the dense form has to actually be dense.
TEST(DescriptorTest, DenseStridesDropThePadding) {
    const RpptDesc d = make_descriptor({2, 3, 4, 12}, DType::U8, Layout::PLN3, /*pad=*/false);
    EXPECT_EQ(d.strides.hStride, 12u);
    EXPECT_EQ(d.strides.cStride, 4u * 12u);
    EXPECT_EQ(d.strides.nStride, 3u * 4u * 12u);
    EXPECT_EQ(element_count(d), static_cast<std::size_t>(2 * 3 * 4 * 12));
}

// A width that is already a multiple of 8 still gains a full vector of slack -- the kernels store
// unconditionally, so an exactly-full last row would be written past its end.
TEST(DescriptorTest, PaddedWidthAlwaysLeavesSlack) {
    EXPECT_EQ(padded_width(8), 16u);
    EXPECT_EQ(padded_width(16), 24u);
    EXPECT_EQ(padded_width(1), 8u);
}
