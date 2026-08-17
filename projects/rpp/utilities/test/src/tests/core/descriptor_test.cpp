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

TEST(RoiTest, XywhAliasesLtrb) {
    RpptROI roi;
    roi.xywhROI.xy.x = 5;
    roi.xywhROI.xy.y = 7;
    EXPECT_EQ(roi.ltrbROI.lt.x, 5);
    EXPECT_EQ(roi.ltrbROI.lt.y, 7);
}

TEST(DescriptorTest, PackedStridesLayout) {
    RpptDesc desc{};
    desc.dataType = RpptDataType::U8;
    desc.layout = RpptLayout::NHWC;
    desc.n = 2;
    desc.h = 4;
    desc.w = 8;
    desc.c = 3;
    desc.strides.nStride = desc.c * desc.w * desc.h;
    desc.strides.hStride = desc.c * desc.w;
    desc.strides.wStride = desc.c;
    desc.strides.cStride = 1;

    EXPECT_EQ(desc.strides.nStride, 96u);
    EXPECT_EQ(desc.strides.hStride, 24u);
    EXPECT_EQ(desc.strides.wStride, 3u);
}
