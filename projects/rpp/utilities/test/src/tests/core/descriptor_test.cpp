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
