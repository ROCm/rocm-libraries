#include <rocRoller/CodeGen/Buffer.hpp>

#include "SimpleFixture.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;

class BufferDescriptorTest : public SimpleFixture
{
};

TEST_F(BufferDescriptorTest, Default)
{
    GFX9BufferDescriptorOptions options;

    auto expected = R"(GFX9BufferDescriptorOptions: 0x00020000
        dst_sel_x: 0
        dst_sel_y: 0
        dst_sel_z: 0
        dst_sel_w: 0
        num_format: 0
        data_format: DF32
        user_vm_enable: 0
        user_vm_mode: 0
        index_stride: 0
        add_tid_enable: 0
        _unusedA: 0
        nv: 0
        _unusedB: 0
        type: 0
    )";

    EXPECT_EQ(NormalizedSource(expected), NormalizedSource(options.toString()));

    EXPECT_EQ(0x00020000, options.rawValue());
    EXPECT_EQ(0, options.dst_sel_x);
    EXPECT_EQ(0, options.dst_sel_y);
    EXPECT_EQ(0, options.dst_sel_z);
    EXPECT_EQ(0, options.dst_sel_w);
    EXPECT_EQ(0, options.num_format);
    EXPECT_EQ(GFX9BufferDescriptorOptions::DF32, options.data_format);
    EXPECT_EQ(0, options.user_vm_enable);
    EXPECT_EQ(0, options.user_vm_mode);

    auto literal = options.literal();
    ASSERT_NE(nullptr, literal);
    EXPECT_EQ(CommandArgumentValue(0x00020000u), literal->getLiteralValue());
}

TEST_F(BufferDescriptorTest, Values)
{
    GFX9BufferDescriptorOptions options;
    options.dst_sel_x = 1;
    EXPECT_EQ(0x00020001, options.rawValue());

    EXPECT_THAT(options.toString(), ::testing::HasSubstr("dst_sel_x: 1\n"));

    options._unusedA = 3;
    EXPECT_ANY_THROW(options.validate());
    options._unusedA = 0;
    EXPECT_NO_THROW(options.validate());

    options.nv = 1;
    EXPECT_EQ(0x08020001, options.rawValue());
}
