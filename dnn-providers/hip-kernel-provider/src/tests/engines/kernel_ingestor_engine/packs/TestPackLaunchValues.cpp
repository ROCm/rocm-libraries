// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <cstring>
#include <string>
#include <variant>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/packs/ConvFwdLaunchValues.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionDenseLaunchValues.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseLaunchValues.hpp"

/**
 * @file TestPackLaunchValues.cpp
 * @brief The saved launch values of each pack: every member of the launch state maps to
 *        the value of its own name and type, and to no other value.
 *
 * Each member gets a distinct value, so a swapped or dropped mapping shows. The expected
 * names are written out again here: they are the stored contract, and a test that shared
 * the constants would not see a rename.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using hipdnn_plugin_sdk::ingestor::MetadataValues;

TEST(TestPackLaunchValues, PointwiseValuesMatchTheV1Contract)
{
    const PointwiseBinding binding{11, 12, 13};

    const MetadataValues values = pointwiseLaunchValues(binding, 256);

    const MetadataValues expected{{"input_a.uid", int64_t{11}},
                                  {"input_b.uid", int64_t{12}},
                                  {"output.uid", int64_t{13}},
                                  {"block_size", int64_t{256}}};
    EXPECT_EQ(values, expected);
    EXPECT_EQ(POINTWISE_DISPATCH_SYMBOL_V1, "hipkernel.pointwise.dispatch.v1");
}

TEST(TestPackLaunchValues, ConvFwdValuesMatchTheV1Contract)
{
    const ConvFwdBinding binding{21, 22, 23};
    const ConvFwdExtents extents{31, 32, 33, 34, 35, 36, 37};

    const MetadataValues values = convFwdLaunchValues(binding, extents, 64);

    const MetadataValues expected{{"x.uid", int64_t{21}},
                                  {"w.uid", int64_t{22}},
                                  {"y.uid", int64_t{23}},
                                  {"n", int64_t{31}},
                                  {"c", int64_t{32}},
                                  {"h", int64_t{33}},
                                  {"width", int64_t{34}},
                                  {"k", int64_t{35}},
                                  {"r", int64_t{36}},
                                  {"s", int64_t{37}},
                                  {"block_size", int64_t{64}}};
    EXPECT_EQ(values, expected);
    EXPECT_EQ(CONV_FWD_DISPATCH_SYMBOL_V1, "hipkernel.conv_fwd.dispatch.v1");
}

TEST(TestPackLaunchValues, Gfx950AttentionDenseValuesMatchTheV1Contract)
{
    AttentionDenseBinding binding;
    binding.q = 41;
    binding.k = 42;
    binding.v = 43;
    binding.o = 44;
    binding.causal = 1;
    binding.slidingWindow = 45;
    binding.scale = 0.1F;

    AttentionDenseProblem problem;
    problem.batch = 51;
    problem.seqLenQ = 52;
    problem.seqLenKv = 53;
    problem.numQueryHeads = 54;
    problem.numKvHeads = 55;
    problem.headSize = 56;
    problem.dataType = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16;

    const MetadataValues values = gfx950AttentionDenseLaunchValues(binding, problem, 128);

    const MetadataValues expected{{"q.uid", int64_t{41}},
                                  {"k.uid", int64_t{42}},
                                  {"v.uid", int64_t{43}},
                                  {"o.uid", int64_t{44}},
                                  {"causal", int64_t{1}},
                                  {"sliding_window", int64_t{45}},
                                  {"batch", int64_t{51}},
                                  {"seqlen_q", int64_t{52}},
                                  {"seqlen_kv", int64_t{53}},
                                  {"num_query_heads", int64_t{54}},
                                  {"num_kv_heads", int64_t{55}},
                                  {"head_size", int64_t{56}},
                                  {"block_m", int64_t{128}},
                                  {"scale", static_cast<double>(0.1F)},
                                  {"data_type", std::string("BFLOAT16")},
                                  {"stride_layout", std::string("bshd")}};
    EXPECT_EQ(values, expected);
    EXPECT_EQ(GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V1,
              "hipkernel.gfx950_attention_dense.dispatch.v1");

    // The stored double converts back to the float the kernel takes, bit for bit.
    const auto stored = values.find("scale");
    ASSERT_NE(stored, values.end());
    const auto* storedScale = std::get_if<double>(&stored->second);
    ASSERT_NE(storedScale, nullptr);
    const auto restored = static_cast<float>(*storedScale);
    uint32_t restoredBits = 0;
    uint32_t originalBits = 0;
    std::memcpy(&restoredBits, &restored, sizeof(restoredBits));
    std::memcpy(&originalBits, &binding.scale, sizeof(originalBits));
    EXPECT_EQ(restoredBits, originalBits);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
