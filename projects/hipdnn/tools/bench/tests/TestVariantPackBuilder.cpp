// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestVariantPackBuilder.cpp
 * @brief Covers buffer sizing, where being wrong yields a number rather than an error.
 */

#include <gtest/gtest.h>

#include <hipdnn_bench/VariantPackBuilder.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace hipdnn_bench
{

TEST(TestVariantPackBuilder, SpansThePackedTensorExactly)
{
    EXPECT_EQ(elementSpan({2, 3, 4}, {12, 4, 1}), 24);
    EXPECT_EQ(elementSpan({1}, {1}), 1);
}

TEST(TestVariantPackBuilder, SpansAPaddedTensorPastItsElementCount)
{
    // Row stride 8: element [3][3] sits at 3*8 + 3 = 27, so 28 must be addressable.
    EXPECT_EQ(elementSpan({4, 4}, {8, 1}), 28);
    EXPECT_GT(elementSpan({4, 4}, {8, 1}), 4 * 4);
}

TEST(TestVariantPackBuilder, TakesTheLastIndexOfEveryDimensionAtOnce)
{
    // (2-1)*12 + (3-1)*4 + (4-1)*1 + 1 = 24; the largest single term would give 13.
    EXPECT_EQ(elementSpan({2, 3, 4}, {12, 4, 1}), 24);
    EXPECT_GT(elementSpan({2, 3, 4}, {12, 4, 1}), (2 - 1) * 12 + 1);
}

TEST(TestVariantPackBuilder, RefusesAMalformedTensorRatherThanSizingIt)
{
    EXPECT_EQ(elementSpan({}, {}), 0);
    EXPECT_EQ(elementSpan({2, 3}, {1}), 0) << "rank mismatch";
    EXPECT_EQ(elementSpan({2, 0}, {1, 1}), 0) << "zero extent";
    EXPECT_EQ(elementSpan({2, -1}, {1, 1}), 0) << "negative extent";
}

TEST(TestVariantPackBuilder, SizesSubByteTypesInBitsNotBytes)
{
    using hipdnn_frontend::DataType;

    EXPECT_EQ(tensorBytes({10}, {1}, DataType::FP4_E2M1), 5);
    EXPECT_EQ(tensorBytes({10}, {1}, DataType::INT4), 5);

    // Nine elements is 36 bits: rounds up to five bytes.
    EXPECT_EQ(tensorBytes({9}, {1}, DataType::FP4_E2M1), 5);

    // 54 bits -> 7 bytes; 18 bits -> 3 bytes.
    EXPECT_EQ(tensorBytes({9}, {1}, DataType::FP6_E2M3), 7);
    EXPECT_EQ(tensorBytes({3}, {1}, DataType::FP6_E3M2), 3);
}

TEST(TestVariantPackBuilder, SizesTheOrdinaryTypes)
{
    using hipdnn_frontend::DataType;
    EXPECT_EQ(tensorBytes({4, 4}, {4, 1}, DataType::FLOAT), 64);
    EXPECT_EQ(tensorBytes({4, 4}, {4, 1}, DataType::HALF), 32);
    EXPECT_EQ(tensorBytes({4, 4}, {4, 1}, DataType::BFLOAT16), 32);
    EXPECT_EQ(tensorBytes({4, 4}, {4, 1}, DataType::DOUBLE), 128);
    EXPECT_EQ(tensorBytes({4, 4}, {4, 1}, DataType::INT8), 16);
}

TEST(TestVariantPackBuilder, RefusesATypeItCannotSize)
{
    using hipdnn_frontend::DataType;
    EXPECT_FALSE(tensorBytes({4}, {1}, DataType::NOT_SET).has_value());
    EXPECT_EQ(elementBits(DataType::NOT_SET), 0);
}

TEST(TestVariantPackBuilder, RefusesASpanThatWouldOverflow)
{
    using hipdnn_frontend::DataType;
    const auto huge = std::numeric_limits<int64_t>::max() / 4;
    EXPECT_FALSE(tensorBytes({huge, 2}, {2, 1}, DataType::DOUBLE).has_value());
}

TEST(TestVariantPackBuilder, RefusesASpanWhoseElementArithmeticWouldWrap)
{
    using hipdnn_frontend::DataType;
    constexpr int64_t TWO_TO_62 = int64_t{1} << 62;
    constexpr int64_t LIMIT = std::numeric_limits<int64_t>::max();

    // Overflow in the product, the accumulation, and the closing +1, respectively.
    EXPECT_EQ(elementSpan({3}, {TWO_TO_62}), 0);
    EXPECT_FALSE(tensorBytes({3}, {TWO_TO_62}, DataType::FLOAT).has_value());
    EXPECT_EQ(elementSpan({2, 2}, {TWO_TO_62, TWO_TO_62}), 0);
    EXPECT_EQ(elementSpan({2, 2}, {TWO_TO_62, TWO_TO_62 - 1}), 0);
    EXPECT_EQ(elementSpan({2}, {-1}), 0) << "negative stride";

    // The largest representable span is still a span.
    EXPECT_EQ(elementSpan({2}, {LIMIT - 1}), LIMIT);
}

TEST(TestVariantPackBuilder, RoundsSubByteSpansNearTheLimitWithoutWrapping)
{
    using hipdnn_frontend::DataType;
    constexpr int64_t LIMIT = std::numeric_limits<int64_t>::max();

    // `span * bits + 7` would overflow, but the true byte count, 2^60, is representable.
    EXPECT_EQ(tensorBytes({LIMIT / 4}, {1}, DataType::FP4_E2M1), int64_t{1} << 60);
    EXPECT_EQ(tensorBytes({LIMIT / 6}, {1}, DataType::FP6_E2M3), int64_t{1} << 60);

    // 2^58 elements of a 256-bit type is 2^63 bytes: does not fit.
    EXPECT_FALSE(tensorBytes({int64_t{1} << 58}, {1}, DataType::INT8x32).has_value());
}

namespace
{

using hipdnn_frontend::DataType;
using hipdnn_frontend::graph::Graph;
using hipdnn_frontend::graph::PointwiseAttributes;
using hipdnn_frontend::graph::TensorAttributes;

std::shared_ptr<TensorAttributes> vectorTensor(int64_t uid, const char* name)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_uid(uid).set_name(name).set_dim({4}).set_stride({1}).set_data_type(DataType::FLOAT);
    return tensor;
}

std::shared_ptr<TensorAttributes> multiply(Graph& graph,
                                           const std::shared_ptr<TensorAttributes>& left,
                                           const std::shared_ptr<TensorAttributes>& right)
{
    PointwiseAttributes attributes;
    attributes.set_mode(hipdnn_frontend::PointwiseMode::MUL);
    return graph.pointwise(left, right, attributes);
}

const TensorRequirement* planned(const VariantPackPlan& plan, int64_t uid)
{
    for(const auto& tensor : plan.tensors)
    {
        if(tensor.uid == uid)
        {
            return &tensor;
        }
    }
    return nullptr;
}

} // namespace

TEST(TestVariantPackBuilder, MarksTheTensorsTheGraphWritesAndOnlyThose)
{
    Graph graph;
    const auto x = vectorTensor(1, "X");
    const auto w = vectorTensor(2, "W");
    const auto product = multiply(graph, x, w);
    product->set_uid(3).set_name("XW").set_is_virtual(true);
    const auto y = multiply(graph, product, w);
    y->set_uid(4).set_name("Y").set_dim({4}).set_stride({1}).set_data_type(DataType::FLOAT);
    y->set_output(true);

    const auto plan = planVariantPack(graph);
    ASSERT_TRUE(plan.error.empty()) << plan.error;
    ASSERT_EQ(plan.tensors.size(), 3U) << "the virtual intermediate is not planned";
    EXPECT_FALSE(planned(plan, 1)->produced);
    EXPECT_FALSE(planned(plan, 2)->produced);
    EXPECT_TRUE(planned(plan, 4)->produced);
}

TEST(TestVariantPackBuilder, PutsARuntimeScalarInHostMemoryAndPlansNoSlotForABakedOne)
{
    // RFC 0016: the provider reads a runtime scalar's slot on the CPU; baked scalars have no
    // slot.
    Graph graph;
    const auto x = vectorTensor(1, "X");

    auto runtimeScale = std::make_shared<TensorAttributes>();
    runtimeScale->set_uid(2).set_name("scale").set_dim({1}).set_stride({1}).set_data_type(
        DataType::FLOAT);
    runtimeScale->set_as_runtime_parameter();

    auto constant = std::make_shared<TensorAttributes>(2.0F);
    constant->set_uid(3).set_name("constant");

    auto withDefault = std::make_shared<TensorAttributes>(
        3.0F, hipdnn_frontend::graph::ScalarType::RUNTIME_PARAM);
    withDefault->set_uid(4).set_name("with_default");

    auto scaled = multiply(graph, x, runtimeScale);
    scaled->set_uid(5).set_is_virtual(true);
    auto doubled = multiply(graph, scaled, constant);
    doubled->set_uid(6).set_is_virtual(true);
    auto y = multiply(graph, doubled, withDefault);
    y->set_uid(7).set_name("Y").set_dim({4}).set_stride({1}).set_data_type(DataType::FLOAT);
    y->set_output(true);

    const auto plan = planVariantPack(graph);
    ASSERT_TRUE(plan.error.empty()) << plan.error;

    ASSERT_NE(planned(plan, 2), nullptr);
    EXPECT_EQ(planned(plan, 2)->storage, TensorStorage::HOST);
    EXPECT_EQ(planned(plan, 2)->bytes, 4);
    EXPECT_FALSE(planned(plan, 2)->produced);

    EXPECT_EQ(planned(plan, 3), nullptr) << "a compile-time constant has no slot";
    EXPECT_EQ(planned(plan, 4), nullptr) << "a runtime-with-default scalar has no slot";

    ASSERT_NE(planned(plan, 1), nullptr);
    EXPECT_EQ(planned(plan, 1)->storage, TensorStorage::DEVICE);
    ASSERT_NE(planned(plan, 7), nullptr);
    EXPECT_EQ(planned(plan, 7)->storage, TensorStorage::DEVICE);
}

} // namespace hipdnn_bench
