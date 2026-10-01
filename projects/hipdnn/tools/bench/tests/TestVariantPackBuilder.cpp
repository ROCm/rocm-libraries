// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestVariantPackBuilder.cpp
 * @brief Covers buffer sizing, where being wrong yields a number rather than an error.
 *
 * Both ways of getting this wrong pass a benchmark run. Under-sizing a padded tensor means
 * the kernel writes past the buffer and the corpus records the time it took to do so;
 * mis-sizing a sub-byte type allocates nothing at all. Neither raises anything a fleet would
 * notice, so the arithmetic is pinned here against tensors the test defines.
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
    // A packed tensor's span is its element count, which is the only case where the naive
    // count-based sizing happens to be right -- and the reason it survives casual testing.
    EXPECT_EQ(elementSpan({2, 3, 4}, {12, 4, 1}), 24);
    EXPECT_EQ(elementSpan({1}, {1}), 1);
}

TEST(TestVariantPackBuilder, SpansAPaddedTensorPastItsElementCount)
{
    // A 4x4 tensor with a row stride of 8: element [3][3] sits at 3*8 + 3 = 27, so 28 must be
    // addressable while the tensor holds only 16. Sizing by element count allocates 16 and the
    // kernel writes twelve elements past the end -- and still returns a time.
    EXPECT_EQ(elementSpan({4, 4}, {8, 1}), 28);
    EXPECT_GT(elementSpan({4, 4}, {8, 1}), 4 * 4);
}

TEST(TestVariantPackBuilder, TakesTheLastIndexOfEveryDimensionAtOnce)
{
    // The furthest element is reached with every index at its maximum, so the span is a sum
    // over dimensions. Taking the largest single term instead would size this at 13 and come
    // up four elements short.
    //
    //   (2-1)*12 + (3-1)*4 + (4-1)*1 + 1 = 12 + 8 + 3 + 1 = 24
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

    // Four-bit types are the reason the arithmetic is in bits. Ten FP4 elements are five
    // bytes; a per-element byte size would be 0 (allocating nothing) or 1 (harmless but
    // wasteful), and only the first of those is a bug that runs.
    EXPECT_EQ(tensorBytes({10}, {1}, DataType::FP4_E2M1), 5);
    EXPECT_EQ(tensorBytes({10}, {1}, DataType::INT4), 5);

    // Odd spans round up: nine elements is 36 bits, which needs five bytes.
    EXPECT_EQ(tensorBytes({9}, {1}, DataType::FP4_E2M1), 5);

    // Six-bit types round on a remainder that is not a whole nibble: nine elements are 54
    // bits, seven bytes; three are 18 bits, three bytes.
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

    // Refusing by name beats guessing a width. A guess that is too small is a buffer overrun
    // that still produces a time.
    EXPECT_FALSE(tensorBytes({4}, {1}, DataType::NOT_SET).has_value());
    EXPECT_EQ(elementBits(DataType::NOT_SET), 0);
}

TEST(TestVariantPackBuilder, RefusesASpanThatWouldOverflow)
{
    using hipdnn_frontend::DataType;

    // The corpus search proposes across orders of magnitude, so this is reachable rather than
    // theoretical. Wrapping would produce a small, plausible allocation.
    const auto huge = std::numeric_limits<int64_t>::max() / 4;
    EXPECT_FALSE(tensorBytes({huge, 2}, {2, 1}, DataType::DOUBLE).has_value());
}

TEST(TestVariantPackBuilder, RefusesASpanWhoseElementArithmeticWouldWrap)
{
    using hipdnn_frontend::DataType;
    constexpr int64_t TWO_TO_62 = int64_t{1} << 62;
    constexpr int64_t LIMIT = std::numeric_limits<int64_t>::max();

    // Each step that forms the span can leave int64_t on its own; checking only the final
    // product against the element width is too late, because by then the span has wrapped
    // to something small. (3 - 1) * 2^62 is the product, 2^62 + 2^62 the accumulation, and
    // (2^62) + (2^62 - 1) + 1 the closing increment.
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

    // Both spans pass a `span * bits <= LIMIT` bound, and `span * bits + 7` then overflows.
    // The true byte counts, ceil(span * bits / 8), are both exactly 2^60 and representable,
    // so the right answer is that number, not a refusal and not a wrapped negative size.
    EXPECT_EQ(tensorBytes({LIMIT / 4}, {1}, DataType::FP4_E2M1), int64_t{1} << 60);
    EXPECT_EQ(tensorBytes({LIMIT / 6}, {1}, DataType::FP6_E2M3), int64_t{1} << 60);

    // A byte count that really does not fit is refused: 2^58 elements of a 256-bit type is
    // 2^63 bytes.
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
    // The flat plan carries direction because both the input fill and the correctness gate
    // need it: an input counted as output is agreement every candidate was handed.
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
    // RFC 0016: a pure runtime scalar is read by the provider on the CPU from its slot
    // (hipdnn_plugin_sdk::resolveScalarOperand), so a device allocation there is a pointer
    // the provider cannot portably dereference. A scalar with a baked value -- constant or
    // runtime-with-default -- reaches the provider through the op graph and has no slot.
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
