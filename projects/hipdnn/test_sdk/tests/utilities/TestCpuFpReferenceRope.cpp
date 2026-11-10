// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceRope.hpp>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;

namespace
{

constexpr float K_HALF_PI = 1.57079632679489661923f;
constexpr float K_PI = 3.14159265358979323846f;

template <typename Type>
Tensor<Type> makeRopeTensor(const std::vector<int64_t>& dims)
{
    return Tensor<Type>(dims, generateStrides(dims));
}

// Writes `values` in logical row-major order through the tensor's own strides.
template <typename Type>
void setLogicalValues(Tensor<Type>& tensor, const std::vector<float>& values)
{
    ASSERT_EQ(static_cast<size_t>(tensor.elementCount()), values.size());
    const auto& dims = tensor.dims();
    std::vector<int64_t> index(dims.size(), 0);
    for(const auto value : values)
    {
        tensor.setHostValue(static_cast<Type>(value), index);
        for(size_t d = dims.size(); d-- > 0;)
        {
            if(++index[d] < dims[d])
            {
                break;
            }
            index[d] = 0;
        }
    }
}

// Reads the tensor in logical row-major order through its own strides.
template <typename Type>
std::vector<float> logicalValues(const Tensor<Type>& tensor)
{
    std::vector<float> result;
    const auto& dims = tensor.dims();
    const auto count = tensor.elementCount();
    std::vector<int64_t> index(dims.size(), 0);
    for(size_t i = 0; i < count; ++i)
    {
        result.push_back(static_cast<float>(tensor.getHostValue(index)));
        for(size_t d = dims.size(); d-- > 0;)
        {
            if(++index[d] < dims[d])
            {
                break;
            }
            index[d] = 0;
        }
    }
    return result;
}

void expectNear(const std::vector<float>& actual,
                const std::vector<float>& expected,
                float tolerance = 1e-5f)
{
    ASSERT_EQ(actual.size(), expected.size());
    for(size_t i = 0; i < actual.size(); ++i)
    {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "at index " << i;
    }
}

} // namespace

template <typename T>
class CpuFpReferenceRopeTyped : public ::testing::Test
{
};

using RopeDataTypes = ::testing::Types<float, half, bfloat16>;
TYPED_TEST_SUITE(CpuFpReferenceRopeTyped, RopeDataTypes, );

// Qwen-style partial rotary: D = 8, r = 4, leading segment (offset 0). With every angle pi/2,
// y[i] = -x[i + 2] and y[i + 2] = x[i] for i < 2; dims 4..7 pass through.
TYPED_TEST(CpuFpReferenceRopeTyped, LeadingPartialRotation)
{
    auto x = makeRopeTensor<TypeParam>({1, 8});
    setLogicalValues(x, {1, 2, 3, 4, 5, 6, 7, 8});
    auto freqs = makeRopeTensor<float>({1, 4});
    setLogicalValues(freqs, {K_HALF_PI, K_HALF_PI, K_HALF_PI, K_HALF_PI});
    auto y = makeRopeTensor<TypeParam>({1, 8});

    CpuFpReferenceRope::forward(x, freqs, y, 1.0f, 4, std::optional<int64_t>{0});

    expectNear(logicalValues(y), {-3, -4, 1, 2, 5, 6, 7, 8}, 1e-2f);
}

// Full rotation (rope_dim 0 means D = 4): angles {0, pi/2} for the pairs (0, 2) and (1, 3).
TEST(TestCpuFpReferenceRope, FullRotation)
{
    auto x = makeRopeTensor<float>({1, 4});
    setLogicalValues(x, {1, 2, 3, 4});
    auto freqs = makeRopeTensor<float>({1, 4});
    setLogicalValues(freqs, {0, K_HALF_PI, 0, K_HALF_PI});
    auto y = makeRopeTensor<float>({1, 4});

    CpuFpReferenceRope::forward(x, freqs, y, 1.0f, 0, std::nullopt);

    // i = 0, f = 0: y0 = 1, y2 = 3. i = 1, f = pi/2: y1 = -x3 = -4, y3 = x1 = 2.
    expectNear(logicalValues(y), {1, -4, 3, 2});
}

// No rope_offset: the cuDNN trailing segment, o = D - r = 2.
TEST(TestCpuFpReferenceRope, TrailingSegmentByDefault)
{
    auto x = makeRopeTensor<float>({1, 6});
    setLogicalValues(x, {1, 2, 3, 4, 5, 6});
    auto freqs = makeRopeTensor<float>({1, 4});
    setLogicalValues(freqs, {K_HALF_PI, 0, K_HALF_PI, 0});
    auto y = makeRopeTensor<float>({1, 6});

    CpuFpReferenceRope::forward(x, freqs, y, 1.0f, 4, std::nullopt);

    // Segment [3, 4, 5, 6]: i = 0 (pi/2): y2 = -5, y4 = 3; i = 1 (0): y3 = 4, y5 = 6.
    expectNear(logicalValues(y), {1, 2, -5, 4, 3, 6});
}

// output_scale multiplies the rotated segment and the passthrough dims alike.
TEST(TestCpuFpReferenceRope, OutputScaleAppliesToPassthrough)
{
    auto x = makeRopeTensor<float>({1, 4});
    setLogicalValues(x, {1, 2, 3, 4});
    auto freqs = makeRopeTensor<float>({1, 2});
    setLogicalValues(freqs, {K_PI, K_PI});
    auto y = makeRopeTensor<float>({1, 4});

    CpuFpReferenceRope::forward(x, freqs, y, 2.0f, 2, std::optional<int64_t>{0});

    // h = 1, f = pi: y0 = 2 * -1 = -2, y1 = 2 * -2 = -4; passthrough 2 * {3, 4}.
    expectNear(logicalValues(y), {-2, -4, 6, 8});
}

// freqs [1, 1, S, r] broadcasts over batch and heads: per-position angles {0, pi/2, pi}.
TEST(TestCpuFpReferenceRope, FreqsBroadcastOverBatchAndHeads)
{
    auto x = makeRopeTensor<float>({2, 1, 3, 2});
    setLogicalValues(x, {1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4});
    auto freqs = makeRopeTensor<float>({1, 1, 3, 2});
    setLogicalValues(freqs, {0, 0, K_HALF_PI, K_HALF_PI, K_PI, K_PI});
    auto y = makeRopeTensor<float>({2, 1, 3, 2});

    CpuFpReferenceRope::forward(x, freqs, y, 1.0f, 0, std::nullopt);

    // s0: [x0, x1]; s1: [-x1, x0]; s2: [-x0, -x1].
    expectNear(logicalValues(y), {1, 2, -2, 1, -1, -2, 3, 4, -4, 3, -3, -4});
}

// freqs [B, 1, S, r]: each batch has its own angles (batch 0: 0, batch 1: pi/2).
TEST(TestCpuFpReferenceRope, PerBatchFreqs)
{
    auto x = makeRopeTensor<float>({2, 1, 3, 2});
    setLogicalValues(x, {1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4});
    auto freqs = makeRopeTensor<float>({2, 1, 3, 2});
    setLogicalValues(
        freqs,
        {0, 0, 0, 0, 0, 0, K_HALF_PI, K_HALF_PI, K_HALF_PI, K_HALF_PI, K_HALF_PI, K_HALF_PI});
    auto y = makeRopeTensor<float>({2, 1, 3, 2});

    CpuFpReferenceRope::forward(x, freqs, y, 1.0f, 0, std::nullopt);

    expectNear(logicalValues(y), {1, 2, 1, 2, 1, 2, -4, 3, -4, 3, -4, 3});
}

// The two halves of each pair may carry different angles (f[i] vs f[i + h]).
TEST(TestCpuFpReferenceRope, DistinctAnglesPerHalf)
{
    auto x = makeRopeTensor<float>({1, 2});
    setLogicalValues(x, {1, 2});
    auto freqs = makeRopeTensor<float>({1, 2});
    setLogicalValues(freqs, {0, K_HALF_PI});
    auto y = makeRopeTensor<float>({1, 2});

    CpuFpReferenceRope::forward(x, freqs, y, 1.0f, 0, std::nullopt);

    // y0 = x0 cos 0 - x1 sin 0 = 1; y1 = x1 cos(pi/2) + x0 sin(pi/2) = 1.
    expectNear(logicalValues(y), {1, 1});
}

// y may alias x: every element is read before its output is written.
TEST(TestCpuFpReferenceRope, InPlace)
{
    auto x = makeRopeTensor<float>({1, 4});
    setLogicalValues(x, {1, 2, 3, 4});
    auto freqs = makeRopeTensor<float>({1, 4});
    setLogicalValues(freqs, {0, K_HALF_PI, 0, K_HALF_PI});

    CpuFpReferenceRope::forward(x, freqs, x, 1.0f, 0, std::nullopt);

    expectNear(logicalValues(x), {1, -4, 3, 2});
}

// Non-packed x and y: the last dim is not unit-stride.
TEST(TestCpuFpReferenceRope, StridedInputAndOutput)
{
    Tensor<float> x(std::vector<int64_t>{2, 4}, std::vector<int64_t>{1, 2});
    setLogicalValues(x, {1, 2, 3, 4, 5, 6, 7, 8});
    auto freqs = makeRopeTensor<float>({1, 4});
    setLogicalValues(freqs, {K_HALF_PI, K_HALF_PI, K_HALF_PI, K_HALF_PI});
    Tensor<float> y(std::vector<int64_t>{2, 4}, std::vector<int64_t>{1, 2});

    CpuFpReferenceRope::forward(x, freqs, y, 1.0f, 0, std::nullopt);

    // Rows [a, b, c, d] -> [-c, -d, a, b].
    expectNear(logicalValues(y), {-3, -4, 1, 2, -7, -8, 5, 6});
}

TEST(TestCpuFpReferenceRope, OddRopeDimThrows)
{
    auto x = makeRopeTensor<float>({1, 4});
    auto freqs = makeRopeTensor<float>({1, 3});
    auto y = makeRopeTensor<float>({1, 4});

    EXPECT_THROW(CpuFpReferenceRope::forward(x, freqs, y, 1.0f, 3, std::nullopt),
                 std::runtime_error);
}

TEST(TestCpuFpReferenceRope, RopeDimAboveHeadDimThrows)
{
    auto x = makeRopeTensor<float>({1, 4});
    auto freqs = makeRopeTensor<float>({1, 6});
    auto y = makeRopeTensor<float>({1, 4});

    EXPECT_THROW(CpuFpReferenceRope::forward(x, freqs, y, 1.0f, 6, std::nullopt),
                 std::runtime_error);
}

TEST(TestCpuFpReferenceRope, OffsetPastHeadDimThrows)
{
    auto x = makeRopeTensor<float>({1, 4});
    auto freqs = makeRopeTensor<float>({1, 2});
    auto y = makeRopeTensor<float>({1, 4});

    EXPECT_THROW(CpuFpReferenceRope::forward(x, freqs, y, 1.0f, 2, std::optional<int64_t>{3}),
                 std::runtime_error);
}

TEST(TestCpuFpReferenceRope, FreqsLastDimMismatchThrows)
{
    auto x = makeRopeTensor<float>({1, 4});
    auto freqs = makeRopeTensor<float>({1, 2});
    auto y = makeRopeTensor<float>({1, 4});

    EXPECT_THROW(CpuFpReferenceRope::forward(x, freqs, y, 1.0f, 0, std::nullopt),
                 std::runtime_error);
}

TEST(TestCpuFpReferenceRope, NonBroadcastableFreqsThrows)
{
    auto x = makeRopeTensor<float>({3, 4});
    auto freqs = makeRopeTensor<float>({2, 4});
    auto y = makeRopeTensor<float>({3, 4});

    EXPECT_THROW(CpuFpReferenceRope::forward(x, freqs, y, 1.0f, 0, std::nullopt),
                 std::runtime_error);
}

TEST(TestCpuFpReferenceRope, RankOneThrows)
{
    auto x = makeRopeTensor<float>({4});
    auto freqs = makeRopeTensor<float>({4});
    auto y = makeRopeTensor<float>({4});

    EXPECT_THROW(CpuFpReferenceRope::forward(x, freqs, y, 1.0f, 0, std::nullopt),
                 std::runtime_error);
}
