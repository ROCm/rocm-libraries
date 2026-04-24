// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/reduction_attributes_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceReduction.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

#include <cmath>
#include <vector>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using hipdnn_test_sdk::detail::safeTestTypeCast;

namespace
{

template <typename Type>
Tensor<Type> createTensor(const std::vector<int64_t>& dims)
{
    return Tensor<Type>(dims, generateStrides(dims));
}

template <typename Type>
void fillSequential(Tensor<Type>& tensor, float start = 1.0f)
{
    const auto count = static_cast<int>(tensor.elementCount());
    for(int i = 0; i < count; ++i)
    {
        tensor.memory().hostData()[i] = safeTestTypeCast<Type>(start + static_cast<float>(i));
    }
}

template <typename Type>
void fillValues(Tensor<Type>& tensor, const std::vector<float>& values)
{
    ASSERT_EQ(static_cast<size_t>(tensor.elementCount()), values.size());
    for(size_t i = 0; i < values.size(); ++i)
    {
        tensor.memory().hostData()[i] = safeTestTypeCast<Type>(values[i]);
    }
}

template <typename Type>
void expectTensorValues(const Tensor<Type>& tensor,
                        const std::vector<float>& expected,
                        float tolerance = 0.0f)
{
    ASSERT_EQ(static_cast<size_t>(tensor.elementCount()), expected.size());

    const auto* data = tensor.memory().hostData();
    for(size_t idx = 0; idx < expected.size(); ++idx)
    {
        if(tolerance == 0.0f)
        {
            EXPECT_EQ(data[idx], static_cast<Type>(expected[idx]))
                << "Mismatch at flat index " << idx;
        }
        else
        {
            EXPECT_NEAR(static_cast<float>(data[idx]), expected[idx], tolerance)
                << "Mismatch at flat index " << idx;
        }
    }
}

} // namespace

/* ============================= Validation tests ============================= */

class TestCpuFpReferenceReduction : public ::testing::Test
{
};

TEST_F(TestCpuFpReferenceReduction, ValidateRankMismatch)
{
    auto x = createTensor<float>({2, 3, 4});
    auto y = createTensor<float>({2, 3});

    EXPECT_THROW((CpuFpReferenceReduction::reduce<float, float, float>(x, y, ReductionMode::ADD)),
                 std::invalid_argument);
}

TEST_F(TestCpuFpReferenceReduction, ValidateInvalidYDim)
{
    auto x = createTensor<float>({2, 3, 4});
    auto y = createTensor<float>({2, 2, 4}); // dim 1: 2 != 3 and != 1

    EXPECT_THROW((CpuFpReferenceReduction::reduce<float, float, float>(x, y, ReductionMode::ADD)),
                 std::invalid_argument);
}

TEST_F(TestCpuFpReferenceReduction, ValidateNoReducedDim)
{
    auto x = createTensor<float>({2, 3, 4});
    auto y = createTensor<float>({2, 3, 4}); // no dim reduced

    EXPECT_THROW((CpuFpReferenceReduction::reduce<float, float, float>(x, y, ReductionMode::ADD)),
                 std::invalid_argument);
}

TEST_F(TestCpuFpReferenceReduction, NotSetModeThrows)
{
    auto x = createTensor<float>({2, 3});
    auto y = createTensor<float>({2, 1});
    fillSequential(x);

    EXPECT_THROW(
        (CpuFpReferenceReduction::reduce<float, float, float>(x, y, ReductionMode::NOT_SET)),
        std::invalid_argument);
}

/* ============================= Mode tests ============================= */
// All mode tests use X = [[1,2,3],[4,5,6]] shape (2,3), reduce dim 1 -> Y shape (2,1)

TEST_F(TestCpuFpReferenceReduction, Add)
{
    auto x = createTensor<float>({2, 3});
    auto y = createTensor<float>({2, 1});
    fillSequential(x); // [1,2,3, 4,5,6]

    CpuFpReferenceReduction::reduce<float, float, float>(x, y, ReductionMode::ADD);

    // Y[0,0] = 1+2+3 = 6,  Y[1,0] = 4+5+6 = 15
    expectTensorValues(y, {6.0f, 15.0f});
}

TEST_F(TestCpuFpReferenceReduction, Mul)
{
    auto x = createTensor<float>({2, 3});
    auto y = createTensor<float>({2, 1});
    fillSequential(x);

    CpuFpReferenceReduction::reduce<float, float, float>(x, y, ReductionMode::MUL);

    // Y[0,0] = 1*2*3 = 6,  Y[1,0] = 4*5*6 = 120
    expectTensorValues(y, {6.0f, 120.0f});
}

TEST_F(TestCpuFpReferenceReduction, Min)
{
    auto x = createTensor<float>({2, 3});
    auto y = createTensor<float>({2, 1});
    fillSequential(x);

    CpuFpReferenceReduction::reduce<float, float, float>(x, y, ReductionMode::MIN_OP);

    // Y[0,0] = min(1,2,3) = 1,  Y[1,0] = min(4,5,6) = 4
    expectTensorValues(y, {1.0f, 4.0f});
}

TEST_F(TestCpuFpReferenceReduction, Max)
{
    auto x = createTensor<float>({2, 3});
    auto y = createTensor<float>({2, 1});
    fillSequential(x);

    CpuFpReferenceReduction::reduce<float, float, float>(x, y, ReductionMode::MAX_OP);

    // Y[0,0] = max(1,2,3) = 3,  Y[1,0] = max(4,5,6) = 6
    expectTensorValues(y, {3.0f, 6.0f});
}

TEST_F(TestCpuFpReferenceReduction, Avg)
{
    auto x = createTensor<float>({2, 3});
    auto y = createTensor<float>({2, 1});
    fillSequential(x);

    CpuFpReferenceReduction::reduce<float, float, float>(x, y, ReductionMode::AVG);

    // Y[0,0] = (1+2+3)/3 = 2,  Y[1,0] = (4+5+6)/3 = 5
    expectTensorValues(y, {2.0f, 5.0f});
}

TEST_F(TestCpuFpReferenceReduction, Amax)
{
    auto x = createTensor<float>({2, 3});
    auto y = createTensor<float>({2, 1});
    fillValues(x, {-3.0f, 1.0f, 2.0f, -5.0f, 4.0f, 0.0f});

    CpuFpReferenceReduction::reduce<float, float, float>(x, y, ReductionMode::AMAX);

    // Y[0,0] = max(|-3|,|1|,|2|) = 3,  Y[1,0] = max(|-5|,|4|,|0|) = 5
    expectTensorValues(y, {3.0f, 5.0f});
}

TEST_F(TestCpuFpReferenceReduction, Norm1)
{
    auto x = createTensor<float>({2, 3});
    auto y = createTensor<float>({2, 1});
    fillValues(x, {-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f});

    CpuFpReferenceReduction::reduce<float, float, float>(x, y, ReductionMode::NORM1);

    // Y[0,0] = |-1|+|2|+|-3| = 6,  Y[1,0] = |4|+|-5|+|6| = 15
    expectTensorValues(y, {6.0f, 15.0f});
}

TEST_F(TestCpuFpReferenceReduction, Norm2)
{
    auto x = createTensor<float>({2, 3});
    auto y = createTensor<float>({2, 1});
    fillSequential(x);

    CpuFpReferenceReduction::reduce<float, float, float>(x, y, ReductionMode::NORM2);

    // Y[0,0] = sqrt(1+4+9) = sqrt(14),  Y[1,0] = sqrt(16+25+36) = sqrt(77)
    expectTensorValues(y, {std::sqrt(14.0f), std::sqrt(77.0f)}, 1e-5f);
}

TEST_F(TestCpuFpReferenceReduction, MulNoZeros)
{
    auto x = createTensor<float>({2, 3});
    auto y = createTensor<float>({2, 1});
    fillValues(x, {2.0f, 0.0f, 3.0f, 0.0f, 5.0f, 4.0f});

    CpuFpReferenceReduction::reduce<float, float, float>(x, y, ReductionMode::MUL_NO_ZEROS);

    // Y[0,0] = 2*3 = 6 (skip 0),  Y[1,0] = 5*4 = 20 (skip 0)
    expectTensorValues(y, {6.0f, 20.0f});
}

TEST_F(TestCpuFpReferenceReduction, MinWithNegativeValues)
{
    auto x = createTensor<float>({2, 3});
    auto y = createTensor<float>({2, 1});
    fillValues(x, {-3.0f, 1.0f, -7.0f, 4.0f, -2.0f, 5.0f});

    CpuFpReferenceReduction::reduce<float, float, float>(x, y, ReductionMode::MIN_OP);

    // Y[0,0] = min(-3, 1, -7) = -7,  Y[1,0] = min(4, -2, 5) = -2
    expectTensorValues(y, {-7.0f, -2.0f});
}

TEST_F(TestCpuFpReferenceReduction, MaxWithNegativeValues)
{
    auto x = createTensor<float>({2, 3});
    auto y = createTensor<float>({2, 1});
    fillValues(x, {-3.0f, -1.0f, -7.0f, -4.0f, -2.0f, -5.0f});

    CpuFpReferenceReduction::reduce<float, float, float>(x, y, ReductionMode::MAX_OP);

    // Y[0,0] = max(-3, -1, -7) = -1,  Y[1,0] = max(-4, -2, -5) = -2
    expectTensorValues(y, {-1.0f, -2.0f});
}

/* ============================= Multi-dim reduction ============================= */

TEST_F(TestCpuFpReferenceReduction, AddReduceMultipleDims)
{
    // X shape (2, 3, 4), Y shape (1, 3, 1) — reduce dims 0 and 2
    auto x = createTensor<float>({2, 3, 4});
    auto y = createTensor<float>({1, 3, 1});
    fillSequential(x, 0.0f); // [0, 1, 2, ..., 23]

    CpuFpReferenceReduction::reduce<float, float, float>(x, y, ReductionMode::ADD);

    // Y[0,0,0] = sum(X[:,0,:]) = (0+1+2+3)+(12+13+14+15) = 60
    // Y[0,1,0] = sum(X[:,1,:]) = (4+5+6+7)+(16+17+18+19) = 92
    // Y[0,2,0] = sum(X[:,2,:]) = (8+9+10+11)+(20+21+22+23) = 124
    expectTensorValues(y, {60.0f, 92.0f, 124.0f});
}

TEST_F(TestCpuFpReferenceReduction, AddReduceAllDims)
{
    // X shape (2, 3), Y shape (1, 1) — reduce all dims
    auto x = createTensor<float>({2, 3});
    auto y = createTensor<float>({1, 1});
    fillSequential(x); // [1, 2, 3, 4, 5, 6]

    CpuFpReferenceReduction::reduce<float, float, float>(x, y, ReductionMode::ADD);

    // Y[0,0] = 1+2+3+4+5+6 = 21
    expectTensorValues(y, {21.0f});
}

TEST_F(TestCpuFpReferenceReduction, MaxReduceDim0)
{
    // X shape (2, 3), Y shape (1, 3) — reduce dim 0
    auto x = createTensor<float>({2, 3});
    auto y = createTensor<float>({1, 3});
    fillSequential(x); // [1,2,3, 4,5,6]

    CpuFpReferenceReduction::reduce<float, float, float>(x, y, ReductionMode::MAX_OP);

    // Y[0,0] = max(1,4) = 4,  Y[0,1] = max(2,5) = 5,  Y[0,2] = max(3,6) = 6
    expectTensorValues(y, {4.0f, 5.0f, 6.0f});
}

TEST_F(TestCpuFpReferenceReduction, AddReduceSingleInnerAxis)
{
    // X shape (2,3,4,8), Y shape (2,3,1,8) — reduce dim 2 only
    //
    // X = [[[[  0,   1,   2,   3,   4,   5,   6,   7],
    //        [  8,   9,  10,  11,  12,  13,  14,  15],
    //        [ 16,  17,  18,  19,  20,  21,  22,  23],
    //        [ 24,  25,  26,  27,  28,  29,  30,  31]],
    //
    //       [[ 32,  33,  34,  35,  36,  37,  38,  39],
    //        [ 40,  41,  42,  43,  44,  45,  46,  47],
    //        [ 48,  49,  50,  51,  52,  53,  54,  55],
    //        [ 56,  57,  58,  59,  60,  61,  62,  63]],
    //
    //       [[ 64,  65,  66,  67,  68,  69,  70,  71],
    //        [ 72,  73,  74,  75,  76,  77,  78,  79],
    //        [ 80,  81,  82,  83,  84,  85,  86,  87],
    //        [ 88,  89,  90,  91,  92,  93,  94,  95]]],
    //
    //
    //      [[[ 96,  97,  98,  99, 100, 101, 102, 103],
    //        [104, 105, 106, 107, 108, 109, 110, 111],
    //        [112, 113, 114, 115, 116, 117, 118, 119],
    //        [120, 121, 122, 123, 124, 125, 126, 127]],
    //
    //       [[128, 129, 130, 131, 132, 133, 134, 135],
    //        [136, 137, 138, 139, 140, 141, 142, 143],
    //        [144, 145, 146, 147, 148, 149, 150, 151],
    //        [152, 153, 154, 155, 156, 157, 158, 159]],
    //
    //       [[160, 161, 162, 163, 164, 165, 166, 167],
    //        [168, 169, 170, 171, 172, 173, 174, 175],
    //        [176, 177, 178, 179, 180, 181, 182, 183],
    //        [184, 185, 186, 187, 188, 189, 190, 191]]]])
    //
    // Y = [[[[ 48,  52,  56,  60,  64,  68,  72,  76]],
    //
    //       [[176, 180, 184, 188, 192, 196, 200, 204]],
    //
    //       [[304, 308, 312, 316, 320, 324, 328, 332]]],
    //
    //
    //      [[[432, 436, 440, 444, 448, 452, 456, 460]],
    //
    //       [[560, 564, 568, 572, 576, 580, 584, 588]],
    //
    //       [[688, 692, 696, 700, 704, 708, 712, 716]]]])
    auto x = createTensor<float>({2, 3, 4, 8});
    auto y = createTensor<float>({2, 3, 1, 8});
    // X[n,c,h,w] = n*96 + c*32 + h*8 + w
    fillSequential(x, 0.0f);

    CpuFpReferenceReduction::reduce<float, float, float>(x, y, ReductionMode::ADD);

    // Y[n,c,0,w] = sum over h: X[n,c,h,w] for h in {0,1,2,3}
    // Y[0,0,0,0] = X[0,0,0,0]+X[0,0,1,0]+X[0,0,2,0]+X[0,0,3,0] = 0+8+16+24 = 48
    // clang-format off
    expectTensorValues(y, {
         48,  52,  56,  60,  64,  68,  72,  76,
        176, 180, 184, 188, 192, 196, 200, 204,
        304, 308, 312, 316, 320, 324, 328, 332,
        432, 436, 440, 444, 448, 452, 456, 460,
        560, 564, 568, 572, 576, 580, 584, 588,
        688, 692, 696, 700, 704, 708, 712, 716,
    });
    // clang-format on
}

/* ============================= Multi-type tests ============================= */

template <typename T1, typename T2, typename T3>
struct ReductionTypeTuple
{
    using XDataType = T1;
    using YDataType = T2;
    using ComputeDataType = T3;
};

using ReductionTypes = ::testing::Types<ReductionTypeTuple<float, float, float>,
                                        ReductionTypeTuple<half, half, float>,
                                        ReductionTypeTuple<bfloat16, bfloat16, float>,
                                        ReductionTypeTuple<half, float, float>,
                                        ReductionTypeTuple<bfloat16, float, float>>;

template <class T>
class CpuFpReferenceReductionTyped : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceReductionTyped, ReductionTypes, );

TYPED_TEST(CpuFpReferenceReductionTyped, AddReduceDim1)
{
    using X = typename TypeParam::XDataType;
    using Y = typename TypeParam::YDataType;
    using C = typename TypeParam::ComputeDataType;

    auto x = createTensor<X>({2, 3});
    auto y = createTensor<Y>({2, 1});
    fillSequential(x);

    CpuFpReferenceReduction::reduce<X, Y, C>(x, y, ReductionMode::ADD);

    expectTensorValues(y, {6.0f, 15.0f});
}

TYPED_TEST(CpuFpReferenceReductionTyped, MinReduceDim1)
{
    using X = typename TypeParam::XDataType;
    using Y = typename TypeParam::YDataType;
    using C = typename TypeParam::ComputeDataType;

    auto x = createTensor<X>({2, 3});
    auto y = createTensor<Y>({2, 1});
    fillSequential(x);

    CpuFpReferenceReduction::reduce<X, Y, C>(x, y, ReductionMode::MIN_OP);

    expectTensorValues(y, {1.0f, 4.0f});
}

TYPED_TEST(CpuFpReferenceReductionTyped, AddReduceMultipleDims)
{
    using X = typename TypeParam::XDataType;
    using Y = typename TypeParam::YDataType;
    using C = typename TypeParam::ComputeDataType;

    auto x = createTensor<X>({2, 3, 4});
    auto y = createTensor<Y>({1, 3, 1});
    fillSequential(x, 0.0f);

    CpuFpReferenceReduction::reduce<X, Y, C>(x, y, ReductionMode::ADD);

    expectTensorValues(y, {60.0f, 92.0f, 124.0f});
}
