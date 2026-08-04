// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn-gpu-ref/GpuFpReferenceCommon.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_gpu_ref::common;
using HalfType = hipdnn_data_sdk::types::half;
using BFloat16Type = hipdnn_data_sdk::types::bfloat16;

#if defined(USE_ROCRAND)

// ---------------------------------------------------------------------
// No Nan/Inf and range checks
// ---------------------------------------------------------------------

TEST(TestFillTensorWithRandomValues, FloatValuesAreWithinRange)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> tensor({10, 10, 100, 100});
    GpuFpReferenceTensor::fillWithRandomValues(tensor, -1.0f, 10.0f, 42);

    const auto* data = static_cast<const float*>(tensor.rawHostData());
    const auto count = tensor.elementCount();
    for(size_t i = 0; i < count; ++i)
    {
        const auto value = data[i];
        EXPECT_FALSE(std::isnan(value));
        EXPECT_FALSE(std::isinf(value));
        EXPECT_GE(value, -1.0f);
        EXPECT_LE(value, 10.0f);
    }
}

TEST(TestFillTensorWithRandomValues, DoubleValuesAreWithinRange)
{
    SKIP_IF_NO_DEVICES();

    Tensor<double> tensor({10, 10, 100, 100});
    GpuFpReferenceTensor::fillWithRandomValues(tensor, -1.0, 10.0, 42);

    const auto* data = static_cast<const double*>(tensor.rawHostData());
    const auto count = tensor.elementCount();
    for(size_t i = 0; i < count; ++i)
    {
        const auto value = data[i];
        EXPECT_FALSE(std::isnan(value));
        EXPECT_FALSE(std::isinf(value));
        EXPECT_GE(value, -1.0);
        EXPECT_LE(value, 10.0);
    }
}

TEST(TestFillTensorWithRandomValues, HalfValuesAreWithinRange)
{
    SKIP_IF_NO_DEVICES();

    Tensor<HalfType> tensor({10, 10, 100, 100});
    GpuFpReferenceTensor::fillWithRandomValues<HalfType>(
        tensor, static_cast<HalfType>(-1.0f), static_cast<HalfType>(10.0f), 42);

    const auto* data = static_cast<const HalfType*>(tensor.rawHostData());
    const auto count = tensor.elementCount();
    for(size_t i = 0; i < count; ++i)
    {
        const auto value = static_cast<float>(data[i]);
        EXPECT_FALSE(std::isnan(value));
        EXPECT_FALSE(std::isinf(value));
        EXPECT_GE(value, -1.0f);
        EXPECT_LE(value, 10.0f);
    }
}

TEST(TestFillTensorWithRandomValues, BFloat16ValuesAreWithinRange)
{
    SKIP_IF_NO_DEVICES();

    Tensor<BFloat16Type> tensor({10, 10, 100, 100});
    GpuFpReferenceTensor::fillWithRandomValues<BFloat16Type>(
        tensor, static_cast<BFloat16Type>(-1.0f), static_cast<BFloat16Type>(10.0f), 42);

    const auto* data = static_cast<const BFloat16Type*>(tensor.rawHostData());
    const auto count = tensor.elementCount();
    for(size_t i = 0; i < count; ++i)
    {
        const auto value = static_cast<float>(data[i]);
        EXPECT_FALSE(std::isnan(value));
        EXPECT_FALSE(std::isinf(value));
        EXPECT_GE(value, -1.0f);
        EXPECT_LE(value, 10.0f);
    }
}

// ---------------------------------------------------------------------
// Mean / variance checks
// ---------------------------------------------------------------------

TEST(TestFillTensorWithRandomValues, FloatMeanAndVariance)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> tensor({10, 10, 100, 100});
    GpuFpReferenceTensor::fillWithRandomValues(tensor, 2.0f, 20.0f, 42);

    const auto* data = static_cast<const float*>(tensor.rawHostData());
    const auto count = tensor.elementCount();

    double sum = 0.0;
    double sumSq = 0.0;
    for(size_t i = 0; i < count; ++i)
    {
        const auto val = static_cast<double>(data[i]);
        sum += val;
        sumSq += val * val;
    }

    const double mean = sum / static_cast<double>(count);
    const double variance = (sumSq / static_cast<double>(count)) - (mean * mean);

    EXPECT_NEAR(mean, 11.0, 1.0e-02);
    EXPECT_NEAR(variance, 27.0, 1.0e-01);
}

TEST(TestFillTensorWithRandomValues, HalfMeanAndVariance)
{
    SKIP_IF_NO_DEVICES();

    Tensor<HalfType> tensor({10, 10, 100, 100});
    GpuFpReferenceTensor::fillWithRandomValues<HalfType>(
        tensor, static_cast<HalfType>(2.0f), static_cast<HalfType>(20.0f), 42);

    const auto* data = static_cast<const HalfType*>(tensor.rawHostData());
    const auto count = tensor.elementCount();

    double sum = 0.0;
    double sumSq = 0.0;
    for(size_t i = 0; i < count; ++i)
    {
        const auto val = static_cast<double>(data[i]);
        sum += val;
        sumSq += val * val;
    }

    const double mean = sum / static_cast<double>(count);
    const double variance = (sumSq / static_cast<double>(count)) - (mean * mean);

    EXPECT_NEAR(mean, 11.0, 1.0e-02);
    EXPECT_NEAR(variance, 27.0, 1.0e-01);
}

TEST(TestFillTensorWithRandomValues, BFloat16MeanAndVariance)
{
    SKIP_IF_NO_DEVICES();

    Tensor<BFloat16Type> tensor({10, 10, 100, 100});
    GpuFpReferenceTensor::fillWithRandomValues<BFloat16Type>(
        tensor, static_cast<BFloat16Type>(2.0f), static_cast<BFloat16Type>(20.0f), 42);

    const auto* data = static_cast<const BFloat16Type*>(tensor.rawHostData());
    const auto count = tensor.elementCount();

    double sum = 0.0;
    double sumSq = 0.0;
    for(size_t i = 0; i < count; ++i)
    {
        const auto val = static_cast<double>(data[i]);
        sum += val;
        sumSq += val * val;
    }

    const double mean = sum / static_cast<double>(count);
    const double variance = (sumSq / static_cast<double>(count)) - (mean * mean);

    EXPECT_NEAR(mean, 11.0, 1.0e-02);
    EXPECT_NEAR(variance, 27.0, 1.0e-01);
}

// ---------------------------------------------------------------------
// Reproducibility
// ---------------------------------------------------------------------

TEST(TestFillTensorWithRandomValues, SameSeedProducesSameValues)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> tensor1({10, 10, 100, 100});
    Tensor<float> tensor2({10, 10, 100, 100});

    GpuFpReferenceTensor::fillWithRandomValues(tensor1, 5.0f, 100.0f, 42);
    GpuFpReferenceTensor::fillWithRandomValues(tensor2, 5.0f, 100.0f, 42);

    const auto* data1 = static_cast<const float*>(tensor1.rawHostData());
    const auto* data2 = static_cast<const float*>(tensor2.rawHostData());
    const auto count = tensor1.elementCount();

    for(size_t i = 0; i < count; ++i)
    {
        EXPECT_EQ(data1[i], data2[i]);
    }
}

TEST(TestFillTensorWithRandomValues, DifferentSeedsProduceDifferentValues)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> tensor1({10, 10, 100, 100});
    Tensor<float> tensor2({10, 10, 100, 100});

    GpuFpReferenceTensor::fillWithRandomValues(tensor1, 5.0f, 100.0f, 42);
    GpuFpReferenceTensor::fillWithRandomValues(tensor2, 5.0f, 100.0f, 43);

    const auto* data1 = static_cast<const float*>(tensor1.rawHostData());
    const auto* data2 = static_cast<const float*>(tensor2.rawHostData());
    const auto count = tensor1.elementCount();

    bool areDifferent = false;
    for(size_t i = 0; i < count; ++i)
    {
        if(data1[i] != data2[i])
        {
            areDifferent = true;
            break;
        }
    }

    EXPECT_TRUE(areDifferent);
}

// ---------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------

TEST(TestFillTensorWithRandomValues, ConstantTensor)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> tensor({10, 10, 100, 100});
    GpuFpReferenceTensor::fillWithRandomValues(tensor, 5.0f, 5.0f, 42);

    const auto* data = static_cast<const float*>(tensor.rawHostData());
    const auto count = tensor.elementCount();

    for(size_t i = 0; i < count; ++i)
    {
        EXPECT_EQ(data[i], 5.0f);
    }
}

TEST(TestFillTensorWithRandomValues, SingleElementTensor)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> tensor({1, 1, 1, 1});
    GpuFpReferenceTensor::fillWithRandomValues(tensor, -10.0f, 10.0f, 42);

    const auto* data = static_cast<const float*>(tensor.rawHostData());
    EXPECT_EQ(tensor.elementCount(), 1);
    EXPECT_GE(data[0], -10.0f);
    EXPECT_LE(data[0], 10.0f);
}

TEST(TestFillTensorWithRandomValues, TensorSizeNotMultipleOfBlockSize)
{
    SKIP_IF_NO_DEVICES();

    constexpr size_t TENSOR_SIZE = 256 * 10000 + 123; // Not a multiple of BLOCK_SIZE (256)
    Tensor<float> tensor({1, 1, 1, TENSOR_SIZE});
    GpuFpReferenceTensor::fillWithRandomValues(tensor, 3.0f, 10.0f, 42);

    const auto* data = static_cast<const float*>(tensor.rawHostData());
    const auto count = tensor.elementCount();

    double sum = 0.0;
    double sumSq = 0.0;
    for(size_t i = 0; i < count; ++i)
    {
        const auto val = static_cast<double>(data[i]);
        sum += val;
        sumSq += val * val;
    }

    const double mean = sum / static_cast<double>(count);
    const double variance = (sumSq / static_cast<double>(count)) - (mean * mean);

    EXPECT_NEAR(mean, 6.5, 1.0e-02);
    EXPECT_NEAR(variance, 4.083333, 1.0e-01);
}

#endif // defined(USE_ROCRAND)
