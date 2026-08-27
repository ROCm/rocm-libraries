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
#define SKIP_IF_NO_ROCRAND() (void)0
#else
#define SKIP_IF_NO_ROCRAND() GTEST_SKIP() << "rocRAND not available. Skipping test."
#endif

// ---------------------------------------------------------------------
// No Nan/Inf and range checks
// ---------------------------------------------------------------------

TEST(TestFillTensorWithRandomValues, FloatValuesAreWithinRange)
{
    SKIP_IF_NO_ROCRAND();
    SKIP_IF_NO_DEVICES();

    Tensor<float> tensor({10, 10, 100, 100});
    gpu_fp_reference_tensor::fillWithRandomValues(tensor, -1.0f, 10.0f, 42);

    const auto* data = static_cast<const float*>(tensor.rawHostData());
    const auto count = tensor.elementSpace();
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
    SKIP_IF_NO_ROCRAND();
    SKIP_IF_NO_DEVICES();

    Tensor<double> tensor({10, 10, 100, 100});
    gpu_fp_reference_tensor::fillWithRandomValues(tensor, -1.0, 10.0, 42);

    const auto* data = static_cast<const double*>(tensor.rawHostData());
    const auto count = tensor.elementSpace();
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
    SKIP_IF_NO_ROCRAND();
    SKIP_IF_NO_DEVICES();

    Tensor<HalfType> tensor({10, 10, 100, 100});
    gpu_fp_reference_tensor::fillWithRandomValues<HalfType>(
        tensor, static_cast<HalfType>(-1.0f), static_cast<HalfType>(10.0f), 42);

    const auto* data = static_cast<const HalfType*>(tensor.rawHostData());
    const auto count = tensor.elementSpace();
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
    SKIP_IF_NO_ROCRAND();
    SKIP_IF_NO_DEVICES();

    Tensor<BFloat16Type> tensor({10, 10, 100, 100});
    gpu_fp_reference_tensor::fillWithRandomValues<BFloat16Type>(
        tensor, static_cast<BFloat16Type>(-1.0f), static_cast<BFloat16Type>(10.0f), 42);

    const auto* data = static_cast<const BFloat16Type*>(tensor.rawHostData());
    const auto count = tensor.elementSpace();
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
    SKIP_IF_NO_ROCRAND();
    SKIP_IF_NO_DEVICES();

    Tensor<float> tensor({10, 10, 100, 100});
    gpu_fp_reference_tensor::fillWithRandomValues(tensor, 2.0f, 20.0f, 42);

    const auto* data = static_cast<const float*>(tensor.rawHostData());
    const auto count = tensor.elementSpace();

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
    SKIP_IF_NO_ROCRAND();
    SKIP_IF_NO_DEVICES();

    Tensor<HalfType> tensor({10, 10, 100, 100});
    gpu_fp_reference_tensor::fillWithRandomValues<HalfType>(
        tensor, static_cast<HalfType>(2.0f), static_cast<HalfType>(20.0f), 42);

    const auto* data = static_cast<const HalfType*>(tensor.rawHostData());
    const auto count = tensor.elementSpace();

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
    SKIP_IF_NO_ROCRAND();
    SKIP_IF_NO_DEVICES();

    Tensor<BFloat16Type> tensor({10, 10, 100, 100});
    gpu_fp_reference_tensor::fillWithRandomValues<BFloat16Type>(
        tensor, static_cast<BFloat16Type>(2.0f), static_cast<BFloat16Type>(20.0f), 42);

    const auto* data = static_cast<const BFloat16Type*>(tensor.rawHostData());
    const auto count = tensor.elementSpace();

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

    EXPECT_NEAR(mean, 11.0, 0.125);
    EXPECT_NEAR(variance, 27.0, 0.25);
}

// ---------------------------------------------------------------------
// Reproducibility
// ---------------------------------------------------------------------

TEST(TestFillTensorWithRandomValues, SameSeedProducesSameValues)
{
    SKIP_IF_NO_ROCRAND();
    SKIP_IF_NO_DEVICES();

    Tensor<float> tensor1({10, 10, 100, 100});
    Tensor<float> tensor2({10, 10, 100, 100});

    gpu_fp_reference_tensor::fillWithRandomValues(tensor1, 5.0f, 100.0f, 42);
    gpu_fp_reference_tensor::fillWithRandomValues(tensor2, 5.0f, 100.0f, 42);

    const auto* data1 = static_cast<const float*>(tensor1.rawHostData());
    const auto* data2 = static_cast<const float*>(tensor2.rawHostData());
    const auto count = tensor1.elementSpace();

    for(size_t i = 0; i < count; ++i)
    {
        EXPECT_EQ(data1[i], data2[i]);
    }
}

TEST(TestFillTensorWithRandomValues, DifferentSeedsProduceDifferentValues)
{
    SKIP_IF_NO_ROCRAND();
    SKIP_IF_NO_DEVICES();

    Tensor<float> tensor1({10, 10, 100, 100});
    Tensor<float> tensor2({10, 10, 100, 100});

    gpu_fp_reference_tensor::fillWithRandomValues(tensor1, 5.0f, 100.0f, 42);
    gpu_fp_reference_tensor::fillWithRandomValues(tensor2, 5.0f, 100.0f, 43);

    const auto* data1 = static_cast<const float*>(tensor1.rawHostData());
    const auto* data2 = static_cast<const float*>(tensor2.rawHostData());
    const auto count = tensor1.elementSpace();

    size_t equalCount = 0;
    for(size_t i = 0; i < count; ++i)
    {
        if(data1[i] == data2[i])
        {
            ++equalCount;
        }
    }

    EXPECT_LE(equalCount, 1);
}

// ---------------------------------------------------------------------
// Strided tensors
// ---------------------------------------------------------------------

TEST(TestFillTensorWithRandomValues, FloatStridedTensor)
{
    SKIP_IF_NO_ROCRAND();
    SKIP_IF_NO_DEVICES();

    Tensor<float> tensor({10, 10, 100, 100}, {100, 1000, 10000, 1}); // Strided tensor
    gpu_fp_reference_tensor::fillWithRandomValues(tensor, 2.0f, 50.0f, 42);

    const auto* data = static_cast<const float*>(tensor.rawHostData());
    const auto count = tensor.elementSpace();

    // Range checks
    for(size_t i = 0; i < count; ++i)
    {
        const auto value = data[i];
        EXPECT_FALSE(std::isnan(value));
        EXPECT_FALSE(std::isinf(value));
        EXPECT_GE(value, 2.0f);
        EXPECT_LE(value, 50.0f);
    }

    // Mean and variance checks
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

    EXPECT_NEAR(mean, 26.0, 1.0e-02);
    EXPECT_NEAR(variance, 192.0, 1.0e-01);
}

TEST(TestFillTensorWithRandomValues, HalfStridedTensor)
{
    SKIP_IF_NO_ROCRAND();
    SKIP_IF_NO_DEVICES();

    Tensor<HalfType> tensor({1000, 10, 10, 1000}, {100, 1, 100, 10}); // Strided tensor
    gpu_fp_reference_tensor::fillWithRandomValues<HalfType>(
        tensor, static_cast<HalfType>(1.0f), static_cast<HalfType>(10.0f), 42);

    const auto* data = static_cast<const HalfType*>(tensor.rawHostData());
    const auto count = tensor.elementSpace();

    // Range checks
    for(size_t i = 0; i < count; ++i)
    {
        const auto value = static_cast<float>(data[i]);
        EXPECT_FALSE(std::isnan(value));
        EXPECT_FALSE(std::isinf(value));
        EXPECT_GE(value, 1.0f);
        EXPECT_LE(value, 10.0f);
    }

    // Mean and variance checks
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

    EXPECT_NEAR(mean, 5.5, 1.0e-02);
    EXPECT_NEAR(variance, 6.75, 1.0e-01);
}

TEST(TestFillTensorWithRandomValues, BFloat16StridedTensor)
{
    SKIP_IF_NO_ROCRAND();
    SKIP_IF_NO_DEVICES();

    Tensor<BFloat16Type> tensor({100, 1, 10, 100}, {1000, 1, 10, 100}); // Strided tensor
    gpu_fp_reference_tensor::fillWithRandomValues<BFloat16Type>(
        tensor, static_cast<BFloat16Type>(3.0f), static_cast<BFloat16Type>(10.0f), 42);

    const auto* data = static_cast<const BFloat16Type*>(tensor.rawHostData());
    const auto count = tensor.elementSpace();

    // Range checks
    for(size_t i = 0; i < count; ++i)
    {
        const auto value = static_cast<float>(data[i]);
        EXPECT_FALSE(std::isnan(value));
        EXPECT_FALSE(std::isinf(value));
        EXPECT_GE(value, 3.0f);
        EXPECT_LE(value, 10.0f);
    }

    // Mean and variance checks
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

    EXPECT_NEAR(mean, 6.5, 0.125);
    EXPECT_NEAR(variance, 4.083333, 0.25);
}

// ---------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------

TEST(TestFillTensorWithRandomValues, ConstantTensor)
{
    SKIP_IF_NO_ROCRAND();
    SKIP_IF_NO_DEVICES();

    Tensor<float> tensor({10, 10, 100, 100});
    gpu_fp_reference_tensor::fillWithRandomValues(tensor, 5.0f, 5.0f, 42);

    const auto* data = static_cast<const float*>(tensor.rawHostData());
    const auto count = tensor.elementSpace();

    for(size_t i = 0; i < count; ++i)
    {
        EXPECT_EQ(data[i], 5.0f);
    }
}

TEST(TestFillTensorWithRandomValues, SingleElementTensor)
{
    SKIP_IF_NO_ROCRAND();
    SKIP_IF_NO_DEVICES();

    Tensor<float> tensor({1, 1, 1, 1});
    gpu_fp_reference_tensor::fillWithRandomValues(tensor, -10.0f, 10.0f, 42);

    const auto* data = static_cast<const float*>(tensor.rawHostData());
    EXPECT_EQ(tensor.elementSpace(), 1);
    EXPECT_GE(data[0], -10.0f);
    EXPECT_LE(data[0], 10.0f);
}

TEST(TestFillTensorWithRandomValues, TensorSizeNotMultipleOfBlockSize)
{
    SKIP_IF_NO_ROCRAND();
    SKIP_IF_NO_DEVICES();

    constexpr size_t TENSOR_SIZE = 256 * 10000 + 123; // Not a multiple of BLOCK_SIZE (256)
    Tensor<float> tensor({1, 1, 1, TENSOR_SIZE});
    gpu_fp_reference_tensor::fillWithRandomValues(tensor, 3.0f, 10.0f, 42);

    const auto* data = static_cast<const float*>(tensor.rawHostData());
    const auto count = tensor.elementSpace();

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
