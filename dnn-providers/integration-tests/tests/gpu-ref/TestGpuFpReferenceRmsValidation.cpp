// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMiopenRmsValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <hipdnn-gpu-ref/GpuFpReferenceRmsValidation.hpp>

#include <cstdint>
#include <limits>
#include <vector>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_gpu_ref;

namespace
{

template <typename T>
void fill(Tensor<T>& tensor, float value)
{
    auto* host = tensor.memory().hostData();
    for(size_t i = 0; i < tensor.memory().count(); ++i)
    {
        host[i] = static_cast<T>(value);
    }
}

template <typename T>
class TestGpuFpReferenceRmsValidation : public ::testing::Test
{
};

using RmsTypes = ::testing::Types<float, half, bfloat16, double>;
TYPED_TEST_SUITE(TestGpuFpReferenceRmsValidation, RmsTypes, );

// One element drifts by 0.5 on a tensor of ones: relative RMS is
// 0.5 / (sqrt(32) * 1.5) ~= 0.0589. Both sites must land on the same side of a
// threshold either way, and it must be the right side.
TYPED_TEST(TestGpuFpReferenceRmsValidation, AgreesWithCpuOnBothSidesOfTheThreshold)
{
    SKIP_IF_NO_DEVICES();

    Tensor<TypeParam> ref({4, 8});
    Tensor<TypeParam> impl({4, 8});
    fill(ref, 1.0f);
    fill(impl, 1.0f);
    impl.memory().hostData()[0] = static_cast<TypeParam>(1.5f);

    for(const float threshold : {0.05f, 0.07f})
    {
        const bool expected = threshold > 0.0589f;
        const GpuFpReferenceRmsValidation<TypeParam> gpu(threshold);
        const CpuFpReferenceMiopenRmsValidation<TypeParam> cpu(static_cast<TypeParam>(threshold));

        EXPECT_EQ(gpu.allClose(ref, impl), expected) << "threshold " << threshold;
        EXPECT_EQ(cpu.allClose(ref, impl), expected) << "threshold " << threshold;
    }
}

// 64K elements is 256 blocks. The only drift sits in the last one, so both the sum and
// the maximum magnitude have to survive the cross-block fold: relative RMS is
// 1 / (sqrt(65536) * 2) ~= 1.953e-3. Losing the last block's sum passes 1.9e-3; losing
// its maximum (2.0 -> 1.0) doubles the ratio and fails 2.0e-3.
TEST(TestGpuFpReferenceRmsValidationReduction, EveryBlockContributesSumAndMaximum)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> ref({64, 32, 32});
    Tensor<float> impl({64, 32, 32});
    fill(ref, 1.0f);
    fill(impl, 1.0f);
    impl.memory().hostData()[impl.elementCount() - 1] = 2.0f;

    EXPECT_FALSE(GpuFpReferenceRmsValidation<float>(1.9e-3f).allClose(ref, impl));
    EXPECT_TRUE(GpuFpReferenceRmsValidation<float>(2.0e-3f).allClose(ref, impl));
}

// An unwritten output element is still the NaN sentinel. No tolerance may excuse it,
// wherever in the tensor it sits.
TEST(TestGpuFpReferenceRmsValidationReduction, NanAnywhereFails)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> ref({64, 32, 32});
    Tensor<float> impl({64, 32, 32});
    fill(ref, 1.0f);
    fill(impl, 1.0f);
    impl.memory().hostData()[impl.elementCount() - 1] = std::numeric_limits<float>::quiet_NaN();

    EXPECT_FALSE(GpuFpReferenceRmsValidation<float>(1e30f).allClose(ref, impl));
}

// Strided tensors: only logical elements count. The padding between N slices holds
// values that would dominate the ratio if the kernel walked the buffer linearly.
TEST(TestGpuFpReferenceRmsValidationReduction, StridedPaddingIsNotCompared)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {2, 3, 4, 5};
    const std::vector<int64_t> strides = {120, 20, 5, 1};
    Tensor<float> ref(dims, strides);
    Tensor<float> impl(dims, strides);
    ASSERT_FALSE(ref.isPacked());

    fill(ref, 1e30f);
    fill(impl, -1e30f);
    for(int64_t n = 0; n < dims[0]; ++n)
    {
        for(int64_t c = 0; c < dims[1]; ++c)
        {
            for(int64_t h = 0; h < dims[2]; ++h)
            {
                for(int64_t w = 0; w < dims[3]; ++w)
                {
                    ref(n, c, h, w) = 1.0f;
                    impl(n, c, h, w) = 1.0f;
                }
            }
        }
    }

    EXPECT_TRUE(GpuFpReferenceRmsValidation<float>(0.0f).allClose(ref, impl));
}

} // namespace
