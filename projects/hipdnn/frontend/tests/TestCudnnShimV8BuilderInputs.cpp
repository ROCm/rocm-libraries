// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Optional values the ported v0.x builders send, against the mock hipDNN backend:
// an RNG seed as a value or a tensor, and a resample window left unset.
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Operation.h>

#include <gtest/gtest.h>

#include "fake_backend/ShimV8BuilderPlanFixture.hpp"

#include <cstdint>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using hipdnn_shim_test::BuilderFailure;
using hipdnn_shim_test::builderFailureOf;

class TestCudnnShimV8BuilderInputs : public hipdnn_shim_test::ShimV8BuilderPlanFixture
{
protected:
    static fe::Tensor_v8 seedTensor(std::int64_t uid)
    {
        const Dims dims = {1, 1, 1, 1};
        const Dims strides = packedStrides(dims);
        return fe::TensorBuilder_v8()
            .setDataType(fe::DataType_t::INT64)
            .setDim(static_cast<std::int64_t>(dims.size()), dims.data())
            .setStride(static_cast<std::int64_t>(strides.size()), strides.data())
            .setId(uid)
            .setAlignment(16)
            .setByValue()
            .build();
    }

    static fe::RngDesc_v8 bernoulli()
    {
        return fe::RngDescBuilder_v8()
            .setRngDistribution(fe::RngDistribution_t::BERNOULLI)
            .setBernoulliDistProbability(0.5)
            .build();
    }
};

// Either seed form is accepted; the RNG op itself is what is not supported.
TEST_F(TestCudnnShimV8BuilderInputs, RngOpNotSupportedWithEitherSeed)
{
    const Dims dims = {2, 4, 8, 8};
    const BuilderFailure byValue = builderFailureOf([&dims] {
        [[maybe_unused]] auto op
            = fe::OperationBuilder_v8(fe::DescriptorType_t::OPERATION_RNG_DESCRIPTOR)
                  .setyDesc(tensor(Y_UID, dims))
                  .setRngDesc(bernoulli())
                  .setSeed(42)
                  .build();
    });
    ASSERT_TRUE(byValue.thrown);
    EXPECT_EQ(byValue.status, CUDNN_STATUS_NOT_SUPPORTED);

    const BuilderFailure byTensor = builderFailureOf([&dims] {
        [[maybe_unused]] auto op
            = fe::OperationBuilder_v8(fe::DescriptorType_t::OPERATION_RNG_DESCRIPTOR)
                  .setyDesc(tensor(Y_UID, dims))
                  .setRngDesc(bernoulli())
                  .setSeedDesc(seedTensor(10))
                  .setOffsetDesc(seedTensor(11))
                  .build();
    });
    ASSERT_TRUE(byTensor.thrown);
    EXPECT_EQ(byTensor.status, CUDNN_STATUS_NOT_SUPPORTED);
}

// Upstream sends the window with count 0 when no spatial dims were set; the
// descriptor then has no spatial dims.
TEST_F(TestCudnnShimV8BuilderInputs, ResampleWithoutWindowRejected)
{
    fe::ResampleDescBuilder_v8 builder;
    builder.setResampleMode(CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING)
        .setComputeType(fe::DataType_t::FLOAT)
        .setPaddingMode(CUDNN_ZERO_PAD);
    const BuilderFailure failure
        = builderFailureOf([&builder] { [[maybe_unused]] auto built = builder.build(); });
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
}

} // namespace
