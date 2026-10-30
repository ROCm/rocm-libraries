// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Optional values the ported v0.x builders send, against the mock hipDNN backend: an RNG seed and
// offset as tensors.
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Operation.h>

#include <gtest/gtest.h>

#include "fake_backend/ShimV8BuilderPlanFixture.hpp"

#include <array>
#include <cstdint>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using hipdnn_shim_test::BuildFailure;
using hipdnn_shim_test::Dims;
using hipdnn_shim_test::failureOf;

class TestCudnnShimV8BuilderInputs : public hipdnn_shim_test::ShimV8BuilderPlanFixture
{
protected:
    static fe::Tensor_v8 seedTensor(std::int64_t uid)
    {
        return scalar(uid, fe::DataType_t::INT64);
    }

    static fe::RngDesc_v8 bernoulli()
    {
        return fe::RngDescBuilder_v8()
            .setRngDistribution(fe::RngDistribution_t::BERNOULLI)
            .setBernoulliDistProbability(0.5)
            .build();
    }
};

// Built seed and offset tensors are accepted and the RNG op itself is what is not supported; an
// unbuilt one is checked first.
TEST_F(TestCudnnShimV8BuilderInputs, RngSeedTensorsCheckedBeforeNotSupported)
{
    const Dims dims = {2, 4, 8, 8};
    fe::Tensor_v8 unbuilt = seedTensor(12);
    unbuilt.initialize_managed_backend_pointer(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    struct Case
    {
        const char* label;
        const fe::Tensor_v8* seed;
        const fe::Tensor_v8* offset;
        cudnnStatus_t status;
    };
    const fe::Tensor_v8 seed = seedTensor(10);
    const fe::Tensor_v8 offset = seedTensor(11);
    const std::array<Case, 3> cases = {{
        {"built", &seed, &offset, CUDNN_STATUS_NOT_SUPPORTED},
        {"unbuilt seed", &unbuilt, &offset, CUDNN_STATUS_BAD_PARAM},
        {"unbuilt offset", &seed, &unbuilt, CUDNN_STATUS_BAD_PARAM},
    }};
    for(const Case& c : cases)
    {
        SCOPED_TRACE(c.label);
        const BuildFailure failure = failureOf([&dims, &c] {
            [[maybe_unused]] auto op
                = fe::OperationBuilder_v8(fe::DescriptorType_t::OPERATION_RNG_DESCRIPTOR)
                      .setyDesc(tensor(Y_UID, dims))
                      .setRngDesc(bernoulli())
                      .setSeedDesc(*c.seed)
                      .setOffsetDesc(*c.offset)
                      .build();
        });
        ASSERT_TRUE(failure.thrown);
        EXPECT_EQ(failure.status, c.status);
    }
}

} // namespace
