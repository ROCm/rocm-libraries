// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "engines/kernel_ingestor_engine/descriptors/hip/standalone_pointwise/kernels/StandalonePointwiseContract.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include <gtest/gtest.h>

namespace
{
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

TEST(TestStandalonePointwisePacks, ShipsOneShapeDynamicKernelPerArchitecture)
{
    const auto& set = loadedSet("hipkernel:StandalonePointwise");
    ASSERT_EQ(distinctPackIdCount(set), 1U);
    for(const auto& pack : set.packs)
    {
        ASSERT_EQ(pack.kernels.size(), 1U);
        const auto& kernel = pack.kernels.front();
        EXPECT_EQ(kernel.source.kind, hipdnn_plugin_sdk::ingestor::KernelSourceKind::KPACK);
        EXPECT_EQ(kernel.source.symbol, "StandalonePointwise");
        EXPECT_EQ(std::get<int64_t>(kernel.metadata.at("contract_version")), 1);
    }
}

TEST(TestStandalonePointwisePacks, EveryKernelDeclaresTheDynamicLaunchSignature)
{
    const auto& set = loadedSet("hipkernel:StandalonePointwise");
    for(const auto& pack : set.packs)
    {
        ASSERT_FALSE(pack.kernels.empty());
        for(const auto& kernel : pack.kernels)
        {
            const auto& signature = kernel.source.signature;
            ASSERT_EQ(signature.size(), 4U);
            for(size_t operand = 0; operand < 3; ++operand)
            {
                EXPECT_EQ(signature[operand].kind, "global_buffer");
                EXPECT_EQ(signature[operand].size, sizeof(void*));
                EXPECT_EQ(signature[operand].offset, operand * sizeof(void*));
            }
            EXPECT_EQ(signature[3].kind, "by_value");
            EXPECT_EQ(signature[3].size, sizeof(hkp_standalone_pointwise::Metadata));
            EXPECT_EQ(signature[3].offset, 3 * sizeof(void*));
        }
    }
}
} // namespace
#endif
