// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvFwdLaunchValues.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionDenseLaunchValues.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseLaunchValues.hpp"

/**
 * @file TestIngestorDispatchAliases.cpp
 * @brief Each pack's dispatch handler answers to the unversioned name its descriptors
 *        carry and to the versioned name a saved plan stores.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using hipdnn_plugin_sdk::ingestor::DispatchRegistry;

void expectOneHandler(std::string_view unversioned, std::string_view versioned)
{
    const auto* byName = DispatchRegistry<Handle>::tryResolve(std::string(unversioned));
    const auto* byAlias = DispatchRegistry<Handle>::tryResolve(std::string(versioned));

    ASSERT_NE(byName, nullptr) << unversioned << " is not registered";
    EXPECT_EQ(byAlias, byName) << versioned << " does not resolve to the handler of "
                               << unversioned;
}

TEST(TestIngestorDispatchAliases, EachVersionedAliasResolvesToTheSameHandler)
{
    registerNativeIngestorSymbols();

    expectOneHandler("hipkernel.pointwise.dispatch", POINTWISE_DISPATCH_SYMBOL_V1);
    expectOneHandler("hipkernel.conv_fwd.dispatch", CONV_FWD_DISPATCH_SYMBOL_V1);
    expectOneHandler("hipkernel.gfx950_attention_dense.dispatch",
                     GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V1);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
