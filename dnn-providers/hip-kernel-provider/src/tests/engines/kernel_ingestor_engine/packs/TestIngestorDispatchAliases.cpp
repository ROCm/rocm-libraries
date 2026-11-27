// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <string>
#include <string_view>
#include <utility>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvFwdLaunchValues.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionDenseLaunchValues.hpp"
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
                     GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_STRIDED);
    expectOneHandler("hipkernel.gfx942_attention_dense.dispatch",
                     GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_STRIDED);
}

// A superseded contract keeps its own handler, so a plan saved under it still restores
// and launches the arguments its code object takes, not the current contract's: on gfx950
// v1 (eight arguments), v2 (ten) and v3 (fourteen, with a bias), on gfx942 v1 (no strides)
// and v2 (with a bias, no strides).
TEST(TestIngestorDispatchAliases, ASupersededContractKeepsItsOwnHandler)
{
    registerNativeIngestorSymbols();

    for(const auto& [unversioned, superseded] :
        {std::pair<std::string_view, std::string_view>{"hipkernel.gfx950_attention_dense.dispatch",
                                                       GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V1},
         std::pair<std::string_view, std::string_view>{"hipkernel.gfx950_attention_dense.dispatch",
                                                       GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V2},
         std::pair<std::string_view, std::string_view>{"hipkernel.gfx950_attention_dense.dispatch",
                                                       GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V3},
         std::pair<std::string_view, std::string_view>{"hipkernel.gfx942_attention_dense.dispatch",
                                                       GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V1},
         std::pair<std::string_view, std::string_view>{"hipkernel.gfx942_attention_dense.dispatch",
                                                       GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V2}})
    {
        const auto* current = DispatchRegistry<Handle>::tryResolve(std::string(unversioned));
        const auto* legacy = DispatchRegistry<Handle>::tryResolve(std::string(superseded));

        ASSERT_NE(legacy, nullptr) << superseded << " is not registered";
        EXPECT_NE(legacy, current) << superseded << " resolves to the current contract's handler";
    }

    EXPECT_NE(DispatchRegistry<Handle>::tryResolve(
                  std::string(GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V1)),
              DispatchRegistry<Handle>::tryResolve(
                  std::string(GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V2)))
        << "gfx950's v1 and v2 launch different argument lists, so they need their own handlers";
    EXPECT_NE(DispatchRegistry<Handle>::tryResolve(
                  std::string(GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V2)),
              DispatchRegistry<Handle>::tryResolve(
                  std::string(GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V3)))
        << "gfx950's v2 and v3 launch different argument lists, so they need their own handlers";
    EXPECT_NE(DispatchRegistry<Handle>::tryResolve(
                  std::string(GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V1)),
              DispatchRegistry<Handle>::tryResolve(
                  std::string(GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V2)))
        << "gfx942's v1 and v2 launch different argument lists, so they need their own handlers";
}

// The catalog's handler saves a bias plan under the strided bias contract, and a plan may be
// saved only under a name registered to the handler that saves it, so that name resolves to
// the current handler.
TEST(TestIngestorDispatchAliases, TheBiasContractIsServedByTheCurrentHandler)
{
    registerNativeIngestorSymbols();

    expectOneHandler("hipkernel.gfx950_attention_dense.dispatch",
                     GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_STRIDED_BIAS);
    expectOneHandler("hipkernel.gfx942_attention_dense.dispatch",
                     GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_STRIDED_BIAS);
}

// No dense engine answers to a version it never defined: the next number past each
// engine's strided bias contract.
TEST(TestIngestorDispatchAliases, AttentionDenseHasNoVersionItNeverDefined)
{
    registerNativeIngestorSymbols();

    EXPECT_EQ(DispatchRegistry<Handle>::tryResolve("hipkernel.gfx950_attention_dense.dispatch.v6"),
              nullptr);
    EXPECT_EQ(DispatchRegistry<Handle>::tryResolve("hipkernel.gfx942_attention_dense.dispatch.v5"),
              nullptr);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
