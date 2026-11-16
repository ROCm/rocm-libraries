// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvFwdLaunchValues.hpp"
#ifdef HIPDNN_ENGINE_FLYDSL
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdLaunchValues.hpp"
#endif
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
                     GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V2);
}

// A superseded contract keeps its own handler, so a plan saved under it still restores
// and launches the arguments its code object takes, not the current contract's.
TEST(TestIngestorDispatchAliases, ASupersededContractKeepsItsOwnHandler)
{
    registerNativeIngestorSymbols();

    const auto* current
        = DispatchRegistry<Handle>::tryResolve("hipkernel.gfx950_attention_dense.dispatch");
    const auto* v1 = DispatchRegistry<Handle>::tryResolve(
        std::string(GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V1));

    ASSERT_NE(v1, nullptr) << GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V1 << " is not registered";
    EXPECT_NE(v1, current) << GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V1
                           << " resolves to the current contract's handler";
}

// gfx942 has one contract. Its versioned name resolves to the handler of the unversioned
// one, so a plan saved through the catalog's handler names that same handler, and it has
// no version it never defined.
TEST(TestIngestorDispatchAliases, Gfx942AttentionDenseHasOneContractOnItsCatalogHandler)
{
    registerNativeIngestorSymbols();

    expectOneHandler("hipkernel.gfx942_attention_dense.dispatch",
                     GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V1);
    EXPECT_EQ(DispatchRegistry<Handle>::tryResolve("hipkernel.gfx942_attention_dense.dispatch.v2"),
              nullptr);
}

#ifdef HIPDNN_ENGINE_FLYDSL
struct FlydslSdpaBwdDispatch
{
    std::string_view unversioned;
    std::string_view versioned;
};

const std::array<FlydslSdpaBwdDispatch, 8> FLYDSL_SDPA_BWD_DISPATCHES = {{
    {"hipkernel.flydsl_sdpa_bwd_split.prep.dispatch",
     FLYDSL_SDPA_BWD_SPLIT_PREP_DISPATCH_SYMBOL_V1},
    {"hipkernel.flydsl_sdpa_bwd_split.dkdv.dispatch",
     FLYDSL_SDPA_BWD_SPLIT_DKDV_DISPATCH_SYMBOL_V1},
    {"hipkernel.flydsl_sdpa_bwd_split.dq.dispatch", FLYDSL_SDPA_BWD_SPLIT_DQ_DISPATCH_SYMBOL_V1},
    {"hipkernel.flydsl_sdpa_bwd_split.dq_reduce.dispatch",
     FLYDSL_SDPA_BWD_SPLIT_DQ_REDUCE_DISPATCH_SYMBOL_V1},
    {"hipkernel.flydsl_sdpa_bwd_split.dkdv_u2.dispatch",
     FLYDSL_SDPA_BWD_SPLIT_DKDV_U2_DISPATCH_SYMBOL_V1},
    {"hipkernel.flydsl_sdpa_bwd_split.dq_u2.dispatch",
     FLYDSL_SDPA_BWD_SPLIT_DQ_U2_DISPATCH_SYMBOL_V1},
    {"hipkernel.flydsl_sdpa_bwd_merged.dispatch", FLYDSL_SDPA_BWD_MERGED_DISPATCH_SYMBOL_V1},
    {"hipkernel.flydsl_sdpa_bwd_merged_reduce.dispatch",
     FLYDSL_SDPA_BWD_MERGED_REDUCE_DISPATCH_SYMBOL_V1},
}};

TEST(TestIngestorDispatchAliases,
     EachFlydslSdpaBwdAliasResolvesToTheHandlerOfItsUnversionedNameAndSavesEveryLaunch)
{
    registerNativeIngestorSymbols();

    for(const auto& [unversioned, versioned] : FLYDSL_SDPA_BWD_DISPATCHES)
    {
        expectOneHandler(unversioned, versioned);
        const auto* handler = DispatchRegistry<Handle>::tryResolve(std::string(unversioned));
        ASSERT_NE(handler, nullptr) << unversioned;
        EXPECT_TRUE(handler->supportsSaving()) << unversioned;
        EXPECT_TRUE(handler->supportsSavingAllLaunches()) << unversioned;
    }
}
#endif // HIPDNN_ENGINE_FLYDSL

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
