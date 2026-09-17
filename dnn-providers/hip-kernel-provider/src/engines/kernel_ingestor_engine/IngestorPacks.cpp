// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

namespace hip_kernel_provider::kernel_ingestor_engine
{

const std::vector<IngestorPack>& ingestorPacks()
{
    // Function-local static: entries are plain function pointers, so this cannot fail
    // in a way that matters before main().
    static const std::vector<IngestorPack> s_packs = {
        {"hipkernel:Pointwise", &registerPointwiseSymbols, true, &resetPointwiseModuleCache},
        // No kpack archive: its kernels are embedded_source, so there is no module to
        // drop and nothing for a reset to do.
        {"hipkernel:ConvFwd", &registerConvFwdSymbols, false, nullptr},
        // THROWAWAY POC. Its dispatch raw-loads a build-time flyDSL->HSACO code object
        // (hipModuleLoadData), never touching buildIngestorKernelCode/kpack, so there is
        // no module cache to reset -- same {false, nullptr} shape as ConvFwd above.
        {"hipkernel:Flydsl", &registerFlydslSymbols, false, nullptr},
        // THROWAWAY POC. Family variant of the above: one flyDSL->HSACO per hidden size N,
        // raw-loaded (hipModuleLoadData) by the dispatch from the matched instance's baked
        // N. No kpack, no module cache -- same {false, nullptr} shape as Flydsl above.
        {"hipkernel:FlydslRmsNorm", &registerFlydslRmsNormSymbols, false, nullptr},
        // THROWAWAY POC (M3). Family of flyDSL flash-attention HSACOs (one per
        // num_heads/head_dim/causal/dtype tuple), raw-loaded (hipModuleLoadData) by the
        // dispatch from the matched instance. Its own graph_match symbol => its own
        // StateManager, so a nullopt decline empties only ITS catalog, never the
        // dense/tiled/asm SDPA engines it competes with. No kpack, no module cache.
        {"hipkernel:FlydslAttention", &registerFlydslAttentionSymbols, false, nullptr},
        // Packaged/kpack: its kernels are lowered rocKE builders resolved out of the
        // per-arch .kpack archive, so it owns a module cache the reset sweep must
        // reach. TestIngestorPacksModuleCacheOwnership asserts `ownsModuleCache` and
        // `resetModuleCache` agree for every entry in this table, so a pack that sets
        // `ownsModuleCache = true` here without wiring the reset pointer -- the shape
        // the generator's fragment used to emit, `nullptr` in this slot -- now fails
        // that test by name instead of leaving this pack silently out of
        // resetIngestorModuleCachesForTesting().
        {"hipkernel:Gfx942AttentionDense",
         &registerGfx942AttentionDenseSymbols,
         true,
         &resetGfx942AttentionDenseModuleCache},
        // Packaged/kpack, same shape as the gfx942 entry above and for the same
        // reason: its module cache must be reachable by the reset sweep. The two
        // packs never compete -- each declares a single, different `arch`, and packs
        // arch-prune before the matcher runs -- so both are registered
        // unconditionally and at most one can ever match on a given device.
        {"hipkernel:Gfx950AttentionDense",
         &registerGfx950AttentionDenseSymbols,
         true,
         &resetGfx950AttentionDenseModuleCache},
        // Packaged/kpack. THE FIRST PAIR OF PACKS THAT SHARE AN ARCH: this and
        // Gfx950AttentionDense both declare gfx950 alone, so unlike the gfx942/gfx950
        // dense pair above they are NOT separated by arch-pruning and both reach the
        // matcher on the same device.
        //
        // They still cannot both serve one graph, and the separation is structural
        // rather than a priority tie-break: this engine REQUIRES page tables and
        // per-sequence lengths (its 18-slot ABI declares block_tables_ptr and
        // seq_lens_ptr unconditionally and it has no dense mode), while the dense
        // engine DECLINES any graph carrying either. The two graph_match bodies are
        // exact complements on that one axis, so every SDPA graph is claimed by at
        // most one of them.
        {"hipkernel:Gfx950AttentionTiled",
         &registerGfx950AttentionTiledSymbols,
         true,
         &resetGfx950AttentionTiledModuleCache},
    };
    return s_packs;
}

void resetIngestorModuleCachesForTesting()
{
    // Driven off the same table as registration, so a pack that gains a kpack cache
    // cannot be left out of the reset by someone who only edited its own file.
    for(const auto& pack : ingestorPacks())
    {
        if(pack.resetModuleCache != nullptr)
        {
            pack.resetModuleCache();
        }
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
