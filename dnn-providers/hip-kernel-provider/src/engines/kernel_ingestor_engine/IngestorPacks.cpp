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
        // Packaged/kpack since the conv pack moved to lowered rocKE builders: it owns a
        // module cache (ConvNative.cpp's convFwdKpackModuleCache), so the reset sweep must
        // reach it. The entry that previously declared it cacheless would leave this pack
        // out of that sweep -- and TestIngestorPacksModuleCacheOwnership would not catch
        // it, because `false, nullptr` is self-consistent.
        {"hipkernel:ConvFwd", &registerConvFwdSymbols, true, &resetConvFwdModuleCache},
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
