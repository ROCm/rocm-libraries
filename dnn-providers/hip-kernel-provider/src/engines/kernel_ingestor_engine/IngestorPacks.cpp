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
        {"hipkernel:Pointwise", &registerPointwiseSymbols, &resetPointwiseModuleCache},
        // No kpack archive: its kernels are embedded_source, so there is no module to
        // drop and nothing for a reset to do.
        {"hipkernel:ConvFwd", &registerConvFwdSymbols, nullptr},
        // Packaged/kpack: its kernels are lowered rocKE builders resolved out of the
        // per-arch .kpack archive, so it owns a module cache the reset sweep must
        // reach. The generator's fragment emitted `nullptr` in this slot; that would
        // have left this pack out of resetIngestorModuleCachesForTesting() while its
        // own fragment header states the rule that says otherwise.
        {"hipkernel:Gfx942AttentionDense",
         &registerGfx942AttentionDenseSymbols,
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
