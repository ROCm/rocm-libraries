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
        {"hipkernel:ConvFwd", &registerConvFwdSymbols, &resetConvFwdModuleCache},
    };
    return s_packs;
}

void resetIngestorModuleCachesForTesting()
{
    // Walks the registration table rather than a list of its own, so the sweep and the
    // inventory cannot drift apart as two lists. A pack that acquires a kpack cache but
    // leaves resetModuleCache null is still skipped, silently.
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
