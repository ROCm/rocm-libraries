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
        // Owns a kpack module cache because its dispatch handler goes through
        // buildIngestorKernelCode, which takes a loader for the kinds it also serves.
        // Its own descriptors are all embedded_source, so today the cache stays empty.
        {"hipkernel:BatchnormInference",
         &registerBatchnormInferenceSymbols,
         true,
         &resetBatchnormInferenceModuleCache},
        // Same case as Pointwise and BatchnormInference: its dispatch handler routes
        // through buildIngestorKernelCode, so it holds a KpackKernelLoader and owns a
        // module cache. Its own descriptors are all embedded_source, so today the cache
        // stays empty -- ownership follows the handler, not the dialect.
        {"Hackweek:ConvBias", &registerConvBiasSymbols, true, &resetConvBiasModuleCache},
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
