// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "compilation/KpackModuleCache.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

const std::vector<IngestorPack>& ingestorPacks()
{
    // Function-local static: entries are plain function pointers, so this cannot fail
    // in a way that matters before main().
    static const std::vector<IngestorPack> s_packs = {
        {"hipkernel:Pointwise", &registerPointwiseSymbols, &resetPointwiseModuleCache},
        {"hipkernel:ConvFwd", &registerConvFwdSymbols, &resetConvFwdModuleCache},
#ifdef HIPDNN_ENGINE_FLYDSL
        // Compiled only when FlyDSL staged a shard; see the gate in this directory's
        // CMakeLists.txt.
        {"hipkernel:FlydslRmsNorm", &registerFlydslRmsNormSymbols, &resetFlydslRmsNormModuleCache},
        {"hipkernel:flydsl_rmsnorm_bwd",
         &registerFlydslRmsNormBwdSymbols,
         &resetFlydslRmsNormBwdModuleCache},
        {"hipkernel:FlydslSdpaRdna",
         &registerFlydslSdpaRdnaSymbols,
         &resetFlydslSdpaRdnaModuleCache},
        {"hipkernel:FlydslSdpaRdnaDecode",
         &registerFlydslSdpaRdnaDecodeSymbols,
         &resetFlydslSdpaRdnaDecodeModuleCache},
        {"hipkernel:FlydslSdpaAiw", &registerFlydslSdpaAiwSymbols, &resetFlydslSdpaAiwModuleCache},
        {"hipkernel:FlydslSdpaCdna",
         &registerFlydslSdpaCdnaSymbols,
         &resetFlydslSdpaCdnaModuleCache},
        {"hipkernel:FlydslSdpaCdnaDualwave",
         &registerFlydslSdpaCdnaDualwaveSymbols,
         &resetFlydslSdpaCdnaDualwaveModuleCache},
        {"hipkernel:FlydslSdpaCdnaDecode",
         &registerFlydslSdpaCdnaDecodeSymbols,
         &resetFlydslSdpaCdnaDecodeModuleCache},
#endif
        {GFX950_ATTENTION_DENSE_ENGINE_NAME,
         &registerGfx950AttentionDenseSymbols,
         &resetGfx950AttentionDenseModuleCache},
        {GFX950_ATTENTION_UNIFIED_2D_ENGINE_NAME,
         &registerGfx950AttentionUnified2DSymbols,
         &resetGfx950AttentionUnified2DModuleCache},
        {"hipkernel:Gfx1151WmmaAttention",
         &registerGfx1151WmmaAttentionSymbols,
         &resetGfx1151WmmaAttentionModuleCache},
        {"hipkernel:Gfx1151WmmaAttentionDecode",
         &registerGfx1151WmmaAttentionDecodeSymbols,
         &resetGfx1151WmmaAttentionDecodeModuleCache},
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
    compilation::SharedKpackArchives::resetForTesting();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
