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
        {"hipkernel:Pointwise", &registerPointwiseSymbols},
        {"hipkernel:ConvFwd", &registerConvFwdSymbols},
#ifdef HIPDNN_ENGINE_ASM_SDPA
        {"hipkernel:AsmSdpaForward", &registerAsmSdpaForwardSymbols},
        {"hipkernel:AsmSdpaBackward", &registerAsmSdpaBackwardSymbols},
#endif
#ifdef HIPDNN_ENGINE_HIP_MLOPS
        {"hipkernel:LayernormForward", &registerLayernormForwardSymbols},
        {"hipkernel:Resample", &registerResampleSymbols},
        {"hipkernel:RMSnorm", &registerRMSnormSymbols},
        {"hipkernel:Batchnorm", &registerBatchnormSymbols},
#endif
    };
    return s_packs;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
