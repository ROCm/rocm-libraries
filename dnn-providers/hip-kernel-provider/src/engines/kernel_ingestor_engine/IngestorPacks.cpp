// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "engines/kernel_ingestor_engine/packs/PointwiseAddMatchers.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseAddPack.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

const std::vector<IngestorPack>& ingestorPacks()
{
    // Function-local static: no dependence on namespace-scope init order, and the
    // entries are plain function pointers, so building this list cannot fail in a way
    // that matters before main().
    static const std::vector<IngestorPack> s_packs = {
        {"hipkernel:PointwiseAdd", &registerPointwiseAddSymbols, &buildPointwiseAddDescriptorSet},
    };
    return s_packs;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
