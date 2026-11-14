// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "include/rocroller_unbuildable.hpp"

#include "kernel_type.hpp"

#include <algorithm>
#include <iterator>
#include <tuple>
#include <vector>

namespace
{
    auto fields(const RocRollerConfigKey& c)
    {
        return std::tie(c.transA,
                        c.transB,
                        c.typeA,
                        c.typeB,
                        c.typeC,
                        c.typeD,
                        c.tileM,
                        c.tileN,
                        c.tileK,
                        c.workgroupMapping,
                        c.streamK);
    }

    bool before(const RocRollerConfigKey& a, const RocRollerConfigKey& b)
    {
        return fields(a) < fields(b);
    }

    // A plain array of aggregates: thousands of rows stay static data.
    constexpr RocRollerConfigKey recordedConfigs[] = {
#define X(transA, transB, typeA, typeB, typeC, typeD, tileM, tileN, tileK, wgm, streamK) \
    {transA, transB, typeA, typeB, typeC, typeD, tileM, tileN, tileK, wgm, streamK},
#include "unbuildable_configs.def"
#undef X
    };

    const std::vector<RocRollerConfigKey>& unbuildableConfigs()
    {
        static const std::vector<RocRollerConfigKey> configs = [] {
            std::vector<RocRollerConfigKey> keys(std::begin(recordedConfigs),
                                                 std::end(recordedConfigs));
            std::sort(keys.begin(), keys.end(), before);
            return keys;
        }();
        return configs;
    }
} // namespace

bool rocRollerConfigUnbuildable(const RocRollerConfigKey& config)
{
    RocRollerConfigKey key = config;
    key.workgroupMapping   = config.workgroupMapping && !config.streamK;
    const auto& configs    = unbuildableConfigs();
    return std::binary_search(configs.begin(), configs.end(), key, before);
}

bool rocRollerTileUnbuildable(const KernelType& kernelType, const RocRollerSelectedTile& tile)
{
    using Format = RocblasltContractionProblem::ScalingFormat;
    const bool recordedScales
        = kernelType.scaleTypeA.format == Format::Block_32_UE8M0
          && kernelType.scaleTypeB.format == Format::Block_32_UE8M0
          && kernelType.scaleTypeA.preSwizzleTile.empty()
          && kernelType.scaleTypeB.preSwizzleTile.empty() && kernelType.scaleTypeA.preTile.empty()
          && kernelType.scaleTypeB.preTile.empty() && !kernelType.swizzleA && !kernelType.swizzleB;
    return recordedScales
           && rocRollerConfigUnbuildable({kernelType.transA,
                                          kernelType.transB,
                                          kernelType.typeA,
                                          kernelType.typeB,
                                          kernelType.typeC,
                                          kernelType.typeD,
                                          tile.m,
                                          tile.n,
                                          tile.k,
                                          tile.workgroupMapping,
                                          tile.streamK});
}
