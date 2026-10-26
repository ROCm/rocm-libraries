// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file FlydslNormDeviceTestSupport.hpp
 * @brief What the FlyDSL RMS-norm forward and backward device suites share: the device as
 *        the packer names it, the staged objects for it, the comparison every launch is
 *        judged by, and the rule for a case whose objects are not there.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// The current device, with the arch stripped of its feature suffix.
///
/// hipGetDeviceProperties reports `gfx1151:xnack-`; the packer names shards with the bare
/// arch, and prepare() passes this string straight through to the archive lookup. Asking
/// in the wrong spelling fails to find a shard that is sitting right there.
inline hipdnn_plugin_sdk::ingestor::DeviceProperties packedArchDeviceProperties()
{
    auto properties = currentDeviceProperties();
    const auto colon = properties.gcnArchName.find(':');
    if(colon != std::string::npos)
    {
        properties.gcnArchName.resize(colon);
    }
    return properties;
}

/// Every kernel @p engineName staged for this device, promoted from the loaded descriptors.
///
/// Promoted rather than hand-written because a kpack source is only usable with the
/// packer's own `library`, `toc_key`, `symbol`, `sha256` and the two loader-filled path
/// fields. A kernel may narrow the arch list its pack claims but never reach outside it,
/// so an empty list on the kernel means "the pack's". A build for several arches loads
/// every arch's shard, so a kernel staged for another device is passed over.
inline std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition>
    stagedKernels(std::string_view engineName)
{
    std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition> kernels;
    const auto* staged = findLoadedSet(engineName);
    if(staged == nullptr)
    {
        return kernels;
    }
    const auto device = packedArchDeviceProperties().gcnArchName;
    for(const auto& pack : staged->packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto& arches = kernel.arch.empty() ? pack.arch : kernel.arch;
            if(!device.empty() && !arches.empty()
               && std::find(arches.begin(), arches.end(), device) == arches.end())
            {
                continue;
            }
            kernels.push_back(hipdnn_plugin_sdk::ingestor::KernelDefinition{kernel.id,
                                                                            pack.id,
                                                                            pack.dispatchId,
                                                                            kernel.source,
                                                                            kernel.metadata,
                                                                            kernel.priority,
                                                                            arches,
                                                                            kernel.originDirectory,
                                                                            kernel.name,
                                                                            kernel.treeRoot,
                                                                            kernel.sources});
        }
    }
    return kernels;
}

/// The worst element of a comparison, and where it is.
struct WorstError
{
    double error = 0.0;
    size_t index = 0;
};

/// Largest error of @p actual against @p expected, relative to max(|expected|, floor).
///
/// A non-finite element is an infinite error. A plain `error > worst` never records a NaN,
/// so a kernel that wrote NaN everywhere would compare as exact; NaN is how a norm kernel
/// usually fails (a bad epsilon, a fast-math poison), and the output buffers start as NaN
/// so an element no kernel wrote fails the same way.
template <typename V>
WorstError worstRelativeError(const std::vector<V>& actual,
                              const std::vector<double>& expected,
                              double floor)
{
    WorstError worst;
    const auto count = std::min(actual.size(), expected.size());
    for(size_t i = 0; i < count; ++i)
    {
        const auto got = static_cast<double>(static_cast<float>(actual[i]));
        const double error = std::isfinite(got) ? std::abs(got - expected[i])
                                                      / std::max(std::abs(expected[i]), floor)
                                                : std::numeric_limits<double>::infinity();
        if(error > worst.error)
        {
            worst = WorstError{error, i};
        }
    }
    if(actual.size() != expected.size())
    {
        worst.error = std::numeric_limits<double>::infinity();
    }
    return worst;
}

/// The architectures FlyDSL has @p op's objects for, as CMake passed them
/// (HIPKERNELPROVIDER_FLYDSL_ARCHS_<op>, comma-separated): the content list, the same in
/// every build whatever its GPU_TARGETS. Whether this install staged them for the device
/// is a run-time question (flydslShipsForThisDevice()), because a multi-arch build
/// (TheRock RFC 0008) compiles this binary in one architecture's shard and runs it on all.
inline std::vector<std::string> flydslShippedArches(std::string_view op)
{
    // A table, not an if/else chain: the two ops usually ship the same arches, and
    // identical branch bodies trip bugprone-branch-clone.
    constexpr std::array<std::pair<std::string_view, std::string_view>, 12> SHIPPED{{
        {"rmsnorm", HIPKERNELPROVIDER_FLYDSL_ARCHS_RMSNORM},
        {"rmsnorm_bwd", HIPKERNELPROVIDER_FLYDSL_ARCHS_RMSNORM_BWD},
        {"sdpa_cdna", HIPKERNELPROVIDER_FLYDSL_ARCHS_SDPA_CDNA},
        {"sdpa_cdna_dualwave", HIPKERNELPROVIDER_FLYDSL_ARCHS_SDPA_CDNA_DUALWAVE},
        {"sdpa_cdna_decode", HIPKERNELPROVIDER_FLYDSL_ARCHS_SDPA_CDNA_DECODE},
        {"sdpa_rdna", HIPKERNELPROVIDER_FLYDSL_ARCHS_SDPA_RDNA},
        {"sdpa_rdna_decode", HIPKERNELPROVIDER_FLYDSL_ARCHS_SDPA_RDNA_DECODE},
        {"sdpa_aiw", HIPKERNELPROVIDER_FLYDSL_ARCHS_SDPA_AIW},
        {"sdpa_bwd_merged", HIPKERNELPROVIDER_FLYDSL_ARCHS_SDPA_BWD_MERGED},
        {"sdpa_bwd_prep", HIPKERNELPROVIDER_FLYDSL_ARCHS_SDPA_BWD_PREP},
        {"sdpa_bwd_dkdv", HIPKERNELPROVIDER_FLYDSL_ARCHS_SDPA_BWD_DKDV},
        {"sdpa_bwd_dq", HIPKERNELPROVIDER_FLYDSL_ARCHS_SDPA_BWD_DQ},
    }};
    std::string list;
    bool known = false;
    for(const auto& [name, archList] : SHIPPED)
    {
        if(name == op)
        {
            list = archList;
            known = true;
        }
    }
    // An op name the table lacks would read as "ships nowhere" and turn every missing
    // object into a skip, which is the gap this table exists to close.
    if(!known)
    {
        ADD_FAILURE() << "no shipped-arch list for FlyDSL op " << op;
    }
    std::vector<std::string> arches;
    std::stringstream stream(list);
    for(std::string arch; std::getline(stream, arch, ',');)
    {
        if(!arch.empty())
        {
            arches.push_back(arch);
        }
    }
    return arches;
}

/// Whether this install staged a product shard for @p arch: a directory named for it
/// under the additive descriptor root main() points at the product tree. A root that holds
/// no architecture's shard at all is a mis-pointed or empty root rather than a build that
/// left this one out, so it counts as staged and the cases that need objects fail.
inline bool flydslShardStagedFor(const std::string& arch)
{
    const std::filesystem::path root
        = hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR");
    std::error_code failed;
    if(root.empty() || !std::filesystem::is_directory(root, failed))
    {
        return true;
    }
    if(std::filesystem::is_directory(root / arch, failed))
    {
        return true;
    }
    for(const auto& entry : std::filesystem::directory_iterator(root, failed))
    {
        if(entry.is_directory() && entry.path().filename().string().rfind("gfx", 0) == 0)
        {
            return false;
        }
    }
    return true;
}

/// Whether this install ships @p op's objects for the current device: FlyDSL has them for
/// its arch and the install staged the arch's shard. A case that then finds none has found
/// a packaging fault rather than an arch FlyDSL, or this install, does not serve.
inline bool flydslShipsForThisDevice(std::string_view op)
{
    const auto arches = flydslShippedArches(op);
    const auto device = packedArchDeviceProperties().gcnArchName;
    return std::find(arches.begin(), arches.end(), device) != arches.end()
           && flydslShardStagedFor(device);
}

/// Whether this install ships @p op's objects for any architecture it staged.
inline bool flydslShipsAnywhere(std::string_view op)
{
    const auto arches = flydslShippedArches(op);
    return std::any_of(arches.begin(), arches.end(), [](const std::string& arch) {
        return flydslShardStagedFor(arch);
    });
}

/// Records that a case did not find the objects it needs: a failure on a device this build
/// ships @p op for, where the shard has to hold them, and a skip elsewhere. The caller
/// returns right after. Skipping on a shipped arch is how a mis-pointed descriptor root or
/// a dropped object went unnoticed: the suite stayed green with every device case skipped.
inline void reportUnstaged(std::string_view op, const std::string& what)
{
    if(flydslShipsForThisDevice(op))
    {
        ADD_FAILURE() << "this build ships " << op << " for this device, but " << what;
        return;
    }
    GTEST_SKIP() << what;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
