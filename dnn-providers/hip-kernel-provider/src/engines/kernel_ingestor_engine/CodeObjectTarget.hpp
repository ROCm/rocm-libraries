// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <string_view>

#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

namespace hip_kernel_provider::kernel_ingestor_engine
{

/// True when a device that reports @p deviceArch can run a code object of source kind
/// @p kind that was built for @p target.
///
/// A kpack archive entry serves each device whose arch name starts with the entry name,
/// so a kpack target follows the rule that selects the entry. Every other kind needs
/// the exact target, feature flags included.
inline bool isCodeObjectTargetCompatible(hipdnn_plugin_sdk::ingestor::KernelSourceKind kind,
                                         std::string_view target,
                                         std::string_view deviceArch)
{
    if(kind == hipdnn_plugin_sdk::ingestor::KernelSourceKind::KPACK)
    {
        return hipdnn_plugin_sdk::archMatches(
            deviceArch, target, hipdnn_plugin_sdk::ArchMatchMode::PREFIX);
    }
    return target == deviceArch;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
