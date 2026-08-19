// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// One catalog entry: a kernel that passed every matcher for a graph. Copied out of
/// KernelIngestorStateManager rather than referenced, so a caller holds a stable
/// snapshot while the source cache is concurrently evicted or refilled.
struct KernelDefinition
{
    DescriptorId kernelId;
    DescriptorId packId;
    DescriptorId dispatchId; ///< Denormalized from the pack for direct lookup.
    KernelSource source;
    MetadataValues metadata;
    int64_t priority = 0;
    /// Devices this kernel runs on, already resolved: its own list when it declared one,
    /// otherwise the pack's. Empty means every device, which is also what an unrestricted
    /// kernel of an unrestricted pack gets. Read at match time, so one pack can hold an
    /// implementation per capability -- an MFMA build beside a portable one.
    std::vector<std::string> arch;

    std::optional<MetadataValue> tryGetMetadata(const std::string& field) const
    {
        auto it = metadata.find(field);
        if(it == metadata.end())
        {
            return std::nullopt;
        }
        return it->second;
    }

    /// @throws std::out_of_range if absent, std::invalid_argument if a different
    ///         alternative is held. getStringMetadata/getIntListMetadata throw on
    ///         the same terms.
    int64_t getIntMetadata(const std::string& field) const
    {
        auto it = metadata.find(field);
        if(it == metadata.end())
        {
            throw std::out_of_range("kernel '" + toString(kernelId) + "' has no metadata field '"
                                    + field + "'");
        }
        const auto* value = std::get_if<int64_t>(&it->second);
        if(value == nullptr)
        {
            throw std::invalid_argument("metadata field '" + field + "' of kernel '"
                                        + toString(kernelId) + "' is not an integer");
        }
        return *value;
    }

    const std::string& getStringMetadata(const std::string& field) const
    {
        auto it = metadata.find(field);
        if(it == metadata.end())
        {
            throw std::out_of_range("kernel '" + toString(kernelId) + "' has no metadata field '"
                                    + field + "'");
        }
        const auto* value = std::get_if<std::string>(&it->second);
        if(value == nullptr)
        {
            throw std::invalid_argument("metadata field '" + field + "' of kernel '"
                                        + toString(kernelId) + "' is not a string");
        }
        return *value;
    }

    const std::vector<int64_t>& getIntListMetadata(const std::string& field) const
    {
        auto it = metadata.find(field);
        if(it == metadata.end())
        {
            throw std::out_of_range("kernel '" + toString(kernelId) + "' has no metadata field '"
                                    + field + "'");
        }
        const auto* value = std::get_if<std::vector<int64_t>>(&it->second);
        if(value == nullptr)
        {
            throw std::invalid_argument("metadata field '" + field + "' of kernel '"
                                        + toString(kernelId) + "' is not an int list");
        }
        return *value;
    }
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
