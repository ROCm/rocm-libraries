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

/**
 * @brief One catalog entry: a kernel that passed every matcher for a graph.
 *
 * Copied out of KernelIngestorStateManager rather than referenced, so a caller holds a
 * stable snapshot while the source cache is concurrently evicted or refilled.
 *
 * Carries the kernel's completed metadata tuple (its catalog key), enough source
 * detail to load it, and the ids needed to find its dispatch descriptor.
 */
struct KernelDefinition
{
    DescriptorId kernelId;
    /// The pack this kernel came from; owns the matchers and UDD that apply to it.
    DescriptorId packId;
    /// Dispatch descriptor id, denormalized from the pack so a caller holding only a
    /// KernelDefinition can reach it without walking the pack list.
    DescriptorId dispatchId;
    /// Where this kernel's code comes from and how to load it.
    KernelSource source;
    /// Every KMD field, with defaults filled in for fields the UKD omitted.
    MetadataValues metadata;
    int64_t priority = 0;

    /// @brief The value of a KMD field, or nullopt if the kernel has no such field.
    std::optional<MetadataValue> tryGetMetadata(const std::string& field) const
    {
        auto it = metadata.find(field);
        if(it == metadata.end())
        {
            return std::nullopt;
        }
        return it->second;
    }

    /// @brief The integer value of a KMD field.
    /// @throws std::out_of_range if absent, std::invalid_argument if a different
    ///         alternative is held.
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

    /// @brief The string value of a KMD field. Throws on the same terms as getIntMetadata.
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

    /// @brief The int-list value of a KMD field (e.g. `stride_order`).
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
