// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/**
 * @brief One catalog entry: a kernel that passed every matcher for a graph.
 *
 * This is the POD that leaves KernelIngestorStateManager. It is copied out rather than
 * handed out by reference so a caller holds a stable snapshot while the cache it came
 * from is concurrently evicted or refilled.
 *
 * It carries the kernel's completed metadata tuple (its catalog key), enough source
 * detail to load it, and the ids needed to find the dispatch descriptor that launches
 * it — but no pointer back into the descriptor set.
 */
struct KernelDefinition
{
    DescriptorId kernelId;
    /// The pack this kernel came from, which owns the matchers and UDD that apply to it.
    DescriptorId packId;
    /// The dispatch descriptor id, denormalized from the pack so a caller holding only a
    /// KernelDefinition can reach its dispatch without walking the pack list again.
    DescriptorId dispatchId;
    std::string sourceFile;
    std::string entryPoint;
    /// Complete: every KMD field, with defaults filled in for fields the UKD omitted.
    MetadataValues metadata;
    int64_t priority = 0;

    /// @brief The value of a KMD field, or nullopt when this kernel has no such field.
    ///
    /// Returns nullopt rather than throwing so a matcher or scorer written against a
    /// newer schema can ask about a field an older kernel predates.
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
    /// @throws std::out_of_range if the field is absent, std::invalid_argument if it
    ///         holds a different alternative. Both are author errors a validating
    ///         loader would have caught, so they throw rather than returning a default
    ///         that would silently mis-rank or mis-launch a kernel.
    int64_t getIntMetadata(const std::string& field) const
    {
        auto it = metadata.find(field);
        if(it == metadata.end())
        {
            throw std::out_of_range("kernel '" + kernelId + "' has no metadata field '" + field
                                    + "'");
        }
        const auto* value = std::get_if<int64_t>(&it->second);
        if(value == nullptr)
        {
            throw std::invalid_argument("metadata field '" + field + "' of kernel '" + kernelId
                                        + "' is not an integer");
        }
        return *value;
    }

    /// @brief The string value of a KMD field. Throws on the same terms as getIntMetadata.
    const std::string& getStringMetadata(const std::string& field) const
    {
        auto it = metadata.find(field);
        if(it == metadata.end())
        {
            throw std::out_of_range("kernel '" + kernelId + "' has no metadata field '" + field
                                    + "'");
        }
        const auto* value = std::get_if<std::string>(&it->second);
        if(value == nullptr)
        {
            throw std::invalid_argument("metadata field '" + field + "' of kernel '" + kernelId
                                        + "' is not a string");
        }
        return *value;
    }
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
