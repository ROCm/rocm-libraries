// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <filesystem>
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
    /// Copied from KernelDescriptor::originDirectory; `source.library` is relative to it,
    /// so an adapter needs both to name a file.
    std::filesystem::path originDirectory;
    /// The kernel's authored name, carried so a dispatch-time diagnostic can name the
    /// descriptor the way the loader does.
    std::string name;
    /// Copied from KernelDescriptor::treeRoot: the descriptor tree originDirectory was
    /// found under, which is the boundary `source.library` may not resolve outside of.
    ///
    /// Appended rather than placed beside originDirectory deliberately. The two are the
    /// same type, and `name` between them is a std::string that converts implicitly to
    /// path, so inserting a path field there would silently rebind every positional
    /// initializer after it while still compiling -- the exact hazard the assertion below
    /// exists to catch.
    std::filesystem::path treeRoot;

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

// KernelDefinition is built positionally, ten initializers at a time, so its field count
// is pinned here the way KernelSource's is in Descriptors.hpp. std::string converts
// implicitly to std::filesystem::path, so a path field inserted ahead of originDirectory
// would otherwise compile and silently rebind every initializer after it. Only the count
// -- two same-typed members swapped past each other still brace-initialize.
static_assert(detail::IS_BRACE_INITIALIZABLE_V<KernelDefinition,
                                               DescriptorId,
                                               DescriptorId,
                                               DescriptorId,
                                               KernelSource,
                                               MetadataValues,
                                               int64_t,
                                               std::vector<std::string>,
                                               std::filesystem::path,
                                               std::string,
                                               std::filesystem::path>
                  && !detail::IS_BRACE_INITIALIZABLE_V<KernelDefinition,
                                                       DescriptorId,
                                                       DescriptorId,
                                                       DescriptorId,
                                                       KernelSource,
                                                       MetadataValues,
                                                       int64_t,
                                                       std::vector<std::string>,
                                                       std::filesystem::path,
                                                       std::string,
                                                       std::filesystem::path,
                                                       std::string>,
              "KernelDefinition gained or lost a field; append only, then extend this "
              "assertion and every positional construction of it.");

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
