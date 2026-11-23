// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

/// LLVM generic GPU targets (`gfx11-generic`, ...) and how an `arch` entry ranks for a device.
/// Membership comes only from the generated table, never from a name's shape.

#include <cstddef>
#include <optional>
#include <string_view>

#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/GpuGenericTargetsTable.hpp>

namespace hipdnn_plugin_sdk
{

/// Shape only (`...-generic`); use findGenericTarget for membership.
inline bool isGenericShapedArchName(std::string_view name)
{
    constexpr std::string_view SUFFIX = "-generic";
    return name.size() >= SUFFIX.size()
           && name.compare(name.size() - SUFFIX.size(), std::string_view::npos, SUFFIX) == 0;
}

/// The table row for @p name, or nullptr.
inline const generated::GenericTargetRow* findGenericTarget(std::string_view name)
{
    for(const auto& row : generated::GENERIC_TARGET_ROWS)
    {
        if(row.name == name)
        {
            return &row;
        }
    }
    return nullptr;
}

/// Is @p baseDeviceId (features stripped) a member of @p generic? False if @p generic is unknown.
inline bool genericTargetContains(std::string_view generic, std::string_view baseDeviceId)
{
    const auto* row = findGenericTarget(generic);
    if(row == nullptr)
    {
        return false;
    }
    for(std::size_t i = 0; i < row->memberCount; ++i)
    {
        if(row->members[i] == baseDeviceId)
        {
            return true;
        }
    }
    return false;
}

/// How specifically an `arch` entry names a device; a lower value wins.
enum class ArchTier : int
{
    EXPLICIT = 0, ///< The entry is the device's own base id.
    GENERIC = 1, ///< A table generic whose members contain the device.
    UNRESTRICTED = 2, ///< An empty list: any device.
};

/// Rank of @p entry for @p rawDeviceArch (features included); nullopt when it does not
/// admit the device. A generic-shaped entry is never EXPLICIT.
inline std::optional<ArchTier> entryTier(std::string_view entry, std::string_view rawDeviceArch)
{
    if(isGenericShapedArchName(entry))
    {
        return genericTargetContains(entry, stripArchFeatures(rawDeviceArch))
                   ? std::optional<ArchTier>(ArchTier::GENERIC)
                   : std::nullopt;
    }
    return archMatches(rawDeviceArch, entry, ArchMatchMode::PREFIX)
               ? std::optional<ArchTier>(ArchTier::EXPLICIT)
               : std::nullopt;
}

} // namespace hipdnn_plugin_sdk
