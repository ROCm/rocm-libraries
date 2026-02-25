// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "BackendEnumStringUtils.hpp"
#include "hipdnn_backend.h"
#include <spdlog/fmt/fmt.h>

template <>
struct fmt::formatter<hipdnnStatus_t> : fmt::formatter<std::string_view>
{
    auto format(hipdnnStatus_t status, fmt::format_context& ctx) const -> decltype(ctx.out())
    {
        return fmt::formatter<std::string_view>::format(
            hipdnn_backend::hipdnnGetStatusString(status), ctx);
    }
};

template <>
struct fmt::formatter<hipdnnBackendAttributeType_t> : fmt::formatter<std::string_view>
{
    auto format(hipdnnBackendAttributeType_t type,
                fmt::format_context& ctx) const -> decltype(ctx.out())
    {
        return fmt::formatter<std::string_view>::format(
            hipdnn_backend::hipdnnGetAttributeTypeString(type), ctx);
    }
};

template <>
struct fmt::formatter<hipdnnBackendDescriptorType_t> : fmt::formatter<std::string_view>
{
    auto format(hipdnnBackendDescriptorType_t type,
                fmt::format_context& ctx) const -> decltype(ctx.out())
    {
        return fmt::formatter<std::string_view>::format(
            hipdnn_backend::hipdnnGetBackendDescriptorTypeName(type), ctx);
    }
};

template <>
struct fmt::formatter<hipdnnBackendAttributeName_t> : fmt::formatter<std::string_view>
{
    auto format(hipdnnBackendAttributeName_t attr,
                fmt::format_context& ctx) const -> decltype(ctx.out())
    {
        return fmt::formatter<std::string_view>::format(
            hipdnn_backend::hipdnnGetAttributeNameString(attr), ctx);
    }
};

template <>
struct fmt::formatter<hipdnnPluginLoadingMode_ext_t> : fmt::formatter<std::string_view>
{
    auto format(hipdnnPluginLoadingMode_ext_t mode,
                fmt::format_context& ctx) const -> decltype(ctx.out())
    {
        return fmt::formatter<std::string_view>::format(
            hipdnn_backend::hipdnnGetPluginLoadingModeString(mode), ctx);
    }
};

template <>
struct fmt::formatter<hipdnnPluginUnloadingMode_ext_t> : fmt::formatter<std::string_view>
{
    auto format(hipdnnPluginUnloadingMode_ext_t mode,
                fmt::format_context& ctx) const -> decltype(ctx.out())
    {
        return fmt::formatter<std::string_view>::format(
            hipdnn_backend::hipdnnGetPluginUnloadingModeString(mode), ctx);
    }
};
