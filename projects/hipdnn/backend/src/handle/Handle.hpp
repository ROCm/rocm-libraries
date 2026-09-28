// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "plugin/EnginePluginResourceManager.hpp"
#include "plugin/HeuristicPluginResourceManager.hpp"
#include <cstdint>
#include <hip/hip_runtime.h>
#include <memory>
#include <optional>
#include <spdlog/fmt/fmt.h>
#include <string>
#include <string_view>
#include <vector>

struct hipdnnHandle // NOLINT
{
public:
    hipdnnHandle();
    virtual ~hipdnnHandle() = default;
    virtual void setStream(hipStream_t stream);
    virtual hipStream_t getStream() const;
    virtual std::shared_ptr<hipdnn_backend::plugin::EnginePluginResourceManager>
        getPluginResourceManager() const;
    virtual std::shared_ptr<hipdnn_backend::plugin::HeuristicPluginResourceManager>
        getHeuristicPluginResourceManager() const;
    virtual size_t getEngineCount() const;
    virtual std::vector<hipdnn_backend::plugin::EngineInfo> getEngineInfos() const;
    virtual std::optional<int64_t> findEngineIdByName(std::string_view engineName) const;
    virtual std::optional<std::string> findEngineNameById(int64_t engineId) const;
    virtual std::string toString() const;

private:
    hipStream_t _stream = nullptr;
    std::shared_ptr<hipdnn_backend::plugin::EnginePluginResourceManager> _pluginResourceManager;
    std::shared_ptr<hipdnn_backend::plugin::HeuristicPluginResourceManager>
        _heuristicPluginResourceManager;
};

template <>
struct fmt::formatter<hipdnnHandle> : fmt::formatter<std::string>
{
    auto format(const hipdnnHandle& handle, format_context& ctx) const
    {
        return fmt::formatter<std::string>::format(handle.toString(), ctx);
    }
};
