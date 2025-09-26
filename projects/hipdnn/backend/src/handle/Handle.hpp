// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "plugin/EnginePluginResourceManager.hpp"
#include <cstdint>
#include <hip/hip_runtime.h>
#include <memory>

struct hipdnnHandle // NOLINT
{
public:
    hipdnnHandle();
    virtual ~hipdnnHandle() = default;
    virtual void setStream(hipStream_t stream);
    virtual hipStream_t getStream() const;
    virtual std::shared_ptr<hipdnn_backend::plugin::EnginePluginResourceManager>
        getPluginResourceManager() const;

private:
    hipStream_t _stream = nullptr;
    std::shared_ptr<hipdnn_backend::plugin::EnginePluginResourceManager> _pluginResourceManager;
};
