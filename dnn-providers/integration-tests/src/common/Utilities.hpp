// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_backend.h>

#include <string>
#include <vector>

namespace hipdnn_integration_tests
{

struct EngineInfo
{
    int64_t engineId = 0;
    std::string engineName;
    std::string pluginName;
    std::string version;
    std::string type;
};

// Wraps the two-call pattern for hipdnnGetEngineInfo_ext:
// first call with nullptr buffers to query sizes, second call to fetch data.
inline EngineInfo getEngineInfo(hipdnnHandle_t handle, size_t engineIndex)
{
    EngineInfo info;
    size_t engineNameLen = 0;
    size_t pluginNameLen = 0;
    size_t versionLen = 0;
    size_t typeLen = 0;

    hipdnnGetEngineInfo_ext(handle,
                            engineIndex,
                            &info.engineId,
                            nullptr,
                            &engineNameLen,
                            nullptr,
                            &pluginNameLen,
                            nullptr,
                            &versionLen,
                            nullptr,
                            &typeLen);

    info.engineName.resize(engineNameLen);
    info.pluginName.resize(pluginNameLen);
    info.version.resize(versionLen);
    info.type.resize(typeLen);

    hipdnnGetEngineInfo_ext(handle,
                            engineIndex,
                            &info.engineId,
                            info.engineName.data(),
                            &engineNameLen,
                            info.pluginName.data(),
                            &pluginNameLen,
                            info.version.data(),
                            &versionLen,
                            info.type.data(),
                            &typeLen);

    if(!info.engineName.empty() && info.engineName.back() == '\0')
    {
        info.engineName.pop_back();
    }

    return info;
}

// Enumerates every engine currently loaded on `handle` (RFC 0015 §7.3/§8:
// support-claim enforcement is multi-engine -- it evaluates every loaded
// engine, not just a pinned --test-engine). Empty on failure or when no
// engines are loaded. Cheap in-process bookkeeping (no GPU dispatch), so
// callers may call it once per test rather than caching across the run.
inline std::vector<std::string> listLoadedEngineNames(hipdnnHandle_t handle)
{
    std::vector<std::string> names;
    size_t numEngines = 0;
    if(hipdnnGetEngineCount_ext(handle, &numEngines) != HIPDNN_STATUS_SUCCESS)
    {
        return names;
    }

    names.reserve(numEngines);
    for(size_t i = 0; i < numEngines; ++i)
    {
        names.push_back(getEngineInfo(handle, i).engineName);
    }
    return names;
}

} // namespace hipdnn_integration_tests
