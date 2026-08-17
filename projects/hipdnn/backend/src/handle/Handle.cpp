// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "Handle.hpp"
#include <spdlog/fmt/fmt.h>

using namespace hipdnn_backend::plugin;

hipdnnHandle::hipdnnHandle()
    : _pluginResourceManager(EnginePluginResourceManager::create())
    , _heuristicPluginResourceManager(HeuristicPluginResourceManager::create())
{
}

void hipdnnHandle::setStream(hipStream_t stream)
{
    _stream = stream;
    _pluginResourceManager->setStream(stream);
}

hipStream_t hipdnnHandle::getStream() const
{
    return _stream;
}

std::shared_ptr<EnginePluginResourceManager> hipdnnHandle::getPluginResourceManager() const
{
    return _pluginResourceManager;
}

std::shared_ptr<HeuristicPluginResourceManager>
    hipdnnHandle::getHeuristicPluginResourceManager() const
{
    return _heuristicPluginResourceManager;
}

size_t hipdnnHandle::getEngineCount() const
{
    return _pluginResourceManager->getEngineCount();
}

std::vector<EngineInfo> hipdnnHandle::getEngineInfos() const
{
    return _pluginResourceManager->getEngineInfos();
}

std::string hipdnnHandle::toString() const
{
    std::string str = "hipdnnHandle: {";
    str += "stream="
           + (_stream != nullptr ? fmt::format("{:p}", static_cast<void*>(_stream)) : "null");
    str += ", "
           + (_pluginResourceManager != nullptr ? _pluginResourceManager->toString()
                                                : "pluginResourceManager=null");
    str += '}';
    return str;
}
