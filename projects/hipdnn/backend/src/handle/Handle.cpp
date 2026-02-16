// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "Handle.hpp"
#include "utilities/PointerToString.hpp"

using namespace hipdnn_backend::plugin;

hipdnnHandle::hipdnnHandle()
    : _pluginResourceManager(EnginePluginResourceManager::create())
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

std::string hipdnnHandle::toString() const
{
    std::string str = "hipdnnHandle: {";
    str += "stream=" + hipdnn_backend::ptrToString(_stream);
    str += ", "
           + (_pluginResourceManager != nullptr ? _pluginResourceManager->toString()
                                                : "pluginResourceManager=null");
    str += "}";
    return str;
}
