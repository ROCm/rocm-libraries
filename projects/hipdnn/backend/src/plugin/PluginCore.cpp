// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cassert>

#include "PluginCore.hpp"

namespace hipdnn_backend
{
namespace plugin
{

PluginBase::PluginBase(SharedLibrary&& lib)
    : _lib(std::move(lib))
{
    resolveSymbols();
}

PluginBase::PluginBase()
{
    // This constructor is used for mocking purposes in tests.
#ifndef NDEBUG
    _initialized = false;
#endif
}

void PluginBase::resolveSymbols()
{
    _funcGetName = _lib.getSymbol<decltype(_funcGetName)>("hipdnnPluginGetName");
    _funcGetVersion = _lib.getSymbol<decltype(_funcGetVersion)>("hipdnnPluginGetVersion");
    _funcGetType = _lib.getSymbol<decltype(_funcGetType)>("hipdnnPluginGetType");
    _funcGetLastErrorStr
        = _lib.getSymbol<decltype(_funcGetLastErrorStr)>("hipdnnPluginGetLastErrorString");

    if(!tryAssignSymbol(_funcGetApiVersion, "hipdnnPluginGetApiVersion"))
    {
        HIPDNN_BACKEND_LOG_INFO("Plugin does not support returning plugin API version");
    }
    if(!tryAssignSymbol(_funcSetLoggingCallback, "hipdnnPluginSetLoggingCallback"))
    {
        HIPDNN_BACKEND_LOG_INFO("Plugin does not support logging callback");
    }
    if(!tryAssignSymbol(_funcSetLogLevel, "hipdnnPluginSetLogLevel"))
    {
        HIPDNN_BACKEND_LOG_INFO("Plugin does not support log level synchronization");
    }

#ifndef NDEBUG
    _initialized = true;
#endif
}

std::string_view PluginBase::name() const
{
    assert(_initialized);
    const char* name;
    invokePluginFunction("get plugin name", _funcGetName, &name);
    return name;
}

std::string_view PluginBase::version() const
{
    assert(_initialized);
    const char* version;
    invokePluginFunction("get plugin version", _funcGetVersion, &version);
    return version;
}

std::string_view PluginBase::apiVersion() const
{
    assert(_initialized);
    const char* version;
    if(_funcGetApiVersion == nullptr)
    {
        return "0.0.0";
    }
    invokePluginFunction("get plugin api version", _funcGetApiVersion, &version);
    return version;
}

hipdnnPluginType_t PluginBase::type() const
{
    assert(_initialized);
    hipdnnPluginType_t type;
    invokePluginFunction("get plugin type", _funcGetType, &type);
    return type;
}

std::string_view PluginBase::getLastErrorString() const noexcept
{
    assert(_initialized);
    const char* errorStr = nullptr;
    _funcGetLastErrorStr(&errorStr);
    return errorStr;
}

hipdnnPluginStatus_t PluginBase::setLoggingCallback(hipdnnCallback_t callback) const
{
    assert(_initialized);
    if(_funcSetLoggingCallback == nullptr)
    {
        // Plugin does not support logging callback, so we vacuously return success
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }

    return _funcSetLoggingCallback(callback);
}

hipdnnPluginStatus_t PluginBase::setLogLevel(hipdnnSeverity_t level) const
{
    assert(_initialized);
    if(_funcSetLogLevel == nullptr)
    {
        // Plugin does not support log level synchronization, so we vacuously return success
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }

    return _funcSetLogLevel(level);
}

} // namespace plugin
} // hipdnn_backend
