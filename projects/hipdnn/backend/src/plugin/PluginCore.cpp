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
    const auto funcNameGetName = "hipdnnPluginGetName";
    _funcGetName = _lib.getSymbol<decltype(_funcGetName)>(funcNameGetName);

    const auto funcNameGetVersion = "hipdnnPluginGetVersion";
    _funcGetVersion = _lib.getSymbol<decltype(_funcGetVersion)>(funcNameGetVersion);

    const auto funcNameGetType = "hipdnnPluginGetType";
    _funcGetType = _lib.getSymbol<decltype(_funcGetType)>(funcNameGetType);

    const auto funcNameGetLastErrorStr = "hipdnnPluginGetLastErrorString";
    _funcGetLastErrorStr = _lib.getSymbol<decltype(_funcGetLastErrorStr)>(funcNameGetLastErrorStr);

    // Logging callback is optional
    try
    {
        const auto funcNameSetLoggingCallback = "hipdnnPluginSetLoggingCallback";
        _funcSetLoggingCallback
            = _lib.getSymbol<decltype(_funcSetLoggingCallback)>(funcNameSetLoggingCallback);
    }
    catch(const HipdnnException&)
    {
        _funcSetLoggingCallback = nullptr;

        // Add name of plugin if ever possible
        HIPDNN_LOG_INFO("Plugin does not support logging callback");
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

} // namespace plugin
} // hipdnn_backend
