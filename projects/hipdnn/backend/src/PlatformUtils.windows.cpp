// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "PlatformUtils.hpp"

#ifdef _WIN32

#include "HipdnnException.hpp"

namespace hipdnn_backend::platform_utilities
{

std::filesystem::path getCurrentModuleDirectory()
{
    std::filesystem::path modulePath;

    HMODULE moduleHandle = nullptr;
    if(GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                              | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCSTR>(&getCurrentModuleDirectory),
                          &moduleHandle)
       == TRUE)
    {
        char* dst = new char[MAX_PATH];
        DWORD len = GetModuleFileNameA(moduleHandle, dst, MAX_PATH);
        std::string modulePathStr(dst);
        delete[] dst;

        if(len > 0 && len < MAX_PATH)
        {
            modulePath = std::filesystem::path(modulePathStr).parent_path();
        }
        else
        {
            throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR, "Failed to get module file name.");
        }
    }
    else
    {
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR, "Failed to get module handle.");
    }

    return std::filesystem::weakly_canonical(std::filesystem::absolute(modulePath));
}

PluginLibHandle openLibrary(const std::filesystem::path& libraryPath)
{
    PluginLibHandle handle = LoadLibraryW(libraryPath.wstring().c_str());
    if(handle == nullptr)
    {
        auto errorCode = GetLastError();
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                              "Failed to load library: " + libraryPath.string()
                                  + " (Error Code: " + std::to_string(errorCode) + ")");
    }

    return handle;
}

void closeLibrary(PluginLibHandle handle)
{
    FreeLibrary(handle);
}

void* getSymbol(PluginLibHandle handle, const char* symbolName)
{
    void* symbol = reinterpret_cast<void*>(GetProcAddress(handle, symbolName));
    if(symbol == nullptr)
    {
        auto errorCode = GetLastError();
        throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                              "Failed to get symbol: " + std::string(symbolName)
                                  + " (Error Code: " + std::to_string(errorCode) + ")");
    }

    return symbol;
}

}

#endif // _WIN32
