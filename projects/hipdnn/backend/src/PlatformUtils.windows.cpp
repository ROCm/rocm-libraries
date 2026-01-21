// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "PlatformUtils.hpp"

#ifdef _WIN32

#include "HipdnnException.hpp"
#include <spdlog/fmt/fmt.h>
#include <winternl.h>

namespace hipdnn_backend::platform_utilities
{

std::filesystem::path getCurrentModuleDirectory()
{
    HMODULE moduleHandle = nullptr;
    if(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                              | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCWSTR>(&getCurrentModuleDirectory),
                          &moduleHandle)
       == 0)
    {
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR, "Failed to get module handle.");
    }

    // Windows supports long paths up to 32,767 characters.
    // Allocate the maximum possible size to ensure we can fetch the module path.
    const DWORD maxPathSize = 32768;
    std::vector<wchar_t> buffer(maxPathSize);

    DWORD len = GetModuleFileNameW(moduleHandle, buffer.data(), static_cast<DWORD>(buffer.size()));
    if(len == 0)
    {
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR, "Failed to get module file name.");
    }

    // If len == buffer.size(), the path was truncated. However, since we allocated
    // the maximum supported path length, this should practically never happen.
    if(len == buffer.size())
    {
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR,
                              "Module file name exceeds maximum supported length.");
    }

    std::filesystem::path modulePath(std::wstring(buffer.data(), len));
    return std::filesystem::weakly_canonical(std::filesystem::absolute(modulePath)).parent_path();
}

PluginLibHandle openLibrary(const std::filesystem::path& libraryPath)
{
    // 1. Determine absolute paths
    auto absLibraryPath = std::filesystem::absolute(libraryPath);
    auto pluginDir = absLibraryPath.parent_path();

    // getCurrentModuleDirectory() returns the folder where hipdnn_backend.dll (and other ROCm
    // libraries) resides.
    auto baseDir = getCurrentModuleDirectory();

    // 2. Add specific directories to the DLL search path
    // cookies are used to remove them later
    DLL_DIRECTORY_COOKIE cookiePlugin = AddDllDirectory(pluginDir.wstring().c_str());
    DLL_DIRECTORY_COOKIE cookieBase = AddDllDirectory(baseDir.wstring().c_str());

    // 3. Load with LOAD_LIBRARY_SEARCH_USER_DIRS
    // This looks in: Application Dir, System32, and paths added via AddDllDirectory.
    // It specifically includes our 'baseDir' where other ROCm libraries (e.g. MIOpen.dll) are
    // located.
    // NOTE: This EXCLUDES the PATH environment variable.
    PluginLibHandle handle
        = LoadLibraryExW(absLibraryPath.wstring().c_str(),
                         nullptr,
                         LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);

    // 4. Cleanup search paths
    if(cookiePlugin)
    {
        RemoveDllDirectory(cookiePlugin);
    }
    if(cookieBase)
    {
        RemoveDllDirectory(cookieBase);
    }

    // 5. Fallback: If enhanced load failed, try standard load (searches PATH)
    if(handle == nullptr)
    {
        // This attempts to load using the standard search order, which includes PATH.
        // This ensures we don't break setups where dependencies are in the system PATH.
        handle = LoadLibraryW(libraryPath.wstring().c_str());
    }

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

std::string getSystemInfo()
{
    // Get Windows version using RtlGetVersion (more reliable than deprecated GetVersionEx)
    typedef LONG(WINAPI * RtlGetVersionPtr)(PRTL_OSVERSIONINFOW);
    RTL_OSVERSIONINFOW versionInfo;
    versionInfo.dwOSVersionInfoSize = sizeof(versionInfo);

    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    if(ntdll != nullptr)
    {
        auto rtlGetVersion
            = reinterpret_cast<RtlGetVersionPtr>(GetProcAddress(ntdll, "RtlGetVersion"));
        if(rtlGetVersion != nullptr)
        {
            rtlGetVersion(&versionInfo);
        }
    }

    // Get computer name
    std::array<char, MAX_COMPUTERNAME_LENGTH + 1> computerName;
    auto size = static_cast<DWORD>(computerName.size());
    if(GetComputerNameA(computerName.data(), &size) == FALSE)
    {
        strcpy_s(computerName.data(), computerName.size(), "Unknown");
    }

    // Get system architecture
    SYSTEM_INFO sysInfo;
    GetNativeSystemInfo(&sysInfo);

    std::string architecture;
    switch(sysInfo.wProcessorArchitecture)
    {
    case PROCESSOR_ARCHITECTURE_AMD64:
        architecture = "x86_64";
        break;
    case PROCESSOR_ARCHITECTURE_ARM64:
        architecture = "ARM64";
        break;
    case PROCESSOR_ARCHITECTURE_INTEL:
        architecture = "x86";
        break;
    default:
        architecture = "Unknown";
    }

    return fmt::format("System Information: {{System Name: Windows, Node Name: {}, Release: {}.{}, "
                       "Version: {}, Machine: {}}}",
                       computerName.data(),
                       versionInfo.dwMajorVersion,
                       versionInfo.dwMinorVersion,
                       versionInfo.dwBuildNumber,
                       architecture);
}

}

#endif // _WIN32
