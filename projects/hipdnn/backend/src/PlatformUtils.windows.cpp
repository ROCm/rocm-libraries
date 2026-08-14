// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef _WIN32
// Must precede every include: <stdlib.h> only declares rand_s when this is already defined, and
// any header that reaches it first silently leaves rand_s undeclared.
#define _CRT_RAND_S
#endif

#include "PlatformUtils.hpp"

#ifdef _WIN32

#include "HipdnnException.hpp"
#include <array>
#include <cstdlib>
#include <cstring>
#include <spdlog/fmt/fmt.h>
#include <winternl.h>

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
        std::array<char, MAX_PATH> dst{};
        DWORD len = GetModuleFileNameA(moduleHandle, dst.data(), MAX_PATH);
        std::string modulePathStr(dst.data());

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
    try
    {
        return hipdnn_data_sdk::utilities::openLibrary(libraryPath);
    }
    catch(const std::runtime_error& ex)
    {
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM, ex.what());
    }
}

void closeLibrary(PluginLibHandle handle)
{
    hipdnn_data_sdk::utilities::closeLibrary(handle);
}

void* getSymbol(PluginLibHandle handle, const char* symbolName)
{
    void* symbol = hipdnn_data_sdk::utilities::getSymbol(handle, symbolName);
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
    RTL_OSVERSIONINFOW versionInfo = {};
    versionInfo.dwOSVersionInfoSize = sizeof(versionInfo);

    bool versionInfoValid = false;
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    if(ntdll != nullptr)
    {
        auto rtlGetVersion
            = reinterpret_cast<RtlGetVersionPtr>(GetProcAddress(ntdll, "RtlGetVersion"));
        if(rtlGetVersion != nullptr)
        {
            versionInfoValid = (rtlGetVersion(&versionInfo) == 0);
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

    if(versionInfoValid)
    {
        return fmt::format(
            "System Information: {{System Name: Windows, Node Name: {}, Release: {}.{}, "
            "Version: {}, Machine: {}}}",
            computerName.data(),
            versionInfo.dwMajorVersion,
            versionInfo.dwMinorVersion,
            versionInfo.dwBuildNumber,
            architecture);
    }

    return fmt::format(
        "System Information: {{System Name: Windows, Node Name: {}, Release: unknown, "
        "Version: unknown, Machine: {}}}",
        computerName.data(),
        architecture);
}

std::array<uint8_t, 16> generateUuidV4()
{
    std::array<uint8_t, 16> bytes{};
    for(size_t offset = 0; offset < bytes.size(); offset += sizeof(unsigned int))
    {
        unsigned int randomValue;
        if(rand_s(&randomValue) != 0)
        {
            throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR,
                                  "Failed to generate graph UUID using rand_s.");
        }
        std::memcpy(bytes.data() + offset, &randomValue, sizeof(randomValue));
    }

    bytes[6] = static_cast<uint8_t>((bytes[6] & 0x0fU) | 0x40U);
    bytes[8] = static_cast<uint8_t>((bytes[8] & 0x3fU) | 0x80U);
    return bytes;
}

}

#endif // _WIN32
