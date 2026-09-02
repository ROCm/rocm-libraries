// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "PlatformUtils.hpp"

#if defined(__linux__)

#include "HipdnnException.hpp"
#include <dlfcn.h>
#include <spdlog/fmt/fmt.h>
#include <sys/random.h>
#include <sys/utsname.h>

namespace hipdnn_backend::platform_utilities
{

std::filesystem::path getCurrentModuleDirectory()
{
    std::filesystem::path modulePath;

    Dl_info info;
    if(dladdr(reinterpret_cast<const void*>(&getCurrentModuleDirectory), &info) != 0
       && info.dli_fname != nullptr && info.dli_fname[0] != '\0')
    {
        modulePath = std::filesystem::path(info.dli_fname).parent_path();
    }
    else
    {
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR, "Failed to get module file name.");
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
    const char* error = dlerror();
    if(error != nullptr)
    {
        throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                              "Failed to get symbol: " + std::string(symbolName)
                                  + " (Error: " + error + ")");
    }
    return symbol;
}

std::string getSystemInfo()
{
    struct utsname buffer;
    if(uname(&buffer) != 0)
    {
        return "Failed to retrieve system information using uname";
    }

    return fmt::format(
        "System Information: {{System Name: {}, Node Name: {}, Release: {}, Version: "
        "{}, Machine: {}}}",
        buffer.sysname,
        buffer.nodename,
        buffer.release,
        buffer.version,
        buffer.machine);
}

std::array<uint8_t, 16> generateUuidV4()
{
    // getentropy() fills the whole buffer or fails, for any request up to 256 bytes.
    std::array<uint8_t, 16> bytes{};
    if(getentropy(bytes.data(), bytes.size()) != 0)
    {
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR,
                              "Failed to generate graph UUID using getentropy.");
    }

    bytes[6] = static_cast<uint8_t>((bytes[6] & 0x0fU) | 0x40U);
    bytes[8] = static_cast<uint8_t>((bytes[8] & 0x3fU) | 0x80U);
    return bytes;
}

}

#endif // defined(__linux__)
