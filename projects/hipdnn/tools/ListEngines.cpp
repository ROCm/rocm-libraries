// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_backend.h"

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace
{

void printHelp(const char* programName)
{
    std::cout << "Usage: " << programName << " [OPTIONS]\n"
              << "Options:\n"
              << "  --plugin-dir <path>  Directory to scan for plugins (repeatable)\n"
              << "  --help, -h           Show this help message\n";
}

void printEngineInfo(hipdnnHandle_t handle, size_t engineIndex)
{
    // Query required buffer sizes and engine ID
    int64_t engineId = 0;
    size_t nameLen = 0;
    size_t versionLen = 0;
    size_t typeLen = 0;
    hipdnnGetEngineInfo_ext(
        handle, engineIndex, &engineId, nullptr, &nameLen, nullptr, &versionLen, nullptr, &typeLen);

    // Retrieve engine info strings
    std::vector<char> name(nameLen);
    std::vector<char> version(versionLen);
    std::vector<char> type(typeLen);
    hipdnnGetEngineInfo_ext(handle,
                            engineIndex,
                            nullptr,
                            name.data(),
                            &nameLen,
                            version.data(),
                            &versionLen,
                            type.data(),
                            &typeLen);

    std::cout << "  " << name.data() << " (0x" << std::hex << std::uppercase << std::setw(16)
              << std::setfill('0') << engineId << std::dec << ")\n"
              << "    Version: " << version.data() << "\n"
              << "    Type:    " << type.data() << "\n";
}

} // anonymous namespace

int main(int argc, char* argv[])
{
    std::vector<std::string> pluginDirs;

    for(int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if(arg == "--help" || arg == "-h")
        {
            printHelp(argv[0]);
            return 0;
        }
        else if(arg == "--plugin-dir")
        {
            if(i + 1 >= argc)
            {
                std::cerr << "Error: --plugin-dir requires a path argument\n";
                return 1;
            }
            pluginDirs.emplace_back(argv[++i]);
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << "\n";
            printHelp(argv[0]);
            return 1;
        }
    }

    if(!pluginDirs.empty())
    {
        std::vector<const char*> paths;
        paths.reserve(pluginDirs.size());
        for(const auto& dir : pluginDirs)
        {
            paths.push_back(dir.c_str());
        }

        auto status = hipdnnSetEnginePluginPaths_ext(
            paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE);
        if(status != HIPDNN_STATUS_SUCCESS)
        {
            std::cerr << "Error: failed to set plugin paths\n";
            return 1;
        }
    }

    hipdnnHandle_t handle = nullptr;
    if(hipdnnCreate(&handle) != HIPDNN_STATUS_SUCCESS)
    {
        std::cerr << "Error: failed to create handle\n";
        return 1;
    }

    size_t numEngines = 0;
    hipdnnGetEngineCount_ext(handle, &numEngines);

    if(numEngines == 0)
    {
        std::cout << "No engines found.\n";
        hipdnnDestroy(handle);
        return 0;
    }

    std::cout << "Loaded engines:\n";
    for(size_t i = 0; i < numEngines; ++i)
    {
        printEngineInfo(handle, i);
    }

    hipdnnDestroy(handle);
    return 0;
}
