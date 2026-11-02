// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <mutex>
#include <unordered_map>

#include <hip/hip_runtime_api.h>

#include "HipblasltUtils.hpp"
#include "ProgramCache.hpp"

namespace hipblaslt_plugin::compilation
{

std::shared_ptr<const Program> getProgram(const std::string& sourceName,
                                          const std::vector<std::string>& options)
{
    int device = -1;
    THROW_ON_HIP_FAILURE(hipGetDevice(&device));
    std::string key = std::to_string(device) + "::" + sourceName;
    for(const auto& option : options)
    {
        key.append("::");
        key.append(option);
    }

    static std::mutex s_mutex;
    static std::unordered_map<std::string, std::shared_ptr<const Program>> s_programs;
    // Held across the compile, so a second request for a key waits for the first
    // instead of compiling it again.
    const std::lock_guard<std::mutex> lock(s_mutex);
    const auto it = s_programs.find(key);
    if(it != s_programs.end())
    {
        return it->second;
    }

    hipDeviceProp_t properties{};
    THROW_ON_HIP_FAILURE(hipGetDeviceProperties(&properties, device));
    auto compileOptions = options;
    compileOptions.push_back(std::string("--offload-arch=") + properties.gcnArchName);
    auto program = std::make_shared<const Program>(sourceName, compileOptions);
    s_programs.emplace(std::move(key), program);
    return program;
}

} // namespace hipblaslt_plugin::compilation
