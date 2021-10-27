#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <string>
#include <algorithm>
#include <cstdlib>
#include <ctype.h>

namespace unittest
{

int get_device_from_ctest()
{
    static const std::string rg0 = "CTEST_RESOURCE_GROUP_0";
    if (std::getenv(rg0.c_str()) != nullptr)
    {
        std::string amdgpu_target = std::getenv(rg0.c_str());
        std::transform(amdgpu_target.cbegin(), amdgpu_target.cend(), amdgpu_target.begin(), ::toupper);
        std::string reqs = std::getenv((rg0 + "_" + amdgpu_target).c_str());
        int device_id = std::atoi(reqs.substr(reqs.find(':') + 1, reqs.find(',') - (reqs.find(':') + 1)).c_str());
        return device_id; 
    }
    else
        return -1;
}

}