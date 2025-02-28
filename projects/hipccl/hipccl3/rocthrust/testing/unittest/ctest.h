/*
 *  Copyright© 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

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