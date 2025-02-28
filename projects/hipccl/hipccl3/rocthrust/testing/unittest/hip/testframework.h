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

#include <unittest/testframework.h>
#include <thrust/system/hip/memory.h>
#include <thrust/system_error.h>
#include <vector>

class HIPTestDriver
  : public UnitTestDriver
{
  public:
    int current_device_architecture() const;

    bool supports_managed_memory() const;

  private:
    std::vector<int> target_devices(const ArgumentMap &kwargs);

    bool check_hip_error(bool concise);

    virtual bool post_test_smoke_check(const UnitTest &test, bool concise);

    virtual bool run_tests(const ArgumentSet &args, const ArgumentMap &kwargs);
};

UnitTestDriver &driver_instance(thrust::system::hip::tag);
