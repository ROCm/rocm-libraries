/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/system/cuda/memory.h>
#include <thrust/system_error.h>

#include <vector>

#include <unittest/testframework.h>

class CUDATestDriver : public UnitTestDriver
{
public:
  int current_device_architecture() const;

private:
  std::vector<int> target_devices(const ArgumentMap& kwargs);

  bool check_cuda_error(bool concise);

  virtual bool post_test_smoke_check(const UnitTest& test, bool concise);

  virtual bool run_tests(const ArgumentSet& args, const ArgumentMap& kwargs);
};

UnitTestDriver& driver_instance(thrust::system::cuda::tag);
