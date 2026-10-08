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

#include <thrust/system/hip/memory.h>

#include <hip/hip_runtime.h>

#include <numeric>

#include <unittest/ctest.h>
#include <unittest/hip/testframework.h>
#include <unittest/testframework.h>

#define HIP_CHECK(condition)                                                                                         \
  do                                                                                                                 \
  {                                                                                                                  \
    hipError_t error = condition;                                                                                    \
    if (error != hipSuccess)                                                                                         \
    {                                                                                                                \
      std::cout << "HIP error: " << hipGetErrorString(error) << " on: " << __FILE__ << ":" << __LINE__ << std::endl; \
      exit(EXIT_FAILURE);                                                                                            \
    }                                                                                                                \
  } while (0)

__global__ void dummy_kernel() {}

bool binary_exists_for_current_device()
{
  // check against the dummy_kernel
  // if we're unable to get the attributes, then
  // we didn't compile a binary compatible with the current device
  hipFuncAttributes attr;
  hipError_t error = hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(dummy_kernel));

  // clear the HIP global error state if we just set it, so that
  // check_hip_error doesn't complain
  if (hipSuccess != error)
  {
    (void) hipGetLastError();
  }

  return hipSuccess == error;
}

void list_devices(void)
{
  int deviceCount;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount == 0)
  {
    std::cout << "There is no device supporting HIP" << std::endl;
  }

  int selected_device;
  HIP_CHECK(hipGetDevice(&selected_device));

  for (int dev = 0; dev < deviceCount; ++dev)
  {
    hipDeviceProp_t deviceProp;
    HIP_CHECK(hipGetDeviceProperties(&deviceProp, dev));

    if (dev == 0)
    {
      if (deviceProp.major == 9999 && deviceProp.minor == 9999)
      {
        std::cout << "There is no device supporting HIP." << std::endl;
      }
      else if (deviceCount == 1)
      {
        std::cout << "There is 1 device supporting HIP" << std::endl;
      }
      else
      {
        std::cout << "There are " << deviceCount << " devices supporting HIP" << std::endl;
      }
    }

    std::cout << "\nDevice " << dev << ": \"" << deviceProp.name << "\"";
    if (dev == selected_device)
    {
      std::cout << "  [SELECTED]";
    }
    std::cout << std::endl;

    std::cout << "  Major revision number:                         " << deviceProp.major << std::endl;
    std::cout << "  Minor revision number:                         " << deviceProp.minor << std::endl;
    std::cout
      << "  Total amount of global memory:                 " << deviceProp.totalGlobalMem << " bytes" << std::endl;
  }
  std::cout << std::endl;
}

std::vector<int> HIPTestDriver::target_devices(const ArgumentMap& kwargs)
{
  std::vector<int> result;

  // by default, test all devices in the system (device id -1)
  int device_id = kwargs.count("device") ? atoi(kwargs.find("device")->second.c_str()) : -1;
  // unless the target device is set by ctest
  int device_from_ctest = unittest::get_device_from_ctest();

  if (device_id < 0 && device_from_ctest >= 0)
  {
    device_id = device_from_ctest;
  }

  if (device_id < 0)
  {
    // target all devices in the system
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));

    result.resize(count);
    std::iota(result.begin(), result.end(), 0);
  }
  else
  {
    // target the specified device
    result = std::vector<int>(1, device_id);
  }

  return result;
}

bool HIPTestDriver::check_hip_error(bool concise)
{
  hipError_t const error = hipGetLastError();
  if (hipSuccess != error)
  {
    if (!concise)
    {
      std::cout << "[ERROR] HIP error detected before running tests: [" << std::string(hipGetErrorName(error)) << ": "
                << std::string(hipGetErrorString(error)) << "]" << std::endl;
    }
  }

  return hipSuccess != error;
}

bool HIPTestDriver::post_test_smoke_check(const UnitTest& test, bool concise)
{
  hipError_t const error = hipDeviceSynchronize();
  if (hipSuccess != error)
  {
    if (!concise)
    {
      std::cout
        << "\t[ERROR] HIP error detected after running " << test.name << ": [" << std::string(hipGetErrorName(error))
        << ": " << std::string(hipGetErrorString(error)) << "]" << std::endl;
    }
  }

  return hipSuccess == error;
}

bool HIPTestDriver::run_tests(const ArgumentSet& args, const ArgumentMap& kwargs)
{
  bool verbose = kwargs.count("verbose");
  bool concise = kwargs.count("concise");

  if (verbose && concise)
  {
    std::cout << "--verbose and --concise cannot be used together" << std::endl;
    exit(EXIT_FAILURE);
    return false;
  }

  // check error status before doing anything
  if (check_hip_error(concise))
  {
    return false;
  }

  bool result = true;

  if (kwargs.count("verbose"))
  {
    list_devices();
  }

  // figure out which devices to target
  std::vector<int> devices = target_devices(kwargs);

  // target each device
  for (std::vector<int>::iterator device = devices.begin(); device != devices.end(); ++device)
  {
    HIP_CHECK(hipDeviceSynchronize());

    // set the device
    HIP_CHECK(hipSetDevice(*device));

    // check if a binary exists for this device
    // if none exists, skip the device silently unless this is the only one we're targeting
    if (devices.size() > 1 && !binary_exists_for_current_device())
    {
      // note which device we're skipping
      hipDeviceProp_t deviceProp;
      HIP_CHECK(hipGetDeviceProperties(&deviceProp, *device));

      std::cout << "Skipping Device " << *device << ": \"" << deviceProp.name << "\"" << std::endl;

      continue;
    }

    if (!concise)
    {
      // note which device we're testing
      hipDeviceProp_t deviceProp;
      HIP_CHECK(hipGetDeviceProperties(&deviceProp, *device));

      std::cout << "Testing Device " << *device << ": \"" << deviceProp.name << "\"" << std::endl;
    }

    // check error status before running any tests
    if (check_hip_error(concise))
    {
      return false;
    }

    // run tests
    result &= UnitTestDriver::run_tests(args, kwargs);

    if (!concise && std::next(device) != devices.end())
    {
      // provide some separation between the output of separate tests
      std::cout << std::endl;
    }
  }

  return result;
}

int HIPTestDriver::current_device_architecture() const
{
  int current = -1;
  HIP_CHECK(hipGetDevice(&current));
  hipDeviceProp_t deviceProp;
  HIP_CHECK(hipGetDeviceProperties(&deviceProp, current));

  return 100 * deviceProp.major + 10 * deviceProp.minor;
}

UnitTestDriver& driver_instance(thrust::system::hip::tag)
{
  static HIPTestDriver s_instance;
  return s_instance;
}

bool HIPTestDriver::supports_managed_memory() const
{
  int current = -1;
  HIP_CHECK(hipGetDevice(&current));

  int managedMemory = 0;
  HIP_CHECK(hipDeviceGetAttribute(&managedMemory, hipDeviceAttributeManagedMemory, current));

  return managedMemory;
}
