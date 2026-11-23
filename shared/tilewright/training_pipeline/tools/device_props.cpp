// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// device_props: the device values hipBLASLt hands tilewright, read through HIP
// for device 0 of the GPUs the environment shows (lib/device_values.py runs
// it with ROCR_VISIBLE_DEVICES naming one GPU).
//
// TensileLite takes the CU count from hipDeviceAttributePhysicalMultiProcessorCount
// in place of hipDeviceProp_t::multiProcessorCount, LDS from sharedMemPerBlock and
// L2 from l2CacheSize; warpSize is the wave size. One JSON line on stdout:
//   {"arch", "multi_processor_count", "physical_multi_processor_count",
//    "shared_mem_per_block", "l2_cache_size", "warp_size"}
// Exit status 1 when no device or property can be read.
#include <hip/hip_runtime_api.h>

#include <cstdio>
#include <string>

int main() {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess || count < 1) {
    std::fprintf(stderr, "device_props: no HIP device\n");
    return 1;
  }
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
    std::fprintf(stderr, "device_props: hipGetDeviceProperties failed\n");
    return 1;
  }
  int physical = 0;
  if (hipDeviceGetAttribute(&physical, hipDeviceAttributePhysicalMultiProcessorCount, 0) !=
      hipSuccess) {
    physical = 0;
  }
  std::string arch;
  for (char const* c = prop.gcnArchName; *c != '\0'; ++c) {
    arch += (*c == '"' || *c == '\\') ? '_' : *c;
  }
  std::printf("{\"arch\": \"%s\", \"multi_processor_count\": %d, "
              "\"physical_multi_processor_count\": %d, \"shared_mem_per_block\": %zu, "
              "\"l2_cache_size\": %d, \"warp_size\": %d}\n",
              arch.c_str(),
              prop.multiProcessorCount,
              physical,
              static_cast<size_t>(prop.sharedMemPerBlock),
              prop.l2CacheSize,
              prop.warpSize);
  return 0;
}
