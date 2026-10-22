// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef ROCM_TENSILELITE_TESTS_CODEGEN_HIPTESTSUPPORT_HPP
#define ROCM_TENSILELITE_TESTS_CODEGEN_HIPTESTSUPPORT_HPP

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <memory>
#include <type_traits>

#define ASSERT_HIP_SUCCESS(expr)                                               \
  do {                                                                         \
    hipError_t status = (expr);                                                \
    ASSERT_EQ(status, hipSuccess)                                              \
        << #expr << ": " << hipGetErrorString(status);                         \
  } while (0)

namespace rocm::tensilelite::codegen::test {

// Deleters cannot assert, so they ignore errors.
struct ModuleUnload {
  void operator()(hipModule_t module) const { (void)hipModuleUnload(module); }
};

using ModulePtr =
    std::unique_ptr<std::remove_pointer_t<hipModule_t>, ModuleUnload>;

struct DeviceFree {
  void operator()(void *ptr) const { (void)hipFree(ptr); }
};

inline bool hasHipDevice() {
  int deviceCount = 0;
  return hipGetDeviceCount(&deviceCount) == hipSuccess && deviceCount > 0;
}

} // namespace rocm::tensilelite::codegen::test

#endif // ROCM_TENSILELITE_TESTS_CODEGEN_HIPTESTSUPPORT_HPP
