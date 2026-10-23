// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/API/Compiler.hpp>

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#define ASSERT_HIP_SUCCESS(expr)                                               \
  do {                                                                         \
    hipError_t status = (expr);                                                \
    ASSERT_EQ(status, hipSuccess)                                              \
        << #expr << ": " << hipGetErrorString(status);                         \
  } while (0)

using namespace rocm::tensilelite;

constexpr int64_t blockSize = 64;
constexpr int64_t gridSize = 4;
constexpr int64_t elementCount = blockSize * gridSize;
constexpr const char *kernelName = "jit_compiler_kernel";
constexpr std::string_view targetPrefix = "amdgcn-amd-amdhsa--";
constexpr std::string_view directivePrefix =
    ".amdgcn_target \"amdgcn-amd-amdhsa-";

// The global id is computed from a block size fixed at compile time, which
// avoids reading the dispatch packet. The attribute makes the runtime reject a
// launch with a different block size.
constexpr std::string_view kernelIR = R"(target triple = "amdgcn-amd-amdhsa"

declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.workgroup.id.x()

define amdgpu_kernel void @jit_compiler_kernel(ptr addrspace(1) %out) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %wg = call i32 @llvm.amdgcn.workgroup.id.x()
  %base = mul i32 %wg, 64
  %gid = add i32 %base, %tid
  %idx64 = zext i32 %gid to i64
  %ptr = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %idx64
  store i32 %gid, ptr addrspace(1) %ptr, align 4
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="64,64" }
)";

namespace {

// Deleters cannot assert, so they ignore errors.
struct DeviceFree {
  void operator()(int32_t *data) const { (void)hipFree(data); }
};
struct ModuleUnload {
  void operator()(hipModule_t module) const { (void)hipModuleUnload(module); }
};

} // namespace

static void launchAndVerify(const std::vector<uint8_t> &hsaco) {
  hipModule_t rawModule = nullptr;
  ASSERT_HIP_SUCCESS(hipModuleLoadData(&rawModule, hsaco.data()));
  std::unique_ptr<std::remove_pointer_t<hipModule_t>, ModuleUnload> module(
      rawModule);
  hipFunction_t function = nullptr;
  ASSERT_HIP_SUCCESS(hipModuleGetFunction(&function, rawModule, kernelName));

  int64_t bytes = elementCount * static_cast<int64_t>(sizeof(int32_t));
  int32_t *out = nullptr;
  ASSERT_HIP_SUCCESS(hipMalloc(&out, static_cast<size_t>(bytes)));
  std::unique_ptr<int32_t, DeviceFree> outGuard(out);
  // Every element becomes -1, which catches elements the kernel never writes.
  ASSERT_HIP_SUCCESS(hipMemset(out, 0xff, static_cast<size_t>(bytes)));
  void *args[] = {&out};
  ASSERT_HIP_SUCCESS(hipModuleLaunchKernel(
      function, static_cast<unsigned int>(gridSize), 1, 1,
      static_cast<unsigned int>(blockSize), 1, 1, 0, nullptr, args, nullptr));
  ASSERT_HIP_SUCCESS(hipDeviceSynchronize());

  std::vector<int32_t> result(elementCount);
  ASSERT_HIP_SUCCESS(hipMemcpy(result.data(), out, static_cast<size_t>(bytes),
                               hipMemcpyDeviceToHost));
  std::vector<int32_t> expected(elementCount);
  std::iota(expected.begin(), expected.end(), 0);
  EXPECT_EQ(result, expected);
}

// The device's ISA name carries feature suffixes such as :sramecc+:xnack-; the
// bare processor gives a code object that loads in any of those modes.
static std::string processorName(std::string_view gcnArchName) {
  return std::string(gcnArchName.substr(0, gcnArchName.find(':')));
}

// Returns the .amdgcn_target directive line, or an empty string without one.
static std::string targetDirective(std::string_view assembly) {
  size_t begin = assembly.find(".amdgcn_target ");
  if (begin == std::string_view::npos)
    return {};
  size_t end = assembly.find('\n', begin);
  return std::string(assembly.substr(begin, end - begin));
}

static bool startsWith(std::string_view text, std::string_view prefix) {
  return text.substr(0, prefix.size()) == prefix;
}

static bool endsWith(std::string_view text, std::string_view suffix) {
  return text.size() >= suffix.size() &&
         text.substr(text.size() - suffix.size()) == suffix;
}

TEST(JitCompiler, CompilesAssemblesAndLaunches) {
  int deviceCount = 0;
  bool hasDevice =
      hipGetDeviceCount(&deviceCount) == hipSuccess && deviceCount > 0;
  // Without a device the ASM checks still run against a fixed processor.
  std::string processor = "gfx950";
  if (hasDevice) {
    int device = 0;
    ASSERT_HIP_SUCCESS(hipGetDevice(&device));
    hipDeviceProp_t properties;
    ASSERT_HIP_SUCCESS(hipGetDeviceProperties(&properties, device));
    processor = processorName(properties.gcnArchName);
  }
  std::string target = std::string(targetPrefix) + processor;

  std::vector<uint8_t> asmBytes = compileToASM(kernelIR, target);
  std::string assembly(asmBytes.begin(), asmBytes.end());
  // LLVM spells the triple's empty environment differently across versions
  // (amdgcn-amd-amdhsa--gfx950 or amdgcn-amd-amdhsa-unknown-gfx950).
  std::string directive = targetDirective(assembly);
  EXPECT_TRUE(startsWith(directive, directivePrefix)) << directive;
  EXPECT_TRUE(endsWith(directive, "-" + processor + "\"")) << directive;
  EXPECT_NE(assembly.find(kernelName), std::string::npos);
  EXPECT_NE(assembly.find("s_endpgm"), std::string::npos);

  if (!hasDevice)
    GTEST_SKIP() << "no HIP device";
  {
    SCOPED_TRACE("assembleToHSACO");
    launchAndVerify(assembleToHSACO(assembly, target));
  }
  {
    SCOPED_TRACE("compileToHSACO");
    launchAndVerify(compileToHSACO(kernelIR, target));
  }
}
