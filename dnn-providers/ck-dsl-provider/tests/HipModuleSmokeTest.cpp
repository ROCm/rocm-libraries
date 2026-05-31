// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <string>
#include <vector>

#include "CkDslContainer.hpp"
#include "TestUtils.hpp"
#include "python/CompileServiceBridge.hpp"
#include "runtime/HipModule.hpp"
#include "runtime/KernelArtifact.hpp"
#include "runtime/LaunchAbi.hpp"

namespace {

using ck_dsl_provider::ArgValue;
using ck_dsl_provider::CkDslContainer;
using ck_dsl_provider::HipModule;
using ck_dsl_provider::KernelArtifact;
using ck_dsl_provider::LaunchAbi;

/// I-4 smoke test: drive a real HSACO from the embedded compile service
/// through hipModuleLoadData + hipModuleLaunchKernel against a real GPU.
///
/// Gated on hipGetDeviceCount() > 0 and a DSL-supported arch
/// (gfx942/gfx950/gfx1151) so the host-only CI lane and unsupported
/// devices stay green; this is the same gating pattern other providers
/// use for HIP-dependent unit tests.
class HipModuleSmoke : public ::testing::Test {
   protected:
    void SetUp() override {
        CK_DSL_PROVIDER_SKIP_IF_UNSUPPORTED_ARCH("HipModuleSmoke", _arch);
    }

    std::string _arch;
};

TEST_F(HipModuleSmoke, CompileServiceSmokeRoundTrip) {
    CkDslContainer container;
    auto& bridge = container.compileServiceBridge();

    KernelArtifact artifact = bridge.compileSmoke(_arch);

    // Sanity-check the artifact matches the compile_smoke contract
    // (elementwise copy built for the present arch: kernel name suffix,
    // non-empty HSACO, exactly three args, one-block grid, single-wave block).
    EXPECT_FALSE(artifact.hsaco.empty()) << "compileSmoke returned empty HSACO";
    EXPECT_FALSE(artifact.kernelName.empty()) << "compileSmoke returned empty kernel name";
    EXPECT_EQ(artifact.kind, "elementwise_copy_smoke");
    ASSERT_EQ(artifact.argSchema.size(), 3u);
    EXPECT_EQ(artifact.grid.x, 1u);
    EXPECT_EQ(artifact.grid.y, 1u);
    EXPECT_EQ(artifact.grid.z, 1u);
    EXPECT_EQ(artifact.block.x, 64u);
    EXPECT_EQ(artifact.block.y, 1u);
    EXPECT_EQ(artifact.block.z, 1u);
    EXPECT_EQ(artifact.ldsBytes, 0u);
}

TEST_F(HipModuleSmoke, LoadAndLaunchSucceeds) {
    CkDslContainer container;
    auto& bridge = container.compileServiceBridge();

    KernelArtifact artifact = bridge.compileSmoke(_arch);

    // Load the HSACO into a real HIP module. The ctor throws on any
    // hipModuleLoadData / hipModuleGetFunction failure with the HIP
    // error string in the message, so a green ctor is meaningful
    // signal that the HSACO format and kernel symbol both match.
    HipModule mod{artifact};
    EXPECT_NE(mod.moduleHandle(), nullptr);
    EXPECT_NE(mod.functionHandle(), nullptr);
    EXPECT_EQ(mod.kernelName(), artifact.kernelName);

    // Allocate one FP16 element of input and one of output, prime the
    // input with a known bit pattern, and launch the elementwise-copy
    // kernel over N=1.
    constexpr std::int32_t kNumel = 1;
    constexpr std::uint16_t kInputBits = 0x3C00;  // FP16 1.0

    std::uint16_t* dA = nullptr;
    std::uint16_t* dC = nullptr;
    ASSERT_EQ(hipMalloc(&dA, sizeof(std::uint16_t) * static_cast<std::size_t>(kNumel)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dC, sizeof(std::uint16_t) * static_cast<std::size_t>(kNumel)), hipSuccess);

    // Seed input = 1.0_f16 and output = 0xBEEF so a failed launch
    // (output unchanged) is distinguishable from a successful one.
    std::uint16_t hostInput = kInputBits;
    std::uint16_t hostSentinel = 0xBEEF;
    ASSERT_EQ(hipMemcpy(dA, &hostInput, sizeof(hostInput), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dC, &hostSentinel, sizeof(hostSentinel), hipMemcpyHostToDevice),
              hipSuccess);

    std::vector<ArgValue> values = {
        ArgValue::pointer(dA),
        ArgValue::pointer(dC),
        ArgValue::i32(kNumel),
    };
    std::vector<std::byte> packed = LaunchAbi::pack(artifact.argSchema, values);

    // The smoke kernel is a 3-arg (ptr, ptr, i32) layout. With natural
    // alignment the I32 sits at offset 16; total buffer size is 20.
    EXPECT_EQ(packed.size(), 20u);

    mod.launch(artifact, packed, /*stream=*/nullptr);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    std::uint16_t hostOutput = 0;
    ASSERT_EQ(hipMemcpy(&hostOutput, dC, sizeof(hostOutput), hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(hostOutput, kInputBits)
        << "elementwise copy did not write expected FP16 1.0 bit pattern; got 0x" << std::hex
        << hostOutput;

    EXPECT_EQ(hipFree(dA), hipSuccess);
    EXPECT_EQ(hipFree(dC), hipSuccess);
}

// Device-free coverage that the arch-aware smoke compile threads its
// target through to comgr for every supported arch. comgr cross-compiles
// without the matching device, so this runs on any box -- the
// verification that compileSmoke(arch) is genuinely multi-arch (the GPU
// smoke tests above only exercise whatever device is present).
class CompileSmokeHost : public ::testing::TestWithParam<std::string> {};

TEST_P(CompileSmokeHost, CompilesForArch) {
    const std::string arch = GetParam();
    CkDslContainer container;
    KernelArtifact artifact = container.compileServiceBridge().compileSmoke(arch);
    EXPECT_FALSE(artifact.hsaco.empty()) << arch << ": empty HSACO";
    EXPECT_NE(artifact.isa.find(arch), std::string::npos)
        << "compiled ISA '" << artifact.isa << "' does not target " << arch;
    EXPECT_EQ(artifact.kind, "elementwise_copy_smoke");
}

INSTANTIATE_TEST_SUITE_P(Arches, CompileSmokeHost, ::testing::Values("gfx942", "gfx950", "gfx1151"),
                         [](const ::testing::TestParamInfo<std::string>& info) {
                             return info.param;
                         });

}  // namespace
