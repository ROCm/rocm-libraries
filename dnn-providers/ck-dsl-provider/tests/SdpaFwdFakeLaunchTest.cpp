// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "CkDslContainer.hpp"
#include "CkDslHandle.hpp"
#include "TestUtils.hpp"
#include "engines/sdpa/SdpaFwdPlan.hpp"
#include "python/CompileServiceBridge.hpp"
#include "runtime/DeviceArch.hpp"
#include "runtime/HipModule.hpp"
#include "runtime/KernelArtifact.hpp"

namespace {

using ck_dsl_provider::CkDslContainer;
using ck_dsl_provider::HipModule;
using ck_dsl_provider::KernelArtifact;
using ck_dsl_provider::SdpaFwdPlan;

/// Phase-4 full-plumbing verification (ALMIOPEN-2002). This drives the
/// PRODUCTION ``SdpaFwdPlan::execute`` against a FAKE kernel that has the
/// exact 18-slot unified-SDPA ABI but no attention math: thread 0 writes
/// an ABI-slot fingerprint into the output buffer. Asserting the
/// fingerprint proves execute() bound all 18 args correctly, marshalled +
/// uploaded the three host integer arrays into the workspace, and
/// launched -- all on real hardware (this dev box is gfx90a). Only the
/// gfx950 kernel's numerics defer; the launch plumbing is verified here.
///
/// Gated by CK_DSL_PROVIDER_SKIP_IF_NO_GPU: the fake kernel compiles for
/// whatever arch the device reports (the POC added a gfx90a arch_specs
/// row), so this is not gfx950-pinned.
TEST(SdpaFwdFakeLaunch, ExecuteBindsAllAbiSlots) {
    CK_DSL_PROVIDER_SKIP_IF_NO_GPU("SdpaFwdFakeLaunch.ExecuteBindsAllAbiSlots");

    CkDslContainer container;
    ::CkDslHandle handle;

    std::optional<std::string> arch = ck_dsl_provider::detectDeviceArch(handle.getStream());
    ASSERT_TRUE(arch.has_value()) << "a device is present but its arch could not be detected";

    // Compile the fake kernel for the detected arch and load it into a
    // real HIP module (the same path production uses for a JitCache miss).
    KernelArtifact artifact = container.compileServiceBridge().compileSdpaFwdFake(*arch);
    ASSERT_NE(artifact.isa.find(*arch), std::string::npos)
        << "fake kernel ISA '" << artifact.isa << "' does not target " << *arch;
    ASSERT_EQ(artifact.argSchema.size(), 18u) << "fake kernel must expose the 18-slot unified ABI";
    auto module = std::make_shared<HipModule>(artifact);

    // Small dense shape. block_size=32 with Skv=64 gives
    // block_table_stride = ceil(64/32) = 2.
    constexpr std::int32_t kBatch = 2;
    constexpr std::int32_t kSeqlenQ = 64;
    constexpr std::int32_t kSeqlenK = 64;
    constexpr std::int32_t kBlockSize = 32;
    constexpr std::int32_t kExpectedStride = 2;  // ceil(64/32)

    // A known scale in log2 space; execute() converts to the RAW softmax
    // scale via raw = scaleLog2 * ln2 before binding slot 10. ln2 is
    // spelled out locally to mirror SdpaFwdPlan.cpp (POSIX-only M_LN2
    // avoided).
    constexpr float kScaleLog2 = 1.5f;
    constexpr float kLn2 = 0.69314718055994530942f;
    const float kExpectedRawScale = kScaleLog2 * kLn2;

    // Tensor UIDs match the production graph contract: q=1, k=2, v=3, o=4.
    SdpaFwdPlan plan(module, /*qUid=*/1, /*kUid=*/2, /*vUid=*/3, /*oUid=*/4, kScaleLog2, kSeqlenQ,
                     kSeqlenK, /*strideQToken=*/0, /*strideQHead=*/0, /*strideKToken=*/0,
                     /*strideKHead=*/0, /*strideVToken=*/0, /*strideVHead=*/0, /*strideOToken=*/0,
                     /*strideOHead=*/0, kBatch, kBlockSize, /*isPaged=*/false, /*isVarlen=*/false,
                     /*useSinks=*/false, /*sinkUid=*/-1);

    // Output holds the fingerprint: 8 int32 elements. Query is read at
    // element 0 to prove pointer slot 1 is bound; seed it with a sentinel.
    constexpr std::int32_t kFingerprintElems = 8;
    constexpr std::int32_t kQuerySentinel = 0x515ABCDE;  // arbitrary known i32

    void* dO = nullptr;
    void* dQ = nullptr;
    void* dK = nullptr;
    void* dV = nullptr;
    ASSERT_EQ(hipMalloc(&dO, sizeof(std::int32_t) * kFingerprintElems), hipSuccess);
    ASSERT_EQ(hipMalloc(&dQ, sizeof(std::int32_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dK, sizeof(std::int32_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dV, sizeof(std::int32_t)), hipSuccess);

    ASSERT_EQ(hipMemset(dO, 0xff, sizeof(std::int32_t) * kFingerprintElems),
              hipSuccess);  // sentinel: did the launch write?
    std::int32_t querySeed = kQuerySentinel;
    ASSERT_EQ(hipMemcpy(dQ, &querySeed, sizeof(querySeed), hipMemcpyHostToDevice), hipSuccess);

    std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers = {
        {1, dQ},
        {2, dK},
        {3, dV},
        {4, dO},
    };

    // Caller-owned workspace for the three marshalled i32 arrays.
    const std::size_t wsBytes = plan.getWorkspaceSize(handle);
    ASSERT_GT(wsBytes, 0u) << "dense path must report a positive workspace size";
    void* dWorkspace = nullptr;
    ASSERT_EQ(hipMalloc(&dWorkspace, wsBytes), hipSuccess);

    EXPECT_NO_THROW(plan.execute(handle, deviceBuffers.data(),
                                 static_cast<std::uint32_t>(deviceBuffers.size()), dWorkspace));
    ASSERT_EQ(hipStreamSynchronize(handle.getStream()), hipSuccess);

    std::int32_t out[kFingerprintElems] = {0};
    ASSERT_EQ(hipMemcpy(out, dO, sizeof(out), hipMemcpyDeviceToHost), hipSuccess);

    // out[0] = num_seqs            -> proves i32 slot 15 + output slot 0
    EXPECT_EQ(out[0], kBatch) << "num_seqs (slot 15) / output (slot 0) binding";
    // out[1] = block_table_stride  -> proves i32 slot 16
    EXPECT_EQ(out[1], kExpectedStride) << "block_table_stride (slot 16) binding";
    // out[2] = qq_bias_stride_0    -> proves i32 slot 17 (= 0)
    EXPECT_EQ(out[2], 0) << "qq_bias_stride_0 (slot 17) binding";
    // out[3] = bitcast<i32>(scale) -> proves f32 slot 10 carries raw scale
    float gotScale = 0.0f;
    std::memcpy(&gotScale, &out[3], sizeof(gotScale));
    EXPECT_FLOAT_EQ(gotScale, kExpectedRawScale)
        << "scale (slot 10) must be the RAW softmax scale = scaleLog2 * ln2";
    // out[5] = query_ptr[0]        -> proves pointer slot 1 bound to Q
    EXPECT_EQ(out[5], kQuerySentinel) << "query_ptr (slot 1) binding";
    // out[6] = block_tables_ptr[0] -> proves pointer slot 5 bound to the
    //          uploaded degenerate table (first physical block id = 0)
    EXPECT_EQ(out[6], 0) << "block_tables_ptr (slot 5) binding + upload";
    // out[7] = query_start_len[0]  -> proves pointer slot 9 bound to the
    //          uploaded cu_seqlens_q (prefix sum starts at 0)
    EXPECT_EQ(out[7], 0) << "query_start_len_ptr (slot 9) binding + upload";

    EXPECT_EQ(hipFree(dO), hipSuccess);
    EXPECT_EQ(hipFree(dQ), hipSuccess);
    EXPECT_EQ(hipFree(dK), hipSuccess);
    EXPECT_EQ(hipFree(dV), hipSuccess);
    EXPECT_EQ(hipFree(dWorkspace), hipSuccess);
}

}  // namespace
