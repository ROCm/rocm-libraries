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

    // Small dense shape chosen so the marshalled arrays have DISTINCT
    // sentinels at the elements the fake kernel reads back -- this gives
    // the test discriminating power against a cross-slot mis-bind. With
    // B=2, Sq=64, Skv=128, block_size=32:
    //   block_table_stride = ceil(128/32) = 4
    //   block_tables  = [0,1,2,3,4,5,6,7]  -> [0]=0, [1]=1
    //   cu_seqlens_q  = [0,64,128]         -> [0]=0, [1]=64 (=Sq)
    //   seqused_k     = [128,128]          -> [0]=128 (=Skv)
    // So block_tables[1]=1, cu_seqlens_q[1]=64 and seq_lens[0]=128 are all
    // distinct from each other AND from the [0]=0 values: a slot-5 <-> slot-9
    // swap (block_tables vs query_start_len) is now detectable even though
    // block_tables[0] == cu_seqlens_q[0] == 0.
    constexpr std::int32_t kBatch = 2;
    constexpr std::int32_t kSeqlenQ = 64;
    constexpr std::int32_t kSeqlenK = 128;
    constexpr std::int32_t kBlockSize = 32;
    constexpr std::int32_t kExpectedStride = 4;  // ceil(128/32)

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

    // Output holds the fingerprint: 11 int32 elements. Query is read at
    // element 0 to prove pointer slot 1 is bound; seed it with a sentinel.
    constexpr std::int32_t kFingerprintElems = 11;
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
    // out[4] = bitcast<i32>(k_scale) -> proves f32 slot 11 carries the
    //          dense-path identity dequant scale (1.0).
    float gotKScale = 0.0f;
    std::memcpy(&gotKScale, &out[4], sizeof(gotKScale));
    EXPECT_FLOAT_EQ(gotKScale, 1.0f) << "k_scale (slot 11) must be the 1.0 identity dequant";
    // out[5] = query_ptr[0]        -> proves pointer slot 1 bound to Q
    EXPECT_EQ(out[5], kQuerySentinel) << "query_ptr (slot 1) binding";
    // out[6] = block_tables_ptr[0] -> proves pointer slot 5 bound to the
    //          uploaded degenerate table (first physical block id = 0)
    EXPECT_EQ(out[6], 0) << "block_tables_ptr[0] (slot 5) binding + upload";
    // out[7] = query_start_len[0]  -> proves pointer slot 9 bound to the
    //          uploaded cu_seqlens_q (prefix sum starts at 0)
    EXPECT_EQ(out[7], 0) << "query_start_len_ptr[0] (slot 9) binding + upload";
    // out[8] = block_tables_ptr[1] -> second element of the degenerate table
    //          is the physical block id 1: distinguishes slot 5 from slot 9.
    EXPECT_EQ(out[8], 1) << "block_tables_ptr[1] (slot 5) -- distinct from slot 9";
    // out[9] = query_start_len[1]  -> second cu_seqlens_q prefix sum is Sq:
    //          distinguishes slot 9 from slot 5 (a 5<->9 swap is detectable).
    EXPECT_EQ(out[9], kSeqlenQ) << "query_start_len_ptr[1] (slot 9) = Sq -- distinct from slot 5";
    // out[10] = seq_lens[0]        -> proves pointer slot 6 (seqused_k) bound;
    //          dense per-sequence seqused_k = Skv.
    EXPECT_EQ(out[10], kSeqlenK) << "seq_lens_ptr[0] (slot 6) = Skv binding + upload";

    EXPECT_EQ(hipFree(dO), hipSuccess);
    EXPECT_EQ(hipFree(dQ), hipSuccess);
    EXPECT_EQ(hipFree(dK), hipSuccess);
    EXPECT_EQ(hipFree(dV), hipSuccess);
    EXPECT_EQ(hipFree(dWorkspace), hipSuccess);
}

}  // namespace
