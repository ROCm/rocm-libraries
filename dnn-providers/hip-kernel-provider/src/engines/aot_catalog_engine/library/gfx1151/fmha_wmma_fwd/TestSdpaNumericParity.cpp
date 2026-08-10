// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// GPU numeric-parity test for the AOT catalog engine's SDPA (flash-attention
// forward) path. Like the GEMM/RMSNorm parity tests, this drives the *substrate*
// directly (Catalog load -> candidate selection -> module load -> LaunchAbi
// pack/grid via CatalogPlan::execute) against the shipped gfx1151 WMMA
// flash-attention .co, and compares to a CPU reference softmax(scale.Q.K^T).V.
//
// It bypasses the hipDNN frontend on purpose: this exercises exactly the
// catalog/selection/launch-ABI code plus the SdpaAdapter's ABI contract (the
// 15-arg Q,K,V,O + scale_log2 + seqlen_q/k + 8 strides), and the two correctness
// traps the plan calls out -- scale_log2 = scale * log2(e) (base-2 exp2 softmax,
// NOT the raw scale) and the BHSD stride mapping (token = stride[2], head =
// stride[1]; grid y = H, z = B).
//
// The shipped kernel bakes D=64 and H=32 (MHA), so those are exact-match family
// constraints; only the sequence lengths are runtime. The test therefore runs at
// H=32, D=64 with an ASYMMETRIC, tile-crossing S_q=32 / S_kv=48 (each a multiple
// of 16): S_q spans 2 query tiles, S_kv spans 3 key tiles, and S_q != S_kv is the
// self-vs-cross-attention distinction one kernel serves. Getting the KV-seqlen
// stride or the softmax base wrong shows up here.

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "catalog/Catalog.hpp"
#include "catalog/CatalogTypes.hpp"
#include "core/Handle.hpp"
#include "launch/LaunchAbi.hpp"
#include "launch/ModuleLoader.hpp"
#include "plans/CatalogPlan.hpp"

namespace
{

using namespace aot_catalog_engine;

// Baked in by CMake: the build-tree copy of the catalog (<arch>/<family>/...).
constexpr const char* CATALOG_DIR = AOT_CATALOG_TEST_DIR;

// The shipped .co is gfx1151-only; the test is meaningful only on that arch.
constexpr const char* ARCH = "gfx1151";

// Baked kernel geometry (D and H are compile-time facts of the shipped .co).
constexpr int64_t HEAD_DIM = 64; // D
constexpr int64_t NUM_HEADS = 32; // H == H_kv (MHA)
constexpr int64_t SEQ_Q = 32; // 2 query tiles of 16
constexpr int64_t SEQ_KV = 48; // 3 key tiles of 16 (asymmetric on purpose)

bool gpuIsArch(const std::string& arch)
{
    int count = 0;
    if(hipGetDeviceCount(&count) != hipSuccess || count == 0)
    {
        return false;
    }
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, 0) != hipSuccess)
    {
        return false;
    }
    // gcnArchName looks like "gfx1151:sramecc+:xnack-"; match the leading token.
    const std::string name(props.gcnArchName);
    return name.rfind(arch, 0) == 0;
}

// Deterministic small inputs kept in a tight range so the softmax and the D-length
// dot products stay well within f16/bf16 precision.
float qVal(int64_t h, int64_t i, int64_t d)
{
    return static_cast<float>((h * 3 + i * 7 + d * 2) % 5) * 0.125f - 0.25f;
}
float kVal(int64_t h, int64_t j, int64_t d)
{
    return static_cast<float>((h * 5 + j * 3 + d * 2) % 4) * 0.125f - 0.1875f;
}
float vVal(int64_t h, int64_t j, int64_t d)
{
    return static_cast<float>((h * 2 + j * 5 + d * 3) % 6) * 0.125f - 0.3125f;
}

// bf16 host storage: bfloat16 == the top 16 bits of an IEEE f32, so the device
// buffers are just uint16_t byte-for-byte what the kernel reads -- no HIP bf16
// header needed. (Same helpers as the GEMM/RMSNorm parity tests.)
uint16_t floatToBf16(float f)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t roundingBias = ((bits >> 16) & 1u) + 0x7fffu; // round to nearest even
    bits += roundingBias;
    return static_cast<uint16_t>(bits >> 16);
}
float bf16ToFloat(uint16_t b)
{
    const uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// CPU reference: O[h,i,:] = sum_j softmax_j(scale * Q[h,i,:].K[h,j,:]) * V[h,j,:],
// non-causal, no mask. Computed in float from the (already dtype-rounded) inputs.
std::vector<float> referenceAttention(const std::vector<float>& q,
                                      const std::vector<float>& k,
                                      const std::vector<float>& v,
                                      float scale)
{
    std::vector<float> out(static_cast<size_t>(NUM_HEADS * SEQ_Q * HEAD_DIM), 0.0f);
    for(int64_t h = 0; h < NUM_HEADS; ++h)
    {
        for(int64_t i = 0; i < SEQ_Q; ++i)
        {
            std::vector<float> scores(static_cast<size_t>(SEQ_KV), 0.0f);
            float maxScore = -std::numeric_limits<float>::infinity();
            for(int64_t j = 0; j < SEQ_KV; ++j)
            {
                float dot = 0.0f;
                for(int64_t d = 0; d < HEAD_DIM; ++d)
                {
                    dot += q[static_cast<size_t>((h * SEQ_Q + i) * HEAD_DIM + d)]
                           * k[static_cast<size_t>((h * SEQ_KV + j) * HEAD_DIM + d)];
                }
                const float s = dot * scale;
                scores[static_cast<size_t>(j)] = s;
                maxScore = std::max(maxScore, s);
            }
            float denom = 0.0f;
            for(int64_t j = 0; j < SEQ_KV; ++j)
            {
                const float e = std::exp(scores[static_cast<size_t>(j)] - maxScore);
                scores[static_cast<size_t>(j)] = e;
                denom += e;
            }
            for(int64_t d = 0; d < HEAD_DIM; ++d)
            {
                float acc = 0.0f;
                for(int64_t j = 0; j < SEQ_KV; ++j)
                {
                    acc += scores[static_cast<size_t>(j)]
                           * v[static_cast<size_t>((h * SEQ_KV + j) * HEAD_DIM + d)];
                }
                out[static_cast<size_t>((h * SEQ_Q + i) * HEAD_DIM + d)] = acc / denom;
            }
        }
    }
    return out;
}

// Runs the full substrate parity for one dtype. StoreT is the 2-byte device
// element (_Float16 or uint16_t); toStore/fromStore round-trip through float.
template <typename StoreT>
void runSdpaParity(const std::string& dtypeTok,
                   StoreT (*toStore)(float),
                   float (*fromStore)(StoreT),
                   float absTol,
                   float relTol)
{
    const catalog::Catalog cat = catalog::Catalog::loadForDevice(CATALOG_DIR, ARCH);
    if(cat.empty())
    {
        GTEST_SKIP() << "empty AOT catalog at " << CATALOG_DIR
                     << "; build with -DROCKE_PYTHON_DIR to populate it (see the engine README)";
    }

    // Problem carries every key the family.json constrains, so candidatesFor's
    // dtype filter alone disambiguates the f16 vs bf16 sibling family. The
    // universal adapter constrains a full capability vocabulary (satisfies() fails
    // closed on any absent constrained key), so this hand-built shape mirrors what
    // SdpaAdapter::decode would publish for the contiguous, non-causal, MHA gfx1151
    // problem. TestSdpaDecode covers decode itself; this stays substrate-only.
    catalog::ProblemShape problem;
    problem.emplace("dtype", catalog::ShapeValue{dtypeTok});
    problem.emplace("D", catalog::ShapeValue{HEAD_DIM});
    problem.emplace("H", catalog::ShapeValue{NUM_HEADS});
    problem.emplace("H_kv", catalog::ShapeValue{NUM_HEADS});
    problem.emplace("S_q", catalog::ShapeValue{SEQ_Q});
    problem.emplace("S_kv", catalog::ShapeValue{SEQ_KV});
    problem.emplace("gqa_ratio", catalog::ShapeValue{static_cast<int64_t>(1)});
    problem.emplace("d_contiguous", catalog::ShapeValue{true});
    problem.emplace("batch_foldable", catalog::ShapeValue{true});
    problem.emplace("causal", catalog::ShapeValue{false});
    problem.emplace("causal_bottom_right", catalog::ShapeValue{false});
    problem.emplace("has_alibi", catalog::ShapeValue{false});
    problem.emplace("has_padding_mask", catalog::ShapeValue{false});
    problem.emplace("has_attn_mask", catalog::ShapeValue{false});
    problem.emplace("has_block_mask", catalog::ShapeValue{false});
    problem.emplace("has_sink", catalog::ShapeValue{false});
    problem.emplace("has_dropout", catalog::ShapeValue{false});
    problem.emplace("paged", catalog::ShapeValue{false});
    problem.emplace("varlen", catalog::ShapeValue{false});
    problem.emplace("gen_stats", catalog::ShapeValue{false});
    problem.emplace("fp8", catalog::ShapeValue{false});
    problem.emplace("runtime_scale", catalog::ShapeValue{false});

    const std::vector<catalog::Catalog::Candidate> candidates = cat.candidatesFor("sdpa", problem);
    ASSERT_FALSE(candidates.empty()) << "no sdpa candidate for the " << dtypeTok << " problem";
    const catalog::KernelEntry& kernel = *candidates.front().kernel;
    ASSERT_NE(kernel.symbol.find(dtypeTok), std::string::npos)
        << "selected kernel '" << kernel.symbol << "' is not the " << dtypeTok << " family";

    std::optional<launch::HipModuleGuard> module
        = launch::loadKernelModule(kernel.coPath, kernel.symbol);
    ASSERT_TRUE(module.has_value()) << "failed to load " << kernel.coPath;

    // Contiguous BHSD, B=1: stride_token = D (S-axis step), stride_head = S*D.
    const int64_t strideQtoken = HEAD_DIM;
    const int64_t strideQhead = SEQ_Q * HEAD_DIM;
    const int64_t strideKtoken = HEAD_DIM;
    const int64_t strideKhead = SEQ_KV * HEAD_DIM;

    constexpr float SCALE = 0.125f; // 1/sqrt(64)
    constexpr float LOG2E = 1.4426950408889634f;

    catalog::LaunchBindings bindings;
    bindings.pointerUids.emplace("Q", 1);
    bindings.pointerUids.emplace("K", 2);
    bindings.pointerUids.emplace("V", 3);
    bindings.pointerUids.emplace("O", 4);
    bindings.scalars.emplace("scale_log2", catalog::ScalarValue{SCALE * LOG2E});
    bindings.scalars.emplace("seqlen_q", catalog::ScalarValue{SEQ_Q});
    bindings.scalars.emplace("seqlen_k", catalog::ScalarValue{SEQ_KV});
    bindings.scalars.emplace("stride_q_token", catalog::ScalarValue{strideQtoken});
    bindings.scalars.emplace("stride_q_head", catalog::ScalarValue{strideQhead});
    bindings.scalars.emplace("stride_k_token", catalog::ScalarValue{strideKtoken});
    bindings.scalars.emplace("stride_k_head", catalog::ScalarValue{strideKhead});
    bindings.scalars.emplace("stride_v_token", catalog::ScalarValue{strideKtoken});
    bindings.scalars.emplace("stride_v_head", catalog::ScalarValue{strideKhead});
    bindings.scalars.emplace("stride_o_token", catalog::ScalarValue{strideQtoken});
    bindings.scalars.emplace("stride_o_head", catalog::ScalarValue{strideQhead});

    launch::SymbolTable gridSymbols;
    gridSymbols.emplace("S_q", SEQ_Q);
    gridSymbols.emplace("H", NUM_HEADS);
    gridSymbols.emplace("B", static_cast<int64_t>(1));

    const CatalogPlan plan(std::move(*module),
                           kernel.launch,
                           std::move(bindings),
                           std::move(gridSymbols),
                           kernel.workspaceBytes,
                           kernel.symbol);

    const auto qElems = static_cast<size_t>(NUM_HEADS * SEQ_Q * HEAD_DIM);
    const auto kvElems = static_cast<size_t>(NUM_HEADS * SEQ_KV * HEAD_DIM);

    // Build device inputs as dtype, and a float mirror (post-rounding) for the
    // reference so the comparison isolates kernel error, not dtype rounding.
    std::vector<StoreT> hostQ(qElems);
    std::vector<StoreT> hostK(kvElems);
    std::vector<StoreT> hostV(kvElems);
    std::vector<StoreT> hostO(qElems, toStore(0.0f));
    std::vector<float> refQ(qElems);
    std::vector<float> refK(kvElems);
    std::vector<float> refV(kvElems);

    for(int64_t h = 0; h < NUM_HEADS; ++h)
    {
        for(int64_t i = 0; i < SEQ_Q; ++i)
        {
            for(int64_t d = 0; d < HEAD_DIM; ++d)
            {
                const auto idx = static_cast<size_t>((h * SEQ_Q + i) * HEAD_DIM + d);
                hostQ[idx] = toStore(qVal(h, i, d));
                refQ[idx] = fromStore(hostQ[idx]);
            }
        }
        for(int64_t j = 0; j < SEQ_KV; ++j)
        {
            for(int64_t d = 0; d < HEAD_DIM; ++d)
            {
                const auto idx = static_cast<size_t>((h * SEQ_KV + j) * HEAD_DIM + d);
                hostK[idx] = toStore(kVal(h, j, d));
                hostV[idx] = toStore(vVal(h, j, d));
                refK[idx] = fromStore(hostK[idx]);
                refV[idx] = fromStore(hostV[idx]);
            }
        }
    }

    const std::vector<float> reference = referenceAttention(refQ, refK, refV, SCALE);

    void* deviceQ = nullptr;
    void* deviceK = nullptr;
    void* deviceV = nullptr;
    void* deviceO = nullptr;
    ASSERT_EQ(hipMalloc(&deviceQ, hostQ.size() * sizeof(StoreT)), hipSuccess);
    ASSERT_EQ(hipMalloc(&deviceK, hostK.size() * sizeof(StoreT)), hipSuccess);
    ASSERT_EQ(hipMalloc(&deviceV, hostV.size() * sizeof(StoreT)), hipSuccess);
    ASSERT_EQ(hipMalloc(&deviceO, hostO.size() * sizeof(StoreT)), hipSuccess);
    ASSERT_EQ(
        hipMemcpy(deviceQ, hostQ.data(), hostQ.size() * sizeof(StoreT), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpy(deviceK, hostK.data(), hostK.size() * sizeof(StoreT), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpy(deviceV, hostV.data(), hostV.size() * sizeof(StoreT), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(hipMemset(deviceO, 0, hostO.size() * sizeof(StoreT)), hipSuccess);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    Handle handle;
    handle.setStream(stream);

    const std::array<hipdnnPluginDeviceBuffer_t, 4> buffers = {{
        {1, deviceQ},
        {2, deviceK},
        {3, deviceV},
        {4, deviceO},
    }};

    ASSERT_NO_THROW(plan.execute(handle, buffers.data(), 4, nullptr));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    ASSERT_EQ(
        hipMemcpy(hostO.data(), deviceO, hostO.size() * sizeof(StoreT), hipMemcpyDeviceToHost),
        hipSuccess);

    for(int64_t h = 0; h < NUM_HEADS; ++h)
    {
        for(int64_t i = 0; i < SEQ_Q; ++i)
        {
            for(int64_t d = 0; d < HEAD_DIM; ++d)
            {
                const auto idx = static_cast<size_t>((h * SEQ_Q + i) * HEAD_DIM + d);
                const float got = fromStore(hostO[idx]);
                const float want = reference[idx];
                const float tol = std::max(absTol, relTol * std::fabs(want));
                ASSERT_NEAR(got, want, tol)
                    << "mismatch at head " << h << " query " << i << " dim " << d;
            }
        }
    }

    (void)hipFree(deviceQ);
    (void)hipFree(deviceK);
    (void)hipFree(deviceV);
    (void)hipFree(deviceO);
    (void)hipStreamDestroy(stream);
}

_Float16 f16FromFloat(float f)
{
    return static_cast<_Float16>(f);
}
float f16ToFloat(_Float16 h)
{
    return static_cast<float>(h);
}

} // namespace

TEST(TestAotCatalogSdpaNumericParity, WmmaFmhaFwdF16MatchesReference)
{
    if(!gpuIsArch(ARCH))
    {
        GTEST_SKIP() << "no " << ARCH << " GPU present";
    }
    runSdpaParity<_Float16>("f16", f16FromFloat, f16ToFloat, 2e-2f, 2e-2f);
}

TEST(TestAotCatalogSdpaNumericParity, WmmaFmhaFwdBf16MatchesReference)
{
    if(!gpuIsArch(ARCH))
    {
        GTEST_SKIP() << "no " << ARCH << " GPU present";
    }
    // bf16 has a 7-bit mantissa (~2 decimal digits) -> looser tolerance than f16.
    runSdpaParity<uint16_t>("bf16", floatToBf16, bf16ToFloat, 5e-2f, 5e-2f);
}
