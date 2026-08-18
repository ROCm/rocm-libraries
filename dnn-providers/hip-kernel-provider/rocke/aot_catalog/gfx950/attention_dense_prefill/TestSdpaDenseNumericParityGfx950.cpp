// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// GPU numeric-parity test for the AOT catalog engine's gfx950 dense flash-attention
// prefill family. Like the gfx1151 SDPA test it drives the substrate directly
// (Catalog load -> candidate selection -> module load -> LaunchAbi pack/grid via
// CatalogPlan::execute) and compares to a CPU reference softmax(scale.Q.K^T).V,
// bypassing the hipDNN frontend on purpose.
//
// Three contract points differ from the gfx1151 family and are exactly what this
// test is here to pin:
//
//  1. RAW SCALE, NOT scale_log2. The gfx1151 kernel takes a pre-multiplied
//     scale_log2 because its softmax is base-2 and the host does the conversion.
//     This kernel does `qk_scale = scale * log2(e)` INTERNALLY, so the binding is
//     `scale_raw`. Passing a pre-multiplied value would apply log2(e) twice and
//     sharpen the softmax; the ramped scores below make that visible.
//
//  2. PACKED BSHD, NO STRIDE ARGUMENTS. The ABI is five arguments (Q, K, V, O and
//     the f32 scale). The kernel hardcodes stride_token = H*D, stride_head = D,
//     stride_batch = S*H*D, i.e. physical [B, S, H, D]. The host buffers here are
//     laid out that way. (hipDNN's canonical contiguous layout is BHSD, which is
//     why the family constrains the bshd_packed fact.)
//
//  3. CAUSAL, TOP-LEFT, AND GQA. The kernel is causal-only and maps query head h to
//     KV head h / (H / H_kv). The shape used here is H=4, H_kv=1 so the GQA mapping
//     is exercised (every query head shares one KV head) at a size whose CPU
//     reference is cheap.
//
// Geometry is the family's smallest entry: S_q = S_kv = 256 (exactly one BLOCK_M
// query block, so the non-persistent grid is 1 x H x B), D = 128, f16.

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "catalog/Catalog.hpp"
#include "catalog/CatalogTypes.hpp"
#include "core/Handle.hpp"
#include "engines/aot_catalog_engine/AotCatalogTestSupport.hpp"
#include "launch/LaunchAbi.hpp"
#include "launch/ModuleLoader.hpp"
#include "plans/CatalogPlan.hpp"

namespace
{

using namespace aot_catalog_engine;

// Baked in by CMake: the build-tree copy of the catalog (<arch>/<family>/...).
const std::string CATALOG_DIR = aotResolveTestCatalogDir();

// The dense kernel is gfx950-only (supports_attention_dense rejects other arches).
constexpr const char* ARCH = "gfx950";

// Baked kernel geometry -- every one of these is compile-time in the .co, so they
// must match a shipped family.json entry exactly. Batch is compile-time too (it sizes
// the K/V buffer extents), so it is a parameter here and each value must have its own
// shipped .co; the batched case is the one that would silently attend over zeros if a
// B=1 binary were ever selected for B>1.
constexpr int64_t SEQ_Q = 256;
constexpr int64_t SEQ_KV = 256;
constexpr int64_t NUM_HEADS = 4;    // H
constexpr int64_t NUM_KV_HEADS = 1; // H_kv -> gqa_ratio 4
constexpr int64_t HEAD_DIM = 128;   // D

constexpr int64_t GQA = NUM_HEADS / NUM_KV_HEADS;

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
    // gcnArchName looks like "gfx950:sramecc+:xnack-"; match the leading token.
    const std::string name(props.gcnArchName);
    return name.rfind(arch, 0) == 0;
}

// Packed BSHD index: [B, S, H, D] with D innermost.
inline size_t bshd(int64_t b, int64_t s, int64_t h, int64_t d, int64_t heads)
{
    return static_cast<size_t>(((b * SEQ_Q + s) * heads + h) * HEAD_DIM + d);
}
inline size_t bshdKv(int64_t b, int64_t s, int64_t h, int64_t d)
{
    return static_cast<size_t>(((b * SEQ_KV + s) * NUM_KV_HEADS + h) * HEAD_DIM + d);
}

// Inputs shaped so the reference output is a genuinely non-uniform softmax mixture
// of V, not a near-average that would pass even for a broken kernel:
//
//  * One "signal" lane (d == 0) carries a wide ramp across KV tokens, so after the
//    scale the scores span a few units and the softmax is sharply peaked. That makes
//    the raw-vs-log2 scale confusion change the answer instead of hiding in noise.
//  * The ramp direction alternates by head, so the weight mass sits at high j for
//    even heads and low j for odd heads -- a kernel that dropped part of the KV axis
//    (or ignored the causal mask) moves O materially for at least one head.
//  * V is strictly positive with period 7, which does not divide 256, so it has a
//    nonzero KV-axis mean and O is O(1) rather than ~1e-4.
constexpr float SIGNAL_SPAN = 16.0f;

// Every generator is a function of the batch index as well, and not by a mere offset:
// the ramp direction flips with b, so batch 1's answer is not a shifted copy of batch
// 0's. A kernel that addressed the wrong batch slice -- or read past a B=1 buffer
// extent and got zeros -- moves O materially rather than landing inside tolerance.
float qVal(int64_t b, int64_t h, int64_t i, int64_t d)
{
    if(d == 0)
    {
        return 0.5f + 0.5f * static_cast<float>(i) / static_cast<float>(SEQ_Q - 1);
    }
    return 0.0625f * static_cast<float>(((h + i + d + b) % 3) - 1);
}
float kVal(int64_t b, int64_t h, int64_t j, int64_t d)
{
    if(d == 0)
    {
        const float t = static_cast<float>(j) / static_cast<float>(SEQ_KV - 1);
        const bool rising = ((h + b) % 2 == 0);
        return SIGNAL_SPAN * (rising ? t : (1.0f - t));
    }
    return 0.0625f * static_cast<float>(((h + j + d + b) % 5) - 2);
}
float vVal(int64_t b, int64_t h, int64_t j, int64_t d)
{
    return 0.25f + 0.125f * static_cast<float>((j * 3 + d + h + 2 * b) % 7);
}

} // namespace

void runDenseParity(int64_t BATCH)
{
    if(!gpuIsArch(ARCH))
    {
        GTEST_SKIP() << "no " << ARCH << " GPU present";
    }

    const catalog::Catalog cat = catalog::Catalog::loadForDevice(CATALOG_DIR, ARCH);
    if(cat.empty())
    {
        AOT_SKIP_OR_FAIL_ON_EMPTY_CATALOG(CATALOG_DIR);
    }

    // 1. Publish the same problem facts SdpaAdapter::decode would, for this graph.
    //    Every shape axis is baked into the .co, so all of them must match exactly.
    catalog::ProblemShape problem;
    problem.emplace("dtype", catalog::ShapeValue{std::string("f16")});
    problem.emplace("B", catalog::ShapeValue{BATCH});
    problem.emplace("S_q", catalog::ShapeValue{SEQ_Q});
    problem.emplace("S_kv", catalog::ShapeValue{SEQ_KV});
    problem.emplace("H", catalog::ShapeValue{NUM_HEADS});
    problem.emplace("H_kv", catalog::ShapeValue{NUM_KV_HEADS});
    problem.emplace("D", catalog::ShapeValue{HEAD_DIM});
    problem.emplace("gqa_ratio", catalog::ShapeValue{GQA});
    problem.emplace("d_contiguous", catalog::ShapeValue{true});
    problem.emplace("batch_foldable", catalog::ShapeValue{true});
    problem.emplace("bshd_packed", catalog::ShapeValue{true});
    problem.emplace("causal", catalog::ShapeValue{true});
    for(const char* k : {"causal_bottom_right",
                         "has_diagonal_band",
                         "has_mma_core_mode",
                         "has_alibi",
                         "has_padding_mask",
                         "has_attn_mask",
                         "has_block_mask",
                         "has_sink",
                         "has_dropout",
                         "paged",
                         "varlen",
                         "gen_stats",
                         "fp8",
                         "runtime_scale"})
    {
        problem.emplace(k, catalog::ShapeValue{false});
    }

    const std::vector<catalog::Catalog::Candidate> candidates = cat.candidatesFor("sdpa", problem);
    ASSERT_FALSE(candidates.empty()) << "no sdpa candidate for the f16 causal "
                                     << "S=" << SEQ_Q << " H=" << NUM_HEADS << "/" << NUM_KV_HEADS
                                     << " D=" << HEAD_DIM << " problem";
    const catalog::KernelEntry& kernel = *candidates.front().kernel;

    std::optional<launch::HipModuleGuard> module
        = launch::loadKernelModule(kernel.coPath, kernel.symbol);
    ASSERT_TRUE(module.has_value()) << "failed to load " << kernel.coPath;

    // 2. Bind the five-argument ABI. NOTE scale_raw: this kernel multiplies by
    //    log2(e) itself, unlike the gfx1151 family which wants scale_log2.
    const float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

    catalog::LaunchBindings bindings;
    bindings.pointerUids.emplace("Q", 1);
    bindings.pointerUids.emplace("K", 2);
    bindings.pointerUids.emplace("V", 3);
    bindings.pointerUids.emplace("O", 4);
    bindings.scalars.emplace("scale_raw", catalog::ScalarValue{scale});

    launch::SymbolTable gridSymbols;
    gridSymbols.emplace("B", BATCH);
    gridSymbols.emplace("S_q", SEQ_Q);
    gridSymbols.emplace("S_kv", SEQ_KV);
    gridSymbols.emplace("H", NUM_HEADS);
    gridSymbols.emplace("H_kv", NUM_KV_HEADS);
    gridSymbols.emplace("D", HEAD_DIM);

    const auto workspaceBytes
        = static_cast<size_t>(launch::evalWorkspace(kernel.workspace, gridSymbols));

    const CatalogPlan plan(std::move(*module),
                           kernel.launch,
                           std::move(bindings),
                           std::move(gridSymbols),
                           workspaceBytes,
                           kernel.symbol);

    // 3. Host buffers in packed BSHD, plus an f32 causal reference from the SAME
    //    f16-rounded inputs (so the tolerance covers kernel arithmetic only).
    const size_t qElems = static_cast<size_t>(BATCH * SEQ_Q * NUM_HEADS * HEAD_DIM);
    const size_t kvElems = static_cast<size_t>(BATCH * SEQ_KV * NUM_KV_HEADS * HEAD_DIM);

    std::vector<_Float16> hostQ(qElems), hostK(kvElems), hostV(kvElems);
    std::vector<_Float16> hostO(qElems, static_cast<_Float16>(0.0f));
    std::vector<float> reference(qElems, 0.0f);

    for(int64_t b = 0; b < BATCH; ++b)
    {
        for(int64_t h = 0; h < NUM_HEADS; ++h)
        {
            for(int64_t i = 0; i < SEQ_Q; ++i)
            {
                for(int64_t d = 0; d < HEAD_DIM; ++d)
                {
                    hostQ[bshd(b, i, h, d, NUM_HEADS)] = static_cast<_Float16>(qVal(b, h, i, d));
                }
            }
        }
        for(int64_t h = 0; h < NUM_KV_HEADS; ++h)
        {
            for(int64_t j = 0; j < SEQ_KV; ++j)
            {
                for(int64_t d = 0; d < HEAD_DIM; ++d)
                {
                    hostK[bshdKv(b, j, h, d)] = static_cast<_Float16>(kVal(b, h, j, d));
                    hostV[bshdKv(b, j, h, d)] = static_cast<_Float16>(vVal(b, h, j, d));
                }
            }
        }
    }

    std::vector<float> scores(static_cast<size_t>(SEQ_KV));
    for(int64_t b = 0; b < BATCH; ++b)
    for(int64_t h = 0; h < NUM_HEADS; ++h)
    {
        const int64_t hkv = h / GQA; // GQA head mapping
        for(int64_t i = 0; i < SEQ_Q; ++i)
        {
            // Top-left causal: query i attends to keys j <= i (S_q == S_kv here, so
            // top-left and bottom-right coincide).
            const int64_t jMax = i;
            float maxScore = -std::numeric_limits<float>::infinity();
            for(int64_t j = 0; j <= jMax; ++j)
            {
                float dot = 0.0f;
                for(int64_t d = 0; d < HEAD_DIM; ++d)
                {
                    dot += static_cast<float>(hostQ[bshd(b, i, h, d, NUM_HEADS)])
                           * static_cast<float>(hostK[bshdKv(b, j, hkv, d)]);
                }
                scores[static_cast<size_t>(j)] = dot * scale;
                maxScore = std::max(maxScore, scores[static_cast<size_t>(j)]);
            }
            float denom = 0.0f;
            for(int64_t j = 0; j <= jMax; ++j)
            {
                const float e = std::exp(scores[static_cast<size_t>(j)] - maxScore);
                scores[static_cast<size_t>(j)] = e;
                denom += e;
            }
            for(int64_t d = 0; d < HEAD_DIM; ++d)
            {
                float acc = 0.0f;
                for(int64_t j = 0; j <= jMax; ++j)
                {
                    acc += scores[static_cast<size_t>(j)]
                           * static_cast<float>(hostV[bshdKv(b, j, hkv, d)]);
                }
                reference[bshd(b, i, h, d, NUM_HEADS)] = acc / denom;
            }
        }
    }

    // 4. Device buffers + execute through the plan.
    void* dQ = nullptr;
    void* dK = nullptr;
    void* dV = nullptr;
    void* dO = nullptr;
    ASSERT_EQ(hipMalloc(&dQ, hostQ.size() * sizeof(_Float16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dK, hostK.size() * sizeof(_Float16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dV, hostV.size() * sizeof(_Float16)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dO, hostO.size() * sizeof(_Float16)), hipSuccess);
    ASSERT_EQ(hipMemcpy(dQ, hostQ.data(), hostQ.size() * sizeof(_Float16), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dK, hostK.data(), hostK.size() * sizeof(_Float16), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dV, hostV.data(), hostV.size() * sizeof(_Float16), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemset(dO, 0, hostO.size() * sizeof(_Float16)), hipSuccess);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    Handle handle;
    handle.setStream(stream);

    const std::array<hipdnnPluginDeviceBuffer_t, 4> buffers = {{
        {1, dQ},
        {2, dK},
        {3, dV},
        {4, dO},
    }};

    ASSERT_NO_THROW(
        plan.execute(handle, buffers.data(), static_cast<uint32_t>(buffers.size()), nullptr));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(hipMemcpy(hostO.data(), dO, hostO.size() * sizeof(_Float16), hipMemcpyDeviceToHost),
              hipSuccess);

    // 5. Compare. Tolerance covers the P->f16 cast before the second MFMA, the
    //    hardware exp2, and lazy_rescale's approximate re-anchoring (bounded, and
    //    measured parity-identical at ~1.5e-3 on this kernel).
    size_t mismatches = 0;
    for(int64_t b = 0; b < BATCH && mismatches < 10; ++b)
    {
        for(int64_t h = 0; h < NUM_HEADS && mismatches < 10; ++h)
        {
            for(int64_t i = 0; i < SEQ_Q && mismatches < 10; ++i)
            {
                for(int64_t d = 0; d < HEAD_DIM && mismatches < 10; ++d)
                {
                    const size_t idx = bshd(b, i, h, d, NUM_HEADS);
                    const auto got = static_cast<float>(hostO[idx]);
                    const float want = reference[idx];
                    const float tol = std::max(2e-2f, 3e-2f * std::fabs(want));
                    if(std::fabs(got - want) > tol)
                    {
                        ++mismatches;
                        ADD_FAILURE() << "mismatch at b=" << b << " h=" << h << " i=" << i
                                      << " d=" << d << ": got " << got << " want " << want
                                      << " tol " << tol;
                    }
                }
            }
        }
    }

    (void)hipFree(dQ);
    (void)hipFree(dK);
    (void)hipFree(dV);
    (void)hipFree(dO);
    (void)hipStreamDestroy(stream);
}

TEST(TestAotCatalogSdpaDenseNumericParityGfx950, DensePrefillF16CausalMatchesReference)
{
    runDenseParity(1);
}

// The batched case needs its own .co (B is baked into the K/V buffer extents). Without
// it the catalog would either decline -- or, if B were ever loosened to a range, select
// the B=1 binary and read zeros past the descriptor. This test is what makes that
// loud instead of silent.
TEST(TestAotCatalogSdpaDenseNumericParityGfx950, DensePrefillF16CausalBatchedMatchesReference)
{
    runDenseParity(2);
}
