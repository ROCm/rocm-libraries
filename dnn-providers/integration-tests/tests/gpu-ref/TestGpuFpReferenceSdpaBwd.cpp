// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU SDPA backward reference vs CPU backward reference. O and LSE come from the CPU forward, so
// the inputs are a real attention; GPU outputs start as NaN so an element the kernels skip fails.

#include "ConvShapeCase.hpp" // for gpu_conv_ref_test::assertAllClose
#include <gtest/gtest.h>
#include <hipdnn-gpu-ref/GpuFpReferenceSdpa.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_gpu_ref;
using gpu_conv_ref_test::assertAllClose;

namespace
{

struct BwdCase
{
    const char* name;
    int64_t batch;
    int64_t heads;
    int64_t headsK;
    int64_t headsV;
    int64_t seqQ;
    int64_t seqKv;
    int64_t headDim;
    int64_t headDimV;
    bool useLse;
    std::vector<int64_t> biasDims; // empty = no bias
    bool dBias;
    int64_t leftBound;
    int64_t rightBound;
    bool topLeft;
    std::vector<int32_t> lensQ; // empty = no padding
    std::vector<int32_t> lensKv;
    bool deadRow; // bias masks every key of query row 1
};

std::string caseName(const ::testing::TestParamInfo<BwdCase>& info)
{
    return info.param.name;
}

// fp32 CPU-vs-GPU bound for O, dQ, dK, dV and dBias. Measured worst case on gfx942 across every
// case in this file is 7.2e-7 (|values| up to ~2.7): different accumulation order and FMA
// contraction in the GPU kernels. 1e-4 keeps over 100x margin for cross-architecture device-math
// variation; a gap approaching it is a real divergence.
constexpr float TOL_FP32 = 1e-4f;

} // namespace

class TestGpuSdpaBwdFp32 : public ::testing::TestWithParam<BwdCase>
{
};

TEST_P(TestGpuSdpaBwdFp32, MatchesCpuReference)
{
    SKIP_IF_NO_DEVICES();
    const auto& c = GetParam();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<int64_t> qDims = {c.batch, c.heads, c.seqQ, c.headDim};
    const std::vector<int64_t> kDims = {c.batch, c.headsK, c.seqKv, c.headDim};
    const std::vector<int64_t> vDims = {c.batch, c.headsV, c.seqKv, c.headDimV};
    const std::vector<int64_t> oDims = {c.batch, c.heads, c.seqQ, c.headDimV};

    Tensor<float> q(qDims);
    Tensor<float> k(kDims);
    Tensor<float> v(vDims);
    Tensor<float> o(oDims);
    Tensor<float> dO(oDims);
    Tensor<float> lse({c.batch, c.heads, c.seqQ, 1});
    q.fillWithRandomValues(-1.0f, 1.0f, 201);
    k.fillWithRandomValues(-1.0f, 1.0f, 202);
    v.fillWithRandomValues(-1.0f, 1.0f, 203);
    dO.fillWithRandomValues(-1.0f, 1.0f, 204);

    std::optional<Tensor<float>> bias;
    if(!c.biasDims.empty())
    {
        bias.emplace(c.biasDims);
        bias->fillWithRandomValues(-1.0f, 1.0f, 205);
        if(c.deadRow)
        {
            // Bias [Sq, Skv]-shaped cases only: row 1 sees no key.
            for(int64_t j = 0; j < c.seqKv; ++j)
            {
                std::vector<int64_t> idx(c.biasDims.size(), 0);
                idx[idx.size() - 2] = 1;
                idx[idx.size() - 1] = j;
                bias->setHostValue(-std::numeric_limits<float>::infinity(), idx);
            }
        }
    }

    std::optional<Tensor<int32_t>> seqLenQ;
    std::optional<Tensor<int32_t>> seqLenKv;
    if(!c.lensQ.empty())
    {
        seqLenQ.emplace(std::vector<int64_t>{c.batch, 1, 1, 1});
        seqLenKv.emplace(std::vector<int64_t>{c.batch, 1, 1, 1});
        for(int64_t b = 0; b < c.batch; ++b)
        {
            seqLenQ->setHostValue(c.lensQ[static_cast<size_t>(b)], b, 0, 0, 0);
            seqLenKv->setHostValue(c.lensKv[static_cast<size_t>(b)], b, 0, 0, 0);
            // NaN in every pad: neither reference may read it.
            for(int64_t h = 0; h < c.heads; ++h)
            {
                for(int64_t s = c.lensQ[static_cast<size_t>(b)]; s < c.seqQ; ++s)
                {
                    for(int64_t d = 0; d < c.headDim; ++d)
                    {
                        q.setHostValue(nan, b, h, s, d);
                    }
                    for(int64_t d = 0; d < c.headDimV; ++d)
                    {
                        dO.setHostValue(nan, b, h, s, d);
                    }
                }
            }
            for(int64_t s = c.lensKv[static_cast<size_t>(b)]; s < c.seqKv; ++s)
            {
                for(int64_t h = 0; h < c.headsK; ++h)
                {
                    for(int64_t d = 0; d < c.headDim; ++d)
                    {
                        k.setHostValue(nan, b, h, s, d);
                    }
                }
                for(int64_t h = 0; h < c.headsV; ++h)
                {
                    for(int64_t d = 0; d < c.headDimV; ++d)
                    {
                        v.setHostValue(nan, b, h, s, d);
                    }
                }
            }
        }
    }
    auto* biasPtr = bias.has_value() ? &bias.value() : nullptr;
    auto* seqQPtr = seqLenQ.has_value() ? &seqLenQ.value() : nullptr;
    auto* seqKvPtr = seqLenKv.has_value() ? &seqLenKv.value() : nullptr;

    CpuFpReferenceSdpa::forward(q,
                                k,
                                v,
                                o,
                                SdpaFwdOptions<float>{}
                                    .setScale(0.4f)
                                    .setMask(static_cast<const TensorBase<float>*>(biasPtr))
                                    .setBand(c.leftBound, c.rightBound, c.topLeft)
                                    .setSeqLens(seqQPtr, seqKvPtr)
                                    .setLse(&lse));

    Tensor<float> dQCpu(qDims);
    Tensor<float> dKCpu(kDims);
    Tensor<float> dVCpu(vDims);
    Tensor<float> dQGpu(qDims);
    Tensor<float> dKGpu(kDims);
    Tensor<float> dVGpu(vDims);
    std::optional<Tensor<float>> dBiasCpu;
    std::optional<Tensor<float>> dBiasGpu;
    if(c.dBias)
    {
        dBiasCpu.emplace(c.biasDims);
        dBiasGpu.emplace(c.biasDims);
        dBiasGpu->fillWithValue(nan);
    }
    dQGpu.fillWithValue(nan);
    dKGpu.fillWithValue(nan);
    dVGpu.fillWithValue(nan);
    for(auto* t : {&q, &k, &v, &o, &dO, &lse})
    {
        t->memory().hostData();
    }

    CpuFpReferenceSdpa::backward(q,
                                 k,
                                 v,
                                 o,
                                 dO,
                                 dQCpu,
                                 dKCpu,
                                 dVCpu,
                                 SdpaBwdOptions<float>{}
                                     .setScale(0.4f)
                                     .setLse(c.useLse ? &lse : nullptr)
                                     .setMask(static_cast<const TensorBase<float>*>(biasPtr))
                                     .setDBias(dBiasCpu.has_value() ? &dBiasCpu.value() : nullptr)
                                     .setBand(c.leftBound, c.rightBound, c.topLeft)
                                     .setSeqLens(seqQPtr, seqKvPtr));
    GpuFpReferenceSdpa::bprop(q,
                              k,
                              v,
                              o,
                              dO,
                              dQGpu,
                              dKGpu,
                              dVGpu,
                              GpuSdpaBwdOptions<float>{}
                                  .setScale(0.4f)
                                  .setLse(c.useLse ? &lse : nullptr)
                                  .setMask(static_cast<TensorBase<float>*>(biasPtr))
                                  .setDBias(dBiasGpu.has_value() ? &dBiasGpu.value() : nullptr)
                                  .setBand(c.leftBound, c.rightBound, c.topLeft)
                                  .setSeqLens(seqQPtr, seqKvPtr));

    assertAllClose(dQCpu, dQGpu, TOL_FP32);
    assertAllClose(dKCpu, dKGpu, TOL_FP32);
    assertAllClose(dVCpu, dVGpu, TOL_FP32);
    if(c.dBias)
    {
        assertAllClose(dBiasCpu.value(), dBiasGpu.value(), TOL_FP32);
    }
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestGpuSdpaBwdFp32,
    ::testing::Values(
        BwdCase{"Plain", 2, 4, 4, 4, 7, 11, 16, 16, true, {}, false, -1, -1, true, {}, {}, false},
        BwdCase{"PlainRecompute",
                2,
                4,
                4,
                4,
                7,
                11,
                16,
                16,
                false,
                {},
                false,
                -1,
                -1,
                true,
                {},
                {},
                false},
        BwdCase{"GqaKvHeadsDiffer",
                1,
                8,
                2,
                4,
                5,
                9,
                8,
                8,
                true,
                {},
                false,
                -1,
                -1,
                true,
                {},
                {},
                false},
        BwdCase{"GqaKvHeadsDifferSwapped",
                1,
                8,
                4,
                2,
                5,
                9,
                8,
                8,
                true,
                {},
                false,
                -1,
                -1,
                true,
                {},
                {},
                false},
        BwdCase{"Mqa", 2, 4, 1, 1, 5, 9, 8, 8, true, {}, false, -1, -1, true, {}, {}, false},
        BwdCase{"DqkGtDv", 1, 2, 2, 2, 6, 8, 24, 8, true, {}, false, -1, -1, true, {}, {}, false},
        BwdCase{"DqkLtDv", 1, 2, 2, 2, 6, 8, 8, 24, true, {}, false, -1, -1, true, {}, {}, false},
        BwdCase{"CausalTopLeftSqLtSkv",
                2,
                2,
                2,
                2,
                5,
                9,
                8,
                8,
                true,
                {},
                false,
                -1,
                0,
                true,
                {},
                {},
                false},
        BwdCase{"CausalBottomRightSqLtSkv",
                2,
                2,
                2,
                2,
                5,
                9,
                8,
                8,
                true,
                {},
                false,
                -1,
                0,
                false,
                {},
                {},
                false},
        BwdCase{"CausalBottomRightSqGtSkv",
                2,
                2,
                2,
                2,
                9,
                5,
                8,
                8,
                true,
                {},
                false,
                -1,
                0,
                false,
                {},
                {},
                false},
        BwdCase{"DecodeBottomRight",
                2,
                2,
                2,
                2,
                1,
                7,
                8,
                8,
                true,
                {},
                false,
                -1,
                0,
                false,
                {},
                {},
                false},
        BwdCase{
            "BandBothBounds", 1, 2, 2, 2, 7, 9, 8, 8, true, {}, false, 2, 1, true, {}, {}, false},
        BwdCase{"BandBottomRightRecompute",
                1,
                2,
                2,
                2,
                7,
                9,
                8,
                8,
                false,
                {},
                false,
                3,
                0,
                false,
                {},
                {},
                false},
        BwdCase{"BiasFullDBias",
                2,
                2,
                2,
                2,
                5,
                7,
                8,
                8,
                true,
                {2, 2, 5, 7},
                true,
                -1,
                -1,
                true,
                {},
                {},
                false},
        BwdCase{"BiasPerHeadDBias",
                2,
                2,
                2,
                2,
                5,
                7,
                8,
                8,
                true,
                {1, 2, 5, 7},
                true,
                -1,
                -1,
                true,
                {},
                {},
                false},
        BwdCase{"BiasPerKeyDBias",
                2,
                2,
                2,
                2,
                5,
                7,
                8,
                8,
                true,
                {2, 1, 1, 7},
                true,
                -1,
                -1,
                true,
                {},
                {},
                false},
        BwdCase{"BiasSqSkvDBiasGqa",
                2,
                4,
                2,
                2,
                5,
                7,
                8,
                8,
                true,
                {5, 7},
                true,
                -1,
                0,
                false,
                {},
                {},
                false},
        BwdCase{"BiasScalarDBias",
                1,
                2,
                2,
                2,
                5,
                7,
                8,
                8,
                true,
                {1, 1, 1, 1},
                true,
                -1,
                -1,
                true,
                {},
                {},
                false},
        BwdCase{"DeadRowWithLse",
                1,
                2,
                2,
                2,
                4,
                6,
                8,
                8,
                true,
                {4, 6},
                true,
                -1,
                -1,
                true,
                {},
                {},
                true},
        BwdCase{"DeadRowRecompute",
                1,
                2,
                2,
                2,
                4,
                6,
                8,
                8,
                false,
                {4, 6},
                true,
                -1,
                -1,
                true,
                {},
                {},
                true},
        BwdCase{"PaddingNoMask",
                3,
                2,
                2,
                2,
                9,
                12,
                8,
                8,
                true,
                {},
                false,
                -1,
                -1,
                true,
                {4, 9, 2},
                {12, 5, 7},
                false},
        BwdCase{"PaddingCausalBottomRight",
                3,
                2,
                2,
                2,
                9,
                12,
                8,
                8,
                true,
                {},
                false,
                -1,
                0,
                false,
                {4, 9, 2},
                {12, 5, 7},
                false},
        BwdCase{"PaddingRecomputeTopLeft",
                3,
                2,
                2,
                2,
                9,
                12,
                8,
                8,
                false,
                {},
                false,
                -1,
                0,
                true,
                {4, 9, 2},
                {12, 5, 7},
                false},
        BwdCase{"PaddingBiasDBiasBottomRight",
                3,
                2,
                2,
                2,
                9,
                12,
                8,
                8,
                true,
                {9, 12},
                true,
                -1,
                0,
                false,
                {4, 9, 2},
                {12, 5, 7},
                false}),
    caseName);

// Explicit-mask dropout (cuDNN custom dropout) in backward: GPU vs CPU with GQA, bias + dBias and
// bottom-right causal; with and without LSE and with and without dropout_scale_inv.
// Standard tier: the same CPU-vs-GPU comparison at medium shapes (hundreds of keys, GQA, Dqk != Dv,
// bias with dBias, bands, padding), where accumulation order and long rows matter. Quick runs the
// small cases above on every change.
INSTANTIATE_TEST_SUITE_P(Standard,
                         TestGpuSdpaBwdFp32,
                         ::testing::Values(BwdCase{"MediumGqaCausalBr",
                                                   1,
                                                   8,
                                                   2,
                                                   2,
                                                   320,
                                                   576,
                                                   64,
                                                   64,
                                                   true,
                                                   {},
                                                   false,
                                                   -1,
                                                   0,
                                                   false,
                                                   {},
                                                   {},
                                                   false},
                                           BwdCase{"MediumBiasDbiasWindow",
                                                   1,
                                                   4,
                                                   4,
                                                   4,
                                                   256,
                                                   384,
                                                   64,
                                                   32,
                                                   true,
                                                   {1, 4, 256, 384},
                                                   true,
                                                   96,
                                                   0,
                                                   false,
                                                   {},
                                                   {},
                                                   false},
                                           BwdCase{"MediumPaddedRecompute",
                                                   2,
                                                   4,
                                                   2,
                                                   1,
                                                   288,
                                                   448,
                                                   32,
                                                   32,
                                                   false,
                                                   {},
                                                   false,
                                                   -1,
                                                   -1,
                                                   true,
                                                   {200, 288},
                                                   {448, 301},
                                                   false}),
                         caseName);

struct BwdDropoutCase
{
    const char* name;
    bool useLse;
    bool scaleInv;
    std::vector<int64_t> maskDims;
};

class TestGpuSdpaBwdDropoutFp32 : public ::testing::TestWithParam<BwdDropoutCase>
{
};

TEST_P(TestGpuSdpaBwdDropoutFp32, MatchesCpuReference)
{
    SKIP_IF_NO_DEVICES();
    const auto& c = GetParam();
    const std::vector<int64_t> qDims = {2, 4, 5, 8};
    const std::vector<int64_t> kDims = {2, 2, 7, 8};
    Tensor<float> q(qDims);
    Tensor<float> k(kDims);
    Tensor<float> v(kDims);
    Tensor<float> o(qDims);
    Tensor<float> dO(qDims);
    Tensor<float> lse({2, 4, 5, 1});
    Tensor<float> bias({5, 7});
    Tensor<float> keepMask(c.maskDims);
    q.fillWithRandomValues(-1.0f, 1.0f, 211);
    k.fillWithRandomValues(-1.0f, 1.0f, 212);
    v.fillWithRandomValues(-1.0f, 1.0f, 213);
    dO.fillWithRandomValues(-1.0f, 1.0f, 214);
    bias.fillWithRandomValues(-1.0f, 1.0f, 215);
    for(size_t i = 0; i < keepMask.elementCount(); ++i)
    {
        keepMask.memory().hostData()[i] = (i * 5 % 7 < 4) ? 1.0f : 0.0f;
    }
    constexpr float KEEP = 0.6f;
    const float scale = 1.0f / KEEP;
    const std::optional<float> scaleInv = c.scaleInv ? std::optional<float>(KEEP) : std::nullopt;
    CpuFpReferenceSdpa::forward(q,
                                k,
                                v,
                                o,
                                SdpaFwdOptions<float>{}
                                    .setScale(0.4f)
                                    .setMask(static_cast<const TensorBase<float>*>(&bias))
                                    .setBand(-1, 0, false)
                                    .setDropout(&keepMask, scale)
                                    .setLse(&lse));

    Tensor<float> dQCpu(qDims);
    Tensor<float> dKCpu(kDims);
    Tensor<float> dVCpu(kDims);
    Tensor<float> dBiasCpu({5, 7});
    Tensor<float> dQGpu(qDims);
    Tensor<float> dKGpu(kDims);
    Tensor<float> dVGpu(kDims);
    Tensor<float> dBiasGpu({5, 7});
    for(auto* t : {&dQGpu, &dKGpu, &dVGpu, &dBiasGpu})
    {
        t->fillWithValue(std::numeric_limits<float>::quiet_NaN());
    }
    for(auto* t : {&q, &k, &v, &o, &dO, &lse, &bias, &keepMask})
    {
        t->memory().hostData();
    }
    CpuFpReferenceSdpa::backward(q,
                                 k,
                                 v,
                                 o,
                                 dO,
                                 dQCpu,
                                 dKCpu,
                                 dVCpu,
                                 SdpaBwdOptions<float>{}
                                     .setScale(0.4f)
                                     .setLse(c.useLse ? &lse : nullptr)
                                     .setMask(static_cast<const TensorBase<float>*>(&bias))
                                     .setDBias(&dBiasCpu)
                                     .setBand(-1, 0, false)
                                     .setDropout(&keepMask, scale, scaleInv));
    GpuFpReferenceSdpa::bprop(q,
                              k,
                              v,
                              o,
                              dO,
                              dQGpu,
                              dKGpu,
                              dVGpu,
                              GpuSdpaBwdOptions<float>{}
                                  .setScale(0.4f)
                                  .setLse(c.useLse ? &lse : nullptr)
                                  .setMask(static_cast<TensorBase<float>*>(&bias))
                                  .setDBias(&dBiasGpu)
                                  .setBand(-1, 0, false)
                                  .setDropout(&keepMask, scale, scaleInv));
    assertAllClose(dQCpu, dQGpu, TOL_FP32);
    assertAllClose(dKCpu, dKGpu, TOL_FP32);
    assertAllClose(dVCpu, dVGpu, TOL_FP32);
    assertAllClose(dBiasCpu, dBiasGpu, TOL_FP32);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestGpuSdpaBwdDropoutFp32,
    ::testing::Values(BwdDropoutCase{"LseScaleInvBroadcastBatch", true, true, {1, 4, 5, 7}},
                      BwdDropoutCase{"RecomputeScaleInvFullMask", false, true, {2, 4, 5, 7}},
                      BwdDropoutCase{"LseNoScaleInv", true, false, {2, 1, 5, 7}}),
    [](const ::testing::TestParamInfo<BwdDropoutCase>& info) {
        return std::string(info.param.name);
    });

// Score modifiers: logit softcap (reference-only extension, SEM-020) and cuDNN ALiBi (SEM-021).
// GPU forward and backward vs CPU over 2 K/V heads, with bias + dBias, a causal band and padding.
// The ALiBi case uses 6 query heads, so its slopes are not all powers of two.
struct ScoreModCase
{
    const char* name;
    int64_t heads;
    bool topLeft;
    std::optional<float> softcap;
    bool alibi;
    unsigned seed;
};

class TestGpuSdpaScoreModFp32 : public ::testing::TestWithParam<ScoreModCase>
{
};

TEST_P(TestGpuSdpaScoreModFp32, ForwardAndBackwardMatchCpuReference)
{
    SKIP_IF_NO_DEVICES();
    const auto& c = GetParam();
    const std::vector<int64_t> qDims = {2, c.heads, 6, 8};
    const std::vector<int64_t> kDims = {2, 2, 9, 8};
    constexpr float SCALE = 1.2f;
    Tensor<float> q(qDims);
    Tensor<float> k(kDims);
    Tensor<float> v(kDims);
    Tensor<float> oCpu(qDims);
    Tensor<float> oGpu(qDims);
    Tensor<float> dO(qDims);
    Tensor<float> lse({2, c.heads, 6, 1});
    Tensor<float> bias({6, 9});
    Tensor<int32_t> seqLenQ({2, 1, 1, 1});
    Tensor<int32_t> seqLenKv({2, 1, 1, 1});
    q.fillWithRandomValues(-1.5f, 1.5f, c.seed);
    k.fillWithRandomValues(-1.5f, 1.5f, c.seed + 1);
    v.fillWithRandomValues(-1.0f, 1.0f, c.seed + 2);
    dO.fillWithRandomValues(-1.0f, 1.0f, c.seed + 3);
    bias.fillWithRandomValues(-1.0f, 1.0f, c.seed + 4);
    seqLenQ.setHostValue(6, 0, 0, 0, 0);
    seqLenQ.setHostValue(4, 1, 0, 0, 0);
    seqLenKv.setHostValue(9, 0, 0, 0, 0);
    seqLenKv.setHostValue(7, 1, 0, 0, 0);
    oGpu.fillWithValue(std::numeric_limits<float>::quiet_NaN());
    for(auto* t : {&q, &k, &v, &dO, &bias})
    {
        t->memory().hostData();
    }
    seqLenQ.memory().hostData();
    seqLenKv.memory().hostData();

    CpuFpReferenceSdpa::forward(q,
                                k,
                                v,
                                oCpu,
                                SdpaFwdOptions<float>{}
                                    .setScale(SCALE)
                                    .setMask(static_cast<const TensorBase<float>*>(&bias))
                                    .setBand(-1, 0, c.topLeft)
                                    .setSeqLens(&seqLenQ, &seqLenKv)
                                    .setSoftcap(c.softcap)
                                    .setAlibi(c.alibi)
                                    .setLse(&lse));
    GpuFpReferenceSdpa::fprop(q,
                              k,
                              v,
                              oGpu,
                              GpuSdpaFwdOptions<float>{}
                                  .setScale(SCALE)
                                  .setMask(static_cast<TensorBase<float>*>(&bias))
                                  .setBand(-1, 0, c.topLeft)
                                  .setSeqLens(&seqLenQ, &seqLenKv)
                                  .setSoftcap(c.softcap)
                                  .setAlibi(c.alibi));
    assertAllClose(oCpu, oGpu, TOL_FP32);

    Tensor<float> dQCpu(qDims);
    Tensor<float> dKCpu(kDims);
    Tensor<float> dVCpu(kDims);
    Tensor<float> dBiasCpu({6, 9});
    Tensor<float> dQGpu(qDims);
    Tensor<float> dKGpu(kDims);
    Tensor<float> dVGpu(kDims);
    Tensor<float> dBiasGpu({6, 9});
    for(auto* t : {&dQGpu, &dKGpu, &dVGpu, &dBiasGpu})
    {
        t->fillWithValue(std::numeric_limits<float>::quiet_NaN());
    }
    oCpu.memory().hostData();
    lse.memory().hostData();
    CpuFpReferenceSdpa::backward(q,
                                 k,
                                 v,
                                 oCpu,
                                 dO,
                                 dQCpu,
                                 dKCpu,
                                 dVCpu,
                                 SdpaBwdOptions<float>{}
                                     .setScale(SCALE)
                                     .setLse(&lse)
                                     .setMask(static_cast<const TensorBase<float>*>(&bias))
                                     .setDBias(&dBiasCpu)
                                     .setBand(-1, 0, c.topLeft)
                                     .setSeqLens(&seqLenQ, &seqLenKv)
                                     .setSoftcap(c.softcap)
                                     .setAlibi(c.alibi));
    GpuFpReferenceSdpa::bprop(q,
                              k,
                              v,
                              oCpu,
                              dO,
                              dQGpu,
                              dKGpu,
                              dVGpu,
                              GpuSdpaBwdOptions<float>{}
                                  .setScale(SCALE)
                                  .setLse(&lse)
                                  .setMask(static_cast<TensorBase<float>*>(&bias))
                                  .setDBias(&dBiasGpu)
                                  .setBand(-1, 0, c.topLeft)
                                  .setSeqLens(&seqLenQ, &seqLenKv)
                                  .setSoftcap(c.softcap)
                                  .setAlibi(c.alibi));
    assertAllClose(dQCpu, dQGpu, TOL_FP32);
    assertAllClose(dKCpu, dKGpu, TOL_FP32);
    assertAllClose(dVCpu, dVGpu, TOL_FP32);
    assertAllClose(dBiasCpu, dBiasGpu, TOL_FP32);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestGpuSdpaScoreModFp32,
    ::testing::Values(ScoreModCase{"SoftcapGqaBottomRight", 4, false, 1.5f, false, 221},
                      ScoreModCase{"AlibiGqaTopLeft", 6, true, std::nullopt, true, 231}),
    [](const ::testing::TestParamInfo<ScoreModCase>& info) {
        return std::string(info.param.name);
    });
