// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>
#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/TensorView.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::conv;
using namespace hip_kernel_provider::test_utilities;

namespace
{

struct SdpaBwdTestCase
{
    SdpaBwdTestCase(std::vector<int64_t> qDimsIn,
                    std::vector<int64_t> vDimsIn,
                    std::optional<float> attnScaleValueIn = std::nullopt,
                    int64_t leftBoundIn = -1,
                    int64_t rightBoundIn = -1,
                    bool topLeftAlignmentIn = true)
        : qDims(std::move(qDimsIn))
        , vDims(std::move(vDimsIn))
        , qStrides(generateStrides(qDims))
        , vStrides(generateStrides(vDims))
        , attnScaleValue(attnScaleValueIn)
        , leftBound(leftBoundIn)
        , rightBound(rightBoundIn)
        , topLeftAlignment(topLeftAlignmentIn)
    {
        // K tensor is [B, H_kv, S_kv, D_qk]: B and D_qk from Q, H_kv and S_kv from V
        kDims = {qDims[0], vDims[1], vDims[2], qDims[3]};
        kStrides = generateStrides(kDims);
    }

    // Non-packed layouts the kernels address: Q, K, V and dO stored [B, S, H, D],
    // and O and dQ rows padded to one element past their head dim.
    SdpaBwdTestCase& withNonPackedLayouts()
    {
        // Stride order (lower = tighter) for [B, H, S, D] dims stored as [B, S, H, D].
        const std::vector<int64_t> bshdOrder = {3, 1, 2, 0};
        qStrides = generateStrides(qDims, bshdOrder);
        kStrides = generateStrides(kDims, bshdOrder);
        vStrides = generateStrides(vDims, bshdOrder);
        doStrides = generateStrides(oDims(), bshdOrder);
        const auto paddedRows = [&](int64_t headDim) {
            const int64_t row = headDim + 1;
            return std::vector<int64_t>{qDims[1] * qDims[2] * row, qDims[2] * row, row, 1};
        };
        oStrides = paddedRows(vDims[3]);
        dqStrides = paddedRows(qDims[3]);
        return *this;
    }

    // dO alone stored [B, S, H, D]; every other tensor stays packed. The gfx942
    // kernels without sequence padding (pssk = 0) assume dO shares Q's strides,
    // so this layout must run the padded (psskddv) kernel instead.
    SdpaBwdTestCase& withBshdDo()
    {
        doStrides = generateStrides(oDims(), {3, 1, 2, 0});
        return *this;
    }

    // Run the case only on these architectures; the others skip it.
    SdpaBwdTestCase& onlyOn(std::vector<std::string> archsIn)
    {
        archs = std::move(archsIn);
        return *this;
    }

    // Build the plan on the ASM SDPA engine with the a16 accumulator knob
    // instead of letting the heuristics pick the engine and its defaults.
    SdpaBwdTestCase& withA16()
    {
        a16 = true;
        return *this;
    }

    // Grade this case at this absolute tolerance instead of its path's. Needed where the
    // reference values stay so far below the path's absolute tolerance that an all-zero
    // output would pass.
    SdpaBwdTestCase& withAbsoluteTolerance(float absoluteToleranceIn)
    {
        absoluteTolerance = absoluteToleranceIn;
        return *this;
    }

    // O and dO: Q's batch, heads and sequence with V's head dim.
    std::vector<int64_t> oDims() const
    {
        return {qDims[0], qDims[1], qDims[2], vDims[3]};
    }

    std::vector<int64_t> qDims;
    std::vector<int64_t> kDims;
    std::vector<int64_t> vDims;
    std::vector<int64_t> qStrides;
    std::vector<int64_t> kStrides;
    std::vector<int64_t> vStrides;
    // Empty: packed strides for O's dims (dO takes O's strides).
    std::vector<int64_t> oStrides;
    std::vector<int64_t> doStrides;
    std::vector<int64_t> dqStrides;

    std::optional<float> attnScaleValue;
    int64_t leftBound;
    int64_t rightBound;
    bool topLeftAlignment;

    // Empty: every architecture the engine serves.
    std::vector<std::string> archs;

    bool a16 = false;

    // Empty: the path's absolute tolerance.
    std::optional<float> absoluteTolerance;
};

// gfx942 serves 64 <= D <= 192 with D_qk == D_v; gfx950 serves 64 < D <= 128
// with D_qk == D_v, and D_qk = 192 with D_v = 128 (AITER v0.1.29 aiter/ops/mha.py
// can_impl_fmha_v3_bwd and can_impl_fmha_v3_bwd_gfx950).
const std::vector<std::string> K_GFX942_ONLY = {"gfx942"};
const std::vector<std::string> K_GFX950_ONLY = {"gfx950"};

// Shapes stay small because the CPU backward reference costs O(B*H*S_q*S_kv*D).

// Quick tier: one case per engine path, at the cheapest shape that reaches it.
// getSdpaBwdComprehensiveTestCases() covers the larger shapes and the rest of
// the mask, layout and head-dim combinations.
std::vector<SdpaBwdTestCase> getSdpaBwdSmokeTestCases()
{
    return {
        // hd128 A32 psskddv on gfx942 (S = 256 is not a multiple of the 192-row
        // K/V tile) and hd128 A32 on gfx950: no mask (mask=0), top-left causal
        // (mask=1), bottom-right causal (mask=2).
        SdpaBwdTestCase({1, 1, 256, 128}, {1, 1, 256, 128}),
        SdpaBwdTestCase({1, 1, 256, 128}, {1, 1, 256, 128}, std::nullopt, -1, 0, true),
        SdpaBwdTestCase({1, 1, 256, 128}, {1, 1, 256, 128}, std::nullopt, -1, 0, false),
        // Bottom-right sliding window: mask=3 on gfx942, mask=4 on gfx950.
        SdpaBwdTestCase({1, 1, 256, 128}, {1, 1, 256, 128}, std::nullopt, 64, 64, false),
        // S_q = 1: the gfx950 hd128 kernels carry a seqlen_q == 1 buffer-range fix.
        // With one query row the reference |dQ| and |dK| stay near 0.01, so the case
        // tightens atol to keep an all-zero dQ or dK failing.
        SdpaBwdTestCase({1, 1, 1, 128}, {1, 1, 256, 128}).withAbsoluteTolerance(2e-3f),
        // The a16 accumulator: the gfx942 A16 rows without sequence padding
        // (S_q == S_kv, S_kv % 64 == 0, packed strides) and the gfx950 A16
        // psskddv rows with the dq_shuffle post kernel.
        SdpaBwdTestCase({1, 1, 256, 128}, {1, 1, 256, 128}).withA16(),
        // A padded head dim: D = 72 runs the hd128 kernels on both architectures.
        SdpaBwdTestCase({1, 1, 256, 72}, {1, 1, 256, 72}),
        // D_qk = 192 with D_v = 128 on gfx950.
        SdpaBwdTestCase({1, 1, 256, 192}, {1, 1, 256, 128}).onlyOn(K_GFX950_ONLY),
        // S_q == S_kv on a multiple of the gfx942 192-row K/V tile: packed strides
        // run the dqdkdv kernel without sequence padding (pssk = 0); dO stored
        // [B, S, H, D] falls back to the psskddv kernel.
        SdpaBwdTestCase({1, 1, 384, 128}, {1, 1, 384, 128}).onlyOn(K_GFX942_ONLY),
        SdpaBwdTestCase({1, 1, 384, 128}, {1, 1, 384, 128}).withBshdDo().onlyOn(K_GFX942_ONLY),
    };
}

// Comprehensive and full tiers: every case getSdpaBwdSmokeTestCases() leaves out.
std::vector<SdpaBwdTestCase> getSdpaBwdComprehensiveTestCases()
{
    return {
        // SLIDING_WINDOW / SWA (mask=3): top-left alignment
        SdpaBwdTestCase({1, 1, 256, 128}, {1, 1, 256, 128}, std::nullopt, 64, 64, true),
        // Asymmetric Sq != Skv — no mask
        SdpaBwdTestCase({1, 1, 256, 128}, {1, 1, 512, 128}),
        // Asymmetric Sq != Skv — top-left causal
        SdpaBwdTestCase({1, 1, 256, 128}, {1, 1, 512, 128}, std::nullopt, -1, 0, true),
        // Asymmetric Sq != Skv — bottom-right causal
        SdpaBwdTestCase({1, 1, 256, 128}, {1, 1, 512, 128}, std::nullopt, -1, 0, false),
        // Asymmetric Sq != Skv — sliding windows. Unlike Sq == Skv, the alignment
        // moves the window, and on gfx950 each alignment has its own kernel
        // (mask 3 top-left, mask 4 bottom-right) that takes the raw window sizes.
        SdpaBwdTestCase({1, 1, 256, 128}, {1, 1, 512, 128}, std::nullopt, 64, 64, true),
        SdpaBwdTestCase({1, 1, 256, 128}, {1, 1, 512, 128}, std::nullopt, 64, 64, false),
        // One-sided bottom-right window: the right bound is unbounded (-1).
        SdpaBwdTestCase({1, 1, 256, 128}, {1, 1, 512, 128}, std::nullopt, 64, -1, false),
        // Small S_q (1 and 16). AITER's Python layer sends these to CK for speed,
        // so the ASM kernels see them less often. S_q = 1 tightens atol as the
        // Smoke case does.
        SdpaBwdTestCase({1, 1, 1, 128}, {1, 1, 256, 128}, std::nullopt, -1, 0, false)
            .withAbsoluteTolerance(2e-3f),
        SdpaBwdTestCase({1, 1, 16, 128}, {1, 1, 256, 128}),
        SdpaBwdTestCase({1, 1, 16, 128}, {1, 1, 16, 128}, std::nullopt, -1, 0, true),
        // B=2 and H=2, so every batch and head stride, and the packed LSE/D
        // layout that buildPlan derives from Q's dims, reach the kernels.
        SdpaBwdTestCase({2, 2, 256, 128}, {2, 2, 256, 128}),
        SdpaBwdTestCase({2, 2, 256, 128}, {2, 2, 256, 128}, std::nullopt, -1, 0, true),
        // Bottom-right window with Sq != Skv (mask 4 on gfx950).
        SdpaBwdTestCase({2, 2, 256, 128}, {2, 2, 512, 128}, std::nullopt, 64, 64, false),
        // The same with non-packed layouts that isApplicable accepts.
        SdpaBwdTestCase({2, 2, 256, 128}, {2, 2, 256, 128}).withNonPackedLayouts(),
        SdpaBwdTestCase({2, 2, 256, 128}, {2, 2, 256, 128}, std::nullopt, -1, 0, true)
            .withNonPackedLayouts(),
        // Head dims other than 128. Padded ones (72, 96, 136) run the next
        // registry head dim's kernel with the graph's head dim in the kernargs.
        SdpaBwdTestCase({1, 2, 256, 64}, {1, 2, 256, 64}).onlyOn(K_GFX942_ONLY),
        SdpaBwdTestCase({1, 2, 256, 64}, {1, 2, 512, 64}, std::nullopt, -1, 0, false)
            .onlyOn(K_GFX942_ONLY),
        SdpaBwdTestCase({1, 2, 256, 72}, {1, 2, 256, 72}),
        SdpaBwdTestCase({1, 2, 256, 96}, {1, 2, 256, 96}),
        SdpaBwdTestCase({1, 2, 256, 96}, {1, 2, 256, 96}, std::nullopt, -1, 0, true),
        SdpaBwdTestCase({1, 2, 256, 96}, {1, 2, 256, 96}, std::nullopt, 64, 64, false),
        SdpaBwdTestCase({1, 2, 256, 136}, {1, 2, 256, 136}).onlyOn(K_GFX942_ONLY),
        SdpaBwdTestCase({1, 2, 256, 192}, {1, 2, 256, 192}).onlyOn(K_GFX942_ONLY),
        SdpaBwdTestCase({1, 2, 256, 192}, {1, 2, 256, 192}, std::nullopt, -1, 0, true)
            .onlyOn(K_GFX942_ONLY),
        // D_qk = 192 with D_v = 128: the hd128 odo, the 192 x 128 dqdkdv and the
        // hd192 dq_convert kernels.
        SdpaBwdTestCase({1, 2, 256, 192}, {1, 2, 256, 128}).onlyOn(K_GFX950_ONLY),
        SdpaBwdTestCase({1, 2, 256, 192}, {1, 2, 512, 128}, std::nullopt, -1, 0, false)
            .onlyOn(K_GFX950_ONLY),
        // S_q == S_kv on a multiple of the gfx942 192-row K/V tile: with packed
        // strides gfx942 runs the dqdkdv kernels without sequence padding
        // (pssk = 0); with dO stored [B, S, H, D] it keeps the padded kernels.
        SdpaBwdTestCase({1, 2, 384, 128}, {1, 2, 384, 128}),
        SdpaBwdTestCase({1, 2, 384, 128}, {1, 2, 384, 128}, std::nullopt, -1, 0, true),
        SdpaBwdTestCase({1, 2, 384, 128}, {1, 2, 384, 128}).withBshdDo(),
        SdpaBwdTestCase({1, 2, 384, 128}, {1, 2, 384, 128}, std::nullopt, -1, 0, true).withBshdDo(),
        // The a16 accumulator. gfx942 ships only A16 rows without sequence
        // padding (S_q == S_kv, S_kv % 64 == 0, packed strides) for D <= 128,
        // no mask or causal, and writes dQ directly; a bottom-right causal mask
        // runs the top-left kernel (AITER mha_bwd.cu:374-377).
        SdpaBwdTestCase({1, 2, 256, 128}, {1, 2, 256, 128}).withA16().onlyOn(K_GFX942_ONLY),
        SdpaBwdTestCase({2, 2, 256, 128}, {2, 2, 256, 128}).withA16().onlyOn(K_GFX942_ONLY),
        SdpaBwdTestCase({1, 2, 256, 128}, {1, 2, 256, 128}, std::nullopt, -1, 0, true)
            .withA16()
            .onlyOn(K_GFX942_ONLY),
        SdpaBwdTestCase({1, 2, 256, 128}, {1, 2, 256, 128}, std::nullopt, -1, 0, false)
            .withA16()
            .onlyOn(K_GFX942_ONLY),
        SdpaBwdTestCase({1, 2, 256, 64}, {1, 2, 256, 64}).withA16().onlyOn(K_GFX942_ONLY),
        SdpaBwdTestCase({1, 2, 256, 96}, {1, 2, 256, 96}, std::nullopt, -1, 0, true)
            .withA16()
            .onlyOn(K_GFX942_ONLY),
        // gfx950 accumulates into a 16-bit dq_acc of ceil16(S_q) rows and runs
        // the dq_shuffle post kernel (the bf16 rows, also for fp16).
        SdpaBwdTestCase({2, 2, 256, 128}, {2, 2, 256, 128}).withA16().onlyOn(K_GFX950_ONLY),
        SdpaBwdTestCase({1, 2, 200, 128}, {1, 2, 256, 128}).withA16().onlyOn(K_GFX950_ONLY),
        SdpaBwdTestCase({1, 2, 256, 128}, {1, 2, 256, 128}, std::nullopt, -1, 0, true)
            .withA16()
            .onlyOn(K_GFX950_ONLY),
        SdpaBwdTestCase({1, 2, 256, 128}, {1, 2, 512, 128}, std::nullopt, -1, 0, false)
            .withA16()
            .onlyOn(K_GFX950_ONLY),
        SdpaBwdTestCase({1, 2, 256, 128}, {1, 2, 256, 128}, std::nullopt, 64, 64, true)
            .withA16()
            .onlyOn(K_GFX950_ONLY),
        SdpaBwdTestCase({2, 2, 256, 128}, {2, 2, 512, 128}, std::nullopt, 64, 64, false)
            .withA16()
            .onlyOn(K_GFX950_ONLY),
        SdpaBwdTestCase({1, 2, 256, 96}, {1, 2, 256, 96}).withA16().onlyOn(K_GFX950_ONLY),
        SdpaBwdTestCase({1, 2, 256, 192}, {1, 2, 256, 128}).withA16().onlyOn(K_GFX950_ONLY),
        SdpaBwdTestCase({1, 2, 256, 192}, {1, 2, 512, 128}, std::nullopt, -1, 0, false)
            .withA16()
            .onlyOn(K_GFX950_ONLY),
        // dq_shuffle writes dQ through its own strides.
        SdpaBwdTestCase({2, 2, 256, 128}, {2, 2, 256, 128})
            .withNonPackedLayouts()
            .withA16()
            .onlyOn(K_GFX950_ONLY),
        // Top-left causal with S_q != S_kv, where the top-left and the bottom-right
        // kernels differ.
        SdpaBwdTestCase({1, 1, 128, 192}, {1, 1, 256, 128}, std::nullopt, -1, 0, true)
            .onlyOn(K_GFX950_ONLY),
        SdpaBwdTestCase({1, 1, 128, 192}, {1, 1, 256, 128}, std::nullopt, -1, 0, true)
            .withA16()
            .onlyOn(K_GFX950_ONLY),
        SdpaBwdTestCase({1, 1, 128, 64}, {1, 1, 256, 64}, std::nullopt, -1, 0, true)
            .onlyOn(K_GFX942_ONLY),
        // gfx942 A16 causal at hd64; its A16 rows need S_q == S_kv, a multiple of 64.
        SdpaBwdTestCase({1, 1, 128, 64}, {1, 1, 128, 64}, std::nullopt, -1, 0, true)
            .withA16()
            .onlyOn(K_GFX942_ONLY),
        // A padded head dim on the gfx942 A16 no-mask rows (pddv = 1).
        SdpaBwdTestCase({1, 1, 128, 72}, {1, 1, 128, 72}).withA16().onlyOn(K_GFX942_ONLY),
        // gfx942 hd192 bottom-right causal with S_q != S_kv.
        SdpaBwdTestCase({1, 1, 128, 192}, {1, 1, 256, 192}, std::nullopt, -1, 0, false)
            .onlyOn(K_GFX942_ONLY),
        // ALMIOPEN-2079: Re-enable when GQA support is implemented
        // // GQA: 4 Q heads, 1 KV head — no mask
        // SdpaBwdTestCase({1, 4, 256, 128}, {1, 1, 256, 128}),
        // // GQA: 4 Q heads, 1 KV head — top-left causal
        // SdpaBwdTestCase({1, 4, 256, 128}, {1, 1, 256, 128}, std::nullopt, -1, 0, true),
    };
}

// The tolerances an output is graded at: an element passes when
// |gpu - ref| <= absolute + relative * |ref|.
struct SdpaBwdTolerance
{
    float absolute;
    float relative;
};

// Grades one output as the default validator does, and prints one line that records
// how far inside its tolerance the output lands:
//   SDPA_BWD_ERROR <test> <arch> <a32|a16> <output> max_abs=.. ref_at_max=.. max_ref=..
//   worst_ratio=.. atol=.. rtol=..
// max_abs is the largest |gpu - ref|, ref_at_max the |ref| at that element, max_ref the
// largest |ref|, and worst_ratio the largest |gpu - ref| / (atol + rtol * |ref|); the
// output passes while worst_ratio stays at or below 1.
template <typename DataType>
class RecordingSdpaBwdValidator : public IReferenceValidation
{
public:
    RecordingSdpaBwdValidator(std::string label, SdpaBwdTolerance tolerance)
        : _label(std::move(label))
        , _tolerance(tolerance)
        , _validator(tolerance.absolute, tolerance.relative)
    {
    }

    bool allClose(ITensor& reference, ITensor& implementation) const override
    {
        if(reference.dims() == implementation.dims())
        {
            TensorView<DataType> refView(reference);
            TensorView<DataType> implView(implementation);
            const auto atol = static_cast<double>(_tolerance.absolute);
            const auto rtol = static_cast<double>(_tolerance.relative);
            double maxAbs = 0.0;
            double refAtMax = 0.0;
            double maxRef = 0.0;
            double worstRatio = 0.0;
            iterateAlongDimensions(reference.dims(), [&](const std::vector<int64_t>& indices) {
                const auto want
                    = static_cast<double>(static_cast<float>(refView.getHostValue(indices)));
                const auto got
                    = static_cast<double>(static_cast<float>(implView.getHostValue(indices)));
                const double error = std::fabs(got - want);
                const double wantMagnitude = std::fabs(want);
                if(error > maxAbs)
                {
                    maxAbs = error;
                    refAtMax = wantMagnitude;
                }
                maxRef = std::max(maxRef, wantMagnitude);
                worstRatio = std::max(worstRatio, error / (atol + rtol * wantMagnitude));
            });
            std::cout << "SDPA_BWD_ERROR " << _label << " max_abs=" << maxAbs
                      << " ref_at_max=" << refAtMax << " max_ref=" << maxRef
                      << " worst_ratio=" << worstRatio << " atol=" << atol << " rtol=" << rtol
                      << '\n';
        }
        return _validator.allClose(reference, implementation);
    }

private:
    std::string _label;
    SdpaBwdTolerance _tolerance;
    CpuFpReferenceValidation<DataType> _validator;
};

template <typename DataType>
class SdpaBackward : public IntegrationGraphVerificationHarness<DataType, SdpaBwdTestCase>
{
protected:
    void initializeBundle(const hipdnn_frontend::graph::Graph& /*graph*/,
                          GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        // Step 1: Randomize Q, K, V and dO, each from its own seed (seed + UID). One seed
        // for every tensor would make same-shaped tensors equal (Q == K == V == dO when
        // S_q == S_kv and D_qk == D_v), which collapses the softmax to the identity and
        // leaves dQ and dK near zero. UIDs match between the GPU and CPU bundles, so both
        // still hold identical inputs.
        for(auto& tensorPair : bundle.tensors)
        {
            bundle.randomizeTensor(tensorPair.first,
                                   _minVal,
                                   _maxVal,
                                   seed + static_cast<unsigned int>(tensorPair.first));
        }

        // Step 2: Compute valid O and stats from Q, K, V using CPU forward reference.
        // The backward pass requires O and stats (LSE) that are mathematically consistent
        // with Q, K, V — random values would cause GPU vs CPU divergence.
        //
        // UIDs are resolved here (not in runGraphTest) because graph.build() —
        // called by verifyGraph() before initializeBundle() — is what assigns
        // UIDs via assignUnsetTensorUids().  Querying get_uid() before build()
        // returns the default (0) for all tensors.
        auto& qTensor = bundle.getTensor(_qAttr->get_uid());
        auto& kTensor = bundle.getTensor(_kAttr->get_uid());
        auto& vTensor = bundle.getTensor(_vAttr->get_uid());
        auto& oTensor = bundle.getTensor(_oAttr->get_uid());
        auto& statsTensor = bundle.getTensor(_statsAttr->get_uid());

        const ShallowTensor<DataType> shallowQ(
            qTensor.rawHostData(), qTensor.dims(), qTensor.strides());
        const ShallowTensor<DataType> shallowK(
            kTensor.rawHostData(), kTensor.dims(), kTensor.strides());
        const ShallowTensor<DataType> shallowV(
            vTensor.rawHostData(), vTensor.dims(), vTensor.strides());
        ShallowTensor<DataType> shallowO(oTensor.rawHostData(), oTensor.dims(), oTensor.strides());
        // Stats (LSE) is [B, H_q, S_q, 1] for the CPU reference
        const std::vector<int64_t> statsDims
            = {qTensor.dims()[0], qTensor.dims()[1], qTensor.dims()[2], 1};
        const std::vector<int64_t> statsStrides = generateStrides(statsDims);
        ShallowTensor<float> shallowStats(statsTensor.rawHostData(), statsDims, statsStrides);

        CpuFpReferenceSdpa::forward<DataType, DataType, DataType, DataType, float>(
            shallowQ,
            shallowK,
            shallowV,
            shallowO,
            SdpaFwdOptions<float>{}
                .setScale(_attnScaleValue)
                .setBand(_leftBound, _rightBound, _topLeftAlignment)
                .setLse(&shallowStats));

        // Mark O and stats as host-modified so the GPU bundle syncs to device
        oTensor.markHostModified();
        statsTensor.markHostModified();
    }

    // Grades dQ, dK and dV at a32Tolerance or, for a case that sets withA16(), at
    // a16Tolerance; a case's withAbsoluteTolerance() overrides the absolute part.
    void runGraphTest(SdpaBwdTolerance a32Tolerance, SdpaBwdTolerance a16Tolerance)
    {
        const auto deviceString = hip_kernel_provider_common::getDeviceString(this->stream());
        if(deviceString != "gfx942" && deviceString != "gfx950")
        {
            GTEST_SKIP() << "Skipped: ASM SDPA backward supports gfx942 and gfx950 only.";
        }

        const SdpaBwdTestCase& testCase = this->GetParam();
        if(!testCase.archs.empty()
           && std::find(testCase.archs.begin(), testCase.archs.end(), deviceString)
                  == testCase.archs.end())
        {
            GTEST_SKIP() << "Skipped: " << deviceString
                         << " backward does not serve this case (see SdpaBwdTestCase::archs).";
        }

        // O has Q's batch, heads and sequence with V's head dim
        const auto oDims = testCase.oDims();
        auto oStrides = testCase.oStrides.empty() ? generateStrides(oDims) : testCase.oStrides;

        // dO has same dims as O, and O's strides unless the test case sets its own
        const auto& doDims = oDims;
        const auto doStrides = testCase.doStrides.empty() ? oStrides : testCase.doStrides;

        // Stats (LSE) dims: [B, H_q, S_q] with contiguous strides
        const std::vector<int64_t> statsDims
            = {testCase.qDims[0], testCase.qDims[1], testCase.qDims[2], 1};
        auto statsStrides = generateStrides(statsDims);

        Graph graph;
        graph.set_io_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT);

        auto q = std::make_shared<TensorAttributes>();
        q->set_dim(testCase.qDims)
            .set_stride(testCase.qStrides)
            .set_data_type(getDataTypeEnumFromType<DataType>());

        auto k = std::make_shared<TensorAttributes>();
        k->set_dim(testCase.kDims)
            .set_stride(testCase.kStrides)
            .set_data_type(getDataTypeEnumFromType<DataType>());

        auto v = std::make_shared<TensorAttributes>();
        v->set_dim(testCase.vDims)
            .set_stride(testCase.vStrides)
            .set_data_type(getDataTypeEnumFromType<DataType>());

        auto o = std::make_shared<TensorAttributes>();
        o->set_dim(oDims).set_stride(oStrides).set_data_type(getDataTypeEnumFromType<DataType>());

        auto dO = std::make_shared<TensorAttributes>();
        dO->set_dim(doDims).set_stride(doStrides).set_data_type(
            getDataTypeEnumFromType<DataType>());

        auto stats = std::make_shared<TensorAttributes>();
        stats->set_dim(statsDims)
            .set_stride(statsStrides)
            .set_data_type(hipdnn_frontend::DataType::FLOAT);

        SdpaBackwardAttributes bwdAttributes;
        bwdAttributes.set_name("SdpaBwdNode");
        if(testCase.attnScaleValue.has_value())
        {
            bwdAttributes.set_attn_scale(testCase.attnScaleValue.value());
        }
        if(testCase.leftBound >= 0)
        {
            bwdAttributes.set_diagonal_band_left_bound(testCase.leftBound);
        }
        if(testCase.rightBound >= 0)
        {
            bwdAttributes.set_diagonal_band_right_bound(testCase.rightBound);
        }
        if(testCase.leftBound >= 0 || testCase.rightBound >= 0)
        {
            bwdAttributes.set_diagonal_alignment(testCase.topLeftAlignment
                                                     ? DiagonalAlignment::TOP_LEFT
                                                     : DiagonalAlignment::BOTTOM_RIGHT);
        }

        auto [dq, dk, dv] = graph.sdpa_backward(q, k, v, o, dO, stats, std::move(bwdAttributes));

        dq->set_output(true).set_data_type(getDataTypeEnumFromType<DataType>());
        if(!testCase.dqStrides.empty())
        {
            dq->set_dim(testCase.qDims).set_stride(testCase.dqStrides);
        }
        dk->set_output(true).set_data_type(getDataTypeEnumFromType<DataType>());
        dv->set_output(true).set_data_type(getDataTypeEnumFromType<DataType>());

        auto validationResult = graph.validate();
        EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();

        // Store tensor attribute pointers — NOT UIDs.  UIDs are assigned by
        // graph.build() (via assignUnsetTensorUids()), which runs inside
        // verifyGraph().  Storing get_uid() here would return 0 for all tensors
        // because validate() does not assign UIDs.  initializeBundle() resolves
        // UIDs lazily via _qAttr->get_uid() after build() has run.
        _qAttr = q;
        _kAttr = k;
        _vAttr = v;
        _oAttr = o;
        _statsAttr = stats;
        _attnScaleValue = testCase.attnScaleValue;
        _leftBound = testCase.leftBound;
        _rightBound = testCase.rightBound;
        _topLeftAlignment = testCase.topLeftAlignment;

        SdpaBwdTolerance tolerance = testCase.a16 ? a16Tolerance : a32Tolerance;
        if(testCase.absoluteTolerance.has_value())
        {
            tolerance.absolute = testCase.absoluteTolerance.value();
        }
        const auto* testInfo = ::testing::UnitTest::GetInstance()->current_test_info();
        const std::string caseLabel = std::string(testInfo->test_suite_name()) + "."
                                      + testInfo->name() + " " + deviceString
                                      + (testCase.a16 ? " a16" : " a32");

        GraphVerificationContext context(graph);
        const auto registerOutput
            = [&](const std::shared_ptr<TensorAttributes>& output, const char* outputName) {
                  this->registerValidator(context,
                                          output,
                                          std::make_unique<RecordingSdpaBwdValidator<DataType>>(
                                              caseLabel + " " + outputName, tolerance));
              };
        ASSERT_NO_FATAL_FAILURE(registerOutput(dq, "dQ"));
        ASSERT_NO_FATAL_FAILURE(registerOutput(dk, "dK"));
        ASSERT_NO_FATAL_FAILURE(registerOutput(dv, "dV"));
        if(!testCase.a16)
        {
            this->verifyGraph(context, 0);
            return;
        }

        ASSERT_NO_FATAL_FAILURE(this->buildPlansForEngine(
            graph,
            ASM_SDPA_ENGINE_ID,
            {KnobSetting("hipkernel.sdpa.bwd.accumulator_type", std::string("a16"))}));
        this->verifyBuiltGraph(context, 0);
    }

    float _minVal = -1.0;
    float _maxVal = 1.0;

private:
    std::shared_ptr<TensorAttributes> _qAttr;
    std::shared_ptr<TensorAttributes> _kAttr;
    std::shared_ptr<TensorAttributes> _vAttr;
    std::shared_ptr<TensorAttributes> _oAttr;
    std::shared_ptr<TensorAttributes> _statsAttr;
    std::optional<float> _attnScaleValue;
    int64_t _leftBound = -1;
    int64_t _rightBound = -1;
    bool _topLeftAlignment = true;
};

using IntegrationGpuSdpaBwdBf16 = SdpaBackward<bfloat16>;

using IntegrationGpuSdpaBwdFp16 = SdpaBackward<hipdnn_data_sdk::types::half>;

} // namespace

TEST_P(IntegrationGpuSdpaBwdBf16, Correctness)
{
    // dQ, dK and dV reach BF16 through these paths:
    // - A32 (gfx942 and gfx950): DQDKDV accumulates dQ in FP32 and dq_convert rounds it
    //   to BF16 once.
    // - A16 on gfx942: DQDKDV writes dQ directly.
    // - A16 on gfx950: DQDKDV accumulates into a 16-bit dq_acc and dq_shuffle writes dQ.
    // The CPU reference accumulates in FP32 and rounds to BF16 once. The kernels also
    // recompute the softmax with their own rounding, so near-zero probabilities differ
    // from the reference's, and the dS = P * (dP - D) subtraction amplifies the
    // difference.
    //
    // PROVISIONAL, pending the gfx942 and gfx950 measurement: atol = rtol = 0.1 on both
    // paths. The values come from a CPU emulation, at scale 1.0 on this suite's
    // seed + UID fill, of a kernel that rounds to 16 bits: its error uses at most 16% of
    // the bound, and an all-zero or halved dQ, dK or dV fails every case. Replace them
    // with values measured from the SDPA_BWD_ERROR lines of a run on both architectures
    // over several seeds, keeping worst_ratio at or below 0.5.
    const SdpaBwdTolerance a32{0.1f, 0.1f};
    const SdpaBwdTolerance a16{0.1f, 0.1f};

    runGraphTest(a32, a16);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuSdpaBwdBf16,
                         testing::ValuesIn(getSdpaBwdSmokeTestCases()));

INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         IntegrationGpuSdpaBwdBf16,
                         testing::ValuesIn(getSdpaBwdComprehensiveTestCases()));

TEST_P(IntegrationGpuSdpaBwdFp16, Correctness)
{
    // FP16 takes the paths the BF16 test lists; its 10-bit mantissa (BF16 has 7) leaves
    // less rounding error on each.
    //
    // PROVISIONAL, pending the gfx942 and gfx950 measurement: atol = rtol = 0.1 on both
    // paths, from the same CPU emulation as BF16 (its error uses at most 16% of the
    // bound). Replace them with measured values as the BF16 test describes.
    const SdpaBwdTolerance a32{0.1f, 0.1f};
    const SdpaBwdTolerance a16{0.1f, 0.1f};

    runGraphTest(a32, a16);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuSdpaBwdFp16,
                         testing::ValuesIn(getSdpaBwdSmokeTestCases()));

INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         IntegrationGpuSdpaBwdFp16,
                         testing::ValuesIn(getSdpaBwdComprehensiveTestCases()));
