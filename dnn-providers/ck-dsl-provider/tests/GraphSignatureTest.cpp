// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "adapters/conv_implicit_gemm/ConvImplicitGemmAdapter.hpp"
#include "adapters/conv_implicit_gemm/ConvImplicitGemmSpec.hpp"
#include "adapters/sdpa/SdpaSpec.hpp"
#include "graph/GraphSignature.hpp"

namespace {

using ck_dsl_provider::ConvImplicitGemmAdapter;
using ck_dsl_provider::ConvImplicitGemmSpec;
using ck_dsl_provider::ConvProblem;
using ck_dsl_provider::GraphSignature;
using ck_dsl_provider::SdpaSpec;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr const char* kOpKind = "conv_implicit_gemm";

// Baseline target arch for the signature tests that don't vary it. arch
// is a separate computeForSpec argument (an orthogonal compile target,
// mirroring the DSL), so the field-perturbation tests below pin it here.
constexpr const char* kArch = "gfx950";

/// Baseline spec for the example conv shape (N=8, H=W=56, C=64, K=64,
/// R=S=3, stride=1, pad=1, dilation=1). Only ``problem`` is set; the
/// codegen knobs keep their example constexpr defaults. The
/// ``computeForSpec`` contract is that *any* field here -- problem or
/// knob -- changes the hash, so the tests below perturb one at a time.
ConvImplicitGemmSpec makeSpec() {
    ConvImplicitGemmSpec spec;
    ConvProblem& p = spec.problem;
    p.N = 8;
    p.Hi = 56;
    p.Wi = 56;
    p.C = 64;
    p.K = 64;
    p.R = 3;
    p.S = 3;
    p.sH = 1;
    p.sW = 1;
    p.pH = 1;
    p.pW = 1;
    p.dH = 1;
    p.dW = 1;
    return spec;
}

TEST(TestGraphSignature, DeterministicForSameSpec) {
    EXPECT_EQ(GraphSignature::computeForSpec(kOpKind, makeSpec(), kArch),
              GraphSignature::computeForSpec(kOpKind, makeSpec(), kArch));
}

TEST(TestGraphSignature, ChangesWithOpKind) {
    auto spec = makeSpec();
    EXPECT_NE(GraphSignature::computeForSpec(kOpKind, spec, kArch),
              GraphSignature::computeForSpec("conv_other_op", spec, kArch));
}

// The target arch must move the hash: the HSACO is arch-specific, so a
// gfx942 build and a gfx950 build of the same shape must land on
// distinct keys (otherwise a multi-arch process would alias them and
// hand back a module for the wrong device). arch is a separate argument
// to computeForSpec, not a spec field (mirroring the DSL).
TEST(TestGraphSignature, ChangesWithArch) {
    EXPECT_NE(GraphSignature::computeForSpec(kOpKind, makeSpec(), "gfx942"),
              GraphSignature::computeForSpec(kOpKind, makeSpec(), "gfx950"))
        << "arch did not affect the signature";
}

// Every one of the 13 ConvProblem fields must move the hash. This is
// the load-bearing coverage: a fold that drops or mis-orders a field
// would let two distinct shapes collide on the same cache key.
TEST(TestGraphSignature, ChangesWithEachProblemField) {
    const auto baseline = GraphSignature::computeForSpec(kOpKind, makeSpec(), kArch);

    const std::vector<std::pair<const char*, std::function<void(ConvProblem&)>>> mutators = {
        {"N", [](ConvProblem& p) { p.N += 1; }},   {"Hi", [](ConvProblem& p) { p.Hi += 1; }},
        {"Wi", [](ConvProblem& p) { p.Wi += 1; }}, {"C", [](ConvProblem& p) { p.C += 1; }},
        {"K", [](ConvProblem& p) { p.K += 1; }},   {"R", [](ConvProblem& p) { p.R += 1; }},
        {"S", [](ConvProblem& p) { p.S += 1; }},   {"sH", [](ConvProblem& p) { p.sH += 1; }},
        {"sW", [](ConvProblem& p) { p.sW += 1; }}, {"pH", [](ConvProblem& p) { p.pH += 1; }},
        {"pW", [](ConvProblem& p) { p.pW += 1; }}, {"dH", [](ConvProblem& p) { p.dH += 1; }},
        {"dW", [](ConvProblem& p) { p.dW += 1; }},
    };

    for (const auto& [name, mutate] : mutators) {
        auto spec = makeSpec();
        mutate(spec.problem);
        EXPECT_NE(GraphSignature::computeForSpec(kOpKind, spec, kArch), baseline)
            << "ConvProblem field '" << name << "' did not affect the signature";
    }
}

// Folding must be position-sensitive: two specs holding the same set of
// values in different fields must hash differently. Swapping N (8) and
// K (64) -- distinct values in distinct fields -- guards against a
// regression that drops a group separator or reorders the folds.
TEST(TestGraphSignature, ProblemFieldsAreNotPositionAliased) {
    auto swapped = makeSpec();
    std::swap(swapped.problem.N, swapped.problem.K);
    EXPECT_NE(GraphSignature::computeForSpec(kOpKind, swapped, kArch),
              GraphSignature::computeForSpec(kOpKind, makeSpec(), kArch));
}

// Every codegen knob must move the hash. M1 never varies these (they
// are constexpr defaults), but M2 autotuning will -- and an autotuned
// kernel that collides with the default-tuned one of the same shape
// would silently load the wrong module. This test pins that contract;
// when autotuning lands it should keep passing, not be deleted.
TEST(TestGraphSignature, ChangesWithEachCodegenKnob) {
    const auto baseline = GraphSignature::computeForSpec(kOpKind, makeSpec(), kArch);

    const std::vector<std::pair<const char*, std::function<void(ConvImplicitGemmSpec&)>>> mutators =
        {
            {"name", [](ConvImplicitGemmSpec& s) { s.name = "other_kernel"; }},
            {"tile_m", [](ConvImplicitGemmSpec& s) { s.tile_m += 1; }},
            {"tile_n", [](ConvImplicitGemmSpec& s) { s.tile_n += 1; }},
            {"tile_k", [](ConvImplicitGemmSpec& s) { s.tile_k += 1; }},
            {"warp_m", [](ConvImplicitGemmSpec& s) { s.warp_m += 1; }},
            {"warp_n", [](ConvImplicitGemmSpec& s) { s.warp_n += 1; }},
            {"warp_tile_m", [](ConvImplicitGemmSpec& s) { s.warp_tile_m += 1; }},
            {"warp_tile_n", [](ConvImplicitGemmSpec& s) { s.warp_tile_n += 1; }},
            {"warp_tile_k", [](ConvImplicitGemmSpec& s) { s.warp_tile_k += 1; }},
            {"wave_size", [](ConvImplicitGemmSpec& s) { s.wave_size += 1; }},
            {"pipeline", [](ConvImplicitGemmSpec& s) { s.pipeline = "other_pipeline"; }},
            {"epilogue", [](ConvImplicitGemmSpec& s) { s.epilogue = "cshuffle"; }},
            {"async_dma", [](ConvImplicitGemmSpec& s) { s.async_dma = !s.async_dma; }},
            {"unroll_k", [](ConvImplicitGemmSpec& s) { s.unroll_k = !s.unroll_k; }},
            {"lds_k_pad", [](ConvImplicitGemmSpec& s) { s.lds_k_pad = 8; }},
            {"chiplet_swizzle",
             [](ConvImplicitGemmSpec& s) { s.chiplet_swizzle = !s.chiplet_swizzle; }},
            {"chiplet_wgm", [](ConvImplicitGemmSpec& s) { s.chiplet_wgm += 1; }},
            {"chiplet_num_xcds", [](ConvImplicitGemmSpec& s) { s.chiplet_num_xcds += 1; }},
            {"chiplet_chunk_size", [](ConvImplicitGemmSpec& s) { s.chiplet_chunk_size += 1; }},
            {"waves_per_eu", [](ConvImplicitGemmSpec& s) { s.waves_per_eu = 4; }},
        };

    for (const auto& [name, mutate] : mutators) {
        auto spec = makeSpec();
        mutate(spec);
        EXPECT_NE(GraphSignature::computeForSpec(kOpKind, spec, kArch), baseline)
            << "codegen knob '" << name << "' did not affect the signature";
    }
}

// An optional knob set to a value must differ from the same knob unset,
// even when the value is 0 -- the presence discriminator, not just the
// payload, has to participate in the fold.
TEST(TestGraphSignature, OptionalKnobPresenceIsDistinctFromZero) {
    auto unset = makeSpec();  // lds_k_pad / waves_per_eu default to nullopt
    auto setZero = makeSpec();
    setZero.lds_k_pad = 0;
    EXPECT_NE(GraphSignature::computeForSpec(kOpKind, setZero, kArch),
              GraphSignature::computeForSpec(kOpKind, unset, kArch));
}

/// Minimal FlatBuffer graph for the example shape so the integration
/// test can drive the real adapter -> signature path. Logical NCHW dims
/// for X/Y and KCRS for W, HALF dtype; strides are required by the
/// schema but unused by the adapter/signature.
struct ConvGraphFixture {
    flatbuffers::FlatBufferBuilder convBuilder;
    flatbuffers::FlatBufferBuilder xBuilder;
    flatbuffers::FlatBufferBuilder wBuilder;
    flatbuffers::FlatBufferBuilder yBuilder;
    const data_objects::ConvolutionFwdAttributes* convAttr{nullptr};
    std::unordered_map<std::int64_t, const data_objects::TensorAttributes*> tensorMap;

    static const data_objects::TensorAttributes* makeTensor(
        flatbuffers::FlatBufferBuilder& builder, std::int64_t uid, const std::string& name,
        const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& strides) {
        auto offset = data_objects::CreateTensorAttributesDirect(
            builder, uid, name.c_str(), data_objects::DataType::HALF, &strides, &dims,
            /*virtual=*/false);
        builder.Finish(offset);
        return flatbuffers::GetRoot<data_objects::TensorAttributes>(builder.GetBufferPointer());
    }

    ConvGraphFixture() {
        tensorMap[1] =
            makeTensor(xBuilder, 1, "X", {8, 64, 56, 56}, {64 * 56 * 56, 1, 56 * 64, 64});
        tensorMap[2] = makeTensor(wBuilder, 2, "W", {64, 64, 3, 3}, {64 * 3 * 3, 1, 3 * 64, 64});
        tensorMap[3] =
            makeTensor(yBuilder, 3, "Y", {8, 64, 56, 56}, {64 * 56 * 56, 1, 56 * 64, 64});

        std::vector<std::int64_t> prePadding{1, 1};
        std::vector<std::int64_t> postPadding{1, 1};
        std::vector<std::int64_t> stride{1, 1};
        std::vector<std::int64_t> dilation{1, 1};
        auto convOffset = data_objects::CreateConvolutionFwdAttributesDirect(
            convBuilder, 1, 2, 3, &prePadding, &postPadding, &stride, &dilation,
            data_objects::ConvMode::CROSS_CORRELATION);
        convBuilder.Finish(convOffset);
        convAttr = flatbuffers::GetRoot<data_objects::ConvolutionFwdAttributes>(
            convBuilder.GetBufferPointer());
    }
};

// End-to-end: the spec the adapter lifts from the example FlatBuffer
// must hash identically to the hand-built ``makeSpec``. This ties the
// adapter's field mapping to the signature through the production path
// and catches drift between the two without an interpreter (buildSpec
// is pure C++).
TEST(TestGraphSignature, MatchesAdapterBuiltSpecForExampleShape) {
    ConvGraphFixture fx;
    ConvImplicitGemmSpec built = ConvImplicitGemmAdapter::buildSpec(*fx.convAttr, fx.tensorMap);
    EXPECT_EQ(GraphSignature::computeForSpec(kOpKind, built, kArch),
              GraphSignature::computeForSpec(kOpKind, makeSpec(), kArch));
}

// ---- SDPA signature coverage --------------------------------------------

constexpr const char* kSdpaOpKind = "sdpa_fmha_fwd";

/// Baseline FMHA-forward spec [B=2, Hq=Hkv=8, Sq=Skv=16, D=64], FP16, no
/// mask, with non-trivial stride/scale launch-time scalars so the
/// "do-not-fold" test below has something to perturb. The contract is
/// that the shape fields plus dtype/mask_mode/name move the hash while
/// the eight stride_* scalars and scale_log2 do NOT.
SdpaSpec makeSdpaSpec() {
    SdpaSpec spec;
    spec.problem.B = 2;
    spec.problem.Hq = 8;
    spec.problem.Hkv = 8;
    spec.problem.Sq = 16;
    spec.problem.Skv = 16;
    spec.problem.D = 64;
    spec.problem.stride_q_token = 512;
    spec.problem.stride_q_head = 64;
    spec.problem.stride_k_token = 512;
    spec.problem.stride_k_head = 64;
    spec.problem.stride_v_token = 512;
    spec.problem.stride_v_head = 64;
    spec.problem.stride_o_token = 512;
    spec.problem.stride_o_head = 64;
    spec.problem.scale_log2 = 0.18033688f;  // (1/sqrt(64)) * log2(e)
    return spec;
}

TEST(TestGraphSignature, SdpaDeterministicForSameSpec) {
    EXPECT_EQ(GraphSignature::computeForSpec(kSdpaOpKind, makeSdpaSpec(), kArch),
              GraphSignature::computeForSpec(kSdpaOpKind, makeSdpaSpec(), kArch));
}

// Every codegen-relevant field -- the six shape ints plus dtype,
// mask_mode, and name -- must move the hash. A fold that drops one would
// let two distinct kernels collide on the same cache key.
TEST(TestGraphSignature, SdpaChangesWithEachShapeField) {
    const auto baseline = GraphSignature::computeForSpec(kSdpaOpKind, makeSdpaSpec(), kArch);

    const std::vector<std::pair<const char*, std::function<void(SdpaSpec&)>>> mutators = {
        {"B", [](SdpaSpec& s) { s.problem.B += 1; }},
        {"Hq", [](SdpaSpec& s) { s.problem.Hq += 1; }},
        {"Hkv", [](SdpaSpec& s) { s.problem.Hkv += 1; }},
        {"Sq", [](SdpaSpec& s) { s.problem.Sq += 1; }},
        {"Skv", [](SdpaSpec& s) { s.problem.Skv += 1; }},
        {"D", [](SdpaSpec& s) { s.problem.D += 1; }},
        {"dtype", [](SdpaSpec& s) { s.dtype = "bf16"; }},
        {"mask_mode", [](SdpaSpec& s) { s.mask_mode = "causal"; }},
        {"name", [](SdpaSpec& s) { s.name = "other_kernel"; }},
    };

    for (const auto& [name, mutate] : mutators) {
        auto spec = makeSdpaSpec();
        mutate(spec);
        EXPECT_NE(GraphSignature::computeForSpec(kSdpaOpKind, spec, kArch), baseline)
            << "SdpaSpec field '" << name << "' did not affect the signature";
    }
}

// The eight stride_* scalars and scale_log2 are launch-time kernel
// arguments, NOT codegen inputs: the compiled kernel + grid are identical
// regardless of stride or scale. They must be DELIBERATELY excluded from
// the signature (folding them would thrash the cache with redundant
// recompiles of a byte-identical kernel). This test pins that non-fold
// contract.
TEST(TestGraphSignature, SdpaStrideAndScaleDoNotChangeSignature) {
    const auto baseline = GraphSignature::computeForSpec(kSdpaOpKind, makeSdpaSpec(), kArch);

    const std::vector<std::pair<const char*, std::function<void(SdpaSpec&)>>> mutators = {
        {"stride_q_token", [](SdpaSpec& s) { s.problem.stride_q_token += 1; }},
        {"stride_q_head", [](SdpaSpec& s) { s.problem.stride_q_head += 1; }},
        {"stride_k_token", [](SdpaSpec& s) { s.problem.stride_k_token += 1; }},
        {"stride_k_head", [](SdpaSpec& s) { s.problem.stride_k_head += 1; }},
        {"stride_v_token", [](SdpaSpec& s) { s.problem.stride_v_token += 1; }},
        {"stride_v_head", [](SdpaSpec& s) { s.problem.stride_v_head += 1; }},
        {"stride_o_token", [](SdpaSpec& s) { s.problem.stride_o_token += 1; }},
        {"stride_o_head", [](SdpaSpec& s) { s.problem.stride_o_head += 1; }},
        {"scale_log2", [](SdpaSpec& s) { s.problem.scale_log2 += 0.5f; }},
    };

    for (const auto& [name, mutate] : mutators) {
        auto spec = makeSdpaSpec();
        mutate(spec);
        EXPECT_EQ(GraphSignature::computeForSpec(kSdpaOpKind, spec, kArch), baseline)
            << "launch-time scalar '" << name << "' must NOT affect the signature";
    }
}

// op_kind partitions the cache and arch is a separate compile target;
// either changing must move the hash so cross-op / cross-arch lookups
// never alias.
TEST(TestGraphSignature, SdpaOpKindAndArchChangeSignature) {
    const auto baseline = GraphSignature::computeForSpec(kSdpaOpKind, makeSdpaSpec(), kArch);
    EXPECT_NE(GraphSignature::computeForSpec("sdpa_other_op", makeSdpaSpec(), kArch), baseline)
        << "op_kind did not affect the signature";
    EXPECT_NE(GraphSignature::computeForSpec(kSdpaOpKind, makeSdpaSpec(), "gfx942"), baseline)
        << "arch did not affect the signature";
}

// Each unified paged/varlen problem lane is codegen-relevant: a paged
// build, a varlen build, a windowed build, and a sinks build each emit a
// distinct kernel/grid and so must hash distinctly. A fold that dropped
// one would let two distinct kernels collide on the same cache key.
TEST(TestGraphSignature, SdpaChangesWithEachProblemLane) {
    const auto baseline = GraphSignature::computeForSpec(kSdpaOpKind, makeSdpaSpec(), kArch);

    const std::vector<std::pair<const char*, std::function<void(SdpaSpec&)>>> mutators = {
        {"is_paged", [](SdpaSpec& s) { s.is_paged = true; }},
        {"block_size", [](SdpaSpec& s) { s.block_size = 32; }},
        {"is_varlen", [](SdpaSpec& s) { s.is_varlen = true; }},
        {"sliding_window", [](SdpaSpec& s) { s.sliding_window = 128; }},
        {"use_sinks", [](SdpaSpec& s) { s.use_sinks = true; }},
    };

    for (const auto& [name, mutate] : mutators) {
        auto spec = makeSdpaSpec();
        mutate(spec);
        EXPECT_NE(GraphSignature::computeForSpec(kSdpaOpKind, spec, kArch), baseline)
            << "SdpaSpec problem lane '" << name << "' did not affect the signature";
    }
}

// Every folded perf knob must move the hash. The scorer-driven selection
// writes these onto the spec before the key is computed, so two distinct
// scored configs of the same shape must land on distinct cache keys --
// otherwise one would silently load the other's module.
TEST(TestGraphSignature, SdpaChangesWithEachPerfKnob) {
    const auto baseline = GraphSignature::computeForSpec(kSdpaOpKind, makeSdpaSpec(), kArch);

    const std::vector<std::pair<const char*, std::function<void(SdpaSpec&)>>> mutators = {
        {"num_warps", [](SdpaSpec& s) { s.knobs.num_warps += 1; }},
        {"block_m_per_warp", [](SdpaSpec& s) { s.knobs.block_m_per_warp += 1; }},
        {"tile_size", [](SdpaSpec& s) { s.knobs.tile_size += 1; }},
        {"waves_per_eu", [](SdpaSpec& s) { s.knobs.waves_per_eu += 1; }},
        {"use_mfma_32x32", [](SdpaSpec& s) { s.knobs.use_mfma_32x32 = !s.knobs.use_mfma_32x32; }},
        {"use_transposed_qk_32x32",
         [](SdpaSpec& s) { s.knobs.use_transposed_qk_32x32 = !s.knobs.use_transposed_qk_32x32; }},
        {"use_register_pv",
         [](SdpaSpec& s) { s.knobs.use_register_pv = !s.knobs.use_register_pv; }},
        {"use_early_v_schedule",
         [](SdpaSpec& s) { s.knobs.use_early_v_schedule = !s.knobs.use_early_v_schedule; }},
        {"use_fast_paged_kv_desc",
         [](SdpaSpec& s) { s.knobs.use_fast_paged_kv_desc = !s.knobs.use_fast_paged_kv_desc; }},
    };

    for (const auto& [name, mutate] : mutators) {
        auto spec = makeSdpaSpec();
        mutate(spec);
        EXPECT_NE(GraphSignature::computeForSpec(kSdpaOpKind, spec, kArch), baseline)
            << "SdpaSpec perf knob '" << name << "' did not affect the signature";
    }
}

}  // namespace
