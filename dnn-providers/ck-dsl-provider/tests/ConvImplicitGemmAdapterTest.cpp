// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

#include <cstdint>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "CkDslContainer.hpp"
#include "adapters/conv_implicit_gemm/ConvImplicitGemmAdapter.hpp"
#include "adapters/conv_implicit_gemm/ConvImplicitGemmPayload.hpp"
#include "adapters/conv_implicit_gemm/ConvImplicitGemmSpec.hpp"
#include "python/CompilePayload.hpp"
#include "python/CompileServiceBridge.hpp"

namespace {

using ck_dsl_provider::CkDslContainer;
using ck_dsl_provider::ConvImplicitGemmAdapter;
using ck_dsl_provider::ConvImplicitGemmSpec;
using ck_dsl_provider::convImplicitGemmSpecToPayload;
using ck_dsl_provider::PayloadDict;
using ck_dsl_provider::PayloadValue;

// Look up a key in an (interpreter-neutral) PayloadDict; nullptr if absent.
const PayloadValue* payloadFind(const PayloadDict& dict, const char* key) {
    for (const auto& kv : dict) {
        if (kv.first == key) {
            return &kv.second;
        }
    }
    return nullptr;
}

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// Example conv shape from plan §4: N=8, H=W=56, C=64, K=64, R=S=3,
/// stride=1, pad=1, dilation=1. FP16. NHWC physical layout; logical
/// NCHW dims per the hipDNN convention.
struct ExampleShape {
    static constexpr std::int64_t N = 8;
    static constexpr std::int64_t C = 64;
    static constexpr std::int64_t Hi = 56;
    static constexpr std::int64_t Wi = 56;
    static constexpr std::int64_t K = 64;
    static constexpr std::int64_t R = 3;
    static constexpr std::int64_t S = 3;
    static constexpr std::int64_t Ho = 56;  // (56 + 2*1 - 1*(3-1) - 1)/1 + 1
    static constexpr std::int64_t Wo = 56;
};

/// Build a FlatBuffer with one ConvolutionFwdAttributes + three
/// TensorAttributes (X / W / Y) for the example shape and return the
/// builder + the parsed tables + the tensor map the adapter expects.
struct ConvGraphFixture {
    flatbuffers::FlatBufferBuilder convBuilder;
    flatbuffers::FlatBufferBuilder xBuilder;
    flatbuffers::FlatBufferBuilder wBuilder;
    flatbuffers::FlatBufferBuilder yBuilder;
    const data_objects::ConvolutionFwdAttributes* convAttr{nullptr};
    const data_objects::TensorAttributes* x{nullptr};
    const data_objects::TensorAttributes* w{nullptr};
    const data_objects::TensorAttributes* y{nullptr};
    std::unordered_map<std::int64_t, const data_objects::TensorAttributes*> tensorMap;

    static void finishTensor(flatbuffers::FlatBufferBuilder& builder, std::int64_t uid,
                             const std::string& name, data_objects::DataType dtype,
                             const std::vector<std::int64_t>& dims,
                             const std::vector<std::int64_t>& strides) {
        auto attrOffset = data_objects::CreateTensorAttributesDirect(
            builder, uid, name.c_str(), dtype, &strides, &dims, /*virtual=*/false);
        builder.Finish(attrOffset);
    }

    ConvGraphFixture() {
        // X tensor: NHWC physical, logical dims [N, C, Hi, Wi], NHWC
        // strides [C*Hi*Wi, 1, Wi*C, C]. We don't actually use the
        // strides in the adapter, but populate them realistically for
        // documentation value.
        std::vector<std::int64_t> xDims{ExampleShape::N, ExampleShape::C, ExampleShape::Hi,
                                        ExampleShape::Wi};
        std::vector<std::int64_t> xStrides{ExampleShape::C * ExampleShape::Hi * ExampleShape::Wi, 1,
                                           ExampleShape::Wi * ExampleShape::C, ExampleShape::C};
        finishTensor(xBuilder, /*uid=*/1, "X", data_objects::DataType::HALF, xDims, xStrides);
        x = flatbuffers::GetRoot<data_objects::TensorAttributes>(xBuilder.GetBufferPointer());

        // W tensor: KRSC physical, logical dims [K, C, R, S].
        std::vector<std::int64_t> wDims{ExampleShape::K, ExampleShape::C, ExampleShape::R,
                                        ExampleShape::S};
        std::vector<std::int64_t> wStrides{ExampleShape::R * ExampleShape::S * ExampleShape::C, 1,
                                           ExampleShape::S * ExampleShape::C, ExampleShape::C};
        finishTensor(wBuilder, /*uid=*/2, "W", data_objects::DataType::HALF, wDims, wStrides);
        w = flatbuffers::GetRoot<data_objects::TensorAttributes>(wBuilder.GetBufferPointer());

        // Y tensor: NHWK physical, logical dims [N, K, Ho, Wo].
        std::vector<std::int64_t> yDims{ExampleShape::N, ExampleShape::K, ExampleShape::Ho,
                                        ExampleShape::Wo};
        std::vector<std::int64_t> yStrides{ExampleShape::K * ExampleShape::Ho * ExampleShape::Wo, 1,
                                           ExampleShape::Wo * ExampleShape::K, ExampleShape::K};
        finishTensor(yBuilder, /*uid=*/3, "Y", data_objects::DataType::HALF, yDims, yStrides);
        y = flatbuffers::GetRoot<data_objects::TensorAttributes>(yBuilder.GetBufferPointer());

        tensorMap[1] = x;
        tensorMap[2] = w;
        tensorMap[3] = y;

        // Conv attributes: stride 1, padding 1, dilation 1, 2-D.
        std::vector<std::int64_t> prePadding{1, 1};
        std::vector<std::int64_t> postPadding{1, 1};
        std::vector<std::int64_t> stride{1, 1};
        std::vector<std::int64_t> dilation{1, 1};
        auto convOffset = data_objects::CreateConvolutionFwdAttributesDirect(
            convBuilder, /*x_uid=*/1, /*w_uid=*/2, /*y_uid=*/3, &prePadding, &postPadding, &stride,
            &dilation, data_objects::ConvMode::CROSS_CORRELATION);
        convBuilder.Finish(convOffset);
        convAttr = flatbuffers::GetRoot<data_objects::ConvolutionFwdAttributes>(
            convBuilder.GetBufferPointer());
    }
};

TEST(TestConvImplicitGemmAdapter, BuildSpecForExampleShape) {
    ConvGraphFixture fx;

    ConvImplicitGemmSpec spec = ConvImplicitGemmAdapter::buildSpec(*fx.convAttr, fx.tensorMap);

    // 13 graph-derived ConvProblem fields.
    EXPECT_EQ(spec.problem.N, 8);
    EXPECT_EQ(spec.problem.C, 64);
    EXPECT_EQ(spec.problem.Hi, 56);
    EXPECT_EQ(spec.problem.Wi, 56);
    EXPECT_EQ(spec.problem.K, 64);
    EXPECT_EQ(spec.problem.R, 3);
    EXPECT_EQ(spec.problem.S, 3);
    EXPECT_EQ(spec.problem.sH, 1);
    EXPECT_EQ(spec.problem.sW, 1);
    EXPECT_EQ(spec.problem.pH, 1);
    EXPECT_EQ(spec.problem.pW, 1);
    EXPECT_EQ(spec.problem.dH, 1);
    EXPECT_EQ(spec.problem.dW, 1);

    // Derived geometry (cross-check the helpers match the example
    // shape's expected output dims).
    EXPECT_EQ(spec.problem.Ho(), 56);
    EXPECT_EQ(spec.problem.Wo(), 56);
    EXPECT_EQ(spec.problem.M(), 8 * 56 * 56);
    EXPECT_EQ(spec.problem.Ngemm(), 64);
    EXPECT_EQ(spec.problem.Kgemm(), 3 * 3 * 64);

    // Codegen knob defaults -- these mirror the DSL ImplicitGemmConvSpec
    // dataclass defaults field-for-field (see ConvImplicitGemmSpec.hpp).
    EXPECT_EQ(spec.tile_m, 64);
    EXPECT_EQ(spec.tile_n, 64);
    EXPECT_EQ(spec.tile_k, 64);
    EXPECT_EQ(spec.warp_m, 2);
    EXPECT_EQ(spec.warp_n, 2);
    EXPECT_EQ(spec.warp_tile_m, 32);  // mirrors dataclass default
    EXPECT_EQ(spec.warp_tile_n, 32);  // mirrors dataclass default
    EXPECT_EQ(spec.warp_tile_k, 16);  // mirrors dataclass default
    EXPECT_EQ(spec.wave_size, 64);
    EXPECT_EQ(spec.pipeline, "mem");
    EXPECT_EQ(spec.epilogue, "default");  // mirrors dataclass default
    EXPECT_FALSE(spec.async_dma);
    EXPECT_FALSE(spec.unroll_k);
    EXPECT_FALSE(spec.lds_k_pad.has_value());
    EXPECT_FALSE(spec.chiplet_swizzle);
    EXPECT_FALSE(spec.waves_per_eu.has_value());
    EXPECT_EQ(spec.block_size(), 256);
}

TEST(TestConvImplicitGemmAdapter, ApplyArchCodegenConfigSelectsPerArchKnobs) {
    ConvGraphFixture fx;

    // gfx950 keeps the wide 32x32x16 f16 MFMA atom + wave64 (the
    // historical default), so its kernel selection is unchanged.
    {
        ConvImplicitGemmSpec spec = ConvImplicitGemmAdapter::buildSpec(*fx.convAttr, fx.tensorMap);
        EXPECT_TRUE(ConvImplicitGemmAdapter::applyArchCodegenConfig(spec, "gfx950"));
        EXPECT_EQ(spec.warp_tile_m, 32);
        EXPECT_EQ(spec.warp_tile_n, 32);
        EXPECT_EQ(spec.warp_tile_k, 16);
        EXPECT_EQ(spec.wave_size, 64);
        EXPECT_EQ(spec.block_size(), 256);  // warp_m * warp_n * wave_size
        // Problem geometry must be left untouched by the codegen knobs.
        EXPECT_EQ(spec.problem.N, 8);
        EXPECT_EQ(spec.problem.K, 64);
    }

    // gfx942 lacks the 32x32x16 f16 atom -> 16x16x16 atom, still wave64.
    {
        ConvImplicitGemmSpec spec = ConvImplicitGemmAdapter::buildSpec(*fx.convAttr, fx.tensorMap);
        EXPECT_TRUE(ConvImplicitGemmAdapter::applyArchCodegenConfig(spec, "gfx942"));
        EXPECT_EQ(spec.warp_tile_m, 16);
        EXPECT_EQ(spec.warp_tile_n, 16);
        EXPECT_EQ(spec.warp_tile_k, 16);
        EXPECT_EQ(spec.wave_size, 64);
        EXPECT_EQ(spec.block_size(), 256);
    }

    // gfx1151 is wave32 RDNA with the 16x16x16 WMMA atom.
    {
        ConvImplicitGemmSpec spec = ConvImplicitGemmAdapter::buildSpec(*fx.convAttr, fx.tensorMap);
        EXPECT_TRUE(ConvImplicitGemmAdapter::applyArchCodegenConfig(spec, "gfx1151"));
        EXPECT_EQ(spec.warp_tile_m, 16);
        EXPECT_EQ(spec.warp_tile_n, 16);
        EXPECT_EQ(spec.warp_tile_k, 16);
        EXPECT_EQ(spec.wave_size, 32);
        EXPECT_EQ(spec.block_size(), 128);
    }

    // An arch with no known config is reported as unsupported and the
    // spec is left unchanged (the caller declines / fails closed).
    {
        ConvImplicitGemmSpec spec = ConvImplicitGemmAdapter::buildSpec(*fx.convAttr, fx.tensorMap);
        const ConvImplicitGemmSpec before = spec;
        EXPECT_FALSE(ConvImplicitGemmAdapter::applyArchCodegenConfig(spec, "gfx777"));
        EXPECT_EQ(spec.warp_tile_m, before.warp_tile_m);
        EXPECT_EQ(spec.warp_tile_n, before.warp_tile_n);
        EXPECT_EQ(spec.warp_tile_k, before.warp_tile_k);
        EXPECT_EQ(spec.wave_size, before.wave_size);
    }
}

TEST(TestConvImplicitGemmAdapter, RejectsAsymmetricPadding) {
    flatbuffers::FlatBufferBuilder builder;
    std::vector<std::int64_t> prePadding{1, 1};
    std::vector<std::int64_t> postPadding{2, 1};  // asymmetric on H
    std::vector<std::int64_t> stride{1, 1};
    std::vector<std::int64_t> dilation{1, 1};
    auto convOffset = data_objects::CreateConvolutionFwdAttributesDirect(
        builder, 1, 2, 3, &prePadding, &postPadding, &stride, &dilation,
        data_objects::ConvMode::CROSS_CORRELATION);
    builder.Finish(convOffset);
    auto convAttr =
        flatbuffers::GetRoot<data_objects::ConvolutionFwdAttributes>(builder.GetBufferPointer());

    ConvGraphFixture fx;  // re-use the fixture's tensor map
    EXPECT_THROW(ConvImplicitGemmAdapter::buildSpec(*convAttr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestConvImplicitGemmAdapter, Rejects3DConv) {
    flatbuffers::FlatBufferBuilder builder;
    std::vector<std::int64_t> prePadding{0, 0, 0};
    std::vector<std::int64_t> postPadding{0, 0, 0};
    std::vector<std::int64_t> stride{1, 1, 1};
    std::vector<std::int64_t> dilation{1, 1, 1};
    auto convOffset = data_objects::CreateConvolutionFwdAttributesDirect(
        builder, 1, 2, 3, &prePadding, &postPadding, &stride, &dilation,
        data_objects::ConvMode::CROSS_CORRELATION);
    builder.Finish(convOffset);
    auto convAttr =
        flatbuffers::GetRoot<data_objects::ConvolutionFwdAttributes>(builder.GetBufferPointer());

    ConvGraphFixture fx;
    EXPECT_THROW(ConvImplicitGemmAdapter::buildSpec(*convAttr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestConvImplicitGemmAdapter, RejectsTrueConvolutionMode) {
    flatbuffers::FlatBufferBuilder builder;
    std::vector<std::int64_t> padding{1, 1};
    std::vector<std::int64_t> stride{1, 1};
    std::vector<std::int64_t> dilation{1, 1};
    auto convOffset = data_objects::CreateConvolutionFwdAttributesDirect(
        builder, 1, 2, 3, &padding, &padding, &stride, &dilation,
        data_objects::ConvMode::CONVOLUTION);  // not CROSS_CORRELATION
    builder.Finish(convOffset);
    auto convAttr =
        flatbuffers::GetRoot<data_objects::ConvolutionFwdAttributes>(builder.GetBufferPointer());

    ConvGraphFixture fx;
    EXPECT_THROW(ConvImplicitGemmAdapter::buildSpec(*convAttr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestConvImplicitGemmAdapter, RejectsMissingTensor) {
    ConvGraphFixture fx;
    decltype(fx.tensorMap) emptyMap;
    EXPECT_THROW(ConvImplicitGemmAdapter::buildSpec(*fx.convAttr, emptyMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestConvImplicitGemmAdapter, RejectsNonHalfDtype) {
    // Build an X tensor with FLOAT instead of HALF; keep W/Y as HALF.
    flatbuffers::FlatBufferBuilder xBuilder;
    std::vector<std::int64_t> dims{ExampleShape::N, ExampleShape::C, ExampleShape::Hi,
                                   ExampleShape::Wi};
    std::vector<std::int64_t> strides{ExampleShape::C * ExampleShape::Hi * ExampleShape::Wi, 1,
                                      ExampleShape::Wi * ExampleShape::C, ExampleShape::C};
    auto attrOffset = data_objects::CreateTensorAttributesDirect(
        xBuilder, 1, "X", data_objects::DataType::FLOAT, &strides, &dims, /*virtual=*/false);
    xBuilder.Finish(attrOffset);
    auto xFloat = flatbuffers::GetRoot<data_objects::TensorAttributes>(xBuilder.GetBufferPointer());

    ConvGraphFixture fx;
    fx.tensorMap[1] = xFloat;  // override the HALF X with a FLOAT one
    EXPECT_THROW(ConvImplicitGemmAdapter::buildSpec(*fx.convAttr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

/// The payload->dataclass round-trip test needs the embedded interpreter; we
/// construct a CkDslContainer so the per-process MicroPython runtime is up and
/// the frozen ck_dsl modules are importable. (The pure-C++ payload conversion
/// itself needs no interpreter.)
class TestConvImplicitGemmPayload : public ::testing::Test {
   protected:
    void SetUp() override {
        _container = std::make_unique<CkDslContainer>();
    }

    std::unique_ptr<CkDslContainer> _container;
};

TEST_F(TestConvImplicitGemmPayload, PayloadDictForExampleShape) {
    ConvGraphFixture fx;
    auto spec = ConvImplicitGemmAdapter::buildSpec(*fx.convAttr, fx.tensorMap);

    const PayloadDict payload = convImplicitGemmSpecToPayload(spec);

    // Top-level fields the dataclass takes as kwargs.
    ASSERT_NE(payloadFind(payload, "problem"), nullptr);
    ASSERT_NE(payloadFind(payload, "name"), nullptr);
    ASSERT_NE(payloadFind(payload, "tile_m"), nullptr);
    ASSERT_NE(payloadFind(payload, "tile_n"), nullptr);
    ASSERT_NE(payloadFind(payload, "tile_k"), nullptr);
    ASSERT_NE(payloadFind(payload, "warp_tile_m"), nullptr);
    ASSERT_NE(payloadFind(payload, "warp_tile_n"), nullptr);
    ASSERT_NE(payloadFind(payload, "warp_tile_k"), nullptr);
    ASSERT_NE(payloadFind(payload, "epilogue"), nullptr);
    ASSERT_NE(payloadFind(payload, "pipeline"), nullptr);
    ASSERT_NE(payloadFind(payload, "lds_k_pad"), nullptr);
    ASSERT_NE(payloadFind(payload, "waves_per_eu"), nullptr);
    // Deliberately NOT present (the dataclass owns the derivation):
    EXPECT_EQ(payloadFind(payload, "lds_layout"), nullptr);

    EXPECT_EQ(payloadFind(payload, "tile_m")->intVal, 64);
    EXPECT_EQ(payloadFind(payload, "tile_k")->intVal, 64);
    EXPECT_EQ(payloadFind(payload, "warp_tile_m")->intVal, 32);
    EXPECT_EQ(payloadFind(payload, "warp_tile_k")->intVal, 16);
    EXPECT_EQ(payloadFind(payload, "epilogue")->strVal, "default");
    EXPECT_EQ(payloadFind(payload, "pipeline")->strVal, "mem");
    EXPECT_EQ(payloadFind(payload, "name")->strVal, "ck_dsl_conv_igemm");
    EXPECT_FALSE(payloadFind(payload, "async_dma")->boolVal);
    EXPECT_EQ(payloadFind(payload, "lds_k_pad")->kind, PayloadValue::Kind::None);
    EXPECT_EQ(payloadFind(payload, "waves_per_eu")->kind, PayloadValue::Kind::None);

    // Nested ConvProblem dict.
    const PayloadValue* problemVal = payloadFind(payload, "problem");
    ASSERT_EQ(problemVal->kind, PayloadValue::Kind::Dict);
    const PayloadDict& problem = problemVal->dictVal;
    EXPECT_EQ(payloadFind(problem, "N")->intVal, 8);
    EXPECT_EQ(payloadFind(problem, "C")->intVal, 64);
    EXPECT_EQ(payloadFind(problem, "Hi")->intVal, 56);
    EXPECT_EQ(payloadFind(problem, "Wi")->intVal, 56);
    EXPECT_EQ(payloadFind(problem, "K")->intVal, 64);
    EXPECT_EQ(payloadFind(problem, "R")->intVal, 3);
    EXPECT_EQ(payloadFind(problem, "S")->intVal, 3);
    EXPECT_EQ(payloadFind(problem, "sH")->intVal, 1);
    EXPECT_EQ(payloadFind(problem, "pH")->intVal, 1);
    EXPECT_EQ(payloadFind(problem, "dH")->intVal, 1);
}

TEST_F(TestConvImplicitGemmPayload, CompilesThroughRealDataclass) {
    // Cross-check: feed the payload through the real ck_dsl
    // ImplicitGemmConvSpec / ConvProblem dataclasses. compile_service.compile
    // constructs the spec from this exact field set (and _reject_unexpected
    // rejects stray keys), so a drift between the C++ payload and the
    // dataclass fields fails here -- the divergence canary the plan §3.3
    // cost-of-mirroring note asks for. comgr cross-compiles, so no GPU needed.
    ConvGraphFixture fx;
    auto spec = ConvImplicitGemmAdapter::buildSpec(*fx.convAttr, fx.tensorMap);

    const PayloadDict payload = convImplicitGemmSpecToPayload(spec);
    auto artifact =
        _container->compileServiceBridge().compile("conv_implicit_gemm", payload, "gfx950");

    EXPECT_FALSE(artifact.hsaco.empty()) << "round-tripped payload must compile";
    EXPECT_NE(artifact.isa.find("gfx950"), std::string::npos);
}

}  // namespace
