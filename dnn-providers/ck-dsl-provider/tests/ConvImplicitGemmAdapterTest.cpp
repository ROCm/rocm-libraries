// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <pybind11/embed.h>

#include <cstdint>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "CkDslContainer.hpp"
#include "adapters/conv_implicit_gemm/ConvImplicitGemmAdapter.hpp"
#include "adapters/conv_implicit_gemm/ConvImplicitGemmPayload.hpp"
#include "adapters/conv_implicit_gemm/ConvImplicitGemmSpec.hpp"

namespace py = pybind11;

namespace {

using ck_dsl_provider::CkDslContainer;
using ck_dsl_provider::ConvImplicitGemmAdapter;
using ck_dsl_provider::ConvImplicitGemmSpec;
using ck_dsl_provider::convImplicitGemmSpecToPayload;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// Bake-off conv shape from plan §4: N=8, H=W=56, C=64, K=64, R=S=3,
/// stride=1, pad=1, dilation=1. FP16. NHWC physical layout; logical
/// NCHW dims per the hipDNN convention (see PREP_FINDINGS P-6).
struct BakeOffShape {
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
/// TensorAttributes (X / W / Y) for the bake-off shape and return the
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
        std::vector<std::int64_t> xDims{BakeOffShape::N, BakeOffShape::C, BakeOffShape::Hi,
                                        BakeOffShape::Wi};
        std::vector<std::int64_t> xStrides{BakeOffShape::C * BakeOffShape::Hi * BakeOffShape::Wi, 1,
                                           BakeOffShape::Wi * BakeOffShape::C, BakeOffShape::C};
        finishTensor(xBuilder, /*uid=*/1, "X", data_objects::DataType::HALF, xDims, xStrides);
        x = flatbuffers::GetRoot<data_objects::TensorAttributes>(xBuilder.GetBufferPointer());

        // W tensor: KRSC physical, logical dims [K, C, R, S].
        std::vector<std::int64_t> wDims{BakeOffShape::K, BakeOffShape::C, BakeOffShape::R,
                                        BakeOffShape::S};
        std::vector<std::int64_t> wStrides{BakeOffShape::R * BakeOffShape::S * BakeOffShape::C, 1,
                                           BakeOffShape::S * BakeOffShape::C, BakeOffShape::C};
        finishTensor(wBuilder, /*uid=*/2, "W", data_objects::DataType::HALF, wDims, wStrides);
        w = flatbuffers::GetRoot<data_objects::TensorAttributes>(wBuilder.GetBufferPointer());

        // Y tensor: NHWK physical, logical dims [N, K, Ho, Wo].
        std::vector<std::int64_t> yDims{BakeOffShape::N, BakeOffShape::K, BakeOffShape::Ho,
                                        BakeOffShape::Wo};
        std::vector<std::int64_t> yStrides{BakeOffShape::K * BakeOffShape::Ho * BakeOffShape::Wo, 1,
                                           BakeOffShape::Wo * BakeOffShape::K, BakeOffShape::K};
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

TEST(TestConvImplicitGemmAdapter, BuildSpecForBakeOffShape) {
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

    // Derived geometry (cross-check the helpers match the bake-off
    // shape's expected output dims).
    EXPECT_EQ(spec.problem.Ho(), 56);
    EXPECT_EQ(spec.problem.Wo(), 56);
    EXPECT_EQ(spec.problem.M(), 8 * 56 * 56);
    EXPECT_EQ(spec.problem.Ngemm(), 64);
    EXPECT_EQ(spec.problem.Kgemm(), 3 * 3 * 64);

    // Bake-off constexpr defaults (deltas vs the dataclass defaults
    // are the load-bearing checks; see ConvImplicitGemmSpec.hpp
    // header comment for the full delta list).
    EXPECT_EQ(spec.tile_m, 64);
    EXPECT_EQ(spec.tile_n, 64);
    EXPECT_EQ(spec.tile_k, 64);  // bake-off override (dataclass: 128)
    EXPECT_EQ(spec.warp_m, 2);
    EXPECT_EQ(spec.warp_n, 2);
    EXPECT_EQ(spec.warp_tile_m, 32);  // bake-off override (dataclass: 16)
    EXPECT_EQ(spec.warp_tile_n, 32);  // bake-off override (dataclass: 16)
    EXPECT_EQ(spec.warp_tile_k, 16);  // bake-off override (dataclass: 32)
    EXPECT_EQ(spec.wave_size, 64);
    EXPECT_EQ(spec.pipeline, "mem");
    EXPECT_EQ(spec.epilogue, "cshuffle");  // bake-off override (dataclass: "default")
    EXPECT_FALSE(spec.async_dma);
    EXPECT_FALSE(spec.unroll_k);
    EXPECT_FALSE(spec.lds_k_pad.has_value());
    EXPECT_FALSE(spec.chiplet_swizzle);
    EXPECT_FALSE(spec.waves_per_eu.has_value());
    EXPECT_EQ(spec.block_size(), 256);
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
    std::vector<std::int64_t> dims{BakeOffShape::N, BakeOffShape::C, BakeOffShape::Hi,
                                   BakeOffShape::Wi};
    std::vector<std::int64_t> strides{BakeOffShape::C * BakeOffShape::Hi * BakeOffShape::Wi, 1,
                                      BakeOffShape::Wi * BakeOffShape::C, BakeOffShape::C};
    auto attrOffset = data_objects::CreateTensorAttributesDirect(
        xBuilder, 1, "X", data_objects::DataType::FLOAT, &strides, &dims, /*virtual=*/false);
    xBuilder.Finish(attrOffset);
    auto xFloat = flatbuffers::GetRoot<data_objects::TensorAttributes>(xBuilder.GetBufferPointer());

    ConvGraphFixture fx;
    fx.tensorMap[1] = xFloat;  // override the HALF X with a FLOAT one
    EXPECT_THROW(ConvImplicitGemmAdapter::buildSpec(*fx.convAttr, fx.tensorMap),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

/// Payload conversion needs the embedded interpreter. We construct a
/// CkDslContainer so the per-process interpreter is up, then exercise
/// the spec -> py::dict translation under the GIL.
class ConvImplicitGemmPayload : public ::testing::Test {
   protected:
    void SetUp() override {
        // The container owns the embedded interpreter; constructing
        // it ensures Py_Initialize has run before we touch any py::*.
        _container = std::make_unique<CkDslContainer>();
    }

    std::unique_ptr<CkDslContainer> _container;
};

TEST_F(ConvImplicitGemmPayload, PayloadDictForBakeOffShape) {
    ConvGraphFixture fx;
    auto spec = ConvImplicitGemmAdapter::buildSpec(*fx.convAttr, fx.tensorMap);

    py::gil_scoped_acquire gil;
    py::dict payload = convImplicitGemmSpecToPayload(spec);

    // Top-level fields the dataclass takes as kwargs.
    ASSERT_TRUE(payload.contains("problem"));
    ASSERT_TRUE(payload.contains("name"));
    ASSERT_TRUE(payload.contains("tile_m"));
    ASSERT_TRUE(payload.contains("tile_n"));
    ASSERT_TRUE(payload.contains("tile_k"));
    ASSERT_TRUE(payload.contains("warp_tile_m"));
    ASSERT_TRUE(payload.contains("warp_tile_n"));
    ASSERT_TRUE(payload.contains("warp_tile_k"));
    ASSERT_TRUE(payload.contains("epilogue"));
    ASSERT_TRUE(payload.contains("pipeline"));
    ASSERT_TRUE(payload.contains("lds_k_pad"));
    ASSERT_TRUE(payload.contains("waves_per_eu"));
    // Deliberately NOT present (the dataclass owns the derivation):
    EXPECT_FALSE(payload.contains("lds_layout"));

    EXPECT_EQ(payload["tile_m"].cast<int>(), 64);
    EXPECT_EQ(payload["tile_k"].cast<int>(), 64);
    EXPECT_EQ(payload["warp_tile_m"].cast<int>(), 32);
    EXPECT_EQ(payload["warp_tile_k"].cast<int>(), 16);
    EXPECT_EQ(payload["epilogue"].cast<std::string>(), "cshuffle");
    EXPECT_EQ(payload["pipeline"].cast<std::string>(), "mem");
    EXPECT_EQ(payload["name"].cast<std::string>(), "ck_dsl_conv_igemm");
    EXPECT_FALSE(payload["async_dma"].cast<bool>());
    EXPECT_TRUE(payload["lds_k_pad"].is_none());
    EXPECT_TRUE(payload["waves_per_eu"].is_none());

    // Nested ConvProblem dict.
    auto problem = payload["problem"].cast<py::dict>();
    EXPECT_EQ(problem["N"].cast<int>(), 8);
    EXPECT_EQ(problem["C"].cast<int>(), 64);
    EXPECT_EQ(problem["Hi"].cast<int>(), 56);
    EXPECT_EQ(problem["Wi"].cast<int>(), 56);
    EXPECT_EQ(problem["K"].cast<int>(), 64);
    EXPECT_EQ(problem["R"].cast<int>(), 3);
    EXPECT_EQ(problem["S"].cast<int>(), 3);
    EXPECT_EQ(problem["sH"].cast<int>(), 1);
    EXPECT_EQ(problem["pH"].cast<int>(), 1);
    EXPECT_EQ(problem["dH"].cast<int>(), 1);
}

TEST_F(ConvImplicitGemmPayload, RoundTripsThroughPythonDataclass) {
    // Cross-check: splat the payload into the actual Python
    // ImplicitGemmConvSpec dataclass. If our field set drifts from
    // the dataclass (extra/missing field) this fails loudly with a
    // TypeError -- exactly the divergence canary the plan §3.3 cost-
    // of-mirroring note asks for.
    ConvGraphFixture fx;
    auto spec = ConvImplicitGemmAdapter::buildSpec(*fx.convAttr, fx.tensorMap);

    py::gil_scoped_acquire gil;
    py::dict payload = convImplicitGemmSpecToPayload(spec);

    // The container already injected ck_dsl onto sys.path via the
    // compile-service bridge, so this import succeeds.
    py::module_ conv = py::module_::import("ck_dsl.instances.conv_implicit_gemm");
    py::object ConvProblem = conv.attr("ConvProblem");
    py::object ImplicitGemmConvSpec = conv.attr("ImplicitGemmConvSpec");

    // Build the nested ConvProblem first (it goes into the spec ctor
    // as a positional/kwarg). Splatting our nested dict means the
    // ConvProblem dataclass also enforces field-set parity.
    py::object problemInst = ConvProblem(**payload["problem"].cast<py::dict>());

    // Reassemble the kwargs the dataclass expects: same dict minus
    // the nested 'problem' key, plus the constructed ConvProblem.
    py::dict kwargs = payload.attr("copy")().cast<py::dict>();
    PyDict_DelItemString(kwargs.ptr(), "problem");
    kwargs["problem"] = problemInst;

    py::object specInst = ImplicitGemmConvSpec(**kwargs);

    EXPECT_EQ(specInst.attr("tile_k").cast<int>(), 64);
    EXPECT_EQ(specInst.attr("warp_tile_m").cast<int>(), 32);
    EXPECT_EQ(specInst.attr("epilogue").cast<std::string>(), "cshuffle");
    EXPECT_EQ(specInst.attr("block_size").cast<int>(), 256);
    EXPECT_EQ(specInst.attr("problem").attr("Ho").cast<int>(), 56);
}

}  // namespace
