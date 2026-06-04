// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ConvImplicitGemmAdapter.hpp"

#include <cstdint>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <limits>
#include <sstream>
#include <string>

namespace ck_dsl_provider {

namespace {

using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
using ConvMode = hipdnn_flatbuffers_sdk::data_objects::ConvMode;
using TensorAttributes = hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;
using TensorMap = ConvImplicitGemmAdapter::TensorMap;

[[noreturn]] void throwBadParam(const std::string& msg) {
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                   "ConvImplicitGemmAdapter: " + msg);
}

const TensorAttributes& lookupTensor(const TensorMap& tensorMap, std::int64_t uid,
                                     const char* role) {
    auto it = tensorMap.find(uid);
    if (it == tensorMap.end() || it->second == nullptr) {
        std::ostringstream oss;
        oss << "tensor map missing entry for " << role << " uid=" << uid;
        throwBadParam(oss.str());
    }
    return *it->second;
}

std::int32_t narrowToI32(std::int64_t value, const char* fieldName) {
    if (value < std::numeric_limits<std::int32_t>::min() ||
        value > std::numeric_limits<std::int32_t>::max()) {
        std::ostringstream oss;
        oss << "field '" << fieldName << "' value " << value << " does not fit in int32_t";
        throwBadParam(oss.str());
    }
    return static_cast<std::int32_t>(value);
}

void checkDtypeFp16(const TensorAttributes& t, const char* role) {
    // The DSL build_implicit_gemm_conv kernel currently only emits FP16
    // I/O (see kernel signature: ptr<f16>, ptr<f16>, ptr<f16>). Reject
    // anything else at the adapter boundary so applicability + the
    // engine selection layer can fall through cleanly to other engines.
    if (t.data_type() != DataType::HALF) {
        std::ostringstream oss;
        oss << role << " data_type must be HALF (FP16); got " << static_cast<int>(t.data_type());
        throwBadParam(oss.str());
    }
}

void check4dDims(const TensorAttributes& t, const char* role) {
    if (t.dims() == nullptr || t.dims()->size() != 4) {
        std::ostringstream oss;
        oss << role << " dims must be 4-D (logical NCHW for X/Y, KCRS for W); got size "
            << (t.dims() == nullptr ? 0u : t.dims()->size());
        throwBadParam(oss.str());
    }
}

std::int32_t getDim(const TensorAttributes& t, std::uint32_t idx, const char* role,
                    const char* fieldName) {
    // Caller has already validated dims is 4-D.
    auto raw = t.dims()->Get(idx);
    return narrowToI32(raw, (std::string(role) + "." + fieldName).c_str());
}

void checkSpatialAttr(const flatbuffers::Vector<std::int64_t>* attr, const char* name) {
    if (attr == nullptr) {
        throwBadParam(std::string("conv attribute '") + name + "' must be set");
    }
    if (attr->size() != 2) {
        std::ostringstream oss;
        oss << "conv attribute '" << name << "' must have size 2 (2-D conv only for M1); got size "
            << attr->size();
        throwBadParam(oss.str());
    }
}

}  // namespace

ConvImplicitGemmSpec ConvImplicitGemmAdapter::buildSpec(const ConvolutionFwdAttributes& convAttr,
                                                        const TensorMap& tensorMap) {
    if (convAttr.conv_mode() != ConvMode::CROSS_CORRELATION) {
        throwBadParam(
            "conv_mode must be CROSS_CORRELATION (true convolution is unsupported for M1)");
    }

    const auto& X = lookupTensor(tensorMap, convAttr.x_tensor_uid(), "X");
    const auto& W = lookupTensor(tensorMap, convAttr.w_tensor_uid(), "W");
    const auto& Y = lookupTensor(tensorMap, convAttr.y_tensor_uid(), "Y");

    check4dDims(X, "X");
    check4dDims(W, "W");
    check4dDims(Y, "Y");

    checkDtypeFp16(X, "X");
    checkDtypeFp16(W, "W");
    checkDtypeFp16(Y, "Y");

    // Tensor dim convention (miopen-provider precedent):
    // TensorAttributes::dims is the logical NCHW order
    // regardless of physical layout. NHWC strides describe the
    // memory layout but the dims index by logical axis. So
    //   X.dims = [N, C, Hi, Wi]
    //   W.dims = [K, C, R,  S ]   (KCRS logical for KRSC layout)
    //   Y.dims = [N, K, Ho, Wo]
    auto N = getDim(X, 0, "X", "N");
    auto Cx = getDim(X, 1, "X", "C");
    auto Hi = getDim(X, 2, "X", "Hi");
    auto Wi = getDim(X, 3, "X", "Wi");

    auto K = getDim(W, 0, "W", "K");
    auto Cw = getDim(W, 1, "W", "C");
    auto R = getDim(W, 2, "W", "R");
    auto S = getDim(W, 3, "W", "S");

    if (Cx != Cw) {
        std::ostringstream oss;
        oss << "X.C (" << Cx << ") must equal W.C (" << Cw
            << "); grouped convolutions are unsupported for M1";
        throwBadParam(oss.str());
    }

    // We don't lift K/R/S from Y -- the conv-fwd math determines Y's
    // spatial dims via Ho, Wo. We do cross-check N + K against Y to
    // catch a malformed graph early; the spatial-dim cross-check is
    // deferred to I-7 (where it can use the spec's Ho()/Wo() helpers
    // once the spec is built).
    if (getDim(Y, 0, "Y", "N") != N) {
        throwBadParam("Y.N must equal X.N");
    }
    if (getDim(Y, 1, "Y", "K") != K) {
        throwBadParam("Y.K must equal W.K");
    }

    checkSpatialAttr(convAttr.pre_padding(), "pre_padding");
    checkSpatialAttr(convAttr.post_padding(), "post_padding");
    checkSpatialAttr(convAttr.stride(), "stride");
    checkSpatialAttr(convAttr.dilation(), "dilation");

    // Asymmetric padding is unsupported (matches miopen-provider's
    // policy). The CK DSL's ConvProblem has a single pH/pW per spatial
    // axis; encoding asymmetric pads would require descriptor changes.
    if (convAttr.pre_padding()->Get(0) != convAttr.post_padding()->Get(0) ||
        convAttr.pre_padding()->Get(1) != convAttr.post_padding()->Get(1)) {
        throwBadParam("asymmetric padding is not supported");
    }

    ConvImplicitGemmSpec spec{};
    spec.problem.N = N;
    spec.problem.Hi = Hi;
    spec.problem.Wi = Wi;
    spec.problem.C = Cx;
    spec.problem.K = K;
    spec.problem.R = R;
    spec.problem.S = S;
    spec.problem.pH = narrowToI32(convAttr.pre_padding()->Get(0), "pre_padding[0]");
    spec.problem.pW = narrowToI32(convAttr.pre_padding()->Get(1), "pre_padding[1]");
    spec.problem.sH = narrowToI32(convAttr.stride()->Get(0), "stride[0]");
    spec.problem.sW = narrowToI32(convAttr.stride()->Get(1), "stride[1]");
    spec.problem.dH = narrowToI32(convAttr.dilation()->Get(0), "dilation[0]");
    spec.problem.dW = narrowToI32(convAttr.dilation()->Get(1), "dilation[1]");

    // The arch-dependent codegen knobs (warp-tile atom + wave size) are
    // left at the struct defaults here and overwritten per-arch by
    // applyArchCodegenConfig once the device arch is known. Every other
    // knob (tile_*, warp_*, pipeline, epilogue, etc.) keeps its example
    // constexpr default from ConvImplicitGemmSpec. Autotuning is M2+ work.
    return spec;
}

bool ConvImplicitGemmAdapter::applyArchCodegenConfig(ConvImplicitGemmSpec& spec,
                                                     const std::string& arch) {
    // 16x16x16 is the f16 MMA/WMMA atom the DSL validates on all three
    // M1 targets (an MFMA op on the CDNA targets gfx942/gfx950, a WMMA
    // op on the RDNA target gfx1151). gfx950 additionally supports the
    // wider 32x32x16 f16 MFMA atom and keeps it as its historical
    // default so gfx950 kernel selection is unchanged by arch-awareness.
    // wave_size tracks the hardware: 64 on the wave64 CDNA targets, 32
    // on the wave32 RDNA target gfx1151. These are the same per-arch
    // example configs the provider's cross-arch tests exercise; the DSL
    // rejects a mismatched atom/wave at compile time (is_valid_spec), so
    // an unrecognised arch here means "the provider cannot target this
    // device" -- report it as false rather than guessing knobs.
    if (arch == "gfx950") {
        spec.warp_tile_m = 32;
        spec.warp_tile_n = 32;
        spec.warp_tile_k = 16;
        spec.wave_size = 64;
        return true;
    }
    if (arch == "gfx942") {
        spec.warp_tile_m = 16;
        spec.warp_tile_n = 16;
        spec.warp_tile_k = 16;
        spec.wave_size = 64;
        return true;
    }
    if (arch == "gfx1151") {
        spec.warp_tile_m = 16;
        spec.warp_tile_n = 16;
        spec.warp_tile_k = 16;
        spec.wave_size = 32;
        return true;
    }
    return false;
}

}  // namespace ck_dsl_provider
