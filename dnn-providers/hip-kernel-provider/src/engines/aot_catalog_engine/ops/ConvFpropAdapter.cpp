// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ops/ConvFpropAdapter.hpp"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/TensorAttributesWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "launch/PluginError.hpp"

namespace aot_catalog_engine::ops
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper;

namespace
{

// hipDNN element dtype -> the provider dtype token kernel authors use in
// family.json constraints. Only the two dtypes the WMMA/MFMA implicit-GEMM conv
// serves; anything else yields nullopt (decode declines -> another engine runs).
std::optional<std::string> providerDtype(data_objects::DataType dtype)
{
    switch(dtype)
    {
    case data_objects::DataType::HALF:
        return std::string("f16");
    case data_objects::DataType::BFLOAT16:
        return std::string("bf16");
    default:
        return std::nullopt;
    }
}

int64_t problemInt(const catalog::ProblemShape& problem, const std::string& key)
{
    auto it = problem.find(key);
    if(it == problem.end() || !std::holds_alternative<int64_t>(it->second))
    {
        throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                         "aot-catalog(conv_fprop): problem missing integer key '" + key + "'");
    }
    return std::get<int64_t>(it->second);
}

// The runtime conv kernel addresses input as NHWC, weight as KRSC, output as
// NHWK -- i.e. channels-last packed on every operand. Logical dims are canonical
// NCHW-order [d0, channels, spatial0, spatial1]; channels-last packing puts the
// channel axis (index 1) at unit stride. Any other layout is a conv we have no
// kernel for, so decode fails closed on it.
bool isPackedChannelsLast(const TensorAttributesWrapper& t)
{
    const auto dims = t.dims();
    const auto strides = t.strides();
    if(dims.size() != 4 || strides.size() != 4)
    {
        return false;
    }
    const int64_t ch = dims[1];
    const int64_t sp0 = dims[2];
    const int64_t sp1 = dims[3];
    return strides[1] == 1 && strides[3] == ch && strides[2] == sp1 * ch
           && strides[0] == sp0 * sp1 * ch;
}

// numel * elemBytes, declining (nullopt) if it would exceed INT32_MAX -- the
// kernel takes each buffer size as an i32 for the hardware OOB clamp, so a
// larger tensor must decline rather than launch with a truncated size.
std::optional<int64_t> bufferBytes(int64_t numel, int64_t elemBytes)
{
    if(numel <= 0 || elemBytes <= 0)
    {
        return std::nullopt;
    }
    const int64_t bytes = numel * elemBytes;
    if(bytes > std::numeric_limits<int32_t>::max())
    {
        return std::nullopt;
    }
    return bytes;
}

} // namespace

std::optional<catalog::ProblemShape> ConvFpropAdapter::decode(const IGraph& graph) const
{
    // Single ConvolutionFwdAttributes node only (the op adapter matches one node).
    if(!graph.isValid() || graph.nodeCount() != 1)
    {
        return std::nullopt;
    }
    const auto& node = graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::ConvolutionFwdAttributes)
    {
        return std::nullopt;
    }
    const auto& attrs = node.attributesAs<data_objects::ConvolutionFwdAttributes>();

    const auto& tensorMap = graph.getTensorMap();
    auto findTensor = [&](int64_t uid) -> const data_objects::TensorAttributes* {
        auto it = tensorMap.find(uid);
        return it == tensorMap.end() ? nullptr : it->second;
    };

    const auto* xTensor = findTensor(attrs.x_tensor_uid());
    const auto* wTensor = findTensor(attrs.w_tensor_uid());
    const auto* yTensor = findTensor(attrs.y_tensor_uid());
    if(xTensor == nullptr || wTensor == nullptr || yTensor == nullptr)
    {
        return std::nullopt;
    }

    // TensorAttributesWrapper::dims()/strides()/dataType() throw on null
    // dims/strides; a malformed graph should make the engine decline, not throw
    // out of isApplicable, so treat any decode failure as "not applicable".
    try
    {
        const TensorAttributesWrapper x(xTensor);
        const TensorAttributesWrapper w(wTensor);
        const TensorAttributesWrapper y(yTensor);

        // ---- Universal safety gates ----------------------------------------
        // Correctness invariants no forward-conv kernel can waive. 2-D conv only:
        // x=[N,C,Hi,Wi], w=[K,C/groups,R,S], y=[N,K,Ho,Wo], all rank 4.
        const auto xDims = x.dims();
        const auto wDims = w.dims();
        const auto yDims = y.dims();
        if(xDims.size() != 4 || wDims.size() != 4 || yDims.size() != 4)
        {
            return std::nullopt;
        }

        const int64_t nBatch = xDims[0];
        const int64_t cIn = xDims[1];
        const int64_t hIn = xDims[2];
        const int64_t wIn = xDims[3];
        const int64_t kOut = wDims[0];
        const int64_t cPerGroup = wDims[1];
        const int64_t rFilt = wDims[2];
        const int64_t sFilt = wDims[3];
        const int64_t hOut = yDims[2];
        const int64_t wOut = yDims[3];

        // Output must mirror [N, K, Ho, Wo].
        if(yDims[0] != nBatch || yDims[1] != kOut)
        {
            return std::nullopt;
        }
        // Every extent that indexes memory or acts as a divisor must be positive.
        if(nBatch <= 0 || cIn <= 0 || kOut <= 0 || hIn <= 0 || wIn <= 0 || rFilt <= 0 || sFilt <= 0
           || cPerGroup <= 0 || hOut <= 0 || wOut <= 0)
        {
            return std::nullopt;
        }
        // Grouping: the weight carries C/groups. A channel count the weight does
        // not evenly divide is a malformed grouping -- decline. groups is
        // published; per-kernel constraints pin it (the gfx1151 kernel -> 1).
        if(cIn % cPerGroup != 0)
        {
            return std::nullopt;
        }
        const int64_t groups = cIn / cPerGroup;

        // Channels-last packed layout on all three operands (NHWC / KRSC / NHWK) --
        // the exact strides the runtime kernel addresses with.
        if(!isPackedChannelsLast(x) || !isPackedChannelsLast(w) || !isPackedChannelsLast(y))
        {
            return std::nullopt;
        }

        // Spatial hyperparameters, ordered [H, W]. Symmetric padding only (the
        // runtime ABI carries one pad per axis).
        const auto* strideV = attrs.stride();
        const auto* preV = attrs.pre_padding();
        const auto* postV = attrs.post_padding();
        const auto* dilV = attrs.dilation();
        if(strideV == nullptr || preV == nullptr || postV == nullptr || dilV == nullptr)
        {
            return std::nullopt;
        }
        if(strideV->size() != 2 || preV->size() != 2 || postV->size() != 2 || dilV->size() != 2)
        {
            return std::nullopt;
        }
        const int64_t sH = strideV->Get(0);
        const int64_t sW = strideV->Get(1);
        const int64_t pH = preV->Get(0);
        const int64_t pW = preV->Get(1);
        const int64_t dH = dilV->Get(0);
        const int64_t dW = dilV->Get(1);
        if(postV->Get(0) != pH || postV->Get(1) != pW)
        {
            return std::nullopt; // asymmetric padding: no runtime ABI for it
        }
        // Positive stride/dilation (they divide / scale the coordinate maps);
        // non-negative padding.
        if(sH <= 0 || sW <= 0 || dH <= 0 || dW <= 0 || pH < 0 || pW < 0)
        {
            return std::nullopt;
        }

        // Ho/Wo must equal the conv arithmetic for the decoded geometry; a graph
        // whose declared output extent disagrees is inconsistent -- fail closed.
        const int64_t expectHo = (hIn + 2 * pH - dH * (rFilt - 1) - 1) / sH + 1;
        const int64_t expectWo = (wIn + 2 * pW - dW * (sFilt - 1) - 1) / sW + 1;
        if(expectHo != hOut || expectWo != wOut)
        {
            return std::nullopt;
        }

        // A single supported element dtype across x/w/y.
        const auto dtype = providerDtype(x.dataType());
        if(!dtype.has_value() || w.dataType() != x.dataType() || y.dataType() != x.dataType())
        {
            return std::nullopt;
        }

        catalog::ProblemShape shape;
        shape.emplace("dtype", catalog::ShapeValue{*dtype});

        // Buffer sizes in bytes (i32-clamped), used both as the OOB-clamp scalar
        // args and as the decode-time overflow decline.
        const auto elemBytes = catalog::elementSizeBytes(shape);
        if(!elemBytes.has_value())
        {
            return std::nullopt;
        }
        const auto aBytes = bufferBytes(nBatch * cIn * hIn * wIn, *elemBytes);
        const auto bBytes = bufferBytes(kOut * cPerGroup * rFilt * sFilt, *elemBytes);
        const auto dBytes = bufferBytes(nBatch * kOut * hOut * wOut, *elemBytes);
        if(!aBytes.has_value() || !bBytes.has_value() || !dBytes.has_value())
        {
            return std::nullopt;
        }

        // ---- Published facts ------------------------------------------------
        // Raw geometry (the runtime scalar ABI).
        shape.emplace("N", catalog::ShapeValue{nBatch});
        shape.emplace("C", catalog::ShapeValue{cIn});
        shape.emplace("K", catalog::ShapeValue{kOut});
        shape.emplace("Hi", catalog::ShapeValue{hIn});
        shape.emplace("Wi", catalog::ShapeValue{wIn});
        shape.emplace("R", catalog::ShapeValue{rFilt});
        shape.emplace("S", catalog::ShapeValue{sFilt});
        shape.emplace("Ho", catalog::ShapeValue{hOut});
        shape.emplace("Wo", catalog::ShapeValue{wOut});
        shape.emplace("sH", catalog::ShapeValue{sH});
        shape.emplace("sW", catalog::ShapeValue{sW});
        shape.emplace("pH", catalog::ShapeValue{pH});
        shape.emplace("pW", catalog::ShapeValue{pW});
        shape.emplace("dH", catalog::ShapeValue{dH});
        shape.emplace("dW", catalog::ShapeValue{dW});
        shape.emplace("groups", catalog::ShapeValue{groups});
        shape.emplace("conv_mode", catalog::ShapeValue{static_cast<int64_t>(attrs.conv_mode())});

        // Buffer sizes.
        shape.emplace("A_bytes", catalog::ShapeValue{*aBytes});
        shape.emplace("B_bytes", catalog::ShapeValue{*bBytes});
        shape.emplace("D_bytes", catalog::ShapeValue{*dBytes});

        // Derived implicit-GEMM extents (grid + selection convenience).
        shape.emplace("M", catalog::ShapeValue{nBatch * hOut * wOut});
        shape.emplace("N_gemm", catalog::ShapeValue{kOut / groups});
        shape.emplace("K_gemm", catalog::ShapeValue{rFilt * sFilt * cPerGroup});
        return shape;
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(
            "aot-catalog(conv_fprop): declining graph, decode failed: " << e.what());
        return std::nullopt;
    }
}

catalog::LaunchBindings ConvFpropAdapter::buildBindings(const IGraph& graph,
                                                        const catalog::ProblemShape& problem,
                                                        const catalog::KernelEntry& kernel) const
{
    // The adapter emits a SUPERSET of named quantities; each family's
    // args_signature selects and orders the subset its kernel takes (launch::
    // bindArgs resolves by name and fails closed on an unemitted name). The
    // gfx1151 kernel's ABI is [A,B,D, A_bytes,B_bytes,D_bytes, N,C,K,Hi,Wi,R,S,
    // sH,sW,pH,pW,dH,dW]; a grouped/split-k CDNA kernel names its own subset.
    (void)kernel;

    const auto& node = graph.getNodeWrapper(0);
    const auto& attrs = node.attributesAs<data_objects::ConvolutionFwdAttributes>();

    catalog::LaunchBindings bindings;
    bindings.pointerUids.emplace("A", attrs.x_tensor_uid());
    bindings.pointerUids.emplace("B", attrs.w_tensor_uid());
    bindings.pointerUids.emplace("D", attrs.y_tensor_uid());

    // Buffer-resource sizes for the hardware OOB clamp.
    bindings.scalars.emplace("A_bytes", catalog::ScalarValue{problemInt(problem, "A_bytes")});
    bindings.scalars.emplace("B_bytes", catalog::ScalarValue{problemInt(problem, "B_bytes")});
    bindings.scalars.emplace("D_bytes", catalog::ScalarValue{problemInt(problem, "D_bytes")});

    // Runtime geometry scalars; the kernel derives Ho/Wo/M/N_gemm/K_gemm from
    // these, so only the raw geometry crosses the ABI.
    for(const char* key : {"N", "C", "K", "Hi", "Wi", "R", "S", "sH", "sW", "pH", "pW", "dH", "dW"})
    {
        bindings.scalars.emplace(key, catalog::ScalarValue{problemInt(problem, key)});
    }
    return bindings;
}

launch::SymbolTable ConvFpropAdapter::gridSymbols(const catalog::ProblemShape& problem,
                                                  const catalog::KernelEntry& kernel) const
{
    // Superset of grid symbols; a family's grid formula references whichever it
    // needs (gfx1151: ceil_div(N_gemm,tile_n) x ceil_div(M,tile_m)) and extra
    // symbols are harmless.
    (void)kernel;

    launch::SymbolTable symbols;
    for(const char* key :
        {"M", "N_gemm", "K_gemm", "N", "C", "K", "Hi", "Wi", "R", "S", "Ho", "Wo"})
    {
        symbols.emplace(key, problemInt(problem, key));
    }
    return symbols;
}

} // namespace aot_catalog_engine::ops
