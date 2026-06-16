// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkDslConvParamParser.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <stdexcept>

namespace ck_dsl_plugin {
namespace CkDslConvParamParser {

namespace {
namespace fb = hipdnn_flatbuffers_sdk::data_objects;
std::string mapDataType(fb::DataType dt) {
    switch (dt) {
        case fb::DataType::HALF:
            return "fp16";
        case fb::DataType::BFLOAT16:
            return "bf16";
        default:
            return "";
    }
}
const fb::TensorAttributes* lookup(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& g,
                                   int64_t uid) {
    auto it = g.getTensorMap().find(uid);
    return it != g.getTensorMap().end() ? it->second : nullptr;
}
int v2(const ::flatbuffers::Vector<int64_t>* v, int i, int dflt) {
    return (v && (int)v->size() > i) ? (int)v->Get(i) : dflt;
}
}  // namespace

bool isConvGraph(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& g) {
    if (g.nodeCount() != 1) return false;
    return g.getNode(0).attributes_type() == fb::NodeAttributes::ConvolutionFwdAttributes;
}

ParsedConvParams parseConvGraph(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& g) {
    const auto* a = g.getNode(0).attributes_as_ConvolutionFwdAttributes();
    if (!a) throw std::runtime_error("CkDslConv: not a ConvolutionFwdAttributes node");
    ParsedConvParams p;
    p.x_uid = a->x_tensor_uid();
    p.w_uid = a->w_tensor_uid();
    p.y_uid = a->y_tensor_uid();
    const auto* x = lookup(g, p.x_uid);
    const auto* w = lookup(g, p.w_uid);
    if (!x || !w) throw std::runtime_error("CkDslConv: missing X/W tensors");
    p.dtype = mapDataType(x->data_type());
    if (p.dtype.empty()) throw std::runtime_error("CkDslConv: unsupported dtype");
    // hipDNN uses cuDNN-style logical dims: X=[N,C,H,W], W=[K,C,R,S]
    // (NHWC/KRSC physical is expressed via strides, which the kernel consumes).
    const auto* xd = x->dims();
    const auto* wd = w->dims();
    if (!xd || !wd || xd->size() < 4 || wd->size() < 4)
        throw std::runtime_error("CkDslConv: expected rank-4 X[N,C,H,W] and W[K,C/G,R,S]");
    p.N  = xd->Get(0);
    p.C  = xd->Get(1);
    p.Hi = xd->Get(2);
    p.Wi = xd->Get(3);
    p.K  = wd->Get(0);
    // W=[K, C/G, R, S]: group count is implicit in the per-group channel dim.
    {
        const auto cpg = wd->Get(1);
        if (cpg <= 0 || p.C % cpg != 0)
            throw std::runtime_error("CkDslConv: weight dim[1] must be a positive divisor of C");
        p.G = static_cast<int>(std::max((int64_t)1, p.C / cpg));
        if (p.K % p.G != 0)
            throw std::runtime_error("CkDslConv: K must be divisible by G (grouped conv)");
    }
    p.R  = wd->Get(2);
    p.S  = wd->Get(3);
    p.sH = v2(a->stride(), 0, 1);
    p.sW = v2(a->stride(), 1, 1);
    p.pH = v2(a->pre_padding(), 0, 0);
    p.pW = v2(a->pre_padding(), 1, 0);
    p.dH = v2(a->dilation(), 0, 1);
    p.dW = v2(a->dilation(), 1, 1);
    return p;
}

ck_dsl::Problem buildProblem(const ParsedConvParams& p, const std::string& arch) {
    ck_dsl::Problem prob;
    prob.op = "conv";
    prob.dtype = p.dtype;
    prob.layout = "NHWC";
    prob.arch = arch;
    prob.M = (long)p.N * p.Ho() * p.Wo();
    prob.N = p.K;
    prob.K = (long)p.R * p.S * (p.C / p.G);  // per-group reduction dim
    // Conv-specific dims for the 97-feature ML extractor.
    prob.conv_N = p.N;
    prob.conv_C = p.C;
    prob.conv_K = p.K;
    prob.conv_G = p.G;
    prob.Hi = p.Hi;
    prob.Wi = p.Wi;
    prob.Y = p.R;
    prob.X = p.S;
    prob.stride_h = p.sH;
    prob.stride_w = p.sW;
    prob.pad_h = p.pH;
    prob.pad_w = p.pW;
    prob.dilation_h = p.dH;
    prob.dilation_w = p.dW;
    return prob;
}

}  // namespace CkDslConvParamParser
}  // namespace ck_dsl_plugin
