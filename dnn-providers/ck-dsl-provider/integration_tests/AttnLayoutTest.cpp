// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-only unit test for SDPA metadata parsing in CkDslAttnParamParser.
// hipDNN SDPA tensors are logically [B,H,S,D]; the stride vector is indexed in
// that same logical order. The ck_dsl attention kernel consumes physical BSHD
// storage and must decline physical BHSD-contiguous storage.
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <cstdint>
#include <cstdio>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <vector>

#include "engines/CkDslAttnParamParser.hpp"

namespace parser = ck_dsl_plugin::CkDslAttnParamParser;

namespace {
int g_fail = 0;
constexpr int64_t B = 2;
constexpr int64_t H = 8;
constexpr int64_t Hkv = 2;
constexpr int64_t S = 64;
constexpr int64_t D = 128;

void check(bool cond, const char* what) {
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++g_fail;
}

parser::ParsedAttnParams parseGraphWithQStrides(const std::vector<int64_t>& qStrides) {
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;

    const std::vector<int64_t> qDims{B, H, S, D};
    const std::vector<int64_t> kvDims{B, Hkv, S, D};
    const std::vector<int64_t> oDims{B, H, S, D};
    const std::vector<int64_t> kvStrides{S * Hkv * D, D, Hkv * D, 1};

    auto q = CreateTensorAttributesDirect(builder, 1, "q", DataType::HALF, &qStrides, &qDims);
    auto k = CreateTensorAttributesDirect(builder, 2, "k", DataType::HALF, &kvStrides, &kvDims);
    auto v = CreateTensorAttributesDirect(builder, 3, "v", DataType::HALF, &kvStrides, &kvDims);
    auto o = CreateTensorAttributesDirect(builder, 4, "o", DataType::HALF, &qStrides, &oDims);
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors{q, k, v, o};

    auto sdpa = CreateSdpaAttributes(builder, 1, 2, 3, 4);
    auto node = CreateNodeDirect(builder, "sdpa", DataType::FLOAT, NodeAttributes::SdpaAttributes,
                                 sdpa.Union());
    std::vector<flatbuffers::Offset<Node>> nodes{node};

    auto graph = CreateGraphDirect(builder, "sdpa_graph", DataType::FLOAT, DataType::HALF,
                                   DataType::HALF, &tensors, &nodes);
    builder.Finish(graph);

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper(builder.GetBufferPointer(),
                                                                       builder.GetSize());
    check(wrapper.isValid(), "test graph flatbuffer is valid");
    return parser::parseSdpaGraph(wrapper);
}

void run() {
    // Logical [B,H,S,D], physical BSHD: H axis is inner to S.
    check(!parser::isPhysicalBhsdLayout(/*strideH=*/D, /*strideS=*/H * D),
          "BSHD physical strides {D,H*D} are not BHSD");

    // Logical [B,H,S,D], physical BHSD contiguous: S axis is inner to H.
    check(parser::isPhysicalBhsdLayout(/*strideH=*/S * D, /*strideS=*/D),
          "BHSD physical strides {S*D,D} are detected");

    // Ambiguous equal strides are not classified as physical BHSD.
    check(!parser::isPhysicalBhsdLayout(/*strideH=*/D, /*strideS=*/D),
          "equal H/S strides are not classified as BHSD");

    auto bshd = parseGraphWithQStrides({S * H * D, D, H * D, 1});
    check(bshd.batch == B && bshd.nhead_q == H && bshd.seqlen_q == S && bshd.hdim_q == D,
          "parser reads logical dims as [B,H,S,D]");
    check(!bshd.is_bhsd, "parser accepts physical BSHD strides in logical [B,H,S,D] order");

    auto bhsd = parseGraphWithQStrides({H * S * D, S * D, D, 1});
    check(bhsd.is_bhsd, "parser detects physical BHSD strides in logical [B,H,S,D] order");
}
}  // namespace

int main() {
    std::printf("=== ck-dsl-provider SDPA layout detector unit test ===\n");
    run();
    std::printf(g_fail == 0 ? "ALL PASS\n" : "FAILURES (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
