// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// libFuzzer entry point (b): the matcher itself. Arbitrary bytes are decoded by
// the bounded PatternCodec::deserialize; any pattern that survives is run by the
// Matcher against a fixed valid graph. This exercises the matcher's hot path
// (candidate walk, unification, constraint + predicate evaluation) on adversarial
// but structurally-decodable patterns -- ids, role indices, axes, counts all
// attacker-chosen within the decoder's bounds. A crash, hang, sanitizer report,
// or step-budget runaway is a real defect.
//
// Build (Linux/CI clang): see fuzz_fromjson.cpp; target graph_matcher_fuzz_match.

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <cstddef>
#include <cstdint>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_graph_matcher/GraphView.hpp>
#include <hipdnn_graph_matcher/Matcher.hpp>
#include <hipdnn_graph_matcher/PatternCodec.hpp>
#include <vector>

namespace {
namespace data = hipdnn_flatbuffers_sdk::data_objects;
namespace fb = hipdnn_flatbuffers_sdk::flatbuffer_utilities;

// A fixed, valid single-matmul graph the fuzzed patterns are matched against.
const std::vector<uint8_t>& fixedGraphBytes() {
    static const std::vector<uint8_t> bytes = [] {
        flatbuffers::FlatBufferBuilder b;
        std::vector<flatbuffers::Offset<data::TensorAttributes>> tensors;
        const std::vector<int64_t> dims{4, 8};
        const std::vector<int64_t> strides{8, 1};
        for (int64_t uid = 1; uid <= 3; ++uid) {
            tensors.push_back(data::CreateTensorAttributesDirect(b, uid, "t", data::DataType::FLOAT,
                                                                 &strides, &dims));
        }
        auto mm = data::CreateMatmulAttributes(b, 1, 2, 3);
        std::vector<flatbuffers::Offset<data::Node>> nodes;
        nodes.push_back(data::CreateNodeDirect(b, "matmul", data::DataType::FLOAT,
                                               data::NodeAttributes::MatmulAttributes, mm.Union()));
        b.Finish(data::CreateGraphDirect(b, "fuzz", data::DataType::FLOAT, data::DataType::FLOAT,
                                         data::DataType::FLOAT, &tensors, &nodes));
        return std::vector<uint8_t>(b.GetBufferPointer(), b.GetBufferPointer() + b.GetSize());
    }();
    return bytes;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    using namespace hipdnn::graph_matcher;
    auto decoded = PatternCodec::deserialize(data, size);
    if (!decoded.ok) {
        return 0;
    }
    const auto& gbytes = fixedGraphBytes();
    fb::GraphWrapper graph(gbytes.data(), gbytes.size());
    GraphView view(graph);
    MatchOptions options;
    options.stepBudget = 1u << 14;  // tight: a decoded-but-pathological pattern must not hang
    volatile auto r = Matcher::match(decoded.pattern, view, options).matched;
    (void)r;
    return 0;
}
