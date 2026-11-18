// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/SdpaGraphMutators.hpp>

#include <functional>
#include <ostream>
#include <string>
#include <vector>

namespace asm_sdpa_engine
{

// One property of a forward SDPA node that neither forward engine serves. `mutate` sets only
// that property on an otherwise legal graph.
struct SdpaFwdUnservedAttributeCase
{
    std::string name;
    std::function<void(hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesT&,
                       hipdnn_flatbuffers_sdk::data_objects::GraphT&)>
        mutate;
};

inline std::ostream& operator<<(std::ostream& os, const SdpaFwdUnservedAttributeCase& value)
{
    return os << value.name;
}

inline flatbuffers::FlatBufferBuilder
    applyUnservedAttribute(flatbuffers::FlatBufferBuilder&& base,
                           const SdpaFwdUnservedAttributeCase& attributeCase)
{
    return hipdnn_test_sdk::utilities::mutateGraph(
        base, [&attributeCase](hipdnn_flatbuffers_sdk::data_objects::GraphT& graph) {
            attributeCase.mutate(hipdnn_test_sdk::utilities::sdpa::detail::getSdpaAttributes(graph),
                                 graph);
        });
}

// seed and offset feed dropout only, so with dropout_probability 0 they are inert and both
// forward engines accept them.
inline SdpaFwdUnservedAttributeCase seedAndOffsetAtZeroDropoutCase()
{
    using hipdnn_flatbuffers_sdk::data_objects::GraphT;
    using hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesT;
    using hipdnn_test_sdk::utilities::sdpa::detail::appendIndexTensor;

    return {"SeedAndOffsetAtZeroDropout", [](SdpaAttributesT& a, GraphT& g) {
                a.dropout_probability = 0.f;
                a.seed_tensor_uid = appendIndexTensor(g, "seed");
                a.offset_tensor_uid = appendIndexTensor(g, "offset");
            }};
}

// Attributes both forward engines decline: dropout mask, scale and rng_dump operands, block
// mask, max and sum_exp outputs, and paged-KV length. The frontend and the cuDNN-compatibility
// layer leave each at its default (tensor absent, max_seq_len_kv unset).
inline std::vector<SdpaFwdUnservedAttributeCase> commonUnservedFwdAttributeCases()
{
    using hipdnn_flatbuffers_sdk::data_objects::GraphT;
    using hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesT;
    using hipdnn_test_sdk::utilities::sdpa::detail::appendIndexTensor;

    return {{"DropoutMaskTensor",
             [](SdpaAttributesT& a, GraphT& g) {
                 a.dropout_mask_tensor_uid = appendIndexTensor(g, "dropout_mask");
             }},
            {"DropoutScaleTensor",
             [](SdpaAttributesT& a, GraphT& g) {
                 a.dropout_scale_tensor_uid = appendIndexTensor(g, "dropout_scale");
             }},
            {"RngDumpTensor",
             [](SdpaAttributesT& a, GraphT& g) {
                 a.rng_dump_tensor_uid = appendIndexTensor(g, "rng_dump");
             }},
            {"BlockMaskTensor",
             [](SdpaAttributesT& a, GraphT& g) {
                 a.block_mask_tensor_uid = appendIndexTensor(g, "block_mask");
             }},
            {"MaxTensor",
             [](SdpaAttributesT& a, GraphT& g) { a.max_tensor_uid = appendIndexTensor(g, "max"); }},
            {"SumExpTensor",
             [](SdpaAttributesT& a, GraphT& g) {
                 a.sum_exp_tensor_uid = appendIndexTensor(g, "sum_exp");
             }},
            {"MaxSeqLenKv", [](SdpaAttributesT& a, GraphT&) { a.max_seq_len_kv = 256; }}};
}

// Attributes the ASM forward builder declines on top of the common set: dropout at a non-zero
// probability, page tables, and a stats tensor without generate_stats.
inline std::vector<SdpaFwdUnservedAttributeCase> asmUnservedFwdAttributeCases()
{
    using hipdnn_flatbuffers_sdk::data_objects::GraphT;
    using hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesT;
    using hipdnn_test_sdk::utilities::sdpa::detail::appendIndexTensor;

    auto cases = commonUnservedFwdAttributeCases();
    cases.push_back({"DropoutProbabilityWithSeedAndOffset", [](SdpaAttributesT& a, GraphT& g) {
                         a.dropout_probability = 0.1f;
                         a.seed_tensor_uid = appendIndexTensor(g, "seed");
                         a.offset_tensor_uid = appendIndexTensor(g, "offset");
                     }});
    cases.push_back({"PageTableK", [](SdpaAttributesT& a, GraphT& g) {
                         a.page_table_k_tensor_uid = appendIndexTensor(g, "page_table_k");
                     }});
    cases.push_back({"PageTableV", [](SdpaAttributesT& a, GraphT& g) {
                         a.page_table_v_tensor_uid = appendIndexTensor(g, "page_table_v");
                     }});
    cases.push_back({"StatsTensorWithGenerateStatsUnset", [](SdpaAttributesT& a, GraphT& g) {
                         a.generate_stats.reset();
                         a.stats_tensor_uid = appendIndexTensor(g, "stats");
                     }});
    cases.push_back({"StatsTensorWithGenerateStatsFalse", [](SdpaAttributesT& a, GraphT& g) {
                         a.generate_stats = false;
                         a.stats_tensor_uid = appendIndexTensor(g, "stats");
                     }});
    return cases;
}

// Inert attributes the ASM forward builder accepts: the implementation hint selects between
// frontend lowerings and does not change what the kernels compute.
inline std::vector<SdpaFwdUnservedAttributeCase> asmInertFwdAttributeCases()
{
    using hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation;
    using hipdnn_flatbuffers_sdk::data_objects::GraphT;
    using hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesT;

    return {seedAndOffsetAtZeroDropoutCase(),
            {"ImplementationComposite",
             [](SdpaAttributesT& a, GraphT&) {
                 a.implementation = AttentionImplementation::COMPOSITE;
             }},
            {"ImplementationUnified", [](SdpaAttributesT& a, GraphT&) {
                 a.implementation = AttentionImplementation::UNIFIED;
             }}};
}

} // namespace asm_sdpa_engine
