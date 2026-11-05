// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

#include "engines/kernel_ingestor_engine/packs/SdpaGraphMatch.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestSdpaGraphMatch.cpp
 * @brief sdpa_graph::readProblem, the checked Q/K reader both unified packs share.
 *
 * kernel_match and prepare() call readProblem on a graph graph_match may never have seen,
 * so it must return nullopt, not index an axis, for every operand problemFor cannot read:
 * a uid naming no tensor, a rank other than 4, null dims, a non-positive extent, and a
 * virtual or pass-by-value tensor. Each case breaks Q or K alone, so a reader that checks
 * only one of them fails here. The graphs hold the two tensors and no node: readProblem
 * reads tensors by uid and never looks at a node.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using sdpa_graph::readProblem;

/// Distinct, and none a small ordinal, so a swapped uid cannot read as correct.
constexpr int64_t Q_UID = 11;
constexpr int64_t K_UID = 22;
constexpr int64_t MISSING_UID = 77;

/// One operand as the graph serializes it. Defaults are a well-formed BSHD tensor.
struct Operand
{
    std::optional<std::vector<int64_t>> dims;
    std::optional<std::vector<int64_t>> strides;
    bool isVirtual = false;
    bool passByValue = false;
};

Operand operand(std::vector<int64_t> dims)
{
    const int64_t heads = dims.size() > 1 ? dims[1] : 1;
    const int64_t headSize = dims.empty() ? 1 : dims.back();
    std::vector<int64_t> strides(dims.size(), 1);
    if(dims.size() == 4)
    {
        strides = {dims[2] * heads * headSize, headSize, heads * headSize, 1};
    }
    return Operand{std::move(dims), std::move(strides)};
}

/// Q (2, 8, 16, 128) and K (2, 2, 256, 128): every field of the problem distinct.
Operand defaultQ()
{
    return operand({2, 8, 16, 128});
}

Operand defaultK()
{
    return operand({2, 2, 256, 128});
}

GraphFixture graphOf(const Operand& q, const Operand& k)
{
    flatbuffers::FlatBufferBuilder builder;
    const auto tensorFor = [&](int64_t uid, const Operand& op) {
        return data_objects::CreateTensorAttributesDirect(builder,
                                                          uid,
                                                          nullptr,
                                                          data_objects::DataType::BFLOAT16,
                                                          op.strides.has_value() ? &*op.strides
                                                                                 : nullptr,
                                                          op.dims.has_value() ? &*op.dims : nullptr,
                                                          op.isVirtual,
                                                          data_objects::TensorValue::NONE,
                                                          0,
                                                          op.passByValue);
    };
    const std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors{
        tensorFor(Q_UID, q), tensorFor(K_UID, k)};
    const std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    auto name = builder.CreateString("sdpa_graph_read_problem_test");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);
    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);
    builder.Finish(graphBuilder.Finish());
    return GraphFixture(std::move(builder));
}

/// A named Q or K the reader must refuse.
struct BadOperand
{
    std::string name;
    Operand operand;
};

std::vector<BadOperand> badOperands()
{
    Operand nullDims = operand({2, 8, 16, 128});
    nullDims.dims.reset();
    Operand nullStrides = operand({2, 8, 16, 128});
    nullStrides.strides.reset();
    Operand virtualTensor = operand({2, 8, 16, 128});
    virtualTensor.isVirtual = true;
    Operand passByValue = operand({2, 8, 16, 128});
    passByValue.passByValue = true;
    return {
        {"rank 3", operand({8, 16, 128})},
        {"rank 5", operand({2, 8, 16, 128, 1})},
        {"null dims", nullDims},
        {"null strides", nullStrides},
        {"zero extent", operand({2, 8, 0, 128})},
        {"negative extent", operand({2, -8, 16, 128})},
        {"virtual", virtualTensor},
        {"pass by value", passByValue},
    };
}

} // namespace

TEST(TestSdpaGraphMatch, ReadsTheProblemFromQAndK)
{
    const auto graph = graphOf(defaultQ(), defaultK());
    const auto problem = readProblem(graph.context(), Q_UID, K_UID);
    ASSERT_TRUE(problem.has_value());
    EXPECT_EQ(problem->batch, 2);
    EXPECT_EQ(problem->numQueryHeads, 8);
    EXPECT_EQ(problem->seqLenQ, 16);
    EXPECT_EQ(problem->headSize, 128);
    EXPECT_EQ(problem->numKvHeads, 2);
    EXPECT_EQ(problem->seqLenKv, 256);
    EXPECT_EQ(problem->dataType, data_objects::DataType::BFLOAT16);
}

TEST(TestSdpaGraphMatch, DeclinesAUidNamingNoTensor)
{
    const auto graph = graphOf(defaultQ(), defaultK());
    EXPECT_FALSE(readProblem(graph.context(), MISSING_UID, K_UID).has_value());
    EXPECT_FALSE(readProblem(graph.context(), Q_UID, MISSING_UID).has_value());
}

TEST(TestSdpaGraphMatch, DeclinesAMalformedQ)
{
    for(const auto& bad : badOperands())
    {
        SCOPED_TRACE(bad.name);
        const auto graph = graphOf(bad.operand, defaultK());
        EXPECT_FALSE(readProblem(graph.context(), Q_UID, K_UID).has_value());
    }
}

TEST(TestSdpaGraphMatch, DeclinesAMalformedK)
{
    for(const auto& bad : badOperands())
    {
        SCOPED_TRACE(bad.name);
        const auto graph = graphOf(defaultQ(), bad.operand);
        EXPECT_FALSE(readProblem(graph.context(), Q_UID, K_UID).has_value());
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
