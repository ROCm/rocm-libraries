// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The CPU and GPU pointwise references list their modes and datatype tuples in unrelated places,
// and the GPU key has no mode. TestPointwiseReferenceParity asks both executors about every
// (mode, datatype tuple) either registry can express. Each difference has to be listed in
// knownGaps(), and each listed gap has to still be a difference. TestGpuPointwiseExecutor runs one
// mode of every tuple the GPU registers and compares it with the CPU reference.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/PointwiseValidation.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PointwiseSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/pointwise/CpuReferencePointwise.hpp>

#include "harness/CpuReferenceGraphExecutorAdapter.hpp"
#include "harness/gpu-graph-executor/GpuReferenceGraphExecutor.hpp"
#include "harness/gpu-graph-executor/detail/GpuPointwiseSignatureKey.hpp"
#include "harness/reference-validation/ReferenceOpCoverage.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_data_sdk::utilities::Tensor;
using hipdnn_integration_tests::CpuReferenceGraphExecutorAdapter;
using hipdnn_integration_tests::ReferenceExecutorType;
using hipdnn_integration_tests::bundle::referenceSupportedOps;
using hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor;
using hipdnn_integration_tests::gpu_graph_executor::detail::GpuPointwiseSignatureKey;
using hipdnn_test_sdk::detail::PointwiseSignatureKey;
using hipdnn_test_sdk::utilities::CpuFpReferenceValidation;
using hipdnn_test_sdk::utilities::CpuReferencePointwiseImpl;
using hipdnn_test_sdk::utilities::datatypeToNativeVariant;
using hipdnn_test_sdk::utilities::pointwise::getTolerance;

using HalfType = hipdnn_data_sdk::types::half;
using BFloat16Type = hipdnn_data_sdk::types::bfloat16;

namespace
{

constexpr DataType K_FP32 = DataType::FLOAT;
constexpr DataType K_FP16 = DataType::HALF;
constexpr DataType K_BFP16 = DataType::BFLOAT16;
constexpr DataType K_FP64 = DataType::DOUBLE;
constexpr DataType K_INT32 = DataType::INT32;
constexpr DataType K_FP8_E4M3 = DataType::FP8_E4M3;
constexpr DataType K_FP8_E5M2 = DataType::FP8_E5M2;
constexpr DataType K_NONE = DataType::UNSET;

// datatypeToNativeVariant() holds more types than the GPU kernels take
template <typename T>
constexpr bool IS_GPU_TYPE
    = std::is_same_v<T, float> || std::is_same_v<T, HalfType> || std::is_same_v<T, BFloat16Type>;

// What the references dispatch on. An operand a mode does not use is UNSET; with the mode UNSET it
// is a datatype signature.
struct PointwiseTuple
{
    PointwiseMode mode = PointwiseMode::UNSET;
    DataType in0 = DataType::UNSET;
    DataType in1 = DataType::UNSET;
    DataType in2 = DataType::UNSET;
    DataType out = DataType::UNSET;
    DataType compute = DataType::UNSET;
};

bool operator<(const PointwiseTuple& lhs, const PointwiseTuple& rhs)
{
    return std::tie(lhs.mode, lhs.in0, lhs.in1, lhs.in2, lhs.out, lhs.compute)
           < std::tie(rhs.mode, rhs.in0, rhs.in1, rhs.in2, rhs.out, rhs.compute);
}

bool operator==(const PointwiseTuple& lhs, const PointwiseTuple& rhs)
{
    return std::tie(lhs.mode, lhs.in0, lhs.in1, lhs.in2, lhs.out, lhs.compute)
           == std::tie(rhs.mode, rhs.in0, rhs.in1, rhs.in2, rhs.out, rhs.compute);
}

PointwiseTuple withoutMode(const PointwiseTuple& tuple)
{
    PointwiseTuple signature = tuple;
    signature.mode = PointwiseMode::UNSET;
    return signature;
}

PointwiseTuple unarySignature(DataType in, DataType out, DataType compute)
{
    return {PointwiseMode::UNSET, in, K_NONE, K_NONE, out, compute};
}

PointwiseTuple binarySignature(DataType in0, DataType in1, DataType out, DataType compute)
{
    return {PointwiseMode::UNSET, in0, in1, K_NONE, out, compute};
}

// A mode, a datatype signature or a full tuple, for failure messages.
std::string describe(const PointwiseTuple& tuple)
{
    std::ostringstream text;
    if(tuple.mode != PointwiseMode::UNSET)
    {
        text << EnumNamePointwiseMode(tuple.mode);
    }
    if(tuple.in0 != DataType::UNSET)
    {
        text << "(in0=" << EnumNameDataType(tuple.in0);
        if(tuple.in1 != DataType::UNSET)
        {
            text << ", in1=" << EnumNameDataType(tuple.in1);
        }
        if(tuple.in2 != DataType::UNSET)
        {
            text << ", in2=" << EnumNameDataType(tuple.in2);
        }
        text << ", out=" << EnumNameDataType(tuple.out)
             << ", compute=" << EnumNameDataType(tuple.compute) << ")";
    }
    return text.str();
}

constexpr int64_t IN0_UID = 1;
constexpr int64_t OUT_UID = 2;
constexpr int64_t IN1_UID = 3;
constexpr int64_t IN2_UID = 4;

// A single-node pointwise graph for `tuple`; `dims` is the first input and the output,
// `operandDims` the other inputs. createPointwiseGraph() gives input and output one datatype, so it
// cannot express a mixed tuple.
flatbuffers::FlatBufferBuilder buildGraph(const PointwiseTuple& tuple,
                                          const std::vector<int64_t>& dims,
                                          const std::vector<int64_t>& operandDims)
{
    flatbuffers::FlatBufferBuilder builder;
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    const auto operandStrides = hipdnn_data_sdk::utilities::generateStrides(operandDims);

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(
        CreateTensorAttributesDirect(builder, IN0_UID, "in0", tuple.in0, &strides, &dims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, OUT_UID, "out", tuple.out, &strides, &dims));

    flatbuffers::Optional<int64_t> in1Uid = flatbuffers::nullopt;
    if(tuple.in1 != DataType::UNSET)
    {
        tensors.push_back(CreateTensorAttributesDirect(
            builder, IN1_UID, "in1", tuple.in1, &operandStrides, &operandDims));
        in1Uid = IN1_UID;
    }

    flatbuffers::Optional<int64_t> in2Uid = flatbuffers::nullopt;
    if(tuple.in2 != DataType::UNSET)
    {
        tensors.push_back(CreateTensorAttributesDirect(
            builder, IN2_UID, "in2", tuple.in2, &operandStrides, &operandDims));
        in2Uid = IN2_UID;
    }

    // The CPU declines GEN_INDEX without an axis, whatever its datatypes.
    flatbuffers::Optional<int64_t> axis = flatbuffers::nullopt;
    if(tuple.mode == PointwiseMode::GEN_INDEX)
    {
        axis = int64_t{0};
    }

    const auto attributes = CreatePointwiseAttributes(builder,
                                                      tuple.mode,
                                                      flatbuffers::nullopt,
                                                      flatbuffers::nullopt,
                                                      flatbuffers::nullopt,
                                                      axis,
                                                      IN0_UID,
                                                      in1Uid,
                                                      in2Uid,
                                                      OUT_UID);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "pointwise",
                                     tuple.compute,
                                     NodeAttributes::PointwiseAttributes,
                                     attributes.Union()));

    builder.Finish(CreateGraphDirect(builder,
                                     "ParityProbe",
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     &tensors,
                                     &nodes));
    return builder;
}

// Every key the CPU registry holds, mode included.
std::set<PointwiseTuple> cpuRegistryTuples()
{
    std::set<PointwiseTuple> tuples;
    for(const auto& entry : PointwiseSignatureKey::getPlanBuilders())
    {
        const auto& key = entry.first;
        tuples.insert(PointwiseTuple{key.operation,
                                     key.inputDataType,
                                     key.input1DataType,
                                     key.input2DataType,
                                     key.outputDataType,
                                     key.computeDataType});
    }
    return tuples;
}

// The GPU key has no mode, so each key is crossed with every mode of its arity; the plan builders
// then say which of those they take. It has no third operand either, so no ternary mode appears.
std::set<PointwiseTuple> gpuCandidateTuples()
{
    std::set<PointwiseTuple> tuples;
    for(const auto& entry : GpuPointwiseSignatureKey::getPlanBuilders())
    {
        const auto& key = entry.first;
        const bool unary = key.input1DataType == DataType::UNSET;
        for(const auto mode : EnumValuesPointwiseMode())
        {
            if(unary ? hipdnn_flatbuffers_sdk::utilities::isUnaryPointwiseMode(mode)
                     : hipdnn_flatbuffers_sdk::utilities::isBinaryPointwiseMode(mode))
            {
                tuples.insert(PointwiseTuple{mode,
                                             key.input0DataType,
                                             key.input1DataType,
                                             K_NONE,
                                             key.outputDataType,
                                             key.computeDataType});
            }
        }
    }
    return tuples;
}

// One mode of each arity stands in for all of them: NEG for unary tuples, SUB for binary ones.
std::vector<PointwiseTuple> executedTuples()
{
    std::vector<PointwiseTuple> tuples;
    for(const auto& tuple : gpuCandidateTuples())
    {
        if(tuple.mode == PointwiseMode::NEG || tuple.mode == PointwiseMode::SUB)
        {
            tuples.push_back(tuple);
        }
    }
    return tuples;
}

// What each executor accepts, out of every tuple either registry can express.
struct Accepted
{
    std::set<PointwiseTuple> cpu;
    std::set<PointwiseTuple> gpu;
};

Accepted probeReferences()
{
    auto candidates = cpuRegistryTuples();
    const auto gpuTuples = gpuCandidateTuples();
    candidates.insert(gpuTuples.begin(), gpuTuples.end());

    const std::vector<int64_t> dims = {1, 2, 3, 4};
    CpuReferenceGraphExecutorAdapter cpuExecutor;
    GpuReferenceGraphExecutor gpuExecutor;
    Accepted accepted;
    for(const auto& tuple : candidates)
    {
        const auto graph = buildGraph(tuple, dims, dims);
        if(cpuExecutor.isApplicable(graph.GetBufferPointer(), graph.GetSize()))
        {
            accepted.cpu.insert(tuple);
        }
        if(gpuExecutor.isApplicable(graph.GetBufferPointer(), graph.GetSize()))
        {
            accepted.gpu.insert(tuple);
        }
    }
    return accepted;
}

enum class Reference
{
    CPU,
    GPU,
};

// Something the other reference accepts and `lacking` does not: a mode (every datatype tuple of
// it) when `pattern.mode` is set, otherwise a datatype signature (every mode of it).
struct Gap
{
    Reference lacking;
    PointwiseTuple pattern;
};

bool matches(const Gap& gap, const PointwiseTuple& tuple)
{
    if(gap.pattern.mode != PointwiseMode::UNSET)
    {
        return gap.pattern.mode == tuple.mode;
    }
    return gap.pattern == withoutMode(tuple);
}

const std::vector<Gap>& knownGaps()
{
    static const std::vector<Gap> s_gaps = {
        // No GPU kernel. The comparison, logical, GEN_INDEX and BINARY_SELECT modes also need a
        // BOOLEAN or INT32 tensor, and BINARY_SELECT a third operand, which the GPU plans and key
        // do not have.
        {Reference::GPU, {PointwiseMode::EXP}},
        {Reference::GPU, {PointwiseMode::LOG}},
        {Reference::GPU, {PointwiseMode::SQRT}},
        {Reference::GPU, {PointwiseMode::RSQRT}},
        {Reference::GPU, {PointwiseMode::RECIPROCAL}},
        {Reference::GPU, {PointwiseMode::SIN}},
        {Reference::GPU, {PointwiseMode::TAN}},
        {Reference::GPU, {PointwiseMode::ERF}},
        {Reference::GPU, {PointwiseMode::CEIL}},
        {Reference::GPU, {PointwiseMode::FLOOR}},
        {Reference::GPU, {PointwiseMode::ELU_FWD}},
        {Reference::GPU, {PointwiseMode::SOFTPLUS_FWD}},
        {Reference::GPU, {PointwiseMode::LOGICAL_NOT}},
        {Reference::GPU, {PointwiseMode::GEN_INDEX}},
        {Reference::GPU, {PointwiseMode::DIV}},
        {Reference::GPU, {PointwiseMode::ADD_SQUARE}},
        {Reference::GPU, {PointwiseMode::MAX_OP}},
        {Reference::GPU, {PointwiseMode::MIN_OP}},
        {Reference::GPU, {PointwiseMode::CMP_EQ}},
        {Reference::GPU, {PointwiseMode::CMP_NEQ}},
        {Reference::GPU, {PointwiseMode::CMP_GT}},
        {Reference::GPU, {PointwiseMode::CMP_GE}},
        {Reference::GPU, {PointwiseMode::CMP_LT}},
        {Reference::GPU, {PointwiseMode::CMP_LE}},
        {Reference::GPU, {PointwiseMode::LOGICAL_AND}},
        {Reference::GPU, {PointwiseMode::LOGICAL_OR}},
        {Reference::GPU, {PointwiseMode::GELU_BWD}},
        {Reference::GPU, {PointwiseMode::GELU_APPROX_TANH_BWD}},
        {Reference::GPU, {PointwiseMode::SWISH_BWD}},
        {Reference::GPU, {PointwiseMode::ELU_BWD}},
        {Reference::GPU, {PointwiseMode::SOFTPLUS_BWD}},
        {Reference::GPU, {PointwiseMode::BINARY_SELECT}},

        // CPU only: FP16 or BFP16 compute, and an FP32 operand beside a 16-bit one. The last two
        // read like compute and output swapped against the tuples before them; pinned as
        // registered.
        {Reference::GPU, binarySignature(K_FP16, K_FP16, K_FP16, K_FP16)},
        {Reference::GPU, binarySignature(K_BFP16, K_BFP16, K_BFP16, K_BFP16)},
        {Reference::GPU, binarySignature(K_FP16, K_FP32, K_FP32, K_FP32)},
        {Reference::GPU, binarySignature(K_BFP16, K_FP32, K_FP32, K_FP32)},
        {Reference::GPU, binarySignature(K_FP32, K_FP16, K_FP32, K_FP32)},
        {Reference::GPU, binarySignature(K_FP32, K_BFP16, K_FP32, K_FP32)},
        {Reference::GPU, binarySignature(K_FP32, K_FP16, K_FP32, K_FP16)},
        {Reference::GPU, binarySignature(K_FP32, K_BFP16, K_FP32, K_BFP16)},

        // CPU only: FP64, FP8 and INT32 tensors, which the GPU key has no entry for. Modes only the
        // CPU has are listed above whatever their datatypes.
        {Reference::GPU, unarySignature(K_FP64, K_FP64, K_FP64)},
        {Reference::GPU, unarySignature(K_FP8_E4M3, K_FP8_E4M3, K_FP32)},
        {Reference::GPU, unarySignature(K_FP8_E4M3, K_FP32, K_FP32)},
        {Reference::GPU, unarySignature(K_FP8_E5M2, K_FP8_E5M2, K_FP32)},
        {Reference::GPU, unarySignature(K_FP8_E5M2, K_FP32, K_FP32)},
        {Reference::GPU, unarySignature(K_INT32, K_INT32, K_FP32)},
        {Reference::GPU, binarySignature(K_FP64, K_FP64, K_FP64, K_FP64)},
        {Reference::GPU, binarySignature(K_FP8_E4M3, K_FP8_E4M3, K_FP8_E4M3, K_FP32)},
        {Reference::GPU, binarySignature(K_FP8_E4M3, K_FP8_E4M3, K_FP32, K_FP32)},
        {Reference::GPU, binarySignature(K_FP8_E5M2, K_FP8_E5M2, K_FP8_E5M2, K_FP32)},
        {Reference::GPU, binarySignature(K_FP8_E5M2, K_FP8_E5M2, K_FP32, K_FP32)},
        {Reference::GPU, binarySignature(K_INT32, K_INT32, K_INT32, K_FP32)},

        // GPU only: the CPU narrows an FP32 input but never widens a 16-bit one, and mixes FP16
        // with BFP16 only in CMP_GT.
        {Reference::CPU, unarySignature(K_FP16, K_FP32, K_FP32)},
        {Reference::CPU, unarySignature(K_BFP16, K_FP32, K_FP32)},
        {Reference::CPU, unarySignature(K_FP16, K_BFP16, K_FP32)},
        {Reference::CPU, unarySignature(K_BFP16, K_FP16, K_FP32)},
        {Reference::CPU, binarySignature(K_BFP16, K_FP16, K_FP32, K_FP32)},
        {Reference::CPU, binarySignature(K_FP16, K_BFP16, K_FP32, K_FP32)},
        {Reference::CPU, binarySignature(K_BFP16, K_FP16, K_FP16, K_FP32)},
        {Reference::CPU, binarySignature(K_FP16, K_BFP16, K_BFP16, K_FP32)},
        {Reference::CPU, binarySignature(K_BFP16, K_FP16, K_BFP16, K_FP32)},
        {Reference::CPU, binarySignature(K_FP16, K_BFP16, K_FP16, K_FP32)},
    };
    return s_gaps;
}

} // namespace

TEST(TestGpuPointwiseSignatureKey, EqualityOperator)
{
    const GpuPointwiseSignatureKey unary{K_FP32, K_FP32, K_FP32};
    EXPECT_TRUE(unary == GpuPointwiseSignatureKey(K_FP32, K_FP32, K_FP32));

    EXPECT_FALSE(unary == GpuPointwiseSignatureKey(K_FP16, K_FP32, K_FP32));
    EXPECT_FALSE(unary == GpuPointwiseSignatureKey(K_FP32, K_FP16, K_FP32));
    EXPECT_FALSE(unary == GpuPointwiseSignatureKey(K_FP32, K_FP32, K_FP16));

    // A binary key is never a unary one, even with every shared field equal.
    const GpuPointwiseSignatureKey binary{K_FP32, K_FP32, K_FP32, K_FP32};
    EXPECT_FALSE(unary == binary);
    EXPECT_TRUE(binary == GpuPointwiseSignatureKey(K_FP32, K_FP32, K_FP32, K_FP32));
    EXPECT_FALSE(binary == GpuPointwiseSignatureKey(K_FP32, K_FP16, K_FP32, K_FP32));
}

TEST(TestGpuPointwiseSignatureKey, HashFunction)
{
    const GpuPointwiseSignatureKey key1{K_FP32, K_FP32, K_FP32};
    const GpuPointwiseSignatureKey key2{K_FP32, K_FP32, K_FP32};
    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());
    EXPECT_EQ(key1(key2), key1.hashSelf());

    EXPECT_NE(key1.hashSelf(), GpuPointwiseSignatureKey(K_FP16, K_FP32, K_FP32).hashSelf());
    EXPECT_NE(key1.hashSelf(), GpuPointwiseSignatureKey(K_FP32, K_FP16, K_FP32).hashSelf());
    EXPECT_NE(key1.hashSelf(), GpuPointwiseSignatureKey(K_FP32, K_FP32, K_FP16).hashSelf());
    EXPECT_NE(key1.hashSelf(), GpuPointwiseSignatureKey(K_FP32, K_FP32, K_FP32, K_FP32).hashSelf());
}

TEST(TestGpuPointwiseSignatureKey, Copy)
{
    const GpuPointwiseSignatureKey original{K_FP16, K_BFP16, K_FP16, K_FP32};
    const GpuPointwiseSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(original.input0DataType, copied.input0DataType);
    EXPECT_EQ(original.input1DataType, copied.input1DataType);
    EXPECT_EQ(original.outputDataType, copied.outputDataType);
    EXPECT_EQ(original.computeDataType, copied.computeDataType);
}

TEST(TestGpuPointwiseSignatureKey, CreateFromNodeAndTensorMapUnary)
{
    const std::vector<int64_t> dims = {1, 2, 3, 4};
    const auto graph
        = buildGraph({PointwiseMode::ABS, K_FP16, K_NONE, K_NONE, K_FP32, K_FP32}, dims, dims);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        graph.GetBufferPointer(), graph.GetSize());

    const GpuPointwiseSignatureKey key(
        graphWrapper.getNode(0), graphWrapper.getTensorMap(), K_FP32);
    EXPECT_TRUE(key == GpuPointwiseSignatureKey(K_FP16, K_FP32, K_FP32));
}

TEST(TestGpuPointwiseSignatureKey, CreateFromNodeAndTensorMapBinary)
{
    const std::vector<int64_t> dims = {1, 2, 3, 4};
    const auto graph
        = buildGraph({PointwiseMode::ADD, K_BFP16, K_FP16, K_NONE, K_FP32, K_FP32}, dims, dims);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        graph.GetBufferPointer(), graph.GetSize());

    const GpuPointwiseSignatureKey key(
        graphWrapper.getNode(0), graphWrapper.getTensorMap(), K_FP32);
    EXPECT_TRUE(key == GpuPointwiseSignatureKey(K_BFP16, K_FP16, K_FP32, K_FP32));
}

// A probe that answered "no" to everything would pass every parity check below; both references
// take plain FP32 ADD.
TEST(TestPointwiseReferenceParity, TheProbeSeesAGraphBothReferencesTake)
{
    const auto accepted = probeReferences();
    const PointwiseTuple add{PointwiseMode::ADD, K_FP32, K_FP32, K_NONE, K_FP32, K_FP32};

    EXPECT_EQ(accepted.cpu.count(add), 1U) << describe(add);
    EXPECT_EQ(accepted.gpu.count(add), 1U) << describe(add);
}

// A registered key whose plan builder then rejects it is dead: the comparison would never see it.
TEST(TestPointwiseReferenceParity, EveryRegisteredTupleIsAcceptedByItsOwnReference)
{
    const auto accepted = probeReferences();
    std::ostringstream dead;

    for(const auto& tuple : cpuRegistryTuples())
    {
        if(accepted.cpu.count(tuple) == 0)
        {
            dead << "  the CPU registers " << describe(tuple)
                 << " but its plan builder rejects it\n";
        }
    }

    // The GPU key has no mode: it is dead when no mode passes its plan builder.
    std::set<PointwiseTuple> registered;
    for(const auto& tuple : gpuCandidateTuples())
    {
        registered.insert(withoutMode(tuple));
    }
    std::set<PointwiseTuple> taken;
    for(const auto& tuple : accepted.gpu)
    {
        taken.insert(withoutMode(tuple));
    }
    for(const auto& signature : registered)
    {
        if(taken.count(signature) == 0)
        {
            dead << "  the GPU registers " << describe(signature)
                 << " but no mode passes its plan builder\n";
        }
    }

    EXPECT_TRUE(dead.str().empty()) << "\n" << dead.str();
}

TEST(TestPointwiseReferenceParity, EveryDifferenceBetweenTheReferencesIsListed)
{
    const auto accepted = probeReferences();
    std::set<PointwiseTuple> either = accepted.cpu;
    either.insert(accepted.gpu.begin(), accepted.gpu.end());

    const auto& gaps = knownGaps();
    std::ostringstream unlisted;
    for(const auto& tuple : either)
    {
        const bool onCpu = accepted.cpu.count(tuple) != 0;
        if(onCpu == (accepted.gpu.count(tuple) != 0))
        {
            continue;
        }

        const Reference lacking = onCpu ? Reference::GPU : Reference::CPU;
        const bool listed = std::any_of(gaps.begin(), gaps.end(), [&](const Gap& gap) {
            return gap.lacking == lacking && matches(gap, tuple);
        });
        if(!listed)
        {
            unlisted << "  " << (onCpu ? "GPU" : "CPU") << " does not accept " << describe(tuple)
                     << "\n";
        }
    }

    EXPECT_TRUE(unlisted.str().empty())
        << "\nThe references disagree on graphs no knownGaps() entry explains. Make both accept\n"
           "them, or list them there and update the Pointwise sections of\n"
           "OperationSupport-ReferenceImpl.md and OperationSupport-GpuReferenceImpl.md:\n"
        << unlisted.str();
}

TEST(TestPointwiseReferenceParity, EveryListedDifferenceStillExists)
{
    const auto accepted = probeReferences();

    std::ostringstream stale;
    for(const auto& gap : knownGaps())
    {
        const auto& lacking = gap.lacking == Reference::CPU ? accepted.cpu : accepted.gpu;
        const auto& other = gap.lacking == Reference::CPU ? accepted.gpu : accepted.cpu;
        const auto isMatch = [&gap](const PointwiseTuple& tuple) { return matches(gap, tuple); };
        if(std::any_of(lacking.begin(), lacking.end(), isMatch)
           || std::none_of(other.begin(), other.end(), isMatch))
        {
            stale << "  " << describe(gap.pattern) << "\n";
        }
    }

    EXPECT_TRUE(stale.str().empty())
        << "\nThese knownGaps() entries no longer describe a difference between the references.\n"
           "Remove them and update the Pointwise sections of OperationSupport-ReferenceImpl.md\n"
           "and OperationSupport-GpuReferenceImpl.md:\n"
        << stale.str();
}

// gpuSupportedOps() is a commitment with no skip path, so it can list Pointwise only once the GPU
// lacks nothing the CPU has, and has to by then.
TEST(TestPointwiseReferenceParity, RequiredOpSetIncludesPointwiseExactlyWhenNothingIsCpuOnly)
{
    const auto accepted = probeReferences();
    const bool gpuLacksSomething = std::any_of(
        accepted.cpu.begin(), accepted.cpu.end(), [&accepted](const PointwiseTuple& tuple) {
            return accepted.gpu.count(tuple) == 0;
        });
    const auto& gpuRequiredOps = referenceSupportedOps(ReferenceExecutorType::GPU);
    const bool pointwiseIsRequired = gpuRequiredOps.count(NodeAttributes::PointwiseAttributes) != 0;

    if(gpuLacksSomething)
    {
        EXPECT_FALSE(pointwiseIsRequired)
            << "PointwiseAttributes is in gpuSupportedOps() (ReferenceOpCoverage.cpp), but the\n"
               "GPU reference still lacks pointwise graphs the CPU takes: see the GPU entries of\n"
               "knownGaps().";
    }
    else
    {
        EXPECT_TRUE(pointwiseIsRequired)
            << "The GPU reference now takes every pointwise graph the CPU does. Add\n"
               "NodeAttributes::PointwiseAttributes to gpuSupportedOps() in\n"
               "ReferenceOpCoverage.cpp, so Convolution Forward + Pointwise bundles stop\n"
               "reporting OUTSIDE_OP_SET.";
    }
}

// The GPU kernels address at most GpuReferencePointwise::MAX_DIMS dimensions and the CPU has no
// such limit, so the GPU has to decline deeper graphs up front for the harness to fall through.
TEST(TestPointwiseReferenceParity, FiveDimensionCapDeclinesDeeperGraphsWhileTheCpuAccepts)
{
    CpuReferenceGraphExecutorAdapter cpuExecutor;
    GpuReferenceGraphExecutor gpuExecutor;

    const std::vector<PointwiseTuple> tuples = {
        {PointwiseMode::ABS, K_FP32, K_NONE, K_NONE, K_FP32, K_FP32},
        {PointwiseMode::ADD, K_FP32, K_FP32, K_NONE, K_FP32, K_FP32},
    };
    const std::vector<int64_t> dimsAtCap = {2, 3, 2, 2, 2};
    const std::vector<int64_t> broadcastAtCap = {1, 3, 1, 1, 1};
    const std::vector<int64_t> dimsOverCap = {2, 3, 2, 2, 2, 2};
    const std::vector<int64_t> broadcastOverCap = {1, 3, 1, 1, 1, 1};

    for(const auto& tuple : tuples)
    {
        const auto atCap = buildGraph(tuple, dimsAtCap, broadcastAtCap);
        const auto overCap = buildGraph(tuple, dimsOverCap, broadcastOverCap);

        EXPECT_TRUE(gpuExecutor.isApplicable(atCap.GetBufferPointer(), atCap.GetSize()))
            << describe(tuple) << " at 5-D";
        EXPECT_TRUE(cpuExecutor.isApplicable(atCap.GetBufferPointer(), atCap.GetSize()))
            << describe(tuple) << " at 5-D";
        EXPECT_FALSE(gpuExecutor.isApplicable(overCap.GetBufferPointer(), overCap.GetSize()))
            << describe(tuple) << " at 6-D";
        EXPECT_TRUE(cpuExecutor.isApplicable(overCap.GetBufferPointer(), overCap.GetSize()))
            << describe(tuple) << " at 6-D";
    }
}

// The parity tests only ask isApplicable(); a plan that passes its datatypes to the kernel in the
// wrong order is only seen when it runs. The CPU reference is the oracle because the CPU executor
// declines the GPU-only tuples.
class TestGpuPointwiseExecutor : public ::testing::TestWithParam<PointwiseTuple>
{
};

TEST_P(TestGpuPointwiseExecutor, MatchesCpuReference)
{
    SKIP_IF_NO_DEVICES();

    const auto& tuple = GetParam();
    const bool binary = tuple.in1 != K_NONE;
    const std::vector<int64_t> dims = {2, 3, 4, 4};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    const auto graph = buildGraph(tuple, dims, dims);
    GpuReferenceGraphExecutor gpuExecutor;

    // A unary tuple has no second input, so in0's type stands in for it.
    std::visit(
        [&](auto in0, auto in1, auto out) {
            using In0 = decltype(in0);
            using In1 = decltype(in1);
            using Out = decltype(out);
            if constexpr(IS_GPU_TYPE<In0> && IS_GPU_TYPE<In1> && IS_GPU_TYPE<Out>)
            {
                Tensor<In0> input0(dims, strides);
                Tensor<In1> input1(dims, strides);
                Tensor<Out> gpuOutput(dims, strides);
                Tensor<Out> cpuOutput(dims, strides);
                input0.fillTensorWithRandomValues(-1.0f, 1.0f, 42);
                input1.fillTensorWithRandomValues(-1.0f, 1.0f, 43);

                std::unordered_map<int64_t, void*> variantPack
                    = {{IN0_UID, input0.rawDeviceData()}, {OUT_UID, gpuOutput.rawDeviceData()}};
                if(binary)
                {
                    variantPack[IN1_UID] = input1.rawDeviceData();
                    CpuReferencePointwiseImpl<Out, double>::pointwiseCompute(
                        tuple.mode, cpuOutput, input0, input1);
                }
                else
                {
                    CpuReferencePointwiseImpl<Out, double>::pointwiseCompute(
                        tuple.mode, cpuOutput, input0);
                }
                gpuExecutor.execute(graph.GetBufferPointer(), graph.GetSize(), variantPack);
                gpuOutput.markDeviceModified();

                const float tolerance = getTolerance<Out>();
                const CpuFpReferenceValidation<Out> validation(tolerance, tolerance);
                EXPECT_TRUE(validation.allClose(cpuOutput, gpuOutput));
            }
            else
            {
                ADD_FAILURE() << "A datatype of this tuple is not covered by IS_GPU_TYPE";
            }
        },
        datatypeToNativeVariant(tuple.in0),
        datatypeToNativeVariant(binary ? tuple.in1 : tuple.in0),
        datatypeToNativeVariant(tuple.out));
}

// The name holds the mode and every datatype, so no two tuples share one.
INSTANTIATE_TEST_SUITE_P(Registered,
                         TestGpuPointwiseExecutor,
                         ::testing::ValuesIn(executedTuples()),
                         [](const ::testing::TestParamInfo<PointwiseTuple>& info) {
                             const auto& tuple = info.param;
                             std::string name = EnumNamePointwiseMode(tuple.mode);
                             name += std::string("_") + EnumNameDataType(tuple.in0);
                             if(tuple.in1 != K_NONE)
                             {
                                 name += std::string("_") + EnumNameDataType(tuple.in1);
                             }
                             return name + "_" + EnumNameDataType(tuple.out) + "_"
                                    + EnumNameDataType(tuple.compute);
                         });
