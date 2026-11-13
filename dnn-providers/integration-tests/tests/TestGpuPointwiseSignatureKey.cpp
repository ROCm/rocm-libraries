// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The CPU and GPU pointwise references list their modes and datatype tuples in unrelated places,
// and the GPU key has no mode. TestPointwiseReferenceParity asks both executors about every
// (mode, datatype tuple) either registry can express. Each difference has to be listed in
// knownGaps(), and each listed gap has to still be a difference.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/PointwiseValidation.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PointwiseSignatureKey.hpp>

#include "harness/CpuReferenceGraphExecutorAdapter.hpp"
#include "harness/gpu-graph-executor/GpuReferenceGraphExecutor.hpp"
#include "harness/gpu-graph-executor/detail/GpuPointwiseSignatureKey.hpp"
#include "harness/reference-validation/ReferenceOpCoverage.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_integration_tests::CpuReferenceGraphExecutorAdapter;
using hipdnn_integration_tests::ReferenceExecutorType;
using hipdnn_integration_tests::bundle::referenceSupportedOps;
using hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor;
using hipdnn_integration_tests::gpu_graph_executor::detail::GpuPointwiseSignatureKey;
using hipdnn_test_sdk::detail::PointwiseSignatureKey;

namespace
{

constexpr DataType K_FP32 = DataType::FLOAT;
constexpr DataType K_FP16 = DataType::HALF;
constexpr DataType K_BFP16 = DataType::BFLOAT16;
constexpr DataType K_NONE = DataType::UNSET;

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

// A single-node pointwise graph for `tuple`; `dims` is the first input and the output,
// `operandDims` the other inputs. createPointwiseGraph() gives input and output one datatype, so it
// cannot express a mixed tuple.
flatbuffers::FlatBufferBuilder buildGraph(const PointwiseTuple& tuple,
                                          const std::vector<int64_t>& dims,
                                          const std::vector<int64_t>& operandDims)
{
    constexpr int64_t IN0_UID = 1;
    constexpr int64_t OUT_UID = 2;
    constexpr int64_t IN1_UID = 3;
    constexpr int64_t IN2_UID = 4;

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

    const auto attributes = CreatePointwiseAttributes(builder,
                                                      tuple.mode,
                                                      flatbuffers::nullopt,
                                                      flatbuffers::nullopt,
                                                      flatbuffers::nullopt,
                                                      flatbuffers::nullopt,
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
        // No GPU kernel; CMP_GT and BINARY_SELECT also need a BOOLEAN tensor and a third operand,
        // which the GPU plans and key do not have.
        {Reference::GPU, {PointwiseMode::MAX_OP}},
        {Reference::GPU, {PointwiseMode::MIN_OP}},
        {Reference::GPU, {PointwiseMode::CMP_GT}},
        {Reference::GPU, {PointwiseMode::BINARY_SELECT}},
        // Implemented by ReferencePointwiseBase, but left out of the CPU registry.
        {Reference::CPU, {PointwiseMode::IDENTITY}},

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

        // GPU only: the CPU narrows an FP32 input but never widens a 16-bit one, and never mixes
        // FP16 with BFP16.
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
