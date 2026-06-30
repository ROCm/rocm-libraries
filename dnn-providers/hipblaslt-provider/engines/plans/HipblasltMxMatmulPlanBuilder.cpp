// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <array>
#include <string>

#include <hipblaslt/hipblaslt.h>
#include <hipdnn_flatbuffers_sdk/data_objects/block_scale_dequantize_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/matmul_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "HipblasltMxMatmulPlan.hpp"
#include "HipblasltMxMatmulPlanBuilder.hpp"
#include "HipblasltUtils.hpp"

namespace hipblaslt_plugin
{
namespace
{

using namespace hipdnn_flatbuffers_sdk::data_objects;

// Extract (deqAttrA, deqAttrB, matmulAttr) from a 3-node dequant+dequant+matmul graph.
// The two BlockScaleDequantize nodes may appear in either order; which one feeds
// matmul input A vs B is resolved by matching each dequant's Y output uid to the
// matmul's a/b input uids.
std::tuple<const BlockScaleDequantizeAttributes&,
           const BlockScaleDequantizeAttributes&,
           const MatmulAttributes&>
    getNodeAttrs(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    if(opGraph.nodeCount() != 3)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "MX matmul plan builder requires exactly 3 nodes. Graph has "
                + std::to_string(opGraph.nodeCount()) + " nodes");
    }

    // Nodes 0 and 1 must both be BlockScaleDequantize, in either order: the two
    // dequant nodes are independent, so their relative position is not fixed.
    const auto& nodeWrap0 = opGraph.getNodeWrapper(0);
    if(nodeWrap0.attributesType() != NodeAttributes::BlockScaleDequantizeAttributes)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Node 0 must be BlockScaleDequantize. Found: "
                + std::string(toString(nodeWrap0.attributesType())));
    }
    const auto& deq0 = nodeWrap0.attributesAs<BlockScaleDequantizeAttributes>();

    const auto& nodeWrap1 = opGraph.getNodeWrapper(1);
    if(nodeWrap1.attributesType() != NodeAttributes::BlockScaleDequantizeAttributes)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Node 1 must be BlockScaleDequantize. Found: "
                + std::string(toString(nodeWrap1.attributesType())));
    }
    const auto& deq1 = nodeWrap1.attributesAs<BlockScaleDequantizeAttributes>();

    // Node 2 must be Matmul. It depends on both dequant outputs, so in topological
    // order it is always last.
    const auto& nodeWrap2 = opGraph.getNodeWrapper(2);
    if(nodeWrap2.attributesType() != NodeAttributes::MatmulAttributes)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Node 2 must be Matmul. Found: " + std::string(toString(nodeWrap2.attributesType())));
    }
    const auto& matmulAttr = nodeWrap2.attributesAs<MatmulAttributes>();

    // Resolve which dequant feeds matmul input A and which feeds B by matching each
    // dequant's Y output uid against the matmul's a/b input uids.
    const int64_t matmulA = matmulAttr.a_tensor_uid();
    const int64_t matmulB = matmulAttr.b_tensor_uid();

    if(deq0.y_tensor_uid() == matmulA && deq1.y_tensor_uid() == matmulB)
    {
        return {deq0, deq1, matmulAttr};
    }
    if(deq1.y_tensor_uid() == matmulA && deq0.y_tensor_uid() == matmulB)
    {
        return {deq1, deq0, matmulAttr};
    }

    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM,
        "Matmul inputs do not match dequantize outputs: "
        "matmulA="
            + std::to_string(matmulA) + " matmulB=" + std::to_string(matmulB)
            + " deq0Y=" + std::to_string(deq0.y_tensor_uid())
            + " deq1Y=" + std::to_string(deq1.y_tensor_uid()));
}

using TensorWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper;
using DT = hipdnn_flatbuffers_sdk::data_objects::DataType;

// Our current implementation limitations (distinct from hipBLASLt's own
// restrictions, checked later). For now we only support OCP FP8 inputs
// (E4M3 / E5M2).
void checkImplementationLimitations(const TensorWrapper& tXA, const TensorWrapper& tXB)
{
    if(!hipblaslt_utils::isTypeFp8Ocp(tXA.dataType()))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "MX matmul: A input must be FP8 OCP (E4M3 or E5M2)");
    }
    if(!hipblaslt_utils::isTypeFp8Ocp(tXB.dataType()))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "MX matmul: B input must be FP8 OCP (E4M3 or E5M2)");
    }
}

// Compute data type must be FP32 for every node, mirroring the plain matmul
// builder (see HipblasltMatmulPlanBuilder::checkComputeTypes). Node 2 is the
// matmul; nodes 0 and 1 are the two dequantize nodes (topology already
// validated by getNodeAttrs).
void checkComputeTypes(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph)
{
    constexpr uint32_t MATMUL_IDX = 2;
    if(graph.getNode(MATMUL_IDX).compute_data_type() != DT::FLOAT)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "MX matmul: matmul node compute data type must be float");
    }
    for(const uint32_t idx : {0U, 1U})
    {
        if(graph.getNode(idx).compute_data_type() != graph.getNode(MATMUL_IDX).compute_data_type())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "MX matmul: dequantize node compute data type must equal matmul compute data type");
        }
    }
}

// Validate the virtual/non-virtual contract of the fused graph. The inputs,
// their scales, and the output are real tensors that the plan resolves to
// device buffers at execute time, so they must be non-virtual. The two dequantize
// outputs (Y_A, Y_B) are fused intermediates the MX plan never materializes, so
// they must be virtual.
void checkVirtualTensors(const BlockScaleDequantizeAttributes& deqAttrA,
                         const BlockScaleDequantizeAttributes& deqAttrB,
                         const MatmulAttributes& matmulAttr,
                         const std::unordered_map<int64_t, const TensorAttributes*>& tensorMap)
{
    const auto requireNonVirtual = [&tensorMap](int64_t uid, const char* name) {
        if(hipblaslt_utils::findTensorAttributes(tensorMap, uid).isVirtual())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                std::string("MX matmul: ") + name
                    + " tensor must be non-virtual (it needs a device buffer)");
        }
    };
    const auto requireVirtual = [&tensorMap](int64_t uid, const char* name) {
        if(!hipblaslt_utils::findTensorAttributes(tensorMap, uid).isVirtual())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                std::string("MX matmul: ") + name
                    + " tensor must be virtual (it is a fused dequantize output)");
        }
    };

    requireNonVirtual(deqAttrA.x_tensor_uid(), "A (FP8 input)");
    requireNonVirtual(deqAttrB.x_tensor_uid(), "B (FP8 input)");
    requireNonVirtual(deqAttrA.scale_tensor_uid(), "A scale");
    requireNonVirtual(deqAttrB.scale_tensor_uid(), "B scale");
    requireNonVirtual(matmulAttr.c_tensor_uid(), "matmul output");
    requireVirtual(deqAttrA.y_tensor_uid(), "A dequantize output");
    requireVirtual(deqAttrB.y_tensor_uid(), "B dequantize output");
}

// Verify the input shapes are consistent with the output. The matmul is
// A[M, K] x B[K, N] = C[M, N], so logically (independent of physical strides):
//   A dims[-2]=M, dims[-1]=K; B dims[-2]=K, dims[-1]=N; C dims[-2]=M, dims[-1]=N.
void checkShapesMatchOutput(const TensorWrapper& tXA,
                            const TensorWrapper& tXB,
                            const TensorWrapper& tC)
{
    const auto& dimsA = tXA.dims();
    const auto& dimsB = tXB.dims();
    const auto& dimsC = tC.dims();

    PLUGIN_THROW_IF_FALSE(
        dimsA.size() >= 2, HIPDNN_PLUGIN_STATUS_BAD_PARAM, "MX matmul: A tensor rank must be >= 2");
    PLUGIN_THROW_IF_FALSE(
        dimsB.size() >= 2, HIPDNN_PLUGIN_STATUS_BAD_PARAM, "MX matmul: B tensor rank must be >= 2");
    PLUGIN_THROW_IF_FALSE(
        dimsC.size() >= 2, HIPDNN_PLUGIN_STATUS_BAD_PARAM, "MX matmul: C tensor rank must be >= 2");

    const int64_t aM = dimsA[dimsA.size() - 2];
    const int64_t aK = dimsA[dimsA.size() - 1];
    const int64_t bK = dimsB[dimsB.size() - 2];
    const int64_t bN = dimsB[dimsB.size() - 1];
    const int64_t cM = dimsC[dimsC.size() - 2];
    const int64_t cN = dimsC[dimsC.size() - 1];

    if(aK != bK)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "MX matmul: A K-dim (" + std::to_string(aK)
                                                           + ") must equal B K-dim ("
                                                           + std::to_string(bK) + ")");
    }
    if(aM != cM)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "MX matmul: A M-dim (" + std::to_string(aM)
                                                           + ") must equal C M-dim ("
                                                           + std::to_string(cM) + ")");
    }
    if(bN != cN)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "MX matmul: B N-dim (" + std::to_string(bN)
                                                           + ") must equal C N-dim ("
                                                           + std::to_string(cN) + ")");
    }
}

// Validate the block-scale tensors. The VEC32_UE8M0 mode stores one UE8M0
// (DataType::FP8_E8M0) value per 32-element block along the operand's K axis, so
// each scale tensor must be FP8_E8M0 and mirror its operand's shape with the K
// axis divided by 32: [M, K/32] for A (K innermost) and [K/32, N] for B (K is the
// second-to-last axis).
void checkScaleTensors(const BlockScaleDequantizeAttributes& deqAttrA,
                       const BlockScaleDequantizeAttributes& deqAttrB,
                       const TensorWrapper& tXA,
                       const TensorWrapper& tXB,
                       const std::unordered_map<int64_t, const TensorAttributes*>& tensorMap)
{
    const auto tScaleA
        = hipblaslt_utils::findTensorAttributes(tensorMap, deqAttrA.scale_tensor_uid());
    const auto tScaleB
        = hipblaslt_utils::findTensorAttributes(tensorMap, deqAttrB.scale_tensor_uid());

    if(tScaleA.dataType() != DT::FP8_E8M0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "MX matmul: A scale must be FP8_E8M0 (UE8M0) for VEC32 block scaling");
    }
    if(tScaleB.dataType() != DT::FP8_E8M0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "MX matmul: B scale must be FP8_E8M0 (UE8M0) for VEC32 block scaling");
    }

    // Validate that a scale tensor mirrors its operand's shape with the K axis
    // divided into VEC32_BLOCK_SIZE-wide blocks. kAxisFromEnd is 1 for A (K is the
    // innermost axis) and 2 for B (K is the second-to-last axis). Operand rank
    // (>= 2) is already guaranteed by checkShapesMatchOutput.
    const auto checkScaleShape = [](const TensorWrapper& tOperand,
                                    const TensorWrapper& tScale,
                                    size_t kAxisFromEnd,
                                    const char* operand) {
        const auto& opDims = tOperand.dims();
        const auto& scaleDims = tScale.dims();
        if(scaleDims.size() != opDims.size())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                std::string("MX matmul: ") + operand + " scale rank ("
                    + std::to_string(scaleDims.size()) + ") must equal operand rank ("
                    + std::to_string(opDims.size()) + ")");
        }

        const size_t kIdx = opDims.size() - kAxisFromEnd;
        if(opDims[kIdx] % VEC32_BLOCK_SIZE != 0)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                std::string("MX matmul: ") + operand + " K-dim (" + std::to_string(opDims[kIdx])
                    + ") must be a multiple of " + std::to_string(VEC32_BLOCK_SIZE)
                    + " for VEC32 block scaling");
        }

        for(size_t i = 0; i < opDims.size(); ++i)
        {
            const int64_t expected = (i == kIdx) ? opDims[i] / VEC32_BLOCK_SIZE : opDims[i];
            if(scaleDims[i] != expected)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    std::string("MX matmul: ") + operand + " scale dim[" + std::to_string(i) + "] ("
                        + std::to_string(scaleDims[i]) + ") must equal " + std::to_string(expected)
                        + " (operand shape with the K axis divided by "
                        + std::to_string(VEC32_BLOCK_SIZE) + ")");
            }
        }
    };

    checkScaleShape(tXA, tScaleA, 1, "A");
    checkScaleShape(tXB, tScaleB, 2, "B");
}

// hipBLASLt's own restrictions for the VEC32_UE8M0 scale mode.
void checkHipblasltConstraints(const BlockScaleDequantizeAttributes& deqAttrA,
                               const BlockScaleDequantizeAttributes& deqAttrB,
                               const TensorWrapper& tXA,
                               const TensorWrapper& tXB,
                               const TensorWrapper& tC)
{
    // Output C must be FP32, FP16, or BF16 (VEC32_UE8M0 Dtype restriction)
    static constexpr std::array<DT, 3> VALID_OUT_TYPES = {DT::FLOAT, DT::HALF, DT::BFLOAT16};
    if(std::find(VALID_OUT_TYPES.begin(), VALID_OUT_TYPES.end(), tC.dataType())
       == VALID_OUT_TYPES.end())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "MX matmul: output C must be FP32, FP16, or BF16");
    }

    // Batch must be 1: hipBLASLt requires B==1 for VEC32_UE8M0, and there is no
    // batch-stride attribute for the A/B scale pointers to advance per batch.
    const auto& dimsA = tXA.dims();
    const auto& dimsB = tXB.dims();
    const auto& dimsC = tC.dims();

    const auto checkNoBatch = [](const auto& dims, const char* operand) {
        for(size_t i = 0; i + 2 < dims.size(); ++i)
        {
            if(dims[i] != 1)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    std::string("MX matmul: batch must be 1 (got ") + operand + " batch>1)");
            }
        }
    };
    checkNoBatch(dimsA, "A");
    checkNoBatch(dimsB, "B");
    checkNoBatch(dimsC, "C");

    // opA = T, opB = N — inferred from FP8 X tensor strides.
    // Rule: row-major (stride[-1]==1) → HIPBLAS_OP_N; col-major (stride[-2]==1) → HIPBLAS_OP_T
    const auto& stridesA = tXA.strides();
    const auto& stridesB = tXB.strides();

    PLUGIN_THROW_IF_FALSE(stridesA.size() >= 2,
                          HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                          "MX matmul: A tensor rank must be >= 2");
    PLUGIN_THROW_IF_FALSE(stridesB.size() >= 2,
                          HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                          "MX matmul: B tensor rank must be >= 2");

    if(stridesA[stridesA.size() - 2] != 1)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "MX matmul: A (FP8) must have opA=T (column-major strides, stride[-2]==1)");
    }
    if(stridesB[stridesB.size() - 1] != 1)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "MX matmul: B (FP8) must have opB=N (row-major strides, stride[-1]==1)");
    }

    // Alignment: m % 16 == 0, n % 16 == 0, K % 128 == 0.
    // Logical matmul dims: A[M, K], B[K, N].
    const int64_t matM = dimsA[dimsA.size() - 2];
    const int64_t matK = dimsA[dimsA.size() - 1];
    const int64_t matN = dimsB[dimsB.size() - 1];

    if(matM % 16 != 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "MX matmul: M must be divisible by 16 (M=" + std::to_string(matM) + ")");
    }
    if(matN % 16 != 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "MX matmul: N must be divisible by 16 (N=" + std::to_string(matN) + ")");
    }
    if(matK % 128 != 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "MX matmul: K must be divisible by 128 (K=" + std::to_string(matK) + ")");
    }

    // Inner block_size must be VEC32_BLOCK_SIZE for both scales (VEC32)
    const auto* blockSizeA = deqAttrA.block_size();
    if(blockSizeA == nullptr || blockSizeA->empty() || (*blockSizeA)[0] != VEC32_BLOCK_SIZE)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "MX matmul: A scale block_size[0] must be "
                                                           + std::to_string(VEC32_BLOCK_SIZE)
                                                           + " (VEC32)");
    }
    const auto* blockSizeB = deqAttrB.block_size();
    if(blockSizeB == nullptr || blockSizeB->empty() || (*blockSizeB)[0] != VEC32_BLOCK_SIZE)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "MX matmul: B scale block_size[0] must be "
                                                           + std::to_string(VEC32_BLOCK_SIZE)
                                                           + " (VEC32)");
    }
}

/// Run every applicability check in order. Plugin-specific limitations, compute
/// types, and shape consistency are validated first; hipBLASLt's own
/// VEC32_UE8M0 constraints are checked last. The current GPU's ability to run
/// the MX GEMM is the authoritative gate and is enforced later when the plan
/// constructor calls hipblasLtMatmulAlgoGetHeuristic.
void checkConstraints(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const BlockScaleDequantizeAttributes& deqAttrA,
                      const BlockScaleDequantizeAttributes& deqAttrB,
                      const MatmulAttributes& matmulAttr,
                      const std::unordered_map<int64_t, const TensorAttributes*>& tensorMap)
{
    const auto tXA = hipblaslt_utils::findTensorAttributes(tensorMap, deqAttrA.x_tensor_uid());
    const auto tXB = hipblaslt_utils::findTensorAttributes(tensorMap, deqAttrB.x_tensor_uid());
    const auto tC = hipblaslt_utils::findTensorAttributes(tensorMap, matmulAttr.c_tensor_uid());

    checkImplementationLimitations(tXA, tXB);
    checkVirtualTensors(deqAttrA, deqAttrB, matmulAttr, tensorMap);
    checkComputeTypes(graph);
    checkShapesMatchOutput(tXA, tXB, tC);
    checkScaleTensors(deqAttrA, deqAttrB, tXA, tXB, tensorMap);
    checkHipblasltConstraints(deqAttrA, deqAttrB, tXA, tXB, tC);
}

} // namespace

bool HipblasltMxMatmulPlanBuilder::isApplicable(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    try
    {
        auto [deqAttrA, deqAttrB, matmulAttr] = getNodeAttrs(opGraph);
        checkConstraints(opGraph, deqAttrA, deqAttrB, matmulAttr, opGraph.getTensorMap());
        // Constructing the plan runs hipblasLtMatmulAlgoGetHeuristic, the
        // authoritative check that this GPU + hipBLASLt build actually has an MX
        // GEMM kernel for this problem. MX support is arch/library-dependent, so
        // (unlike plain GEMM) the constraint checks above cannot confirm it. No
        // algorithm -> the constructor throws -> the graph is reported unsupported
        // here rather than failing later in buildPlan()/execute().
        MxMatmulParams params(deqAttrA, deqAttrB, matmulAttr, opGraph.getTensorMap());
        MxMatmulPlan const plan(handle, std::move(params));
        return true;
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(e.what());
        return false;
    }
}

size_t HipblasltMxMatmulPlanBuilder::getWorkspaceSize(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    auto [deqAttrA, deqAttrB, matmulAttr] = getNodeAttrs(opGraph);
    MxMatmulParams params(deqAttrA, deqAttrB, matmulAttr, opGraph.getTensorMap());
    MxMatmulPlan const plan(handle, std::move(params));
    return plan.getWorkspaceSize(handle);
}

void HipblasltMxMatmulPlanBuilder::buildPlan(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    HipdnnEnginePluginExecutionContext& executionContext) const
{
    auto [deqAttrA, deqAttrB, matmulAttr] = getNodeAttrs(opGraph);

    HIPDNN_PLUGIN_LOG_INFO("Building MX matmul plan for graph with " << opGraph.nodeCount()
                                                                     << " nodes");

    MxMatmulParams params(deqAttrA, deqAttrB, matmulAttr, opGraph.getTensorMap());
    auto plan = std::make_unique<MxMatmulPlan>(handle, std::move(params));
    executionContext.setPlan(std::move(plan));
}

} // namespace hipblaslt_plugin
