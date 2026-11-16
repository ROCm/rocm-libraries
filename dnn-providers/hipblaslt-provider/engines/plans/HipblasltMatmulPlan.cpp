// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <cstddef>
#include <limits>
#include <numeric>
#include <string>

#include <hipblaslt/hipblaslt.h>
#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "HipblasltMatmulPlan.hpp"
#include "HipblasltUtils.hpp"
#include "HipdnnEnginePluginHandle.hpp"

namespace hipblaslt_plugin
{
namespace
{
inline int64_t getBatchCount(const std::vector<int64_t>& dims)
{
    PLUGIN_THROW_IF_TRUE(dims.size() < 2,
                         HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                         "Failed to calculate batch count:expected at least 2 dimensions");
    return std::accumulate(dims.begin(), dims.end() - 2, int64_t{1}, std::multiplies<>());
}
} // namespace

hipblasOperation_t MatmulParams::getTrans(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper& t)
{
    const auto order = hipblaslt_utils::matrixOrder(t.dims(), t.strides());
    // Row-major: dims are swapped in layout, no additional transpose
    if(order == hipblaslt_utils::MatrixOrder::ROW_MAJOR)
    {
        return HIPBLAS_OP_N;
    }
    // Column-major: dims not swapped, need transpose
    if(order == hipblaslt_utils::MatrixOrder::COLUMN_MAJOR)
    {
        return HIPBLAS_OP_T;
    }
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Unsupported stride for input matrix: " + t.name());
}

hipblasComputeType_t MatmulParams::getComputeDataType(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper& tA,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper& tB)
{
    auto hipDataTypeA = hipblaslt_utils::tensorDataTypeToHipDataType(tA.dataType());
    auto hipDataTypeB = hipblaslt_utils::tensorDataTypeToHipDataType(tB.dataType());
    if(hipDataTypeA == hipDataTypeB && hipDataTypeA == HIP_R_16F)
    {
        return HIPBLAS_COMPUTE_32F_FAST_16F;
    }
    if(hipDataTypeA == hipDataTypeB && hipDataTypeA == HIP_R_16BF)
    {
        return HIPBLAS_COMPUTE_32F_FAST_16BF;
    }
    return HIPBLAS_COMPUTE_32F;
}

MatmulParams::MatmulParams(
    const hipdnn_flatbuffers_sdk::data_objects::MatmulAttributes& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : MatmulParams(attributes, nullptr, nullptr, tensorMap)
{
}

MatmulParams::MatmulParams(
    const hipdnn_flatbuffers_sdk::data_objects::MatmulAttributes& attributes,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* biasAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* activAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    const auto tA = hipblaslt_utils::findTensorAttributes(tensorMap, attributes.a_tensor_uid());
    const auto tB = hipblaslt_utils::findTensorAttributes(tensorMap, attributes.b_tensor_uid());
    const auto tC = hipblaslt_utils::findTensorAttributes(tensorMap, attributes.c_tensor_uid());

    _matrixLayoutA = HipblasltMatrixLayout(tA);
    _matrixLayoutB = HipblasltMatrixLayout(tB);

    // The layout of C/D comes from the graph's final output tensor.
    int64_t outputUid = attributes.c_tensor_uid();
    if(activAttr != nullptr)
    {
        outputUid = activAttr->out_0_tensor_uid();
    }
    else if(biasAttr != nullptr)
    {
        outputUid = biasAttr->out_0_tensor_uid();
    }
    const auto tOut = hipblaslt_utils::findTensorAttributes(tensorMap, outputUid);
    PLUGIN_THROW_IF_FALSE(hipblaslt_utils::matrixOrder(tOut.dims(), tOut.strides())
                              == hipblaslt_utils::MatrixOrder::ROW_MAJOR,
                          HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                          "Unsupported output layout: hipBLASLt engine requires a row-major "
                          "output matrix: "
                              + tOut.name());
    _matrixLayoutC = HipblasltMatrixLayout(tOut);

    const auto& aDims = tA.dims();
    const auto& bDims = tB.dims();
    const auto& cDims = tC.dims();
    if(aDims.size() != bDims.size() || aDims.size() != cDims.size())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported input matrix ranks: they should be the same");
    }

    hipDataType biasDataType = HIP_R_32F;
    if(biasAttr != nullptr)
    {
        if(!biasAttr->in_1_tensor_uid().has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "MatmulParams: biasAttr missing in_1_tensor_uid");
        }

        if(biasAttr->in_0_tensor_uid() == attributes.c_tensor_uid())
        {
            _biasUid = biasAttr->in_1_tensor_uid();
        }
        else if(biasAttr->in_1_tensor_uid().value() == attributes.c_tensor_uid())
        {
            _biasUid = biasAttr->in_0_tensor_uid();
        }
        else
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "MatmulParams: biasAttr tensor UIDs do not match c_tensor_uid");
        }

        const auto tBias = hipblaslt_utils::findTensorAttributes(tensorMap, _biasUid.value());
        const auto& biasDims = tBias.dims();

        PLUGIN_THROW_IF_TRUE(
            biasDims.empty() || biasDims.back() != cDims.back()
                || std::accumulate(
                       biasDims.cbegin(), biasDims.cend(), int64_t(1), std::multiplies<>())
                       != biasDims.back(),
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Bias tensor dims must be equal to column dimension of output matrix");
        biasDataType = hipblaslt_utils::tensorDataTypeToHipDataType(tBias.dataType());
    }

    // Row-major BLAS trick: to compute C = A * B with row-major data,
    // we compute C^T = B^T * A^T with column-major BLAS.
    // So we swap the transpose operations: transA in desc = getTrans(B), transB in desc = getTrans(A)
    _matmulDesc
        = HipblasltMatmulDesc(getTrans(tB), getTrans(tA), getComputeDataType(tA, tB), HIP_R_32F);
    setEpilogue(activAttr, biasDataType);

    if(aDims.size() > 2)
    {
        setBatchInfo(tA, tB, tC);
    }
}

void MatmulParams::setBatchInfo(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper& tA,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper& tB,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper& tC)
{
    // Batch support: we flatten all batch dimensions (all except last two) into a single batch count.
    // hipBLASLt uses a single strided batch offset, so we can only support uniform batch strides.
    // Supported: [3, M, K] x [1, K, N] or [3, M, K] x [3, K, N] - single batch dimension, uniform stride.
    // Not supported: [3, 1, M, K] x [1, 3, K, N] - would require non-uniform stride pattern for broadcasting.
    const int64_t aBatch = getBatchCount(tA.dims());
    const int64_t bBatch = getBatchCount(tB.dims());
    const int64_t cBatch = getBatchCount(tC.dims());

    if(aBatch != bBatch && aBatch != 1 && bBatch != 1)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported input matrix batch dimensions: they should be equal or one of them "
            "should be 1");
    }

    if(cBatch != std::max(aBatch, bBatch))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported matrix batch dimensions: output batch should be the same as the max "
            "of the input batch dimensions");
    }

    if(cBatch > 1)
    {
        _matrixLayoutA.setBatchCount(cBatch);
        _matrixLayoutB.setBatchCount(cBatch);
        _matrixLayoutC.setBatchCount(cBatch);

        size_t const rank = tA.dims().size();
        if(aBatch > 1)
        {
            _matrixLayoutA.setStridedBatchOffset(tA.strides()[rank - 3]);
        }
        if(bBatch > 1)
        {
            _matrixLayoutB.setStridedBatchOffset(tB.strides()[rank - 3]);
        }
        _matrixLayoutC.setStridedBatchOffset(tC.strides()[rank - 3]);
    }
}

void MatmulParams::setEpilogue(
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* activAttr,
    hipDataType biasDataType)
{
    auto epilogueParams
        = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(activAttr, _biasUid.has_value());

    _matmulDesc.setEpilogue(epilogueParams.epilogue);
    if(epilogueParams.act0 != 0.0f)
    {
        _matmulDesc.setEpilogueActArg0(epilogueParams.act0);
    }
    if(epilogueParams.act1 != 0.0f)
    {
        _matmulDesc.setEpilogueActArg1(epilogueParams.act1);
    }

    if(_biasUid.has_value())
    {
        _matmulDesc.setBiasDataType(biasDataType);

        // hipBLASLt requires initialized bias pointer in matmul descriptor for algorithm search.
        // Since the pointer is unknown on this stage, we initialize it by dummy pointer to update it in execution stage.
        _matmulDesc.setBiasPointer(reinterpret_cast<const void*>(0x1));
    }
}

const HipblasltMatrixLayout& MatmulParams::a() const
{
    return _matrixLayoutA;
}

const HipblasltMatrixLayout& MatmulParams::b() const
{
    return _matrixLayoutB;
}

const HipblasltMatrixLayout& MatmulParams::c() const
{
    return _matrixLayoutC;
}

const HipblasltMatmulDesc& MatmulParams::desc() const
{
    return _matmulDesc;
}

const std::optional<int64_t>& MatmulParams::biasUid() const
{
    return _biasUid;
}

MatmulPlan::MatmulPlan(const HipdnnEnginePluginHandle& handle, MatmulParams&& params)
    : _params(std::move(params))
{
    // hipBLASLt requests the max workspace size for the search algorithm
    // so that it fits within the available memory size.
    // So for better performance we set 128 MB here since
    // it is enough to get the most performant solution from hipblaslt.
    auto maxWorkspaceSize = static_cast<size_t>(128 * 1024 * 1024); // 128MB
    hipblasLtMatmulPreference_t prefHandle;
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulPreferenceCreate(&prefHandle));
    // Own the preference via RAII.
    hipdnn_data_sdk::utilities::ScopedResource<hipblasLtMatmulPreference_t> const pref(
        prefHandle, [](hipblasLtMatmulPreference_t p) {
            LOG_ON_HIPBLASLT_FAILURE(hipblasLtMatmulPreferenceDestroy(p));
        });

    THROW_ON_HIPBLASLT_FAILURE(
        hipblasLtMatmulPreferenceSetAttribute(pref.get(),
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &maxWorkspaceSize,
                                              sizeof(maxWorkspaceSize)));

    // Row-major BLAS trick: swap A and B layouts
    constexpr int REQUEST_SOLUTIONS = 1;
    std::array<hipblasLtMatmulHeuristicResult_t, REQUEST_SOLUTIONS> heuristicResult{};
    int returnedAlgoCount = 0;
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulAlgoGetHeuristic(handle.hipblasltHandle,
                                                               _params.desc().matmulDesc(),
                                                               _params.b().matrixLayout(),
                                                               _params.a().matrixLayout(),
                                                               _params.c().matrixLayout(),
                                                               _params.c().matrixLayout(),
                                                               pref.get(),
                                                               REQUEST_SOLUTIONS,
                                                               heuristicResult.data(),
                                                               &returnedAlgoCount));

    PLUGIN_THROW_IF_FALSE(returnedAlgoCount > 0,
                          HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                          "hipBLASLt has not found algorithm!");

    _heuristicResult = heuristicResult[0];
    _workspaceSize = _heuristicResult.workspaceSize;
}

size_t MatmulPlan::getWorkspaceSize([[maybe_unused]] const HipdnnEnginePluginHandle& handle) const
{
    return _workspaceSize;
}

void MatmulPlan::execute(const HipdnnEnginePluginHandle& handle,
                         const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                         uint32_t numDeviceBuffers,
                         void* workspace) const
{
    auto aBuffer
        = hipblaslt_utils::findDeviceBuffer(_params.a().uid(), deviceBuffers, numDeviceBuffers);
    auto bBuffer
        = hipblaslt_utils::findDeviceBuffer(_params.b().uid(), deviceBuffers, numDeviceBuffers);
    auto cBuffer
        = hipblaslt_utils::findDeviceBuffer(_params.c().uid(), deviceBuffers, numDeviceBuffers);

    // hipBLASLt writes per-call problem state through the descriptor and the bias pointer is
    // per call, so a descriptor shared across concurrent execute() calls on this plan would race.
    HipblasltMatmulDesc desc = _params.desc().clone();
    if(_params.biasUid().has_value())
    {
        auto biasBuffer = hipblaslt_utils::findDeviceBuffer(
            _params.biasUid().value(), deviceBuffers, numDeviceBuffers);
        desc.setBiasPointer(biasBuffer.ptr);
    }
    // A, B and C matrices are row-major. But hipBLASLt works with column-major matrices
    // To work with row-major matrices, we transpose them.
    // This is done by changing the order of A and B matrices in arguments:
    //  C = A * B => C^T = (A * B)^T => C^T = B^T * A^T
    // Due to this formula, we changed the order of A and B matrices in arguments
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmul(handle.hipblasltHandle,
                                               desc.matmulDesc(),
                                               &ALPHA,
                                               bBuffer.ptr,
                                               _params.b().matrixLayout(),
                                               aBuffer.ptr,
                                               _params.a().matrixLayout(),
                                               &BETA,
                                               cBuffer.ptr,
                                               _params.c().matrixLayout(),
                                               cBuffer.ptr,
                                               _params.c().matrixLayout(),
                                               &_heuristicResult.algo,
                                               workspace,
                                               _workspaceSize,
                                               handle.getStream()));
}

} // namespace hipblaslt_plugin
