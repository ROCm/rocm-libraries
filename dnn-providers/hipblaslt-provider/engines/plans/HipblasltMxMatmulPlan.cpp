// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <cstddef>
#include <string>

#include <hipblaslt/hipblaslt.h>
#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "HipblasltMxMatmulPlan.hpp"
#include "HipblasltUtils.hpp"
#include "HipdnnEnginePluginHandle.hpp"

namespace hipblaslt_plugin
{
namespace
{

// Infer the hipBLASLt transpose op from a tensor's strides: row-major
// (stride[-1]==1) → HIPBLAS_OP_N; column-major (stride[-2]==1) → HIPBLAS_OP_T.
// Mirrors MatmulParams::getTrans for the plain matmul plan.
hipblasOperation_t getTransFromStrides(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper& t)
{
    const auto& strides = t.strides();
    PLUGIN_THROW_IF_FALSE(strides.size() > 1,
                          HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                          "Unsupported stride for MX input matrix: " + t.name());
    if(strides[strides.size() - 1] == 1)
    {
        return HIPBLAS_OP_N;
    }
    if(strides[strides.size() - 2] == 1)
    {
        return HIPBLAS_OP_T;
    }
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Unsupported stride for MX input matrix: " + t.name());
}

} // namespace

MxMatmulParams::MxMatmulParams(
    const hipdnn_flatbuffers_sdk::data_objects::BlockScaleDequantizeAttributes& deqAttrA,
    const hipdnn_flatbuffers_sdk::data_objects::BlockScaleDequantizeAttributes& deqAttrB,
    const hipdnn_flatbuffers_sdk::data_objects::MatmulAttributes& matmulAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    const auto tXA = hipblaslt_utils::findTensorAttributes(tensorMap, deqAttrA.x_tensor_uid());
    const auto tXB = hipblaslt_utils::findTensorAttributes(tensorMap, deqAttrB.x_tensor_uid());
    const auto tC = hipblaslt_utils::findTensorAttributes(tensorMap, matmulAttr.c_tensor_uid());

    _matrixLayoutA = HipblasltMatrixLayout(tXA);
    _matrixLayoutB = HipblasltMatrixLayout(tXB);
    _matrixLayoutC = HipblasltMatrixLayout(tC);

    _aScaleUid = deqAttrA.scale_tensor_uid();
    _bScaleUid = deqAttrB.scale_tensor_uid();

    // Row-major BLAS trick: swap transA/transB (same as MatmulParams).
    // FP8 OCP MX GEMM always uses HIPBLAS_COMPUTE_32F.
    // Row-major swap: desc transA = getTrans(B), desc transB = getTrans(A)
    _matmulDesc = HipblasltMatmulDesc(
        getTransFromStrides(tXB), getTransFromStrides(tXA), HIPBLAS_COMPUTE_32F, HIP_R_32F);

    _matmulDesc.setAScaleMode(HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
    _matmulDesc.setBScaleMode(HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
}

const HipblasltMatrixLayout& MxMatmulParams::a() const
{
    return _matrixLayoutA;
}

const HipblasltMatrixLayout& MxMatmulParams::b() const
{
    return _matrixLayoutB;
}

const HipblasltMatrixLayout& MxMatmulParams::c() const
{
    return _matrixLayoutC;
}

const HipblasltMatmulDesc& MxMatmulParams::desc() const
{
    return _matmulDesc;
}

HipblasltMatmulDesc& MxMatmulParams::desc()
{
    return _matmulDesc;
}

int64_t MxMatmulParams::aScaleUid() const
{
    return _aScaleUid;
}

int64_t MxMatmulParams::bScaleUid() const
{
    return _bScaleUid;
}

MxMatmulPlan::MxMatmulPlan(const HipdnnEnginePluginHandle& handle, MxMatmulParams&& params)
    : _params(std::move(params))
{
    // Same max workspace approach as MatmulPlan: 128 MB to allow hipBLASLt to
    // find the most performant MX GEMM algorithm.
    auto maxWorkspaceSize = static_cast<size_t>(128 * 1024 * 1024); // 128 MB
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

    // Row-major BLAS trick: swap A and B layouts (same as MatmulPlan)
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
                          "hipBLASLt has not found MX GEMM algorithm!");

    _heuristicResult = heuristicResult[0];
    _workspaceSize = _heuristicResult.workspaceSize;
}

size_t MxMatmulPlan::getWorkspaceSize([[maybe_unused]] const HipdnnEnginePluginHandle& handle) const
{
    return _workspaceSize;
}

void MxMatmulPlan::execute(const HipdnnEnginePluginHandle& handle,
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
    auto aScaleBuffer
        = hipblaslt_utils::findDeviceBuffer(_params.aScaleUid(), deviceBuffers, numDeviceBuffers);
    auto bScaleBuffer
        = hipblaslt_utils::findDeviceBuffer(_params.bScaleUid(), deviceBuffers, numDeviceBuffers);

    // Set scale pointers at execute time.
    // Due to the row-major B/A swap trick, desc "A" in hipBLASLt is our B and vice-versa.
    // So we swap the scale pointer assignment to match:
    //   hipBLASLt A_SCALE_POINTER ← our B scale
    //   hipBLASLt B_SCALE_POINTER ← our A scale
    _params.desc().setBScalePointer(aScaleBuffer.ptr);
    _params.desc().setAScalePointer(bScaleBuffer.ptr);

    // Row-major BLAS trick: swap A and B (C = A*B → C^T = B^T * A^T)
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmul(handle.hipblasltHandle,
                                               _params.desc().matmulDesc(),
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
