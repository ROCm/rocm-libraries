// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <cstddef>
#include <cstdint>
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

// Matches hipMalloc's 256-byte alignment guarantee, so the matmul workspace
// following the reserved scale region stays as aligned as a fresh allocation.
constexpr size_t WORKSPACE_ALIGNMENT = 256;

size_t alignUp(size_t value, size_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
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

    // A is [..., M, K], scales are blocked 32-wide along K (innermost). M and the
    // K-block count drive the scale_A transpose performed at execute time.
    const auto& aDims = tXA.dims();
    _m = aDims[aDims.size() - 2];
    _kBlocks = aDims[aDims.size() - 1] / 32;

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

int64_t MxMatmulParams::m() const
{
    return _m;
}

int64_t MxMatmulParams::kBlocks() const
{
    return _kBlocks;
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

    // scale_A is transposed on-device at execute time; reserve aligned room for
    // it at the front of the workspace so the plan owns no device memory itself.
    const auto scaleBytes = static_cast<size_t>(_params.m() * _params.kBlocks());
    _scaleBufferBytes = alignUp(scaleBytes, WORKSPACE_ALIGNMENT);

    // Prebuild the scale_A transpose descriptors; reused on every execute. scale_A
    // [M, K/32] row-major is viewed column-major as (K/32, M) on input and written
    // as (M, K/32) column-major == [K/32, M] row-major on output. R_8I moves the
    // UE8M0 bytes verbatim without interpreting them numerically.
    const auto m = static_cast<uint64_t>(_params.m());
    const auto kBlocks = static_cast<uint64_t>(_params.kBlocks());
    _scaleTransposeDesc = HipblasltMatrixTransformDesc(HIP_R_32F, HIPBLAS_OP_T);
    _scaleSrcLayout = HipblasltMatrixLayout(HIP_R_8I, kBlocks, m, static_cast<int64_t>(kBlocks));
    _scaleDstLayout = HipblasltMatrixLayout(HIP_R_8I, m, kBlocks, static_cast<int64_t>(m));
}

size_t MxMatmulPlan::getWorkspaceSize([[maybe_unused]] const HipdnnEnginePluginHandle& handle) const
{
    return _scaleBufferBytes + _workspaceSize;
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

    // Workspace layout: [ transposed scale_A | hipBLASLt matmul workspace ].
    void* transposedAScale = workspace;
    void* matmulWorkspace = static_cast<void*>(static_cast<char*>(workspace) + _scaleBufferBytes);

    // hipBLASLt expects A_SCALE as [k/32, m'] and B_SCALE as [k/32, n'] for the
    // GEMM it runs. Under the row-major A/B operand swap (m'=N, n'=M), its A
    // operand is our B and its B operand is our A:
    //   A_SCALE ← our scale_B ([K/32, N]) — already in the expected layout.
    //   B_SCALE ← our scale_A transposed from [M, K/32] to [K/32, M].
    // The operands are swapped for free via their layout handles (below), but
    // scale pointers carry no layout, so scale_A must be physically transposed.
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatrixTransform(handle.hipblasltHandle,
                                                        _scaleTransposeDesc.transformDesc(),
                                                        &ALPHA,
                                                        aScaleBuffer.ptr,
                                                        _scaleSrcLayout.matrixLayout(),
                                                        &BETA,
                                                        nullptr,
                                                        nullptr,
                                                        transposedAScale,
                                                        _scaleDstLayout.matrixLayout(),
                                                        handle.getStream()));

    // Scale pointers are device addresses, so they are set here rather than at
    // build time (where the scale modes were set).
    _params.desc().setAScalePointer(bScaleBuffer.ptr);
    _params.desc().setBScalePointer(transposedAScale);

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
                                               matmulWorkspace,
                                               _workspaceSize,
                                               handle.getStream()));
}

} // namespace hipblaslt_plugin
