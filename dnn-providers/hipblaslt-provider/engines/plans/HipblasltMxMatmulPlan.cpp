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
    // tXA/tXB are the graph's logical A/B (the block-scaled X inputs of dequant A/B).
    const auto tXA = hipblaslt_utils::findTensorAttributes(tensorMap, deqAttrA.x_tensor_uid());
    const auto tXB = hipblaslt_utils::findTensorAttributes(tensorMap, deqAttrB.x_tensor_uid());
    const auto tC = hipblaslt_utils::findTensorAttributes(tensorMap, matmulAttr.c_tensor_uid());

    // Row-major BLAS trick: hipDNN tensors are row-major but hipBLASLt is
    // column-major, so we compute C^T = B^T * A^T by presenting our B as
    // hipBLAS's A operand and our A as its B operand. The swap is applied once,
    // here, so every member is already in hipBLAS's frame and execute() needs no
    // further swapping: a()/aScaleUid() are what hipBLAS receives as A/A_SCALE.
    _matrixLayoutA = HipblasltMatrixLayout(tXB); // hipBLAS A <- our B
    _matrixLayoutB = HipblasltMatrixLayout(tXA); // hipBLAS B <- our A
    _matrixLayoutC = HipblasltMatrixLayout(tC);

    _aScaleUid = deqAttrB.scale_tensor_uid(); // hipBLAS A_SCALE <- our B's scale
    _bScaleUid = deqAttrA.scale_tensor_uid(); // hipBLAS B_SCALE <- our A's scale

    // Our A is [..., M, K], scales blocked 32-wide along K (innermost). Our A's
    // scale becomes hipBLAS's B_SCALE and is transposed [M, K/32] -> [K/32, M] at
    // execute time; M and the K-block count drive that transpose.
    const auto& aDims = tXA.dims();
    _m = aDims[aDims.size() - 2];
    _kBlocks = aDims[aDims.size() - 1] / VEC32_BLOCK_SIZE;

    // FP8 OCP MX GEMM always uses HIPBLAS_COMPUTE_32F. desc transA/transB are in
    // hipBLAS's frame too: transA = getTrans(hipBLAS A = our B), transB = our A.
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

    // Operands are already in hipBLAS's frame (swapped in MxMatmulParams), so
    // a()/b() are passed straight through as hipBLAS A/B.
    constexpr int REQUEST_SOLUTIONS = 1;
    std::array<hipblasLtMatmulHeuristicResult_t, REQUEST_SOLUTIONS> heuristicResult{};
    int returnedAlgoCount = 0;
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulAlgoGetHeuristic(handle.hipblasltHandle,
                                                               _params.desc().matmulDesc(),
                                                               _params.a().matrixLayout(),
                                                               _params.b().matrixLayout(),
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

    // Our A's scale (hipBLAS B_SCALE) is transposed on-device at execute time;
    // reserve aligned room for it at the front of the workspace so the plan owns
    // no device memory itself.
    const auto scaleBytes = static_cast<size_t>(_params.m() * _params.kBlocks());
    _scaleBufferBytes = alignUp(scaleBytes, WORKSPACE_ALIGNMENT);

    // Prebuild the transpose descriptors; reused on every execute. Our A's scale
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
    // getWorkspaceSize() always reports a non-zero size for MX GEMM (the scale
    // transpose region is reserved unconditionally), so a null workspace here is
    // a caller contract violation. Validate before dereferencing it below.
    PLUGIN_THROW_IF_TRUE(workspace == nullptr,
                         HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                         "MxMatmulPlan::execute: workspace is null but MX GEMM requires a "
                         "non-zero workspace (see getWorkspaceSize())");

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

    // Workspace layout: [ transposed B_SCALE | hipBLASLt matmul workspace ].
    void* transposedBScale = workspace;
    void* matmulWorkspace = static_cast<void*>(static_cast<char*>(workspace) + _scaleBufferBytes);

    // hipBLASLt expects A_SCALE as [K/32, m] and B_SCALE as [K/32, n]. A_SCALE
    // (our B's scale) is already in that layout. B_SCALE (our A's scale) is
    // [M, K/32] and must be physically transposed to [K/32, M], because scale
    // pointers carry no layout handle to swap for free.
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatrixTransform(handle.hipblasltHandle,
                                                        _scaleTransposeDesc.transformDesc(),
                                                        &ALPHA,
                                                        bScaleBuffer.ptr,
                                                        _scaleSrcLayout.matrixLayout(),
                                                        &BETA,
                                                        nullptr,
                                                        nullptr,
                                                        transposedBScale,
                                                        _scaleDstLayout.matrixLayout(),
                                                        handle.getStream()));

    // Clone the plan's descriptor so this call owns its own copy: scale pointers
    // are per-call device addresses, and mutating a shared descriptor would make
    // concurrent execute() calls on one plan instance race. Already in hipBLAS's
    // frame (swapped in MxMatmulParams).
    HipblasltMatmulDesc desc = _params.desc().clone();
    desc.setAScalePointer(aScaleBuffer.ptr);
    desc.setBScalePointer(transposedBScale);

    // Operands are already in hipBLAS's frame (swapped in MxMatmulParams), so
    // pass them straight through.
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmul(handle.hipblasltHandle,
                                               desc.matmulDesc(),
                                               &ALPHA,
                                               aBuffer.ptr,
                                               _params.a().matrixLayout(),
                                               bBuffer.ptr,
                                               _params.b().matrixLayout(),
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
