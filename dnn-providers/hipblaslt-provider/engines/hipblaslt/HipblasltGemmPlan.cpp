// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/hipblaslt/HipblasltGemmPlan.hpp"

#include <algorithm>
#include <array>
#include <vector>

#include <hipblaslt/hipblaslt.h>
#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "HipblasltUtils.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "common/GemmFormulation.hpp"

namespace hipblaslt_plugin
{

namespace
{

/// The workspace hipBLASLt may ask for; the same ceiling the matmul engine gives it.
constexpr size_t MAX_WORKSPACE_BYTES = static_cast<size_t>(128) * 1024 * 1024;

constexpr float ALPHA = 1.0f;

/// A column-major layout for @p operand, @p rows x @p cols as stored, batched like the call.
HipblasltMatrixLayout
    layoutFor(const GemmOperand& operand, int64_t rows, int64_t cols, int64_t batchCount)
{
    HipblasltMatrixLayout layout(hipblaslt_utils::tensorDataTypeToHipDataType(operand.dataType),
                                 static_cast<uint64_t>(rows),
                                 static_cast<uint64_t>(cols),
                                 operand.leadingDimension);
    if(batchCount > 1)
    {
        layout.setBatchCount(batchCount);
        layout.setStridedBatchOffset(operand.batchStride);
    }
    return layout;
}

} // namespace

HipblasltGemmPlan::HipblasltGemmPlan(const HipdnnEnginePluginHandle& handle,
                                     std::vector<GemmCall> calls)
{
    hipblasLtMatmulPreference_t preferenceHandle = nullptr;
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulPreferenceCreate(&preferenceHandle));
    const hipdnn_data_sdk::utilities::ScopedResource<hipblasLtMatmulPreference_t> preference(
        preferenceHandle, [](hipblasLtMatmulPreference_t p) {
            LOG_ON_HIPBLASLT_FAILURE(hipblasLtMatmulPreferenceDestroy(p));
        });
    const size_t maxWorkspace = MAX_WORKSPACE_BYTES;
    THROW_ON_HIPBLASLT_FAILURE(
        hipblasLtMatmulPreferenceSetAttribute(preference.get(),
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &maxWorkspace,
                                              sizeof(maxWorkspace)));

    for(auto& call : fitLibraryLimits(std::move(calls)))
    {
        // Stored shapes: op(a) is m x k, op(b) is k x n.
        PreparedCall prepared{call,
                              {},
                              layoutFor(call.a,
                                        call.transposeA ? call.k : call.m,
                                        call.transposeA ? call.m : call.k,
                                        call.batchCount),
                              layoutFor(call.b,
                                        call.transposeB ? call.n : call.k,
                                        call.transposeB ? call.k : call.n,
                                        call.batchCount),
                              layoutFor(call.d, call.m, call.n, call.batchCount),
                              {}};

        // TF32 where the call permits it and hipBLASLt has an algorithm for it on this GPU;
        // fp32 otherwise, the fallback TF32 asks for.
        std::vector<hipblasComputeType_t> computeTypes;
        if(call.tf32 && call.a.dataType == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT
           && call.b.dataType == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
        {
            computeTypes.push_back(HIPBLAS_COMPUTE_32F_FAST_TF32);
        }
        computeTypes.push_back(HIPBLAS_COMPUTE_32F);

        int returned = 0;
        std::array<hipblasLtMatmulHeuristicResult_t, 1> results{};
        for(const auto computeType : computeTypes)
        {
            prepared.desc = HipblasltMatmulDesc(call.transposeA ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                                                call.transposeB ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                                                computeType,
                                                HIP_R_32F);
            returned = 0;
            const auto status = hipblasLtMatmulAlgoGetHeuristic(handle.hipblasltHandle,
                                                                prepared.desc.matmulDesc(),
                                                                prepared.a.matrixLayout(),
                                                                prepared.b.matrixLayout(),
                                                                prepared.d.matrixLayout(),
                                                                prepared.d.matrixLayout(),
                                                                preference.get(),
                                                                1,
                                                                results.data(),
                                                                &returned);
            if(status == HIPBLAS_STATUS_SUCCESS && returned > 0)
            {
                break;
            }
            if(computeType == HIPBLAS_COMPUTE_32F)
            {
                THROW_ON_HIPBLASLT_FAILURE(status);
            }
        }
        PLUGIN_THROW_IF_FALSE(returned > 0,
                              HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                              "hipBLASLt found no algorithm for a convolution GEMM");
        prepared.heuristic = results[0];
        _workspaceSize = std::max(_workspaceSize, prepared.heuristic.workspaceSize);
        _calls.push_back(std::move(prepared));
    }
}

size_t HipblasltGemmPlan::getWorkspaceSize(const HipdnnEnginePluginHandle& /*handle*/) const
{
    return _workspaceSize;
}

void HipblasltGemmPlan::execute(const HipdnnEnginePluginHandle& handle,
                                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                uint32_t numDeviceBuffers,
                                void* workspace) const
{
    for(const auto& prepared : _calls)
    {
        const auto& call = prepared.call;
        const auto a
            = hipdnn_plugin_sdk::findDeviceBuffer(call.a.uid, deviceBuffers, numDeviceBuffers);
        const auto b
            = hipdnn_plugin_sdk::findDeviceBuffer(call.b.uid, deviceBuffers, numDeviceBuffers);
        const auto d
            = hipdnn_plugin_sdk::findDeviceBuffer(call.d.uid, deviceBuffers, numDeviceBuffers);
        void* output = operandAddress(d.ptr, call.d);
        const float beta = call.accumulate ? 1.0f : 0.0f;

        // c is d: read only when accumulating.
        THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmul(handle.hipblasltHandle,
                                                   prepared.desc.matmulDesc(),
                                                   &ALPHA,
                                                   operandAddress(a.ptr, call.a),
                                                   prepared.a.matrixLayout(),
                                                   operandAddress(b.ptr, call.b),
                                                   prepared.b.matrixLayout(),
                                                   &beta,
                                                   output,
                                                   prepared.d.matrixLayout(),
                                                   output,
                                                   prepared.d.matrixLayout(),
                                                   &prepared.heuristic.algo,
                                                   workspace,
                                                   prepared.heuristic.workspaceSize,
                                                   handle.getStream()));
    }
}

} // namespace hipblaslt_plugin
