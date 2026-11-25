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
#include "common/ConvEpilogueFormulation.hpp"
#include "common/GemmFormulation.hpp"

namespace hipblaslt_plugin
{

namespace
{

/// The workspace hipBLASLt may ask for; the same ceiling the matmul engine gives it.
constexpr size_t MAX_WORKSPACE_BYTES = static_cast<size_t>(128) * 1024 * 1024;

constexpr float ALPHA = 1.0f;

static_assert(NATIVE_EPILOGUE_RELU == HIPBLASLT_EPILOGUE_RELU
                  && NATIVE_EPILOGUE_BIAS == HIPBLASLT_EPILOGUE_BIAS
                  && NATIVE_EPILOGUE_RELU_BIAS == HIPBLASLT_EPILOGUE_RELU_BIAS
                  && NATIVE_EPILOGUE_GELU == HIPBLASLT_EPILOGUE_GELU
                  && NATIVE_EPILOGUE_GELU_BIAS == HIPBLASLT_EPILOGUE_GELU_BIAS
                  && NATIVE_EPILOGUE_SIGMOID == HIPBLASLT_EPILOGUE_SIGMOID
                  && NATIVE_EPILOGUE_SWISH == HIPBLASLT_EPILOGUE_SWISH_EXT
                  && NATIVE_EPILOGUE_SWISH_BIAS == HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT
                  && NATIVE_EPILOGUE_CLAMP == HIPBLASLT_EPILOGUE_CLAMP_EXT
                  && NATIVE_EPILOGUE_CLAMP_BIAS == HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT,
              "the formulation's native epilogue values are hipBLASLt's");

/// Sets @p epilogue on @p desc. The bias pointer is a placeholder until execution: hipBLASLt
/// wants one set for its algorithm search.
void setNativeEpilogue(const HipblasltMatmulDesc& desc, const NativeEpilogue& epilogue)
{
    const auto mode = static_cast<hipblasLtEpilogue_t>(epilogue.mode);
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulDescSetAttribute(
        desc.matmulDesc(), HIPBLASLT_MATMUL_DESC_EPILOGUE, &mode, sizeof(mode)));
    if(epilogue.mode == NATIVE_EPILOGUE_CLAMP || epilogue.mode == NATIVE_EPILOGUE_CLAMP_BIAS)
    {
        THROW_ON_HIPBLASLT_FAILURE(
            hipblasLtMatmulDescSetAttribute(desc.matmulDesc(),
                                            HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT,
                                            &epilogue.act0,
                                            sizeof(epilogue.act0)));
        THROW_ON_HIPBLASLT_FAILURE(
            hipblasLtMatmulDescSetAttribute(desc.matmulDesc(),
                                            HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT,
                                            &epilogue.act1,
                                            sizeof(epilogue.act1)));
    }
    if(epilogue.biasUid.has_value())
    {
        const hipDataType biasType
            = hipblaslt_utils::tensorDataTypeToHipDataType(epilogue.biasType);
        THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulDescSetAttribute(
            desc.matmulDesc(), HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &biasType, sizeof(biasType)));
        const void* placeholder = reinterpret_cast<const void*>(0x1);
        THROW_ON_HIPBLASLT_FAILURE(
            hipblasLtMatmulDescSetAttribute(desc.matmulDesc(),
                                            HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                            static_cast<const void*>(&placeholder),
                                            sizeof(placeholder)));
    }
}

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
                                     const std::vector<GemmCall>& calls,
                                     bool deterministic)
{
    hipblasLtMatmulPreference_t preferenceHandle = nullptr;
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulPreferenceCreate(&preferenceHandle));
    const hipdnn_data_sdk::utilities::ScopedResource<hipblasLtMatmulPreference_t> preference(
        preferenceHandle, [](hipblasLtMatmulPreference_t p) {
            LOG_ON_HIPBLASLT_FAILURE(hipblasLtMatmulPreferenceDestroy(p));
        });
    // Deterministic: no hipBLASLt workspace, so no workspace split of a reduction.
    const size_t maxWorkspace = deterministic ? 0 : MAX_WORKSPACE_BYTES;
    THROW_ON_HIPBLASLT_FAILURE(
        hipblasLtMatmulPreferenceSetAttribute(preference.get(),
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &maxWorkspace,
                                              sizeof(maxWorkspace)));

    for(auto& call : fitLibraryLimits(calls))
    {
        // A fused epilogue applies once, to the whole result: never to a part of a split call.
        PLUGIN_THROW_IF_FALSE(!(call.epilogue.has_value() && call.accumulate),
                              HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                              "a native epilogue on an accumulating GEMM");
        PLUGIN_THROW_IF_FALSE(!(call.c.has_value() && call.accumulate),
                              HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                              "an addend on an accumulating GEMM");
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
                              {},
                              std::nullopt};
        if(call.c.has_value())
        {
            prepared.c = layoutFor(*call.c, call.m, call.n, call.batchCount);
        }
        const auto& cLayout = prepared.c.has_value() ? *prepared.c : prepared.d;

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
            if(call.epilogue.has_value())
            {
                setNativeEpilogue(prepared.desc, *call.epilogue);
            }
            returned = 0;
            const auto status = hipblasLtMatmulAlgoGetHeuristic(handle.hipblasltHandle,
                                                                prepared.desc.matmulDesc(),
                                                                prepared.a.matrixLayout(),
                                                                prepared.b.matrixLayout(),
                                                                cLayout.matrixLayout(),
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
        const float beta = call.accumulate || call.c.has_value() ? 1.0f : 0.0f;
        // c is d (read only when accumulating), or the addend.
        const void* addend = output;
        if(call.c.has_value())
        {
            addend = operandAddress(
                hipdnn_plugin_sdk::findDeviceBuffer(call.c->uid, deviceBuffers, numDeviceBuffers)
                    .ptr,
                *call.c);
        }
        const auto& cLayout = prepared.c.has_value() ? *prepared.c : prepared.d;
        if(call.epilogue.has_value() && call.epilogue->biasUid.has_value())
        {
            const void* bias = hipdnn_plugin_sdk::findDeviceBuffer(
                                   *call.epilogue->biasUid, deviceBuffers, numDeviceBuffers)
                                   .ptr;
            THROW_ON_HIPBLASLT_FAILURE(
                hipblasLtMatmulDescSetAttribute(prepared.desc.matmulDesc(),
                                                HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                static_cast<const void*>(&bias),
                                                sizeof(bias)));
        }

        THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmul(handle.hipblasltHandle,
                                                   prepared.desc.matmulDesc(),
                                                   &ALPHA,
                                                   operandAddress(a.ptr, call.a),
                                                   prepared.a.matrixLayout(),
                                                   operandAddress(b.ptr, call.b),
                                                   prepared.b.matrixLayout(),
                                                   &beta,
                                                   addend,
                                                   cLayout.matrixLayout(),
                                                   output,
                                                   prepared.d.matrixLayout(),
                                                   &prepared.heuristic.algo,
                                                   workspace,
                                                   prepared.heuristic.workspaceSize,
                                                   handle.getStream()));
    }
}

} // namespace hipblaslt_plugin
