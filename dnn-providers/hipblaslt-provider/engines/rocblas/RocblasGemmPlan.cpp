// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/rocblas/RocblasGemmPlan.hpp"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>

#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "HipdnnEnginePluginHandle.hpp"
#include "common/GemmFormulation.hpp"
#include "engines/rocblas/RocblasGemm.hpp"

namespace hipblaslt_plugin
{

RocblasGemmPlan::RocblasGemmPlan(const std::vector<GemmCall>& calls, const std::string& gcnArchName)
    : _calls(fitLibraryLimits(calls))
{
    // rocBLAS has no epilogue: a fused convolution applies its own in a repack pass.
    if(std::any_of(_calls.begin(), _calls.end(), [](const GemmCall& call) {
           return call.epilogue.has_value();
       }))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "rocBLAS GEMMs take no native epilogue");
    }
    // A reduction rocBLAS would chunk into a half or bf16 output rounds once per chunk:
    // declined, so the result is summed in fp32 and rounded once, or not computed here.
    if(std::any_of(_calls.begin(), _calls.end(), rocblasRoundsReductionChunks))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "rocBLAS would split a reduction into a narrow output, rounding between the parts");
    }
    // So is a bf16 GEMV whose kernel rounds each product to bf16.
    const std::string arch
        = gcnArchName.empty() ? rocblas_backend::currentGcnArchName() : gcnArchName;
    if(std::any_of(_calls.begin(), _calls.end(), [&arch](const GemmCall& call) {
           return rocblasGemvRoundsBf16Products(call, arch);
       }))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "rocBLAS would run a bf16 GEMV that rounds every product to bf16 on " + arch);
    }
}

void requireTunedRocblasKernels(const std::vector<GemmCall>& calls, const std::string& gcnArchName)
{
    const std::string arch
        = gcnArchName.empty() ? rocblas_backend::currentGcnArchName() : gcnArchName;
    if(std::any_of(calls.begin(), calls.end(), [&arch](const GemmCall& call) {
           return rocblasFp32OutputIsUntuned(call, arch);
       }))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "rocBLAS has only untuned kernels for a 16-bit GEMM into fp32 on " + arch);
    }
}

size_t RocblasGemmPlan::getWorkspaceSize(const HipdnnEnginePluginHandle& /*handle*/) const
{
    return 0;
}

void RocblasGemmPlan::execute(const HipdnnEnginePluginHandle& handle,
                              const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                              uint32_t numDeviceBuffers,
                              void* /*workspace*/) const
{
    rocblas_handle rocblasHandle = handle.rocblasHandle();
    if(rocblas_set_stream(rocblasHandle, handle.getStream()) != rocblas_status_success)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "rocblas_set_stream failed");
    }
    std::optional<rocblas_backend::ScopedNoAtomics> noAtomics;
    if(_bitwiseReproducible)
    {
        noAtomics.emplace(rocblasHandle);
    }
    std::optional<rocblas_backend::ScopedXf32> xf32;
    if(std::any_of(_calls.begin(), _calls.end(), rocblas_backend::permitsXf32))
    {
        xf32.emplace(rocblasHandle);
    }
    for(const auto& call : _calls)
    {
        const auto a
            = hipdnn_plugin_sdk::findDeviceBuffer(call.a.uid, deviceBuffers, numDeviceBuffers);
        const auto b
            = hipdnn_plugin_sdk::findDeviceBuffer(call.b.uid, deviceBuffers, numDeviceBuffers);
        const auto d
            = hipdnn_plugin_sdk::findDeviceBuffer(call.d.uid, deviceBuffers, numDeviceBuffers);
        const void* addend = nullptr;
        if(call.c.has_value())
        {
            addend = operandAddress(
                hipdnn_plugin_sdk::findDeviceBuffer(call.c->uid, deviceBuffers, numDeviceBuffers)
                    .ptr,
                *call.c);
        }
        rocblas_backend::runGemm(rocblasHandle,
                                 call,
                                 operandAddress(a.ptr, call.a),
                                 operandAddress(b.ptr, call.b),
                                 operandAddress(d.ptr, call.d),
                                 addend);
    }
}

} // namespace hipblaslt_plugin
