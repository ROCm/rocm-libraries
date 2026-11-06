// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/rocblas/RocblasMatmulPlanBuilder.hpp"

#include <memory>

#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "common/MatmulProblem.hpp"
#include "engines/rocblas/RocblasGemm.hpp"

namespace hipblaslt_plugin
{

bool RocblasMatmulPlanBuilder::isApplicable(
    const HipdnnEnginePluginHandle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return gemmForMatmul(opGraph).has_value();
}

size_t RocblasMatmulPlanBuilder::getWorkspaceSize(
    const HipdnnEnginePluginHandle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/) const
{
    return 0;
}

void RocblasMatmulPlanBuilder::buildPlan(
    const HipdnnEnginePluginHandle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    HipdnnEnginePluginExecutionContext& executionContext) const
{
    auto call = gemmForMatmul(opGraph);
    if(!call.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "rocBLAS matmul does not apply to this graph");
    }
    HIPDNN_PLUGIN_LOG_INFO(
        "Building rocBLAS matmul plan for node: " << opGraph.getNodeWrapper(0).name());
    executionContext.setPlan(std::make_unique<RocblasMatmulPlan>(*call));
}

RocblasMatmulPlan::RocblasMatmulPlan(GemmCall call)
    : _call(call)
{
}

size_t RocblasMatmulPlan::getWorkspaceSize(const HipdnnEnginePluginHandle& /*handle*/) const
{
    return 0;
}

void RocblasMatmulPlan::execute(const HipdnnEnginePluginHandle& handle,
                                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                uint32_t numDeviceBuffers,
                                void* /*workspace*/) const
{
    const auto a
        = hipdnn_plugin_sdk::findDeviceBuffer(_call.a.uid, deviceBuffers, numDeviceBuffers);
    const auto b
        = hipdnn_plugin_sdk::findDeviceBuffer(_call.b.uid, deviceBuffers, numDeviceBuffers);
    const auto d
        = hipdnn_plugin_sdk::findDeviceBuffer(_call.d.uid, deviceBuffers, numDeviceBuffers);

    const rocblas_handle rocblasHandle = handle.rocblasHandle();
    if(rocblas_set_stream(rocblasHandle, handle.getStream()) != rocblas_status_success)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "rocblas_set_stream failed");
    }
    rocblas_backend::runGemm(rocblasHandle, _call, a.ptr, b.ptr, d.ptr);
}

} // namespace hipblaslt_plugin
