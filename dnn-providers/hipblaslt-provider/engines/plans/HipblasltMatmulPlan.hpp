// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include <hipblaslt/hipblaslt.h>
#include <hipdnn_flatbuffers_sdk/data_objects/matmul_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

#include "HipblasltMatmulDesc.hpp"
#include "HipblasltMatrixLayout.hpp"
#include "PlanInterface.hpp"

namespace hipblaslt_plugin
{
class MatmulParams
{
public:
    MatmulParams(
        const hipdnn_flatbuffers_sdk::data_objects::MatmulAttributes& attributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);
    MatmulParams(
        const hipdnn_flatbuffers_sdk::data_objects::MatmulAttributes& attributes,
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* biasAttr,
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* activAttr,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    MatmulParams(const MatmulParams&) = delete;
    MatmulParams& operator=(const MatmulParams&) = delete;

    MatmulParams(MatmulParams&&) = default;
    MatmulParams& operator=(MatmulParams&&) = default;

    const HipblasltMatrixLayout& a() const;
    const HipblasltMatrixLayout& b() const;
    const HipblasltMatrixLayout& c() const;

    const HipblasltMatmulDesc& desc() const;
    const std::optional<int64_t>& biasUid() const;

private:
    static hipblasOperation_t
        getTrans(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper& t);
    static hipblasComputeType_t getComputeDataType(
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper& tA,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper& tB);

    void setBatchInfo(
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper& tA,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper& tB,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper& tC);

    void setEpilogue(const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* activAttr,
                     hipDataType biasDataType);

    HipblasltMatmulDesc _matmulDesc;
    HipblasltMatrixLayout _matrixLayoutA;
    HipblasltMatrixLayout _matrixLayoutB;
    HipblasltMatrixLayout _matrixLayoutC;
    std::optional<int64_t> _biasUid;
};

class MatmulPlan : public IPlan
{
public:
    MatmulPlan(const HipdnnEnginePluginHandle& handle, MatmulParams&& params);
    ~MatmulPlan() override = default;

    MatmulPlan(const MatmulPlan&) = delete;
    MatmulPlan& operator=(const MatmulPlan&) = delete;

    MatmulPlan(MatmulPlan&& other) = default;
    MatmulPlan& operator=(MatmulPlan&& other) = default;

    size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle) const override;

    void execute(const HipdnnEnginePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    MatmulParams _params;
    hipblasLtMatmulHeuristicResult_t _heuristicResult;
    size_t _workspaceSize = 0;

    static constexpr float _alpha = 1.f;
    static constexpr float _beta = 0.f;
};

} // namespace hipblaslt_plugin
