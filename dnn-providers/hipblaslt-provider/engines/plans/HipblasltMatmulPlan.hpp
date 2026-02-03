// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include <hipblaslt/hipblaslt.h>
#include <hipdnn_data_sdk/data_objects/matmul_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>

#include "HipblasltMatmulDesc.hpp"
#include "HipblasltMatrixLayout.hpp"
#include "PlanInterface.hpp"

namespace hipblaslt_plugin
{

class MatmulParams
{
public:
    MatmulParams(
        const hipdnn_data_sdk::data_objects::MatmulAttributes& attributes,
        const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    MatmulParams(const MatmulParams&) = delete;
    MatmulParams& operator=(const MatmulParams&) = delete;

    MatmulParams(MatmulParams&&) = default;
    MatmulParams& operator=(MatmulParams&&) = default;

    const HipblasltMatrixLayout& a() const;
    const HipblasltMatrixLayout& b() const;
    const HipblasltMatrixLayout& c() const;

    const HipblasltMatmulDesc& desc() const;

private:
    static hipblasOperation_t getTrans(const hipdnn_plugin_sdk::TensorAttributesWrapper& t);
    static hipblasComputeType_t
        getComputeDataType(const hipdnn_plugin_sdk::TensorAttributesWrapper& tA,
                           const hipdnn_plugin_sdk::TensorAttributesWrapper& tB);

    HipblasltMatmulDesc _matmulDesc;
    HipblasltMatrixLayout _matrixLayoutA;
    HipblasltMatrixLayout _matrixLayoutB;
    HipblasltMatrixLayout _matrixLayoutC;
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
