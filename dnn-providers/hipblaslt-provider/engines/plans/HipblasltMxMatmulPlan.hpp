// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include <hipblaslt/hipblaslt.h>
#include <hipdnn_flatbuffers_sdk/data_objects/block_scale_dequantize_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/matmul_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

#include "HipblasltMatmulDesc.hpp"
#include "HipblasltMatrixLayout.hpp"
#include "HipblasltMatrixTransformDesc.hpp"
#include "PlanInterface.hpp"

namespace hipblaslt_plugin
{

// VEC32_UE8M0 block scaling stores one scale value per 32-element block along K.
constexpr int64_t VEC32_BLOCK_SIZE = 32;

class MxMatmulParams
{
public:
    MxMatmulParams(
        const hipdnn_flatbuffers_sdk::data_objects::BlockScaleDequantizeAttributes& deqAttrA,
        const hipdnn_flatbuffers_sdk::data_objects::BlockScaleDequantizeAttributes& deqAttrB,
        const hipdnn_flatbuffers_sdk::data_objects::MatmulAttributes& matmulAttr,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    MxMatmulParams(const MxMatmulParams&) = delete;
    MxMatmulParams& operator=(const MxMatmulParams&) = delete;

    MxMatmulParams(MxMatmulParams&&) = default;
    MxMatmulParams& operator=(MxMatmulParams&&) = default;

    const HipblasltMatrixLayout& a() const;
    const HipblasltMatrixLayout& b() const;
    const HipblasltMatrixLayout& c() const;

    const HipblasltMatmulDesc& desc() const;

    // Scale UIDs in hipBLAS's frame: aScaleUid() is our B's scale (A_SCALE),
    // bScaleUid() is our A's scale (B_SCALE). See the constructor for the swap.
    int64_t aScaleUid() const;
    int64_t bScaleUid() const;

    // Logical GEMM M and the number of 32-wide K blocks (K / 32). Used to
    // transpose our A's scale from [M, K/32] to [K/32, M] for hipBLASLt's B_SCALE.
    int64_t m() const;
    int64_t kBlocks() const;

private:
    HipblasltMatmulDesc _matmulDesc;
    HipblasltMatrixLayout _matrixLayoutA;
    HipblasltMatrixLayout _matrixLayoutB;
    HipblasltMatrixLayout _matrixLayoutC;
    int64_t _aScaleUid;
    int64_t _bScaleUid;
    int64_t _m;
    int64_t _kBlocks;
};

class MxMatmulPlan : public IPlan
{
public:
    MxMatmulPlan(const HipdnnEnginePluginHandle& handle, MxMatmulParams&& params);
    ~MxMatmulPlan() override = default;

    MxMatmulPlan(const MxMatmulPlan&) = delete;
    MxMatmulPlan& operator=(const MxMatmulPlan&) = delete;

    MxMatmulPlan(MxMatmulPlan&&) = default;
    MxMatmulPlan& operator=(MxMatmulPlan&&) = default;

    size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle) const override;

    void execute(const HipdnnEnginePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    MxMatmulParams _params;
    hipblasLtMatmulHeuristicResult_t _heuristicResult;

    // hipBLASLt matmul workspace size (from the heuristic).
    size_t _workspaceSize = 0;

    // Aligned front region of the workspace reserved for the transposed B_SCALE
    // (our A's scale).
    size_t _scaleBufferBytes = 0;

    // Prebuilt descriptors for the on-device transpose of our A's scale (fed as
    // hipBLAS B_SCALE).
    HipblasltMatrixTransformDesc _scaleTransposeDesc;
    HipblasltMatrixLayout _scaleSrcLayout;
    HipblasltMatrixLayout _scaleDstLayout;

    static constexpr float ALPHA = 1.f;
    static constexpr float BETA = 0.f;
};

} // namespace hipblaslt_plugin
