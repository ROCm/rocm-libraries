// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <hipblaslt/hipblaslt.h>
#include <hipdnn_flatbuffers_sdk/data_objects/block_scale_dequantize_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/matmul_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/TensorAttributesWrapper.hpp>

#include "HipblasltMatmulDesc.hpp"
#include "HipblasltMatrixLayout.hpp"
#include "HipblasltMxScaleLayout.hpp"
#include "PlanInterface.hpp"
#include "Workarounds.hpp"

namespace hipblaslt_plugin
{

// VEC32_UE8M0 block scaling stores one scale value per 32-element block along K.
constexpr int64_t VEC32_BLOCK_SIZE = 32;

// Order of a densely packed matrix (the two innermost dims of `tensor`; leading
// batch dims are ignored). `kAxisFromEnd` is 1 when the K axis is innermost
// ([MN, K], our A and its scale) and 2 when K is the second-to-last axis
// ([K, MN], our B and its scale). Returns nullopt when the matrix is neither a
// packed row-major nor a packed column-major matrix.
std::optional<MxOrder> packedMxOrder(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper& tensor,
    size_t kAxisFromEnd);

#if WORKAROUND_ISSUE_14286
// The MX scale layout a Tensile kernel reads, from its name: the abbreviated
// MXScaleFormat value after "_MXSF" ("HPS" HostPreSwizzle = gfx950 32x8, "IMS"
// InMemorySwizzle = gfx1250, "NS" NoSwizzle), or "" when the name has no such
// token (rocRoller, or Tensile libraries that predate the parameter).
std::string mxKernelScaleLayout(const std::string& kernelName);

// The scale format a selected kernel reads, given whether it is a rocRoller kernel,
// its mxKernelScaleLayout token and the format the plan prepared; nullopt when they
// do not match and the plan must be declined. A gfx1250 NoSwizzle kernel reads
// natural scales. A Tensile name without the token (older libraries) is trusted for
// the swizzled formats, which is what those libraries shipped.
std::optional<MxScaleFormat>
    kernelScaleFormat(bool isRocRoller, const std::string& layoutToken, MxScaleFormat prepared);
#endif

// Scale format for the device behind the handle's stream: SWIZZLED_GFX1250 on
// gfx1250 (Tensile only, VEC32 + implicit swizzle), NATURAL elsewhere (rocRoller
// on gfx950). SWIZZLED_GFX950 is never selected here; createMxMatmulPlan()
// accepts it.
MxScaleFormat selectMxScaleFormat(const HipdnnEnginePluginHandle& handle);
MxScaleFormat mxScaleFormatForArch(std::string_view arch);

// One hipBLAS scale operand: which tensor feeds it and how it must be prepared.
struct MxScaleOperand
{
    int64_t uid = 0;
    int64_t mn = 0; // M or N extent of the scaled operand
    int64_t kBlocks = 0; // K / VEC32_BLOCK_SIZE
    MxOrder scaleOrder = MxOrder::K_CONTIGUOUS;
    MxOrder operandOrder = MxOrder::K_CONTIGUOUS;
};

class MxMatmulParams
{
public:
    MxMatmulParams(
        const hipdnn_flatbuffers_sdk::data_objects::BlockScaleDequantizeAttributes& deqAttrA,
        const hipdnn_flatbuffers_sdk::data_objects::BlockScaleDequantizeAttributes& deqAttrB,
        const hipdnn_flatbuffers_sdk::data_objects::MatmulAttributes& matmulAttr,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        MxScaleFormat scaleFormat = MxScaleFormat::NATURAL);

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

    // Scale operands in hipBLAS's frame (aScale() = our B's scale).
    const MxScaleOperand& aScale() const;
    const MxScaleOperand& bScale() const;

    MxScaleFormat scaleFormat() const;

private:
    HipblasltMatmulDesc _matmulDesc;
    HipblasltMatrixLayout _matrixLayoutA;
    HipblasltMatrixLayout _matrixLayoutB;
    HipblasltMatrixLayout _matrixLayoutC;
    MxScaleOperand _aScale;
    MxScaleOperand _bScale;
    MxScaleFormat _scaleFormat = MxScaleFormat::NATURAL;
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

    // Scale format the selected kernel reads (see createMxMatmulPlan()).
    MxScaleFormat scaleFormat() const;

    // Scale preparation for hipBLAS A_SCALE (our B's scale) and B_SCALE (our A's).
    const MxScalePrep& aScalePrep() const;
    const MxScalePrep& bScalePrep() const;

private:
    MxMatmulParams _params;
    hipblasLtMatmulHeuristicResult_t _heuristicResult;

    // hipBLASLt matmul workspace size (from the heuristic).
    size_t _workspaceSize = 0;

    // Scale format the selected kernel reads.
    MxScaleFormat _scaleFormat = MxScaleFormat::NATURAL;

    // Workspace layout: [ A_SCALE prep | B_SCALE prep | hipBLASLt matmul workspace ],
    // each region aligned. A pass-through scale reserves nothing.
    MxScalePrep _aScalePrep;
    MxScalePrep _bScalePrep;
    size_t _bScalePrepOffset = 0;
    size_t _matmulWorkspaceOffset = 0;

    static constexpr float ALPHA = 1.f;
    static constexpr float BETA = 0.f;
};

// Builds the MX plan for `scaleFormat`; the constructor runs
// hipblasLtMatmulAlgoGetHeuristic. When SWIZZLED_GFX950 has no usable Tensile
// kernel (no algorithm, or the kernel guard declines the one selected), the plan
// falls back to rocRoller (NATURAL). Other formats have no fallback and throw.
std::unique_ptr<MxMatmulPlan> createMxMatmulPlan(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_flatbuffers_sdk::data_objects::BlockScaleDequantizeAttributes& deqAttrA,
    const hipdnn_flatbuffers_sdk::data_objects::BlockScaleDequantizeAttributes& deqAttrB,
    const hipdnn_flatbuffers_sdk::data_objects::MatmulAttributes& matmulAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    MxScaleFormat scaleFormat);

} // namespace hipblaslt_plugin
