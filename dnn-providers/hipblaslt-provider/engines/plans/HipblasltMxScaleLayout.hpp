// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include <hipblaslt/hipblaslt.h>

#include "HipblasltMatrixLayout.hpp"
#include "HipblasltMatrixTransformDesc.hpp"
#include "compilation/Kernel.hpp"
#include "engines/kernels/MxScaleRelayoutParams.hpp"

namespace hipblaslt_plugin
{

// Physical order of an MX operand or of its block-scale tensor. An operand is
// K_CONTIGUOUS when its K axis has stride 1 (hipBLASLt OP_T for hipBLAS A, OP_N
// for hipBLAS B) and MN_CONTIGUOUS when its M (or N) axis has stride 1. A scale
// tensor uses the same words for its K-block axis: K_CONTIGUOUS is the natural
// [MN][K/32] layout, MN_CONTIGUOUS is [K/32][MN].
enum class MxOrder
{
    K_CONTIGUOUS,
    MN_CONTIGUOUS,
};

// How the plan hands block scales to hipBLASLt. selectMxScaleFormat picks NATURAL or
// SWIZZLED_GFX1250 by device arch; SWIZZLED_GFX950 is used only when a caller asks
// createMxMatmulPlan for it.
enum class MxScaleFormat
{
    // VEC32_UE8M0 with unswizzled scales whose order follows the operand order
    // (rocRoller on gfx950 accepts NN/NT/TN/TT this way).
    NATURAL,
    // BLK32_UE8M0_32_8_EXT with the gfx950 32x8 pre-swizzle
    // (mxDataGenerator preSwizzleScalesGFX950). Tensile on gfx950; TN only.
    SWIZZLED_GFX950,
    // VEC32_UE8M0 with the implicit gfx1250 swizzle
    // (mxDataGenerator preSwizzleScalesGFX1250). Tensile on gfx1250; TN only.
    SWIZZLED_GFX1250,
};

const char* toString(MxOrder order);
const char* toString(MxScaleFormat format);

// The hipBLASLt scale mode that matches a scale format.
hipblasLtMatmulMatrixScale_t scaleModeFor(MxScaleFormat format);

// Number of bytes hipBLASLt reads for one scale operand in the given format:
// mn * kBlocks for NATURAL, padded to (32, 8) for SWIZZLED_GFX950 and to a
// multiple of 4 K blocks for SWIZZLED_GFX1250.
size_t mxScaleBytes(MxScaleFormat format, int64_t mn, int64_t kBlocks);

// Turns one user block-scale tensor (UE8M0 bytes, one per 32-element K block,
// mn x kBlocks, densely packed in either order) into the layout hipBLASLt reads for a
// scale format:
//  - NATURAL: passed through when it already has its operand's order (or is a vector),
//    otherwise transposed by one hipblasLtMatrixTransform.
//  - SWIZZLED_GFX950, SWIZZLED_GFX1250: rewritten by the provider's relayout kernel
//    (engines/kernels/MxScaleRelayout.cpp), which the constructor compiles for the
//    current device on first use. hipBLASLt has no function for these layouts.
// A prepared scale is written to the front of the caller's workspace region.
class MxScalePrep
{
public:
    enum class Method
    {
        PASS_THROUGH,
        TRANSPOSE,
        RELAYOUT_KERNEL,
    };

    MxScalePrep() = default;
    MxScalePrep(MxOrder sourceOrder,
                MxOrder operandOrder,
                MxScaleFormat format,
                int64_t mn,
                int64_t kBlocks);

    MxScalePrep(const MxScalePrep&) = delete;
    MxScalePrep& operator=(const MxScalePrep&) = delete;
    MxScalePrep(MxScalePrep&&) = default;
    MxScalePrep& operator=(MxScalePrep&&) = default;
    ~MxScalePrep() = default;

    // Bytes of workspace run() needs (0 for a pass-through).
    size_t workspaceSize() const;

    Method method() const;
    bool isPassThrough() const;

    // "pass-through", "transpose" or "relayout kernel", for logging and tests.
    std::string describe() const;

    // Enqueues the preparation on `stream` and returns the pointer to hand to hipBLASLt:
    // `source` for a pass-through, otherwise the front of `workspace`.
    const void* run(hipblasLtHandle_t handle,
                    hipStream_t stream,
                    const void* source,
                    void* workspace) const;

    // run() for the A and B scales of one GEMM. When both use the relayout kernel for
    // the same format, they share one launch.
    static std::pair<const void*, const void*> runPair(const MxScalePrep& a,
                                                       const void* sourceA,
                                                       void* workspaceA,
                                                       const MxScalePrep& b,
                                                       const void* sourceB,
                                                       void* workspaceB,
                                                       hipblasLtHandle_t handle,
                                                       hipStream_t stream);

private:
    mx_scale_relayout::Params relayoutParams(const void* source, void* workspace) const;

    Method _method = Method::PASS_THROUGH;
    MxScaleFormat _format = MxScaleFormat::NATURAL;
    MxOrder _sourceOrder = MxOrder::K_CONTIGUOUS;
    int64_t _mn = 0;
    int64_t _kBlocks = 0;
    size_t _workspaceBytes = 0;

    // TRANSPOSE: a HIP_R_8I hipblasLtMatrixTransform.
    HipblasltMatrixTransformDesc _transformDesc;
    HipblasltMatrixLayout _sourceLayout;
    HipblasltMatrixLayout _preparedLayout;

    // RELAYOUT_KERNEL: the kernel with this scale's grid, and the extents of the
    // swizzled (padded) layout.
    std::optional<compilation::Kernel> _kernel;
    uint32_t _outMn = 0;
    uint32_t _outKb = 0;
    uint32_t _tilesK = 0;
    uint32_t _blocks = 0;
};

} // namespace hipblaslt_plugin
