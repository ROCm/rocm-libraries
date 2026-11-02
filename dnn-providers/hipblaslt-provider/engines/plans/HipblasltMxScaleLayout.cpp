// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>

#include <hip/hip_runtime.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "HipblasltMxScaleLayout.hpp"
#include "HipblasltUtils.hpp"
#include "compilation/ProgramCache.hpp"

namespace hipblaslt_plugin
{
namespace
{

constexpr float ALPHA = 1.f;
constexpr float BETA = 0.f;

int64_t roundUp(int64_t value, int64_t multiple)
{
    return ((value + multiple - 1) / multiple) * multiple;
}

uint32_t ceilDiv(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1) / divisor;
}

bool isWordAligned(const void* ptr)
{
    return reinterpret_cast<uintptr_t>(ptr) % 4 == 0;
}

// The relayout kernel's entry point for a swizzled format, compiled for the current
// device on first use.
compilation::Kernel relayoutKernel(MxScaleFormat format)
{
    const auto program = compilation::getProgram(mx_scale_relayout::SOURCE_NAME, {"-std=c++17"});
    const std::string name = format == MxScaleFormat::SWIZZLED_GFX950
                                 ? mx_scale_relayout::GFX950_KERNEL
                                 : mx_scale_relayout::GFX1250_KERNEL;
    compilation::Kernel kernel(program->getKernel(name), name, program);
    kernel.setBlockSize(mx_scale_relayout::THREADS);
    return kernel;
}

} // namespace

const char* toString(MxOrder order)
{
    switch(order)
    {
    case MxOrder::K_CONTIGUOUS:
        return "K_CONTIGUOUS";
    case MxOrder::MN_CONTIGUOUS:
        return "MN_CONTIGUOUS";
    default:
        return "<unknown MxOrder>";
    }
}

const char* toString(MxScaleFormat format)
{
    switch(format)
    {
    case MxScaleFormat::NATURAL:
        return "NATURAL";
    case MxScaleFormat::SWIZZLED_GFX950:
        return "SWIZZLED_GFX950";
    case MxScaleFormat::SWIZZLED_GFX1250:
        return "SWIZZLED_GFX1250";
    default:
        return "<unknown MxScaleFormat>";
    }
}

hipblasLtMatmulMatrixScale_t scaleModeFor(MxScaleFormat format)
{
    return format == MxScaleFormat::SWIZZLED_GFX950
               ? HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT
               : HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
}

size_t mxScaleBytes(MxScaleFormat format, int64_t mn, int64_t kBlocks)
{
    switch(format)
    {
    case MxScaleFormat::SWIZZLED_GFX950:
        return static_cast<size_t>(roundUp(mn, mx_scale_relayout::GFX950_TILE_MN)
                                   * roundUp(kBlocks, mx_scale_relayout::GFX950_TILE_KB));
    case MxScaleFormat::SWIZZLED_GFX1250:
        return static_cast<size_t>(mn * roundUp(kBlocks, mx_scale_relayout::GFX1250_GROUP_KB));
    case MxScaleFormat::NATURAL:
    default:
        return static_cast<size_t>(mn * kBlocks);
    }
}

MxScalePrep::MxScalePrep(
    MxOrder sourceOrder, MxOrder operandOrder, MxScaleFormat format, int64_t mn, int64_t kBlocks)
    : _format(format)
    , _sourceOrder(sourceOrder)
    , _mn(mn)
    , _kBlocks(kBlocks)
{
    PLUGIN_THROW_IF_TRUE(mn <= 0 || kBlocks <= 0,
                         HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                         "MX scale prep: mn and kBlocks must be positive");
    PLUGIN_THROW_IF_TRUE(format != MxScaleFormat::NATURAL && operandOrder != MxOrder::K_CONTIGUOUS,
                         HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                         std::string("MX scale prep: the ") + toString(format)
                             + " scale format needs a K-contiguous operand");

    switch(format)
    {
    case MxScaleFormat::NATURAL:
    {
        // rocRoller reads each scale in its operand's order and never transposes it;
        // a vector scale reads the same in either order.
        if(sourceOrder == operandOrder || mn == 1 || kBlocks == 1)
        {
            return;
        }
        // hipBLASLt layouts are column-major: the row-major [rows][cols] source is
        // the column-major (cols x rows) matrix, and OP_T writes the column-major
        // (rows x cols) matrix, i.e. the row-major [cols][rows] buffer. HIP_R_8I
        // round-trips every byte exactly through the transform's float math
        // (alpha = 1, beta = 0).
        const bool kContiguous = sourceOrder == MxOrder::K_CONTIGUOUS;
        const auto rows = static_cast<uint64_t>(kContiguous ? mn : kBlocks);
        const auto cols = static_cast<uint64_t>(kContiguous ? kBlocks : mn);
        _method = Method::TRANSPOSE;
        _transformDesc = HipblasltMatrixTransformDesc(HIP_R_32F, HIPBLAS_OP_T);
        _sourceLayout = HipblasltMatrixLayout(HIP_R_8I, cols, rows, static_cast<int64_t>(cols));
        _preparedLayout = HipblasltMatrixLayout(HIP_R_8I, rows, cols, static_cast<int64_t>(rows));
        break;
    }
    case MxScaleFormat::SWIZZLED_GFX950:
    case MxScaleFormat::SWIZZLED_GFX1250:
    {
        _method = Method::RELAYOUT_KERNEL;
        const bool gfx950 = format == MxScaleFormat::SWIZZLED_GFX950;
        _outMn
            = static_cast<uint32_t>(gfx950 ? roundUp(mn, mx_scale_relayout::GFX950_TILE_MN) : mn);
        _outKb = static_cast<uint32_t>(roundUp(kBlocks,
                                               gfx950 ? mx_scale_relayout::GFX950_TILE_KB
                                                      : mx_scale_relayout::GFX1250_GROUP_KB));
        _tilesK = ceilDiv(_outKb, mx_scale_relayout::TILE);
        _blocks = ceilDiv(_outMn, mx_scale_relayout::TILE) * _tilesK;
        _kernel = relayoutKernel(format);
        _kernel->setGridSize(_blocks);
        break;
    }
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "MX scale prep: unknown scale format");
    }
    _workspaceBytes = hipblaslt_utils::alignUp(mxScaleBytes(format, mn, kBlocks),
                                               hipblaslt_utils::WORKSPACE_ALIGNMENT);
}

size_t MxScalePrep::workspaceSize() const
{
    return _workspaceBytes;
}

MxScalePrep::Method MxScalePrep::method() const
{
    return _method;
}

bool MxScalePrep::isPassThrough() const
{
    return _method == Method::PASS_THROUGH;
}

std::string MxScalePrep::describe() const
{
    switch(_method)
    {
    case Method::TRANSPOSE:
        return "transpose";
    case Method::RELAYOUT_KERNEL:
        return "relayout kernel";
    case Method::PASS_THROUGH:
    default:
        return "pass-through";
    }
}

mx_scale_relayout::Params MxScalePrep::relayoutParams(const void* source, void* workspace) const
{
    const bool kContiguous = _sourceOrder == MxOrder::K_CONTIGUOUS;
    const auto mn = static_cast<uint32_t>(_mn);
    const auto kBlocks = static_cast<uint32_t>(_kBlocks);
    return mx_scale_relayout::Params{static_cast<const uint8_t*>(source),
                                     static_cast<uint8_t*>(workspace),
                                     mn,
                                     kBlocks,
                                     _outMn,
                                     _outKb,
                                     kContiguous,
                                     isWordAligned(source) && (kContiguous ? kBlocks : mn) % 4 == 0,
                                     isWordAligned(workspace)};
}

std::pair<const void*, const void*> MxScalePrep::runPair(const MxScalePrep& a,
                                                         const void* sourceA,
                                                         void* workspaceA,
                                                         const MxScalePrep& b,
                                                         const void* sourceB,
                                                         void* workspaceB,
                                                         hipblasLtHandle_t handle,
                                                         hipStream_t stream)
{
    if(a._method == Method::RELAYOUT_KERNEL && b._method == Method::RELAYOUT_KERNEL
       && a._format == b._format && workspaceA != nullptr && workspaceB != nullptr)
    {
        mx_scale_relayout::Batch batch{};
        batch.first = a.relayoutParams(sourceA, workspaceA);
        batch.second = b.relayoutParams(sourceB, workspaceB);
        batch.tilesKFirst = a._tilesK;
        batch.tilesKSecond = b._tilesK;
        batch.firstBlockOfSecond = a._blocks;
        // A copy with the pair's grid: a plan's preparations are shared by concurrent
        // execute() calls.
        auto kernel = *a._kernel;
        kernel.setGridSize(a._blocks + b._blocks);
        kernel.launch(stream, batch);
        return {workspaceA, workspaceB};
    }
    return {a.run(handle, stream, sourceA, workspaceA), b.run(handle, stream, sourceB, workspaceB)};
}

const void* MxScalePrep::run(hipblasLtHandle_t handle,
                             hipStream_t stream,
                             const void* source,
                             void* workspace) const
{
    if(_method == Method::PASS_THROUGH)
    {
        return source;
    }
    PLUGIN_THROW_IF_TRUE(workspace == nullptr,
                         HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                         "MX scale prep: workspace is null but the scale needs "
                             + std::to_string(_workspaceBytes) + " bytes");

    if(_method == Method::RELAYOUT_KERNEL)
    {
        mx_scale_relayout::Batch batch{};
        batch.first = relayoutParams(source, workspace);
        batch.second = batch.first;
        batch.tilesKFirst = _tilesK;
        batch.tilesKSecond = _tilesK;
        batch.firstBlockOfSecond = _blocks;
        _kernel->launch(stream, batch);
        return workspace;
    }

    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatrixTransform(handle,
                                                        _transformDesc.transformDesc(),
                                                        &ALPHA,
                                                        source,
                                                        _sourceLayout.matrixLayout(),
                                                        &BETA,
                                                        nullptr,
                                                        nullptr,
                                                        workspace,
                                                        _preparedLayout.matrixLayout(),
                                                        stream));
    return workspace;
}

} // namespace hipblaslt_plugin
