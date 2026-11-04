// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/DeviceQuery.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "HipblasltMxMatmulPlan.hpp"
#include "HipblasltUtils.hpp"
#include "HipdnnEnginePluginHandle.hpp"

namespace hipblaslt_plugin
{
namespace
{

// Infer the hipBLASLt transpose op from a tensor's strides: row-major
// (stride[-1]==1) → HIPBLAS_OP_N; column-major (stride[-2]==1) → HIPBLAS_OP_T.
// Mirrors MatmulParams::getTrans for the plain matmul plan.
hipblasOperation_t getTransFromStrides(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper& t)
{
    const auto& strides = t.strides();
    PLUGIN_THROW_IF_FALSE(strides.size() > 1,
                          HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                          "Unsupported stride for MX input matrix: " + t.name());
    if(strides[strides.size() - 1] == 1)
    {
        return HIPBLAS_OP_N;
    }
    if(strides[strides.size() - 2] == 1)
    {
        return HIPBLAS_OP_T;
    }
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Unsupported stride for MX input matrix: " + t.name());
}

#if WORKAROUND_ISSUE_14286
// hipBLASLt does not match the selected kernel's scale layout to the scale mode
// (#14286). Returns the scale format the selected kernel reads (see
// kernelScaleFormat), or declines the plan.
MxScaleFormat scaleFormatForKernel(const HipdnnEnginePluginHandle& handle,
                                   hipblasLtMatmulAlgo_t algo,
                                   MxScaleFormat format)
{
    const bool isRocRoller = hipblaslt_ext::getIndexFromAlgo(algo) < 0;
    const std::string kernelName
        = isRocRoller ? std::string("<rocRoller>")
                      : hipblaslt_ext::getKernelNameFromAlgo(handle.hipblasltHandle, algo);
    const auto kernelFormat
        = kernelScaleFormat(isRocRoller, mxKernelScaleLayout(kernelName), format);
    PLUGIN_THROW_IF_FALSE(kernelFormat.has_value(),
                          HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                          std::string("[#14286] MX matmul: hipBLASLt selected kernel '")
                              + kernelName + "', whose scale layout does not match the "
                              + toString(format)
                              + " scales this plan prepares (is HIPBLASLT_USE_ROCROLLER set?)");
    return kernelFormat.value();
}
#endif

MxScaleOperand makeScaleOperand(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper& operand,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper& scale,
    size_t kAxisFromEnd)
{
    const auto& dims = operand.dims();
    PLUGIN_THROW_IF_FALSE(dims.size() >= 2,
                          HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                          "MX matmul: operand rank must be >= 2: " + operand.name());
    const auto operandOrder = packedMxOrder(operand, kAxisFromEnd);
    const auto scaleOrder = packedMxOrder(scale, kAxisFromEnd);
    PLUGIN_THROW_IF_FALSE(operandOrder.has_value(),
                          HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                          "MX matmul: operand must be a packed row- or column-major matrix: "
                              + operand.name());
    PLUGIN_THROW_IF_FALSE(scaleOrder.has_value(),
                          HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                          "MX matmul: scale must be a packed row- or column-major matrix: "
                              + scale.name());

    const size_t kIdx = dims.size() - kAxisFromEnd;
    const size_t mnIdx = dims.size() - (kAxisFromEnd == 1 ? 2 : 1);

    MxScaleOperand result;
    result.uid = scale.uid();
    result.mn = dims[mnIdx];
    result.kBlocks = dims[kIdx] / VEC32_BLOCK_SIZE;
    result.scaleOrder = scaleOrder.value();
    result.operandOrder = operandOrder.value();
    return result;
}

} // namespace

#if WORKAROUND_ISSUE_14286
std::string mxKernelScaleLayout(const std::string& kernelName)
{
    constexpr std::string_view TOKEN = "_MXSF";
    const auto start = kernelName.find(TOKEN);
    if(start == std::string::npos)
    {
        return "";
    }
    const auto valueStart = start + TOKEN.size();
    const auto valueEnd = kernelName.find('_', valueStart);
    return kernelName.substr(valueStart,
                             valueEnd == std::string::npos ? valueEnd : valueEnd - valueStart);
}

std::optional<MxScaleFormat>
    kernelScaleFormat(bool isRocRoller, const std::string& layoutToken, MxScaleFormat prepared)
{
    switch(prepared)
    {
    case MxScaleFormat::NATURAL:
        if(isRocRoller || layoutToken == "NS")
        {
            return MxScaleFormat::NATURAL;
        }
        break;
    case MxScaleFormat::SWIZZLED_GFX950:
        if(!isRocRoller && (layoutToken.empty() || layoutToken == "HPS"))
        {
            return MxScaleFormat::SWIZZLED_GFX950;
        }
        break;
    case MxScaleFormat::SWIZZLED_GFX1250:
        if(!isRocRoller && layoutToken == "NS")
        {
            return MxScaleFormat::NATURAL;
        }
        if(!isRocRoller && (layoutToken.empty() || layoutToken == "IMS"))
        {
            return MxScaleFormat::SWIZZLED_GFX1250;
        }
        break;
    default:
        break;
    }
    return std::nullopt;
}
#endif

std::optional<MxOrder> packedMxOrder(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper& tensor,
    size_t kAxisFromEnd)
{
    const auto& dims = tensor.dims();
    const auto& strides = tensor.strides();
    if(dims.size() < 2 || strides.size() != dims.size() || (kAxisFromEnd != 1 && kAxisFromEnd != 2))
    {
        return std::nullopt;
    }
    const auto rank = dims.size();
    const int64_t outer = dims[rank - 2];
    const int64_t inner = dims[rank - 1];
    const bool rowMajor = strides[rank - 1] == 1 && (outer == 1 || strides[rank - 2] == inner);
    const bool colMajor = strides[rank - 2] == 1 && (inner == 1 || strides[rank - 1] == outer);

    // The innermost logical axis is K for kAxisFromEnd == 1 ([MN, K]) and MN for
    // kAxisFromEnd == 2 ([K, MN]); row-major means that axis is contiguous.
    const auto rowMajorOrder = kAxisFromEnd == 1 ? MxOrder::K_CONTIGUOUS : MxOrder::MN_CONTIGUOUS;
    const auto colMajorOrder = kAxisFromEnd == 1 ? MxOrder::MN_CONTIGUOUS : MxOrder::K_CONTIGUOUS;
    if(rowMajor)
    {
        return rowMajorOrder;
    }
    if(colMajor)
    {
        return colMajorOrder;
    }
    return std::nullopt;
}

MxScaleFormat mxScaleFormatForArch(std::string_view arch)
{
    using hipdnn_plugin_sdk::archMatches;
    using hipdnn_plugin_sdk::ArchMatchMode;
    if(archMatches(arch, "gfx1250", ArchMatchMode::PREFIX)
       || archMatches(arch, "gfx1250-strict", ArchMatchMode::PREFIX))
    {
        return MxScaleFormat::SWIZZLED_GFX1250;
    }
    return MxScaleFormat::NATURAL;
}

MxScaleFormat selectMxScaleFormat(const HipdnnEnginePluginHandle& handle)
{
    return mxScaleFormatForArch(hipdnn_plugin_sdk::getDeviceArch(handle.getStream()));
}

MxMatmulParams::MxMatmulParams(
    const hipdnn_flatbuffers_sdk::data_objects::BlockScaleDequantizeAttributes& deqAttrA,
    const hipdnn_flatbuffers_sdk::data_objects::BlockScaleDequantizeAttributes& deqAttrB,
    const hipdnn_flatbuffers_sdk::data_objects::MatmulAttributes& matmulAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    MxScaleFormat scaleFormat)
    : _scaleFormat(scaleFormat)
{
    // tXA/tXB are the graph's logical A/B (the block-scaled X inputs of dequant A/B).
    const auto tXA = hipblaslt_utils::findTensorAttributes(tensorMap, deqAttrA.x_tensor_uid());
    const auto tXB = hipblaslt_utils::findTensorAttributes(tensorMap, deqAttrB.x_tensor_uid());
    const auto tC = hipblaslt_utils::findTensorAttributes(tensorMap, matmulAttr.c_tensor_uid());
    const auto tScaleA
        = hipblaslt_utils::findTensorAttributes(tensorMap, deqAttrA.scale_tensor_uid());
    const auto tScaleB
        = hipblaslt_utils::findTensorAttributes(tensorMap, deqAttrB.scale_tensor_uid());

    // Row-major BLAS trick: hipDNN tensors are row-major but hipBLASLt is
    // column-major, so we compute C^T = B^T * A^T by presenting our B as
    // hipBLAS's A operand and our A as its B operand. The swap is applied once,
    // here, so every member is already in hipBLAS's frame and execute() needs no
    // further swapping: a()/aScale() are what hipBLAS receives as A/A_SCALE.
    _matrixLayoutA = HipblasltMatrixLayout(tXB); // hipBLAS A <- our B
    _matrixLayoutB = HipblasltMatrixLayout(tXA); // hipBLAS B <- our A
    _matrixLayoutC = HipblasltMatrixLayout(tC);

    // Our A is [..., M, K] (K innermost); our B is [..., K, N] (K second-to-last).
    _aScale = makeScaleOperand(tXB, tScaleB, 2); // hipBLAS A_SCALE <- our B's scale
    _bScale = makeScaleOperand(tXA, tScaleA, 1); // hipBLAS B_SCALE <- our A's scale

    // MX GEMM always uses HIPBLAS_COMPUTE_32F. desc transA/transB are in
    // hipBLAS's frame too: transA = getTrans(hipBLAS A = our B), transB = our A.
    // Both K-contiguous operands give hipBLAS TN, the only layout Tensile has MX
    // kernels for; rocRoller (gfx950) accepts every combination.
    _matmulDesc = HipblasltMatmulDesc(
        getTransFromStrides(tXB), getTransFromStrides(tXA), HIPBLAS_COMPUTE_32F, HIP_R_32F);

    _matmulDesc.setAScaleMode(scaleModeFor(scaleFormat));
    _matmulDesc.setBScaleMode(scaleModeFor(scaleFormat));
}

const HipblasltMatrixLayout& MxMatmulParams::a() const
{
    return _matrixLayoutA;
}

const HipblasltMatrixLayout& MxMatmulParams::b() const
{
    return _matrixLayoutB;
}

const HipblasltMatrixLayout& MxMatmulParams::c() const
{
    return _matrixLayoutC;
}

const HipblasltMatmulDesc& MxMatmulParams::desc() const
{
    return _matmulDesc;
}

int64_t MxMatmulParams::aScaleUid() const
{
    return _aScale.uid;
}

int64_t MxMatmulParams::bScaleUid() const
{
    return _bScale.uid;
}

const MxScaleOperand& MxMatmulParams::aScale() const
{
    return _aScale;
}

const MxScaleOperand& MxMatmulParams::bScale() const
{
    return _bScale;
}

MxScaleFormat MxMatmulParams::scaleFormat() const
{
    return _scaleFormat;
}

MxMatmulPlan::MxMatmulPlan(const HipdnnEnginePluginHandle& handle, MxMatmulParams&& params)
    : _params(std::move(params))
{
    // Same max workspace approach as MatmulPlan: 128 MB to allow hipBLASLt to
    // find the most performant MX GEMM algorithm.
    auto maxWorkspaceSize = static_cast<size_t>(128 * 1024 * 1024); // 128 MB
    hipblasLtMatmulPreference_t prefHandle;
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulPreferenceCreate(&prefHandle));
    // Own the preference via RAII.
    hipdnn_data_sdk::utilities::ScopedResource<hipblasLtMatmulPreference_t> const pref(
        prefHandle, [](hipblasLtMatmulPreference_t p) {
            LOG_ON_HIPBLASLT_FAILURE(hipblasLtMatmulPreferenceDestroy(p));
        });

    THROW_ON_HIPBLASLT_FAILURE(
        hipblasLtMatmulPreferenceSetAttribute(pref.get(),
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &maxWorkspaceSize,
                                              sizeof(maxWorkspaceSize)));

    // Operands are already in hipBLAS's frame (swapped in MxMatmulParams), so
    // a()/b() are passed straight through as hipBLAS A/B.
    constexpr int REQUEST_SOLUTIONS = 1;
    std::array<hipblasLtMatmulHeuristicResult_t, REQUEST_SOLUTIONS> heuristicResult{};
    int returnedAlgoCount = 0;
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulAlgoGetHeuristic(handle.hipblasltHandle,
                                                               _params.desc().matmulDesc(),
                                                               _params.a().matrixLayout(),
                                                               _params.b().matrixLayout(),
                                                               _params.c().matrixLayout(),
                                                               _params.c().matrixLayout(),
                                                               pref.get(),
                                                               REQUEST_SOLUTIONS,
                                                               heuristicResult.data(),
                                                               &returnedAlgoCount));

    PLUGIN_THROW_IF_FALSE(returnedAlgoCount > 0,
                          HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                          "hipBLASLt has not found MX GEMM algorithm!");

    _heuristicResult = heuristicResult[0];
    _workspaceSize = _heuristicResult.workspaceSize;

#if WORKAROUND_ISSUE_14286
    _scaleFormat = scaleFormatForKernel(handle, _heuristicResult.algo, _params.scaleFormat());
#else
    _scaleFormat = _params.scaleFormat();
#endif
    const auto scaleFormat = _scaleFormat;

    // Each scale is reshaped on-device at execute time into a region at the front
    // of the workspace, so the plan owns no device memory itself. A swizzled format
    // compiles the relayout kernel here, on the first plan for this device.
    _aScalePrep = MxScalePrep(_params.aScale().scaleOrder,
                              _params.aScale().operandOrder,
                              scaleFormat,
                              _params.aScale().mn,
                              _params.aScale().kBlocks);
    _bScalePrep = MxScalePrep(_params.bScale().scaleOrder,
                              _params.bScale().operandOrder,
                              scaleFormat,
                              _params.bScale().mn,
                              _params.bScale().kBlocks);
    _bScalePrepOffset = hipblaslt_utils::alignUp(_aScalePrep.workspaceSize(),
                                                 hipblaslt_utils::WORKSPACE_ALIGNMENT);
    _matmulWorkspaceOffset = _bScalePrepOffset
                             + hipblaslt_utils::alignUp(_bScalePrep.workspaceSize(),
                                                        hipblaslt_utils::WORKSPACE_ALIGNMENT);

    HIPDNN_PLUGIN_LOG_INFO("MX matmul plan: scale format "
                           << toString(scaleFormat) << ", A_SCALE prep [" << _aScalePrep.describe()
                           << "], B_SCALE prep [" << _bScalePrep.describe() << "]");
}

size_t MxMatmulPlan::getWorkspaceSize([[maybe_unused]] const HipdnnEnginePluginHandle& handle) const
{
    return _matmulWorkspaceOffset + _workspaceSize;
}

MxScaleFormat MxMatmulPlan::scaleFormat() const
{
    return _scaleFormat;
}

const MxScalePrep& MxMatmulPlan::aScalePrep() const
{
    return _aScalePrep;
}

const MxScalePrep& MxMatmulPlan::bScalePrep() const
{
    return _bScalePrep;
}

void MxMatmulPlan::execute(const HipdnnEnginePluginHandle& handle,
                           const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                           uint32_t numDeviceBuffers,
                           void* workspace) const
{
    // A non-zero workspace size is a caller contract: validate before any
    // dereference below. (A pass-through scale layout with a zero-size hipBLASLt
    // workspace legitimately needs none.)
    PLUGIN_THROW_IF_TRUE(workspace == nullptr && getWorkspaceSize(handle) != 0,
                         HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                         "MxMatmulPlan::execute: workspace is null but MX GEMM requires "
                             + std::to_string(getWorkspaceSize(handle))
                             + " bytes (see getWorkspaceSize())");

    auto aBuffer
        = hipblaslt_utils::findDeviceBuffer(_params.a().uid(), deviceBuffers, numDeviceBuffers);
    auto bBuffer
        = hipblaslt_utils::findDeviceBuffer(_params.b().uid(), deviceBuffers, numDeviceBuffers);
    auto cBuffer
        = hipblaslt_utils::findDeviceBuffer(_params.c().uid(), deviceBuffers, numDeviceBuffers);
    auto aScaleBuffer
        = hipblaslt_utils::findDeviceBuffer(_params.aScaleUid(), deviceBuffers, numDeviceBuffers);
    auto bScaleBuffer
        = hipblaslt_utils::findDeviceBuffer(_params.bScaleUid(), deviceBuffers, numDeviceBuffers);

    // Workspace layout: [ A_SCALE prep | B_SCALE prep | hipBLASLt matmul workspace ].
    auto* base = static_cast<char*>(workspace);
    void* matmulWorkspace = base == nullptr ? nullptr : base + _matmulWorkspaceOffset;

    // Scale pointers carry no layout handle, so a scale whose layout differs from
    // what hipBLASLt reads for this format (transposed or swizzled) is rewritten on
    // device first; a matching one is passed through.
    const auto [aScale, bScale]
        = MxScalePrep::runPair(_aScalePrep,
                               aScaleBuffer.ptr,
                               base,
                               _bScalePrep,
                               bScaleBuffer.ptr,
                               base == nullptr ? nullptr : base + _bScalePrepOffset,
                               handle.hipblasltHandle,
                               handle.getStream());

    // Clone the plan's descriptor so this call owns its own copy: scale pointers
    // are per-call device addresses, and mutating a shared descriptor would make
    // concurrent execute() calls on one plan instance race. Already in hipBLAS's
    // frame (swapped in MxMatmulParams).
    HipblasltMatmulDesc desc = _params.desc().clone();
    desc.setAScalePointer(aScale);
    desc.setBScalePointer(bScale);

    // Operands are already in hipBLAS's frame (swapped in MxMatmulParams), so
    // pass them straight through.
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmul(handle.hipblasltHandle,
                                               desc.matmulDesc(),
                                               &ALPHA,
                                               aBuffer.ptr,
                                               _params.a().matrixLayout(),
                                               bBuffer.ptr,
                                               _params.b().matrixLayout(),
                                               &BETA,
                                               cBuffer.ptr,
                                               _params.c().matrixLayout(),
                                               cBuffer.ptr,
                                               _params.c().matrixLayout(),
                                               &_heuristicResult.algo,
                                               matmulWorkspace,
                                               _workspaceSize,
                                               handle.getStream()));
}

std::unique_ptr<MxMatmulPlan> createMxMatmulPlan(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_flatbuffers_sdk::data_objects::BlockScaleDequantizeAttributes& deqAttrA,
    const hipdnn_flatbuffers_sdk::data_objects::BlockScaleDequantizeAttributes& deqAttrB,
    const hipdnn_flatbuffers_sdk::data_objects::MatmulAttributes& matmulAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    MxScaleFormat scaleFormat)
{
    try
    {
        return std::make_unique<MxMatmulPlan>(
            handle, MxMatmulParams(deqAttrA, deqAttrB, matmulAttr, tensorMap, scaleFormat));
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        if(scaleFormat != MxScaleFormat::SWIZZLED_GFX950)
        {
            throw;
        }
        HIPDNN_PLUGIN_LOG_INFO("MX matmul: no usable Tensile kernel (" << e.what()
                                                                       << "); using rocRoller");
        return std::make_unique<MxMatmulPlan>(
            handle,
            MxMatmulParams(deqAttrA, deqAttrB, matmulAttr, tensorMap, MxScaleFormat::NATURAL));
    }
}

} // namespace hipblaslt_plugin
