/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// CK-related solver utilities that do NOT depend on Composable Kernel headers.
//
// This header provides workspace calculation, layout transpose orchestration,
// and perf-config iteration helpers used by CK-based solvers on the MIOpen
// side of the dlopen boundary. It must NEVER include any CK headers directly
// or transitively.

#pragma once

#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/batched_transpose_sol.hpp>
#include <miopen/buffer_info.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/miopen_internal.h>
#include <miopen/fusion/fusion_invoke_params.hpp>
#include <miopen/solver/implicitgemm_util.hpp>

#include <algorithm>
#include <optional>
#include <string>
#include <string_view>

namespace miopen {

namespace conv {
struct ProblemDescription;
} // namespace conv

namespace solver {

static constexpr int CkSplitkAutoDeduce = -1;

template <int L, int H>
inline static bool NextCKSplitkValue(int& v)
{
    assert((IsTwoPower<L, H>(v) || v == CkSplitkAutoDeduce));
    if(v == H)
    {
        v = CkSplitkAutoDeduce;
        return false;
    }
    if(v == CkSplitkAutoDeduce)
    {
        v = L;
        return true;
    }

    v *= 2;
    return false;
}

inline bool IsLinear(int L, int H, const int v)
{
    assert(L <= H);
    return L <= v && v <= H;
}

inline bool NextLinear(int L, int H, int& v)
{
    assert((IsLinear(L, H, v)));
    if(H == v)
    {
        v = L;
        return true;
    }
    ++v;
    return false;
}

struct ConvSolution;

struct CKWorkspaceBufferDescriptor
{
    size_t ck_size;
    size_t ck_offset;

    CKWorkspaceBufferDescriptor(size_t _ck_size, size_t _ck_offset)
        : ck_size(_ck_size), ck_offset(_ck_offset)
    {
    }
};

// Lightweight type tags for AI heuristics template dispatch.
// Replace CK types (ck::half_t, ck::bhalf_t, ck::tf32_t) that were previously
// used as template arguments to RunParameterPredictionModel. The DataType
// template parameter is unused in the function body -- these tags only drive
// the mode_use_tf32 constexpr check.
struct HalfTag
{
};
struct BFloat16Tag
{
};
struct TF32Tag
{
};

namespace internal {

enum class ConvOperandTag : int
{
    Input = 0,
    Weights,
    Output
};

enum class TranposeKind : int
{
    NHWC_TO_NCHW = 0,
    NCHW_TO_NHWC
};

template <int ND, TranposeKind TPOSE_KIND, ConvOperandTag CONV_OP>
struct TransposeOperand
{
    static_assert(ND == 2 || ND == 3, "Num Dimensions must be 2 or 3");
    constexpr static int NDIM                    = ND;
    constexpr static ConvOperandTag CONV_OP_TAG  = CONV_OP;
    constexpr static TranposeKind TRANSPOSE_KIND = TPOSE_KIND;

    using SolverType =
        std::conditional_t<TPOSE_KIND == TranposeKind::NHWC_TO_NCHW,
                           // NHWC_TO_NCHW
                           std::conditional_t<ND == 2,
                                              miopen::TransposeSolutionNhwc2Default,
                                              miopen::TransposeSolutionNdhwc2Default>,
                           // NCHW_TO_NHWC
                           std::conditional_t<ND == 2,
                                              miopen::TransposeSolutionDefault2Nhwc,
                                              miopen::TransposeSolutionDefault2Ndhwc>>;

    template <typename CKArgsType>
    SolverType MakeTransposeSolver(const miopen::ExecutionContext& ctx,
                                   const miopen::conv::ProblemDescription& problem,
                                   const CKArgsType& ck_args) const
    {

        if constexpr(CONV_OP_TAG == ConvOperandTag::Input)
        {
            if constexpr(ND == 3)
            {

                return SolverType{ctx,
                                  problem.GetInDataType(),
                                  static_cast<uint32_t>(ck_args.N),
                                  static_cast<uint32_t>(ck_args.C1),
                                  static_cast<uint32_t>(ck_args.Di),
                                  static_cast<uint32_t>(ck_args.Hi),
                                  static_cast<uint32_t>(ck_args.Wi)};
            }
            else
            {
                return SolverType{ctx,
                                  problem.GetInDataType(),
                                  static_cast<uint32_t>(ck_args.N),
                                  static_cast<uint32_t>(ck_args.C1),
                                  static_cast<uint32_t>(ck_args.Hi),
                                  static_cast<uint32_t>(ck_args.Wi)};
            }
        }
        else if constexpr(CONV_OP_TAG == ConvOperandTag::Weights)
        {
            if constexpr(ND == 3)
            {
                return SolverType{ctx,
                                  problem.GetWeightsDataType(),
                                  static_cast<uint32_t>(ck_args.K1),
                                  static_cast<uint32_t>(ck_args.C),
                                  static_cast<uint32_t>(ck_args.Z),
                                  static_cast<uint32_t>(ck_args.Y),
                                  static_cast<uint32_t>(ck_args.X)};
            }
            else
            {
                return SolverType{ctx,
                                  problem.GetWeightsDataType(),
                                  static_cast<uint32_t>(ck_args.K1),
                                  static_cast<uint32_t>(ck_args.C),
                                  static_cast<uint32_t>(ck_args.Y),
                                  static_cast<uint32_t>(ck_args.X)};
            }
        }
        else
        {
            static_assert(CONV_OP_TAG == ConvOperandTag::Output);
            if constexpr(ND == 3)
            {
                return SolverType{ctx,
                                  problem.GetOutDataType(),
                                  static_cast<uint32_t>(ck_args.N),
                                  static_cast<uint32_t>(ck_args.K1),
                                  static_cast<uint32_t>(ck_args.Do),
                                  static_cast<uint32_t>(ck_args.Ho),
                                  static_cast<uint32_t>(ck_args.Wo)};
            }
            else
            {
                return SolverType{ctx,
                                  problem.GetOutDataType(),
                                  static_cast<uint32_t>(ck_args.N),
                                  static_cast<uint32_t>(ck_args.K1),
                                  static_cast<uint32_t>(ck_args.Ho),
                                  static_cast<uint32_t>(ck_args.Wo)};
            }
        }
    }
};

// Shorthand aliases for CK assuming CK always expects and generates NHWC/NDHWC layouts
template <int ND, ConvOperandTag CONV_OP>
using CKTransposeInputOp = TransposeOperand<ND, TranposeKind::NCHW_TO_NHWC, CONV_OP>;

template <int ND, ConvOperandTag CONV_OP>
using CKTransposeOutputOp = TransposeOperand<ND, TranposeKind::NHWC_TO_NCHW, CONV_OP>;

class TransposeInstance
{
    size_t tensor_sz = 0;
    std::vector<OpKernelArg> kern_args{};
    size_t kern_idx   = std::numeric_limits<size_t>::max();
    size_t buf_offset = 0;
    shared<Data_t> buf_handle{};
    bool skippable = false;

public:
    template <typename TransSolnType>
    TransposeInstance(const TransSolnType& trans_sol,
                      size_t k_idx,
                      const MultiBufferWorkspaceTraits& wt,
                      size_t wspace_index)
        : tensor_sz(trans_sol.GetOutputTensorSize()),
          kern_args(trans_sol.GetKernelArg()),
          kern_idx(k_idx),
          buf_offset(wt.GetOffset(wspace_index)),
          skippable(trans_sol.IsSkippable())
    {
    }

    bool IsSkippable() const noexcept { return skippable; }

    void AssignBuffer(const Handle& handle, Data_t workSpace)
    {
        buf_handle = handle.CreateSubBuffer(workSpace, buf_offset, tensor_sz);
        assert(buf_handle.get());
    }

    Data_t GetBufferPtr() const { return buf_handle.get(); }

    void ConvertFrom(const Handle& handle, const std::vector<Kernel>& kernels, ConstData_t in_ptr)
    {
        Run(handle, kernels, buf_handle.get(), in_ptr);
    }

    void ConvertTo(const Handle& handle, const std::vector<Kernel>& kernels, Data_t out_ptr)
    {
        Run(handle, kernels, out_ptr, buf_handle.get());
    }

    void ZeroOutBuffer(const Handle& handle)
    {
        HipEventProfiler pfr(handle);

        [[maybe_unused]] auto status =
            hipMemsetAsync(buf_handle.get(), 0, tensor_sz, handle.GetStream());
        assert(status == hipSuccess);
    }

    TransposeInstance()                         = delete;
    TransposeInstance(const TransposeInstance&) = default;
    TransposeInstance(TransposeInstance&&)      = default;
    ~TransposeInstance()                        = default;

private:
    void Run(const Handle& handle,
             const std::vector<Kernel>& kernels,
             Data_t out_ptr,
             ConstData_t in_ptr)
    {
        assert(out_ptr);
        assert(in_ptr);
        assert(kernels.size() > kern_idx);

        kern_args[0] = out_ptr;
        kern_args[1] = in_ptr;

        auto save = handle.IsProfilingEnabled() ? handle.GetKernelTime() : 0.0f;
        handle.Run(kernels[kern_idx])(kern_args);
        if(handle.IsProfilingEnabled())
        {
            handle.AccumKernelTime(save);
        }
    }
};

class TransposeInstanceTagged : public TransposeInstance
{

    ConvOperandTag conv_op_tag_;

public:
    template <typename TransSolnType>
    TransposeInstanceTagged(const TransSolnType& sol,
                            size_t k_idx,
                            const MultiBufferWorkspaceTraits& wt,
                            size_t wspace_index,
                            ConvOperandTag conv_op_tag)
        : TransposeInstance(sol, k_idx, wt, wspace_index), conv_op_tag_(conv_op_tag)
    {
    }

    ConvOperandTag GetConvOperandTag() const noexcept { return conv_op_tag_; }

    std::underlying_type_t<ConvOperandTag> GetConvOperandTagAsInt() const noexcept
    {
        using IntType = std::underlying_type_t<ConvOperandTag>;
        return static_cast<IntType>(GetConvOperandTag());
    }
    Data_t GetTensorPtr(const ConvTensors& tensors) const { return pickTensorPtr(tensors); }

    const TensorDescriptor& GetTensorDesc(const ConvTensors& tensors) const
    {
        switch(conv_op_tag_)
        {
        case ConvOperandTag::Input: return tensors.xDesc;
        case ConvOperandTag::Weights: return tensors.wDesc;
        case ConvOperandTag::Output: return tensors.yDesc;
        }
        MIOPEN_THROW(miopenStatusInternalError, "Unknown CK convolution operand tag");
    }

    void ConvertFrom(const Handle& handle,
                     const std::vector<Kernel>& kernels,
                     const ConvTensors& tensors)
    {
        TransposeInstance::ConvertFrom(handle, kernels, pickTensorPtr(tensors));
    }

    void
    ConvertTo(const Handle& handle, const std::vector<Kernel>& kernels, const ConvTensors& tensors)
    {
        TransposeInstance::ConvertTo(handle, kernels, pickTensorPtr(tensors));
    }

    TransposeInstanceTagged()                               = delete;
    TransposeInstanceTagged(const TransposeInstanceTagged&) = default;
    TransposeInstanceTagged(TransposeInstanceTagged&&)      = default;
    ~TransposeInstanceTagged()                              = default;

private:
    Data_t pickTensorPtr(const ConvTensors& tensors) const
    {
        std::array<Data_t, 3> data_ptrs = {
            const_cast<Data_t>(tensors.x), // NOLINT (cppcoreguidelines-pro-type-const-cast)
            const_cast<Data_t>(tensors.w), // NOLINT (cppcoreguidelines-pro-type-const-cast)
            const_cast<Data_t>(tensors.y)  // NOLINT (cppcoreguidelines-pro-type-const-cast)
        };

        return data_ptrs[GetConvOperandTagAsInt()];
    }
};

template <typename CKArgsType,
          typename Input1TposeOp,
          typename Input2TposeOp,
          typename OutputTposeOp>
auto MakeTaggedTransposeInstances(ConvSolution& result,
                                  const ExecutionContext& ctx,
                                  const miopen::conv::ProblemDescription& problem,
                                  const CKArgsType& ck_args,
                                  const Input1TposeOp& input1_op,
                                  const Input2TposeOp& input2_op,
                                  const OutputTposeOp& output_op,
                                  std::optional<CKWorkspaceBufferDescriptor>& ck_buff_des)
{

    auto input1_solver = input1_op.MakeTransposeSolver(ctx, problem, ck_args);
    auto input2_solver = input2_op.MakeTransposeSolver(ctx, problem, ck_args);
    auto output_solver = output_op.MakeTransposeSolver(ctx, problem, ck_args);

    // NOTE: In cases where the convolution updates only a subset of output
    // indices, we need to first initialize the workspace buffer for
    // output with the real tensor for the output and then apply the convolution.
    // This is achieved by creating an input transpose op for the output workspace
    // bufffer.

    using OutputInitOp = CKTransposeInputOp<OutputTposeOp::NDIM, OutputTposeOp::CONV_OP_TAG>;

    auto output_init_solver = OutputInitOp{}.MakeTransposeSolver(ctx, problem, ck_args);

    result.construction_params.insert(result.construction_params.end(),
                                      {input1_solver.GetKernelInfo(),
                                       input2_solver.GetKernelInfo(),
                                       output_solver.GetKernelInfo(),
                                       output_init_solver.GetKernelInfo()});

    if(ck_buff_des.has_value())
    {
        MultiBufferWorkspaceTraits wt({input1_solver.GetOutputTensorSize(),
                                       input2_solver.GetOutputTensorSize(),
                                       output_solver.GetOutputTensorSize(),
                                       ck_buff_des->ck_size});
        ck_buff_des->ck_offset = wt.GetOffset(3);
        return std::make_tuple(
            TransposeInstanceTagged{input1_solver, 0, wt, 0, Input1TposeOp::CONV_OP_TAG},
            TransposeInstanceTagged{input2_solver, 1, wt, 1, Input2TposeOp::CONV_OP_TAG},
            TransposeInstanceTagged{output_solver, 2, wt, 2, OutputTposeOp::CONV_OP_TAG},
            TransposeInstanceTagged{output_init_solver, 3, wt, 2, OutputTposeOp::CONV_OP_TAG});
    }

    MultiBufferWorkspaceTraits wt({input1_solver.GetOutputTensorSize(),
                                   input2_solver.GetOutputTensorSize(),
                                   output_solver.GetOutputTensorSize()});
    return std::make_tuple(
        TransposeInstanceTagged{input1_solver, 0, wt, 0, Input1TposeOp::CONV_OP_TAG},
        TransposeInstanceTagged{input2_solver, 1, wt, 1, Input2TposeOp::CONV_OP_TAG},
        TransposeInstanceTagged{output_solver, 2, wt, 2, OutputTposeOp::CONV_OP_TAG},
        TransposeInstanceTagged{output_init_solver, 3, wt, 2, OutputTposeOp::CONV_OP_TAG});
}

#ifndef NDEBUG // disable for release builds, enable for debug builds

template <typename V>
void DebugPrintVec(const char* name, const V& vec)
{
    std::ostringstream oss;
    oss << name << " = [ ";
    for(const auto& v : vec)
    {
        oss << v << ", ";
    }
    oss << "]";
    MIOPEN_LOG_I(oss.str());
}

#define DEBUG_PRINT_VEC(x) DebugPrintVec(#x, x);

inline void DebugPrintConvTensors(const ConvTensors& conv_tensors)
{
    MIOPEN_LOG_I("in ptr = " << conv_tensors.x);
    MIOPEN_LOG_I("w ptr = " << conv_tensors.w);
    MIOPEN_LOG_I("out ptr = " << conv_tensors.y);

    DEBUG_PRINT_VEC(conv_tensors.xDesc.GetLengths());
    DEBUG_PRINT_VEC(conv_tensors.wDesc.GetLengths());
    DEBUG_PRINT_VEC(conv_tensors.yDesc.GetLengths());
}

#undef DEBUG_PRINT_VEC

#endif // NDEBUG
} // end namespace internal

// packed size in bytes
inline size_t GetPackedSize(const TensorDescriptor& td)
{
    return td.GetElementSize() * GetTypeSize(td.GetType());
}

// Only selected CK families whose complete invocation defines the packed E tensor
// independently of its previous contents may omit the NCHW old-output transpose.
// Keep this list tied to the CK implementations, not to dtype, spatial-size or
// beta-zero heuristics. The caller also checks invocation scalars and disjoint
// descriptors/pointers; unsupported or newly introduced families default to false.
inline bool SelectedNCHWCKOutputIsFullyDefined(const miopen::conv::ProblemDescription& problem,
                                               const std::string& selected_id,
                                               std::optional<int> split_k)
{
    if((!problem.Is2d() && !problem.Is3d()) || !problem.IsLayoutDefault() ||
       problem.GetConv().mode != miopenConvolution ||
       problem.GetConv().trans_output_pads.size() != problem.GetSpatialDims() ||
       std::any_of(problem.GetConv().trans_output_pads.begin(),
                   problem.GetConv().trans_output_pads.end(),
                   [](int pad) { return pad != 0; }) ||
       (!problem.IsFp16() && !problem.IsBfp16()) || problem.HasNonPackedTensors() ||
       problem.GetGroupCount() == 0 || problem.GetAlphaBetaCase() != DEFAULT)
        return false;

    for(const auto* desc : {&problem.GetIn(), &problem.GetWeights(), &problem.GetOut()})
        for(const auto length : desc->GetLengths())
            if(length == 0)
                return false;

    const auto starts_with = [&](const char* prefix) { return selected_id.rfind(prefix, 0) == 0; };
    const auto ends_with   = [&](const char* suffix) {
        const std::string_view end{suffix};
        return selected_id.size() >= end.size() &&
               selected_id.compare(selected_id.size() - end.size(), end.size(), end) == 0;
    };
    if(problem.Is3d())
    {
        // The registered 3D default XDL-v1 backward-data instances have no D tensor.
        // Their bare IDs use CK's implicit split-1 argument and clear the full
        // packed E before the residue GEMMs. Other directions, explicit splits,
        // fused instances and v3 have no selected 3D proof here.
        return problem.GetDirection() == miopen::conv::Direction::BackwardData && !split_k &&
               starts_with("DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<") &&
               (selected_id.find(", Default,") != std::string::npos ||
                selected_id.find(", Filter1x1Stride1Pad0,") != std::string::npos) &&
               ends_with(">");
    }

    if(problem.IsDirectionForward())
    {
        // Ordinary unmerged packed FWD runs a Set epilogue over every logical
        // M/N coordinate, including partial tiles. Exclude both merge modes.
        return !split_k &&
               ((starts_with("DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<") &&
                 ends_with(", 1>")) ||
                ((starts_with("DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<") ||
                  starts_with("DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_WmmaPorted<")) &&
                 ends_with(", 1>")));
    }
    if(problem.GetDirection() == miopen::conv::Direction::BackwardData)
    {
        // No-D XDL v1/v3 always clear full packed E before the residue GEMMs;
        // no-D WMMA v3 either does the same or proves a split-1 pointwise Set.
        // A positive filter guarantees at least one nonempty residue/slice.
        return split_k && *split_k > 0 &&
               (starts_with("DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<") ||
                starts_with("DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffleV3<") ||
                starts_with("DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffleV3_WmmaPorted<") ||
                starts_with("DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<"));
    }
    return false;
}

inline size_t GetCKAlphaBetaWorkspace(const miopen::conv::ProblemDescription& problem)
{
    std::size_t buff_size;

    TensorDescriptor input          = problem.GetIn();
    TensorDescriptor output         = problem.GetOut();
    TensorDescriptor weights        = problem.GetWeights();
    ConvolutionDescriptor conv_desc = problem.GetConv();

    miopenConvolutionABBackwardWeightsGetWorkSpaceSize(
        problem.GetAlphaBetaCase(), &input, &output, &weights, &conv_desc, &buff_size);
    return buff_size;
}

inline bool CKWrwRequireWorkspace(
    size_t G, size_t C, size_t K, miopenDataType_t data_type, miopenAlphaBetaCase_t alpha_beta_case)
{
    auto is_odd        = [](size_t num) { return num % 2 != 0; };
    size_t C_per_group = C / G;
    size_t K_per_group = K / G;

    return (alpha_beta_case == BILINEAR || alpha_beta_case == SCALE) ||
           ((data_type == miopenHalf || data_type == miopenBFloat16) &&
            (is_odd(C_per_group) || is_odd(K_per_group)));
}

/// \todo move to a cpp file
inline size_t GetWorkspaceSizeLayoutTransformConv(const miopen::conv::ProblemDescription& problem,
                                                  size_t ck_ws_size = 0)
{
    if(problem.IsLayoutNHWC())
    {
        if(problem.GetDirection() == ::miopen::conv::Direction::BackwardWeights)
        {
            return (ck_ws_size > 0) ? ck_ws_size : GetCKAlphaBetaWorkspace(problem);
        }
        // Only forward's selected candidate consumes native CK scratch here.
        return problem.IsDirectionForward() ? ck_ws_size : 0;
    }

    assert(problem.IsLayoutDefault());

    if(problem.GetDirection() == ::miopen::conv::Direction::BackwardWeights)
    {
        MultiBufferWorkspaceTraits wt(
            {GetPackedSize(problem.GetIn()),
             GetPackedSize(problem.GetWeights()),
             GetPackedSize(problem.GetOut()),
             (ck_ws_size > 0) ? ck_ws_size : GetCKAlphaBetaWorkspace(problem)});
        return wt.GetSize();
    }

    if(problem.IsDirectionForward())
    {
        MultiBufferWorkspaceTraits wt({GetPackedSize(problem.GetIn()),
                                       GetPackedSize(problem.GetWeights()),
                                       GetPackedSize(problem.GetOut()),
                                       ck_ws_size});
        return wt.GetSize();
    }

    MultiBufferWorkspaceTraits wt({GetPackedSize(problem.GetIn()),
                                   GetPackedSize(problem.GetWeights()),
                                   GetPackedSize(problem.GetOut())});
    return wt.GetSize();
}

inline void
ZeroOutTensor(const Handle& handle, const TensorDescriptor& tensorDesc, Data_t tensorData)
{
#if MIOPEN_BACKEND_HIP
    // SetTensor is required for non-packed tensors, but is also slower.
    // Use faster clear if possible.
    if(tensorDesc.IsPacked())
    {
        HipEventProfiler pfr(handle);

        auto status = hipMemsetAsync(tensorData, 0, tensorDesc.GetNumBytes(), handle.GetStream());
        if(status != hipSuccess)
        {
            MIOPEN_THROW_HIP_STATUS(status, "hipMemsetAsync() failed");
        }
    }
    else
#endif
    {
        auto zero = 0.0f;
        SetTensor(handle, tensorDesc, tensorData, &zero);
    }
}

template <typename CastType>
Data_t GetWorkspacePointer(const CastType& data_ctx)
{
    if constexpr(std::is_same_v<CastType, miopen::conv::DataInvokeParams> ||
                 std::is_same_v<CastType, miopen::conv::WrWInvokeParams> ||
                 std::is_same_v<CastType, miopen::fusion::FusionInvokeParams>)
    {
        return data_ctx.workSpace;
    }
    else
    {
        MIOPEN_THROW(miopenStatusNotImplemented,
                     "Unsupported CastType for workspace extraction: " +
                         std::string(typeid(CastType).name()));
    }
}

template <typename CastType>
void ValidateWorkspacePointer(Data_t workspace_ptr)
{
    if(!workspace_ptr)
    {
        MIOPEN_THROW(miopenStatusInvalidValue, "Workspace pointer is null");
    }
}

template <typename CastType>
ConvTensors GetTensors(const CastType& data_ctx)
{
    if constexpr(std::is_same_v<CastType, miopen::fusion::FusionInvokeParams>)
    {
        const auto& conv_param = dynamic_cast<const miopen::fusion::ConvolutionOpInvokeParam&>(
            *data_ctx.op_args.params[0]);
        assert(&conv_param);

        ConvTensors tensors;
        tensors.x     = data_ctx.in;
        tensors.xDesc = data_ctx.inDesc;
        tensors.w     = conv_param.weights;
        tensors.y     = data_ctx.out;
        tensors.yDesc = data_ctx.outDesc;

        return tensors;
    }
    else
    {
        return ConvTensors(data_ctx.tensors);
    }
}

} // namespace solver
} // namespace miopen
