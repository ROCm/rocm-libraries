// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <miopen/solver/implicitgemm_ck_util_common.hpp>
#include <miopen/kernel_tuning_mode.hpp>

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <ck/utility/data_type.hpp>
#include <ck/utility/numeric_limits.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_weight.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_weight_bilinear.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_weight_scale.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_data.hpp>
#endif // MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

namespace miopen {
namespace solver {

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

namespace internal {
#ifndef NDEBUG

#define DEBUG_PRINT_VEC(x) DebugPrintVec(#x, x);

template <typename CKArgsType, typename ConvPtr>
void DebugPrintCKArgPtrs(
    const CKArgsType& ck_args, const ConvPtr& conv_ptr, ConstData_t x, ConstData_t w, ConstData_t y)
{

    MIOPEN_LOG_I("CK Instance: " << conv_ptr->GetTypeString());
    MIOPEN_LOG_I("in ptr = " << x);
    MIOPEN_LOG_I("w ptr = " << w);
    MIOPEN_LOG_I("out ptr = " << y);

    DEBUG_PRINT_VEC(ck_args.input);
    DEBUG_PRINT_VEC(ck_args.in_strides);
    DEBUG_PRINT_VEC(ck_args.weight);
    DEBUG_PRINT_VEC(ck_args.wei_strides);
    DEBUG_PRINT_VEC(ck_args.output);
    DEBUG_PRINT_VEC(ck_args.out_strides);
}

#undef DEBUG_PRINT_VEC

#endif // NDEBUG
} // namespace internal

namespace conv {
template <typename DataType, typename ComputeType = DataType>
using DeviceOpGWrw = ck::tensor_operation::device::DeviceGroupedConvBwdWeight<
    2,
    ck::tensor_layout::convolution::NHWGC,
    ck::tensor_layout::convolution::GKYXC,
    ck::tensor_layout::convolution::NHWGK,
    DataType,
    DataType,
    DataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ComputeType>;
template <typename DataType, typename ComputeType = DataType>
using DeviceOpGWrwPtrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
    DeviceOpGWrw<DataType, ComputeType>>;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGBwd = ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD<
    2,
    ck::tensor_layout::convolution::NHWGK,
    ck::tensor_layout::convolution::GKYXC,
    ck::Tuple<>,
    ck::tensor_layout::convolution::NHWGC,
    DataType,
    DataType,
    ck::Tuple<>,
    DataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ComputeType>;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGBwdPtrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
    DeviceOpGBwd<DataType, ComputeType>>;

using InLayout    = ck::tensor_layout::convolution::NDHWGC;
using WeiLayout   = ck::tensor_layout::convolution::GKZYXC;
using OutLayout   = ck::tensor_layout::convolution::NDHWGK;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Bilinear    = ck::tensor_operation::element_wise::Bilinear;
using Scale       = ck::tensor_operation::element_wise::Scale;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGBwdWeightDefault =
    ck::tensor_operation::device::DeviceGroupedConvBwdWeight<3,
                                                             InLayout,
                                                             WeiLayout,
                                                             OutLayout,
                                                             DataType,
                                                             DataType,
                                                             DataType,
                                                             PassThrough,
                                                             PassThrough,
                                                             PassThrough,
                                                             ComputeType>;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGBwdWeightBilinear =
    ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD<3,
                                                                      InLayout,
                                                                      WeiLayout,
                                                                      OutLayout,
                                                                      ck::Tuple<WeiLayout>,
                                                                      DataType,
                                                                      DataType,
                                                                      DataType,
                                                                      ck::Tuple<DataType>,
                                                                      PassThrough,
                                                                      Bilinear,
                                                                      PassThrough,
                                                                      ComputeType>;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGBwdWeightScale =
    ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD<3,
                                                                      InLayout,
                                                                      WeiLayout,
                                                                      OutLayout,
                                                                      ck::Tuple<>,
                                                                      DataType,
                                                                      DataType,
                                                                      DataType,
                                                                      ck::Tuple<>,
                                                                      PassThrough,
                                                                      Scale,
                                                                      PassThrough,
                                                                      ComputeType>;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGBwdWeightDefaultPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOpGBwdWeightDefault<DataType, ComputeType>>;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGBwdWeightBilinearPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOpGBwdWeightBilinear<DataType, ComputeType>>;

template <typename DataType, typename ComputeType = DataType>
using DeviceOpGBwdWeightScalePtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOpGBwdWeightScale<DataType, ComputeType>>;

} // namespace conv

#endif

template <typename ConvPtrsType>
typename ConvPtrsType::iterator FindConvPtrByID(ConvPtrsType& conv_ptrs,
                                                const std::string& kernel_id)
{
    return std::find_if(conv_ptrs.begin(), conv_ptrs.end(), [&kernel_id](const auto& ptr) {
        return ptr->GetTypeString() == kernel_id;
    });
}

template <typename DeviceOpType,
          typename CKArgsType,
          typename ProblemDescriptionType = miopen::conv::ProblemDescription>
std::vector<std::string> FillValidKernelsIDs(const ProblemDescriptionType& problem)
{
    const auto args      = CKArgsType{problem};
    const auto conv_ptrs = DeviceOpType::GetInstances();
    assert(!conv_ptrs.empty());

    std::vector<std::string> valid_kernels;
    valid_kernels.reserve(conv_ptrs.size());
    for(size_t idx = 0; idx < conv_ptrs.size(); ++idx)
    {
        if(args.IsSupportedBy(conv_ptrs[idx]))
            valid_kernels.emplace_back(std::move(conv_ptrs[idx]->GetTypeString()));
    }
    assert(!valid_kernels.empty());
    return valid_kernels;
}

/**
 * @brief Generic implementation for filling valid kernel IDs across all data types
 *
 * This template function provides a reusable implementation that handles data type
 * dispatch and TF32 fallback logic. It's used by all hip_implicit_gemm solvers to
 * avoid code duplication when constructing the fill_valid_kernels lambda.
 *
 * @tparam DeviceOpPtrs CK DeviceOp factory template (e.g., DeviceOpGFwdPtrs)
 * @tparam CKArgs CK argument structure for the solver
 * @tparam ProblemDescriptionType Problem description type (default:
 * miopen::conv::ProblemDescription)
 *
 * @param problem Convolution problem description
 * @return Vector of valid kernel ID strings for the problem
 *
 * @note Automatically handles TF32 mode for float data type, falling back to
 *       regular float kernels if TF32 kernels are unavailable.
 *
 * ## Usage Example (in solver HeuristicInit)
 * ```cpp
 * auto fill_valid_kernels = [&](const ProblemDescription& p) {
 *     return FillValidKernelsGeneric<DeviceOpGFwdPtrs, CKArgs>(p);
 * };
 * ```
 */
template <template <typename, typename> class DeviceOpPtrs,
          typename CKArgs,
          typename ProblemDescriptionType = miopen::conv::ProblemDescription>
std::vector<std::string> FillValidKernelsGeneric(const ProblemDescriptionType& problem)
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    switch(problem.GetInDataType())
    {
    case miopenHalf:
        return FillValidKernelsIDs<DeviceOpPtrs<ck::half_t, ck::half_t>, CKArgs>(problem);

    case miopenFloat:
        if(problem.UseTF32())
        {
            auto tf32_kernels =
                FillValidKernelsIDs<DeviceOpPtrs<float, ck::tf32_t>, CKArgs>(problem);
            if(!tf32_kernels.empty())
                return tf32_kernels;
        }
        return FillValidKernelsIDs<DeviceOpPtrs<float, float>, CKArgs>(problem);

    case miopenBFloat16:
        return FillValidKernelsIDs<DeviceOpPtrs<ck::bhalf_t, ck::bhalf_t>, CKArgs>(problem);

    case miopenInt8: return FillValidKernelsIDs<DeviceOpPtrs<int8_t, int8_t>, CKArgs>(problem);

    default: return {};
    }
#else
    (void)problem;
    return {};
#endif
}

/**
 * @brief Helper function to dispatch by alpha/beta case for specific data types
 * It dispatches to the appropriate DeviceOp type based on the problem's alpha/beta case.
 *
 * @tparam BilinearPtrs CK DeviceOp factory for BILINEAR operations
 * @tparam ScalePtrs CK DeviceOp factory for SCALE operations
 * @tparam DefaultPtrs CK DeviceOp factory for DEFAULT (PassThrough) operations
 * @tparam CKArgs CK argument structure (must be templated on DataType, ComputeType)
 * @tparam DataType Input/output data type
 * @tparam ComputeType Computation type (may differ from DataType, e.g., tf32 for float)
 * @tparam ProblemDescriptionType Problem description type
 *
 * @param problem Convolution problem description
 * @return Vector of valid kernel ID strings for the problem
 */
template <template <typename, typename> class BilinearPtrs,
          template <typename, typename>
          class ScalePtrs,
          template <typename, typename>
          class DefaultPtrs,
          template <typename, typename>
          class CKArgs,
          typename DataType,
          typename ComputeType,
          typename ProblemDescriptionType = miopen::conv::ProblemDescription>
std::vector<std::string> FillKernelsByAlphaBeta(const ProblemDescriptionType& problem)
{
    switch(problem.GetAlphaBetaCase())
    {
    case BILINEAR:
        return FillValidKernelsIDs<BilinearPtrs<DataType, ComputeType>,
                                   CKArgs<DataType, ComputeType>>(problem);
    case SCALE:
        return FillValidKernelsIDs<ScalePtrs<DataType, ComputeType>, CKArgs<DataType, ComputeType>>(
            problem);
    default: // DEFAULT case (PassThrough)
        return FillValidKernelsIDs<DefaultPtrs<DataType, ComputeType>,
                                   CKArgs<DataType, ComputeType>>(problem);
    }
}

/**
 * @brief Generic implementation for filling valid kernel IDs with alpha/beta case dispatch
 *
 * This template function extends FillValidKernelsGeneric to handle 3D solvers that support
 * multiple alpha/beta operation types (BILINEAR, SCALE, DEFAULT). It dispatches to the
 * appropriate DeviceOp type based on the problem's alpha/beta case.
 *
 * @tparam BilinearPtrs CK DeviceOp factory for BILINEAR operations
 * @tparam ScalePtrs CK DeviceOp factory for SCALE operations
 * @tparam DefaultPtrs CK DeviceOp factory for DEFAULT (PassThrough) operations
 * @tparam CKArgs CK argument structure (must be templated on DataType, ComputeType)
 * @tparam ProblemDescriptionType Problem description type
 *
 * @param problem Convolution problem description
 * @return Vector of valid kernel ID strings for the problem
 *
 * @note This is used by 3D solvers (Fwd/Bwd/Wrw) that support alpha/beta fusion.
 *       2D solvers don't need this as they use simpler operation types.
 *
 * ## Usage Example (in 3D solver HeuristicInit)
 * ```cpp
 * auto fill_valid_kernels = [&](const ProblemDescription& p) {
 *     return FillValidKernelsWithAlphaBetaGeneric<
 *         DeviceOpGFwdBilinearPtrs,
 *         DeviceOpGFwdScalePtrs,
 *         DeviceOpGFwdDefaultPtrs,
 *         CKArgs>(p);
 * };
 * ```
 */
template <template <typename, typename> class BilinearPtrs,
          template <typename, typename>
          class ScalePtrs,
          template <typename, typename>
          class DefaultPtrs,
          template <typename, typename>
          class CKArgs,
          typename ProblemDescriptionType = miopen::conv::ProblemDescription>
std::vector<std::string> FillValidKernelsWithAlphaBetaGeneric(const ProblemDescriptionType& problem)
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    // Data type dispatch with TF32 support (C++17 compatible - uses helper function)
    switch(problem.GetInDataType())
    {
    case miopenHalf:
        return FillKernelsByAlphaBeta<BilinearPtrs,
                                      ScalePtrs,
                                      DefaultPtrs,
                                      CKArgs,
                                      ck::half_t,
                                      ck::half_t>(problem);

    case miopenFloat:
        if(problem.UseTF32())
        {
            auto tf32_kernels = FillKernelsByAlphaBeta<BilinearPtrs,
                                                       ScalePtrs,
                                                       DefaultPtrs,
                                                       CKArgs,
                                                       float,
                                                       ck::tf32_t>(problem);
            if(!tf32_kernels.empty())
                return tf32_kernels;
        }
        return FillKernelsByAlphaBeta<BilinearPtrs, ScalePtrs, DefaultPtrs, CKArgs, float, float>(
            problem);

    case miopenBFloat16:
        return FillKernelsByAlphaBeta<BilinearPtrs,
                                      ScalePtrs,
                                      DefaultPtrs,
                                      CKArgs,
                                      ck::bhalf_t,
                                      ck::bhalf_t>(problem);

    case miopenInt8:
        return FillKernelsByAlphaBeta<BilinearPtrs, ScalePtrs, DefaultPtrs, CKArgs, int8_t, int8_t>(
            problem);

    default: return {};
    }
#else
    (void)problem;
    return {};
#endif
}

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
template <typename DeviceOpType>
inline constexpr bool IsSplitKNeeded()
{
    return std::is_same_v<DeviceOpType, conv::DeviceOpGWrwPtrs<ck::half_t>> ||
           std::is_same_v<DeviceOpType, conv::DeviceOpGWrwPtrs<float>> ||
           std::is_same_v<DeviceOpType, conv::DeviceOpGWrwPtrs<float, ck::tf32_t>> ||
           std::is_same_v<DeviceOpType, conv::DeviceOpGWrwPtrs<int8_t>> ||
           std::is_same_v<DeviceOpType, conv::DeviceOpGWrwPtrs<ck::bhalf_t>> ||
           std::is_same_v<DeviceOpType, conv::DeviceOpGBwdPtrs<ck::half_t>> ||
           std::is_same_v<DeviceOpType, conv::DeviceOpGBwdPtrs<float>> ||
           std::is_same_v<DeviceOpType, conv::DeviceOpGBwdPtrs<float, ck::tf32_t>> ||
           std::is_same_v<DeviceOpType, conv::DeviceOpGBwdPtrs<int8_t>> ||
           std::is_same_v<DeviceOpType, conv::DeviceOpGBwdPtrs<ck::bhalf_t>> ||
           std::is_same_v<DeviceOpType, conv::DeviceOpGBwdWeightDefaultPtrs<ck::half_t>> ||
           std::is_same_v<DeviceOpType, conv::DeviceOpGBwdWeightDefaultPtrs<float>> ||
           std::is_same_v<DeviceOpType, conv::DeviceOpGBwdWeightDefaultPtrs<float, ck::tf32_t>> ||
           std::is_same_v<DeviceOpType, conv::DeviceOpGBwdWeightDefaultPtrs<int8_t>> ||
           std::is_same_v<DeviceOpType, conv::DeviceOpGBwdWeightDefaultPtrs<ck::bhalf_t>> ||
           std::is_same_v<DeviceOpType, conv::DeviceOpGBwdWeightBilinearPtrs<ck::half_t>> ||
           std::is_same_v<DeviceOpType, conv::DeviceOpGBwdWeightBilinearPtrs<float>> ||
           std::is_same_v<DeviceOpType, conv::DeviceOpGBwdWeightBilinearPtrs<float, ck::tf32_t>> ||
           std::is_same_v<DeviceOpType, conv::DeviceOpGBwdWeightBilinearPtrs<int8_t>> ||
           std::is_same_v<DeviceOpType, conv::DeviceOpGBwdWeightBilinearPtrs<ck::bhalf_t>> ||
           std::is_same_v<DeviceOpType, conv::DeviceOpGBwdWeightScalePtrs<ck::half_t>> ||
           std::is_same_v<DeviceOpType, conv::DeviceOpGBwdWeightScalePtrs<float>> ||
           std::is_same_v<DeviceOpType, conv::DeviceOpGBwdWeightScalePtrs<float, ck::tf32_t>> ||
           std::is_same_v<DeviceOpType, conv::DeviceOpGBwdWeightScalePtrs<int8_t>> ||
           std::is_same_v<DeviceOpType, conv::DeviceOpGBwdWeightScalePtrs<ck::bhalf_t>>;
}

#endif // MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

template <typename DeviceOpType,
          typename CKArgsType,
          typename ProblemDescriptionType = miopen::conv::ProblemDescription,
          bool CheckSplitK                = false>
bool IsCKArgsSupported(const ProblemDescriptionType& problem, const std::string& kernel_id)
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    if(!kernel_id.empty())
    {
        auto conv_ptrs = DeviceOpType::GetInstances();
        if constexpr(IsSplitKNeeded<DeviceOpType>() || CheckSplitK)
        {
            auto pos = kernel_id.find_last_of('+');
            if(pos == std::string::npos)
            {
                MIOPEN_LOG_WE("Unable to parse split_k from kernel_id for wrw: " << kernel_id);
                return false;
            }

            int split_k = 1;
            try
            {
                split_k = std::stoi(kernel_id.substr(pos + 1));
            }
            catch(std::exception& e)
            {
                MIOPEN_LOG_WE("Unable to parse split_k from kernel_id for wrw: "
                              << kernel_id << " : " << e.what());
                return false;
            }

            auto ptr_iter = FindConvPtrByID(conv_ptrs, kernel_id.substr(0, pos));
            return (ptr_iter != conv_ptrs.end()) &&
                   CKArgsType{problem}.IsSupportedBySplitK(*ptr_iter, split_k);
        }
        else
        {
            auto ptr_iter = FindConvPtrByID(conv_ptrs, kernel_id);
            return (ptr_iter != conv_ptrs.end()) && CKArgsType{problem}.IsSupportedBy(*ptr_iter);
        }
    }
#endif
    return false;
}

template <typename DeviceOpType,
          typename CKArgsType,
          typename ProblemDescriptionType = miopen::conv::ProblemDescription>
bool IsCKApplicable(const ProblemDescriptionType& problem)
{
    const auto args = CKArgsType{problem};

    const auto ptrs = DeviceOpType::GetInstances();
    return std::any_of(
        ptrs.begin(), ptrs.end(), [&args](auto& ptr) { return args.IsSupportedBy(ptr); });
}

/**
 * @brief Check if a kernel+split_k combination is supported by CK
 *
 * This function validates whether a specific CK kernel instance supports
 * a given split_k value. It uses CK's IsSupportedBySplitK method to check
 * if the (kernel, split_k) combination is valid.
 *
 * @tparam DeviceOpType CK DeviceOp factory type (e.g., DeviceOpGBwdPtrs<float>)
 * @tparam CKArgsType CK arguments structure type
 * @tparam ProblemDescriptionType Problem description type
 *
 * @param problem The convolution problem description
 * @param kernel_id The kernel type string (without split_k suffix)
 * @param split_k The split_k value to validate
 * @return true if the (kernel_id, split_k) combination is supported by CK
 *
 * ## Usage Example
 * ```cpp
 * bool supported = IsCKSplitKSupported<DeviceOpGBwdPtrs<float>, CKArgs>(
 *     problem, "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<...>", 4);
 * ```
 */
template <typename DeviceOpType,
          typename CKArgsType,
          typename ProblemDescriptionType = miopen::conv::ProblemDescription>
bool IsCKSplitKSupported(const ProblemDescriptionType& problem,
                         const std::string& kernel_id,
                         int split_k)
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    auto conv_ptrs = DeviceOpType::GetInstances();
    auto ptr_iter  = FindConvPtrByID(conv_ptrs, kernel_id);

    if(ptr_iter == conv_ptrs.end())
    {
        return false;
    }

    const auto args = CKArgsType{problem};
    return args.IsSupportedBySplitK(*ptr_iter, split_k);
#else
    (void)problem;
    (void)kernel_id;
    (void)split_k;
    return false;
#endif
}

/**
 * @brief Generic CK split_k validator for multiple data types
 *
 * This template function creates a validator that checks all supported
 * data types for a given DeviceOp template. Use this in solvers to
 * create a CK validator without data type dispatch code.
 *
 * @tparam DeviceOpPtrs CK DeviceOp factory template (templated on DataType, ComputeType)
 * @tparam CKArgs CK argument structure template (templated on DataType, ComputeType)
 *
 * @param problem The convolution problem description
 * @param kernel_id The kernel type string (without split_k suffix)
 * @param split_k The split_k value to validate
 * @return true if any data type supports this (kernel_id, split_k) combination
 */
template <template <typename, typename> class DeviceOpPtrs,
          typename CKArgs,
          typename ProblemDescriptionType = miopen::conv::ProblemDescription>
bool IsCKSplitKSupportedGeneric(const ProblemDescriptionType& problem,
                                const std::string& kernel_id,
                                int split_k)
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    // Check the data type that matches the problem
    switch(problem.GetInDataType())
    {
    case miopenHalf:
        return IsCKSplitKSupported<DeviceOpPtrs<ck::half_t, ck::half_t>, CKArgs>(
            problem, kernel_id, split_k);

    case miopenFloat:
        if(problem.UseTF32())
        {
            if(IsCKSplitKSupported<DeviceOpPtrs<float, ck::tf32_t>, CKArgs>(
                   problem, kernel_id, split_k))
                return true;
        }
        return IsCKSplitKSupported<DeviceOpPtrs<float, float>, CKArgs>(problem, kernel_id, split_k);

    case miopenBFloat16:
        return IsCKSplitKSupported<DeviceOpPtrs<ck::bhalf_t, ck::bhalf_t>, CKArgs>(
            problem, kernel_id, split_k);

    case miopenInt8:
        return IsCKSplitKSupported<DeviceOpPtrs<int8_t, int8_t>, CKArgs>(
            problem, kernel_id, split_k);

    default: return false;
    }
#else
    (void)problem;
    (void)kernel_id;
    (void)split_k;
    return false;
#endif
}

template <typename DeviceOpType,
          typename CKArgsType,
          typename ProblemDescriptionType = miopen::conv::ProblemDescription>
size_t GetCKSplitkMaxWorkspaceSize(const ProblemDescriptionType& problem)
{
    const auto args         = CKArgsType{problem};
    auto max_workspace_size = 0;

    const auto ptrs = DeviceOpType::GetInstances();
    for(auto& ptr : ptrs)
    {
        // Cycle `split_k` over {1,2,4,...,128} then `CkSplitkAutoDeduce`.
        // The loop then restarts from 1 for the next conv instance.
        auto split_k = 1;
        do
        {
            if(args.IsSupportedBySplitK(ptr, split_k))
            {
                auto workspace_size = args.GetCKSplitkWorkspaceSize(ptr, split_k);
                if(workspace_size > max_workspace_size)
                    max_workspace_size = workspace_size;
            }
        } while(!NextCKSplitkValue<1, 128>(split_k));
    }

    MIOPEN_LOG_I("Max workspace size reported by CK: " << max_workspace_size);
    return max_workspace_size;
}

#define WORKAROUND_CK_ISSUE_1184 1
#if WORKAROUND_CK_ISSUE_1184
using WorkAroundHipEventProfiler = HipEventProfiler;
#endif

inline bool isDataTypeHalfAndChannelsEven(const miopen::conv::ProblemDescription& problem)
{
    return (problem.GetOutDataType() == miopenHalf) &&
           ((problem.GetInChannels() & 1) != 0 ||
            (problem.GetOutChannels() & 1) != 0 /* Test if odd*/);
}

inline bool ShouldAllocateWorkSpaceBufferForWRW(const miopen::conv::ProblemDescription& problem)
{
    return (problem.GetAlphaBetaCase() != DEFAULT || isDataTypeHalfAndChannelsEven(problem));
}

template <typename DeviceOpType,
          typename CKArgsType,
          typename CastType,
          typename ProblemDescriptionType = miopen::conv::ProblemDescription>
ConvSolution InitAnyInvokerFactory(const ProblemDescriptionType& problem,
                                   const std::string& kernel_id)
{
    auto conv_ptrs = DeviceOpType::GetInstances();
    auto ptr_iter  = FindConvPtrByID(conv_ptrs, kernel_id);

    if(ptr_iter == conv_ptrs.end())
    {
        MIOPEN_LOG_E("Kernel does not exist.");
        return {miopenStatusInvalidValue};
    }

    ConvSolution result;
#ifdef CK_EXPERIMENTAL_BUILDER
    std::string description = (*ptr_iter)->describe()->detailed();

    if(!description.empty())
    {
        MIOPEN_LOG_I(description);
    }
#endif

    result.invoker_factory =
        [kernel_id_   = kernel_id,
         ck_args_     = CKArgsType{problem},
         sh_conv_ptr_ = std::shared_ptr{std::move(*ptr_iter)}](const std::vector<Kernel>&) mutable {
            return [kernel_id2   = std::move(kernel_id_),
                    ck_args2     = std::move(ck_args_),
                    sh_conv_ptr2 = std::move(sh_conv_ptr_)](
                       const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                const auto& data_ctx = primitive_parameters.CastTo<CastType>();
                auto argument_ptr    = ck_args2.MakeArgPtr(sh_conv_ptr2, data_ctx);
                auto invoker_ptr     = sh_conv_ptr2->MakeInvokerPointer();
                {
                    WorkAroundHipEventProfiler prf(handle);
                    invoker_ptr->Run(argument_ptr.get(), {handle.GetStream(), false});
                }
                if(handle.IsProfilingEnabled())
                {
                    float elapsed_time = handle.GetKernelTime();
                    AddKernelToJsonAccumulator(kernel_id2, elapsed_time, false);
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(elapsed_time);
                }
            };
        };
    return result;
}

template <typename DataType, typename OutElemOp>
OutElemOp GetOutElementOp(const miopen::fusion::ActivationOpInvokeParam& activationOp)
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    auto activationMode = activationOp.activMode;
    switch(activationMode)
    {
    case miopenActivationRELU: return OutElemOp{0, ck::NumericLimits<DataType>::Max()};
    case miopenActivationCLIPPEDRELU: return OutElemOp{0, activationOp.activAlpha};
    case miopenActivationCLAMP: return OutElemOp{activationOp.activAlpha, activationOp.activBeta};
    default:
        MIOPEN_THROW(miopenStatusInternalError,
                     "Unsupported activation type: " + std::to_string(activationMode));
    }
#else
    MIOPEN_THROW(miopenStatusNotImplemented, "Not implemented without ck enabled");
#endif
}

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

template <bool NeedsSplitK, typename DeviceOpType, typename CKArgsType, typename CastType>
std::unique_ptr<ck::tensor_operation::device::BaseArgument>
MakeNCHWCKArgPtr(const CKArgsType& ck_args,
                 const std::shared_ptr<DeviceOpType>& sh_conv_ptr,
                 const std::array<internal::TransposeInstanceTagged*, 3>& tr_ptrs,
                 const CastType& data_ctx,
                 const std::optional<int>& split_k)
{
    std::unique_ptr<ck::tensor_operation::device::BaseArgument> argument_ptr;

    if constexpr(std::is_same_v<CastType, miopen::fusion::FusionInvokeParams>)
    {
        const auto& conv_param = dynamic_cast<const miopen::fusion::ConvolutionOpInvokeParam&>(
            *data_ctx.op_args.params[0]);
        assert(&conv_param);

        const miopen::fusion::ActivationOpInvokeParam* activ_param_ptr = nullptr;
        ConstData_t bias_buf                                           = nullptr;

        if(data_ctx.op_args.params.size() == 2)
        {
            activ_param_ptr = &dynamic_cast<const miopen::fusion::ActivationOpInvokeParam&>(
                *data_ctx.op_args.params[1]);
            assert(activ_param_ptr);
        }
        else if(data_ctx.op_args.params.size() == 3)
        {
            const auto& bias_param =
                dynamic_cast<const miopen::fusion::BiasOpInvokeParam&>(*data_ctx.op_args.params[1]);
            assert(&bias_param);
            bias_buf = bias_param.bdata;

            activ_param_ptr = &dynamic_cast<const miopen::fusion::ActivationOpInvokeParam&>(
                *data_ctx.op_args.params[2]);
            assert(activ_param_ptr);
        }
        else
        {
            throw miopen::Exception(miopenStatusInternalError,
                                    "Unsupported number of parameters for FusionInvokeParams: " +
                                        std::to_string(data_ctx.op_args.params.size()));
        }

        argument_ptr = ck_args.MakeArgPtr(
            sh_conv_ptr,
            tr_ptrs[0]->GetBufferPtr(),
            tr_ptrs[1]->GetBufferPtr(),
            bias_buf,
            tr_ptrs[2]->GetBufferPtr(),
            conv_param.alpha,
            conv_param.beta,
            GetOutElementOp<typename CKArgsType::OutputDataType,
                            typename CKArgsType::OutputElementOpType>(*activ_param_ptr));
    }
    else
    {
        if constexpr(NeedsSplitK)
        {
            if(split_k.has_value())
            {
                argument_ptr = ck_args.MakeArgPtr(sh_conv_ptr,
                                                  tr_ptrs[0]->GetBufferPtr(),
                                                  tr_ptrs[1]->GetBufferPtr(),
                                                  tr_ptrs[2]->GetBufferPtr(),
                                                  data_ctx.alpha.GetAsFloat(),
                                                  data_ctx.beta.GetAsFloat(),
                                                  split_k.value());
            }
            else
            {
                MIOPEN_THROW(miopenStatusInvalidValue, "split_k is required but not provided");
            }
        }
        else
        {
            argument_ptr = ck_args.MakeArgPtr(sh_conv_ptr,
                                              tr_ptrs[0]->GetBufferPtr(),
                                              tr_ptrs[1]->GetBufferPtr(),
                                              tr_ptrs[2]->GetBufferPtr(),
                                              data_ctx.alpha.GetAsFloat(),
                                              data_ctx.beta.GetAsFloat());
        }
    }

    MIOPEN_THROW_IF(argument_ptr == nullptr,
                    "Failed to create argument pointer ck_args argument ptr.");

    return argument_ptr;
}

template <bool NeedsSplitK, typename DeviceOpType, typename CKArgsType, typename CastType>
std::unique_ptr<ck::tensor_operation::device::BaseArgument>
MakeNHWCCKArgPtr(const std::shared_ptr<DeviceOpType>& sh_conv_ptr,
                 const CKArgsType& ck_args,
                 const CastType& data_ctx,
                 const std::optional<int>& split_k)
{
    std::unique_ptr<ck::tensor_operation::device::BaseArgument> argument_ptr;

    if constexpr(std::is_same_v<CastType, miopen::fusion::FusionInvokeParams>)
    {
        const auto& conv_param = dynamic_cast<const miopen::fusion::ConvolutionOpInvokeParam&>(
            *data_ctx.op_args.params[0]);
        assert(&conv_param);

        const miopen::fusion::ActivationOpInvokeParam* activ_param_ptr = nullptr;
        ConstData_t bias_buf                                           = nullptr;

        if(data_ctx.op_args.params.size() == 2)
        {
            activ_param_ptr = &dynamic_cast<const miopen::fusion::ActivationOpInvokeParam&>(
                *data_ctx.op_args.params[1]);
            assert(activ_param_ptr);
        }
        else if(data_ctx.op_args.params.size() == 3)
        {
            const auto& bias_param =
                dynamic_cast<const miopen::fusion::BiasOpInvokeParam&>(*data_ctx.op_args.params[1]);
            assert(&bias_param);
            bias_buf = bias_param.bdata;

            activ_param_ptr = &dynamic_cast<const miopen::fusion::ActivationOpInvokeParam&>(
                *data_ctx.op_args.params[2]);
            assert(activ_param_ptr);
        }
        else
        {
            throw miopen::Exception(miopenStatusInternalError,
                                    "Unsupported number of parameters for FusionInvokeParams: " +
                                        std::to_string(data_ctx.op_args.params.size()));
        }

        ConstData_t weight_buf = conv_param.weights;

        argument_ptr = ck_args.MakeArgPtr(
            sh_conv_ptr,
            data_ctx.in,
            weight_buf,
            bias_buf,
            data_ctx.out,
            conv_param.alpha,
            conv_param.beta,
            GetOutElementOp<typename CKArgsType::OutputDataType,
                            typename CKArgsType::OutputElementOpType>(*activ_param_ptr));
    }
    else
    {
        if constexpr(NeedsSplitK)
        {
            if(split_k.has_value())
            {
                argument_ptr = ck_args.MakeArgPtr(sh_conv_ptr,
                                                  data_ctx.tensors,
                                                  data_ctx.alpha.GetAsFloat(),
                                                  data_ctx.beta.GetAsFloat(),
                                                  split_k.value());
            }
            else
            {
                MIOPEN_THROW(miopenStatusInvalidValue, "split_k is required but not provided");
            }
        }
        else
        {
            std::ignore  = split_k;
            argument_ptr = ck_args.MakeArgPtr(sh_conv_ptr,
                                              data_ctx.tensors,
                                              data_ctx.alpha.GetAsFloat(),
                                              data_ctx.beta.GetAsFloat());
        }
    }

    MIOPEN_THROW_IF(argument_ptr == nullptr,
                    "Failed to create argument pointer ck_args argument ptr.");

    return argument_ptr;
}
#endif

template <bool ZeroOutputs,
          typename DeviceOpType,
          typename CKArgsType,
          typename CastType,
          typename Input1TposeOp,
          typename Input2TposeOp,
          typename OutputTposeOp>
ConvSolution InitInvokerFactoryNCHW(const ExecutionContext& ctx,
                                    const miopen::conv::ProblemDescription& problem,
                                    const std::string& kernel_id,
                                    const Input1TposeOp& input1_op,
                                    const Input2TposeOp& input2_op,
                                    const OutputTposeOp& output_op)
{
    assert(problem.IsLayoutDefault());

    ConvSolution result;
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    auto ck_args = CKArgsType{problem};

    auto conv_ptrs = DeviceOpType::GetInstances();

    std::optional<int> split_k = std::nullopt;
    std::string id_string      = kernel_id;
    auto pos                   = kernel_id.find_last_of('+');
    if(pos != std::string::npos)
    {
        split_k   = std::stoi(kernel_id.substr(pos + 1));
        id_string = kernel_id.substr(0, pos);
    }

    std::optional<CKBWDWeightBufferDescriptor> _ck_buff_des;

    auto ptr_iter = FindConvPtrByID(conv_ptrs, id_string);
    if(ptr_iter == conv_ptrs.end())
    {
        MIOPEN_LOG_E("PerformanceConfig kernel '" + kernel_id + "' does not exist.");
        result = ConvSolution{miopenStatusInvalidValue};
        return result;
    }

#ifdef CK_EXPERIMENTAL_BUILDER
    std::string description = (*ptr_iter)->describe()->detailed();

    if(!description.empty())
    {
        MIOPEN_LOG_I(description);
    }
#endif

    if constexpr(std::is_same_v<CastType, miopen::conv::WrWInvokeParams>)
    {
        auto ck_ws_size = ck_args.GetCKSplitkWorkspaceSize(*ptr_iter, split_k.value_or(1));
        _ck_buff_des.emplace(ck_ws_size, 0);
        result.workspace_sz = GetWorkspaceSizeLayoutTransformConv(problem, ck_ws_size);
    }
    else
    {
        result.workspace_sz = GetWorkspaceSizeLayoutTransformConv(problem);
    }

    auto [_input1_tr_inst, _input2_tr_inst, _output_tr_inst, _output_init_tr_inst] =
        internal::MakeTaggedTransposeInstances<CKArgsType>(
            result, ctx, problem, ck_args, input1_op, input2_op, output_op, _ck_buff_des);

    result.invoker_factory = [kernel_id_           = &kernel_id,
                              split_k_             = split_k,
                              ck_args_             = std::move(ck_args),
                              sh_conv_ptr_         = std::shared_ptr{std::move(*ptr_iter)},
                              input1_tr_inst_      = std::move(_input1_tr_inst),
                              input2_tr_inst_      = std::move(_input2_tr_inst),
                              output_tr_inst_      = std::move(_output_tr_inst),
                              output_init_tr_inst_ = std::move(_output_init_tr_inst),
                              ck_buff_des_ =
                                  _ck_buff_des](const std::vector<Kernel>& kernels) mutable {
        return [kernel_id2 = kernel_id_,
                split_k2   = split_k_,
                kernels,
                ck_args2             = std::move(ck_args_),
                sh_conv_ptr2         = std::move(sh_conv_ptr_),
                input1_tr_inst2      = std::move(input1_tr_inst_),
                input2_tr_inst2      = std::move(input2_tr_inst_),
                output_tr_inst2      = std::move(output_tr_inst_),
                output_init_tr_inst2 = std::move(output_init_tr_inst_),
                ck_buff_des2         = ck_buff_des_](const Handle& handle,
                                             const AnyInvokeParams& primitive_parameters) mutable {
            handle.ResetKernelTime();

            const auto& data_ctx = primitive_parameters.CastTo<CastType>();
            Data_t workspace_ptr = GetWorkspacePointer<CastType>(data_ctx);
            ValidateWorkspacePointer<CastType>(workspace_ptr);

            input1_tr_inst2.AssignBuffer(handle, workspace_ptr);
            input2_tr_inst2.AssignBuffer(handle, workspace_ptr);
            output_tr_inst2.AssignBuffer(handle, workspace_ptr);
            output_init_tr_inst2.AssignBuffer(handle, workspace_ptr);

            // if FusionInvokeParams extract tensors from the params
            // conversion operator applied here to convert to ConvTensors
            auto conv_tensors = GetTensors(data_ctx);

            /// \todo remove this when DataInvokeParams stops swapping
            // "in" and "out" tensors for backward pass
            if(output_tr_inst2.GetConvOperandTag() == internal::ConvOperandTag::Input)
            {
                // this is backward pass, swap back input and output
                std::swap(conv_tensors.x, conv_tensors.y);
                std::swap(conv_tensors.xDesc, conv_tensors.yDesc);
            }

            float elapsed = 0.0f;

            // ConvertFrom automatically keeps kernel time and accumulates
            input1_tr_inst2.ConvertFrom(handle, kernels, conv_tensors);
            input2_tr_inst2.ConvertFrom(handle, kernels, conv_tensors);
            output_init_tr_inst2.ConvertFrom(handle, kernels, conv_tensors);
            elapsed = handle.IsProfilingEnabled() ? handle.GetKernelTime() : 0.0f;

            if constexpr(ZeroOutputs)
            {
                /// Note: Need to clear buffer memory for output since all values may not be set.
                output_tr_inst2.ZeroOutBuffer(handle);
                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();
            }

            std::array<internal::TransposeInstanceTagged*, 3> tr_ptrs = {
                &input1_tr_inst2, &input2_tr_inst2, &output_tr_inst2};

            // sort by tag in order: Input, Weights, Output
            std::sort(tr_ptrs.begin(), tr_ptrs.end(), [](const auto& left, const auto& right) {
                return left->GetConvOperandTagAsInt() < right->GetConvOperandTagAsInt();
            });

            std::unique_ptr<ck::tensor_operation::device::BaseArgument> argument_ptr =
                MakeNCHWCKArgPtr<IsSplitKNeeded<DeviceOpType>(),
                                 std::decay_t<decltype(*sh_conv_ptr2)>,
                                 CKArgsType,
                                 CastType>(ck_args2, sh_conv_ptr2, tr_ptrs, data_ctx, split_k2);

            shared<Data_t> buf_handle{};
            if(ck_buff_des2.has_value() && ck_buff_des2->ck_size && workspace_ptr)
            {
                buf_handle = handle.CreateSubBuffer(
                    workspace_ptr, ck_buff_des2->ck_offset, ck_buff_des2->ck_size);
                assert(buf_handle.get());
                sh_conv_ptr2->SetWorkSpacePointer(argument_ptr.get(), buf_handle.get());
            }

            auto invoker_ptr = sh_conv_ptr2->MakeInvokerPointer();
            {
                WorkAroundHipEventProfiler prf(handle);
                MIOPEN_LOG_I2("kernel_name = " << kernel_id2);
                invoker_ptr->Run(argument_ptr.get(), {handle.GetStream(), false});
            }

            if(handle.IsProfilingEnabled())
            {
                elapsed += handle.GetKernelTime();

                // Kernel logging for CK kernels
                if(IsLoggingKernel())
                {
                    AddKernelToJsonAccumulator(*kernel_id2, elapsed, false);
                }
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }

            // ConvertTo automatically keeps kernel time and accumulates
            output_tr_inst2.ConvertTo(handle, kernels, conv_tensors);
        };
    };
#endif
    return result;
}

template <bool ZeroOutputs,
          typename DeviceOpType,
          typename CKArgsType,
          typename CastType,
          typename ProblemDescriptionType = miopen::conv::ProblemDescription>
ConvSolution InitInvokerFactoryNHWC(const ExecutionContext&,
                                    const ProblemDescriptionType& problem,
                                    const std::string& kernel_id)
{
    ConvSolution result;
    auto conv_ptrs             = DeviceOpType::GetInstances();
    std::optional<int> split_k = std::nullopt;
    std::string id_string      = kernel_id;
    auto pos                   = kernel_id.find_last_of('+');
    if(pos != std::string::npos)
    {
        split_k   = std::stoi(kernel_id.substr(pos + 1));
        id_string = kernel_id.substr(0, pos);
    }

    auto ptr_iter = FindConvPtrByID(conv_ptrs, id_string);

    if(ptr_iter == conv_ptrs.end())
    {
        MIOPEN_LOG_E("PerformanceConfig kernel '" + kernel_id + "' does not exist.");
        result = ConvSolution{miopenStatusInvalidValue};
        return result;
    }

#ifdef CK_EXPERIMENTAL_BUILDER
    std::string description = (*ptr_iter)->describe()->detailed();

    if(!description.empty())
    {
        MIOPEN_LOG_I(description);
    }
#endif
    if constexpr(std::is_same_v<CastType, miopen::conv::WrWInvokeParams>)
    {
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
        miopenAlphaBetaCase_t alpha_beta_case = problem.GetAlphaBetaCase();
        auto ck_args                          = CKArgsType{problem};
        auto ck_ws_size = ck_args.GetCKSplitkWorkspaceSize(*ptr_iter, split_k.value_or(1));
        [[maybe_unused]] bool should_allocated_wrw_buffer = ck_ws_size > 0;

        result.invoker_factory = [kernel_id_                   = kernel_id,
                                  split_k_                     = split_k,
                                  ck_args_                     = CKArgsType{problem},
                                  alpha_beta_case_             = alpha_beta_case,
                                  should_allocated_wrw_buffer_ = should_allocated_wrw_buffer,
                                  sh_conv_ptr_ = std::shared_ptr{std::move(*ptr_iter)}](
                                     const std::vector<Kernel>&) mutable {
            return [kernel_id2                   = kernel_id_,
                    split_k2                     = split_k_,
                    ck_args2                     = std::move(ck_args_),
                    alpha_beta_case2             = alpha_beta_case_,
                    should_allocated_wrw_buffer2 = should_allocated_wrw_buffer_,
                    sh_conv_ptr2                 = std::move(sh_conv_ptr_)](
                       const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                const auto& data_ctx = primitive_parameters.CastTo<CastType>();
                std::unique_ptr<ck::tensor_operation::device::BaseArgument> argument_ptr =
                    MakeNHWCCKArgPtr<IsSplitKNeeded<DeviceOpType>(),
                                     std::decay_t<decltype(*sh_conv_ptr2)>,
                                     CKArgsType,
                                     CastType>(sh_conv_ptr2, ck_args2, data_ctx, split_k2);

                float elapsed = 0.0f;
                if(alpha_beta_case2 == DEFAULT)
                {
                    if constexpr(ZeroOutputs)
                    {
                        ZeroOutTensor(handle, data_ctx.tensors.dwDesc, data_ctx.tensors.dw);

                        if(handle.IsProfilingEnabled())
                        {
                            elapsed += handle.GetKernelTime();
                        }
                    }
                }
                // use captured value, other wise getting warning
                // "lambda capture is not used" since this variable is only used in assert.
                (void)should_allocated_wrw_buffer2;
                assert((should_allocated_wrw_buffer2 && data_ctx.workSpace != nullptr) ||
                       !(should_allocated_wrw_buffer2 && data_ctx.workSpace == nullptr));
                if(data_ctx.workSpace)
                {
                    sh_conv_ptr2->SetWorkSpacePointer(argument_ptr.get(), data_ctx.workSpace);
                }

                auto invoker_ptr = sh_conv_ptr2->MakeInvokerPointer();
                {
                    WorkAroundHipEventProfiler prf(handle);
                    MIOPEN_LOG_I2("kernel_name = " << kernel_id2);
                    invoker_ptr->Run(argument_ptr.get(), {handle.GetStream(), false});
                }

                if(handle.IsProfilingEnabled())
                {
                    elapsed += handle.GetKernelTime();

                    // Kernel logging for CK kernels
                    if(IsLoggingKernel())
                    {
                        AddKernelToJsonAccumulator(kernel_id2, elapsed, false);
                    }
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(elapsed);
                }
            };
        };
        result.workspace_sz = GetWorkspaceSizeLayoutTransformConv(problem, ck_ws_size);
#endif
        return result;
    }
    else
    {
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
        result.invoker_factory = [kernel_id_   = kernel_id,
                                  split_k_     = split_k,
                                  ck_args_     = CKArgsType{problem},
                                  sh_conv_ptr_ = std::shared_ptr{std::move(*ptr_iter)}](
                                     const std::vector<Kernel>&) mutable {
            return [kernel_id2   = kernel_id_,
                    split_k2     = split_k_,
                    ck_args2     = std::move(ck_args_),
                    sh_conv_ptr2 = std::move(sh_conv_ptr_)](
                       const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                const auto& data_ctx = primitive_parameters.CastTo<CastType>();

                std::unique_ptr<ck::tensor_operation::device::BaseArgument> argument_ptr =
                    MakeNHWCCKArgPtr<IsSplitKNeeded<DeviceOpType>(),
                                     std::decay_t<decltype(*sh_conv_ptr2)>,
                                     CKArgsType,
                                     CastType>(sh_conv_ptr2, ck_args2, data_ctx, split_k2);

                auto invoker_ptr = sh_conv_ptr2->MakeInvokerPointer();

                // Zero out the buffer for output data since it won't always write all output
                // values.
                float elapsed = 0.0f;
                if constexpr(std::is_same_v<CastType, miopen::conv::DataInvokeParams> &&
                             ZeroOutputs)
                {
                    ZeroOutTensor(handle, data_ctx.tensors.outDesc, data_ctx.tensors.out);

                    if(handle.IsProfilingEnabled())
                    {
                        elapsed += handle.GetKernelTime();
                    }
                }

                {
                    WorkAroundHipEventProfiler prf(handle);
                    MIOPEN_LOG_I2("kernel_name = " << kernel_id2);
                    invoker_ptr->Run(argument_ptr.get(), {handle.GetStream(), false});
                }

                if(handle.IsProfilingEnabled())
                {
                    elapsed += handle.GetKernelTime();

                    // Kernel logging for CK kernels
                    if(IsLoggingKernel())
                    {
                        AddKernelToJsonAccumulator(kernel_id2, elapsed, false);
                    }

                    handle.ResetKernelTime();
                    handle.AccumKernelTime(elapsed);
                }
            };
        };
#endif
        return result;
    }
}

template <int ND, bool ZeroOutputs, typename DeviceOpType, typename CKArgsType, typename CastType>
ConvSolution InitInvokerFactoryFwdNCHW(const ExecutionContext& ctx,
                                       const miopen::conv::ProblemDescription& problem,
                                       const std::string& kernel_id)
{

    static_assert(ND == 2 || ND == 3, "Num Dimensions must be 2 or 3");

    using Input1 = internal::CKTransposeInputOp<ND, internal::ConvOperandTag::Input>;
    using Input2 = internal::CKTransposeInputOp<ND, internal::ConvOperandTag::Weights>;
    using Output = internal::CKTransposeOutputOp<ND, internal::ConvOperandTag::Output>;

    return InitInvokerFactoryNCHW<ZeroOutputs, DeviceOpType, CKArgsType, CastType>(
        ctx, problem, kernel_id, Input1{}, Input2{}, Output{});
}

template <int ND, bool ZeroOutputs, typename DeviceOpType, typename CKArgsType, typename CastType>
ConvSolution InitInvokerFactoryBwdNCHW(const ExecutionContext& ctx,
                                       const miopen::conv::ProblemDescription& problem,
                                       const std::string& kernel_id)
{

    static_assert(ND == 2 || ND == 3, "Num Dimensions must be 2 or 3");

    using Input1 = internal::CKTransposeInputOp<ND, internal::ConvOperandTag::Output>;
    using Input2 = internal::CKTransposeInputOp<ND, internal::ConvOperandTag::Weights>;
    using Output = internal::CKTransposeOutputOp<ND, internal::ConvOperandTag::Input>;

    return InitInvokerFactoryNCHW<ZeroOutputs, DeviceOpType, CKArgsType, CastType>(
        ctx, problem, kernel_id, Input1{}, Input2{}, Output{});
}

template <int ND, bool ZeroOutputs, typename DeviceOpType, typename CKArgsType, typename CastType>
ConvSolution InitInvokerFactoryWrwNCHW(const ExecutionContext& ctx,
                                       const miopen::conv::ProblemDescription& problem,
                                       const std::string& kernel_id)
{
    static_assert(ND == 2 || ND == 3, "Num Dimensions must be 2 or 3");

    using Input1 = internal::CKTransposeInputOp<ND, internal::ConvOperandTag::Input>;
    using Input2 = internal::CKTransposeInputOp<ND, internal::ConvOperandTag::Output>;
    using Output = internal::CKTransposeOutputOp<ND, internal::ConvOperandTag::Weights>;

    return InitInvokerFactoryNCHW<ZeroOutputs, DeviceOpType, CKArgsType, CastType>(
        ctx, problem, kernel_id, Input1{}, Input2{}, Output{});
}

template <typename InvokerFactoryMakerNCHW, typename InvokerFactoryMakerNHWC>
ConvSolution
MakeSolutionGroupConvImplicitGemmXdlops(const miopen::conv::ProblemDescription& problem,
                                        InvokerFactoryMakerNCHW&& invoker_factory_maker_ncdhw,
                                        InvokerFactoryMakerNHWC&& invoker_factory_maker_ndhwc,
                                        const bool use_tf32 = false)
{

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    if(problem.IsLayoutDefault())
    {
        switch(problem.GetInDataType())
        {
        case miopenInt8: return invoker_factory_maker_ncdhw(int8_t{}, int8_t{});
        case miopenHalf: return invoker_factory_maker_ncdhw(ck::half_t{}, ck::half_t{});
        case miopenFloat:
            if(use_tf32)
                return invoker_factory_maker_ncdhw(float{}, ck::tf32_t{});
            else
                return invoker_factory_maker_ncdhw(float{}, float{});
        case miopenBFloat16: return invoker_factory_maker_ncdhw(ck::bhalf_t{}, ck::bhalf_t{});
        case miopenInt64:
        case miopenInt32:
        case miopenDouble:
        case miopenFloat8_fnuz:
        case miopenBFloat8_fnuz:
        default:
            MIOPEN_THROW(miopenStatusInternalError,
                         "3DGroupConvolutionImplicitGemmXdlops operation not implemented for this "
                         "data type");
        }
    }
    else if(problem.IsLayoutNHWC())
    {
        switch(problem.GetInDataType())
        {
        case miopenInt8: return invoker_factory_maker_ndhwc(int8_t{}, int8_t{});
        case miopenHalf: return invoker_factory_maker_ndhwc(ck::half_t{}, ck::half_t{});
        case miopenFloat:
            if(use_tf32)
                return invoker_factory_maker_ndhwc(float{}, ck::tf32_t{});
            else
                return invoker_factory_maker_ndhwc(float{}, float{});
        case miopenBFloat16: return invoker_factory_maker_ndhwc(ck::bhalf_t{}, ck::bhalf_t{});
        case miopenInt64:
        case miopenInt32:
        case miopenDouble:
        case miopenFloat8_fnuz:
        case miopenBFloat8_fnuz:
        default:
            MIOPEN_THROW(miopenStatusInternalError,
                         "3DGroupConvolutionImplicitGemmXdlops operation not implemented for this "
                         "data type");
        }
    }
    else
    {
        MIOPEN_THROW(
            miopenStatusInternalError,
            "3DGroupConvolutionImplicitGemmXdlops operation not implemented for this data type");
    }
#else
    return {};
#endif
}

/// \todo This check is probably no longer needed, as it was likely related to static_ck or
/// legacy_ck, and was copy-pasted into solvers that use the modern CK.
static inline bool IsIndexRangeLargeEnough(const miopen::conv::ProblemDescription& problem)
{
    // composable kernel use int32_t for memory offset, which covers 2GB of memory maximum
    const std::size_t max_index_range = std::size_t(2) * 1024 * 1024 * 1024;

    return problem.GetInSize() < max_index_range && problem.GetWeightsSize() < max_index_range &&
           problem.GetOutSize() < max_index_range;
}

} // namespace solver
} // namespace miopen
