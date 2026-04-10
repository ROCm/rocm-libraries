// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_grouped_conv_common.hpp"
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/solver/ck_utility_common.hpp>
#include "implicitgemm_ck_util.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

// ---------------------------------------------------------------------------
// CKArgsSplitK — CRTP base for BWD and WRW CKArgs.
//
// Contains all shared members, the constructor (dimension extraction, NHWC
// layout stride handling), and the split-k support methods (IsSupportedBy,
// IsSupportedBySplitK, GetCKSplitkWorkspaceSize).
//
// Derived classes provide only their direction-specific MakeArgPtr overloads.
// FWD has its own CKArgs (no split-k, no NHWC, simpler IsSupportedBy).
// ---------------------------------------------------------------------------

template <typename Derived>
struct CKArgsSplitK
{
    using ProblemDescription = miopen::conv::ProblemDescription;

    CKArgsSplitK(const ProblemDescription& problem)
    {
        using miopen::solver::ProblemInterpreter;

        auto d          = ExtractConvDims(problem);
        G               = d.G;
        N               = d.N;
        K1              = d.K1;
        C1              = d.C1;
        C               = d.C;
        K               = d.K;
        Hi              = d.Hi;
        Wi              = d.Wi;
        Ho              = d.Ho;
        Wo              = d.Wo;
        Y               = d.Y;
        X               = d.X;
        data_type       = ProblemInterpreter::GetOutputDataType(problem);
        alpha_beta_case = ProblemInterpreter::GetAlphaBetaCase(problem);

        input  = {G, N, C, Hi, Wi};
        output = {G, N, K, Ho, Wo};
        weight = {G, K, C, Y, X};

        if(problem.IsLayoutNHWC())
        {
            auto copy_strides = [](const auto& src, auto& dst) {
                assert(dst.size() == (src.size() + 1));
                std::copy(src.begin(), src.end(), dst.begin() + 1);
            };
            copy_strides(problem.GetIn().GetStrides(), in_strides);
            copy_strides(problem.GetOut().GetStrides(), out_strides);
            copy_strides(problem.GetWeights().GetStrides(), wei_strides);

            // On a backward pass, problem.GetIn() means y(or out),
            // and problem.GetOut means x(or in)
            std::swap(in_strides, out_strides);

            in_strides[0]  = C;
            out_strides[0] = K;
            wei_strides[0] = K * wei_strides[1];
        }
        else
        {
            assert(problem.IsLayoutDefault());
            in_strides  = {C, Hi * Wi * G * C, 1, Wi * G * C, G * C};
            out_strides = {K, Ho * Wo * G * K, 1, Wo * G * K, G * K};
            wei_strides = {K * Y * X * C, Y * X * C, 1, X * C, C};
        }

        strides  = {ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                    ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        dilation = {ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                    ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding = {ProblemInterpreter::GetInputLeftPadH(problem),
                    ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding = {ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                    ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
    }

    CKArgsSplitK(const CKArgsSplitK&)            = default;
    CKArgsSplitK(CKArgsSplitK&&)                 = default;
    CKArgsSplitK& operator=(const CKArgsSplitK&) = default;

    template <typename ConvPtr>
    bool IsSupportedBy(const ConvPtr& conv_ptr) const
    {
        auto arg_ptr = derived().MakeArgPtr(conv_ptr, nullptr, nullptr, nullptr, 1.0f, 0.0f, 1);
        auto workspace_size = conv_ptr->GetWorkSpaceSize(arg_ptr.get());
        if(workspace_size != 0)
            conv_ptr->SetWorkSpacePointer(arg_ptr.get(), &workspace_size);
        return conv_ptr->IsSupportedArgument(arg_ptr.get());
    }

    template <typename ConvPtr>
    bool IsSupportedBySplitK(const ConvPtr& conv_ptr, int split_k) const
    {
        auto arg_ptr =
            derived().MakeArgPtr(conv_ptr, nullptr, nullptr, nullptr, 1.0f, 0.0f, split_k);
        auto workspace_size = conv_ptr->GetWorkSpaceSize(arg_ptr.get());
        if(workspace_size != 0)
            conv_ptr->SetWorkSpacePointer(arg_ptr.get(), &workspace_size);
        return conv_ptr->IsSupportedArgument(arg_ptr.get());
    }

    template <typename ConvPtr>
    std::size_t GetCKSplitkWorkspaceSize(const ConvPtr& conv_ptr, int split_k) const
    {
        auto arg_ptr =
            derived().MakeArgPtr(conv_ptr, nullptr, nullptr, nullptr, 1.0f, 0.0f, split_k);
        return conv_ptr->GetWorkSpaceSize(arg_ptr.get());
    }

    int G;
    int N;
    int K;
    int C;
    int C1;
    int K1;
    int Hi;
    int Wi;
    int Ho;
    int Wo;
    int Y;
    int X;
    miopenDataType_t data_type;
    miopenAlphaBetaCase_t alpha_beta_case;
    std::array<ck::index_t, 5> input;
    std::array<ck::index_t, 5> in_strides;
    std::array<ck::index_t, 5> output;
    std::array<ck::index_t, 5> out_strides;
    std::array<ck::index_t, 5> weight;
    std::array<ck::index_t, 5> wei_strides;
    std::array<ck::index_t, 2> strides;
    std::array<ck::index_t, 2> dilation;
    std::array<ck::index_t, 2> lPadding;
    std::array<ck::index_t, 2> rPadding;

private:
    const Derived& derived() const { return static_cast<const Derived&>(*this); }
};

// ---------------------------------------------------------------------------
// Template helpers — parameterized on DeviceOpPtrs and CKArgsType.
// Each direction file instantiates these with its DeviceOpPtrs and CKArgs.
// ---------------------------------------------------------------------------

template <template <typename, typename> class DeviceOpPtrs, typename CKArgsType, typename DataType>
bool CheckCKApplicabilityCommon(const miopen::conv::ProblemDescription& problem, bool use_tf32)
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        if(use_tf32 &&
           miopen::solver::IsCKApplicable<DeviceOpPtrs<DataType, ck::tf32_t>, CKArgsType>(problem))
        {
            return true;
        }
    }
    return miopen::solver::IsCKApplicable<DeviceOpPtrs<DataType, DataType>, CKArgsType>(problem);
}

template <template <typename, typename> class DeviceOpPtrs, typename CKArgsType, typename DataType>
std::vector<std::string> FillValidKernelsCommon(const miopen::conv::ProblemDescription& problem,
                                                bool use_tf32)
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        if(use_tf32)
        {
            return miopen::solver::FillValidKernelsIDs<DeviceOpPtrs<DataType, ck::tf32_t>,
                                                       CKArgsType>(problem);
        }
    }
    return miopen::solver::FillValidKernelsIDs<DeviceOpPtrs<DataType, DataType>, CKArgsType>(
        problem);
}

template <template <typename, typename> class DeviceOpPtrs, typename CKArgsType, typename DataType>
bool CheckIsArgSupportedCommon(const miopen::conv::ProblemDescription& problem,
                               const std::string& kernel_id,
                               bool use_tf32)
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        if(use_tf32 &&
           miopen::solver::IsCKArgsSupported<DeviceOpPtrs<DataType, ck::tf32_t>, CKArgsType>(
               problem, kernel_id))
        {
            return true;
        }
    }
    return miopen::solver::IsCKArgsSupported<DeviceOpPtrs<DataType, DataType>, CKArgsType>(
        problem, kernel_id);
}

template <template <typename, typename> class DeviceOpPtrs, typename CKArgsType, typename DataType>
std::size_t GetWorkspaceSizeCommon(const miopen::conv::ProblemDescription& problem, bool use_tf32)
{
    std::size_t ws =
        miopen::solver::GetCKSplitkMaxWorkspaceSize<DeviceOpPtrs<DataType, DataType>, CKArgsType>(
            problem);
    if constexpr(std::is_same_v<DataType, float>)
    {
        if(use_tf32)
        {
            ws = std::max(
                ws,
                miopen::solver::GetCKSplitkMaxWorkspaceSize<DeviceOpPtrs<DataType, ck::tf32_t>,
                                                            CKArgsType>(problem));
        }
    }
    return ws;
}

// ---------------------------------------------------------------------------
// DispatchByDataType — collapses data-type switch blocks into a single call.
// ---------------------------------------------------------------------------

template <typename Fn>
auto DispatchByDataType(miopenDataType_t dtype, Fn&& fn)
{
    static_assert(std::is_same_v<decltype(fn(ck::half_t{})), decltype(fn(float{}))> &&
                      std::is_same_v<decltype(fn(ck::bhalf_t{})), decltype(fn(float{}))> &&
                      std::is_same_v<decltype(fn(int8_t{})), decltype(fn(float{}))>,
                  "DispatchByDataType requires Fn to return the same type for all data types");
    switch(dtype)
    {
    case miopenHalf: return fn(ck::half_t{});
    case miopenBFloat16: return fn(ck::bhalf_t{});
    case miopenInt8: return fn(int8_t{});
    default: return fn(float{});
    }
}
