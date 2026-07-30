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
// ToCKIndexArray — narrow a long_index_t array to a ck::index_t (int32) array.
//
// Used on the sub-INT_MAX BWD/WRW MakeArgPtr path, where CK's int32
// MakeArgumentPointer overload (e.g. device_grouped_conv_bwd_data_multiple_d.hpp)
// accepts ck::index_t length / stride arrays only. For those shapes the
// narrowing is exact. Large-tensor (>INT_MAX) shapes never reach this helper:
// MakeArgPtr detects a large-tensor CK instance (IsLargeTensorCKInstance) and
// binds CK's int64 long_index_t overload with the un-narrowed members instead.
// The assert below therefore guards the contract -- if a >INT_MAX value is ever
// narrowed here, the large-tensor branch was bypassed and the result would be
// silently wrong.
// ---------------------------------------------------------------------------
template <typename T, std::size_t N>
constexpr std::array<ck::index_t, N> ToCKIndexArray(const std::array<T, N>& src)
{
    std::array<ck::index_t, N> dst{};
    for(std::size_t i = 0; i < N; ++i)
    {
        dst[i] = static_cast<ck::index_t>(src[i]);
        assert(static_cast<T>(dst[i]) == src[i] &&
               "ToCKIndexArray narrowed a value > INT_MAX -- "
               "RequiresLargeTensorCKInstance filter contract was bypassed");
    }
    return dst;
}

// ---------------------------------------------------------------------------
// NarrowedCKArrays3D / NarrowedCKArrays2D — bundles of int32-narrowed
// length/stride arrays handed to CK's int32 MakeArgumentPointer overload on
// the sub-INT_MAX path. Large-tensor (>INT_MAX) shapes bypass these bundles
// and bind CK's int64 long_index_t overload directly (see ToCKIndexArray).
//
// These bundles MUST be stored as members of the owning CKArgs (not as
// function-local temporaries), because CK's MakeArgumentPointer captures
// references to the array elements into the returned Argument object. If
// the bundle goes out of scope before IsSupportedArgument runs, CK reads
// freed stack memory (caught by ASAN as stack-use-after-scope).
// ---------------------------------------------------------------------------
struct NarrowedCKArrays3D
{
    std::array<ck::index_t, 6> in_l;
    std::array<ck::index_t, 6> in_s;
    std::array<ck::index_t, 6> out_l;
    std::array<ck::index_t, 6> out_s;
    std::array<ck::index_t, 6> wei_l;
    std::array<ck::index_t, 6> wei_s;
    std::array<ck::index_t, 3> filter_strides;
    std::array<ck::index_t, 3> filter_dilations;
    std::array<ck::index_t, 3> lPadding;
    std::array<ck::index_t, 3> rPadding;
};

struct NarrowedCKArrays2D
{
    std::array<ck::index_t, 5> in_l;
    std::array<ck::index_t, 5> in_s;
    std::array<ck::index_t, 5> out_l;
    std::array<ck::index_t, 5> out_s;
    std::array<ck::index_t, 5> wei_l;
    std::array<ck::index_t, 5> wei_s;
    std::array<ck::index_t, 2> filter_strides;
    std::array<ck::index_t, 2> filter_dilations;
    std::array<ck::index_t, 2> lPadding;
    std::array<ck::index_t, 2> rPadding;
};

// ---------------------------------------------------------------------------
// MakeNarrowedCKArrays — build a NarrowedCKArrays2D/3D bundle by int32-narrowing
// each int64 length/stride array via ToCKIndexArray. The 2D and 3D bundles
// share field names, so a single Bundle-parameterized helper deduplicates the
// identical field mapping across all five narrowing accessors (2D FWD, the
// CKArgsSplitK base, and 3D FWD/BWD/WRW). Callers pass their own member arrays
// positionally, which differ in name (2D uses input/output/weight/strides/
// dilation; 3D uses in_lengths/out_lengths/wei_lengths/filter_strides/
// filter_dilations) but not in meaning or order.
// ---------------------------------------------------------------------------
template <typename Bundle, typename LenArr, typename FilterArr>
Bundle MakeNarrowedCKArrays(const LenArr& in_lengths,
                            const LenArr& in_strides,
                            const LenArr& out_lengths,
                            const LenArr& out_strides,
                            const LenArr& wei_lengths,
                            const LenArr& wei_strides,
                            const FilterArr& filter_strides,
                            const FilterArr& filter_dilations,
                            const FilterArr& lPadding,
                            const FilterArr& rPadding)
{
    return Bundle{
        .in_l             = ToCKIndexArray(in_lengths),
        .in_s             = ToCKIndexArray(in_strides),
        .out_l            = ToCKIndexArray(out_lengths),
        .out_s            = ToCKIndexArray(out_strides),
        .wei_l            = ToCKIndexArray(wei_lengths),
        .wei_s            = ToCKIndexArray(wei_strides),
        .filter_strides   = ToCKIndexArray(filter_strides),
        .filter_dilations = ToCKIndexArray(filter_dilations),
        .lPadding         = ToCKIndexArray(lPadding),
        .rPadding         = ToCKIndexArray(rPadding),
    };
}

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

    // Populate-and-return the narrowed bundle. Lazy so narrowing only runs
    // for kernels that survived the RequiresLargeTensorCKInstance filter --
    // BWD/WRW CKArgs is constructed unconditionally in FillValidKernelsIDs
    // before filtering, so narrowing in the constructor would assert on
    // >INT_MAX shapes even though no kernel is ultimately selected.
    const NarrowedCKArrays2D& GetNarrowedArrays() const
    {
        narrowed = MakeNarrowedCKArrays<NarrowedCKArrays2D>(input,
                                                            in_strides,
                                                            output,
                                                            out_strides,
                                                            weight,
                                                            wei_strides,
                                                            strides,
                                                            dilation,
                                                            lPadding,
                                                            rPadding);
        return narrowed;
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

    // Dim members are int64 (and length/stride arrays use ck::long_index_t)
    // so the NCHW stride builder above (e.g. Hi*Wi*G*C) does not silently
    // overflow on tensors whose contiguous stride exceeds INT_MAX. Argument
    // construction then binds to CK's long_index_t MakeArgumentPointer
    // overload, which is safe only when paired with a large-tensor instance
    // (see implicitgemm_ck_util.hpp::RequiresLargeTensorCKInstance).
    int64_t G;
    int64_t N;
    int64_t K;
    int64_t C;
    int64_t C1;
    int64_t K1;
    int64_t Hi;
    int64_t Wi;
    int64_t Ho;
    int64_t Wo;
    int64_t Y;
    int64_t X;
    miopenDataType_t data_type;
    miopenAlphaBetaCase_t alpha_beta_case;
    std::array<ck::long_index_t, 5> input;
    std::array<ck::long_index_t, 5> in_strides;
    std::array<ck::long_index_t, 5> output;
    std::array<ck::long_index_t, 5> out_strides;
    std::array<ck::long_index_t, 5> weight;
    std::array<ck::long_index_t, 5> wei_strides;
    std::array<ck::long_index_t, 2> strides;
    std::array<ck::long_index_t, 2> dilation;
    std::array<ck::long_index_t, 2> lPadding;
    std::array<ck::long_index_t, 2> rPadding;
    // mutable: populated lazily by GetNarrowedArrays() (const) so derived
    // MakeArgPtr (also const) can hand CK references that outlive the call.
    mutable NarrowedCKArrays2D narrowed;

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

    case miopenFloat:
    case miopenInt32:
    case miopenInt64:
    case miopenDouble:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz: return fn(float{});
    }

    MIOPEN_THROW(miopenStatusInternalError, "Unhandled miopenDataType_t");
}
