// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"

#include "amdgcn_mma.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "mma_selector.hpp"
#include "mma_traits.hpp"
#include "mma_transforms.hpp"

namespace ck_tile::core::arch::mma {

enum struct MmaPipelineOptionFlag
{
    NONE        = 0x0,
    C_TRANSPOSE = 0x1,
    COMPRESS_A  = 0x2,
};

struct MmaPipelineOptionFlags
{
    using Type = std::underlying_type<MmaPipelineOptionFlag>::type;

    explicit constexpr MmaPipelineOptionFlags() : mFlags(0) {}
    explicit constexpr MmaPipelineOptionFlags(Type value) : mFlags(value) {}
    constexpr MmaPipelineOptionFlags(MmaPipelineOptionFlag singleFlag) : mFlags(toType(singleFlag))
    {
    }
    constexpr MmaPipelineOptionFlags(const MmaPipelineOptionFlags& original)
        : mFlags(original.mFlags)
    {
    }

    constexpr MmaPipelineOptionFlags& operator|=(MmaPipelineOptionFlag addValue)
    {
        mFlags |= toType(addValue);
        return *this;
    }
    constexpr MmaPipelineOptionFlags operator|(MmaPipelineOptionFlag addValue) const
    {
        MmaPipelineOptionFlags result(*this);
        result |= addValue;
        return result;
    }
    constexpr MmaPipelineOptionFlags& operator&=(MmaPipelineOptionFlag maskValue)
    {
        mFlags &= toType(maskValue);
        return *this;
    }
    constexpr MmaPipelineOptionFlags operator&(MmaPipelineOptionFlag maskValue) const
    {
        MmaPipelineOptionFlags result(*this);
        result &= maskValue;
        return result;
    }
    constexpr MmaPipelineOptionFlags operator~() const
    {
        MmaPipelineOptionFlags result(*this);
        result.mFlags = ~result.mFlags;
        return result;
    }
    constexpr bool testFlag(MmaPipelineOptionFlag flag) const
    {
        return (flag == MmaPipelineOptionFlag::NONE) ? mFlags == toType(flag) : *this & flag;
    }
    constexpr operator bool() const { return mFlags != toType(MmaPipelineOptionFlag::NONE); }
    constexpr bool operator==(Type rhs) const { return mFlags == rhs; }

    private:
    Type mFlags;
    static constexpr Type toType(MmaPipelineOptionFlag f) { return static_cast<Type>(f); }
};

constexpr bool operator==(MmaPipelineOptionFlags::Type lhs, const MmaPipelineOptionFlags& rhs)
{
    return rhs == lhs;
}

namespace {
template <typename T>
struct is_tuple : std::false_type
{
};

template <typename... Args>
struct is_tuple<std::tuple<Args...>> : std::true_type
{
};
} // namespace

// TODO: c++20: use MmaPipelineOptionFlags directly
template <MmaPipelineOptionFlags::Type Flags_, typename Derived>
struct MmaPipelineBase
{
    static constexpr auto Flags = MmaPipelineOptionFlags(Flags_);

    private:
    // Helper to reconstruct a tuple with formatted first element and remaining elements
    template <typename DstT, typename SrcT, std::size_t... Is>
    CK_TILE_DEVICE static auto formatBufferTupleImpl(SrcT&& inputTuple, std::index_sequence<Is...>)
    {
        auto&& first_elem = std::get<0>(std::forward<SrcT>(inputTuple));
        return std::make_tuple(formatBuffer<DstT>(std::forward<decltype(first_elem)>(first_elem)),
                               std::get<Is + 1>(std::forward<SrcT>(inputTuple))...);
    }

    template <typename DstT, typename SrcT>
    CK_TILE_DEVICE static auto formatBuffer(SrcT&& inputBuffer)
    {
        using DecayedSrcT = ck_tile::remove_cvref_t<SrcT>;

        // If SrcT is a tuple, extract the first element (the vector) and format it
        // while preserving all remaining elements (metadata)
        if constexpr(is_tuple<DecayedSrcT>::value)
        {
            // Create index sequence for all remaining elements (skip first)
            constexpr std::size_t tuple_size = std::tuple_size_v<DecayedSrcT>;
            return formatBufferTupleImpl<DstT>(std::forward<SrcT>(inputBuffer),
                                               std::make_index_sequence<tuple_size - 1>{});
        }
        else if constexpr(std::is_array_v<DecayedSrcT> || std::is_pointer_v<DecayedSrcT>)
        {
            return std::forward<SrcT>(inputBuffer);
        }
        else
        {
            static_assert(sizeof(DstT) == sizeof(DecayedSrcT), "Size mismatch in formatBuffer");

            using QualifiedDstT =
                std::conditional_t<std::is_const_v<DecayedSrcT>, DstT const, DstT>;

            return reinterpret_cast<QualifiedDstT&>(inputBuffer);
        }
    }

    protected:
    template <MmaPipelineOptionFlag Flag>
    constexpr CK_TILE_DEVICE static bool hasFlag()
    {
        return Flags.testFlag(Flag);
    }

    template <typename DstT, typename Transform, typename... Args>
    CK_TILE_DEVICE static auto preApplyTransform(Args&&... args)
    {
        return formatBuffer<DstT>(Transform::exec(std::forward<Args>(args)...));
    }

    template <typename DstT, typename Transform, typename... Args>
    CK_TILE_DEVICE static auto postApplyTransform(Args&&... args)
    {
        return Transform::exec(formatBuffer<DstT>(std::forward<Args>(args)...));
    }

    template <typename ATransformInputs, typename BTransformInputs, typename CTransformInputs>
    CK_TILE_DEVICE static decltype(auto)
    applyTransformsToInputs(ATransformInputs&& a, BTransformInputs&& b, CTransformInputs&& accum)
    {
        using InternalAVecT = typename Derived::InternalAVecT;
        using InternalBVecT = typename Derived::InternalBVecT;
        using InternalCVecT = typename Derived::InternalCVecT;

        using ATransform = typename Derived::ATransform;
        using BTransform = typename Derived::BTransform;
        using CTransform = typename Derived::CTransform;

        return std::make_tuple(
            preApplyTransform<InternalAVecT, ATransform>(std::forward<ATransformInputs>(a)),
            preApplyTransform<InternalBVecT, BTransform>(std::forward<BTransformInputs>(b)),
            preApplyTransform<InternalCVecT, CTransform>(std::forward<CTransformInputs>(accum)));
    }

    template <typename ATransformResult, typename BTransformResult, typename CTransformResult>
    CK_TILE_DEVICE static decltype(auto)
    applyTransformToOutput(std::tuple<ATransformResult, BTransformResult, CTransformResult>&& vecs)
    {
        auto&& [a_result, b_result, c_result] = vecs;
        static_assert(!is_tuple<decltype(c_result)>::value,
                      "If CTransform returns more than the vector, update this function.");

        using CVecT      = typename Derived::CVecType;
        using DTransform = typename Derived::DTransform;
        return postApplyTransform<CVecT, DTransform>(c_result);
    }

    public:
    template <typename VecTA, typename VecTB, typename VecTC>
    CK_TILE_DEVICE static decltype(auto) exec(VecTA&& a, VecTB&& b, VecTC&& accum)
    {
        if constexpr(MmaOpTraits<typename Derived::MmaOp>::IsSupported)
        {
            auto transformed_inputs = applyTransformsToInputs(
                hasFlag<MmaPipelineOptionFlag::C_TRANSPOSE>() ? std::forward<VecTB>(b)
                                                              : std::forward<VecTA>(a),
                hasFlag<MmaPipelineOptionFlag::C_TRANSPOSE>() ? std::forward<VecTA>(a)
                                                              : std::forward<VecTB>(b),
                std::forward<VecTC>(accum));

            Derived::execImpl(transformed_inputs);

            return applyTransformToOutput(std::move(transformed_inputs));
        }
        else
        {
            // Return the unsupported exec. This should print a runtime warning. (amdgcn_mma.hpp)
            // Code should not reach here, but HOST/DEVICE compile passes are
            // weirdly intertwined and instead of having constexpr in the calling
            // site (tests) we do this. See also changes by this commit.
            return Derived::MmaOp::exec({}, {}, {});
        }
    }
};

#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

#include <concepts>

/**
 * @concept MmaPipelineI
 * @brief  Expresses the meta-data interface required for a CRTP MmaPipeline.
 */
template <typename Derived, MmaPipelineOptionFlags::Type Flags>
concept MmaPipelineInterface = std::derived_from<Derived, MmaPipelineBase<Flags, Derived>>;

#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

} // namespace ck_tile::core::arch::mma
