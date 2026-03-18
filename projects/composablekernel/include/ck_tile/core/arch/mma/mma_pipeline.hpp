// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"

#include "amdgcn_mma.hpp"
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
    constexpr operator bool() const { return mFlags != toType(MmaPipelineOptionFlag::NONE); }

    private:
    Type mFlags;
    static constexpr Type toType(MmaPipelineOptionFlag f) { return static_cast<Type>(f); }
};

// TODO: c++20: use MmaPipelineOptionFlags directly
template <MmaPipelineOptionFlags::Type Flags_, typename Derived>
struct MmaPipelineBase
{
    static constexpr auto Flags = MmaPipelineOptionFlags(Flags_);
    // TODO: Implement those cases
    static_assert(!(Flags & MmaPipelineOptionFlag::C_TRANSPOSE), "Flag not yet implemented");

    private:
    template <typename DstT, typename SrcT>
    CK_TILE_DEVICE static auto formatBuffer(SrcT&& inputBuffer)
    {
        // TODO: Implement formatting logic as needed.
        // This is intended to convert input fragments to the native vector types
        // required by the BlockWiseMma operation for iteration
        static_assert(sizeof(DstT) == sizeof(std::remove_reference_t<SrcT>),
                      "Size mismatch in formatBuffer");

        using QualifiedDstT =
            std::conditional_t<std::is_const_v<std::remove_reference_t<SrcT>>, DstT const, DstT>;

        return reinterpret_cast<QualifiedDstT&>(inputBuffer);
    }

    protected:
    template <MmaPipelineOptionFlag Flag>
    constexpr CK_TILE_DEVICE static bool hasFlag()
    {
        return Flags & Flag;
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

    public:
    template <typename VecTA, typename VecTB, typename VecTC>
    CK_TILE_DEVICE static decltype(auto) exec(VecTA&& a, VecTB&& b, VecTC&& accum)
    {
        // TODO: c++20: Call template functions with MmaPipelineOptionFlags directly
        auto pre = Derived::template preApply<Flags_>(
            std::forward<VecTA>(a), std::forward<VecTB>(b), std::forward<VecTC>(accum));
        Derived::execImpl(pre);
        return Derived::template postApply<Flags_>(std::move(pre));
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
