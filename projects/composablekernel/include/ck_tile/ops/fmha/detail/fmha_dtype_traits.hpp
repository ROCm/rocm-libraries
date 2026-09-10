// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/numeric/numeric.hpp"
#include "ck_tile/core/numeric/type_convert.hpp"

#include <type_traits>

namespace ck_tile {
namespace fmha {

template <typename T>
struct FmhaDTypeTraits
{
    using StorageType                    = T;
    using AccumulatorType                = float;
    using SoftmaxComputeType             = float;
    using LogExpComputeType              = float;
    using OutputType                     = T;
    static constexpr bool kRequiresScale = false;

    CK_TILE_HOST_DEVICE static constexpr StorageType pack(float value)
    {
        return type_convert<StorageType>(value);
    }

    CK_TILE_HOST_DEVICE static constexpr float unpack(StorageType value)
    {
        return type_convert<float>(value);
    }

    CK_TILE_HOST_DEVICE static constexpr StorageType clamp(float value)
    {
        const float clipped =
            ck_tile::clamp(value, -numeric<float>::infinity(), numeric<float>::infinity());
        return type_convert<StorageType>(clipped);
    }

    static constexpr float kClipMin     = -numeric<float>::infinity();
    static constexpr float kClipMax     = numeric<float>::infinity();
    static constexpr float kLog2Rescale = 1.0f;
};

template <>
struct FmhaDTypeTraits<fp8_t>
{
    using StorageType                    = fp8_t;
    using AccumulatorType                = float;
    using SoftmaxComputeType             = float;
    using LogExpComputeType              = float;
    using OutputType                     = fp8_t;
    static constexpr bool kRequiresScale = true;

    CK_TILE_HOST_DEVICE static constexpr StorageType pack(float value)
    {
        return type_convert<StorageType>(value);
    }

    CK_TILE_HOST_DEVICE static constexpr float unpack(StorageType value)
    {
        return type_convert<float>(value);
    }

    CK_TILE_HOST_DEVICE static constexpr StorageType clamp(float value)
    {
        const float clipped =
            ck_tile::clamp(value, -numeric<float>::infinity(), numeric<float>::infinity());
        return type_convert<StorageType>(clipped);
    }

    static constexpr float kClipMin     = -numeric<float>::infinity();
    static constexpr float kClipMax     = numeric<float>::infinity();
    static constexpr float kLog2Rescale = 1.0f;
};

template <>
struct FmhaDTypeTraits<bf8_t>
{
    using StorageType                    = bf8_t;
    using AccumulatorType                = float;
    using SoftmaxComputeType             = float;
    using LogExpComputeType              = float;
    using OutputType                     = bf8_t;
    static constexpr bool kRequiresScale = true;

    CK_TILE_HOST_DEVICE static constexpr StorageType pack(float value)
    {
        return type_convert<StorageType>(value);
    }

    CK_TILE_HOST_DEVICE static constexpr float unpack(StorageType value)
    {
        return type_convert<float>(value);
    }

    CK_TILE_HOST_DEVICE static constexpr StorageType clamp(float value)
    {
        const float clipped =
            ck_tile::clamp(value, -numeric<float>::infinity(), numeric<float>::infinity());
        return type_convert<StorageType>(clipped);
    }

    static constexpr float kClipMin     = -numeric<float>::infinity();
    static constexpr float kClipMax     = numeric<float>::infinity();
    static constexpr float kLog2Rescale = 1.0f;
};

namespace detail {

template <typename Problem, typename = void>
struct FmhaProblemTraitsSelector
{
    using type = void;
};

template <typename Problem>
struct FmhaProblemTraitsSelector<Problem, std::void_t<typename Problem::FmhaTraits>>
{
    using type = typename Problem::FmhaTraits;
};

} // namespace detail

template <typename Problem>
struct FmhaDefaultProblemTraits
{
    using QDataType = remove_cvref_t<typename Problem::QDataType>;
    using KDataType = remove_cvref_t<typename Problem::KDataType>;
    using VDataType = remove_cvref_t<typename Problem::VDataType>;
    using ODataType = remove_cvref_t<typename Problem::ODataType>;

    using QTraits = FmhaDTypeTraits<QDataType>;
    using KTraits = FmhaDTypeTraits<KDataType>;
    using VTraits = FmhaDTypeTraits<VDataType>;
    using OTraits = FmhaDTypeTraits<ODataType>;

    static constexpr index_t kQElementBytes = sizeof(QDataType);
    static constexpr index_t kKElementBytes = sizeof(KDataType);
    static constexpr index_t kVElementBytes = sizeof(VDataType);
    static constexpr index_t kOElementBytes = sizeof(ODataType);

    static constexpr bool kRequiresFp8Scale = QTraits::kRequiresScale || KTraits::kRequiresScale ||
                                              VTraits::kRequiresScale || OTraits::kRequiresScale;

    CK_TILE_HOST_DEVICE static constexpr QDataType pack_q(float value)
    {
        return QTraits::pack(value);
    }

    CK_TILE_HOST_DEVICE static constexpr KDataType pack_k(float value)
    {
        return KTraits::pack(value);
    }

    CK_TILE_HOST_DEVICE static constexpr VDataType pack_v(float value)
    {
        return VTraits::pack(value);
    }

    CK_TILE_HOST_DEVICE static constexpr ODataType pack_o(float value)
    {
        return OTraits::pack(value);
    }
};

template <typename Problem>
struct FmhaProblemTraits
{
    private:
    using Selected = typename detail::FmhaProblemTraitsSelector<Problem>::type;

    public:
    using type =
        std::conditional_t<std::is_void_v<Selected>, FmhaDefaultProblemTraits<Problem>, Selected>;
};

template <typename Problem>
using FmhaProblemTraitsT = typename FmhaProblemTraits<Problem>::type;

} // namespace fmha
} // namespace ck_tile
