// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {

enum struct MaskingSpecialization
{
    MaskDisabled,
    MaskOutUpperTriangle
};

#ifndef __HIPCC_RTC__
inline std::string getMaskingSpecializationString(const MaskingSpecialization& s)
{
    switch(s)
    {
    case MaskingSpecialization::MaskDisabled: return "MaskDisabled";
    case MaskingSpecialization::MaskOutUpperTriangle: return "MaskOutUpperTriangle";
    default: return "Unrecognized specialization!";
    }
}
#endif

struct MaskDisabledPredicate
{
    __host__ __device__ constexpr bool operator()(index_t /*m*/, index_t /*n*/) const
    {
        return false;
    };

    __host__ __device__ constexpr bool
        IsTileSkippable(index_t /*m*/, index_t /*n*/, index_t /*m_tile*/, index_t /*n_tile*/) const
    {
        return false;
    }
};

struct MaskOutUpperTrianglePredicate
{
    __host__ __device__ constexpr bool operator()(index_t m, index_t n) const { return n > m; }

    __host__ __device__ constexpr bool
    IsTileSkippable(index_t m, index_t n, index_t m_tile, index_t /*n_tile*/) const
    {
        return operator()(m + m_tile - 1, n);
    }
};

// to track the points which need to be set to -inf on C0
// Note: no need to reset M padding value, because they will not be stored out.
template <typename MaskOutPredicate>
struct C0MatrixMask_impl
{
    __host__ __device__ constexpr C0MatrixMask_impl(index_t NRaw)
        : NRaw_(NRaw), predicate_(MaskOutPredicate{})
    {
    }

    __host__ __device__ constexpr bool IsNOutOfBound(/*index_t m, */ index_t n) const
    {
        return n >= NRaw_;
    }

    __host__ __device__ constexpr bool IsMaskedElement(index_t m, index_t n) const
    {
        return predicate_(m, n) || IsNOutOfBound(n);
    }

    __host__ __device__ constexpr bool
    IsTileSkippable(index_t m, index_t n, index_t m_tile, index_t n_tile) const
    {
        return predicate_.IsTileSkippable(m, n, m_tile, n_tile);
    }

    private:
    // index_t MRaw_;
    index_t NRaw_;
    MaskOutPredicate predicate_;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
