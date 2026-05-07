// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef CK_STATICALLY_INDEXED_ARRAY_HPP
#define CK_STATICALLY_INDEXED_ARRAY_HPP

#include "functional2.hpp"
#include "tuple.hpp"

namespace ck {

namespace detail {
template <typename X, typename Y>
struct tuple_concat;

template <typename... Xs, typename... Ys>
struct tuple_concat<Tuple<Xs...>, Tuple<Ys...>>
{
    using type = Tuple<Xs..., Ys...>;
};

// StaticallyIndexedArrayImpl uses binary split for O(log N) depth
template <typename T, index_t N>
struct StaticallyIndexedArrayImpl
{
    using type =
        typename tuple_concat<typename StaticallyIndexedArrayImpl<T, N / 2>::type,
                              typename StaticallyIndexedArrayImpl<T, N - N / 2>::type>::type;
};

template <typename T>
struct StaticallyIndexedArrayImpl<T, 0>
{
    using type = Tuple<>;
};

template <typename T>
struct StaticallyIndexedArrayImpl<T, 1>
{
    using type = Tuple<T>;
};
} // namespace detail

template <typename T, index_t N>
using StaticallyIndexedArray = typename detail::StaticallyIndexedArrayImpl<T, N>::type;

template <typename X, typename... Xs>
__host__ __device__ constexpr auto make_statically_indexed_array(const X& x, const Xs&... xs)
{
    return StaticallyIndexedArray<X, sizeof...(Xs) + 1>(x, static_cast<X>(xs)...);
}

// make empty StaticallyIndexedArray
template <typename X>
__host__ __device__ constexpr auto make_statically_indexed_array()
{
    return StaticallyIndexedArray<X, 0>();
}

template <typename T, index_t N>
struct StaticallyIndexedArray_v2
{
    __host__ __device__ constexpr StaticallyIndexedArray_v2() = default;

    __host__ __device__ static constexpr index_t Size() { return N; }

    // read access
    template <index_t I>
    __host__ __device__ constexpr const auto& At(Number<I>) const
    {
        static_assert(I < N, "wrong! out of range");

        return data_[I];
    }

    // write access
    template <index_t I>
    __host__ __device__ constexpr auto& At(Number<I>)
    {
        static_assert(I < N, "wrong! out of range");

        return data_[I];
    }

    // read access
    template <index_t I>
    __host__ __device__ constexpr const auto& operator[](Number<I> i) const
    {
        return At(i);
    }

    // write access
    template <index_t I>
    __host__ __device__ constexpr auto& operator()(Number<I> i)
    {
        return At(i);
    }

    __host__ __device__ static constexpr bool IsStaticBuffer() { return true; }

    T data_[N];
};

// static index array with external buffer support, it is used by lane-shared variable
template <typename T, index_t N>
struct StaticallyIndexedArray_v3
{
    __host__ __device__ constexpr StaticallyIndexedArray_v3(T* data) { data_ = data; };

    __host__ __device__ static constexpr index_t Size() { return N; }

    // read access
    template <index_t I>
    __host__ __device__ constexpr const auto& At(Number<I>) const
    {
        static_assert(I < N, "wrong! out of range");

        return data_[I];
    }

    // write access
    template <index_t I>
    __host__ __device__ constexpr auto& At(Number<I>)
    {
        static_assert(I < N, "wrong! out of range");

        return data_[I];
    }

    // read access
    template <index_t I>
    __host__ __device__ constexpr const auto& operator[](Number<I> i) const
    {
        return At(i);
    }

    // write access
    template <index_t I>
    __host__ __device__ constexpr auto& operator()(Number<I> i)
    {
        return At(i);
    }

    __host__ __device__ static constexpr bool IsStaticBuffer() { return true; }

    __host__ __device__ void SwitchBuffer() {}
    T* data_;
};

// static index array with external double buffer support, it is used by lane-shared variable
template <typename T, index_t N>
struct StaticallyIndexedArray_v4
{
    __host__ __device__ constexpr StaticallyIndexedArray_v4(T* data)
    {
        data_   = data;
        offset_ = 0;
    };

    __host__ __device__ static constexpr index_t Size() { return N; }

    // read access
    template <index_t I>
    __host__ __device__ constexpr const auto& At(Number<I>) const
    {
        static_assert(I < N, "wrong! out of range");

        return (data_ + offset_)[I];
    }

    // write access
    template <index_t I>
    __host__ __device__ constexpr auto& At(Number<I>)
    {
        static_assert(I < N, "wrong! out of range");

        return (data_ + offset_)[I];
    }

    // read access
    template <index_t I>
    __host__ __device__ constexpr const auto& operator[](Number<I> i) const
    {
        return At(i);
    }

    // write access
    template <index_t I>
    __host__ __device__ constexpr auto& operator()(Number<I> i)
    {
        return At(i);
    }

    __host__ __device__ static constexpr bool IsStaticBuffer() { return true; }

    // switch double buffer
    __host__ __device__ void SwitchBuffer() { offset_ = (offset_ == 0) ? N : 0; }
    T* data_;
    index_t offset_;
};

// static index array with external buffer support, it is used by lane-shared variable
template <typename T, index_t N, index_t Offset>
struct StaticallyIndexedArray_v5
{
    __host__ __device__ constexpr StaticallyIndexedArray_v5(T* data) { data_ = data; };

    __host__ __device__ static constexpr index_t Size() { return N; }

    // read access
    template <index_t I>
    __host__ __device__ constexpr const auto& At(Number<I>) const
    {
        static_assert(I < N, "wrong! out of range");

        return data_[I + Offset];
    }

    // write access
    template <index_t I>
    __host__ __device__ constexpr auto& At(Number<I>)
    {
        static_assert(I < N, "wrong! out of range");

        return data_[I + Offset];
    }

    // read access
    template <index_t I>
    __host__ __device__ constexpr const auto& operator[](Number<I> i) const
    {
        return At(i);
    }

    // write access
    template <index_t I>
    __host__ __device__ constexpr auto& operator()(Number<I> i)
    {
        return At(i);
    }

    __host__ __device__ static constexpr bool IsStaticBuffer() { return true; }

    __host__ __device__ void SwitchBuffer() {}
    T* data_;
};

} // namespace ck
#endif
