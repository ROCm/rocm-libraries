// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifndef __HIPCC_RTC__
#include <hip/hip_fp16.h>
#endif
#include <type_traits>

namespace hip_kernel_provider
{

// used by batch norm functions.
template <typename T, int N>
struct MappedVectorType
{
    static_assert(false, "there is no specialization for this T & N combination.");
};

template <typename Vec>
struct MappedVectorInfo;

// NOLINTBEGIN(bugprone-macro-parentheses)
#define DEFINE_VECTOR_MAPPING(ScalarType, N)                                \
    template <>                                                             \
    struct MappedVectorType<ScalarType, N>                                  \
    {                                                                       \
        using type = ScalarType __attribute__((ext_vector_type(N)));        \
    };                                                                      \
    template <>                                                             \
    struct MappedVectorInfo<ScalarType __attribute__((ext_vector_type(N)))> \
    {                                                                       \
        using UnderlyingType = ScalarType;                                  \
        static constexpr size_t SIZE = N;                                   \
    };

#define DEFINE_SCALAR_MAPPING(ScalarType)  \
    template <>                            \
    struct MappedVectorType<ScalarType, 1> \
    {                                      \
        using type = ScalarType;           \
    };                                     \
    template <>                            \
    struct MappedVectorInfo<ScalarType>    \
    {                                      \
        using UnderlyingType = ScalarType; \
        static constexpr size_t SIZE = 1;  \
    };
// NOLINTEND(bugprone-macro-parentheses)

DEFINE_SCALAR_MAPPING(double)
DEFINE_SCALAR_MAPPING(float)
DEFINE_SCALAR_MAPPING(_Float16)
DEFINE_SCALAR_MAPPING(__bf16)
DEFINE_SCALAR_MAPPING(int)
DEFINE_SCALAR_MAPPING(unsigned int)

DEFINE_VECTOR_MAPPING(float, 2)
DEFINE_VECTOR_MAPPING(float, 4)

DEFINE_VECTOR_MAPPING(_Float16, 2)
DEFINE_VECTOR_MAPPING(_Float16, 4)
DEFINE_VECTOR_MAPPING(_Float16, 8)

DEFINE_VECTOR_MAPPING(__bf16, 2)
DEFINE_VECTOR_MAPPING(__bf16, 4)
DEFINE_VECTOR_MAPPING(__bf16, 8)

DEFINE_VECTOR_MAPPING(int, 2)
DEFINE_VECTOR_MAPPING(int, 4)

DEFINE_VECTOR_MAPPING(unsigned int, 2)
DEFINE_VECTOR_MAPPING(unsigned int, 4)

// The following overloads with __half are needed due to a regression
// in the current implementation of the RNNHiddenStateUpdate kernel
// Moreover, __half is defined as a struct, thus the attribute
// ext_vector_type, which is used here extensively, will fail

template <>
struct MappedVectorType<__half, 1>
{
    using type = _Float16;
};

template <>
struct MappedVectorInfo<__half>
{
    using UnderlyingType = _Float16;
    static constexpr size_t SIZE = 1;
};

template <>
struct MappedVectorType<__half, 2>
{
    using type = _Float16 __attribute__((ext_vector_type(2)));
};

template <>
struct MappedVectorType<__half, 4>
{
    using type = _Float16 __attribute__((ext_vector_type(4)));
};

namespace detail
{

template <typename OutType, typename InType>
__forceinline__ __device__ __host__ OutType scalarcast(InType in)
{
    if constexpr(std::is_same_v<OutType, InType>)
    {
        return in;
    }
    else
    {
        return static_cast<OutType>(in);
    }
}

template <typename MappedVectorType, typename T>
__forceinline__ __device__ __host__ MappedVectorType broadcast(const T val)
{
    using VectorInfo = MappedVectorInfo<MappedVectorType>;
    MappedVectorType retval;
    auto* retvalPtr = reinterpret_cast<typename VectorInfo::UnderlyingType*>(&retval);
    for(auto i = 0; i < VectorInfo::size; ++i)
    {
        retvalPtr[i] = detail::scalarcast<typename VectorInfo::UnderlyingType>(val);
    }
    return retval;
};

} // namespace detail

template <typename OutType, typename InType>
__forceinline__ __device__ __host__ OutType cast(InType input)
{
    using InTypeInfo = MappedVectorInfo<InType>;
    using OutTypeInfo = MappedVectorInfo<OutType>;

    constexpr auto IN_SIZE = InTypeInfo::SIZE;
    constexpr auto OUT_SIZE = OutTypeInfo::SIZE;

    if constexpr(IN_SIZE == OUT_SIZE && OUT_SIZE == 4)
    {
        return OutType{detail::scalarcast<typename OutTypeInfo::UnderlyingType>(input.x),
                       detail::scalarcast<typename OutTypeInfo::UnderlyingType>(input.y),
                       detail::scalarcast<typename OutTypeInfo::UnderlyingType>(input.z),
                       detail::scalarcast<typename OutTypeInfo::UnderlyingType>(input.w)};
    }
    else if constexpr(IN_SIZE == OUT_SIZE && OUT_SIZE == 2)
    {
        return OutType{detail::scalarcast<typename OutTypeInfo::UnderlyingType>(input.x),
                       detail::scalarcast<typename OutTypeInfo::UnderlyingType>(input.y)};
    }
    else if constexpr(IN_SIZE == OUT_SIZE && OUT_SIZE == 1)
    {
        return detail::scalarcast<typename OutTypeInfo::UnderlyingType>(input);
    }
    else if constexpr(IN_SIZE == 1 && OUT_SIZE > 1)
    {
        return detail::broadcast<OutType>(input);
    }
    else
    {
        static_assert(false, "Unsupported type cast.");
    }
}

} // namespace hip_kernel_provider
