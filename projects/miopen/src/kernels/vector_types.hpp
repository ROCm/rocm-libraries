// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef VECTOR_TYPES_HPP
#define VECTOR_TYPES_HPP

#include "bfloat16_dev.hpp"
#include "miopen_type_traits.hpp"

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif
namespace miopen {

// used by batch norm functions.
template <typename T, int N>
struct mapped_vector_type
{
    static_assert(false, "there is no specialization for this T & N combination.");
};

template <typename Vec>
struct mapped_vector_info;

#define DEFINE_VECTOR_MAPPING(ScalarType, N)                                  \
    template <>                                                               \
    struct mapped_vector_type<ScalarType, N>                                  \
    {                                                                         \
        using type = ScalarType __attribute__((ext_vector_type(N)));          \
    };                                                                        \
    template <>                                                               \
    struct mapped_vector_info<ScalarType __attribute__((ext_vector_type(N)))> \
    {                                                                         \
        using UnderlyingType         = ScalarType;                            \
        static constexpr size_t size = N;                                     \
    };

#define DEFINE_SCALAR_MAPPING(ScalarType)          \
    template <>                                    \
    struct mapped_vector_type<ScalarType, 1>       \
    {                                              \
        using type = ScalarType;                   \
    };                                             \
    template <>                                    \
    struct mapped_vector_info<ScalarType>          \
    {                                              \
        using UnderlyingType         = ScalarType; \
        static constexpr size_t size = 1;          \
    };

DEFINE_SCALAR_MAPPING(double)
DEFINE_SCALAR_MAPPING(float)
DEFINE_SCALAR_MAPPING(_Float16)
DEFINE_SCALAR_MAPPING(ushort)
DEFINE_SCALAR_MAPPING(int)
DEFINE_SCALAR_MAPPING(unsigned int)

DEFINE_VECTOR_MAPPING(float, 2)
DEFINE_VECTOR_MAPPING(float, 4)

DEFINE_VECTOR_MAPPING(_Float16, 2)
DEFINE_VECTOR_MAPPING(_Float16, 4)
DEFINE_VECTOR_MAPPING(_Float16, 8)

DEFINE_VECTOR_MAPPING(ushort, 2)
DEFINE_VECTOR_MAPPING(ushort, 4)
DEFINE_VECTOR_MAPPING(ushort, 8)

DEFINE_VECTOR_MAPPING(int, 2)
DEFINE_VECTOR_MAPPING(int, 4)

DEFINE_VECTOR_MAPPING(unsigned int, 2)
DEFINE_VECTOR_MAPPING(unsigned int, 4)

// The following overloads with __half are needed due to a regression
// in the current implementation of the RNNHiddenStateUpdate kernel
// Moreover, __half is defined as a struct, thus the attribute
// ext_vector_type, which is used here extensively, will fail

template <>
struct mapped_vector_type<__half, 1>
{
    using type = _Float16;
};

template <>
struct mapped_vector_info<__half>
{
    using UnderlyingType         = _Float16;
    static constexpr size_t size = 1;
};

template <>
struct mapped_vector_type<__half, 2>
{
    using type = _Float16 __attribute__((ext_vector_type(2)));
};

template <>
struct mapped_vector_type<__half, 4>
{
    using type = _Float16 __attribute__((ext_vector_type(4)));
};

namespace detail {

template <typename OutType, typename InType>
__forceinline__ __device__ __host__ OutType scalarcast(InType in)
{
    if constexpr(std::is_same<OutType, InType>::value)
    {
        return in;
    }
    else if constexpr(std::is_same<OutType, ushort>::value && std::is_same<InType, float>::value)
    {
        return float_to_bfloat16(in);
    }
    else if constexpr(std::is_same<InType, ushort>::value && std::is_same<OutType, float>::value)
    {
        return bfloat16_to_float(in);
    }
    else
    {
        return static_cast<OutType>(in);
    }
}

template <typename MappedVectorType, typename T>
__forceinline__ __device__ __host__ MappedVectorType broadcast(const T val)
{
    using VectorInfo = mapped_vector_info<MappedVectorType>;
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
    using InTypeInfo  = mapped_vector_info<InType>;
    using OutTypeInfo = mapped_vector_info<OutType>;

    constexpr auto inSize  = InTypeInfo::size;
    constexpr auto outSize = OutTypeInfo::size;

    if constexpr(inSize == outSize && outSize == 4)
    {
        return OutType{detail::scalarcast<typename OutTypeInfo::UnderlyingType>(input.x),
                       detail::scalarcast<typename OutTypeInfo::UnderlyingType>(input.y),
                       detail::scalarcast<typename OutTypeInfo::UnderlyingType>(input.z),
                       detail::scalarcast<typename OutTypeInfo::UnderlyingType>(input.w)};
    }
    else if constexpr(inSize == outSize && outSize == 2)
    {
        return OutType{detail::scalarcast<typename OutTypeInfo::UnderlyingType>(input.x),
                       detail::scalarcast<typename OutTypeInfo::UnderlyingType>(input.y)};
    }
    else if constexpr(inSize == outSize && outSize == 1)
    {
        return detail::scalarcast<typename OutTypeInfo::UnderlyingType>(input);
    }
    else if constexpr(inSize == 1 && outSize > 1)
    {
        return detail::broadcast<OutType>(input);
    }
    else
    {
        static_assert(false, "Unsupported type cast.");
    }
}

} // namespace miopen

#endif
