/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2025, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef HIBCUB_ROCPRIM_THREAD_THREAD_OPERATORS_HPP_
#define HIBCUB_ROCPRIM_THREAD_THREAD_OPERATORS_HPP_

#include "../../../config.hpp"

#include "../util_macro.hpp"
#include "../util_type.hpp"

#include <rocprim/type_traits.hpp> // IWYU pragma: export
#include <rocprim/type_traits_functions.hpp>

#include <hip/hip_bf16.h>

BEGIN_HIPCUB_NAMESPACE

// TODO: this is deprecated in cub, we should also mark this as deprecated when we have libhipcxx
struct Equality
{
    template<class T, class U>
    HIPCUB_HOST_DEVICE inline constexpr bool operator()(T&& t, U&& u) const
    {
        return std::forward<T>(t) == std::forward<U>(u);
    }
};

// TODO: this is deprecated in cub, we should also mark this as deprecated when we have libhipcxx
struct Inequality
{
    template<class T, class U>
    HIPCUB_HOST_DEVICE inline constexpr bool operator()(T&& t, U&& u) const
    {
        return std::forward<T>(t) != std::forward<U>(u);
    }
};

// TODO: this is deprecated in cub, we should also mark this as deprecated when we have libhipcxx
template <class EqualityOp>
struct InequalityWrapper
{
    EqualityOp op;

    HIPCUB_HOST_DEVICE inline
    InequalityWrapper(EqualityOp op) : op(op) {}

    template<class T, class U>
    HIPCUB_HOST_DEVICE inline bool operator()(T&& t, U&& u)
    {
        return !op(std::forward<T>(t), std::forward<U>(u));
    }
};

// TODO: this is deprecated in cub, we should also mark this as deprecated when we have libhipcxx
struct Sum
{
    template<class T, class U>
    HIPCUB_HOST_DEVICE inline constexpr auto operator()(T&& t, U&& u) const -> decltype(auto)
    {
        return std::forward<T>(t) + std::forward<U>(u);
    }
};

// TODO: this is deprecated in cub, we should also mark this as deprecated when we have libhipcxx
struct Difference
{
    template<class T, class U>
    HIPCUB_HOST_DEVICE inline constexpr auto operator()(T&& t, U&& u) const -> decltype(auto)
    {
        return std::forward<T>(t) - std::forward<U>(u);
    }
};

// TODO: this is deprecated in cub, we should also mark this as deprecated when we have libhipcxx
struct Division
{
    template<class T, class U>
    HIPCUB_HOST_DEVICE inline constexpr auto operator()(T&& t, U&& u) const -> decltype(auto)
    {
        return std::forward<T>(t) / std::forward<U>(u);
    }
};

// TODO: this is deprecated in cub, we should also mark this as deprecated when we have libhipcxx
struct Max
{
    template<class T, class U>
    HIPCUB_HOST_DEVICE inline constexpr typename std::common_type<T, U>::type
        operator()(T&& t, U&& u) const
    {
        // TODO: change to use hip::std::max after libhipcxx is ready
        return (((u) > (t)) ? (u) : (t));
    }
};

// TODO: this is deprecated in cub, we should also mark this as deprecated when we have libhipcxx
struct Min
{
    template<class T, class U>
    HIPCUB_HOST_DEVICE inline constexpr typename std::common_type<T, U>::type
        operator()(T&& t, U&& u) const
    {
        // TODO: change to use hip::std::min after libhipcxx is ready
        return (((u) < (t)) ? (u) : (t));
    }
};

struct ArgMax
{
    template<
        class Key,
        class Value
    >
    HIPCUB_HOST_DEVICE inline
    constexpr KeyValuePair<Key, Value>
    operator()(const KeyValuePair<Key, Value>& a,
               const KeyValuePair<Key, Value>& b) const
    {
        return ((b.value > a.value) || ((a.value == b.value) && (b.key < a.key))) ? b : a;
    }
};

struct ArgMin
{
    template<
        class Key,
        class Value
    >
    HIPCUB_HOST_DEVICE inline
    constexpr KeyValuePair<Key, Value>
    operator()(const KeyValuePair<Key, Value>& a,
               const KeyValuePair<Key, Value>& b) const
    {
        return ((b.value < a.value) || ((a.value == b.value) && (b.key < a.key))) ? b : a;
    }
};

template <typename B>
struct CastOp
{
    template<typename A>
    HIPCUB_HOST_DEVICE inline B operator()(A&& a) const
    {
        return (B)a;
    }
};

template <typename ScanOp>
class SwizzleScanOp
{
private:
    ScanOp scan_op;

public:
    HIPCUB_HOST_DEVICE inline
    SwizzleScanOp(ScanOp scan_op) : scan_op(scan_op)
    {
    }

    template <typename T>
    HIPCUB_HOST_DEVICE inline
    T operator()(const T &a, const T &b)
    {
      T _a(a);
      T _b(b);

      return scan_op(_b, _a);
    }
};

template <typename ReductionOpT>
struct ReduceBySegmentOp
{
    ReductionOpT op;

    HIPCUB_HOST_DEVICE inline
    ReduceBySegmentOp()
    {
    }

    HIPCUB_HOST_DEVICE inline
    ReduceBySegmentOp(ReductionOpT op) : op(op)
    {
    }

    template <typename KeyValuePairT>
    HIPCUB_HOST_DEVICE inline
    KeyValuePairT operator()(
        const KeyValuePairT &first,
        const KeyValuePairT &second)
    {
        KeyValuePairT retval;
        retval.key = first.key + second.key;
        retval.value = (second.key) ?
                second.value :
                op(first.value, second.value);
        return retval;
    }
};

template <typename ReductionOpT>
struct ReduceByKeyOp
{
    ReductionOpT op;

    HIPCUB_HOST_DEVICE inline
    ReduceByKeyOp()
    {
    }

    HIPCUB_HOST_DEVICE inline
    ReduceByKeyOp(ReductionOpT op) : op(op)
    {
    }

    template <typename KeyValuePairT>
    HIPCUB_HOST_DEVICE inline
    KeyValuePairT operator()(
        const KeyValuePairT &first,
        const KeyValuePairT &second)
    {
        KeyValuePairT retval = second;

        if (first.key == second.key)
        {
            retval.value = op(first.value, retval.value);
        }
        return retval;
    }
};

template <typename BinaryOpT>
struct BinaryFlip
{
    BinaryOpT binary_op;

    HIPCUB_HOST_DEVICE
    explicit BinaryFlip(BinaryOpT binary_op) : binary_op(binary_op)
    {
    }

    template<typename T, typename U>
    HIPCUB_DEVICE auto operator()(T&& t, U&& u) -> decltype(auto)
    {
        return binary_op(std::forward<U>(u), std::forward<T>(t));
    }
};

template <typename BinaryOpT>
HIPCUB_HOST_DEVICE
BinaryFlip<BinaryOpT> MakeBinaryFlip(BinaryOpT binary_op)
{
    return BinaryFlip<BinaryOpT>(binary_op);
}

namespace internal
{

template<typename T>
struct [[deprecated(
    "SIMD intrinsics are currently not supported on HIP, use Min instead.")]] SimdMin
{
    static_assert(false, "Unsupported specialization");
};

template<>
struct [[deprecated(
    "SIMD intrinsics are currently not supported on HIP, use Min instead.")]] SimdMin<int16_t>
{
    using simd_type = int32_t;

    HIPCUB_HOST_DEVICE __forceinline__
    constexpr uint32_t
        operator()(int32_t t, int32_t u) const
    {
        return HIPCUB_MIN(t, u);
    }
};

template<>
struct [[deprecated(
    "SIMD intrinsics are currently not supported on HIP, use Min instead.")]] SimdMin<uint16_t>
{
    using simd_type = uint32_t;

    HIPCUB_HOST_DEVICE __forceinline__
    constexpr uint32_t
        operator()(uint32_t t, uint32_t u) const
    {
        return HIPCUB_MIN(t, u);
    }
};

#if !defined(__HIP_NO_HALF_OPERATORS__)
template<>
struct [[deprecated(
    "SIMD intrinsics are currently not supported on HIP, use Min instead.")]] SimdMin<__half>
{
    using simd_type = __half2;

    // This function is not constexpr because the operator< of __half does not produce a constant expression
    HIPCUB_HOST_DEVICE __forceinline__
    __half2
        operator()(__half2 t, __half2 u) const
    {
        return HIPCUB_MIN(t, u);
    }
};
#endif // !defined(__HIP_NO_HALF_OPERATORS__)

template<>
struct [[deprecated("SIMD intrinsics are currently not supported on HIP, use Min "
                    "instead.")]] SimdMin<__hip_bfloat16>
{
    using simd_type = __hip_bfloat162;

    // This function is not constexpr because the operator< of __half does not produce a constant expression
    HIPCUB_HOST_DEVICE __forceinline__
    __hip_bfloat162
        operator()(__hip_bfloat162 t, __hip_bfloat162 u) const
    {
        return HIPCUB_MIN(t, u);
    }
};

template<typename T>
struct [[deprecated(
    "SIMD intrinsics are currently not supported on HIP, use Max instead.")]] SimdMax
{
    static_assert(false, "Unsupported specialization");
};

template<>
struct [[deprecated(
    "SIMD intrinsics are currently not supported on HIP, use Max instead.")]] SimdMax<int16_t>
{
    using simd_type = int32_t;

    HIPCUB_HOST_DEVICE __forceinline__
    constexpr uint32_t
        operator()(int32_t t, int32_t u) const
    {
        return HIPCUB_MAX(t, u);
    }
};

template<>
struct [[deprecated(
    "SIMD intrinsics are currently not supported on HIP, use Max instead.")]] SimdMax<uint16_t>
{
    using simd_type = uint32_t;

    HIPCUB_HOST_DEVICE __forceinline__
    constexpr uint32_t
        operator()(uint32_t t, uint32_t u) const
    {
        return HIPCUB_MAX(t, u);
    }
};

#if !defined(__HIP_NO_HALF_OPERATORS__)
template<>
struct [[deprecated(
    "SIMD intrinsics are currently not supported on HIP, use Max instead.")]] SimdMax<__half>
{
    using simd_type = __half2;

    HIPCUB_HOST_DEVICE __forceinline__
    __half2
        operator()(__half2 t, __half2 u) const
    {
        return HIPCUB_MAX(t, u);
    }
};
#endif // !defined(__HIP_NO_HALF_OPERATORS__)

template<>
struct [[deprecated("SIMD intrinsics are currently not supported on HIP, use Max "
                    "instead.")]] SimdMax<__hip_bfloat16>
{
    using simd_type = __hip_bfloat162;

    HIPCUB_HOST_DEVICE __forceinline__
    __hip_bfloat162
        operator()(__hip_bfloat162 t, __hip_bfloat162 u) const
    {
        return HIPCUB_MAX(t, u);
    }
};

template<typename T>
struct [[deprecated(
    "SIMD intrinsics are currently not supported on HIP, use Sum instead.")]] SimdSum
{
    static_assert(false, "Unsupported specialization");
};

template<>
struct [[deprecated(
    "SIMD intrinsics are currently not supported on HIP, use Sum instead.")]] SimdSum<int16_t>
{
    using simd_type = int32_t;

    HIPCUB_HOST_DEVICE __forceinline__
    constexpr uint32_t
        operator()(int32_t t, int32_t u) const
    {
        return t + u;
    }
};

template<>
struct [[deprecated(
    "SIMD intrinsics are currently not supported on HIP, use Sum instead.")]] SimdSum<uint16_t>
{
    using simd_type = uint32_t;

    HIPCUB_HOST_DEVICE __forceinline__
    constexpr uint32_t
        operator()(uint32_t t, uint32_t u) const
    {
        return t + u;
    }
};
#if !defined(__HIP_NO_HALF_OPERATORS__)
template<>
struct [[deprecated(
    "SIMD intrinsics are currently not supported on HIP, use Sum instead.")]] SimdSum<__half>
{
    using simd_type = __half2;

    HIPCUB_HOST_DEVICE __forceinline__
    __half2
        operator()(__half2 t, __half2 u) const
    {
        return t + u;
    }
};
#endif // !defined(__HIP_NO_HALF_OPERATORS__)

template<>
struct [[deprecated("SIMD intrinsics are currently not supported on HIP, use Sum "
                    "instead.")]] SimdSum<__hip_bfloat16>
{
    using simd_type = __hip_bfloat162;

    HIPCUB_HOST_DEVICE __forceinline__
    __hip_bfloat162
        operator()(__hip_bfloat162 t, __hip_bfloat162 u) const
    {
        return t + u;
    }
};

// TODO: Mul is not available in hipcub. Use hip::std::multiplies from libhipcxx in the future.

template<typename T>
struct [[deprecated("Warning: SIMD intrinsics are currently not supported on HIP, so "
                    "this operator will multiply 2 input values directly")]] SimdMul
{
    static_assert(false, "Unsupported specialization");
};

template<>
struct [[deprecated("Warning: SIMD intrinsics are currently not supported on HIP, so "
                    "this operator will multiply 2 input values directly")]] SimdMul<int16_t>
{
    using simd_type = int32_t;

    HIPCUB_HOST_DEVICE __forceinline__
    constexpr uint32_t
        operator()(int32_t t, int32_t u) const
    {
        return t * u;
    }
};

template<>
struct [[deprecated("Warning: SIMD intrinsics are currently not supported on HIP, so "
                    "this operator will multiply 2 input values directly")]] SimdMul<uint16_t>
{
    using simd_type = uint32_t;

    HIPCUB_HOST_DEVICE __forceinline__
    constexpr uint32_t
        operator()(uint32_t t, uint32_t u) const
    {
        return t * u;
    }
};

#if !defined(__HIP_NO_HALF_OPERATORS__)
template<>
struct [[deprecated("Warning: SIMD intrinsics are currently not supported on HIP, so "
                    "this operator will multiply 2 input values directly")]] SimdMul<__half>
{
    using simd_type = __half2;

    HIPCUB_HOST_DEVICE __forceinline__
    __half2
        operator()(__half2 t, __half2 u) const
    {
        return t * u;
    }
};
#endif // !defined(__HIP_NO_HALF_OPERATORS__)

template<>
struct [[deprecated("Warning: SIMD intrinsics are currently not supported on HIP, so "
                    "this operator will multiply 2 input values directly")]] SimdMul<__hip_bfloat16>
{
    using simd_type = __hip_bfloat162;

    HIPCUB_HOST_DEVICE __forceinline__
    __hip_bfloat162
        operator()(__hip_bfloat162 t, __hip_bfloat162 u) const
    {
        return t * u;
    }
};

template<typename ReduceOp, typename T>
struct HipOperatorToSimdOperator
{
    static_assert(false, "Unsupported specialization");
};

template<typename T>
struct HipOperatorToSimdOperator<Min, T>
{
    using type      = SimdMin<T>;
    using simd_type = typename type::simd_type;
};

template<typename T>
struct HipOperatorToSimdOperator<Max, T>
{
    using type      = SimdMax<T>;
    using simd_type = typename type::simd_type;
};

template<typename T>
struct HipOperatorToSimdOperator<Sum, T>
{
    using type      = SimdSum<T>;
    using simd_type = typename type::simd_type;
};

// TODO: Mul is not available in hipcub. Use hip::std::multiplies from libhipcxx in the future.

} // namespace internal

namespace detail
{

// Non-void value type.
template<typename IteratorT, typename FallbackT>
using non_void_value_t =
    typename std::conditional<std::is_same<IteratorT, void>::value, FallbackT, IteratorT>::type;

/// Intermediate accumulator type.
template<typename Invokable, typename InputT, typename InitT = InputT>
using accumulator_t = ::rocprim::accumulator_t<Invokable, InputT, InitT>;

// CUB uses value_type of OutputIteratorT (if not void) as a type of intermediate results in segmented reduce,
// for example:
//
// /// The output value type
// using OutputT =
//     typename If<(Equals<typename std::iterator_traits<OutputIteratorT>::value_type,
//                         void>::VALUE),           // OutputT =  (if output iterator's value type is void) ?
//                 typename std::iterator_traits<
//                     InputIteratorT>::value_type, // ... then the input iterator's value type,
//                 typename std::iterator_traits<OutputIteratorT>::value_type>::
//         Type;                                    // ... else the output iterator's value type
//
// rocPRIM (as well as Thrust) uses result type of BinaryFunction instead (if not void):
//
// using input_type = typename std::iterator_traits<InputIterator>::value_type;
// using result_type = ::rocprim::accumulator_t<BinaryFunction, input_type>;
//
// For short -> float using Sum()
// CUB:     float Sum(float, float)
// rocPRIM: short Sum(short, short)
//
// This wrapper allows to have compatibility with CUB in hipCUB.
template<
    class InputIteratorT,
    class OutputIteratorT,
    class BinaryFunction
>
struct convert_result_type_wrapper
{
    using input_type  = typename std::iterator_traits<InputIteratorT>::value_type;
    using output_type = typename std::iterator_traits<OutputIteratorT>::value_type;
    using result_type = non_void_value_t<output_type, input_type>;

    convert_result_type_wrapper(BinaryFunction op) : op(op) {}

    template<class T>
    HIPCUB_HOST_DEVICE inline
    constexpr result_type operator()(const T &a, const T &b) const
    {
        return static_cast<result_type>(op(a, b));
    }

    BinaryFunction op;
};

template<
    class InputIteratorT,
    class OutputIteratorT,
    class BinaryFunction
>
inline
convert_result_type_wrapper<InputIteratorT, OutputIteratorT, BinaryFunction>
convert_result_type(BinaryFunction op)
{
    return convert_result_type_wrapper<InputIteratorT, OutputIteratorT, BinaryFunction>(op);
}

// CUB uses the return type of applying BinaryFunction to an Input and an InitialValue (if not void) as the intermediate result type in reduce.
//
// // The accumulator type
// using AccumT = ::rocprim::accumulator_t<BinaryFunction, InputT, InitT>>;
//
// rocPRIM uses the return type of applying BinaryFunction to two Inputs (if not void) instead.
//
// using AccumT = ::rocprim::accumulator_t<BinaryFunction, InputT, InputT>>;
//
// This wrapper allows to have compatibility with CUB in hipCUB.

template<class BinaryFunction, class InputIteratorT, class InitT>
struct convert_binary_result_type_wrapper
{
    using input_type  = typename std::iterator_traits<InputIteratorT>::value_type;
    using init_type   = InitT;
    using accum_type  = accumulator_t<BinaryFunction, input_type, init_type>;

    convert_binary_result_type_wrapper(BinaryFunction op) : op(op) {}

    template<class T>
    HIPCUB_HOST_DEVICE inline constexpr accum_type operator()(const T& a, const T& b) const
    {
        return static_cast<accum_type>(op(a, b));
    }

    BinaryFunction op;
};

template<class InputIteratorT, class InitT, class BinaryFunction>
inline convert_binary_result_type_wrapper<BinaryFunction, InputIteratorT, InitT>
    convert_binary_result_type(BinaryFunction op)
{
    return convert_binary_result_type_wrapper<BinaryFunction, InputIteratorT, InitT>(op);
}

} // end detail namespace

END_HIPCUB_NAMESPACE

#endif // HIBCUB_ROCPRIM_THREAD_THREAD_OPERATORS_HPP_
