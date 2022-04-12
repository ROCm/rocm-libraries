/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2020, Advanced Micro Devices, Inc.  All rights reserved.
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

#include "../util_type.hpp"

BEGIN_HIPCUB_NAMESPACE

struct Equality
{
    template<class T>
    HIPCUB_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a == b;
    }
};

struct Inequality
{
    template<class T>
    HIPCUB_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a != b;
    }
};

template <class EqualityOp>
struct InequalityWrapper
{
    EqualityOp op;

    HIPCUB_HOST_DEVICE inline
    InequalityWrapper(EqualityOp op) : op(op) {}

    template<class T>
    HIPCUB_HOST_DEVICE inline
    bool operator()(const T &a, const T &b)
    {
        return !op(a, b);
    }
};

struct Sum
{
    template<class T>
    HIPCUB_HOST_DEVICE inline
    constexpr T operator()(const T &a, const T &b) const
    {
        return a + b;
    }
};

struct Difference
{
    template <class T>
    HIPCUB_HOST_DEVICE inline
    constexpr T operator()(const T &a, const T &b) const
    {
        return a - b;
    }
};

struct Division
{
    template <class T>
    HIPCUB_HOST_DEVICE inline
    constexpr T operator()(const T &a, const T &b) const
    {
        return a / b;
    }
};

struct Max
{
    template<class T>
    HIPCUB_HOST_DEVICE inline
    constexpr T operator()(const T &a, const T &b) const
    {
        return a < b ? b : a;
    }
};

struct Min
{
    template<class T>
    HIPCUB_HOST_DEVICE inline
    constexpr T operator()(const T &a, const T &b) const
    {
        return a < b ? a : b;
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
    template <typename A>
    HIPCUB_HOST_DEVICE inline
    B operator()(const A &a) const
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

    template <typename T, typename U>
    HIPCUB_DEVICE auto
    operator()(T &&t, U &&u) -> decltype(binary_op(std::forward<U>(u),
                                                    std::forward<T>(t)))
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

namespace detail
{

// CUB uses value_type of OutputIteratorT (if not void) as a type of intermediate results in reduce,
// for example:
//
// /// The output value type
// typedef typename If<(Equals<typename std::iterator_traits<OutputIteratorT>::value_type, void>::VALUE),  // OutputT =  (if output iterator's value type is void) ?
//     typename std::iterator_traits<InputIteratorT>::value_type,                                          // ... then the input iterator's value type,
//     typename std::iterator_traits<OutputIteratorT>::value_type>::Type OutputT;                          // ... else the output iterator's value type
//
// rocPRIM (as well as Thrust) uses result type of BinaryFunction instead (if not void):
//
// using input_type = typename std::iterator_traits<InputIterator>::value_type;
// using result_type = typename ::rocprim::detail::match_result_type<
//     input_type, BinaryFunction
// >::type;
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
    using input_type = typename std::iterator_traits<InputIteratorT>::value_type;
    using output_type = typename std::iterator_traits<OutputIteratorT>::value_type;
    using result_type =
        typename std::conditional<
            std::is_void<output_type>::value, input_type, output_type
        >::type;

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

} // end detail namespace

END_HIPCUB_NAMESPACE

#endif // HIBCUB_ROCPRIM_THREAD_THREAD_OPERATORS_HPP_
