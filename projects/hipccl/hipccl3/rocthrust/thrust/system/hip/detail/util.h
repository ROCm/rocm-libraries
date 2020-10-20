/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights meserved.
 *  Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#include <cstdio>
#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_traits.h>
// Not present in rocPRIM
// #include <thrust/system/cuda/detail/cub/util_arch.cuh>
#include <thrust/system/hip/detail/execution_policy.h>
#include <thrust/system/hip/error.h>
#include <thrust/system_error.h>

// Define the value to 0, if you want to disable printf on device side.
#ifndef THRUST_HIP_PRINTF_ENABLED
#define THRUST_HIP_PRINTF_ENABLED 1
#endif

#if THRUST_HIP_PRINTF_ENABLED == 1
  #define THRUST_HIP_PRINTF(text, ...) \
    printf(text, ##__VA_ARGS__)
#else
  #define THRUST_HIP_PRINTF(text, ...)
#endif

THRUST_BEGIN_NS
namespace hip_rocprim
{

__thrust_exec_check_disable__ template <class Policy>
__host__ __device__ hipError_t synchronize(Policy& policy)
{
#if __THRUST_HAS_HIPRT__
    return synchronize_stream(derived_cast(policy));
#else
    THRUST_UNUSED_VAR(policy);
    return hipSuccess;
#endif
}

template <class Derived>
__host__ __device__ hipStream_t stream(execution_policy<Derived>& policy)
{
    return get_stream(derived_cast(policy));
}

template <class Type>
hipError_t THRUST_HIP_HOST_FUNCTION
trivial_copy_from_device(Type* dst, Type const* src, size_t count, hipStream_t stream)
{
    hipError_t status = hipSuccess;
    if(count == 0)
        return status;

    // hipMemcpyWithStream is only supported on rocm 3.1 and above
    #if HIP_VERSION_MAJOR >= 3
    #if HIP_VERSION_MINOR >= 1 || HIP_VERSION_MAJOR >= 4
    status = ::hipMemcpyWithStream(dst, src, sizeof(Type) * count, hipMemcpyDeviceToHost, stream);
    #else
    status = ::hipMemcpyAsync(dst, src, sizeof(Type) * count, hipMemcpyDeviceToHost, stream);
    if(status != hipSuccess)
      return status;
    status = hipStreamSynchronize(stream);
    #endif
    #endif
    return status;
}

template <class Type>
hipError_t THRUST_HIP_HOST_FUNCTION
trivial_copy_to_device(Type* dst, Type const* src, size_t count, hipStream_t stream)
{
    hipError_t status = hipSuccess;
    if(count == 0)
        return status;

    // hipMemcpyWithStream is only supported on rocm 3.1 and above
    #if HIP_VERSION_MAJOR >= 3
    #if HIP_VERSION_MINOR >= 1 || HIP_VERSION_MAJOR >= 4
    status = ::hipMemcpyWithStream(dst, src, sizeof(Type) * count, hipMemcpyHostToDevice, stream);
    #else
    status = ::hipMemcpyAsync(dst, src, sizeof(Type) * count, hipMemcpyHostToDevice, stream);
    if(status != hipSuccess)
      return status;
    status = hipStreamSynchronize(stream);
    #endif
    #endif
    return status;
}

template <class Policy, class Type>
__host__ __device__ hipError_t
trivial_copy_device_to_device(Policy& policy, Type* dst, Type const* src, size_t count)
{
    hipError_t status = hipSuccess;
    if(count == 0)
        return status;

    hipStream_t stream = hip_rocprim::stream(policy);
    //
    status = ::hipMemcpyAsync(dst, src, sizeof(Type) * count, hipMemcpyDeviceToDevice, stream);
    hip_rocprim::synchronize(policy);
    return status;
}

inline void __host__ __device__ terminate()
{
#ifdef __HIP_DEVICE_COMPILE__
    abort();
#else
    std::terminate();
#endif
}

inline void __host__ __device__ throw_on_error(hipError_t status, char const* msg)
{
    if(hipSuccess != status)
    {
#if !defined(__HIP_DEVICE_COMPILE__)
        throw thrust::system_error(status, thrust::hip_category(), msg);
#else
#if __THRUST_HAS_HIPRT__
        printf("Error after %s: %s\n", msg, hipGetErrorString(status));
#else
        THRUST_HIP_PRINTF("Error %d: %s \n", (int)status, msg);
    #if THRUST_HIP_PRINTF_ENABLED == 0
        THRUST_UNUSED_VAR(status);
        THRUST_UNUSED_VAR(msg);
    #endif
#endif
        hip_rocprim::terminate();
#endif
    }
}

// TODO this overload should be removed and messages should be passed.
inline void __host__ __device__ throw_on_error(hipError_t status)
{
    if(hipSuccess != status)
    {
#if !defined(__HIP_DEVICE_COMPILE__)
        throw thrust::system_error(status, thrust::hip_category());
#else
#if __THRUST_HAS_HIPRT__
        printf("Error %s\n", hipGetErrorString(status));
#else
        THRUST_HIP_PRINTF("Error %d \n", (int)status);
    #if THRUST_HIP_PRINTF_ENABLED == 0
        THRUST_UNUSED_VAR(status);
    #endif
#endif
        hip_rocprim::terminate();
#endif
    }
}

template <class ValueType, class InputIt, class UnaryOp>
struct transform_input_iterator_t
{
    typedef transform_input_iterator_t                         self_t;
    typedef typename iterator_traits<InputIt>::difference_type difference_type;
    typedef ValueType                                          value_type;
    typedef void                                               pointer;
    typedef value_type                                         reference;
    typedef std::random_access_iterator_tag                    iterator_category;

    InputIt         input;
    mutable UnaryOp op;

    THRUST_HIP_FUNCTION transform_input_iterator_t(InputIt input, UnaryOp op)
        : input(input)
        , op(op)
    {
    }

#if THRUST_CPP_DIALECT >= 2011
    transform_input_iterator_t(const self_t &) = default;
#endif


    /// Postfix increment
    THRUST_HIP_FUNCTION self_t operator++(int)
    {
        self_t retval = *this;
        ++input;
        return retval;
    }

    /// Prefix increment
    THRUST_HIP_FUNCTION self_t operator++()
    {
        ++input;
        return *this;
    }

    /// Indirection
    THRUST_HIP_FUNCTION reference operator*() const
    {
        typename thrust::iterator_value<InputIt>::type x = *input;
        return op(x);
    }
    /// Indirection
    THRUST_HIP_FUNCTION reference operator*()
    {
        typename thrust::iterator_value<InputIt>::type x = *input;
        return op(x);
    }

    /// Addition
    THRUST_HIP_FUNCTION self_t operator+(difference_type n) const
    {
        return self_t(input + n, op);
    }

    /// Addition assignment
    THRUST_HIP_FUNCTION self_t& operator+=(difference_type n)
    {
        input += n;
        return *this;
    }

    /// Subtraction
    THRUST_HIP_FUNCTION self_t operator-(difference_type n) const
    {
        return self_t(input - n, op);
    }

    /// Subtraction assignment
    THRUST_HIP_FUNCTION self_t& operator-=(difference_type n)
    {
        input -= n;
        return *this;
    }

    /// Distance
    THRUST_HIP_FUNCTION difference_type operator-(self_t other) const
    {
        return input - other.input;
    }

    /// Array subscript
    THRUST_HIP_FUNCTION reference operator[](difference_type n) const
    {
        return op(input[n]);
    }

#if 0
    /// Structure dereference
    THRUST_HIP_FUNCTION pointer operator->()
    {
        return &op(*input_itr);
    }
#endif

    /// Equal to
    THRUST_HIP_FUNCTION bool operator==(const self_t& rhs) const
    {
        return (input == rhs.input);
    }

    /// Not equal to
    THRUST_HIP_FUNCTION bool operator!=(const self_t& rhs) const
    {
        return (input != rhs.input);
    }

#if 0
    /// ostream operator
    friend std::ostream& operator<<(std::ostream& os, const self& itr)
    {
        return os;
    }
#endif
}; // struct transform_input_iterarot_t

template <class ValueType, class InputIt1, class InputIt2, class BinaryOp>
struct transform_pair_of_input_iterators_t
{
    typedef transform_pair_of_input_iterators_t                 self_t;
    typedef typename iterator_traits<InputIt1>::difference_type difference_type;
    typedef ValueType                                           value_type;
    typedef void                                                pointer;
    typedef value_type                                          reference;
    typedef std::random_access_iterator_tag                     iterator_category;

    InputIt1         input1;
    InputIt2         input2;
    mutable BinaryOp op;

    THRUST_HIP_FUNCTION transform_pair_of_input_iterators_t(InputIt1 input1_,
                                                                            InputIt2 input2_,
                                                                            BinaryOp op_)
        : input1(input1_)
        , input2(input2_)
        , op(op_)
    {
    }

    /// Postfix increment
    THRUST_HIP_FUNCTION self_t operator++(int)
    {
        self_t retval = *this;
        ++input1;
        ++input2;
        return retval;
    }

    /// Prefix increment
    THRUST_HIP_FUNCTION self_t operator++()
    {
        ++input1;
        ++input2;
        return *this;
    }

    /// Indirection
    THRUST_HIP_FUNCTION reference operator*() const
    {
        return op(*input1, *input2);
    }
    /// Indirection
    THRUST_HIP_FUNCTION reference operator*()
    {
        return op(*input1, *input2);
    }

    /// Addition
    THRUST_HIP_FUNCTION self_t operator+(difference_type n) const
    {
        return self_t(input1 + n, input2 + n, op);
    }

    /// Addition assignment
    THRUST_HIP_FUNCTION self_t& operator+=(difference_type n)
    {
        input1 += n;
        input2 += n;
        return *this;
    }

    /// Subtraction
    THRUST_HIP_FUNCTION self_t operator-(difference_type n) const
    {
        return self_t(input1 - n, input2 - n, op);
    }

    /// Subtraction assignment
    THRUST_HIP_FUNCTION self_t& operator-=(difference_type n)
    {
        input1 -= n;
        input2 -= n;
        return *this;
    }

    /// Distance
    THRUST_HIP_FUNCTION difference_type operator-(self_t other) const
    {
        return input1 - other.input1;
    }

    /// Array subscript
    THRUST_HIP_FUNCTION reference operator[](difference_type n) const
    {
        return op(input1[n], input2[n]);
    }

    /// Equal to
    THRUST_HIP_FUNCTION bool operator==(const self_t& rhs) const
    {
        return (input1 == rhs.input1) && (input2 == rhs.input2);
    }

    /// Not equal to
    THRUST_HIP_FUNCTION bool operator!=(const self_t& rhs) const
    {
        return (input1 != rhs.input1) || (input2 != rhs.input2);
    }

}; // struct trasnform_pair_of_input_iterators_t

template <class ValueType, class InputIt1, class InputIt2, class InputIt3, class TransformOp>
struct transform_triple_of_input_iterators_t
{
    typedef transform_triple_of_input_iterators_t               self_t;
    typedef typename iterator_traits<InputIt1>::difference_type difference_type;
    typedef ValueType                                           value_type;
    typedef value_type*                                         pointer;
    typedef value_type                                          reference;
    typedef std::random_access_iterator_tag                     iterator_category;

    InputIt1            input1;
    InputIt2            input2;
    InputIt3            input3;
    mutable TransformOp op;

    THRUST_HIP_FUNCTION transform_triple_of_input_iterators_t(InputIt1 input1_,
                                                                              InputIt2 input2_,
                                                                              InputIt3 input3_,
                                                                              TransformOp op_)
        : input1(input1_)
        , input2(input2_)
        , input3(input3_)
        , op(op_)
    {
    }

    /// Postfix increment
    THRUST_HIP_FUNCTION self_t operator++(int)
    {
        self_t retval = *this;
        ++input1;
        ++input2;
        ++input3;
        return retval;
    }

    /// Prefix increment
    THRUST_HIP_FUNCTION self_t operator++()
    {
        ++input1;
        ++input2;
        ++input3;
        return *this;
    }

    /// Indirection
    THRUST_HIP_FUNCTION reference operator*() const
    {
        return op(*input1, *input2, *input3);
    }
    /// Indirection
    THRUST_HIP_FUNCTION reference operator*()
    {
        return op(*input1, *input2, *input3);
    }

    /// Addition
    THRUST_HIP_FUNCTION self_t operator+(difference_type n) const
    {
        return self_t(input1 + n, input2 + n, input3 + n, op);
    }

    /// Addition assignment
    THRUST_HIP_FUNCTION self_t& operator+=(difference_type n)
    {
        input1 += n;
        input2 += n;
        input3 += n;
        return *this;
    }

    /// Subtraction
    THRUST_HIP_FUNCTION self_t operator-(difference_type n) const
    {
        return self_t(input1 - n, input2 - n, input3 - n, op);
    }

    /// Subtraction assignment
    THRUST_HIP_FUNCTION self_t& operator-=(difference_type n)
    {
        input1 -= n;
        input2 -= n;
        input3 -= n;
        return *this;
    }

    /// Distance
    THRUST_HIP_FUNCTION difference_type operator-(self_t other) const
    {
        return input1 - other.input1;
    }

    /// Array subscript
    THRUST_HIP_FUNCTION reference operator[](difference_type n) const
    {
        return op(input1[n], input2[n], input3[n]);
    }

    /// Equal to
    THRUST_HIP_FUNCTION bool operator==(const self_t& rhs) const
    {
        return (input1 == rhs.input1) && (input2 == rhs.input2) && (input3 == rhs.input3);
    }

    /// Not equal to
    THRUST_HIP_FUNCTION bool operator!=(const self_t& rhs) const
    {
        return (input1 != rhs.input1) || (input2 != rhs.input2) || (input3 != rhs.input3);
    }

}; // struct trasnform_triple_of_input_iterators_t

struct identity
{
    template <class T>
    __host__ __device__ T const& operator()(T const& t) const
    {
        return t;
    }

    template <class T>
    __host__ __device__ T& operator()(T& t) const
    {
        return t;
    }
};

template <class ValueType, class OutputIt, class TransformOp = identity>
struct transform_output_iterator_t
{
    struct proxy_reference
    {
    private:
        OutputIt    output;
        TransformOp op;

    public:
        __host__ __device__ proxy_reference(OutputIt const& output_, TransformOp op_)
            : output(output_)
            , op(op_)
        {
        }

        proxy_reference __host__ __device__ operator=(ValueType const& x)
        {
            *output = op(x);
            return *this;
        }
    };

    typedef transform_output_iterator_t                         self_t;
    typedef typename iterator_traits<OutputIt>::difference_type difference_type;
    typedef void                                                value_type;
    typedef proxy_reference                                     reference;
    typedef std::output_iterator_tag                            iterator_category;

    OutputIt    output;
    TransformOp op;

    THRUST_HIP_FUNCTION transform_output_iterator_t(OutputIt output)
        : output(output)
    {
    }

    THRUST_HIP_FUNCTION transform_output_iterator_t(OutputIt    output,
                                                                    TransformOp op)
        : output(output)
        , op(op)
    {
    }

    /// Postfix increment
    THRUST_HIP_FUNCTION self_t operator++(int)
    {
        self_t retval = *this;
        ++output;
        return retval;
    }

    /// Prefix increment
    THRUST_HIP_FUNCTION self_t operator++()
    {
        ++output;
        return *this;
    }

    /// Indirection
    THRUST_HIP_FUNCTION reference operator*() const
    {
        return proxy_reference(output, op);
    }
    /// Indirection
    THRUST_HIP_FUNCTION reference operator*()
    {
        return proxy_reference(output, op);
    }

    /// Addition
    THRUST_HIP_FUNCTION self_t operator+(difference_type n) const
    {
        return self_t(output + n, op);
    }

    /// Addition assignment
    THRUST_HIP_FUNCTION self_t& operator+=(difference_type n)
    {
        output += n;
        return *this;
    }

    /// Subtraction
    THRUST_HIP_FUNCTION self_t operator-(difference_type n) const
    {
        return self_t(output - n, op);
    }

    /// Subtraction assignment
    THRUST_HIP_FUNCTION self_t& operator-=(difference_type n)
    {
        output -= n;
        return *this;
    }

    /// Distance
    THRUST_HIP_FUNCTION difference_type operator-(self_t other) const
    {
        return output - other.output;
    }

    /// Array subscript
    THRUST_HIP_FUNCTION reference operator[](difference_type n) const
    {
        return *(output + n);
    }

    /// Equal to
    THRUST_HIP_FUNCTION bool operator==(const self_t& rhs) const
    {
        return (output == rhs.output);
    }

    /// Not equal to
    THRUST_HIP_FUNCTION bool operator!=(const self_t& rhs) const
    {
        return (output != rhs.output);
    }
}; // struct transform_output_iterator_

template <class T, T VALUE>
struct static_integer_iterator
{
    typedef static_integer_iterator         self_t;
    typedef int                             difference_type;
    typedef T                               value_type;
    typedef T                               reference;
    typedef std::random_access_iterator_tag iterator_category;

    THRUST_HIP_FUNCTION static_integer_iterator() {}

    /// Postfix increment
    THRUST_HIP_FUNCTION self_t operator++(int)
    {
        return *this;
    }

    /// Prefix increment
    THRUST_HIP_FUNCTION self_t operator++()
    {
        return *this;
    }

    /// Indirection
    THRUST_HIP_FUNCTION reference operator*() const
    {
        return VALUE;
    }
    /// Indirection
    THRUST_HIP_FUNCTION reference operator*()
    {
        return VALUE;
    }

    /// Addition
    THRUST_HIP_FUNCTION self_t operator+(difference_type) const
    {
        return self_t();
    }

    /// Addition assignment
    THRUST_HIP_FUNCTION self_t& operator+=(difference_type)
    {
        return *this;
    }

    /// Subtraction
    THRUST_HIP_FUNCTION self_t operator-(difference_type) const
    {
        return self_t();
    }

    /// Subtraction assignment
    THRUST_HIP_FUNCTION self_t& operator-=(difference_type)
    {
        return *this;
    }

    /// Distance
    THRUST_HIP_FUNCTION difference_type operator-(self_t) const
    {
        return 0;
    }

    /// Array subscript
    THRUST_HIP_FUNCTION reference operator[](difference_type) const
    {
        return VALUE;
    }

    /// Equal to
    THRUST_HIP_FUNCTION bool operator==(const self_t&) const
    {
        return true;
    }

    /// Not equal to
    THRUST_HIP_FUNCTION bool operator!=(const self_t&) const
    {
        return false;
    }

}; // struct static_bool_iterator

template <class T>
struct counting_iterator_t
{
    typedef counting_iterator_t             self_t;
    typedef T                               difference_type;
    typedef T                               value_type;
    typedef void                            pointer;
    typedef T                               reference;
    typedef std::random_access_iterator_tag iterator_category;

    T count;

    THRUST_HIP_FUNCTION counting_iterator_t(T count_)
        : count(count_)
    {
    }

    /// Postfix increment
    THRUST_HIP_FUNCTION self_t operator++(int)
    {
        self_t retval = *this;
        ++count;
        return retval;
    }

    /// Prefix increment
    THRUST_HIP_FUNCTION self_t operator++()
    {
        ++count;
        return *this;
    }

    /// Indirection
    THRUST_HIP_FUNCTION reference operator*() const
    {
        return count;
    }

    /// Indirection
    THRUST_HIP_FUNCTION reference operator*()
    {
        return count;
    }

    /// Addition
    THRUST_HIP_FUNCTION self_t operator+(difference_type n) const
    {
        return self_t(count + n);
    }

    /// Addition assignment
    THRUST_HIP_FUNCTION self_t& operator+=(difference_type n)
    {
        count += n;
        return *this;
    }

    /// Subtraction
    THRUST_HIP_FUNCTION self_t operator-(difference_type n) const
    {
        return self_t(count - n);
    }

    /// Subtraction assignment
    THRUST_HIP_FUNCTION self_t& operator-=(difference_type n)
    {
        count -= n;
        return *this;
    }

    /// Distance
    THRUST_HIP_FUNCTION difference_type operator-(self_t other) const
    {
        return count - other.count;
    }

    /// Array subscript
    THRUST_HIP_FUNCTION reference operator[](difference_type n) const
    {
        return count + n;
    }

    /// Equal to
    THRUST_HIP_FUNCTION bool operator==(const self_t& rhs) const
    {
        return (count == rhs.count);
    }

    /// Not equal to
    THRUST_HIP_FUNCTION bool operator!=(const self_t& rhs) const
    {
        return (count != rhs.count);
    }

}; // struct count_iterator_t

} // hip_rocprim
THRUST_END_NS
