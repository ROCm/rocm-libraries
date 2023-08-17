/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights meserved.
 *  Modifications Copyright© 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <thrust/system/hip/detail/execution_policy.h>
#include <thrust/system/hip/error.h>
#include <thrust/system_error.h>

#include <thrust/system/hip/detail/nv/target.h>

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

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{

inline __host__ __device__ hipStream_t default_stream()
{
  #ifdef HIP_API_PER_THREAD_DEFAULT_STREAM
    return hipStreamPerThread;
  #else
    return hipStreamDefault; // There's not hipStreamLegacy
  #endif
}

template <class Derived>
hipStream_t __host__ __device__
get_stream(execution_policy<Derived>&)
{
    return default_stream();
}

// Fallback implementation of the customization point.
template <class Derived> __host__ __device__
bool must_perform_optional_stream_synchronization(execution_policy<Derived> &)
{
  return true;
}

// Entry point/interface.
template <class Derived> __host__ __device__
bool must_perform_optional_synchronization(execution_policy<Derived> &policy)
{
  return must_perform_optional_stream_synchronization(derived_cast(policy));
}

template <class Derived>
__host__ __device__ hipError_t synchronize_stream(execution_policy<Derived>& policy)
{
  hipError_t result;
  // Can't use #if inside NV_IF_TARGET, use a temp macro to hoist the device
  // instructions out of the target logic.
#if __THRUST_HAS_HIPRT__

#define THRUST_TEMP_DEVICE_CODE result = hipDeviceSynchronize();

#else

#define THRUST_TEMP_DEVICE_CODE result = hipSuccess

#endif

  NV_IF_TARGET(NV_IS_HOST,
               (result = hipStreamSynchronize(stream(policy));),
               (THRUST_UNUSED_VAR(policy); THRUST_TEMP_DEVICE_CODE;));

#undef THRUST_TEMP_DEVICE_CODE

  return result;
}

// Fallback implementation of the customization point.
template <class Derived> __host__ __device__
hipError_t synchronize_stream_optional(execution_policy<Derived> &policy)
{
  hipError_t result;

  if (must_perform_optional_synchronization(policy))
  {
    result = synchronize_stream(policy);
  }
  else
  {
    result = hipSuccess;
  }

  return result;
}

// Entry point/interface.
template <class Policy> __host__ __device__
hipError_t synchronize_optional(Policy &policy)
{
  return synchronize_stream_optional(derived_cast(policy));
}

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
    NV_IF_TARGET(NV_IS_HOST, (std::terminate();), (abort();));
}

inline void __host__ __device__ throw_on_error(hipError_t status, char const* msg)
{
    // Clear the global HIP error state which may have been set by the last
    // call. Otherwise, errors may "leak" to unrelated kernel launches.
    NV_IF_TARGET(NV_IS_HOST, (hipError_t clear_error_status = hipGetLastError(); THRUST_UNUSED_VAR(clear_error_status);));

    if(hipSuccess != status)
    {

        // Can't use #if inside NV_IF_TARGET, use a temp macro to hoist the device
        // instructions out of the target logic.
    #if THRUST_HIP_PRINTF_ENABLED == 0
    #define THRUST_TEMP_DEVICE_CODE                             \
        THRUST_HIP_PRINTF("Error %d :%s \n", (int)status, msg); \
        THRUST_UNUSED_VAR(status);                              \
        THRUST_UNUSED_VAR(msg)
    #else
    #define THRUST_TEMP_DEVICE_CODE THRUST_HIP_PRINTF("Error %d :%s \n", (int)status, msg)
    #endif

            NV_IF_TARGET(NV_IS_HOST,
                        (throw thrust::system_error(status, thrust::hip_category(), msg);),
                        (THRUST_TEMP_DEVICE_CODE; hip_rocprim::terminate();));

#undef THRUST_TEMP_DEVICE_CODE
  }
}

// TODO this overload should be removed and messages should be passed.
inline void __host__ __device__ throw_on_error(hipError_t status)
{
    // Clear the global HIP error state which may have been set by the last
    // call. Otherwise, errors may "leak" to unrelated kernel launches.
    NV_IF_TARGET(NV_IS_HOST, (hipError_t clear_error_status = hipGetLastError(); THRUST_UNUSED_VAR(clear_error_status);));

  if(hipSuccess != status)
  {

        // Can't use #if inside NV_IF_TARGET, use a temp macro to hoist the device
        // instructions out of the target logic.
#if THRUST_HIP_PRINTF_ENABLED == 0
#define THRUST_TEMP_DEVICE_CODE                    \
    THRUST_HIP_PRINTF("Error %d \n", (int)status); \
    THRUST_UNUSED_VAR(status)
#else
#define THRUST_TEMP_DEVICE_CODE THRUST_HIP_PRINTF("Error %d \n", (int)status)
#endif

        NV_IF_TARGET(NV_IS_HOST,
                     (throw thrust::system_error(status, thrust::hip_category());),
                     (THRUST_TEMP_DEVICE_CODE; hip_rocprim::terminate();));

#undef THRUST_TEMP_DEVICE_CODE
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

  // UnaryOp might not be copy assignable, such as when it is a lambda.  Define
  // an explicit copy assignment operator that doesn't try to assign it.
  THRUST_HIP_FUNCTION self_t& operator=(const self_t& o)
  {
    input = o.input;
    return *this;
  }

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

    transform_pair_of_input_iterators_t(const self_t&) = default;

    // BinaryOp might not be copy assignable, such as when it is a lambda.
    // Define an explicit copy assignment operator that doesn't try to assign it.
    self_t& operator=(const self_t& o)
    {
        input1 = o.input1;
        input2 = o.input2;
        return *this;
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
THRUST_NAMESPACE_END
