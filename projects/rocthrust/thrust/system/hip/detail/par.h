/******************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019-2025, Advanced Micro Devices, Inc.  All rights reserved.
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

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/detail/allocator_aware_execution_policy.h>
#include <thrust/detail/dependencies_aware_execution_policy.h>
#include <thrust/system/hip/detail/execution_policy.h>
#include <thrust/system/hip/detail/util.h>

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{

template <class Derived>
struct execute_on_stream_base : execution_policy<Derived>
{
private:
  hipStream_t stream;

public:
  THRUST_EXEC_CHECK_DISABLE
  THRUST_HOST_DEVICE execute_on_stream_base(hipStream_t stream_ = default_stream())
      : stream(stream_)
  {}

  THRUST_HIP_RUNTIME_FUNCTION
  Derived on(hipStream_t const& s) const
  {
    Derived result = derived_cast(*this);
    result.stream  = s;
    return result;
  }

private:
  friend THRUST_HOST_DEVICE hipStream_t get_stream(const execute_on_stream_base& exec)
  {
    return exec.stream;
  }
};

template <class Derived>
struct execute_on_stream_nosync_base : execution_policy<Derived>
{
private:
  hipStream_t stream;

public:
  THRUST_HOST_DEVICE execute_on_stream_nosync_base(hipStream_t stream_ = default_stream())
      : stream(stream_)
  {}

  THRUST_HIP_RUNTIME_FUNCTION Derived on(hipStream_t const& s) const
  {
    Derived result = derived_cast(*this);
    result.stream  = s;
    return result;
  }

private:
  friend THRUST_HOST_DEVICE hipStream_t get_stream(const execute_on_stream_nosync_base& exec)
  {
    return exec.stream;
  }

  friend THRUST_HOST_DEVICE bool must_perform_optional_stream_synchronization(const execute_on_stream_nosync_base&)
  {
    return false;
  }
};

struct execute_on_stream : execute_on_stream_base<execute_on_stream>
{
  using base_t = execute_on_stream_base<execute_on_stream>;

  THRUST_HOST_DEVICE execute_on_stream()
      : base_t(){};
  THRUST_HOST_DEVICE execute_on_stream(hipStream_t stream)
      : base_t(stream){};
};

struct execute_on_stream_nosync : execute_on_stream_nosync_base<execute_on_stream_nosync>
{
  using base_t = execute_on_stream_nosync_base<execute_on_stream_nosync>;

  THRUST_HOST_DEVICE execute_on_stream_nosync()
      : base_t(){};
  THRUST_HOST_DEVICE execute_on_stream_nosync(hipStream_t stream)
      : base_t(stream){};
};

THRUST_SUPPRESS_DEPRECATED_PUSH
struct par_t
    : execution_policy<par_t>
    , thrust::detail::allocator_aware_execution_policy<execute_on_stream_base>
    , thrust::detail::dependencies_aware_execution_policy<execute_on_stream_base>
{
  using base_t = execution_policy<par_t>;

  THRUST_HOST_DEVICE constexpr par_t()
      : base_t()
  {}

  using stream_attachment_type = execute_on_stream;

  THRUST_HIP_RUNTIME_FUNCTION
  stream_attachment_type on(hipStream_t const& stream) const
  {
    return execute_on_stream(stream);
  }
};
THRUST_SUPPRESS_DEPRECATED_POP

THRUST_SUPPRESS_DEPRECATED_PUSH
struct par_nosync_t
    : execution_policy<par_nosync_t>
    , thrust::detail::allocator_aware_execution_policy<execute_on_stream_nosync_base>
    , thrust::detail::dependencies_aware_execution_policy<execute_on_stream_nosync_base>
{
  using base_t = execution_policy<par_nosync_t>;

  THRUST_HOST_DEVICE constexpr par_nosync_t()
      : base_t()
  {}

  using stream_attachment_type = execute_on_stream_nosync;

  THRUST_HIP_RUNTIME_FUNCTION
  stream_attachment_type on(hipStream_t const& stream) const
  {
    return execute_on_stream_nosync(stream);
  }

private:
  // this function is defined to allow non-blocking calls on the default_stream() with thrust::hip::par_nosync
  // without explicitly using thrust::hip::par_nosync.on(default_stream())
  friend THRUST_HOST_DEVICE bool must_perform_optional_stream_synchronization(const par_nosync_t&)
  {
    return false;
  }
};

template <class Derived>
struct execute_on_stream_deterministic_base : execute_on_stream_base<Derived>
{
private:
  using base_t = execute_on_stream_base<Derived>;

public:
  THRUST_HOST_DEVICE execute_on_stream_deterministic_base()
      : base_t(){};
  THRUST_HOST_DEVICE execute_on_stream_deterministic_base(hipStream_t stream)
      : base_t(stream){};

private:
  friend THRUST_HOST_DEVICE integral_constant<bool, false>
  allows_nondeterminism(const execute_on_stream_deterministic_base&)
  {
    return {};
  }
};

struct execute_on_stream_deterministic : execute_on_stream_deterministic_base<execute_on_stream_deterministic>
{
  using base_t = execute_on_stream_deterministic_base<execute_on_stream_deterministic>;

  THRUST_HOST_DEVICE execute_on_stream_deterministic()
      : base_t(){};
  THRUST_HOST_DEVICE execute_on_stream_deterministic(hipStream_t stream)
      : base_t(stream){};
};

struct par_det_t
    : execution_policy<par_det_t>
    , thrust::detail::allocator_aware_execution_policy<execute_on_stream_deterministic_base>
    , thrust::detail::dependencies_aware_execution_policy<execute_on_stream_deterministic_base>
{
  using base_t = execution_policy<par_det_t>;

  THRUST_HOST_DEVICE constexpr par_det_t()
      : base_t()
  {}

  using stream_attachment_type = execute_on_stream_deterministic;

  THRUST_HIP_RUNTIME_FUNCTION
  stream_attachment_type on(hipStream_t const& stream) const
  {
    return execute_on_stream_deterministic(stream);
  }

private:
  friend THRUST_HOST_DEVICE integral_constant<bool, false> allows_nondeterminism(const par_det_t&)
  {
    return {};
  }
};

template <class Derived>
struct execute_on_stream_nosync_deterministic_base : execute_on_stream_nosync_base<Derived>
{
private:
  using base_t = execute_on_stream_nosync_base<Derived>;

public:
  THRUST_HOST_DEVICE execute_on_stream_nosync_deterministic_base()
      : base_t(){};
  THRUST_HOST_DEVICE execute_on_stream_nosync_deterministic_base(hipStream_t stream)
      : base_t(stream){};

private:
  friend THRUST_HOST_DEVICE integral_constant<bool, false>
  allows_nondeterminism(const execute_on_stream_nosync_deterministic_base&)
  {
    return {};
  }
};

struct execute_on_stream_nosync_deterministic
    : execute_on_stream_nosync_deterministic_base<execute_on_stream_nosync_deterministic>
{
  using base_t = execute_on_stream_nosync_deterministic_base<execute_on_stream_nosync_deterministic>;

  THRUST_HOST_DEVICE execute_on_stream_nosync_deterministic()
      : base_t(){};
  THRUST_HOST_DEVICE execute_on_stream_nosync_deterministic(hipStream_t stream)
      : base_t(stream){};
};

struct par_det_nosync_t
    : execution_policy<par_det_nosync_t>
    , thrust::detail::allocator_aware_execution_policy<execute_on_stream_nosync_deterministic_base>
    , thrust::detail::dependencies_aware_execution_policy<execute_on_stream_nosync_deterministic_base>
{
  using base_t = execution_policy<par_det_nosync_t>;

  THRUST_HOST_DEVICE constexpr par_det_nosync_t()
      : base_t()
  {}

  using stream_attachment_type = execute_on_stream_nosync_deterministic;

  THRUST_HIP_RUNTIME_FUNCTION
  stream_attachment_type on(hipStream_t const& stream) const
  {
    return execute_on_stream_nosync_deterministic(stream);
  }

private:
  friend THRUST_HOST_DEVICE integral_constant<bool, false> allows_nondeterminism(const par_det_nosync_t&)
  {
    return {};
  }
};
THRUST_SUPPRESS_DEPRECATED_POP

THRUST_INLINE_CONSTANT par_t par;

/*! \p thrust::hip::par_nosync is a parallel execution policy targeting Thrust's HIP device backend.
 *  Similar to \p thrust::hip::par it allows execution of Thrust algorithms in a specific HIP stream.
 *
 *  \p thrust::hip::par_nosync indicates that an algorithm is free to avoid any synchronization of the
 *  associated stream that is not strictly required for correctness. Additionally, algorithms may return
 *  before the corresponding kernels are completed, similar to asynchronous kernel launches via <<< >>> syntax.
 *  The user must take care to perform explicit synchronization if necessary.
 *
 *  The following code snippet demonstrates how to use \p thrust::hip::par_nosync :
 *
 *  \code
 *    #include <thrust/device_vector.h>
 *    #include <thrust/for_each.h>
 *    #include <thrust/execution_policy.h>
 *
 *    struct IncFunctor{
 *        __host__ __device__
 *        void operator()(std::size_t& x){ x = x + 1; };
 *    };
 *
 *    int main(){
 *        std::size_t N = 1000000;
 *        thrust::device_vector<std::size_t> d_vec(N);
 *
 *        hipStream_t stream;
 *        hipStreamCreate(&stream);
 *        auto nosync_policy = thrust::hip::par_nosync.on(stream);
 *
 *        thrust::for_each(nosync_policy, d_vec.begin(), d_vec.end(), IncFunctor{});
 *        thrust::for_each(nosync_policy, d_vec.begin(), d_vec.end(), IncFunctor{});
 *        thrust::for_each(nosync_policy, d_vec.begin(), d_vec.end(), IncFunctor{});
 *
 *        //for_each may return before completion. Could do other cpu work in the meantime
 *        // ...
 *
 *        //Wait for the completion of all for_each kernels
 *        hipStreamSynchronize(stream);
 *
 *        std::size_t x = thrust::reduce(nosync_policy, d_vec.begin(), d_vec.end());
 *        //Currently, this synchronization is not necessary. reduce will still perform
 *        //implicit synchronization to transfer the reduced value to the host to return it.
 *        hipStreamSynchronize(stream);
 *        hipStreamDestroy(stream);
 *    }
 *  \endcode
 *
 */
THRUST_INLINE_CONSTANT par_nosync_t par_nosync;
THRUST_INLINE_CONSTANT par_det_t par_det;
THRUST_INLINE_CONSTANT par_det_nosync_t par_det_nosync;
} // namespace hip_rocprim

namespace system
{
namespace hip
{
using thrust::hip_rocprim::par;
using thrust::hip_rocprim::par_det;
using thrust::hip_rocprim::par_det_nosync;
using thrust::hip_rocprim::par_nosync;
namespace detail
{
using thrust::hip_rocprim::par_det_nosync_t;
using thrust::hip_rocprim::par_det_t;
using thrust::hip_rocprim::par_nosync_t;
using thrust::hip_rocprim::par_t;
} // namespace detail
} // namespace hip
} // namespace system

namespace hip
{
using thrust::hip_rocprim::par;
using thrust::hip_rocprim::par_det;
using thrust::hip_rocprim::par_det_nosync;
using thrust::hip_rocprim::par_nosync;
} // namespace hip

THRUST_NAMESPACE_END
