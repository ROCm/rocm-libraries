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

#include <thrust/detail/allocator_aware_execution_policy.h>
#include <thrust/detail/config.h>
#include <thrust/detail/dependencies_aware_execution_policy.h>
#include <thrust/system/hip/detail/execution_policy.h>
#include <thrust/system/hip/detail/guarded_hip_runtime_api.h>
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
        {
        }

        THRUST_HIP_RUNTIME_FUNCTION
        Derived
        on(hipStream_t const& s) const
        {
            Derived result = derived_cast(*this);
            result.stream  = s;
            return result;
        }

    private:
        friend hipStream_t THRUST_HOST_DEVICE
        get_stream(const execute_on_stream_base& exec)
        {
            return exec.stream;
        }
    };

    struct execute_on_stream : execute_on_stream_base<execute_on_stream>
    {
        using base_t = execute_on_stream_base<execute_on_stream>;

        THRUST_HOST_DEVICE execute_on_stream()
            : base_t() {};
        THRUST_HOST_DEVICE execute_on_stream(hipStream_t stream)
            : base_t(stream) {};
    };

    template <class Derived>
    struct execute_on_stream_nosync_base : execute_on_stream_base<Derived>
    {
    private:
        using base_t = execute_on_stream_base<Derived>;

    public:
        THRUST_HOST_DEVICE execute_on_stream_nosync_base()
            : base_t() {};
        THRUST_HOST_DEVICE execute_on_stream_nosync_base(hipStream_t stream)
            : base_t(stream) {};

    private:
        friend THRUST_HOST_DEVICE
        bool
        must_perform_optional_stream_synchronization(const execute_on_stream_nosync_base &)
        {
            return false;
        }
    };

    struct execute_on_stream_nosync : execute_on_stream_nosync_base<execute_on_stream_nosync>
    {
        using base_t = execute_on_stream_nosync_base<execute_on_stream_nosync>;

        THRUST_HOST_DEVICE
        execute_on_stream_nosync() : base_t(){};
        THRUST_HOST_DEVICE execute_on_stream_nosync(hipStream_t stream)
            : base_t(stream) {};
    };

    struct par_t : execution_policy<par_t>,
                   thrust::detail::allocator_aware_execution_policy<execute_on_stream_base>,
                   thrust::detail::dependencies_aware_execution_policy<execute_on_stream_base>
    {
        using base_t = execution_policy<par_t>;

        THRUST_DEVICE THRUST_HOST
        constexpr par_t() : base_t()
        {
        }

        using stream_attachment_type = execute_on_stream;

        THRUST_HIP_RUNTIME_FUNCTION
        stream_attachment_type
        on(hipStream_t const& stream) const
        {
            return execute_on_stream(stream);
        }
    };

    struct par_nosync_t
        : execution_policy<par_nosync_t>,
          thrust::detail::allocator_aware_execution_policy<execute_on_stream_nosync_base>,
          thrust::detail::dependencies_aware_execution_policy<execute_on_stream_nosync_base>
    {
        using base_t = execution_policy<par_nosync_t>;

        THRUST_HOST_DEVICE
        constexpr par_nosync_t() : base_t() {}

        using stream_attachment_type = execute_on_stream_nosync;

        THRUST_HIP_RUNTIME_FUNCTION
        stream_attachment_type
        on(hipStream_t const &stream) const
        {
            return execute_on_stream_nosync(stream);
        }

    private:
        //this function is defined to allow non-blocking calls on the default_stream() with thrust::cuda::par_nosync
        //without explicitly using thrust::cuda::par_nosync.on(default_stream())
        friend THRUST_HOST_DEVICE  bool
        must_perform_optional_stream_synchronization(const par_nosync_t &)
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
            : base_t() {};
        THRUST_HOST_DEVICE execute_on_stream_deterministic_base(hipStream_t stream)
            : base_t(stream) {};

    private:
        friend THRUST_HOST_DEVICE integral_constant<bool, false>
        allows_nondeterminism(const execute_on_stream_deterministic_base&)
        {
            return {};
        }
    };

    struct execute_on_stream_deterministic
        : execute_on_stream_deterministic_base<execute_on_stream_deterministic>
    {
        using base_t = execute_on_stream_deterministic_base<execute_on_stream_deterministic>;

        THRUST_HOST_DEVICE execute_on_stream_deterministic()
            : base_t() {};
        THRUST_HOST_DEVICE execute_on_stream_deterministic(hipStream_t stream)
            : base_t(stream) {};
    };

    struct par_det_t
        : execution_policy<par_det_t>,
          thrust::detail::allocator_aware_execution_policy<execute_on_stream_deterministic_base>,
          thrust::detail::dependencies_aware_execution_policy<execute_on_stream_deterministic_base>
    {
        using base_t = execution_policy<par_det_t>;

        THRUST_HOST_DEVICE constexpr par_det_t()
            : base_t()
        {
        }

        using stream_attachment_type = execute_on_stream_deterministic;

        THRUST_HIP_RUNTIME_FUNCTION
        stream_attachment_type on(hipStream_t const& stream) const
        {
            return execute_on_stream_deterministic(stream);
        }

    private:
        friend THRUST_HOST_DEVICE integral_constant<bool, false>
                                   allows_nondeterminism(const par_det_t&)
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
            : base_t() {};
        THRUST_HOST_DEVICE execute_on_stream_nosync_deterministic_base(hipStream_t stream)
            : base_t(stream) {};

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
            : base_t() {};
        THRUST_HOST_DEVICE execute_on_stream_nosync_deterministic(hipStream_t stream)
            : base_t(stream) {};
    };

    struct par_det_nosync_t : execution_policy<par_det_nosync_t>,
                              thrust::detail::allocator_aware_execution_policy<
                                  execute_on_stream_nosync_deterministic_base>,
                              thrust::detail::dependencies_aware_execution_policy<
                                  execute_on_stream_nosync_deterministic_base>
    {
        using base_t = execution_policy<par_det_nosync_t>;

        THRUST_HOST_DEVICE constexpr par_det_nosync_t()
            : base_t()
        {
        }

        using stream_attachment_type = execute_on_stream_nosync_deterministic;

        THRUST_HIP_RUNTIME_FUNCTION
        stream_attachment_type on(hipStream_t const& stream) const
        {
            return execute_on_stream_nosync_deterministic(stream);
        }

    private:
        friend THRUST_HOST_DEVICE integral_constant<bool, false>
                                   allows_nondeterminism(const par_det_nosync_t&)
        {
            return {};
        }
    };

    THRUST_INLINE_CONSTANT par_t            par;
    THRUST_INLINE_CONSTANT par_nosync_t     par_nosync;
    THRUST_INLINE_CONSTANT par_det_t        par_det;
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
        }
    } // namesapce hip
} // namespace system

namespace hip
{
    using thrust::hip_rocprim::par;
    using thrust::hip_rocprim::par_det;
    using thrust::hip_rocprim::par_det_nosync;
    using thrust::hip_rocprim::par_nosync;
} // namespace hip

THRUST_NAMESPACE_END
