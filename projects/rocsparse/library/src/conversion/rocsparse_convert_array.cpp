/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "rocsparse.h"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_convert_array.hpp"

namespace rocsparse
{
    //
    // Kernel to copy array of integers with mix precisions
    //
    template <uint32_t BLOCKSIZE, typename TARGET, typename SOURCE>
    __launch_bounds__(BLOCKSIZE) __global__
        static void copy_indexbase_iarray_mix_safe(const size_t               nitems_,
                                                   TARGET*                    target_,
                                                   const rocsparse_index_base target_indexbase_,
                                                   const SOURCE*              source_,
                                                   const rocsparse_index_base source_indexbase_,
                                                   size_t*                    count_out_of_limits_)
    {
        const size_t      tid = hipThreadIdx_x;
        const size_t      gid = tid + BLOCKSIZE * hipBlockIdx_x;
        __shared__ size_t shd[BLOCKSIZE];
        if(gid < nitems_)
        {
            const SOURCE s = source_[gid];
            if(s > std::numeric_limits<TARGET>::max() || s < std::numeric_limits<TARGET>::min())
            {
                shd[tid] = 1;
            }
            else
            {
                target_[gid] = static_cast<TARGET>(s) - source_indexbase_ + target_indexbase_;
                shd[tid]     = 0;
            }
        }
        else
        {
            shd[tid] = 0;
        }

        __syncthreads();
        rocsparse::blockreduce_sum<BLOCKSIZE>(tid, shd);
        if(tid == 0)
        {
            shd[0] = rocsparse::atomic_add(count_out_of_limits_, shd[0]);
        }
    }

    //
    // Kernel to copy array of integers with mix precisions
    //
    template <uint32_t BLOCKSIZE, typename TARGET, typename SOURCE>
    __launch_bounds__(BLOCKSIZE) __global__
        static void copy_iarray_mix_safe(const size_t  nitems_,
                                         TARGET*       target_,
                                         const size_t  target_inc_,
                                         const SOURCE* source_,
                                         const size_t  source_inc_,
                                         size_t*       count_out_of_limits_)
    {
        const size_t      tid = hipThreadIdx_x;
        const size_t      gid = tid + BLOCKSIZE * hipBlockIdx_x;
        __shared__ size_t shd[BLOCKSIZE];
        if(gid < nitems_)
        {
            const SOURCE s = source_[gid * source_inc_];
            if(s > std::numeric_limits<TARGET>::max() || s < std::numeric_limits<TARGET>::min())
            {
                shd[tid] = 1;
            }
            else
            {
                target_[gid * target_inc_] = static_cast<TARGET>(s);
                shd[tid]                   = 0;
            }
        }
        else
        {
            shd[tid] = 0;
        }

        __syncthreads();
        rocsparse::blockreduce_sum<BLOCKSIZE>(tid, shd);
        if(tid == 0)
        {
            shd[0] = rocsparse::atomic_add(count_out_of_limits_, shd[0]);
        }
    }

    template <typename TARGET, typename SOURCE>
    static rocsparse_status
        convert_indexbase_array_compute_core(rocsparse_handle     handle_,
                                             size_t               nitems_,
                                             void*                target__,
                                             rocsparse_index_base target_indexbase_,
                                             const void*          source__,
                                             rocsparse_index_base source_indexbase_,
                                             size_t*              host_num_invalid)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const SOURCE* source_ = (const SOURCE*)source__;
        const TARGET* target_ = (const TARGET*)target__;

        static constexpr uint32_t BLOCKSIZE = 1024;

        size_t* dnum_out_of_range_values = (size_t*)handle_->buffer;
        RETURN_IF_HIP_ERROR(
            hipMemsetAsync(dnum_out_of_range_values, 0, sizeof(size_t), handle_->stream));

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::copy_indexbase_iarray_mix_safe<BLOCKSIZE, TARGET, SOURCE>),
            dim3((nitems_ - 1) / BLOCKSIZE + 1),
            dim3(BLOCKSIZE),
            0,
            handle_->stream,
            nitems_,
            (TARGET*)target_,
            target_indexbase_,
            (const SOURCE*)source_,
            source_indexbase_,
            dnum_out_of_range_values);
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(host_num_invalid,
                                           dnum_out_of_range_values,
                                           sizeof(size_t),
                                           hipMemcpyDeviceToHost,
                                           handle_->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle_->stream));
        if(host_num_invalid[0] > 0)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_type_mismatch);
        }
        return rocsparse_status_success;
    }

    template <typename T, typename... P>
    static rocsparse_status
        convert_indexbase_array_compute_dispatch(rocsparse_indextype source_indextype_, P... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(source_indextype_)
        {
        case rocsparse_indextype_u16:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        case rocsparse_indextype_i32:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::convert_indexbase_array_compute_core<T, int32_t>(p...)));
            return rocsparse_status_success;
        }
        case rocsparse_indextype_i64:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::convert_indexbase_array_compute_core<T, int64_t>(p...)));
            return rocsparse_status_success;
        }
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    template <typename... P>
    static rocsparse_status convert_indexbase_array_compute(rocsparse_indextype target_indextype_,
                                                            rocsparse_indextype source_indextype_,
                                                            P... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(target_indextype_)
        {
        case rocsparse_indextype_u16:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        case rocsparse_indextype_i32:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::convert_indexbase_array_compute_dispatch<int32_t>(
                source_indextype_, p...));
            return rocsparse_status_success;
        }
        case rocsparse_indextype_i64:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::convert_indexbase_array_compute_dispatch<int64_t>(
                source_indextype_, p...));
            return rocsparse_status_success;
        }
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    template <typename TARGET, typename SOURCE>
    static rocsparse_status convert_indexing_array_compute_core(rocsparse_handle handle_,
                                                                size_t           nitems_,
                                                                void*            target__,
                                                                int64_t          target_inc_,
                                                                const void*      source__,
                                                                int64_t          source_inc_,
                                                                size_t*          host_num_invalid)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const SOURCE* source_ = (const SOURCE*)source__;
        const TARGET* target_ = (const TARGET*)target__;

        static constexpr uint32_t BLOCKSIZE = 1024;

        size_t* dnum_out_of_range_values = (size_t*)handle_->buffer;
        RETURN_IF_HIP_ERROR(
            hipMemsetAsync(dnum_out_of_range_values, 0, sizeof(size_t), handle_->stream));

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::copy_iarray_mix_safe<BLOCKSIZE, TARGET, SOURCE>),
            dim3((nitems_ - 1) / BLOCKSIZE + 1),
            dim3(BLOCKSIZE),
            0,
            handle_->stream,
            nitems_,
            (TARGET*)target_,
            target_inc_,
            (const SOURCE*)source_,
            source_inc_,
            dnum_out_of_range_values);
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(host_num_invalid,
                                           dnum_out_of_range_values,
                                           sizeof(size_t),
                                           hipMemcpyDeviceToHost,
                                           handle_->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle_->stream));
        if(host_num_invalid[0] > 0)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_type_mismatch);
        }
        return rocsparse_status_success;
    }

    template <typename T, typename... P>
    static rocsparse_status
        convert_indexing_array_compute_dispatch(rocsparse_indextype source_indextype_, P... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(source_indextype_)
        {
        case rocsparse_indextype_u16:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        case rocsparse_indextype_i32:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::convert_indexing_array_compute_core<T, int32_t>(p...)));
            return rocsparse_status_success;
        }
        case rocsparse_indextype_i64:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::convert_indexing_array_compute_core<T, int64_t>(p...)));
            return rocsparse_status_success;
        }
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    template <typename... P>
    static rocsparse_status convert_indexing_array_compute(rocsparse_indextype target_indextype_,
                                                           rocsparse_indextype source_indextype_,
                                                           P... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(target_indextype_)
        {
        case rocsparse_indextype_u16:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        case rocsparse_indextype_i32:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::convert_indexing_array_compute_dispatch<int32_t>(
                source_indextype_, p...));
            return rocsparse_status_success;
        }
        case rocsparse_indextype_i64:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::convert_indexing_array_compute_dispatch<int64_t>(
                source_indextype_, p...));
            return rocsparse_status_success;
        }
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    template <typename TARGET, typename SOURCE, class FILTER = void>
    struct copy_farray_mix_safe_kernel_t
    {
        template <uint32_t BLOCKSIZE>
        __launch_bounds__(BLOCKSIZE) __global__
            static void run(const size_t             nitems_,
                            TARGET*                  target_,
                            const SOURCE*            source_,
                            floating_data_t<SOURCE>* conversion_error_)
        {
        }
    };

    template <typename TARGET, typename SOURCE>
    struct copy_farray_mix_safe_kernel_t<
        TARGET,
        SOURCE,
        std::enable_if_t<((std::is_same<TARGET, SOURCE>{}) || //
                          (std::is_same<TARGET, double>{} && std::is_same<SOURCE, float>{}) || //
                          (std::is_same<TARGET, rocsparse_double_complex>{}
                           && std::is_same<SOURCE, rocsparse_float_complex>{}))>>
    {
        template <uint32_t BLOCKSIZE>
        __launch_bounds__(BLOCKSIZE) __global__
            static void run(const size_t             nitems_,
                            TARGET*                  target_,
                            const SOURCE*            source_,
                            floating_data_t<SOURCE>* conversion_error_)
        {
            const size_t tid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
            if(tid < nitems_)
            {
                target_[tid] = source_[tid];
            }
        };
    };

    //
    // Specialized conversion from float to float complex.
    //
    template <>
    struct copy_farray_mix_safe_kernel_t<rocsparse_float_complex, float>
    {
        using SOURCE = float;
        using TARGET = rocsparse_float_complex;

        template <uint32_t BLOCKSIZE>
        __launch_bounds__(BLOCKSIZE) __global__
            static void run(const size_t             nitems_,
                            TARGET*                  target_,
                            const SOURCE*            source_,
                            floating_data_t<SOURCE>* conversion_error_)
        {
            const size_t tid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
            if(tid < nitems_)
            {
                target_[tid] = {static_cast<floating_data_t<TARGET>>(source_[tid]),
                                static_cast<floating_data_t<TARGET>>(0)};
            }
        }
    };

    //
    // Specialized conversion from double to float complex.
    //
    template <>
    struct copy_farray_mix_safe_kernel_t<rocsparse_float_complex, double>
    {
        using SOURCE = double;
        using TARGET = rocsparse_float_complex;

        template <uint32_t BLOCKSIZE>
        __launch_bounds__(BLOCKSIZE) __global__
            static void run(const size_t             nitems_,
                            TARGET*                  target_,
                            const SOURCE*            source_,
                            floating_data_t<SOURCE>* conversion_error_)
        {
            const size_t tid = hipThreadIdx_x;
            const size_t gid = tid + BLOCKSIZE * hipBlockIdx_x;
            __shared__ floating_data_t<SOURCE> shd[BLOCKSIZE];
            if(gid < nitems_)
            {
                const SOURCE s  = source_[gid];
                const SOURCE sf = floating_data_t<SOURCE>(s);
                const TARGET t{static_cast<floating_data_t<TARGET>>(sf),
                               static_cast<floating_data_t<TARGET>>(0)};
                shd[tid]     = rocsparse::abs(s - sf);
                target_[gid] = t;
            }
            else
            {
                shd[tid] = floating_data_t<SOURCE>(0);
            }

            __syncthreads();
            rocsparse::blockreduce_max<BLOCKSIZE>(tid, shd);
            if(tid == 0)
            {
                shd[0] = rocsparse::atomic_max(conversion_error_, shd[0]);
            }
        }
    };

    //
    // Specialized conversion from double complex to float complex.
    //
    template <>
    struct copy_farray_mix_safe_kernel_t<rocsparse_float_complex, rocsparse_double_complex>
    {
        using SOURCE = rocsparse_double_complex;
        using TARGET = rocsparse_float_complex;

        template <uint32_t BLOCKSIZE>
        __launch_bounds__(BLOCKSIZE) __global__
            static void run(const size_t             nitems_,
                            TARGET*                  target_,
                            const SOURCE*            source_,
                            floating_data_t<SOURCE>* conversion_error_)
        {
            const size_t tid = hipThreadIdx_x;
            const size_t gid = tid + BLOCKSIZE * hipBlockIdx_x;
            __shared__ floating_data_t<SOURCE> shd[BLOCKSIZE];
            if(gid < nitems_)
            {
                const SOURCE s = source_[gid];

                TARGET t(static_cast<floating_data_t<TARGET>>(std::real(s)),
                         static_cast<floating_data_t<TARGET>>(std::imag(s)));

                const SOURCE sback{static_cast<floating_data_t<SOURCE>>(std::real(t)),
                                   static_cast<floating_data_t<SOURCE>>(std::imag(t))};
                shd[tid]     = rocsparse::abs(s - sback);
                target_[gid] = t;
            }
            else
            {
                shd[tid] = floating_data_t<SOURCE>(0);
            }

            __syncthreads();
            rocsparse::blockreduce_max<BLOCKSIZE>(tid, shd);
            if(tid == 0)
            {
                shd[0] = rocsparse::atomic_max(conversion_error_, shd[0]);
            }
        }
    };

    //
    // Specialized conversion from float complex to double complex.
    //
    template <>
    struct copy_farray_mix_safe_kernel_t<rocsparse_double_complex, rocsparse_float_complex>
    {
        using SOURCE = rocsparse_float_complex;
        using TARGET = rocsparse_double_complex;

        template <uint32_t BLOCKSIZE>
        __launch_bounds__(BLOCKSIZE) __global__
            static void run(size_t                   nitems_,
                            TARGET*                  target_,
                            const SOURCE*            source_,
                            floating_data_t<SOURCE>* conversion_error_)
        {
            const size_t tid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
            if(tid < nitems_)
            {
                const SOURCE s = source_[tid];
                target_[tid]
                    = TARGET{static_cast<double>(std::real(s)), static_cast<double>(std::imag(s))};
            }
        }
    };

    template <>
    struct copy_farray_mix_safe_kernel_t<float, double>
    {
        using TARGET = float;
        using SOURCE = double;
        template <uint32_t BLOCKSIZE>
        __launch_bounds__(BLOCKSIZE) __global__
            static void run(size_t                   nitems_,
                            TARGET*                  target_,
                            const SOURCE*            source_,
                            floating_data_t<SOURCE>* conversion_error_)
        {
            const size_t tid = hipThreadIdx_x;
            const size_t gid = tid + BLOCKSIZE * hipBlockIdx_x;
            __shared__ floating_data_t<SOURCE> shd[BLOCKSIZE];
            if(gid < nitems_)
            {
                const SOURCE s = source_[gid];
                const TARGET t = static_cast<TARGET>(s);
                shd[tid]       = rocsparse::abs(s - static_cast<SOURCE>(t));
                target_[gid]   = t;
            }
            else
            {
                shd[tid] = floating_data_t<SOURCE>(0);
            }

            __syncthreads();
            rocsparse::blockreduce_max<BLOCKSIZE>(tid, shd);
            if(tid == 0)
            {
                shd[0] = rocsparse::atomic_max(conversion_error_, shd[0]);
            }
        }
    };

    template <typename TARGET, typename SOURCE>
    static rocsparse_status convert_data_array_compute_core(rocsparse_handle handle_,
                                                            size_t           nitems_,
                                                            void*            target__,
                                                            const void*      source__,
                                                            double*          host_error)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const SOURCE*             source_   = (const SOURCE*)source__;
        const TARGET*             target_   = (const TARGET*)target__;
        static constexpr uint32_t BLOCKSIZE = 1024;
        floating_data_t<SOURCE>*  derr      = (floating_data_t<SOURCE>*)handle_->buffer;
        floating_data_t<SOURCE>   herr;
        RETURN_IF_HIP_ERROR(
            hipMemsetAsync(derr, 0, sizeof(floating_data_t<SOURCE>), handle_->stream));
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::copy_farray_mix_safe_kernel_t<TARGET, SOURCE>::template run<BLOCKSIZE>),
            dim3((nitems_ - 1) / BLOCKSIZE + 1),
            dim3(BLOCKSIZE),
            0,
            handle_->stream,
            nitems_,
            (TARGET*)target_,
            (const SOURCE*)source_,
            (floating_data_t<SOURCE>*)derr);
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &herr, derr, sizeof(floating_data_t<SOURCE>), hipMemcpyDeviceToHost, handle_->stream));
        host_error[0] = static_cast<double>(herr);
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle_->stream));
        return rocsparse_status_success;
    }

    template <typename T, typename... P>
    static rocsparse_status convert_data_array_compute_dispatch(rocsparse_datatype source_datatype_,
                                                                P... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(source_datatype_)
        {
        case rocsparse_datatype_i8_r:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::convert_data_array_compute_core<T, int8_t>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_u8_r:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::convert_data_array_compute_core<T, uint8_t>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_i32_r:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::convert_data_array_compute_core<T, int32_t>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_u32_r:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::convert_data_array_compute_core<T, uint32_t>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_f16_r:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::convert_data_array_compute_core<T, _Float16>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_bf16_r:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::convert_data_array_compute_core<T, rocsparse_bfloat16>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_f32_r:
        {
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::convert_data_array_compute_core<T, float>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_f64_r:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::convert_data_array_compute_core<T, double>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_f32_c:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::convert_data_array_compute_core<T, rocsparse_float_complex>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_f64_c:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::convert_data_array_compute_core<T, rocsparse_double_complex>)(p...));
            return rocsparse_status_success;
        }
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    template <typename... P>
    static rocsparse_status convert_data_array_compute(rocsparse_datatype target_datatype_,
                                                       rocsparse_datatype source_datatype_,
                                                       P... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(target_datatype_)
        {
        case rocsparse_datatype_i8_r:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::convert_data_array_compute_dispatch<int8_t>)(source_datatype_, p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_u8_r:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::convert_data_array_compute_dispatch<uint8_t>)(source_datatype_, p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_i32_r:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::convert_data_array_compute_dispatch<int32_t>)(source_datatype_, p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_u32_r:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::convert_data_array_compute_dispatch<uint32_t>)(source_datatype_, p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_f16_r:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::convert_data_array_compute_dispatch<_Float16>(source_datatype_, p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_bf16_r:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::convert_data_array_compute_dispatch<rocsparse_bfloat16>(source_datatype_,
                                                                                   p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_f32_r:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::convert_data_array_compute_dispatch<float>(source_datatype_, p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_f64_r:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::convert_data_array_compute_dispatch<double>(source_datatype_, p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_f32_c:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::convert_data_array_compute_dispatch<rocsparse_float_complex>(
                    source_datatype_, p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_f64_c:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::convert_data_array_compute_dispatch<rocsparse_double_complex>(
                    source_datatype_, p...));
            return rocsparse_status_success;
        }
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
}

//
// Converting indexing arrays.
//
rocsparse_status rocsparse::convert_array(rocsparse_handle     handle_,
                                          size_t               nitems_,
                                          rocsparse_indextype  target_indextype_,
                                          void*                target_,
                                          rocsparse_index_base target_indexbase_,
                                          rocsparse_indextype  source_indextype_,
                                          const void*          source_,
                                          rocsparse_index_base source_indexbase_)
{
    ROCSPARSE_ROUTINE_TRACE;

    if((source_indextype_ == target_indextype_) && (target_indexbase_ == source_indexbase_))
    {
        if(target_ != source_)
        {
            const size_t sizeof_data = rocsparse::indextype_sizeof(source_indextype_);
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                target_, source_, sizeof_data * nitems_, hipMemcpyDeviceToDevice, handle_->stream));
        }
    }
    else
    {
        size_t                 count_out_of_bounds_conversion = 0;
        const rocsparse_status status
            = rocsparse::convert_indexbase_array_compute(target_indextype_,
                                                         source_indextype_,
                                                         handle_,
                                                         nitems_,
                                                         target_,
                                                         target_indexbase_,
                                                         source_,
                                                         source_indexbase_,
                                                         &count_out_of_bounds_conversion);
        if(status != rocsparse_status_success)
        {
            std::cerr << "rocsparse_convert_array_compute has detected "
                      << count_out_of_bounds_conversion << " invalid data." << std::endl;
            RETURN_IF_ROCSPARSE_ERROR(status);
        }
    }
    return rocsparse_status_success;
}

//
// Converting indexing arrays.
//
rocsparse_status rocsparse::convert_array(rocsparse_handle    handle_,
                                          size_t              nitems_,
                                          rocsparse_indextype target_indextype_,
                                          void*               target_,
                                          int64_t             target_inc_,
                                          rocsparse_indextype source_indextype_,
                                          const void*         source_,
                                          int64_t             source_inc_)
{
    ROCSPARSE_ROUTINE_TRACE;

    if((source_indextype_ == target_indextype_) && (target_inc_ == 1 && source_inc_ == 1))
    {
        if(target_ != source_)
        {
            const size_t sizeof_data = rocsparse::indextype_sizeof(source_indextype_);
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                target_, source_, sizeof_data * nitems_, hipMemcpyDeviceToDevice, handle_->stream));
        }
    }
    else
    {
        size_t                 count_out_of_bounds_conversion = 0;
        const rocsparse_status status
            = rocsparse::convert_indexing_array_compute(target_indextype_,
                                                        source_indextype_,
                                                        handle_,
                                                        nitems_,
                                                        target_,
                                                        target_inc_,
                                                        source_,
                                                        source_inc_,
                                                        &count_out_of_bounds_conversion);
        if(status != rocsparse_status_success)
        {
            std::cerr << "rocsparse_convert_array_compute has detected "
                      << count_out_of_bounds_conversion << " invalid data." << std::endl;
            RETURN_IF_ROCSPARSE_ERROR(status);
        }
    }
    return rocsparse_status_success;
}

//
// Converting indexing array.
//
rocsparse_status rocsparse::convert_array(rocsparse_handle    handle_,
                                          size_t              nitems_,
                                          rocsparse_indextype target_indextype_,
                                          void*               target_,
                                          rocsparse_indextype source_indextype_,
                                          const void*         source_)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::convert_array(
        handle_, nitems_, target_indextype_, target_, 1, source_indextype_, source_, 1));
    return rocsparse_status_success;
}

///
/// @brief Convert numerical arrays.
/// @param handle_ The rocsparse handle.
/// @param nitems_ The number of items to copy.
/// @param target_datatype_ The data type of the target array.
/// @param target_ The target array.
/// @param source_datatype_ The data type of the source array.
/// @param source_ The source array.
/// @return The rocsparse status.
///
rocsparse_status rocsparse::convert_array(rocsparse_handle   handle_,
                                          size_t             nitems_,
                                          rocsparse_datatype target_datatype_,
                                          void*              target_,
                                          rocsparse_datatype source_datatype_,
                                          const void*        source_)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(source_datatype_ == target_datatype_)
    {
        //
        // Treating arrays with same data types.
        //
        if(target_ != source_)
        {
            const size_t sizeof_data = rocsparse::datatype_sizeof(source_datatype_);
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                target_, source_, sizeof_data * nitems_, hipMemcpyDeviceToDevice, handle_->stream));
        }
        return rocsparse_status_success;
    }
    else
    {
        //
        // Treating arrays with different data types.
        //
        double conversion_error_max;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::convert_data_array_compute(target_datatype_,
                                                                        source_datatype_,
                                                                        handle_,
                                                                        nitems_,
                                                                        target_,
                                                                        source_,
                                                                        &conversion_error_max));

        std::cout << "rocsparse_convert_array numerical conversion error " << conversion_error_max
                  << " invalid data." << std::endl;
        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::dnvec_transfer_from(rocsparse_handle            handle,
                                                rocsparse_dnvec_descr       target,
                                                rocsparse_const_dnvec_descr source)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, target);
    ROCSPARSE_CHECKARG_POINTER(1, source);
    ROCSPARSE_CHECKARG(0, target, (target->size != source->size), rocsparse_status_invalid_size);
    switch(target->data_type)
    {
    case rocsparse_datatype_f32_c:
    case rocsparse_datatype_f64_c:
    {
        break;
    }
    case rocsparse_datatype_i8_r:
    case rocsparse_datatype_u8_r:
    case rocsparse_datatype_i32_r:
    case rocsparse_datatype_u32_r:
    case rocsparse_datatype_f16_r:
    case rocsparse_datatype_f32_r:
    case rocsparse_datatype_f64_r:
    case rocsparse_datatype_bf16_r:
    {
        switch(source->data_type)
        {
        case rocsparse_datatype_f32_c:
        case rocsparse_datatype_f64_c:
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_not_implemented,
                "source data is defined with complex types whereas "
                "target data is defined with real type");
            break;
        }
        case rocsparse_datatype_i8_r:
        case rocsparse_datatype_u8_r:
        case rocsparse_datatype_i32_r:
        case rocsparse_datatype_u32_r:
        case rocsparse_datatype_f16_r:
        case rocsparse_datatype_f32_r:
        case rocsparse_datatype_f64_r:
        case rocsparse_datatype_bf16_r:
        {
            break;
        }
        }
        break;
    }
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::convert_array(handle,
                                                       target->size,
                                                       target->data_type,
                                                       target->values,
                                                       source->data_type,
                                                       source->values));
    return rocsparse_status_success;
}
