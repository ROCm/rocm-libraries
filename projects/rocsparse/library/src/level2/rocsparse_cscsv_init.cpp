/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_cscsv_solve_kernel.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_grid.hpp"
#include "rocsparse_scalar.hpp"
#include "rocsparse_utility.hpp"

#include <map>
#include <sstream>
#include <tuple>

#if defined(ROCSPARSE_WITH_SPTRSV_CSC)
namespace rocsparse
{
    // For each batch instance, resets the dependency counters from the analysis and starts the
    // partial sums at alpha * x. The solve does not read x, so x and y may be the same vector.
    template <uint32_t BLOCKSIZE, typename J, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsv_init_kernel(J       m,
                           int64_t batch_count,
                           ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                           int64_t alpha_stride,
                           const J* __restrict__ dependency_count,
                           const T* __restrict__ x,
                           int64_t x_inc,
                           int64_t x_stride,
                           J* __restrict__ rem,
                           T* __restrict__ sum,
                           bool is_host_mode)
    {
        for(int64_t i = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;
            i < m * batch_count;
            i += static_cast<int64_t>(hipGridDim_x) * BLOCKSIZE)
        {
            const int64_t batch = i / m;
            const int64_t row   = i % m;
            const T       alpha
                = is_host_mode ? alpha_union.value : alpha_union.pointer[batch * alpha_stride];
            rem[i] = dependency_count[row];
            sum[i] = alpha * x[batch * x_stride + row * x_inc];
        }
    }

    template <uint32_t BLOCKSIZE, typename J, typename T>
    static rocsparse_status cscsv_init_launch(rocsparse_handle handle,
                                              int64_t          m,
                                              int64_t          batch_count,
                                              const void*      alpha,
                                              int64_t          alpha_stride,
                                              const void*      dependency_count,
                                              const void*      x,
                                              int64_t          x_inc,
                                              int64_t          x_stride,
                                              void*            rem,
                                              void*            sum)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::cscsv_init_kernel<BLOCKSIZE, J, T>),
            dim3(rocsparse::get_grid_size_x(
                handle, rocsparse::ceil_div(m * batch_count, BLOCKSIZE), BLOCKSIZE)),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            static_cast<J>(m),
            batch_count,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, reinterpret_cast<const T*>(alpha)),
            alpha_stride,
            reinterpret_cast<const J*>(dependency_count),
            reinterpret_cast<const T*>(x),
            x_inc,
            x_stride,
            reinterpret_cast<J*>(rem),
            reinterpret_cast<T*>(sum),
            handle->pointer_mode == rocsparse_pointer_mode_host);
        return rocsparse_status_success;
    }
    using cscsv_init_tpl_t = std::tuple<rocsparse_indextype, rocsparse_datatype>;

#define CONFIG(J_, T_)                                                          \
    {                                                                           \
        cscsv_init_tpl_t(J_, T_),                                               \
            cscsv_init_launch<rocsparse::cscsv_kernel_constants_t::s_blocksize, \
                              typename rocsparse::indextype_traits<J_>::type_t, \
                              typename rocsparse::datatype_traits<T_>::type_t>  \
    }
#define CONFIG_TYPES(J_)                                                        \
    CONFIG(J_, rocsparse_datatype_f32_r), CONFIG(J_, rocsparse_datatype_f64_r), \
        CONFIG(J_, rocsparse_datatype_f32_c), CONFIG(J_, rocsparse_datatype_f64_c)

    static const std::map<cscsv_init_tpl_t, rocsparse::cscsv_init_launch_t> s_cscsv_init_dispatch{
        {CONFIG_TYPES(rocsparse_indextype_i32), CONFIG_TYPES(rocsparse_indextype_i64)}};

#undef CONFIG_TYPES
#undef CONFIG
}

rocsparse_status rocsparse::cscsv_init_launch_find(rocsparse::cscsv_init_launch_t* launch,
                                                   rocsparse_indextype             j_type,
                                                   rocsparse_datatype              t_type)
{
    const auto& it
        = rocsparse::s_cscsv_init_dispatch.find(rocsparse::cscsv_init_tpl_t(j_type, t_type));
    if(it == rocsparse::s_cscsv_init_dispatch.end())
    {
        // LCOV_EXCL_START
        std::stringstream sstr;
        sstr << "invalid type configuration: j_type: " << rocsparse::enum_utils::to_string(j_type)
             << ", t_type: " << rocsparse::enum_utils::to_string(t_type);
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error, sstr.str().c_str());
        // LCOV_EXCL_STOP
    }
    launch[0] = it->second;
    return rocsparse_status_success;
}
#endif
