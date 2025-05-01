/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "rocblas_dot_kernels.hpp"
#include "check_numerics_vector.hpp"

template <typename T, typename Tex>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_dot_template(rocblas_handle __restrict__ handle,
                                  rocblas_int n,
                                  const T* __restrict__ x,
                                  rocblas_stride offsetx,
                                  rocblas_int    incx,
                                  rocblas_stride stridex,
                                  const T* __restrict__ y,
                                  rocblas_stride offsety,
                                  rocblas_int    incy,
                                  rocblas_stride stridey,
                                  rocblas_int    batch_count,
                                  T* __restrict__ results,
                                  Tex* __restrict__ workspace)
{
    return rocblas_internal_dot_launcher<rocblas_int, ROCBLAS_DOT_NB, false>(handle,
                                                                             n,
                                                                             x,
                                                                             offsetx,
                                                                             incx,
                                                                             stridex,
                                                                             y,
                                                                             offsety,
                                                                             incy,
                                                                             stridey,
                                                                             batch_count,
                                                                             results,
                                                                             workspace);
}

template <typename T, typename Tex>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_dotc_template(rocblas_handle __restrict__ handle,
                                   rocblas_int n,
                                   const T* __restrict__ x,
                                   rocblas_stride offsetx,
                                   rocblas_int    incx,
                                   rocblas_stride stridex,
                                   const T* __restrict__ y,
                                   rocblas_stride offsety,
                                   rocblas_int    incy,
                                   rocblas_stride stridey,
                                   rocblas_int    batch_count,
                                   T* __restrict__ results,
                                   Tex* __restrict__ workspace)
{
    return rocblas_internal_dot_launcher<rocblas_int, ROCBLAS_DOT_NB, true>(handle,
                                                                            n,
                                                                            x,
                                                                            offsetx,
                                                                            incx,
                                                                            stridex,
                                                                            y,
                                                                            offsety,
                                                                            incy,
                                                                            stridey,
                                                                            batch_count,
                                                                            results,
                                                                            workspace);
}

template <typename T, typename Tex>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_dot_batched_template(rocblas_handle __restrict__ handle,
                                          rocblas_int n,
                                          const T* const* __restrict__ x,
                                          rocblas_stride offsetx,
                                          rocblas_int    incx,
                                          rocblas_stride stridex,
                                          const T* const* __restrict__ y,
                                          rocblas_stride offsety,
                                          rocblas_int    incy,
                                          rocblas_stride stridey,
                                          rocblas_int    batch_count,
                                          T* __restrict__ results,
                                          Tex* __restrict__ workspace)
{
    return rocblas_internal_dot_launcher<rocblas_int, ROCBLAS_DOT_NB, false>(handle,
                                                                             n,
                                                                             x,
                                                                             offsetx,
                                                                             incx,
                                                                             stridex,
                                                                             y,
                                                                             offsety,
                                                                             incy,
                                                                             stridey,
                                                                             batch_count,
                                                                             results,
                                                                             workspace);
}

template <typename T, typename Tex>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_dotc_batched_template(rocblas_handle __restrict__ handle,
                                           rocblas_int n,
                                           const T* const* __restrict__ x,
                                           rocblas_stride offsetx,
                                           rocblas_int    incx,
                                           rocblas_stride stridex,
                                           const T* const* __restrict__ y,
                                           rocblas_stride offsety,
                                           rocblas_int    incy,
                                           rocblas_stride stridey,
                                           rocblas_int    batch_count,
                                           T* __restrict__ results,
                                           Tex* __restrict__ workspace)
{
    return rocblas_internal_dot_launcher<rocblas_int, ROCBLAS_DOT_NB, true>(handle,
                                                                            n,
                                                                            x,
                                                                            offsetx,
                                                                            incx,
                                                                            stridex,
                                                                            y,
                                                                            offsety,
                                                                            incy,
                                                                            stridey,
                                                                            batch_count,
                                                                            results,
                                                                            workspace);
}

template <typename T>
rocblas_status rocblas_dot_check_numerics(const char*    function_name,
                                          rocblas_handle handle,
                                          int64_t        n,
                                          T              x,
                                          rocblas_stride offset_x,
                                          int64_t        inc_x,
                                          rocblas_stride stride_x,
                                          T              y,
                                          rocblas_stride offset_y,
                                          int64_t        inc_y,
                                          rocblas_stride stride_y,
                                          int64_t        batch_count,
                                          const int      check_numerics,
                                          bool           is_input)
{
    rocblas_status check_numerics_status
        = rocblas_internal_check_numerics_vector_template(function_name,
                                                          handle,
                                                          n,
                                                          x,
                                                          offset_x,
                                                          inc_x,
                                                          stride_x,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);
    if(check_numerics_status != rocblas_status_success)
        return check_numerics_status;

    check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                            handle,
                                                                            n,
                                                                            y,
                                                                            offset_y,
                                                                            inc_y,
                                                                            stride_y,
                                                                            batch_count,
                                                                            check_numerics,
                                                                            is_input);

    return check_numerics_status;
}

// If there are any changes in template parameters in the files *dot*.cpp
// instantiations below will need to be manually updated to match the changes.

// clang-format off

#ifdef INSTANTIATE_DOT_CHECK_NUMERICS
#error INSTANTIATE_DOT_CHECK_NUMERICS already defined
#endif

#define INSTANTIATE_DOT_CHECK_NUMERICS(T_) \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_dot_check_numerics<T_>(const char* function_name, \
    rocblas_handle handle, \
    int64_t    n, \
    T_             x, \
    rocblas_stride offset_x, \
    int64_t    inc_x, \
    rocblas_stride stride_x, \
    T_             y, \
    rocblas_stride offset_y, \
    int64_t    inc_y, \
    rocblas_stride stride_y, \
    int64_t    batch_count, \
    const int      check_numerics, \
    bool           is_input);

INSTANTIATE_DOT_CHECK_NUMERICS(rocblas_half const*)
INSTANTIATE_DOT_CHECK_NUMERICS(rocblas_half const* const*)
INSTANTIATE_DOT_CHECK_NUMERICS(rocblas_bfloat16 const*)
INSTANTIATE_DOT_CHECK_NUMERICS(rocblas_bfloat16 const* const*)
INSTANTIATE_DOT_CHECK_NUMERICS(float const*)
INSTANTIATE_DOT_CHECK_NUMERICS(float const* const*)
INSTANTIATE_DOT_CHECK_NUMERICS(double const*)
INSTANTIATE_DOT_CHECK_NUMERICS(double const* const*)
INSTANTIATE_DOT_CHECK_NUMERICS(rocblas_float_complex const*)
INSTANTIATE_DOT_CHECK_NUMERICS(rocblas_float_complex const* const*)
INSTANTIATE_DOT_CHECK_NUMERICS(rocblas_double_complex const*)
INSTANTIATE_DOT_CHECK_NUMERICS(rocblas_double_complex const* const*)

#undef INSTANTIATE_DOT_CHECK_NUMERICS


#ifdef INSTANTIATE_DOT_TEMPLATE
#error INSTANTIATE_DOT_TEMPLATE already defined
#endif

#define INSTANTIATE_DOT_TEMPLATE(T_, Tex_)                                                      \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE                                                       \
rocblas_status rocblas_internal_dot_template<T_, Tex_>(rocblas_handle __restrict__ handle,      \
                                                       rocblas_int                 n,           \
                                                       const T_*      __restrict__ x,           \
                                                       rocblas_stride              offsetx,     \
                                                       rocblas_int                 incx,        \
                                                       rocblas_stride              stridex,     \
                                                       const T_*      __restrict__ y,           \
                                                       rocblas_stride              offsety,     \
                                                       rocblas_int                 incy,        \
                                                       rocblas_stride              stridey,     \
                                                       rocblas_int                 batch_count, \
                                                       T_*            __restrict__ results,     \
                                                       Tex_*          __restrict__ workspace);

INSTANTIATE_DOT_TEMPLATE(rocblas_half, rocblas_half)
INSTANTIATE_DOT_TEMPLATE(rocblas_bfloat16, float)
INSTANTIATE_DOT_TEMPLATE(float, float)
INSTANTIATE_DOT_TEMPLATE(double, double)
INSTANTIATE_DOT_TEMPLATE(rocblas_float_complex, rocblas_float_complex)
INSTANTIATE_DOT_TEMPLATE(rocblas_double_complex, rocblas_double_complex)

#undef INSTANTIATE_DOT_TEMPLATE

#ifdef INSTANTIATE_DOTC_TEMPLATE
#error INSTANTIATE_DOTC_TEMPLATE already defined
#endif

#define INSTANTIATE_DOTC_TEMPLATE(T_, Tex_)                                                 \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE                                                   \
rocblas_status rocblas_internal_dotc_template<T_, Tex_>(rocblas_handle __restrict__ handle, \
                                                  rocblas_int                 n,            \
                                                  const T_*      __restrict__ x,            \
                                                  rocblas_stride              offsetx,      \
                                                  rocblas_int                 incx,         \
                                                  rocblas_stride              stridex,      \
                                                  const T_*      __restrict__ y,            \
                                                  rocblas_stride              offsety,      \
                                                  rocblas_int                 incy,         \
                                                  rocblas_stride              stridey,      \
                                                  rocblas_int                 batch_count,  \
                                                  T_*            __restrict__ results,      \
                                                  Tex_*          __restrict__ workspace);

INSTANTIATE_DOTC_TEMPLATE(rocblas_float_complex, rocblas_float_complex)
INSTANTIATE_DOTC_TEMPLATE(rocblas_double_complex, rocblas_double_complex)

#undef INSTANTIATE_DOTC_TEMPLATE

#ifdef INSTANTIATE_DOT_BATCHED_TEMPLATE
#error INSTANTIATE_DOT_BATCHED_TEMPLATE already defined
#endif

#define INSTANTIATE_DOT_BATCHED_TEMPLATE(T_, Tex_)                                                        \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE                                                                 \
rocblas_status rocblas_internal_dot_batched_template<T_, Tex_>(rocblas_handle   __restrict__ handle,      \
                                                               rocblas_int                   n,           \
                                                               const T_* const* __restrict__ x,           \
                                                               rocblas_stride                offsetx,     \
                                                               rocblas_int                   incx,        \
                                                               rocblas_stride                stridex,     \
                                                               const T_* const* __restrict__ y,           \
                                                               rocblas_stride                offsety,     \
                                                               rocblas_int                   incy,        \
                                                               rocblas_stride                stridey,     \
                                                               rocblas_int                   batch_count, \
                                                               T_*              __restrict__ results,     \
                                                               Tex_*            __restrict__ workspace);

INSTANTIATE_DOT_BATCHED_TEMPLATE(rocblas_half, rocblas_half)
INSTANTIATE_DOT_BATCHED_TEMPLATE(rocblas_bfloat16, float)
INSTANTIATE_DOT_BATCHED_TEMPLATE(float, float)
INSTANTIATE_DOT_BATCHED_TEMPLATE(double, double)
INSTANTIATE_DOT_BATCHED_TEMPLATE(rocblas_float_complex, rocblas_float_complex)
INSTANTIATE_DOT_BATCHED_TEMPLATE(rocblas_double_complex, rocblas_double_complex)

#undef INSTANTIATE_DOT_BATCHED_TEMPLATE

#ifdef INSTANTIATE_DOTC_BATCHED_TEMPLATE
#error INSTANTIATE_DOTC_BATCHED_TEMPLATE already defined
#endif

#define INSTANTIATE_DOTC_BATCHED_TEMPLATE(T_, Tex_)                                                        \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE                                                                  \
rocblas_status rocblas_internal_dotc_batched_template<T_, Tex_>(rocblas_handle   __restrict__ handle,      \
                                                                rocblas_int                   n,           \
                                                                const T_* const* __restrict__ x,           \
                                                                rocblas_stride                offsetx,     \
                                                                rocblas_int                   incx,        \
                                                                rocblas_stride                stridex,     \
                                                                const T_* const* __restrict__ y,           \
                                                                rocblas_stride                offsety,     \
                                                                rocblas_int                   incy,        \
                                                                rocblas_stride                stridey,     \
                                                                rocblas_int                   batch_count, \
                                                                T_*              __restrict__ results,     \
                                                                Tex_*            __restrict__ workspace);

INSTANTIATE_DOTC_BATCHED_TEMPLATE(rocblas_float_complex, rocblas_float_complex)
INSTANTIATE_DOTC_BATCHED_TEMPLATE(rocblas_double_complex, rocblas_double_complex)

#undef INSTANTIATE_DOTC_BATCHED_TEMPLATE

// instantiate _ex forms from header file define

// Mixed precision for dot_ex
INST_DOT_EX_LAUNCHER(rocblas_int, ROCBLAS_DOT_NB, false, rocblas_half, rocblas_half const*, float)
INST_DOT_EX_LAUNCHER(rocblas_int, ROCBLAS_DOT_NB, false, rocblas_half, rocblas_half const* const*, float)

// real types are "supported" in dotc_ex
INST_DOT_EX_LAUNCHER(rocblas_int, ROCBLAS_DOT_NB, true, rocblas_half, rocblas_half const*, float)
INST_DOT_EX_LAUNCHER(rocblas_int, ROCBLAS_DOT_NB, true, rocblas_half, rocblas_half const* const*, float)
INST_DOT_EX_LAUNCHER(rocblas_int, ROCBLAS_DOT_NB, true, rocblas_half, rocblas_half const*, rocblas_half)
INST_DOT_EX_LAUNCHER(rocblas_int, ROCBLAS_DOT_NB, true, rocblas_half, rocblas_half const* const*, rocblas_half)
INST_DOT_EX_LAUNCHER(rocblas_int, ROCBLAS_DOT_NB, true, rocblas_bfloat16, rocblas_bfloat16 const*, float)
INST_DOT_EX_LAUNCHER(rocblas_int, ROCBLAS_DOT_NB, true, rocblas_bfloat16, rocblas_bfloat16 const* const*, float)
INST_DOT_EX_LAUNCHER(rocblas_int, ROCBLAS_DOT_NB, true, float, float const*, float)
INST_DOT_EX_LAUNCHER(rocblas_int, ROCBLAS_DOT_NB, true, float, float const* const*, float)
INST_DOT_EX_LAUNCHER(rocblas_int, ROCBLAS_DOT_NB, true, float, double const*, double)
INST_DOT_EX_LAUNCHER(rocblas_int, ROCBLAS_DOT_NB, true, float, double const* const*, double)
INST_DOT_EX_LAUNCHER(rocblas_int, ROCBLAS_DOT_NB, true, double, double const*, double)
INST_DOT_EX_LAUNCHER(rocblas_int, ROCBLAS_DOT_NB, true, double, double const* const*, double)

// clang-format on
