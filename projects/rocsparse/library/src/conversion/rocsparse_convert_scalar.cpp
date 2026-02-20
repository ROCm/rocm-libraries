/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_control.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_convert_scalar.hpp"
#include <map>

namespace rocsparse
{

    template <typename S, typename T>
    __host__ static void host_convert_scalar(const void* source, void* target)
    {
        *reinterpret_cast<T*>(target) = static_cast<T>(*reinterpret_cast<const S*>(source));
    }

    static const std::map<std::pair<rocsparse_datatype, rocsparse_datatype>,
                          void (*)(const void*, void*)>
        s_map_host_convert_scalar{

#define DEFINE_CASE(S, T)                                                \
    {std::pair<rocsparse_datatype, rocsparse_datatype>(S, T),            \
     host_convert_scalar<typename rocsparse::datatype_traits<S>::type_t, \
                         typename rocsparse::datatype_traits<T>::type_t>}

            DEFINE_CASE(rocsparse_datatype_u8_r, rocsparse_datatype_i8_r),
            DEFINE_CASE(rocsparse_datatype_i32_r, rocsparse_datatype_i8_r),
            DEFINE_CASE(rocsparse_datatype_u32_r, rocsparse_datatype_i8_r),

            DEFINE_CASE(rocsparse_datatype_i8_r, rocsparse_datatype_u8_r),
            DEFINE_CASE(rocsparse_datatype_i32_r, rocsparse_datatype_u8_r),
            DEFINE_CASE(rocsparse_datatype_u32_r, rocsparse_datatype_u8_r),

            DEFINE_CASE(rocsparse_datatype_i8_r, rocsparse_datatype_i32_r),
            DEFINE_CASE(rocsparse_datatype_u8_r, rocsparse_datatype_i32_r),
            DEFINE_CASE(rocsparse_datatype_u32_r, rocsparse_datatype_i32_r),

            DEFINE_CASE(rocsparse_datatype_i8_r, rocsparse_datatype_u32_r),
            DEFINE_CASE(rocsparse_datatype_u8_r, rocsparse_datatype_u32_r),
            DEFINE_CASE(rocsparse_datatype_i32_r, rocsparse_datatype_u32_r),

            DEFINE_CASE(rocsparse_datatype_i8_r, rocsparse_datatype_f32_r),
            DEFINE_CASE(rocsparse_datatype_u8_r, rocsparse_datatype_f32_r),
            DEFINE_CASE(rocsparse_datatype_i32_r, rocsparse_datatype_f32_r),
            DEFINE_CASE(rocsparse_datatype_u32_r, rocsparse_datatype_f32_r),
            DEFINE_CASE(rocsparse_datatype_f64_r, rocsparse_datatype_f32_r),

            DEFINE_CASE(rocsparse_datatype_i8_r, rocsparse_datatype_f64_r),
            DEFINE_CASE(rocsparse_datatype_u8_r, rocsparse_datatype_f64_r),
            DEFINE_CASE(rocsparse_datatype_i32_r, rocsparse_datatype_f64_r),
            DEFINE_CASE(rocsparse_datatype_u32_r, rocsparse_datatype_f64_r),
            DEFINE_CASE(rocsparse_datatype_f32_r, rocsparse_datatype_f64_r),

            DEFINE_CASE(rocsparse_datatype_i8_r, rocsparse_datatype_f32_c),
            DEFINE_CASE(rocsparse_datatype_u8_r, rocsparse_datatype_f32_c),
            DEFINE_CASE(rocsparse_datatype_i32_r, rocsparse_datatype_f32_c),
            DEFINE_CASE(rocsparse_datatype_u32_r, rocsparse_datatype_f32_c),
            DEFINE_CASE(rocsparse_datatype_f32_r, rocsparse_datatype_f32_c),
            DEFINE_CASE(rocsparse_datatype_f64_r, rocsparse_datatype_f32_c),

            DEFINE_CASE(rocsparse_datatype_i8_r, rocsparse_datatype_f64_c),
            DEFINE_CASE(rocsparse_datatype_u8_r, rocsparse_datatype_f64_c),
            DEFINE_CASE(rocsparse_datatype_i32_r, rocsparse_datatype_f64_c),
            DEFINE_CASE(rocsparse_datatype_u32_r, rocsparse_datatype_f64_c),
            DEFINE_CASE(rocsparse_datatype_f32_r, rocsparse_datatype_f64_c),
            DEFINE_CASE(rocsparse_datatype_f64_r, rocsparse_datatype_f64_c)

#undef DEFINE_CASE

        };

    host_convert_scalar_t find_host_convert_scalar(rocsparse_datatype source_datatype,
                                                   rocsparse_datatype target_datatype)
    {
        const auto& it = rocsparse::s_map_host_convert_scalar.find(
            std::pair(source_datatype, target_datatype));
        return (it != rocsparse::s_map_host_convert_scalar.end()) ? it->second : nullptr;
    }

    template <typename S, typename T>
    __launch_bounds__(64) __global__
        void convert_device_scalars_kernel(const void* source, void* target)
    {
        const size_t tid = hipBlockIdx_x * 64 + hipThreadIdx_x;
        if(tid == 0)
        {
            *reinterpret_cast<T*>(target) = static_cast<T>(*reinterpret_cast<const S*>(source));
        }
    }

    static const std::map<std::pair<rocsparse_datatype, rocsparse_datatype>,
                          void (*)(const void*, void*)>
        s_map_device_convert_scalar{

#define DEFINE_CASE(S, T)                                                          \
    {std::pair<rocsparse_datatype, rocsparse_datatype>(S, T),                      \
     convert_device_scalars_kernel<typename rocsparse::datatype_traits<S>::type_t, \
                                   typename rocsparse::datatype_traits<T>::type_t>}

            DEFINE_CASE(rocsparse_datatype_u8_r, rocsparse_datatype_i8_r),
            DEFINE_CASE(rocsparse_datatype_i32_r, rocsparse_datatype_i8_r),
            DEFINE_CASE(rocsparse_datatype_u32_r, rocsparse_datatype_i8_r),

            DEFINE_CASE(rocsparse_datatype_i8_r, rocsparse_datatype_u8_r),
            DEFINE_CASE(rocsparse_datatype_i32_r, rocsparse_datatype_u8_r),
            DEFINE_CASE(rocsparse_datatype_u32_r, rocsparse_datatype_u8_r),

            DEFINE_CASE(rocsparse_datatype_i8_r, rocsparse_datatype_i32_r),
            DEFINE_CASE(rocsparse_datatype_u8_r, rocsparse_datatype_i32_r),
            DEFINE_CASE(rocsparse_datatype_u32_r, rocsparse_datatype_i32_r),

            DEFINE_CASE(rocsparse_datatype_i8_r, rocsparse_datatype_u32_r),
            DEFINE_CASE(rocsparse_datatype_u8_r, rocsparse_datatype_u32_r),
            DEFINE_CASE(rocsparse_datatype_i32_r, rocsparse_datatype_u32_r),

            DEFINE_CASE(rocsparse_datatype_i8_r, rocsparse_datatype_f32_r),
            DEFINE_CASE(rocsparse_datatype_u8_r, rocsparse_datatype_f32_r),
            DEFINE_CASE(rocsparse_datatype_i32_r, rocsparse_datatype_f32_r),
            DEFINE_CASE(rocsparse_datatype_u32_r, rocsparse_datatype_f32_r),
            DEFINE_CASE(rocsparse_datatype_f64_r, rocsparse_datatype_f32_r),

            DEFINE_CASE(rocsparse_datatype_i8_r, rocsparse_datatype_f64_r),
            DEFINE_CASE(rocsparse_datatype_u8_r, rocsparse_datatype_f64_r),
            DEFINE_CASE(rocsparse_datatype_i32_r, rocsparse_datatype_f64_r),
            DEFINE_CASE(rocsparse_datatype_u32_r, rocsparse_datatype_f64_r),
            DEFINE_CASE(rocsparse_datatype_f32_r, rocsparse_datatype_f64_r),

            DEFINE_CASE(rocsparse_datatype_i8_r, rocsparse_datatype_f32_c),
            DEFINE_CASE(rocsparse_datatype_u8_r, rocsparse_datatype_f32_c),
            DEFINE_CASE(rocsparse_datatype_i32_r, rocsparse_datatype_f32_c),
            DEFINE_CASE(rocsparse_datatype_u32_r, rocsparse_datatype_f32_c),
            DEFINE_CASE(rocsparse_datatype_f32_r, rocsparse_datatype_f32_c),
            DEFINE_CASE(rocsparse_datatype_f64_r, rocsparse_datatype_f32_c),

            DEFINE_CASE(rocsparse_datatype_i8_r, rocsparse_datatype_f64_c),
            DEFINE_CASE(rocsparse_datatype_u8_r, rocsparse_datatype_f64_c),
            DEFINE_CASE(rocsparse_datatype_i32_r, rocsparse_datatype_f64_c),
            DEFINE_CASE(rocsparse_datatype_u32_r, rocsparse_datatype_f64_c),
            DEFINE_CASE(rocsparse_datatype_f32_r, rocsparse_datatype_f64_c),
            DEFINE_CASE(rocsparse_datatype_f64_r, rocsparse_datatype_f64_c)

#undef DEFINE_CASE

        };
    device_convert_scalar_t find_device_convert_scalar(rocsparse_datatype source_datatype,
                                                       rocsparse_datatype target_datatype)
    {
        const auto& it = rocsparse::s_map_device_convert_scalar.find(
            std::pair(source_datatype, target_datatype));
        return (it != rocsparse::s_map_device_convert_scalar.end()) ? it->second : nullptr;
    }

}
