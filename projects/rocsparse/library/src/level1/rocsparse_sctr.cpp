/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <map>
#include <sstream>

#include "rocsparse_sctr.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    typedef rocsparse_status (*sctr_t)(rocsparse_handle     handle,
                                       int64_t              nnz,
                                       const void*          x_val,
                                       const void*          x_ind,
                                       void*                y,
                                       rocsparse_index_base idx_base);

    using sctr_tuple = std::tuple<rocsparse_indextype, rocsparse_datatype>;

    // clang-format off
#define SCTR_CONFIG(I, T)                                        \
    {sctr_tuple(I, T),                                           \
     sctr_template<typename rocsparse::indextype_traits<I>::type_t, typename rocsparse::datatype_traits<T>::type_t>}
    // clang-format on

    static const std::map<sctr_tuple, sctr_t> s_sctr_dispatch{
        {SCTR_CONFIG(rocsparse_indextype_i32, rocsparse_datatype_i8_r),
         SCTR_CONFIG(rocsparse_indextype_i32, rocsparse_datatype_f16_r),
         SCTR_CONFIG(rocsparse_indextype_i32, rocsparse_datatype_bf16_r),
         SCTR_CONFIG(rocsparse_indextype_i32, rocsparse_datatype_f32_r),
         SCTR_CONFIG(rocsparse_indextype_i32, rocsparse_datatype_f32_r),
         SCTR_CONFIG(rocsparse_indextype_i32, rocsparse_datatype_f64_r),
         SCTR_CONFIG(rocsparse_indextype_i32, rocsparse_datatype_f32_c),
         SCTR_CONFIG(rocsparse_indextype_i32, rocsparse_datatype_f64_c),

         SCTR_CONFIG(rocsparse_indextype_i64, rocsparse_datatype_i8_r),
         SCTR_CONFIG(rocsparse_indextype_i64, rocsparse_datatype_f16_r),
         SCTR_CONFIG(rocsparse_indextype_i64, rocsparse_datatype_bf16_r),
         SCTR_CONFIG(rocsparse_indextype_i64, rocsparse_datatype_f32_r),
         SCTR_CONFIG(rocsparse_indextype_i64, rocsparse_datatype_f64_r),
         SCTR_CONFIG(rocsparse_indextype_i64, rocsparse_datatype_f32_c),
         SCTR_CONFIG(rocsparse_indextype_i64, rocsparse_datatype_f64_c)}};

    static rocsparse_status
        sctr_find(sctr_t* function_, rocsparse_indextype i_type_, rocsparse_datatype t_type_)
    {

        const auto& it = rocsparse::s_sctr_dispatch.find(rocsparse::sctr_tuple(i_type_, t_type_));

        if(it != rocsparse::s_sctr_dispatch.end())
        {
            function_[0] = it->second;
        }
        // LCOV_EXCL_START
        else
        {
#ifndef NDEBUG
            std::cout << "invalid precision configuration: "
                      << "t_type: " << rocsparse::enum_utils::to_string(t_type_) << std::endl
                      << ", i_type: " << rocsparse::enum_utils::to_string(i_type_) << std::endl;

            std::cout << "available configuration are: " << std::endl;
            for(const auto& p : rocsparse::s_sctr_dispatch)
            {
                const auto& t      = p.first;
                const auto  i_type = std::get<0>(t);
                const auto  t_type = std::get<1>(t);
                std::cout << std::endl
                          << std::endl
                          << "t_type: " << rocsparse::enum_utils::to_string(t_type) << std::endl
                          << ", i_type: " << rocsparse::enum_utils::to_string(i_type) << std::endl;
            }
#endif

            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "t_type: " << rocsparse::enum_utils::to_string(t_type_)
                 << ", i_type: " << rocsparse::enum_utils::to_string(i_type_);

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP

        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::sctr(rocsparse_handle     handle,
                                 int64_t              nnz,
                                 rocsparse_datatype   x_datatype,
                                 const void*          x_val,
                                 rocsparse_indextype  x_indextype,
                                 const void*          x_ind,
                                 rocsparse_datatype   y_datatype,
                                 void*                y,
                                 rocsparse_index_base idx_base)
{
    ROCSPARSE_ROUTINE_TRACE;
    rocsparse::sctr_t f;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sctr_find(&f, x_indextype, x_datatype));

    RETURN_IF_ROCSPARSE_ERROR(f(handle, nnz, x_val, x_ind, y, idx_base));

    return rocsparse_status_success;
}
