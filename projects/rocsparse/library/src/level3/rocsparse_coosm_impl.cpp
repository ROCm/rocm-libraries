/*! \file */
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
#include "rocsparse_coosm.hpp"
#include "rocsparse_csrsm.hpp"
#include "rocsparse_utility.hpp"
#if 0


namespace rocsparse
{
    typedef rocsparse_status (*coosm_solve_t)(rocsparse_handle          handle,
                                              rocsparse_operation       trans_A,
                                              rocsparse_operation       trans_B,
                                              int64_t                   m,
                                              int64_t                   nrhs,
                                              int64_t                   nnz,
                                              const void*               alpha_device_host,
                                              const rocsparse_mat_descr descr,
                                              const void*               coo_val,
                                              const void*               coo_row_ind,
                                              const void*               coo_col_ind,
                                              void*                     B,
                                              int64_t                   ldb,
                                              rocsparse_order           order_B,
                                              rocsparse_mat_info        info,
                                              rocsparse_solve_policy    policy,
                                              void*                     temp_buffer);

    using coosm_solve_tuple = std::tuple<rocsparse_indextype, rocsparse_datatype>;

#define COOSM_SOLVE_CONFIG(I_, T_)                                                 \
    {                                                                              \
        coosm_solve_tuple(I_, T_),                                                 \
            coosm_solve_template<typename rocsparse::indextype_traits<I_>::type_t, \
                                 typename rocsparse::datatype_traits<T_>::type_t>  \
    }

    static const std::map<coosm_solve_tuple, coosm_solve_t> s_coosm_solve_dispatch{
        {COOSM_SOLVE_CONFIG(rocsparse_indextype_i32, rocsparse_datatype_f32_r),
         COOSM_SOLVE_CONFIG(rocsparse_indextype_i32, rocsparse_datatype_f64_r),
         COOSM_SOLVE_CONFIG(rocsparse_indextype_i32, rocsparse_datatype_f32_c),
         COOSM_SOLVE_CONFIG(rocsparse_indextype_i32, rocsparse_datatype_f64_c),
         COOSM_SOLVE_CONFIG(rocsparse_indextype_i64, rocsparse_datatype_f32_r),
         COOSM_SOLVE_CONFIG(rocsparse_indextype_i64, rocsparse_datatype_f64_r),
         COOSM_SOLVE_CONFIG(rocsparse_indextype_i64, rocsparse_datatype_f32_c),
         COOSM_SOLVE_CONFIG(rocsparse_indextype_i64, rocsparse_datatype_f64_c)}};

    static rocsparse_status coosm_solve_find(coosm_solve_t*      function_,
                                             rocsparse_indextype i_type_,
                                             rocsparse_datatype  t_type_)
    {
        const auto& it = rocsparse::s_coosm_solve_dispatch.find(
            rocsparse::coosm_solve_tuple(i_type_, t_type_));

        if(it != rocsparse::s_coosm_solve_dispatch.end())
        {
            function_[0] = it->second;
        }
        // LCOV_EXCL_START
        else
        {
#ifndef NDEBUG
            std::cout << "invalid precision configuration: "
                      << "i_type: " << rocsparse::to_string(i_type_) << std::endl
                      << ", t_type: " << rocsparse::to_string(t_type_) << std::endl;

            std::cout << "available configuration are: " << std::endl;
            for(const auto& p : rocsparse::s_coosm_solve_dispatch)
            {
                const auto& t      = p.first;
                const auto  i_type = std::get<0>(t);
                const auto  t_type = std::get<1>(t);
                std::cout << std::endl
                          << std::endl
                          << "i_type: " << rocsparse::to_string(i_type) << std::endl
                          << ", t_type: " << rocsparse::to_string(t_type) << std::endl;
            }
#endif

            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "i_type: " << rocsparse::to_string(i_type_)
                 << ", t_type: " << rocsparse::to_string(t_type_);

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP

        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::coosm_solve(rocsparse_handle          handle,
                                        rocsparse_operation       trans_A,
                                        rocsparse_operation       trans_B,
                                        int64_t                   m,
                                        int64_t                   nrhs,
                                        int64_t                   nnz,
                                        rocsparse_datatype        alpha_device_host_datatype,
                                        const void*               alpha_device_host,
                                        const rocsparse_mat_descr descr,
                                        rocsparse_datatype        coo_val_datatype,
                                        const void*               coo_val,
                                        rocsparse_indextype       coo_row_ind_indextype,
                                        const void*               coo_row_ind,
                                        rocsparse_indextype       coo_col_ind_indextype,
                                        const void*               coo_col_ind,
                                        rocsparse_datatype        B_datatype,
                                        void*                     B,
                                        int64_t                   ldb,
                                        rocsparse_order           order_B,
                                        rocsparse_mat_info        info,
                                        rocsparse_solve_policy    policy,
                                        void*                     temp_buffer)
{
    rocsparse::coosm_solve_t f;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::coosm_solve_find(&f, coo_row_ind_indextype, coo_val_datatype));
    RETURN_IF_ROCSPARSE_ERROR(f(handle,
                                trans_A,
                                trans_B,
                                m,
                                nrhs,
                                nnz,
                                alpha_device_host,
                                descr,
                                coo_val,
                                coo_row_ind,
                                coo_col_ind,
                                B,
                                ldb,
                                order_B,
                                info,
                                policy,
                                temp_buffer));
    return rocsparse_status_success;
}
#endif

#if 1
/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_coosm.hpp"
#include "rocsparse_csrsm.hpp"
#include "rocsparse_utility.hpp"

#include "../conversion/rocsparse_coo2csr.hpp"

template <typename I, typename T>
rocsparse_status rocsparse::coosm_solve_core(rocsparse_handle          handle,
                                             rocsparse_operation       trans_A,
                                             rocsparse_operation       trans_B,
                                             int64_t                   m_,
                                             int64_t                   nrhs_,
                                             int64_t                   nnz,
                                             const void*               alpha_device_host_,
                                             const rocsparse_mat_descr descr,
                                             const void*               coo_val_,
                                             const void*               coo_row_ind_,
                                             const void*               coo_col_ind_,
                                             void*                     B_,
                                             int64_t                   ldb,
                                             rocsparse_order           order_B,
                                             rocsparse_mat_info        info,
                                             rocsparse_solve_policy    policy,
                                             void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    const I  m                 = static_cast<I>(m_);
    const I  nrhs              = static_cast<I>(nrhs_);
    const T* alpha_device_host = reinterpret_cast<const T*>(alpha_device_host_);
    const T* coo_val           = reinterpret_cast<const T*>(coo_val_);
    const I* coo_col_ind       = reinterpret_cast<const I*>(coo_col_ind_);
    T*       B                 = reinterpret_cast<T*>(B_);

    // Buffer
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    if(std::is_same<I, int32_t>() && nnz < std::numeric_limits<int32_t>::max())
    {
        int32_t* csr_row_ptr = reinterpret_cast<int32_t*>(ptr);
        ptr += sizeof(int32_t) * (m / 256 + 1) * 256;

        const I* csr_col_ind = coo_col_ind;
        const T* csr_val     = coo_val;

        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::csrsm_solve_template<int32_t, int32_t, T>(handle,
                                                                  trans_A,
                                                                  trans_B,
                                                                  m,
                                                                  nrhs,
                                                                  nnz,
                                                                  alpha_device_host,
                                                                  descr,
                                                                  csr_val,
                                                                  csr_row_ptr,
                                                                  csr_col_ind,
                                                                  B,
                                                                  ldb,
                                                                  order_B,
                                                                  info,
                                                                  policy,
                                                                  ptr)));
        return rocsparse_status_success;
    }
    else
    {
        int64_t* csr_row_ptr = reinterpret_cast<int64_t*>(ptr);
        ptr += sizeof(int64_t) * (m / 256 + 1) * 256;

        const I* csr_col_ind = coo_col_ind;
        const T* csr_val     = coo_val;

        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::csrsm_solve_template<int64_t, int64_t, T>(handle,
                                                                  trans_A,
                                                                  trans_B,
                                                                  m,
                                                                  nrhs,
                                                                  nnz,
                                                                  alpha_device_host,
                                                                  descr,
                                                                  csr_val,
                                                                  csr_row_ptr,
                                                                  csr_col_ind,
                                                                  B,
                                                                  ldb,
                                                                  order_B,
                                                                  info,
                                                                  policy,
                                                                  ptr)));
        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::coosm_solve_quickreturn(rocsparse_handle          handle,
                                                    rocsparse_operation       trans_A,
                                                    rocsparse_operation       trans_B,
                                                    int64_t                   m,
                                                    int64_t                   nrhs,
                                                    int64_t                   nnz,
                                                    const void*               alpha_device_host,
                                                    const rocsparse_mat_descr descr,
                                                    const void*               coo_val,
                                                    const void*               coo_row_ind,
                                                    const void*               coo_col_ind,
                                                    void*                     B,
                                                    int64_t                   ldb,
                                                    rocsparse_order           order_B,
                                                    rocsparse_mat_info        info,
                                                    rocsparse_solve_policy    policy,
                                                    void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(m == 0 || nrhs == 0)
    {
        return rocsparse_status_success;
    }
    return rocsparse_status_continue;
}

#define INSTANTIATE(I, T)                                                                           \
    template rocsparse_status rocsparse::coosm_solve_core<I, T>(rocsparse_handle    handle,         \
                                                                rocsparse_operation trans_A,        \
                                                                rocsparse_operation trans_B,        \
                                                                int64_t             m,              \
                                                                int64_t             nrhs,           \
                                                                int64_t             nnz,            \
                                                                const void* alpha_device_host,      \
                                                                const rocsparse_mat_descr descr,    \
                                                                const void*               coo_val,  \
                                                                const void*            coo_row_ind, \
                                                                const void*            coo_col_ind, \
                                                                void*                  B,           \
                                                                int64_t                ldb,         \
                                                                rocsparse_order        order_B,     \
                                                                rocsparse_mat_info     info,        \
                                                                rocsparse_solve_policy policy,      \
                                                                void*                  temp_buffer);

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
#undef INSTANTIATE
#endif
