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
#pragma once
#include "auto_testing_bad_arg.hpp"
#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "rocsparse.hpp"
#include "rocsparse_check.hpp"
#include "rocsparse_matrix_dense.hpp"
#include "rocsparse_matrix_factory.hpp"
#include "rocsparse_reproducibility.hpp"
#include "rocsparse_reproducibility_test_save.hpp"
#include "rocsparse_traits.hpp"
#include "rocsparse_vector_utils.hpp"
#include "utility.hpp"

namespace rocsparse_clients
{

    template <typename T>
    struct dnmat_descr
    {

    protected:
        rocsparse_dnmat_descr m_descr{};

        size_t                               m_memory_in_bytes{};
        void*                                m_device_memory{};
        void*                                m_host_memory{};
        host_dense_matrix_view<T, int64_t>   m_host_view{};
        device_dense_matrix_view<T, int64_t> m_device_view{};
        rocsparse_datatype                   m_datatype{};
        int64_t                              m_M{};
        int64_t                              m_N{};
        int64_t                              m_ld{};
        int64_t                              m_batch_count{1};
        int64_t                              m_batch_stride{0};
        rocsparse_direction                  m_batch_layout_direction{};
        rocsparse_order                      m_order{};

    public:
        int64_t                                     get_M() const;
        int64_t                                     get_N() const;
        int64_t                                     get_ld() const;
        int64_t                                     get_batch_stride() const;
        int64_t                                     get_batch_count() const;
        void                                        dzero();
        void                                        hzero();
        host_dense_matrix_view<T, int64_t>&         host();
        device_dense_matrix_view<T, int64_t>&       device();
        const host_dense_matrix_view<T, int64_t>&   host() const;
        const device_dense_matrix_view<T, int64_t>& device() const;
        ~dnmat_descr();

        void to_device();
        void to_host();
        explicit dnmat_descr(rocsparse_order     order,
                             int64_t             M,
                             int64_t             N,
                             rocsparse_direction batch_layout_direction,
                             int64_t             batch_count,
                             bool                non_zero_stride,
                             bool                init);

        void near_check_values(const host_dense_vector<int64_t>& symbolic,
                               const host_dense_vector<int64_t>& numeric);

        void print();

        void unit_check();

        operator rocsparse_dnmat_descr&();
        operator const rocsparse_dnmat_descr&() const;
    };

}
