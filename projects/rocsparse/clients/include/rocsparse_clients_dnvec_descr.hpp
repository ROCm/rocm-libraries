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
#include "rocsparse_graph.hpp"
#include "rocsparse_matrix_factory.hpp"
#include "rocsparse_reproducibility.hpp"
#include "rocsparse_reproducibility_test_save.hpp"
#include "rocsparse_traits.hpp"
#include "rocsparse_vector_utils.hpp"
#include "utility.hpp"

namespace rocsparse_clients
{

    template <typename T>
    struct dnvec_descr
    {
    protected:
        rocsparse_dnvec_descr  descr{};
        host_dense_vector<T>   m_host{};
        device_dense_vector<T> m_device{};
        int64_t                m_batch_count{1};
        int64_t                m_stride{0};
        int64_t                m_size{};

    public:
        host_dense_vector<T>&         host();
        device_dense_vector<T>&       device();
        const host_dense_vector<T>&   host() const;
        const device_dense_vector<T>& device() const;
        int64_t                       get_stride() const;
        int64_t                       get_batch_count() const;
        explicit dnvec_descr(int64_t M, int64_t batch_count, int64_t stride);
        void near_check_values(const host_dense_vector<int64_t>& symbolic,
                               const host_dense_vector<int64_t>& numeric);
             operator rocsparse_dnvec_descr&();
             operator const rocsparse_dnvec_descr&() const;
    };

}
