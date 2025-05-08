/*! \file */
/* ************************************************************************
* Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_reproducibility.hpp"

#define ROCSPARSE_REPRODUCIBILITY \
    (rocsparse_reproducibility_t::instance().is_enabled() && arg.skip_reproducibility == false)

namespace rocsparse_reproducibility
{

    template <typename T>
    static void save_binary_item(rocsparse_reproducibility_t::test_data_t* data,
                                 const std::string&                        name,
                                 const device_dense_matrix_view<T>&        first)
    {
        const host_dense_matrix<T> host_first(first);
        const size_t               numbytes = sizeof(T) * host_first.m * host_first.n;
        void*                      p        = data->add(name, numbytes);
        CHECK_HIP_ERROR(hipMemcpy(p, host_first.data(), numbytes, hipMemcpyHostToHost));
    }

    template <typename T>
    static void save_binary_item(rocsparse_reproducibility_t::test_data_t* data,
                                 const std::string&                        name,
                                 const device_dense_matrix<T>&             first)
    {
        const size_t numbytes = sizeof(T) * first.m * first.n;
        void*        p        = data->add(name, numbytes);
        CHECK_HIP_ERROR(hipMemcpy(p, first.data(), numbytes, hipMemcpyDeviceToHost));
    }

    template <typename T>
    static void save_binary_item(rocsparse_reproducibility_t::test_data_t* data,
                                 const std::string&                        name,
                                 const device_vector<T>&                   first)
    {
        const size_t numbytes = sizeof(T) * first.size();
        void*        p        = data->add(name, numbytes);
        CHECK_HIP_ERROR(hipMemcpy(p, first.data(), numbytes, hipMemcpyDeviceToHost));
    }

    template <typename T>
    static void save_binary_item(rocsparse_reproducibility_t::test_data_t* data,
                                 const std::string&                        name,
                                 const managed_dense_vector<T>&            first)
    {
        const size_t numbytes = sizeof(T) * first.size();
        void*        p        = data->add(name, numbytes);
        CHECK_HIP_ERROR(hipMemcpy(p, first.data(), numbytes, hipMemcpyDeviceToHost));
    }

    template <typename T>
    static void save_binary_item(rocsparse_reproducibility_t::test_data_t* data,
                                 const std::string&                        name,
                                 const host_vector<T>&                     first)
    {
        const size_t numbytes = sizeof(T) * first.size();
        void*        p        = data->add(name, numbytes);
        CHECK_HIP_ERROR(hipMemcpy(p, first.data(), numbytes, hipMemcpyHostToHost));
    }

    template <memory_mode::value_t MODE,
              rocsparse_direction  DIRECTION,
              typename T,
              typename I,
              typename J>
    static void save_binary_item(rocsparse_reproducibility_t::test_data_t*     data,
                                 const std::string&                            name,
                                 const gebsx_matrix<MODE, DIRECTION, T, I, J>& A)
    {
        save_binary_item(data, name + "( ptr )", A.ptr);
        save_binary_item(data, name + "( ind )", A.ind);
        save_binary_item(data, name + "( val )", A.val);
    }

    template <memory_mode::value_t MODE,
              rocsparse_direction  DIRECTION,
              typename T,
              typename I,
              typename J>
    static void save_binary_item(rocsparse_reproducibility_t::test_data_t*   data,
                                 const std::string&                          name,
                                 const csx_matrix<MODE, DIRECTION, T, I, J>& A)
    {
        save_binary_item(data, name + "( ptr )", A.ptr);
        save_binary_item(data, name + "( ind )", A.ind);
        save_binary_item(data, name + "( val )", A.val);
    }

    template <memory_mode::value_t MODE, typename T, typename I>
    static void save_binary_item(rocsparse_reproducibility_t::test_data_t* data,
                                 const std::string&                        name,
                                 const coo_matrix<MODE, T, I>&             A)
    {
        save_binary_item(data, name + "( row_ind )", A.row_ind);
        save_binary_item(data, name + "( col_ind )", A.col_ind);
        save_binary_item(data, name + "( val )", A.val);
    }

    template <memory_mode::value_t MODE, typename T, typename I>
    static void save_binary_item(rocsparse_reproducibility_t::test_data_t* data,
                                 const std::string&                        name,
                                 const coo_aos_matrix<MODE, T, I>&         A)
    {
        save_binary_item(data, name + "( ind )", A.ind);
        save_binary_item(data, name + "( val )", A.val);
    }

    template <memory_mode::value_t MODE, typename T, typename I>
    static void save_binary_item(rocsparse_reproducibility_t::test_data_t* data,
                                 const std::string&                        name,
                                 const ell_matrix<MODE, T, I>&             A)
    {
        save_binary_item(data, name + "( ind )", A.ind);
        save_binary_item(data, name + "( val )", A.val);
    }

    template <typename LAST>
    static void save(const std::string& name, LAST&& last)
    {
        auto& test = rocsparse_reproducibility_t::instance().test();
        auto* data = test.get_current_data();
        save_binary_item(data, name, last);
    }

    template <typename F, typename... R>
    static void save(const std::string& name, F&& first, R&&... rest)
    {
        auto& test = rocsparse_reproducibility_t::instance().test();
        auto* data = test.get_current_data();
        save_binary_item(data, name, first);
        save(rest...);
    }

}
