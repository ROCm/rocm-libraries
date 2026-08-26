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

#include "rocsparse_clients_dnmat_descr.hpp"
template <typename T>
void rocsparse_clients::dnmat_descr<T>::to_host()
{
    CHECK_HIP_ERROR(hipMemcpy(this->m_host_memory,
                              this->m_device_memory,
                              this->m_memory_in_bytes,
                              hipMemcpyDeviceToHost));
}

template <typename T>
void rocsparse_clients::dnmat_descr<T>::to_device()
{
    CHECK_HIP_ERROR(hipMemcpy(this->m_device_memory,
                              this->m_host_memory,
                              this->m_memory_in_bytes,
                              hipMemcpyHostToDevice));
}

template <typename T>
rocsparse_clients::dnmat_descr<T>::dnmat_descr(rocsparse_order     order,
                                               int64_t             M,
                                               int64_t             N,
                                               rocsparse_direction batch_layout_direction,
                                               int64_t             batch_count,
                                               bool                non_zero_stride,
                                               bool                init)
{

    ROCSPARSE_CLIENTS_ROUTINE_TRACE;
    int64_t global_M = M;
    int64_t global_N = N;
    int64_t ld       = std::max(int64_t(1), (order == rocsparse_order_column) ? M : N);
    int64_t stride   = 0;
    if(non_zero_stride)
    {

        switch(batch_layout_direction)
        {
        case rocsparse_direction_column:
        {
            // B = [B_0;
            //      B_1;
            //      B_2;
            //      B_3]; if row oriented B_ld = (B_n+s) and stride = B_ld * B_m
            //            if column oriented B_ld = (B_m+s) * batch_count and stride = B_ld
            //
            global_M = int64_t(M) * batch_count;
            global_N = N;
            //
            switch(order)
            {
            case rocsparse_order_column:
            {
                if(batch_count > 1)
                    ld = std::max(int64_t(1), global_M);
                stride = (batch_count > 1) ? M : 0;
                break;
            }
            case rocsparse_order_row:
            {
                if(batch_count > 1)
                    ld = std::max(int64_t(1), global_N);
                stride = (batch_count > 1) ? (int64_t(N) * M) : 0;
                break;
            }
            }
            break;
        }
        case rocsparse_direction_row:
        {
            global_M = M;
            global_N = N * batch_count;
            //
            // B = [B_0 B_1 B_2 B_3]; if column oriented B_ld = (B_m+s) and stride = B_ld * B_n
            //                        if row oriented B_ld = (B_n+s)  * batch_count and stride = B_ld
            //
            switch(order)
            {
            case rocsparse_order_column:
            {
                if(batch_count > 1)
                    ld = std::max(int64_t(1), global_M);
                stride = (batch_count > 1) ? (int64_t(N) * M) : 0;
                break;
            }
            case rocsparse_order_row:
            {
                if(batch_count > 1)
                    ld = std::max(int64_t(1), global_N);
                stride = (batch_count > 1) ? N : 0;
                break;
            }
            }
            break;
        }
        }
    }

    this->m_M        = M;
    this->m_N        = N;
    this->m_ld       = ld;
    this->m_order    = order;
    this->m_datatype = get_datatype<T>();

    this->m_batch_count            = batch_count;
    this->m_batch_stride           = stride;
    this->m_batch_layout_direction = batch_layout_direction;

    const int64_t size      = ((this->m_batch_stride == 0) ? 1 : batch_count) * M * N;
    this->m_memory_in_bytes = sizeof(T) * size;
    this->m_host_memory     = malloc(this->m_memory_in_bytes);
    auto hip_err            = hipMalloc(&this->m_device_memory, this->m_memory_in_bytes);

    if(hip_err)
    {
        throw(hip_err);
    }

    //
    // Randomize host values.
    //
    if(init)
    {

        rocsparse_init<T>(reinterpret_cast<T*>(this->m_host_memory), size, 1, 1);

        //
        // Host to device.
        //
        hip_err = hipMemcpy(
            m_device_memory, m_host_memory, this->m_memory_in_bytes, hipMemcpyHostToDevice);
        if(hip_err)
        {
            throw(hip_err);
        }
    }
    //
    // Define views.
    //
    this->m_host_view(
        this->m_M, this->m_N, reinterpret_cast<T*>(this->m_host_memory), this->m_ld, this->m_order);

    this->m_device_view(this->m_M,
                        this->m_N,
                        reinterpret_cast<T*>(this->m_device_memory),
                        this->m_ld,
                        this->m_order);

    //
    // Define rocsparse_dnmat_descr..
    //
    rocsparse_status status = rocsparse_create_dnmat_descr(&this->m_descr,
                                                           this->m_M,
                                                           this->m_N,
                                                           this->m_ld,
                                                           this->m_device_memory,
                                                           this->m_datatype,
                                                           this->m_order);
    if(status)
    {
        throw(status);
    }

    status = rocsparse_dnmat_set_strided_batch(
        this->m_descr, this->m_batch_count, this->m_batch_stride);

    if(status)
    {
        throw(status);
    }
}

template <typename T>
void rocsparse_clients::dnmat_descr<T>::near_check_values(
    const host_dense_vector<int64_t>& symbolic, const host_dense_vector<int64_t>& numeric)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    static constexpr bool verbose = false;

    for(int64_t k = 0; k < this->m_batch_count; ++k)
    {
        if(verbose)
        {
            std::cout << "batch: " << (k + 1) << "/" << this->m_batch_count << std::endl;
        }

        if((symbolic[k] == -1) && (numeric[k] == -1))
        {
            device_dense_matrix_view<T, int64_t> d(this->m_device_view.m,
                                                   this->m_device_view.n,
                                                   this->m_device_view.data()
                                                       + this->m_batch_stride * k,
                                                   this->m_device_view.ld,
                                                   this->m_device_view.order);

            host_dense_matrix_view<T, int64_t> h(this->m_host_view.m,
                                                 this->m_host_view.n,
                                                 this->m_host_view.data()
                                                     + this->m_batch_stride * k,
                                                 this->m_host_view.ld,
                                                 this->m_host_view.order);

            h.near_check(d);
        }
    }
}

template <typename T>
void rocsparse_clients::dnmat_descr<T>::print()
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;
    std::cout << "PRINT DNMAT DESCR ------------------------------------ " << std::endl;
    for(int64_t k = 0; k < this->m_batch_count; ++k)
    {
        std::cout << "batch_count  " << k << std::endl;
        device_dense_matrix_view<T, int64_t> d(this->m_device_view.m,
                                               this->m_device_view.n,
                                               this->m_device_view.data()
                                                   + this->m_batch_stride * k,
                                               this->m_device_view.ld,
                                               this->m_device_view.order);

        device_dense_matrix_view<T, int64_t> h(this->m_host_view.m,
                                               this->m_host_view.n,
                                               this->m_host_view.data() + this->m_batch_stride * k,
                                               this->m_host_view.ld,
                                               this->m_host_view.order);

        std::cout << "hprint   " << std::endl;
        h.print();
        std::cout << "dprint   " << std::endl;
        //	  d.print();
        //	  h.near_check(d);
    }
    std::cout << "PRINT DNMAT DONE DESCR ------------------------------------ " << std::endl;
}

template <typename T>
void rocsparse_clients::dnmat_descr<T>::unit_check()
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;
    const int64_t size
        = ((this->m_batch_stride == 0) ? 1 : this->m_batch_count) * this->m_M * this->m_N;
    {
        host_dense_vector<T> device_memory_on_host(size);
        CHECK_HIP_ERROR(hipMemcpy(
            device_memory_on_host, this->m_device_memory, size * sizeof(T), hipMemcpyDeviceToHost));
        unit_check_segments<T>(
            size, reinterpret_cast<const T*>(this->m_host_memory), device_memory_on_host);
    }
}

template <typename T>
host_dense_matrix_view<T, int64_t>& rocsparse_clients::dnmat_descr<T>::host()
{
    return this->m_host_view;
}

template <typename T>
device_dense_matrix_view<T, int64_t>& rocsparse_clients::dnmat_descr<T>::device()
{
    return this->m_device_view;
}

template <typename T>
const host_dense_matrix_view<T, int64_t>& rocsparse_clients::dnmat_descr<T>::host() const
{
    return this->m_host_view;
}

template <typename T>
const device_dense_matrix_view<T, int64_t>& rocsparse_clients::dnmat_descr<T>::device() const
{
    return this->m_device_view;
}

template <typename T>
void rocsparse_clients::dnmat_descr<T>::hzero()
{
    memset(this->m_host_memory, 0, this->m_memory_in_bytes);
}

template <typename T>
void rocsparse_clients::dnmat_descr<T>::dzero()
{
    CHECK_HIP_ERROR(hipMemset(this->m_device_memory, 0, this->m_memory_in_bytes));
}

template <typename T>
int64_t rocsparse_clients::dnmat_descr<T>::get_M() const
{
    return this->m_M;
}

template <typename T>
int64_t rocsparse_clients::dnmat_descr<T>::get_N() const
{
    return this->m_N;
}

template <typename T>
int64_t rocsparse_clients::dnmat_descr<T>::get_ld() const
{
    return this->m_ld;
}

template <typename T>
int64_t rocsparse_clients::dnmat_descr<T>::get_batch_stride() const
{
    return this->m_batch_stride;
}

template <typename T>
int64_t rocsparse_clients::dnmat_descr<T>::get_batch_count() const
{
    return this->m_batch_count;
}

template <typename T>
rocsparse_clients::dnmat_descr<T>::~dnmat_descr()
{
    if(this->m_host_memory)
    {
        free(this->m_host_memory);
        this->m_host_memory = nullptr;
    }

    if(this->m_device_memory)
    {
        std::ignore           = hipFree(this->m_device_memory);
        this->m_device_memory = nullptr;
    }

    if(this->m_descr)
    {
        std::ignore   = rocsparse_destroy_dnmat_descr(this->m_descr);
        this->m_descr = nullptr;
    }
}

template <typename T>
rocsparse_clients::dnmat_descr<T>::operator rocsparse_dnmat_descr&()
{
    return this->m_descr;
}

template <typename T>
rocsparse_clients::dnmat_descr<T>::operator const rocsparse_dnmat_descr&() const
{
    return this->m_descr;
}

template struct rocsparse_clients::dnmat_descr<float>;
template struct rocsparse_clients::dnmat_descr<rocsparse_float_complex>;

template struct rocsparse_clients::dnmat_descr<double>;
template struct rocsparse_clients::dnmat_descr<rocsparse_double_complex>;
