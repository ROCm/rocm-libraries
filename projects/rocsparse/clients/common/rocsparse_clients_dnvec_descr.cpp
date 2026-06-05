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

#include "rocsparse_clients_dnvec_descr.hpp"
template <typename T>
rocsparse_clients::dnvec_descr<T>::dnvec_descr(int64_t M, int64_t batch_count, int64_t stride)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;
    this->m_batch_count = batch_count;
    this->m_stride      = stride;
    this->m_host.resize(stride * batch_count);
    this->m_device.resize(stride * batch_count);
    this->m_size = M;

    rocsparse_init<T>(m_host, stride * batch_count, 1, 1);

    this->m_device.transfer_from(this->m_host);

    rocsparse_status status
        = rocsparse_create_dnvec_descr(&this->descr, M, this->m_device, get_datatype<T>());
    if(status != rocsparse_status_success)
    {
        throw(status);
    }

    status = rocsparse_dnvec_set_strided_batch(this->descr, this->m_batch_count, this->m_stride);

    if(status != rocsparse_status_success)
    {
        throw(status);
    }
}

template <typename T>
void rocsparse_clients::dnvec_descr<T>::near_check_values(
    const host_dense_vector<int64_t>& symbolic, const host_dense_vector<int64_t>& numeric)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;
    for(int64_t i = 0; i < this->m_batch_count; ++i)
    {
        if((symbolic[i] != -1) || (numeric[i] != -1))
        {
            CHECK_HIP_ERROR(
                hipMemset(&this->m_device[this->m_stride * i], 0, sizeof(T) * this->m_size));
            for(int64_t j = 0; j < this->m_size; ++j)
            {
                this->m_host[this->m_stride * i + j] = static_cast<T>(0);
            }
        }
    }
    this->m_host.near_check(this->m_device);
}

template <typename T>
rocsparse_clients::dnvec_descr<T>::operator rocsparse_dnvec_descr&()
{
    return this->descr;
}

template <typename T>
rocsparse_clients::dnvec_descr<T>::operator const rocsparse_dnvec_descr&() const
{
    return this->descr;
}

template <typename T>
host_dense_vector<T>& rocsparse_clients::dnvec_descr<T>::host()
{
    return this->m_host;
}

template <typename T>
device_dense_vector<T>& rocsparse_clients::dnvec_descr<T>::device()
{
    return this->m_device;
}

template <typename T>
const host_dense_vector<T>& rocsparse_clients::dnvec_descr<T>::host() const
{
    return this->m_host;
}

template <typename T>
const device_dense_vector<T>& rocsparse_clients::dnvec_descr<T>::device() const
{
    return this->m_device;
}

template <typename T>
int64_t rocsparse_clients::dnvec_descr<T>::get_stride() const
{
    return this->m_stride;
}

template <typename T>
int64_t rocsparse_clients::dnvec_descr<T>::get_batch_count() const
{
    return this->m_batch_count;
}

template struct rocsparse_clients::dnvec_descr<float>;
template struct rocsparse_clients::dnvec_descr<rocsparse_float_complex>;

template struct rocsparse_clients::dnvec_descr<double>;
template struct rocsparse_clients::dnvec_descr<rocsparse_double_complex>;
