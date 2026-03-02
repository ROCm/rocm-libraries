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

#include "testing.hpp"

namespace rocsparse_clients
{

    template <typename T>
    struct dnvec_descr
    {
    public:
        rocsparse_dnvec_descr  descr{};
        host_dense_vector<T>   m_host{};
        device_dense_vector<T> m_device{};
        int64_t                m_batch_count{1};
        int64_t                m_stride{0};
        int64_t                m_size{};

    public:
        T* get_batched_host_val(int64_t i)
        {
            if(this->m_batch_count > 1)
            {
                return &this->m_host[i * this->m_stride];
            }
            else
            {
                return this->m_host.val;
            }
        }

        explicit dnvec_descr(int64_t M, int64_t batch_count, int64_t stride)
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

            status = rocsparse_dnvec_set_strided_batch(
                this->descr, this->m_batch_count, this->m_stride);

            if(status != rocsparse_status_success)
            {
                throw(status);
            }
        }

        void near_check_values(const host_dense_vector<int64_t>& symbolic,
                               const host_dense_vector<int64_t>& numeric)
        {
            ROCSPARSE_CLIENTS_ROUTINE_TRACE;
            for(int64_t i = 0; i < this->m_batch_count; ++i)
            {
                if((symbolic[i] != -1) || (numeric[i] != -1))
                {
                    std::ignore = hipMemset(
                        &this->m_device[this->m_stride * i], 0, sizeof(T) * this->m_size);
                    for(int64_t j = 0; j < this->m_size; ++j)
                    {
                        this->m_host[this->m_stride * i + j] = static_cast<T>(0);
                    }
                }
            }
            this->m_host.near_check(this->m_device);
        }

        operator rocsparse_dnvec_descr&()
        {
            return this->descr;
        }

        operator const rocsparse_dnvec_descr&() const
        {
            return this->descr;
        }
    };

}
