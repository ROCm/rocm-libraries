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

    template <typename T, typename I, typename J = I>
    struct spmat_descr
    {

    public:
        rocsparse_spmat_descr descr{};
        rocsparse_format      m_format{};

        device_coo_aos_matrix<T, I>  m_device_coo_aos{};
        device_coo_matrix<T, I>      m_device_coo{};
        device_csc_matrix<T, I, J>   m_device_csc{};
        device_csr_matrix<T, I, J>   m_device_csr{};
        device_gebsr_matrix<T, I, J> m_device_gebsr{};
        device_ell_matrix<T, I>      m_device_ell{};
        device_sell_matrix<T, I, J>  m_device_sell{};

        host_csr_matrix<T, I, J>   m_host_csr{};
        host_gebsr_matrix<T, I, J> m_host_gebsr{};

        int64_t                m_batch_count{1};
        int64_t                m_val_stride{0};
        host_dense_vector<T>   m_host_val{};
        device_dense_vector<T> m_device_val{};

    public:
        T* get_batched_host_val(int64_t i)
        {
            switch(this->m_format)
            {
            case rocsparse_format_csr:
            {
                if(this->m_batch_count > 1)
                {
                    return &this->m_host_val[i * this->m_val_stride];
                }
                else
                {
                    return this->m_host_csr.val;
                }
            }

            case rocsparse_format_bsr:
            {
                if(this->m_batch_count > 1)
                {
                    return &this->m_host_val[i * this->m_val_stride];
                }
                else
                {
                    return this->m_host_gebsr.val;
                }
            }
            case rocsparse_format_coo:
            case rocsparse_format_csc:
            case rocsparse_format_coo_aos:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            case rocsparse_format_sell:
            {
                return nullptr;
            }
            }
        }

        void near_check_values(const host_dense_vector<int64_t>& symbolic,
                               const host_dense_vector<int64_t>& numeric)
        {
            ROCSPARSE_CLIENTS_ROUTINE_TRACE;
            switch(this->m_format)
            {
            case rocsparse_format_csr:
            {
                if(this->m_batch_count == 1)
                {
                    if((symbolic[0] == -1) && (numeric[0] == -1))
                    {
                        this->m_host_csr.val.near_check(this->m_device_csr.val);
                    }
                }
                else
                {
                    const int64_t nnz = this->m_device_csr.nnz;
                    for(int64_t i = 0; i < this->m_batch_count; ++i)
                    {
                        if((symbolic[i] != -1) || (numeric[i] != -1))
                        {
                            std::ignore = hipMemset(&this->m_device_val[this->m_val_stride * i],
                                                    0,
                                                    sizeof(T) * this->m_device_csr.nnz);
                            for(int64_t j = 0; j < nnz; ++j)
                            {
                                this->m_host_val[this->m_val_stride * i + j] = static_cast<T>(0);
                            }
                        }
                    }
                    this->m_host_val.near_check(this->m_device_val);
                }

                break;
            }

            case rocsparse_format_bsr:
            {
                if(this->m_batch_count == 1)
                {
                    if((symbolic[0] == -1) && (numeric[0] == -1))
                    {
                        this->m_host_gebsr.val.near_check(this->m_device_gebsr.val);
                    }
                }
                else
                {
                    const int64_t nnz = this->m_device_gebsr.nnzb
                                        * this->m_device_gebsr.row_block_dim
                                        * this->m_device_gebsr.col_block_dim;
                    for(int64_t i = 0; i < this->m_batch_count; ++i)
                    {
                        if((symbolic[i] != -1) || (numeric[i] != -1))
                        {
                            std::ignore = hipMemset(
                                &this->m_device_val[this->m_val_stride * i], 0, sizeof(T) * nnz);
                            for(int64_t j = 0; j < nnz; ++j)
                            {
                                this->m_host_val[this->m_val_stride * i + j] = static_cast<T>(0);
                            }
                        }
                    }
                    this->m_host_val.near_check(this->m_device_val);
                }
                break;
            }

            case rocsparse_format_csc:
            case rocsparse_format_coo:
            case rocsparse_format_coo_aos:
            case rocsparse_format_bell:
            case rocsparse_format_ell:
            case rocsparse_format_sell:
            {
                break;
            }
            }
        }

        bool is_square() const
        {
            ROCSPARSE_CLIENTS_ROUTINE_TRACE;
            switch(this->m_format)
            {

            case rocsparse_format_csr:
            {
                return (this->m_device_csr.m == this->m_device_csr.n);
            }

            case rocsparse_format_bsr:
            {
                return (
                    (this->m_device_gebsr.mb == this->m_device_gebsr.nb)
                    && (this->m_device_gebsr.row_block_dim == this->m_device_gebsr.col_block_dim));
            }

            case rocsparse_format_csc:
            case rocsparse_format_coo:
            case rocsparse_format_coo_aos:
            case rocsparse_format_bell:
            case rocsparse_format_ell:
            case rocsparse_format_sell:
            {
                return true;
            }
            }
        }

        void set_randomized_batch(int64_t batch_count,
                                  int64_t stride_shift,
                                  double  random_multiplier)
        {
            this->m_batch_count = batch_count;
            if(batch_count == 1)
                return;
            switch(this->m_format)
            {
            case rocsparse_format_csr:
            {
                if(batch_count > 1)
                {
                    int64_t nnz        = this->m_device_csr.nnz;
                    int64_t stride     = nnz + stride_shift;
                    this->m_val_stride = stride;

                    this->m_host_val.resize(batch_count * stride);
                    this->m_device_val.resize(batch_count * stride);
                    memset(this->m_host_val, 255 - 1, sizeof(T) * m_host_val.size());
                    for(int i = 0; i < batch_count; ++i)
                    {
                        T* p = &this->m_host_val[i * stride];
                        CHECK_HIP_ERROR(hipMemcpy(
                            p, this->m_device_csr.val, sizeof(T) * nnz, hipMemcpyDefault));
                        if(i > 0)
                        {
                            for(int64_t j = 0; j < nnz; ++j)
                            {
                                p[j] = p[j]
                                       * (1.0
                                          + random_cached_generator<float>(1.0, 2.0)
                                                * random_multiplier);
                            }
                        }
                    }

                    this->m_device_val.transfer_from(this->m_host_val);
                    CHECK_ROCSPARSE_ERROR(
                        rocsparse_csr_set_strided_batch(this->descr, batch_count, 0, stride));

                    if(this->m_device_val != nullptr)
                        CHECK_ROCSPARSE_ERROR(rocsparse_csr_set_pointers(this->descr,
                                                                         this->m_device_csr.ptr,
                                                                         this->m_device_csr.ind,
                                                                         this->m_device_val));
                }
                break;
            }

            case rocsparse_format_bsr:
            {
                if(batch_count > 1)
                {
                    const int64_t nnz = int64_t(this->m_device_gebsr.nnzb)
                                        * this->m_device_gebsr.row_block_dim
                                        * this->m_device_gebsr.row_block_dim;
                    int64_t stride     = nnz + stride_shift;
                    this->m_val_stride = stride;
                    this->m_host_val.resize(batch_count * stride);
                    this->m_device_val.resize(batch_count * stride);

                    memset(this->m_host_val, 255 - 1, sizeof(T) * m_host_val.size());
                    for(int i = 0; i < batch_count; ++i)
                    {
                        T* p = &this->m_host_val[i * stride];
                        CHECK_HIP_ERROR(hipMemcpy(
                            p, this->m_device_gebsr.val, sizeof(T) * nnz, hipMemcpyDefault));
                        if(i > 0)
                        {
                            for(int64_t j = 0; j < nnz; ++j)
                            {
                                p[j] = p[j]
                                       * (1.0
                                          + random_cached_generator<float>(1.0, 2.0)
                                                * random_multiplier);
                            }
                        }
                    }
                    this->m_device_val.transfer_from(this->m_host_val);

                    CHECK_ROCSPARSE_ERROR(
                        rocsparse_csr_set_strided_batch(this->descr, batch_count, 0, stride));
                    if(this->m_device_val != nullptr)
                        CHECK_ROCSPARSE_ERROR(rocsparse_bsr_set_pointers(this->descr,
                                                                         this->m_device_gebsr.ptr,
                                                                         this->m_device_gebsr.ind,
                                                                         this->m_device_val));
                }

                break;
            }

            case rocsparse_format_ell:
            case rocsparse_format_sell:
            case rocsparse_format_bell:
            case rocsparse_format_coo:
            case rocsparse_format_coo_aos:
            case rocsparse_format_csc:
            {
                break;
            }
            }
        }

        explicit spmat_descr(const Arguments& arg, bool full_rank = false)
        {
            ROCSPARSE_CLIENTS_ROUTINE_TRACE;
            const rocsparse_format format = arg.formatA;
            this->m_format                = format;
            switch(format)
            {

            case rocsparse_format_coo:
            case rocsparse_format_coo_aos:
            case rocsparse_format_csc:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            case rocsparse_format_sell:
            {
                break;
            }

            case rocsparse_format_csr:
            {
                const bool                        to_int = arg.timing ? false : true;
                rocsparse_matrix_factory<T, I, J> matrix_factory(arg, to_int, full_rank);
                matrix_factory.init_csr(this->m_host_csr);
                this->m_device_csr(this->m_host_csr);
                const rocsparse_status status = rocsparse_create_csr_descr(&this->descr,
                                                                           this->m_device_csr.m,
                                                                           this->m_device_csr.n,
                                                                           this->m_device_csr.nnz,
                                                                           this->m_device_csr.ptr,
                                                                           this->m_device_csr.ind,
                                                                           this->m_device_csr.val,
                                                                           get_indextype<I>(),
                                                                           get_indextype<J>(),
                                                                           this->m_device_csr.base,
                                                                           get_datatype<T>());

                if(status != rocsparse_status_success)
                {
                    throw(status);
                }
                break;
            }

            case rocsparse_format_bsr:
            {
                static constexpr bool             toint = false;
                rocsparse_matrix_factory<T, I, J> matrix_factory(arg, toint, false);

                {
                    J                   M         = arg.M;
                    J                   N         = arg.N;
                    J                   block_dim = arg.block_dim;
                    rocsparse_direction direction = arg.direction;
                    J                   Mb        = (M + block_dim - 1) / block_dim;
                    J                   Nb        = (N + block_dim - 1) / block_dim;

                    I nnzb = arg.nnz;
                    matrix_factory.init_bsr(this->m_host_gebsr.ptr,
                                            this->m_host_gebsr.ind,
                                            this->m_host_gebsr.val,
                                            direction,
                                            Mb,
                                            Nb,
                                            nnzb,
                                            block_dim,
                                            arg.baseA);

                    this->m_host_gebsr.mb              = Mb;
                    this->m_host_gebsr.nb              = Nb;
                    this->m_host_gebsr.nnzb            = nnzb;
                    this->m_host_gebsr.row_block_dim   = block_dim;
                    this->m_host_gebsr.col_block_dim   = block_dim;
                    this->m_host_gebsr.block_direction = arg.direction;
                    this->m_host_gebsr.base            = arg.baseA;
                    this->m_device_gebsr(this->m_host_gebsr);
                }

                const rocsparse_status status
                    = rocsparse_create_bsr_descr(&this->descr,
                                                 this->m_device_gebsr.mb,
                                                 this->m_device_gebsr.nb,
                                                 this->m_device_gebsr.nnzb,
                                                 this->m_device_gebsr.block_direction,
                                                 this->m_device_gebsr.row_block_dim,
                                                 this->m_device_gebsr.ptr,
                                                 this->m_device_gebsr.ind,
                                                 this->m_device_gebsr.val,
                                                 get_indextype<I>(),
                                                 get_indextype<J>(),
                                                 this->m_device_gebsr.base,
                                                 get_datatype<T>());
                if(status != rocsparse_status_success)
                {
                    throw(status);
                }
                break;
            }
            }

            //
            // Iniitialize batch_count;
            //
            const int64_t stride_shift          = 0;
            const double  randomized_multiplier = 1.0e-6;
            int64_t       batch_count           = arg.batch_count;
            if(batch_count == -1)
                batch_count = 1;

            set_randomized_batch(batch_count, stride_shift, randomized_multiplier);
        }

        operator rocsparse_spmat_descr&()
        {
            return this->descr;
        }
        operator const rocsparse_spmat_descr&() const
        {
            return this->descr;
        }
    };

}
