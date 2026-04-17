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

#include "rocsparse_clients_spmat_descr.hpp"

template <class... Ts>
struct select_lambdas : Ts...
{
    using Ts::operator()...;
};
template <class... Ts>
select_lambdas(Ts...) -> select_lambdas<Ts...>;

template <typename T, typename I, typename J>
const device_dense_vector<T>& rocsparse_clients::spmat_descr<T, I, J>::get_device_values() const
{
    return std::visit(
        [](const auto& d) -> const auto& { return d.device().val; }, this->m_data);
}

template <typename T, typename I, typename J>
const host_vector<T>& rocsparse_clients::spmat_descr<T, I, J>::get_host_values() const
{
    return std::visit(
        [](const auto& d) -> const auto& { return d.host().val; }, this->m_data);
}

template <typename T, typename I, typename J>
device_dense_vector<T>& rocsparse_clients::spmat_descr<T, I, J>::get_device_values()
{
    return std::visit(
        [](auto& d) -> auto& { return d.device().val; }, this->m_data);
}

template <typename T, typename I, typename J>
host_vector<T>& rocsparse_clients::spmat_descr<T, I, J>::get_host_values()
{
    return std::visit(
        [](auto& d) -> auto& { return d.host().val; }, this->m_data);
}

template <typename T, typename I, typename J>
int64_t rocsparse_clients::spmat_descr<T, I, J>::get_batch_count() const
{
    return this->m_batch_count;
}

template <typename T, typename I, typename J>
rocsparse_format rocsparse_clients::spmat_descr<T, I, J>::get_format() const
{
    return this->m_format;
}

template <typename T, typename I, typename J>
int64_t rocsparse_clients::spmat_descr<T, I, J>::get_stride() const
{
    return this->m_batch_val_stride;
}

template <typename T, typename I, typename J>
rocsparse_clients::spmat_descr<T, I, J>::~spmat_descr()
{
    if(this->m_local_descr != nullptr)
    {
        delete this->m_local_descr;
        this->m_local_descr = nullptr;
    }
}

template <typename T>
void batch_count_near_check(int64_t                 batch_count,
                            int64_t                 data_size,
                            host_dense_vector<T>&   host_data,
                            int64_t                 host_data_stride,
                            device_dense_vector<T>& device_data,
                            int64_t                 device_data_stride,
                            const int64_t* __restrict__ symbolic,
                            const int64_t* __restrict__ numeric)
{
    for(int64_t i = 0; i < batch_count; ++i)
    {
        if((symbolic[i] != -1) || (numeric[i] != -1))
        {
            CHECK_HIP_ERROR(
                hipMemset(&device_data[device_data_stride * i], 0, sizeof(T) * data_size));
            std::ignore = memset(&host_data[host_data_stride * i], 0, sizeof(T) * data_size);
        }
    }
    host_data.near_check(device_data);
}

template <typename T>
void batch_count_near_check(int64_t                 batch_count,
                            int64_t                 data_size,
                            host_vector<T>&         host_data,
                            int64_t                 host_data_stride,
                            device_dense_vector<T>& device_data,
                            int64_t                 device_data_stride,
                            const int64_t* __restrict__ symbolic,
                            const int64_t* __restrict__ numeric)
{
    for(int64_t i = 0; i < batch_count; ++i)
    {
        if((symbolic[i] != -1) || (numeric[i] != -1))
        {
            CHECK_HIP_ERROR(
                hipMemset(&device_data[device_data_stride * i], 0, sizeof(T) * data_size));
            std::ignore = memset(&host_data[host_data_stride * i], 0, sizeof(T) * data_size);
        }
    }
    host_data.near_check(device_data);
}

template <typename T, typename I, typename J>
void rocsparse_clients::spmat_descr<T, I, J>::near_check_values(
    const host_dense_vector<int64_t>& symbolic, const host_dense_vector<int64_t>& numeric)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    auto& host_values   = this->get_host_values();
    auto& device_values = this->get_device_values();
    if(this->m_batch_count == 1)
    {
        if((symbolic[0] == -1) && (numeric[0] == -1))
        {
            host_values.near_check(device_values);
        }
    }
    else
    {
        batch_count_near_check(this->m_batch_count,
                               this->get_size_values(),
                               host_values,
                               this->m_batch_val_stride,
                               device_values,
                               this->m_batch_val_stride,
                               symbolic,
                               numeric);
    }
}

template <typename T, typename I, typename J>
bool rocsparse_clients::spmat_descr<T, I, J>::is_square() const
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    return std::visit(
        select_lambdas{
            [](const coo_t& that) -> bool { return (that.host().m == that.host().n); },
            [](const coo_aos_t& that) -> bool { return (that.host().m == that.host().n); },
            [](const csr_t& that) -> bool { return (that.host().m == that.host().n); },
            [](const csc_t& that) -> bool { return (that.host().m == that.host().n); },
            [](const ell_t& that) -> bool { return (that.host().m == that.host().n); },
            [](const bell_t& that) -> bool { return (that.host().m == that.host().n); },
            [](const bsr_t& that) -> bool { return (that.host().mb == that.host().nb); },
            [](const sell_t& that) -> bool { return (that.host().m == that.host().n); }},
        this->m_data);
    return true;
}

template <typename T, typename I, typename J>
void rocsparse_clients::spmat_descr<T, I, J>::reinit_values()
{
    this->get_device_values().transfer_from(this->get_host_values());
}

template <typename T, typename I, typename J>
int64_t rocsparse_clients::spmat_descr<T, I, J>::get_nrows() const
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;
    return std::visit(
        select_lambdas{
            [](const coo_t& that) -> int64_t { return that.host().m; },
            [](const coo_aos_t& that) -> int64_t { return that.host().m; },
            [](const csr_t& that) -> int64_t { return that.host().m; },
            [](const csc_t& that) -> int64_t { return that.host().m; },
            [](const ell_t& that) -> int64_t { return that.host().m; },
            [](const bell_t& that) -> int64_t { return that.host().m; },
            [](const bsr_t& that) -> int64_t { return that.host().mb * that.host().row_block_dim; },
            [](const sell_t& that) -> int64_t { return that.host().m; },
        },
        this->m_data);
}

template <typename T, typename I, typename J>
int64_t rocsparse_clients::spmat_descr<T, I, J>::get_size_values() const
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;
    return std::visit(
        select_lambdas{
            [](const coo_t& that) -> int64_t { return that.host().nnz; },
            [](const coo_aos_t& that) -> int64_t { return that.host().nnz; },
            [](const csr_t& that) -> int64_t { return that.host().nnz; },
            [](const csc_t& that) -> int64_t { return that.host().nnz; },
            [](const ell_t& that) -> int64_t { return that.host().m * that.host().width; },
            [](const bell_t& that) -> int64_t {
                return that.host().width * that.host().m * that.host().bdim * that.host().bdim;
            },
            [](const bsr_t& that) -> int64_t {
                return that.host().nnzb * that.host().row_block_dim * that.host().col_block_dim;
            },
            [](const sell_t& that) -> int64_t { return that.host().sell_colval_size; },
        },
        this->m_data);
}

template <typename T, typename I, typename J>
int64_t rocsparse_clients::spmat_descr<T, I, J>::get_size_cols() const
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;
    return std::visit(
        select_lambdas{
            [](const coo_t& that) -> int64_t { return that.host().nnz; },
            [](const coo_aos_t& that) -> int64_t { return that.host().nnz; },
            [](const csr_t& that) -> int64_t { return that.host().nnz; },
            [](const csc_t& that) -> int64_t { return that.host().n + 1; },
            [](const ell_t& that) -> int64_t { return that.host().m * that.host().width; },
            [](const bell_t& that) -> int64_t { return that.host().width * that.host().m; },
            [](const bsr_t& that) -> int64_t { return that.host().nnzb; },
            [](const sell_t& that) -> int64_t { return that.host().sell_colval_size; },
        },
        this->m_data);
}

template <typename T, typename I, typename J>
int64_t rocsparse_clients::spmat_descr<T, I, J>::get_size_rows() const
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;
    return std::visit(select_lambdas{
                          [](const coo_t& that) -> int64_t { return that.host().nnz; },
                          [](const coo_aos_t& that) -> int64_t { return that.host().nnz; },
                          [](const csr_t& that) -> int64_t { return that.host().m + 1; },
                          [](const csc_t& that) -> int64_t { return that.host().nnz; },
                          [](const ell_t& that) -> int64_t { return 0; },
                          [](const bell_t& that) -> int64_t { return 0; },
                          [](const bsr_t& that) -> int64_t { return that.host().mb + 1; },
                          [](const sell_t& that) -> int64_t {
                              return (that.host().m - 1) / that.host().sell_slice_size + 1;
                          },
                      },
                      this->m_data);
}

template <typename T, typename I, typename J>
rocsparse_clients::spmat_descr<T, I, J>::spmat_descr(const Arguments& arg,
                                                     int64_t          batch_count,
                                                     bool             full_rank)
    : m_data(make_variant(arg.formatA))
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    const rocsparse_format format = arg.formatA;
    this->m_format                = format;
    this->init(arg, batch_count, full_rank);
}

template <typename T, typename I, typename J>
void rocsparse_clients::spmat_descr<T, I, J>::init(const Arguments& arg,
                                                   int64_t          batch_count,
                                                   bool             full_rank)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;
    //
    // Build host matrix.
    //
    std::visit(
        select_lambdas{[&](coo_t& that) {
                           const bool                        to_int = arg.timing ? false : true;
                           rocsparse_matrix_factory<T, I, I> matrix_factory(arg, to_int, full_rank);
                           matrix_factory.init_coo(that.host());
                       },

                       [&](csr_t& that) {
                           const bool                        to_int = arg.timing ? false : true;
                           rocsparse_matrix_factory<T, I, J> matrix_factory(arg, to_int, full_rank);
                           matrix_factory.init_csr(that.host());
                       },

                       [&](bsr_t& that) {
                           static constexpr bool             toint = false;
                           rocsparse_matrix_factory<T, I, J> matrix_factory(arg, toint, false);

                           {
                               J                   M         = arg.M;
                               J                   N         = arg.N;
                               J                   block_dim = arg.block_dim;
                               rocsparse_direction direction = arg.direction;
                               J                   Mb        = (M + block_dim - 1) / block_dim;
                               J                   Nb        = (N + block_dim - 1) / block_dim;
                               auto&               host      = that.host();
                               I                   nnzb      = arg.nnz;
                               matrix_factory.init_bsr(host.ptr,
                                                       host.ind,
                                                       host.val,
                                                       direction,
                                                       Mb,
                                                       Nb,
                                                       nnzb,
                                                       block_dim,
                                                       arg.baseA);

                               host.mb              = Mb;
                               host.nb              = Nb;
                               host.nnzb            = nnzb;
                               host.row_block_dim   = block_dim;
                               host.col_block_dim   = block_dim;
                               host.block_direction = arg.direction;
                               host.base            = arg.baseA;
                           }
                       },
                       [&](coo_aos_t& that) {
                           std::cerr << "handling coo_aos not yet implemented" << std::endl;
                           throw(rocsparse_status_not_implemented);
                       },
                       [&](csc_t& that) {
                           std::cerr << "handling csc not yet implemented" << std::endl;
                           throw(rocsparse_status_not_implemented);
                       },
                       [&](ell_t& that) {
                           std::cerr << "handling ell not yet implemented" << std::endl;
                           throw(rocsparse_status_not_implemented);
                       },
                       [&](bell_t& that) {
                           std::cerr << "handling bell not yet implemented" << std::endl;
                           throw(rocsparse_status_not_implemented);
                       },
                       [&](sell_t& that) {
                           std::cerr << "handling sell not yet implemented" << std::endl;
                           throw(rocsparse_status_not_implemented);
                       }},
        this->m_data);

    //
    // Build device matrix.
    //
    std::visit(
        [&](auto& that) {
            auto& device = that.device();
            device(that.host());
        },
        this->m_data);

    const int64_t stride_shift      = 0;
    const double  random_multiplier = 1.0e-6;
    int64_t       stride            = 0;
    this->m_batch_count             = batch_count;
    if(batch_count > 1)
    {
        auto&        host_values               = this->get_host_values();
        const size_t host_values_size          = host_values.size();
        const size_t host_values_size_in_bytes = sizeof(T) * host_values_size;
        stride                                 = host_values_size + stride_shift;
        this->m_batch_val_stride               = stride;

        host_dense_vector<char> buffer(host_values_size_in_bytes);
        CHECK_HIP_ERROR(
            hipMemcpy(buffer, host_values, host_values_size_in_bytes, hipMemcpyDefault));

        const int64_t size = batch_count * stride;
        host_values.resize(size);
        memset(host_values, 255 - 1, sizeof(T) * size);
        CHECK_HIP_ERROR(
            hipMemcpy(host_values, buffer, host_values_size_in_bytes, hipMemcpyDefault));
        for(int64_t i = 1; i < batch_count; ++i)
        {
            CHECK_HIP_ERROR(hipMemcpy(
                &host_values[i * stride], buffer, host_values_size_in_bytes, hipMemcpyDefault));
            T* p = &host_values[i * stride];

            for(int64_t j = 0; j < host_values_size; ++j)
            {
                p[j] = p[j] * (1.0 + random_cached_generator<float>(1.0, 2.0) * random_multiplier);
            }
        }

        auto& device_values = this->get_device_values();
        device_values.resize(size);
        device_values.transfer_from(host_values);
    }

    //
    // Create local spmat.
    //
    std::visit([&](auto& that) { this->m_local_descr = new rocsparse_local_spmat(that.device()); },
               this->m_data);

    if(batch_count > 1)
    {
        //
        // To fix, exposed C-API is inconsistent.
        //
        switch(this->get_format())
        {
        case rocsparse_format_coo:
        case rocsparse_format_bell:
        case rocsparse_format_coo_aos:
        case rocsparse_format_ell:
        {
            CHECK_ROCSPARSE_ERROR(
                rocsparse_coo_set_strided_batch(this->m_local_descr[0], batch_count, stride));
            break;
        }
        case rocsparse_format_csr:
        case rocsparse_format_sell:
        case rocsparse_format_bsr:
        {
            CHECK_ROCSPARSE_ERROR(
                rocsparse_csr_set_strided_batch(this->m_local_descr[0], batch_count, 0, stride));
            break;
        }
        case rocsparse_format_csc:
        {
            CHECK_ROCSPARSE_ERROR(
                rocsparse_csc_set_strided_batch(this->m_local_descr[0], batch_count, 0, stride));
            break;
        }
        }
    }
}

template <typename T, typename I, typename J>
rocsparse_clients::spmat_descr<T, I, J>::operator rocsparse_spmat_descr&()
{
    return this->m_local_descr[0];
}

template <typename T, typename I, typename J>
rocsparse_clients::spmat_descr<T, I, J>::operator const rocsparse_spmat_descr&() const
{
    return this->m_local_descr[0];
}

template struct rocsparse_clients::spmat_descr<float, int32_t, int32_t>;
template struct rocsparse_clients::spmat_descr<double, int32_t, int32_t>;
template struct rocsparse_clients::spmat_descr<rocsparse_float_complex, int32_t, int32_t>;
template struct rocsparse_clients::spmat_descr<rocsparse_double_complex, int32_t, int32_t>;

template struct rocsparse_clients::spmat_descr<float, int64_t, int32_t>;
template struct rocsparse_clients::spmat_descr<double, int64_t, int32_t>;
template struct rocsparse_clients::spmat_descr<rocsparse_float_complex, int64_t, int32_t>;
template struct rocsparse_clients::spmat_descr<rocsparse_double_complex, int64_t, int32_t>;

template struct rocsparse_clients::spmat_descr<float, int64_t, int64_t>;
template struct rocsparse_clients::spmat_descr<double, int64_t, int64_t>;
template struct rocsparse_clients::spmat_descr<rocsparse_float_complex, int64_t, int64_t>;
template struct rocsparse_clients::spmat_descr<rocsparse_double_complex, int64_t, int64_t>;
