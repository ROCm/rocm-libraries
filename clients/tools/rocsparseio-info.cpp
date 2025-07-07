//
// Copyright (C) 2024-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#include "app.hpp"
#include "rocsparseio.h"
#include <algorithm>
#include <chrono>
#include <complex>
#include <iostream>
#include <math.h>
using rocsparseio_float_complex  = std::complex<float>;
using rocsparseio_double_complex = std::complex<double>;

typedef struct
{
    uint64_t mean_nnz_per_seq;
    uint64_t min_nnz_per_seq;
    uint64_t max_nnz_per_seq;
    uint64_t median_nnz_per_seq;
    bool     full_diagonal;
    bool     symbolic_symetric;
    bool     numeric_symetric;

} rocsparseio_statistics_csx;

template <typename J, typename T>
struct transpose_pair_t
{
    J                        index{-1};
    const T*                 val;
    struct transpose_pair_t* next{};
    transpose_pair_t(J i, const T* v, struct transpose_pair_t* n = nullptr)
        : index(i)
        , val(v)
        , next(n){};
    ~transpose_pair_t()
    {
        index = -1;
        val   = nullptr;
        next  = nullptr;
    }
};

template <typename T>
double abs(T val)
{
    return std::abs(val);
}

template <>
double abs(_Float16 val)
{
    return std::abs((double)val);
}

template <typename I, typename J, typename T>
rocsparseio_status rocsparseio_csx_statistics(rocsparseio_direction dir,
                                              uint64_t              M,
                                              uint64_t              N,
                                              uint64_t              nnz,
                                              const I* __restrict__ ptr_,
                                              const J* __restrict__ ind_,
                                              const T* __restrict__ val_,
                                              rocsparseio_index_base base)
{
    uint64_t K         = M;
    uint64_t len_trseq = N;
    switch(base)
    {
    case rocsparseio_index_base_zero:
    {
        if(ptr_[0] != 0)
        {
            std::cerr << "inconsistent matrix with index base" << std::endl;
            return rocsparseio_status_invalid_value;
        }
        break;
    }
    case rocsparseio_index_base_one:
    {
        if(ptr_[0] != 1)
        {
            std::cerr << "inconsistent matrix with index base" << std::endl;
            return rocsparseio_status_invalid_value;
        }
        break;
    }
    }

    switch(dir)
    {
    case rocsparseio_direction_row:
    {
        break;
    }
    case rocsparseio_direction_column:
    {
        K         = N;
        len_trseq = M;
        break;
    }
    }

    using tr_t    = transpose_pair_t<J, T>;
    tr_t** trseqs = new tr_t*[len_trseq];
    for(uint64_t i = 0; i < len_trseq; ++i)
    {
        trseqs[i] = nullptr;
    }

    {
        uint64_t k = K;
        do
        {
            --k;
            for(uint64_t l = ptr_[k] - base; l < ptr_[k + 1] - base; ++l)
            {
                J ind = ind_[l] - base;
                if(ind < 0 || ind > len_trseq)
                {
                    std::cerr << "inconsistent matrix with overflow of indices" << std::endl;
                    return rocsparseio_status_invalid_value;
                }

                if(trseqs[ind] == nullptr)
                {
                    trseqs[ind] = new tr_t(k, &val_[l]);
                }
                else
                {
                    auto* q     = trseqs[ind];
                    trseqs[ind] = new tr_t(k, &val_[l], q);
                }
            }
        } while(k > 0);
    }

    {
        uint64_t* count = new uint64_t[K];
        for(uint64_t k = 0; k < K; ++k)
        {
            count[k] = ptr_[k + 1] - ptr_[k];
        }
        std::sort(count, count + K);

        std::cout << "min nnz per row: " << count[0] << std::endl;
        std::cout << "max nnz per row: " << count[K - 1] << std::endl;
        if(K % 2 == 0)
        {
            std::cout << "median nnz per row: " << count[(K - 1) / 2 - 1] << " "
                      << count[(K - 1) / 2] << std::endl;
        }
        else
        {
            std::cout << "median nnz per row: " << count[(K - 1) / 2] << std::endl;
        }
    }

    uint64_t b_min   = N;
    uint64_t b_max   = 0;
    double   sym_val = static_cast<double>(0);
    bool     sym     = true;
    for(uint64_t k = 0; k < K; ++k)
    {
        for(uint64_t l = ptr_[k] - base; l < ptr_[k + 1] - base; ++l)
        {
            J        ind = ind_[l] - base;
            T        v   = val_[l];
            uint64_t s   = (ind > k) ? (ind - k) : (k - ind);
            if(k != ind)
            {
                b_max     = std::max(s, b_max);
                b_min     = std::min(s, b_min);
                auto* lst = trseqs[ind];
                for(; lst != nullptr; lst = lst->next)
                {
                    if(lst->index == k)
                    {
                        const double d = abs(lst->val[0] - v);
                        sym_val        = std::max(sym_val, d);
                        break;
                    }
                }
                if(lst == nullptr)
                {
                    sym = false;
                    break;
                }
            }
        }
        if(sym == false)
        {
            break;
        }
    }
    std::cout << "symbolic symmetry " << (sym ? "true" : "false") << std::endl;
    std::cout << "numerical symmetry " << (sym_val > 0.0 ? "false" : "true") << std::endl;
    std::cout << "b_min " << b_min << std::endl;
    std::cout << "b_max " << b_max << std::endl;
    for(uint64_t i = 0; i < len_trseq; ++i)
    {
        auto* p = trseqs[i];
        do
        {
            auto* q = p;
            p       = p->next;
            delete q;
        } while(p != nullptr);
    }
    delete[] trseqs;
    return rocsparseio_status_success;
}

template <typename I, typename J, typename T>
rocsparseio_status rocsparseio_csx_statistics(rocsparseio_direction dir,
                                              uint64_t              M,
                                              uint64_t              N,
                                              uint64_t              nnz,
                                              const void* __restrict__ ptr_,
                                              const void* __restrict__ ind_,
                                              const void* __restrict__ val_,
                                              rocsparseio_index_base base)
{
    const I* __restrict__ ptr = (const I* __restrict__)ptr_;
    const J* __restrict__ ind = (const J* __restrict__)ind_;
    const T* __restrict__ val = (const T* __restrict__)val_;

    return rocsparseio_csx_statistics(dir, M, N, nnz, ptr, ind, val, base);
}

template <typename I, typename J, typename... P>
rocsparseio_status rocsparseio_csx_statistics_val_type(rocsparseio_type val_type, P&&... params)
{
    switch(val_type)
    {
    case rocsparseio_type_int32:
    {
        return rocsparseio_status_invalid_value;
    }
    case rocsparseio_type_int64:
    {
        return rocsparseio_status_invalid_value;
    }
    case rocsparseio_type_int8:
    {
        return rocsparseio_csx_statistics<I, J, int8_t>(params...);
    }
    case rocsparseio_type_float16:
    {
        return rocsparseio_csx_statistics<I, J, _Float16>(params...);
    }
    case rocsparseio_type_float32:
    {
        return rocsparseio_csx_statistics<I, J, float>(params...);
    }
    case rocsparseio_type_float64:
    {
        return rocsparseio_csx_statistics<I, J, double>(params...);
    }
    case rocsparseio_type_complex32:
    {
        return rocsparseio_csx_statistics<I, J, rocsparseio_float_complex>(params...);
    }
    case rocsparseio_type_complex64:
    {
        return rocsparseio_csx_statistics<I, J, rocsparseio_double_complex>(params...);
    }
    case rocsparseio_type_bfloat16:
    {
        return rocsparseio_status_invalid_value;
    }
    }
    return rocsparseio_status_invalid_enum;
}

template <typename I, typename... P>
rocsparseio_status rocsparseio_csx_statistics_ind_type(rocsparseio_type ind_type,
                                                       rocsparseio_type val_type,
                                                       P&&... params)
{
    switch(ind_type)
    {
    case rocsparseio_type_int32:
    {
        return rocsparseio_csx_statistics_val_type<I, int32_t>(val_type, params...);
    }
    case rocsparseio_type_int64:
    {
        return rocsparseio_csx_statistics_val_type<I, int64_t>(val_type, params...);
    }
    case rocsparseio_type_int8:
    case rocsparseio_type_float16:
    case rocsparseio_type_bfloat16:
    case rocsparseio_type_float32:
    case rocsparseio_type_float64:
    case rocsparseio_type_complex32:
    case rocsparseio_type_complex64:
    {
        return rocsparseio_status_invalid_value;
    }
    }
    return rocsparseio_status_invalid_enum;
}

template <typename... P>
rocsparseio_status rocsparseio_csx_statistics_dynamic_dispatch(rocsparseio_type ptr_type,
                                                               rocsparseio_type ind_type,
                                                               rocsparseio_type val_type,
                                                               P&&... params)
{
    switch(ptr_type)
    {
    case rocsparseio_type_int32:
    {
        return rocsparseio_csx_statistics_ind_type<int32_t>(ind_type, val_type, params...);
    }
    case rocsparseio_type_int64:
    {
        return rocsparseio_csx_statistics_ind_type<int64_t>(ind_type, val_type, params...);
    }
    case rocsparseio_type_int8:
    case rocsparseio_type_float16:
    case rocsparseio_type_bfloat16:
    case rocsparseio_type_float32:
    case rocsparseio_type_float64:
    case rocsparseio_type_complex32:
    case rocsparseio_type_complex64:
    {
        return rocsparseio_status_invalid_value;
    }
    }
    return rocsparseio_status_invalid_enum;
}

void usage(const char* appname_)
{
    fprintf(stderr, "NAME\n");
    fprintf(stderr, "       %s -- Get ROCSPARSEIO file information\n", appname_);
    fprintf(stderr, "SYNOPSIS\n");
    fprintf(stderr, "       %s [OPTION]... -o <output file> \n", appname_);
    fprintf(stderr, "DESCRIPTION\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "       List meta-data information.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "OPTIONS\n");
    fprintf(stderr, "       -v, --verbose\n");
    fprintf(stderr, "              use verbose of information.\n");
    fprintf(stderr, "       -h, --help\n");
    fprintf(stderr, "              produces this help and exit.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "NOTES\n");
    fprintf(stderr, "\n");
}

#define ROCSPARSEIO_CHECK(stus)                                             \
    do                                                                      \
    {                                                                       \
        rocsparseio_status check_ = (stus);                                 \
        if(check_ != rocsparseio_status_success)                            \
        {                                                                   \
            std::cerr << "invalid status: '" << check_ << "'" << std::endl; \
            return stus;                                                    \
        }                                                                   \
    } while(false)

int main(int argc, char** argv)
{
    rocsparseio_cmdline_t cmd(argc, argv);

    if(cmd.option("-h") || cmd.option("--help"))
    {
        usage(argv[0]);
        return ROCSPARSEIO_STATUS_SUCCESS;
    }

    rocsparseio_handle handle;
    rocsparseio_status status;
    rocsparseio_rwmode mode = rocsparseio_rwmode_read;
    status                  = rocsparseio_open(&handle, mode, argv[1]);

    rocsparseio_string name;
    rocsparseio_format format;
    status = rocsparseio_read_name(handle, name);
    std::cout << "name       : " << name << std::endl;

    status = rocsparseio_read_format(handle, &format);
    switch(format)
    {
    case rocsparseio_format_dense_vector:
    {
        rocsparseio_type data_type;
        uint64_t         data_nmemb;
        status = rocsparseiox_read_metadata_dense_vector(handle, &data_type, &data_nmemb);
        ROCSPARSEIO_CHECK(status);
        std::cout << "data_type  : " << data_type << std::endl;
        std::cout << "data_nmemb : " << data_nmemb << std::endl;
        break;
    }

    case rocsparseio_format_dense_matrix:
    {
        rocsparseio_order order;
        rocsparseio_type  data_type;
        uint64_t          m, n;
        status = rocsparseiox_read_metadata_dense_matrix(handle, &order, &m, &n, &data_type);
        ROCSPARSEIO_CHECK(status);
        std::cout << "data_type  : " << data_type << std::endl;
        std::cout << "order      : " << order << std::endl;
        std::cout << "m          : " << m << std::endl;
        std::cout << "n          : " << n << std::endl;
        break;
    }
    case rocsparseio_format_sparse_mcsx:
    {
        rocsparseio_type       ptr_type, ind_type, val_type;
        uint64_t               m, n, nnz;
        rocsparseio_direction  dir;
        rocsparseio_index_base base;

        status = rocsparseiox_read_metadata_sparse_mcsx(
            handle, &dir, &m, &n, &nnz, &ptr_type, &ind_type, &val_type, &base);
        ROCSPARSEIO_CHECK(status);
        std::cout << "m          : " << m << std::endl;
        std::cout << "n          : " << n << std::endl;
        std::cout << "nnz        : " << nnz << std::endl;
        std::cout << "ptr_type   : " << ptr_type << std::endl;
        std::cout << "ind_type   : " << ind_type << std::endl;
        std::cout << "val_type   : " << val_type << std::endl;
        std::cout << "base       : " << base << std::endl;

        uint64_t val_type_size, ind_type_size, ptr_type_size;

        status = rocsparseio_type_get_size(val_type, &val_type_size);
        ROCSPARSEIO_CHECK(status);

        status = rocsparseio_type_get_size(ptr_type, &ptr_type_size);
        ROCSPARSEIO_CHECK(status);

        status = rocsparseio_type_get_size(ind_type, &ind_type_size);
        ROCSPARSEIO_CHECK(status);

        uint64_t s = (val_type_size + ind_type_size) * nnz + (m + 1) * ptr_type_size;

        uint64_t nmb = s / (1024 * 1024);
        uint64_t ngb = nmb / 1024;

        nmb += nmb % 1024;
        std::cout << "memory     : " << ngb << " Go," << nmb << " Mb " << s % 1024 << " kb"
                  << std::endl;
        break;
    }
    case rocsparseio_format_sparse_csx:
    {
        rocsparseio_type       ptr_type, ind_type, val_type;
        uint64_t               m, n, nnz;
        rocsparseio_direction  dir;
        rocsparseio_index_base base;

        status = rocsparseiox_read_metadata_sparse_csx(
            handle, &dir, &m, &n, &nnz, &ptr_type, &ind_type, &val_type, &base);
        ROCSPARSEIO_CHECK(status);
        std::cout << "m          : " << m << std::endl;
        std::cout << "n          : " << n << std::endl;
        std::cout << "nnz        : " << nnz << std::endl;
        std::cout << "ptr_type   : " << ptr_type << std::endl;
        std::cout << "ind_type   : " << ind_type << std::endl;
        std::cout << "val_type   : " << val_type << std::endl;
        std::cout << "base       : " << base << std::endl;

        uint64_t val_type_size, ind_type_size, ptr_type_size;

        status = rocsparseio_type_get_size(val_type, &val_type_size);
        ROCSPARSEIO_CHECK(status);

        status = rocsparseio_type_get_size(ptr_type, &ptr_type_size);
        ROCSPARSEIO_CHECK(status);

        status = rocsparseio_type_get_size(ind_type, &ind_type_size);
        ROCSPARSEIO_CHECK(status);

        uint64_t s = (val_type_size + ind_type_size) * nnz + (m + 1) * ptr_type_size;

        uint64_t nmb = s / (1024 * 1024);
        uint64_t ngb = nmb / 1024;

        nmb += nmb % 1024;
        std::cout << "memory     : " << ngb << " Go," << nmb << " Mb " << s % 1024 << " kb"
                  << std::endl;

        uint64_t ptr_size = (dir == rocsparseio_direction_row) ? (m + 1) : (n + 1);
        void*    ptr      = malloc(ptr_type_size * ptr_size);

        uint64_t ind_size = nnz;
        void*    ind      = malloc(ind_type_size * ind_size);

        uint64_t val_size = nnz;
        void*    val      = malloc(val_type_size * val_size);

        status = rocsparseiox_read_sparse_csx(handle, ptr, ind, val);

        status = rocsparseio_csx_statistics_dynamic_dispatch(
            ptr_type, ind_type, val_type, dir, m, n, nnz, ptr, ind, val, base);

        break;
    }
    case rocsparseio_format_sparse_gebsx:
    case rocsparseio_format_sparse_coo:
    case rocsparseio_format_sparse_dia:
    case rocsparseio_format_sparse_ell:
    case rocsparseio_format_sparse_hyb:
    {
        break;
    }
    }

    status = rocsparseio_close(handle);
    ROCSPARSEIO_CHECK(status);
    return 0;
}
