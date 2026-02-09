// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef FFT_HASH_H
#define FFT_HASH_H

#include "../../../library/include/rocfft/rocfft.h"
#include "../../../shared/arithmetic.h"
#include "../../../shared/hostbuf.h"
#include "../../../shared/increment.h"
#include "../../../shared/index_partition_omp.h"
#include "../../../shared/rocfft_complex.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>

struct hash_input
{
    hash_input(rocfft_precision    precision_,
               std::vector<size_t> length_,
               std::vector<size_t> stride_,
               size_t              dist_,
               rocfft_array_type   type_,
               size_t              nbatch_)
        : buf_precision(precision_)
        , buf_length(length_)
        , buf_stride(stride_)
        , buf_dist(dist_)
        , buf_type(type_)
        , nbatch(nbatch_)

    {
    }

    ~hash_input() {}

    rocfft_precision    buf_precision;
    std::vector<size_t> buf_length;
    std::vector<size_t> buf_stride;
    size_t              buf_dist;
    rocfft_array_type   buf_type;

    size_t nbatch;
};

template <typename Tint>
struct hash_output
{
    hash_output()
        : buffer_real(static_cast<Tint>(0))
        , buffer_imag(static_cast<Tint>(0))
    {
    }

    ~hash_output() {}

    bool operator==(const hash_output& rhs) const
    {
        return (buffer_real == rhs.buffer_real && buffer_imag == rhs.buffer_imag);
    }

    Tint buffer_real;
    Tint buffer_imag;
};

static inline double get_weight(const size_t counter, const size_t max_counter)
{
    return (static_cast<double>(counter) / static_cast<double>(max_counter));
}

template <typename Tint>
static inline void hash_value(Tint&              hash_value,
                              const size_t       counter,
                              const size_t       max_counter,
                              const rocfft_fp16& input_value)
{
    auto weight = get_weight(counter, max_counter);
    hash_value += std::hash<float>{}(weight * input_value);
}
template <typename Tint>
static inline void hash_value(Tint&        hash_value,
                              const size_t counter,
                              const size_t max_counter,
                              const float& input_value)
{
    auto weight = get_weight(counter, max_counter);
    hash_value += std::hash<float>{}(weight * input_value);
}
template <typename Tint>
static inline void hash_value(Tint&         hash_value,
                              const size_t  counter,
                              const size_t  max_counter,
                              const double& input_value)
{
    auto weight = get_weight(counter, max_counter);
    hash_value  = std::hash<double>{}(weight * input_value) + hash_value;
}

template <typename Tint>
static inline size_t get_max_counter(const Tint& whole_length, const size_t nbatch)
{
    return static_cast<size_t>(count_iters(whole_length) * nbatch);
}

template <typename T1>
static inline T1 get_unit_value(const T1& val)
{
    return static_cast<T1>(1);
}
template <typename T1>
static inline std::tuple<T1, T1> get_unit_value(const std::tuple<T1, T1>& val)
{
    return std::make_tuple(static_cast<T1>(1), static_cast<T1>(1));
}
template <typename T1>
static inline std::tuple<T1, T1, T1> get_unit_value(const std::tuple<T1, T1, T1>& val)
{
    return std::make_tuple(static_cast<T1>(1), static_cast<T1>(1), static_cast<T1>(1));
}

template <typename T1>
static inline void sum_hash_from_partitions(const std::vector<T1>& partition_hash, T1& hash_value)
{
    hash_value = 0;
    hash_value = std::accumulate(partition_hash.begin(), partition_hash.end(), hash_value);
}

template <typename Tfloat, typename Tint1, typename Tint2>
static inline void compute_real_buffer_hash(const std::vector<hostbuf>& ibuffer,
                                            const Tint1&                whole_length,
                                            const Tint1&                whole_stride,
                                            const size_t                idist,
                                            const size_t                nbatch,
                                            Tint2&                      hash_real,
                                            Tint2&                      hash_imag)
{
    auto unit_stride = get_unit_value(whole_stride);

    size_t max_counter = get_max_counter<Tint1>(whole_length, nbatch);

    auto   idata      = (Tfloat*)ibuffer[0].data();
    size_t i_base     = 0;
    auto   partitions = partition_rowmajor(whole_length);

    std::vector<Tint2> partition_hash_real(partitions.size(), 0);

    for(unsigned int b = 0; b < nbatch; b++, i_base += idist)
    {
#pragma omp parallel for num_threads(partitions.size())
        for(size_t part = 0; part < partitions.size(); ++part)
        {
            auto       index  = partitions[part].first;
            const auto length = partitions[part].second;
            do
            {
                const auto i       = compute_index(index, whole_stride, i_base);
                const auto counter = compute_index(index, unit_stride, i_base) + 1;

                hash_value<Tint2>(partition_hash_real[part], counter, max_counter, idata[i]);
            } while(increment_rowmajor(index, length));
        }
    }

    sum_hash_from_partitions(partition_hash_real, hash_real);
    hash_imag = 0;
}

template <typename Tfloat, typename Tint1, typename Tint2>
static inline void compute_planar_buffer_hash(const std::vector<hostbuf>& ibuffer,
                                              const Tint1&                whole_length,
                                              const Tint1&                whole_stride,
                                              const size_t                idist,
                                              const size_t                nbatch,
                                              Tint2&                      hash_real,
                                              Tint2&                      hash_imag)
{
    auto unit_stride = get_unit_value(whole_stride);

    size_t max_counter = get_max_counter<Tint1>(whole_length, nbatch);

    auto   ireal      = (Tfloat*)ibuffer[0].data();
    auto   iimag      = (Tfloat*)ibuffer[1].data();
    size_t i_base     = 0;
    auto   partitions = partition_rowmajor(whole_length);

    std::vector<Tint2> partition_hash_real(partitions.size(), 0);
    std::vector<Tint2> partition_hash_imag(partitions.size(), 0);

    for(unsigned int b = 0; b < nbatch; b++, i_base += idist)
    {
#pragma omp parallel for num_threads(partitions.size())
        for(size_t part = 0; part < partitions.size(); ++part)
        {
            auto       index  = partitions[part].first;
            const auto length = partitions[part].second;
            do
            {
                const auto i       = compute_index(index, whole_stride, i_base);
                const auto counter = compute_index(index, unit_stride, i_base) + 1;

                hash_value<Tint2>(partition_hash_real[part], counter, max_counter, ireal[i]);
                hash_value<Tint2>(partition_hash_imag[part], counter, max_counter, iimag[i]);
            } while(increment_rowmajor(index, length));
        }
    }

    sum_hash_from_partitions(partition_hash_real, hash_real);
    sum_hash_from_partitions(partition_hash_imag, hash_imag);
}

template <typename Tfloat, typename Tint1, typename Tint2>
static inline void compute_interleaved_buffer_hash(const std::vector<hostbuf>& ibuffer,
                                                   const Tint1&                whole_length,
                                                   const Tint1&                whole_stride,
                                                   const size_t                idist,
                                                   const size_t                nbatch,
                                                   Tint2&                      hash_real,
                                                   Tint2&                      hash_imag)
{
    auto unit_stride = get_unit_value(whole_stride);

    size_t max_counter = get_max_counter<Tint1>(whole_length, nbatch);

    auto   idata      = (rocfft_complex<Tfloat>*)ibuffer[0].data();
    size_t i_base     = 0;
    auto   partitions = partition_rowmajor(whole_length);

    std::vector<Tint2> partition_hash_real(partitions.size(), 0);
    std::vector<Tint2> partition_hash_imag(partitions.size(), 0);

    for(unsigned int b = 0; b < nbatch; b++, i_base += idist)
    {
#pragma omp parallel for num_threads(partitions.size())
        for(size_t part = 0; part < partitions.size(); ++part)
        {
            auto       index  = partitions[part].first;
            const auto length = partitions[part].second;
            do
            {
                const auto i       = compute_index(index, whole_stride, i_base);
                const auto counter = compute_index(index, unit_stride, i_base) + 1;

                hash_value<Tint2>(partition_hash_real[part], counter, max_counter, idata[i].real());
                hash_value<Tint2>(partition_hash_imag[part], counter, max_counter, idata[i].imag());
            } while(increment_rowmajor(index, length));
        }
    }

    sum_hash_from_partitions(partition_hash_real, hash_real);
    sum_hash_from_partitions(partition_hash_imag, hash_imag);
}

template <typename Tfloat, typename Tint1, typename Tint2>
static inline void compute_buffer_hash(const std::vector<hostbuf>& ibuffer,
                                       const rocfft_array_type     itype,
                                       const Tint1&                whole_length,
                                       const Tint1&                whole_stride,
                                       const size_t                idist,
                                       const size_t                nbatch,
                                       Tint2&                      hash_real,
                                       Tint2&                      hash_imag)
{
    switch(itype)
    {
    case rocfft_array_type_complex_interleaved:
    case rocfft_array_type_hermitian_interleaved:
        compute_interleaved_buffer_hash<Tfloat, Tint1, Tint2>(
            ibuffer, whole_length, whole_stride, idist, nbatch, hash_real, hash_imag);
        break;
    case rocfft_array_type_complex_planar:
    case rocfft_array_type_hermitian_planar:
        compute_planar_buffer_hash<Tfloat, Tint1, Tint2>(
            ibuffer, whole_length, whole_stride, idist, nbatch, hash_real, hash_imag);
        break;
    case rocfft_array_type_real:
        compute_real_buffer_hash<Tfloat, Tint1, Tint2>(
            ibuffer, whole_length, whole_stride, idist, nbatch, hash_real, hash_imag);
        break;
    default:
        throw std::runtime_error("Input layout format not yet supported");
    }
}

template <typename Tfloat, typename Tint>
static inline void compute_buffer_hash(const std::vector<hostbuf>& ibuffer,
                                       const rocfft_array_type     itype,
                                       const std::vector<size_t>&  ilength,
                                       const std::vector<size_t>&  istride,
                                       const size_t                idist,
                                       const size_t                nbatch,
                                       Tint&                       hash_real,
                                       Tint&                       hash_imag)
{
    switch(ilength.size())
    {
    case 1:
        compute_buffer_hash<Tfloat, size_t, Tint>(
            ibuffer, itype, ilength[0], istride[0], idist, nbatch, hash_real, hash_imag);
        break;
    case 2:
        compute_buffer_hash<Tfloat, std::tuple<size_t, size_t>, Tint>(
            ibuffer,
            itype,
            std::make_tuple(ilength[0], ilength[1]),
            std::make_tuple(istride[0], istride[1]),
            idist,
            nbatch,
            hash_real,
            hash_imag);
        break;
    case 3:
        compute_buffer_hash<Tfloat, std::tuple<size_t, size_t, size_t>, Tint>(
            ibuffer,
            itype,
            std::make_tuple(ilength[0], ilength[1], ilength[2]),
            std::make_tuple(istride[0], istride[1], istride[2]),
            idist,
            nbatch,
            hash_real,
            hash_imag);
        break;
    case 4:
    {
        // treat 4D brick as 3D + batch, but this needs to be
        // unbatched until we add 4D support for copy_buffers
        if(nbatch != 1)
            throw std::runtime_error("cannot copy batched 4D bricks");
        compute_buffer_hash<Tfloat, std::tuple<size_t, size_t, size_t>, Tint>(
            ibuffer,
            itype,
            std::make_tuple(ilength[1], ilength[2], ilength[3]),
            std::make_tuple(istride[1], istride[2], istride[3]),
            istride[0],
            ilength[0],
            hash_real,
            hash_imag);
        break;
    }
    default:
        abort();
    }
}

template <typename Tint>
static inline void compute_hash(const std::vector<hostbuf>& buffer,
                                const hash_input&           hash_in,
                                hash_output<Tint>&          hash_out)
{

    auto blength = hash_in.buf_length;
    auto bstride = hash_in.buf_stride;
    auto bdist   = hash_in.buf_dist;
    auto btype   = hash_in.buf_type;

    switch(hash_in.buf_precision)
    {
    case rocfft_precision_half:
        compute_buffer_hash<rocfft_fp16>(buffer,
                                         btype,
                                         blength,
                                         bstride,
                                         bdist,
                                         hash_in.nbatch,
                                         hash_out.buffer_real,
                                         hash_out.buffer_imag);
        break;
    case rocfft_precision_double:
        compute_buffer_hash<double>(buffer,
                                    btype,
                                    blength,
                                    bstride,
                                    bdist,
                                    hash_in.nbatch,
                                    hash_out.buffer_real,
                                    hash_out.buffer_imag);
        break;
    case rocfft_precision_single:
        compute_buffer_hash<float>(buffer,
                                   btype,
                                   blength,
                                   bstride,
                                   bdist,
                                   hash_in.nbatch,
                                   hash_out.buffer_real,
                                   hash_out.buffer_imag);
        break;
    default:
        abort();
    }
}

#endif // FFT_HASH_H
