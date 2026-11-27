/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "rocblas_matrix.hpp"
#include "rocblas_vector.hpp"

#include <cstring>
#include <functional>
#include <gtest/gtest.h>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
    // Thread counts to compare: one thread, a few, and everything the host offers
    std::vector<int> fill_thread_counts()
    {
        std::vector<int> counts{1, 2, 3, 7};
#ifdef _OPENMP
        counts.push_back(omp_get_max_threads());
#endif
        return counts;
    }

    template <typename T>
    bool same_bits(const std::vector<T>& a, const std::vector<T>& b)
    {
        return a.size() == b.size() && !memcmp(a.data(), b.data(), a.size() * sizeof(T));
    }

    // Runs fill() once per thread count and checks every run produced the same data
    template <typename T>
    void expect_independent_of_thread_count(const std::function<std::vector<T>()>& fill)
    {
#ifdef _OPENMP
        int saved_threads = omp_get_max_threads();
#endif
        std::vector<T> reference;
        for(int threads : fill_thread_counts())
        {
#ifdef _OPENMP
            omp_set_num_threads(threads);
#endif
            rocblas_seedrand();
            std::vector<T> data = fill();
            if(reference.empty())
                reference = data;
            else
                EXPECT_TRUE(same_bits(data, reference))
                    << "fill differs with " << threads << " threads";
        }
#ifdef _OPENMP
        omp_set_num_threads(saved_threads);
#endif
    }

    template <typename T>
    std::vector<T>
        fill_matrix(rocblas_check_matrix_type type, char uplo, size_t m, size_t n, size_t lda)
    {
        host_matrix<T> hA(m, n, lda);
        rocblas_init_matrix(type, uplo, random_generator<T>, hA);
        // a second fill continues the random sequence, so it must be reproducible too
        host_matrix<T> hB(m, n, lda);
        rocblas_init_matrix(type, uplo, random_generator<T>, hB);
        std::vector<T> out(hA.begin(), hA.end());
        out.insert(out.end(), hB.begin(), hB.end());
        return out;
    }

    TEST(rocblas_init, quick_matrix_fill_is_independent_of_thread_count)
    {
        expect_independent_of_thread_count<float>(
            [] { return fill_matrix<float>(rocblas_client_general_matrix, 'L', 97, 131, 100); });
        expect_independent_of_thread_count<float>(
            [] { return fill_matrix<float>(rocblas_client_triangular_matrix, 'U', 67, 67, 70); });
        expect_independent_of_thread_count<double>([] {
            return fill_matrix<double>(
                rocblas_client_diagonally_dominant_triangular_matrix, 'L', 67, 67, 67);
        });
        expect_independent_of_thread_count<float>(
            [] { return fill_matrix<float>(rocblas_client_symmetric_matrix, 'L', 67, 67, 67); });
        expect_independent_of_thread_count<rocblas_float_complex>([] {
            return fill_matrix<rocblas_float_complex>(
                rocblas_client_hermitian_matrix, 'U', 53, 53, 60);
        });
        expect_independent_of_thread_count<double>([] {
            host_matrix<double> hA(83, 71, 90);
            rocblas_init_matrix_alternating_sign(
                rocblas_client_general_matrix, 'L', random_hpl_generator<double>, hA);
            return std::vector<double>(hA.begin(), hA.end());
        });
        // strided-batched layout: the batch index is part of every block seed
        expect_independent_of_thread_count<float>([] {
            constexpr size_t   m = 37, n = 29, lda = 40, batches = 3;
            host_vector<float> hA(lda * n * batches);
            rocblas_init_matrix(rocblas_client_general_matrix,
                                'L',
                                random_generator<float>,
                                hA,
                                m,
                                n,
                                lda,
                                lda * n,
                                batches);
            return std::vector<float>(hA.begin(), hA.end());
        });
    }

    TEST(rocblas_init, quick_column_fill_is_independent_of_thread_count)
    {
        // rocblas_init fills whole columns at once (and short columns element by element)
        for(size_t m : {3, 61})
            expect_independent_of_thread_count<double>([m] {
                constexpr size_t    n = 47, lda = 64, batches = 2;
                host_vector<double> hA(lda * n * batches);
                rocblas_init(hA.data(), m, n, lda, lda * n, batches);
                return std::vector<double>(hA.begin(), hA.end());
            });
    }

    TEST(rocblas_init, quick_vector_fill_is_independent_of_thread_count)
    {
        // more than one fill block, and a partial last block
        static constexpr int64_t N = 3 * rocblas_fill_block_size + 123;
        for(int64_t incx : {1, -2})
        {
            expect_independent_of_thread_count<float>([incx] {
                host_vector<float> hx(N, incx);
                rocblas_init_vector(random_generator<float>, hx.data(), N, incx);
                return std::vector<float>(hx.begin(), hx.end());
            });
            expect_independent_of_thread_count<double>([incx] {
                host_vector<double> hx(N, incx);
                rocblas_init_vector_alternating_sign(random_generator<double>, hx.data(), N, incx);
                return std::vector<double>(hx.begin(), hx.end());
            });
        }
    }

    TEST(rocblas_init, quick_fill_sequence_semantics)
    {
        static constexpr int64_t N = 2 * rocblas_fill_block_size;
        host_vector<float>       h1(N), h2(N), h3(N);

        rocblas_seedrand();
        rocblas_init_vector(random_generator<float>, h1.data(), N, 1);
        rocblas_init_vector(random_generator<float>, h2.data(), N, 1);
        rocblas_seedrand();
        rocblas_init_vector(random_generator<float>, h3.data(), N, 1);

        EXPECT_TRUE(h1 == h3) << "resetting the sequence must reproduce the first fill";
        EXPECT_FALSE(h1 == h2) << "consecutive fills must not repeat the same data";

        // the caller's random sequence continues from the same point whatever the thread count
        expect_independent_of_thread_count<float>([] {
            host_vector<float> hx(N);
            rocblas_init_vector(random_generator<float>, hx.data(), N, 1);
            return std::vector<float>{random_generator<float>(), random_generator<float>()};
        });

        // every value of the integer generator is in [1, 10], and all ten occur
        bool seen[11] = {};
        for(int i = 0; i < 10000; i++)
        {
            float v = random_generator<float>();
            ASSERT_TRUE(v >= 1 && v <= 10 && v == int(v)) << v;
            seen[int(v)] = true;
        }
        for(int v = 1; v <= 10; v++)
            EXPECT_TRUE(seen[v]) << v << " never generated";
    }

    // The thread-count tests only show that a fill is deterministic: they would also pass if every
    // row, column, block or batch were seeded alike and so held the same values
    TEST(rocblas_init, quick_fill_blocks_differ)
    {
        rocblas_seedrand();

        static constexpr int64_t N = 2 * rocblas_fill_block_size;
        host_vector<float>       hx(N);
        rocblas_init_vector(random_generator<float>, hx.data(), N, 1);
        EXPECT_NE(
            memcmp(hx.data(), hx.data() + rocblas_fill_block_size, rocblas_fill_block_size * 4), 0)
            << "vector blocks 0 and 1 hold the same values";

        constexpr size_t   m = 64, n = 64, lda = 64, batches = 2;
        host_vector<float> hA(lda * n * batches);
        rocblas_init_matrix(rocblas_client_general_matrix,
                            'L',
                            random_generator<float>,
                            hA,
                            m,
                            n,
                            lda,
                            lda * n,
                            batches);
        const float* A         = hA.data();
        bool         rows_same = true;
        for(size_t j = 0; j < n; j++)
            rows_same = rows_same && A[j * lda] == A[1 + j * lda];
        EXPECT_FALSE(rows_same) << "rows 0 and 1 hold the same values";
        EXPECT_NE(memcmp(A, A + lda * n, lda * n * sizeof(float)), 0)
            << "batches 0 and 1 hold the same values";

        host_vector<float> hB(lda * n);
        rocblas_init(hB.data(), m, n, lda);
        EXPECT_NE(memcmp(hB.data(), hB.data() + lda, m * sizeof(float)), 0)
            << "columns 0 and 1 hold the same values";
    }

    // Every batch of a host_batch_matrix, then the next draws of the caller's random sequence,
    // which a batched helper must leave as it found it
    template <typename T>
    std::vector<T> batches_then_caller_draws(const host_batch_matrix<T>& hA)
    {
        std::vector<T> out;
        for(int64_t b = 0; b < hA.batch_count(); b++)
            out.insert(out.end(), hA[b], hA[b] + hA.lda() * hA.n());
        for(int i = 0; i < 4; i++)
            out.push_back(random_generator<T>());
        return out;
    }

    // The batched regular_to_banded and make_unit_diagonal reseed each batch: they fill unused
    // band storage and the unit diagonal with random values from the threads of a parallel loop
    TEST(rocblas_init, quick_banded_and_unit_diagonal_are_independent_of_thread_count)
    {
        constexpr rocblas_int n = 41, k = 5, batches = 5;
        for(bool upper : {false, true})
            expect_independent_of_thread_count<float>([upper] {
                host_batch_matrix<float> hA(n, n, n, batches);
                rocblas_init_matrix(
                    rocblas_client_general_matrix, 'L', random_generator<float>, hA);
                // ldab > k + 1, so the rows below the band are filled with random values too
                host_batch_matrix<float> hAB(k + 1, n, k + 4, batches);
                regular_to_banded(upper, hA, hAB, k);
                return batches_then_caller_draws(hAB);
            });
        for(rocblas_fill uplo : {rocblas_fill_lower, rocblas_fill_upper})
            expect_independent_of_thread_count<double>([uplo] {
                host_batch_matrix<double> hA(n, n, n + 3, batches);
                rocblas_init_matrix(rocblas_client_triangular_matrix,
                                    uplo == rocblas_fill_lower ? 'L' : 'U',
                                    random_generator<double>,
                                    hA);
                make_unit_diagonal(uplo, hA);
                return batches_then_caller_draws(hA);
            });
    }
}
