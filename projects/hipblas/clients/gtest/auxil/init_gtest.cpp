/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ************************************************************************ */

#include "hipblas_init.hpp"
#include "host_matrix.hpp"
#include "host_vector.hpp"
#include "utility.h"

#include <complex>
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
            hipblas_seedrand();
            std::vector<T> data = fill();
            if(reference.empty())
                reference = data;
            else
                EXPECT_TRUE(data == reference) << "fill differs with " << threads << " threads";
        }
#ifdef _OPENMP
        omp_set_num_threads(saved_threads);
#endif
    }

    template <typename T>
    std::vector<T> fill_matrix(hipblas_matrix_type type, char uplo, size_t m, size_t n, size_t lda)
    {
        host_matrix<T> hA(m, n, lda);
        hipblas_init_matrix(type, uplo, random_generator<T>, hA);
        // a second fill continues the random sequence, so it must be reproducible too
        host_matrix<T> hB(m, n, lda);
        hipblas_init_matrix(type, uplo, random_generator<T>, hB);
        std::vector<T> out(hA.begin(), hA.end());
        out.insert(out.end(), hB.begin(), hB.end());
        return out;
    }

    TEST(hipblas_init, matrix_fill_is_independent_of_thread_count)
    {
        expect_independent_of_thread_count<float>(
            [] { return fill_matrix<float>(hipblas_general_matrix, 'L', 97, 131, 100); });
        expect_independent_of_thread_count<float>(
            [] { return fill_matrix<float>(hipblas_triangular_matrix, 'U', 67, 67, 70); });
        expect_independent_of_thread_count<float>(
            [] { return fill_matrix<float>(hipblas_symmetric_matrix, 'L', 67, 67, 67); });
        expect_independent_of_thread_count<std::complex<float>>([] {
            return fill_matrix<std::complex<float>>(hipblas_hermitian_matrix, 'U', 53, 53, 60);
        });
        expect_independent_of_thread_count<double>([] {
            host_matrix<double> hA(83, 71, 90);
            hipblas_init_matrix_alternating_sign(
                hipblas_general_matrix, 'L', random_hpl_generator<double>, hA);
            return std::vector<double>(hA.begin(), hA.end());
        });
    }

    TEST(hipblas_init, vector_fill_is_independent_of_thread_count)
    {
        // more than one fill block, and a partial last block
        constexpr int64_t N = 3 * hipblas_fill_block_size + 123;
        for(int64_t incx : {1, -2})
        {
            expect_independent_of_thread_count<float>([incx] {
                host_vector<float> hx(N, incx);
                hipblas_init_vector(random_generator<float>, hx.data(), N, incx);
                return std::vector<float>(hx.begin(), hx.end());
            });
            expect_independent_of_thread_count<double>([incx] {
                host_vector<double> hx(N, incx);
                hipblas_init_vector_alternating_sign(random_generator<double>, hx.data(), N, incx);
                return std::vector<double>(hx.begin(), hx.end());
            });
        }
    }

    TEST(hipblas_init, fill_sequence_semantics)
    {
        constexpr int64_t  N = 2 * hipblas_fill_block_size;
        host_vector<float> h1(N), h2(N), h3(N);

        hipblas_seedrand();
        hipblas_init_vector(random_generator<float>, h1.data(), N, 1);
        hipblas_init_vector(random_generator<float>, h2.data(), N, 1);
        hipblas_seedrand();
        hipblas_init_vector(random_generator<float>, h3.data(), N, 1);

        EXPECT_TRUE(h1 == h3) << "resetting the sequence must reproduce the first fill";
        EXPECT_FALSE(h1 == h2) << "consecutive fills must not repeat the same data";

        // the caller's random sequence continues from the same point whatever the thread count
        expect_independent_of_thread_count<float>([] {
            host_vector<float> hx(N);
            hipblas_init_vector(random_generator<float>, hx.data(), N, 1);
            return std::vector<float>{random_generator<float>(), random_generator<float>()};
        });
    }
}
