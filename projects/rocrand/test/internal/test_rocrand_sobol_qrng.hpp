// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCRAND_TEST_INTERNAL_TEST_ROCRAND_SOBOL_QRNG_HPP_
#define ROCRAND_TEST_INTERNAL_TEST_ROCRAND_SOBOL_QRNG_HPP_

#include "test_common.hpp"
#include "test_rocrand_common.hpp"
#include <rocrand/rocrand.h>

#include <rng/sobol.hpp>

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <type_traits>
#include <vector>

template<class Params>
struct sobol_qrng_tests : public ::testing::Test
{
    using generator_t                                 = typename Params::generator_t;
    static inline constexpr rocrand_ordering ordering = Params::ordering;
    using constant_t                                  = typename generator_t::constant_type;
    using engine_t                                    = typename generator_t::engine_type;

    auto get_generator() const
    {
        generator_t g;
        if(g.set_order(ordering) != ROCRAND_STATUS_SUCCESS)
        {
            throw std::runtime_error("Could not set ordering for generator");
        }
        return g;
    }

    rocrand_status get_engine(engine_t& engine, unsigned long long int offset)
    {
        const constant_t* direction_vectors;
        if(const rocrand_status status = get_direction_vectors(&direction_vectors);
           status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
        }
        if constexpr(generator_t::is_scrambled)
        {
            const constant_t* scramble_constants;
            if(const rocrand_status status = get_scramble_constants(&scramble_constants);
               status != ROCRAND_STATUS_SUCCESS)
            {
                return status;
            }
            engine = engine_t(direction_vectors, scramble_constants[1], offset);
        }
        else
        {
            engine = engine_t(direction_vectors, offset);
        }
        return ROCRAND_STATUS_SUCCESS;
    }

protected:
    rocrand_status get_scramble_constants(const constant_t** scramble_constants)
    {
        static_assert(generator_t::is_scrambled);
        if constexpr(std::is_same_v<unsigned int, constant_t>)
        {
            if(const rocrand_status status = rocrand_get_scramble_constants32(scramble_constants);
               status != ROCRAND_STATUS_SUCCESS)
            {
                return status;
            }
        }
        else
        {
            if(const rocrand_status status = rocrand_get_scramble_constants64(scramble_constants);
               status != ROCRAND_STATUS_SUCCESS)
            {
                return status;
            }
        }
        return ROCRAND_STATUS_SUCCESS;
    }

    rocrand_status get_direction_vectors(const constant_t** direction_vectors)
    {
        if constexpr(std::is_same_v<unsigned int, constant_t>)
        {
            constexpr rocrand_direction_vector_set direction_vector_set
                = generator_t::is_scrambled ? ROCRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6
                                            : ROCRAND_DIRECTION_VECTORS_32_JOEKUO6;
            if(const rocrand_status status
               = rocrand_get_direction_vectors32(direction_vectors, direction_vector_set);
               status != ROCRAND_STATUS_SUCCESS)
            {
                return status;
            }
        }
        else
        {
            constexpr rocrand_direction_vector_set direction_vector_set
                = generator_t::is_scrambled ? ROCRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6
                                            : ROCRAND_DIRECTION_VECTORS_64_JOEKUO6;
            if(const rocrand_status status
               = rocrand_get_direction_vectors64(direction_vectors, direction_vector_set);
               status != ROCRAND_STATUS_SUCCESS)
            {
                return status;
            }
        }
        return ROCRAND_STATUS_SUCCESS;
    }
};

TYPED_TEST_SUITE_P(sobol_qrng_tests);

template<class Generator, rocrand_ordering Ordering>
struct sobol_qrng_tests_params
{
    using generator_t                                 = Generator;
    static inline constexpr rocrand_ordering ordering = Ordering;
};

TYPED_TEST_P(sobol_qrng_tests, init_test)
{
    auto g = TestFixture::get_generator(); // offset = 0
    ROCRAND_CHECK(g.init());
    HIP_CHECK(hipDeviceSynchronize());

    g.set_offset(1);
    ROCRAND_CHECK(g.init());
    HIP_CHECK(hipDeviceSynchronize());

    g.set_offset(1337);
    ROCRAND_CHECK(g.init());
    HIP_CHECK(hipDeviceSynchronize());

    g.set_offset(1048576);
    ROCRAND_CHECK(g.init());
    HIP_CHECK(hipDeviceSynchronize());

    g.set_offset(1 << 24);
    ROCRAND_CHECK(g.init());
    HIP_CHECK(hipDeviceSynchronize());

    g.set_offset(1 << 28);
    ROCRAND_CHECK(g.init());
    HIP_CHECK(hipDeviceSynchronize());

    g.set_offset((1ULL << 36) + 1234567ULL);
    ROCRAND_CHECK(g.init());
    HIP_CHECK(hipDeviceSynchronize());
}

template<class Generator, class T>
void uniform_integer_test(T max)
{
    const size_t size = 1313;
    T*           data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(T) * size));

    Generator g;
    ROCRAND_CHECK(g.generate_uniform(data, size));

    T host_data[size];
    HIP_CHECK(hipMemcpy(host_data, data, sizeof(T) * size, hipMemcpyDeviceToHost));

    double sum = 0;
    for(size_t i = 0; i < size; i++)
    {
        sum += host_data[i];
    }
    const double mean = sum / size;
    ASSERT_NEAR(mean,
                static_cast<double>(max) / static_cast<double>(2),
                static_cast<double>(max) / static_cast<double>(20));

    HIP_CHECK(hipFree(data));
}

TYPED_TEST_P(sobol_qrng_tests, uniform_uint_test)
{
    using generator_t = typename TestFixture::generator_t;
    using constant_t  = typename TestFixture::constant_t;

    if constexpr(!std::is_same_v<constant_t, unsigned int>)
    {
        return;
    }

    uniform_integer_test<generator_t, constant_t>(UINT_MAX);
}

TYPED_TEST_P(sobol_qrng_tests, uniform_uint64_test)
{
    using generator_t = typename TestFixture::generator_t;
    using constant_t  = typename TestFixture::constant_t;

    if constexpr(!std::is_same_v<constant_t, unsigned long long int>)
    {
        return;
    }

    uniform_integer_test<generator_t, constant_t>(UINT64_MAX);
}

template<class Generator, class T>
void uniform_floating_point_test()
{
    const size_t size = 1313;
    T*           data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(*data) * size));

    Generator g;
    ROCRAND_CHECK(g.generate_uniform(data, size));

    T host_data[size];
    HIP_CHECK(hipMemcpy(host_data, data, sizeof(*host_data) * size, hipMemcpyDeviceToHost));

    double sum = 0;
    for(size_t i = 0; i < size; i++)
    {
        ASSERT_GT(host_data[i], static_cast<T>(0.0));
        ASSERT_LE(host_data[i], static_cast<T>(1.0));
        sum += host_data[i];
    }
    const double mean = sum / size;
    ASSERT_NEAR(mean, 0.5, 0.05);

    HIP_CHECK(hipFree(data));
}

TYPED_TEST_P(sobol_qrng_tests, uniform_float_test)
{
    using generator_t = typename TestFixture::generator_t;

    uniform_floating_point_test<generator_t, float>();
}

TYPED_TEST_P(sobol_qrng_tests, uniform_double_test)
{
    using generator_t = typename TestFixture::generator_t;

    uniform_floating_point_test<generator_t, double>();
}

template<class Generator, class T>
void normal_floating_point_test()
{
    const size_t size = 1313;
    T*           data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(*data) * size));

    Generator g;
    ROCRAND_CHECK(g.generate_normal(data, size, static_cast<T>(2.0), static_cast<T>(5.0)));
    HIP_CHECK(hipDeviceSynchronize());

    T host_data[size];
    HIP_CHECK(hipMemcpy(host_data, data, sizeof(*host_data) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    double mean = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        mean += host_data[i];
    }
    mean = mean / size;

    double stddev = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        stddev += std::pow(host_data[i] - mean, 2);
    }
    stddev = std::sqrt(stddev / size);

    EXPECT_NEAR(2.0, mean, 0.4); // 20%
    EXPECT_NEAR(5.0, stddev, 1.0); // 20%

    HIP_CHECK(hipFree(data));
}

TYPED_TEST_P(sobol_qrng_tests, normal_float_test)
{
    using generator_t = typename TestFixture::generator_t;

    normal_floating_point_test<generator_t, float>();
}

TYPED_TEST_P(sobol_qrng_tests, normal_double_test)
{
    using generator_t = typename TestFixture::generator_t;

    normal_floating_point_test<generator_t, double>();
}

template<class Generator, class T>
void log_normal_floating_point_test()
{
    const size_t size = 131313;
    T*           data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(*data) * size));

    T normal_mean   = static_cast<T>(3.0);
    T normal_stddev = static_cast<T>(1.5);

    Generator g;
    ROCRAND_CHECK(g.generate_log_normal(data, size, normal_mean, normal_stddev));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<T> host_data(size);
    HIP_CHECK(hipMemcpy(host_data.data(), data, sizeof(T) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    double mean = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        host_data[i] = std::log(host_data[i]);
        mean += host_data[i];
    }
    mean = mean / size;

    double stddev = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        stddev += std::pow(host_data[i] - mean, 2);
    }
    stddev = std::sqrt(stddev / size);

    EXPECT_NEAR(normal_mean, mean, normal_mean * 0.05); // 5%
    EXPECT_NEAR(normal_stddev, stddev, normal_stddev * 0.05); // 5%

    HIP_CHECK(hipFree(data));
}

TYPED_TEST_P(sobol_qrng_tests, log_normal_float_test)
{
    using generator_t = typename TestFixture::generator_t;

    log_normal_floating_point_test<generator_t, float>();
}

TYPED_TEST_P(sobol_qrng_tests, log_normal_double_test)
{
    using generator_t = typename TestFixture::generator_t;

    log_normal_floating_point_test<generator_t, double>();
}

TYPED_TEST_P(sobol_qrng_tests, poisson_test)
{
    const size_t  size = 1313;
    unsigned int* data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(unsigned int) * size));

    auto g = TestFixture::get_generator();
    ROCRAND_CHECK(g.generate_poisson(data, size, 5.5));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned int host_data[size];
    HIP_CHECK(hipMemcpy(host_data, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    double mean = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        mean += host_data[i];
    }
    mean = mean / size;

    double var = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        double x = host_data[i] - mean;
        var += x * x;
    }
    var = var / size;

    EXPECT_NEAR(mean, 5.5, std::max(1.0, 5.5 * 1e-2));
    EXPECT_NEAR(var, 5.5, std::max(1.0, 5.5 * 1e-2));

    HIP_CHECK(hipFree(data));
}

TYPED_TEST_P(sobol_qrng_tests, dimensions_test)
{
    const size_t size = 12345;
    float*       data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(float) * size));

    auto g = TestFixture::get_generator();

    ROCRAND_CHECK(g.generate(data, size));

    ROCRAND_CHECK(g.set_dimensions(4));
    EXPECT_EQ(g.generate(data, size), ROCRAND_STATUS_LENGTH_NOT_MULTIPLE);

    ROCRAND_CHECK(g.set_dimensions(15));
    ROCRAND_CHECK(g.generate(data, size));

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(data));
}

// Check if the numbers generated by first generate() call are different from
// the numbers generated by the 2nd call (same generator)
TYPED_TEST_P(sobol_qrng_tests, state_progress_test)
{
    using generator_t = typename TestFixture::generator_t;

    // Device data
    const size_t  size = 1025;
    unsigned int* data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(unsigned int) * size));

    // Generator
    generator_t g0;

    // Generate using g0 and copy to host
    ROCRAND_CHECK(g0.generate(data, size));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned int host_data1[size];
    HIP_CHECK(hipMemcpy(host_data1, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Generate using g0 and copy to host
    ROCRAND_CHECK(g0.generate(data, size));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned int host_data2[size];
    HIP_CHECK(hipMemcpy(host_data2, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    size_t same = 0;
    for(size_t i = 0; i < size; i++)
    {
        if(host_data1[i] == host_data2[i])
        {
            same++;
        }
    }
    // It may happen that numbers are the same, so we
    // just make sure that most of them are different.
    EXPECT_LT(same, static_cast<size_t>(0.01f * size));

    HIP_CHECK(hipFree(data));
}

TYPED_TEST_P(sobol_qrng_tests, discard_test)
{
    using generator_t = typename TestFixture::generator_t;
    using engine_t    = typename generator_t::engine_type;

    engine_t engine1;
    engine_t engine2;
    ROCRAND_CHECK(TestFixture::get_engine(engine1, 678));
    ROCRAND_CHECK(TestFixture::get_engine(engine2, 676));

    EXPECT_NE(engine1(), engine2());

    engine2.discard();

    EXPECT_NE(engine1(), engine2());

    engine2.discard();

    EXPECT_EQ(engine1(), engine2());
    EXPECT_EQ(engine1(), engine2());

    const unsigned int ds[] = {0, 1, 4, 37, 583, 7452, 21032, 35678, 66778, 10313475, 82120230};

    for(auto d : ds)
    {
        for(unsigned int i = 0; i < d; i++)
        {
            engine1.discard();
        }
        engine2.discard(d);

        EXPECT_EQ(engine1(), engine2());
    }
}

TYPED_TEST_P(sobol_qrng_tests, discard_stride_test)
{
    using generator_t = typename TestFixture::generator_t;
    using engine_t    = typename generator_t::engine_type;

    engine_t engine1;
    engine_t engine2;
    ROCRAND_CHECK(TestFixture::get_engine(engine1, 123));
    ROCRAND_CHECK(TestFixture::get_engine(engine2, 123));

    EXPECT_EQ(engine1(), engine2());

    const unsigned int ds[] = {1, 10, 12, 20, 4, 5, 30};

    for(auto d : ds)
    {
        engine1.discard(1 << d);
        engine2.discard_stride(1 << d);

        EXPECT_EQ(engine1(), engine2());
    }
}

TYPED_TEST_P(sobol_qrng_tests, offsets_test)
{
    using generator_t = typename TestFixture::generator_t;

    constexpr unsigned int           dimensions_list[] = {1, 2, 10, 321};
    constexpr unsigned long long int offsets[]         = {0, 1, 11, 112233};

    for(const unsigned int dimensions : dimensions_list)
    {
        SCOPED_TRACE(testing::Message() << "with dimensions = " << dimensions);
        for(const unsigned long long int offset : offsets)
        {
            SCOPED_TRACE(testing::Message() << "with offset = " << offset);
            const size_t  size  = 1313;
            const size_t  size0 = size * dimensions;
            const size_t  size1 = (size + offset) * dimensions;
            unsigned int* data0;
            unsigned int* data1;
            HIP_CHECK(hipMalloc(&data0, sizeof(unsigned int) * size0));
            HIP_CHECK(hipMalloc(&data1, sizeof(unsigned int) * size1));

            generator_t g0;
            g0.set_offset(offset);
            g0.set_dimensions(dimensions);
            g0.generate(data0, size0);

            generator_t g1;
            g1.set_dimensions(dimensions);
            g1.generate(data1, size1);

            std::vector<unsigned int> host_data0(size0);
            std::vector<unsigned int> host_data1(size1);
            HIP_CHECK(hipMemcpy(host_data0.data(),
                                data0,
                                sizeof(unsigned int) * size0,
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(host_data1.data(),
                                data1,
                                sizeof(unsigned int) * size1,
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipDeviceSynchronize());

            for(unsigned int d = 0; d < dimensions; d++)
            {
                for(size_t i = 0; i < size; i++)
                {
                    ASSERT_EQ(host_data0[d * size + i],
                              host_data1[d * (size + offset) + i + offset]);
                }
            }

            HIP_CHECK(hipFree(data0));
            HIP_CHECK(hipFree(data1));
        }
    }
}

// Check that subsequent generations of different sizes produce one Sobol
// sequence without gaps, no matter how many values are generated per call.
TYPED_TEST_P(sobol_qrng_tests, continuity_test)
{
    constexpr unsigned int continuity_test_dimensions[] = {1, 2, 10, 21};

    const std::vector<size_t> sizes0({100, 1, 24783, 3, 2, 776543});
    const std::vector<size_t> sizes1({1024, 56, 65536, 623456, 30, 111330});

    const size_t s0 = std::accumulate(sizes0.cbegin(), sizes0.cend(), std::size_t{0});
    const size_t s1 = std::accumulate(sizes1.cbegin(), sizes1.cend(), std::size_t{0});

    for(auto dimensions : continuity_test_dimensions)
    {
        SCOPED_TRACE(testing::Message() << "with dimensions = " << dimensions);

        const size_t size0 = s0 * dimensions;
        const size_t size1 = s1 * dimensions;

        unsigned int* data0;
        unsigned int* data1;
        HIP_CHECK(hipMalloc(&data0, sizeof(unsigned int) * size0));
        HIP_CHECK(hipMalloc(&data1, sizeof(unsigned int) * size1));

        typename TestFixture::generator_t g0;
        typename TestFixture::generator_t g1;
        g0.set_dimensions(dimensions);
        g1.set_dimensions(dimensions);

        std::vector<unsigned int> host_data0(size0);
        std::vector<unsigned int> host_data1(size1);

        // host_data0 contains all s0 values of dim0, then all s0 values of dim1...
        // host_data1 contains all s1 values of dim0, then all s1 values of dim1...
        size_t current0 = 0;
        for(size_t s : sizes0)
        {
            g0.generate(data0, s * dimensions);
            for(unsigned int d = 0; d < dimensions; d++)
            {
                HIP_CHECK(hipMemcpy(host_data0.data() + s0 * d + current0,
                                    data0 + d * s,
                                    sizeof(unsigned int) * s,
                                    hipMemcpyDefault));
            }
            current0 += s;
        }
        size_t current1 = 0;
        for(size_t s : sizes1)
        {
            g1.generate(data1, s * dimensions);
            for(unsigned int d = 0; d < dimensions; d++)
            {
                HIP_CHECK(hipMemcpy(host_data1.data() + s1 * d + current1,
                                    data1 + d * s,
                                    sizeof(unsigned int) * s,
                                    hipMemcpyDefault));
            }
            current1 += s;
        }

        for(unsigned int d = 0; d < dimensions; d++)
        {
            for(size_t i = 0; i < std::min(s0, s1); i++)
            {
                ASSERT_EQ(host_data0[d * s0 + i], host_data1[d * s1 + i]);
            }
        }

        HIP_CHECK(hipFree(data0));
        HIP_CHECK(hipFree(data1));
    }
}

// Check equidistribution across dyadic intervals (Joe & Kuo 2008):
// The first 2^m points of each dimension must fall into exactly one cell
// across 2^m equal subintervals.
TYPED_TEST_P(sobol_qrng_tests, equidistribution_test)
{
    using generator_t  = typename TestFixture::generator_t;
    using constant_t   = typename TestFixture::constant_t;
    constexpr size_t B = sizeof(constant_t) * 8;

    constexpr unsigned int NUM_DIMS = 4;
    constexpr unsigned int MAX_M    = 14;

    const size_t max_points = 1ULL << MAX_M;

    generator_t g;
    ROCRAND_CHECK(g.set_dimensions(NUM_DIMS));
    ROCRAND_CHECK(g.init());

    constant_t* dev_data;
    HIP_CHECK(hipMallocHelper(&dev_data, NUM_DIMS * max_points * sizeof(constant_t)));
    std::vector<constant_t> host_data(NUM_DIMS * max_points);

    ROCRAND_CHECK(g.generate(dev_data, NUM_DIMS * max_points));
    HIP_CHECK(hipMemcpy(host_data.data(),
                        dev_data,
                        NUM_DIMS * max_points * sizeof(constant_t),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(dev_data));

    for(unsigned int d = 0; d < NUM_DIMS; d++)
    {
        const constant_t* dim_data = host_data.data() + d * max_points;

        for(unsigned int m = 1; m <= MAX_M; m++)
        {
            const size_t num_points = 1ULL << m;

            std::vector<uint32_t> cell_counts(num_points, 0);
            for(size_t i = 0; i < num_points; i++)
            {
                const size_t cell = static_cast<size_t>(dim_data[i] >> (B - m));
                ASSERT_LT(cell, num_points);
                cell_counts[cell]++;
            }

            // Strict equidistribution: every cell must contain exactly 1 point
            for(size_t cell = 0; cell < num_points; cell++)
            {
                ASSERT_EQ(cell_counts[cell], 1u)
                    << "Equidistribution failed for dim=" << d << " m=" << m << " at cell " << cell;
            }
        }

        if constexpr(generator_t::is_scrambled)
        {
            // Scrambling must shift the origin: point 0 must not be all-zeros
            EXPECT_NE(dim_data[0], static_cast<constant_t>(0))
                << "Scrambled Sobol dim=" << d << " should not have point 0 at index 0";
        }
    }
}

// Diagnostic report over NUM_DIMS dimensions on CPU engines.
//
// For each dimension d, an engine is constructed directly from the raw
// direction-vector table (and scramble constants if applicable) and stepped
// sequentially on the CPU.  The test scans
// m = 1 … MAX_M and records the first m at which equidistribution breaks.
//
// Failure metric: fill ratio = distinct_cells / 2^m  (must equal 1.00 for a
// correct (t,s)-sequence).
TYPED_TEST_P(sobol_qrng_tests, equidistribution_report_test)
{
    using generator_t            = typename TestFixture::generator_t;
    using engine_t               = typename TestFixture::engine_t;
    using constant_t             = typename TestFixture::constant_t;
    constexpr size_t B           = sizeof(constant_t) * 8;
    constexpr size_t VECTOR_SIZE = B; // direction vectors stored per dimension

    // Covers the m=13/16/17/18/22
    // clusters from Joe & Kuo.
    constexpr unsigned int NUM_DIMS = 256;
    constexpr unsigned int MAX_M    = 22;
    const size_t           max_pts  = size_t(1) << MAX_M;

    const constant_t* direction_vectors  = nullptr;
    const constant_t* scramble_constants = nullptr;
    ROCRAND_CHECK(this->get_direction_vectors(&direction_vectors));
    if constexpr(generator_t::is_scrambled)
    {
        ROCRAND_CHECK(this->get_scramble_constants(&scramble_constants));
    }

    struct DimResult
    {
        unsigned int first_fail_m   = 0;
        size_t       distinct_cells = 0;
        size_t       total_cells    = 0;
    };
    std::vector<DimResult> results(NUM_DIMS);

    std::vector<constant_t> values(max_pts);
    std::vector<uint32_t>   hist(max_pts);

    for(unsigned int d = 0; d < NUM_DIMS; d++)
    {
        const constant_t* dim_vecs = direction_vectors + d * VECTOR_SIZE;

        engine_t engine;
        if constexpr(generator_t::is_scrambled)
            engine = engine_t(dim_vecs, scramble_constants[d], /*offset=*/0);
        else
            engine = engine_t(dim_vecs, /*offset=*/0);

        for(size_t i = 0; i < max_pts; i++)
            values[i] = engine.next();

        unsigned int first_fail = 0;
        size_t       fail_dist  = 0;
        size_t       fail_total = 0;

        for(unsigned int m = 1; m <= MAX_M && first_fail == 0; m++)
        {
            const size_t n = size_t(1) << m;
            std::fill(hist.begin(), hist.begin() + n, 0u);
            for(size_t i = 0; i < n; i++)
                hist[static_cast<size_t>(values[i] >> (B - m))]++;

            size_t distinct = 0;
            bool   broken   = false;
            for(size_t cell = 0; cell < n; cell++)
            {
                if(hist[cell] > 0)
                    ++distinct;
                if(hist[cell] != 1)
                    broken = true;
            }
            if(broken)
            {
                first_fail = m;
                fail_dist  = distinct;
                fail_total = n;
            }
        }
        results[d] = {first_fail, fail_dist, fail_total};
    }

    size_t                         broken = 0;
    std::map<unsigned int, size_t> fail_by_m;
    int                            first_broken_dim = -1;

    for(unsigned int d = 0; d < NUM_DIMS; d++)
    {
        if(results[d].first_fail_m > 0)
        {
            if(first_broken_dim < 0)
                first_broken_dim = static_cast<int>(d);
            ++broken;
            fail_by_m[results[d].first_fail_m]++;
        }
    }

    const char* label = generator_t::is_scrambled ? "SCRAMBLED" : "NON-SCRAMBLED";
    std::cout << "\nequidistribution -- " << label << " table\n";
    std::cout << "  dimensions that break at some m: " << broken << "  (" << std::fixed
              << std::setprecision(2) << (100.0 * broken / NUM_DIMS) << "%)" << "   [first "
              << NUM_DIMS << " dims, tested m <= " << MAX_M << "]\n";

    for(auto& [m, cnt] : fail_by_m)
    {
        std::cout << "    first fails at m=" << m << " (2^" << m << " =" << std::setw(10)
                  << (size_t(1) << m) << " points): " << cnt << " dims\n";
    }

    if(first_broken_dim >= 0)
    {
        const DimResult& r    = results[first_broken_dim];
        const double     fill = static_cast<double>(r.distinct_cells) / r.total_cells;
        std::cout << "  brute-force confirm dim " << first_broken_dim << " at m=" << r.first_fail_m
                  << ": " << r.distinct_cells << " distinct cells of " << r.total_cells
                  << "  fill ratio " << std::fixed << std::setprecision(3) << fill << " (FAIL)\n";
    }
    std::cout << std::flush;

    EXPECT_EQ(broken, size_t(0)) << broken << " of " << NUM_DIMS
                                 << " dimensions fail equidistribution (see stdout report above)";
}

REGISTER_TYPED_TEST_SUITE_P(sobol_qrng_tests,
                            init_test,
                            uniform_uint_test,
                            uniform_uint64_test,
                            uniform_float_test,
                            uniform_double_test,
                            normal_float_test,
                            normal_double_test,
                            log_normal_float_test,
                            log_normal_double_test,
                            poisson_test,
                            dimensions_test,
                            state_progress_test,
                            discard_test,
                            discard_stride_test,
                            offsets_test,
                            continuity_test,
                            equidistribution_test,
                            equidistribution_report_test);

#endif // ROCRAND_TEST_INTERNAL_TEST_ROCRAND_SOBOL_QRNG_HPP_
