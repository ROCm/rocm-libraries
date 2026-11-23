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
#include "test_rocrand_sobol_equidistribution.hpp"
#include <rocrand/rocrand.h>

#include <rng/sobol.hpp>

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
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

private:
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

// Algebraic check of the generator matrix over GF(2):
// Each dimension's generator matrix is recovered from generated values, then checked for
// the condition that makes the dimension a (0, 1)-sequence: every leading m x m submatrix
// is nonsingular over GF(2).
TYPED_TEST_P(sobol_qrng_tests, generator_matrix_test)
{
    using generator_t = typename TestFixture::generator_t;
    using constant_t  = typename TestFixture::constant_t;

    constexpr unsigned int dimensions = generator_t::is_scrambled ? SCRAMBLED_SOBOL_DIM : SOBOL_DIM;
    constexpr unsigned int bits       = sizeof(constant_t) * 8;

    std::vector<constant_t> direction_vectors;
    std::vector<constant_t> scramble_constants;
    ASSERT_NO_FATAL_FAILURE(
        (rocrand_test::sobol::recover_direction_vectors<generator_t, constant_t>(
            dimensions,
            direction_vectors,
            scramble_constants)));

    for(unsigned int d = 0; d < dimensions; d++)
    {
        const auto         matrix = rocrand_test::sobol::generator_matrix(direction_vectors.data()
                                                                  + static_cast<size_t>(d) * bits);
        const unsigned int m      = rocrand_test::sobol::first_singular_leading_minor(matrix);
        ASSERT_EQ(m, 0u) << "dimension " << d << " is not equidistributed: the leading " << m
                         << " x " << m << " submatrix of its generator matrix is singular, so its"
                         << " first 2^" << m << " values do not cover every " << m
                         << "-bit prefix exactly once";
    }
}

// The scrambled generators must apply a Matousek scramble to the Joe-Kuo direction vectors: a
// linear matrix scramble C -> L * C with L unit lower triangular, plus a per-dimension digital
// shift XORed onto the output. L being unit lower triangular is what keeps the leading minors
// nonsingular.
TYPED_TEST_P(sobol_qrng_tests, linear_matrix_scramble_test)
{
    using generator_t = typename TestFixture::generator_t;
    using constant_t  = typename TestFixture::constant_t;

    constexpr unsigned int dimensions = 1024;
    constexpr unsigned int bits       = sizeof(constant_t) * 8;

    std::vector<constant_t> direction_vectors;
    std::vector<constant_t> scramble_constants;
    ASSERT_NO_FATAL_FAILURE(
        (rocrand_test::sobol::recover_direction_vectors<generator_t, constant_t>(
            dimensions,
            direction_vectors,
            scramble_constants)));

    const constant_t* unscrambled;
    ASSERT_NO_FATAL_FAILURE(rocrand_test::sobol::get_unscrambled_direction_vectors(&unscrambled));

    for(unsigned int d = 0; d < dimensions; d++)
    {
        SCOPED_TRACE(testing::Message() << "with dimension = " << d);

        const auto scrambled = rocrand_test::sobol::generator_matrix(
            direction_vectors.data() + static_cast<size_t>(d) * bits);
        const auto plain
            = rocrand_test::sobol::generator_matrix(unscrambled + static_cast<size_t>(d) * bits);

        rocrand_test::sobol::gf2_matrix<constant_t> plain_inverse;
        ASSERT_TRUE(rocrand_test::sobol::invert(plain, plain_inverse))
            << "the unscrambled generator matrix is singular";

        const auto scramble = rocrand_test::sobol::multiply(scrambled, plain_inverse);
        ASSERT_TRUE(rocrand_test::sobol::is_unit_lower_triangular(scramble))
            << "the scrambled direction vectors are not a linear matrix scramble of the"
            << " unscrambled ones";

        if constexpr(generator_t::is_scrambled)
        {
            // The digital shift is what the scrambling constant contributes, and it is the only
            // reason an unscrambled sequence starts at zero while a scrambled one does not.
            EXPECT_NE(scramble_constants[d], constant_t{})
                << "no digital shift was applied to this dimension";
        }
        else
        {
            // An unscrambled generator starts at zero and must reproduce the Joe-Kuo vectors
            // exactly, which also confirms that the recovery above is truthful.
            EXPECT_EQ(scramble_constants[d], constant_t{});
            for(unsigned int j = 0; j < bits; j++)
            {
                ASSERT_EQ(direction_vectors[static_cast<size_t>(d) * bits + j],
                          unscrambled[static_cast<size_t>(d) * bits + j])
                    << "recovered direction vector " << j << " does not match the table";
            }
        }
    }
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
                            generator_matrix_test,
                            linear_matrix_scramble_test);

#endif // ROCRAND_TEST_INTERNAL_TEST_ROCRAND_SOBOL_QRNG_HPP_
