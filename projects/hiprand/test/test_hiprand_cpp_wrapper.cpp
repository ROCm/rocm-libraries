// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <stdio.h>
#include <gtest/gtest.h>

#include <hip/hip_runtime.h>
#include <hiprand/hiprand.hpp>

#include "test_common.hpp"

constexpr hiprandOrdering_t hiprand_ordering_types[] = {HIPRAND_ORDERING_PSEUDO_BEST,
                                                        HIPRAND_ORDERING_PSEUDO_DEFAULT,
                                                        HIPRAND_ORDERING_PSEUDO_SEEDED,
                                                        HIPRAND_ORDERING_PSEUDO_LEGACY,
                                                        HIPRAND_ORDERING_PSEUDO_DYNAMIC,
                                                        HIPRAND_ORDERING_QUASI_DEFAULT};

template<typename test_type>
struct hiprand_cpp_wrapper : public ::testing::Test
{
    typedef test_type engine_type;
};

template<typename test_type>
struct hiprand_cpp_wrapper_32 : public ::testing::Test
{
    typedef test_type engine_type;
};

template<typename test_type>
struct hiprand_cpp_wrapper_64 : public ::testing::Test
{
    typedef test_type engine_type;
};

template<typename test_type>
struct hiprand_cpp_wrapper_prng : public ::testing::Test
{
    typedef test_type engine_type;
};

template<typename test_type>
struct hiprand_cpp_wrapper_qrng : public ::testing::Test
{
    typedef test_type engine_type;
};

template<typename test_type>
struct hiprand_cpp_wrapper_offset : public ::testing::Test
{
    typedef test_type engine_type;
};

class hiprand_ordering : public ::testing::TestWithParam<hiprandOrdering_t>
{};

typedef ::testing::Types<hiprand_cpp::mrg32k3a,
                         hiprand_cpp::mt19937,
                         hiprand_cpp::mtgp32,
                         hiprand_cpp::philox4x32_10,
                         hiprand_cpp::scrambled_sobol32,
                         hiprand_cpp::scrambled_sobol64,
                         hiprand_cpp::sobol32,
                         hiprand_cpp::sobol64,
                         hiprand_cpp::xorwow>
    hiprand_cpp_test_types;

typedef ::testing::Types<hiprand_cpp::mrg32k3a,
                         hiprand_cpp::mt19937,
                         hiprand_cpp::mtgp32,
                         hiprand_cpp::philox4x32_10,
                         hiprand_cpp::scrambled_sobol32,
                         hiprand_cpp::sobol32,
                         hiprand_cpp::xorwow>
    hiprand_cpp_test_types_32;

typedef ::testing::Types<hiprand_cpp::scrambled_sobol64, hiprand_cpp::sobol64>
    hiprand_cpp_test_types_64;

typedef ::testing::Types<hiprand_cpp::mrg32k3a,
                         hiprand_cpp::mt19937,
                         hiprand_cpp::mtgp32,
                         hiprand_cpp::philox4x32_10,
                         hiprand_cpp::xorwow>
    hiprand_cpp_test_types_prng;

typedef ::testing::Types<hiprand_cpp::scrambled_sobol32,
                         hiprand_cpp::scrambled_sobol64,
                         hiprand_cpp::sobol32,
                         hiprand_cpp::sobol64>
    hiprand_cpp_test_types_qrng;

typedef ::testing::Types<hiprand_cpp::mrg32k3a,
                         hiprand_cpp::philox4x32_10,
                         hiprand_cpp::scrambled_sobol32,
                         hiprand_cpp::scrambled_sobol64,
                         hiprand_cpp::sobol32,
                         hiprand_cpp::sobol64,
                         hiprand_cpp::xorwow>
    hiprand_cpp_test_types_offset;

TYPED_TEST_SUITE(hiprand_cpp_wrapper, hiprand_cpp_test_types);
TYPED_TEST_SUITE(hiprand_cpp_wrapper_32, hiprand_cpp_test_types_32);
TYPED_TEST_SUITE(hiprand_cpp_wrapper_64, hiprand_cpp_test_types_64);
TYPED_TEST_SUITE(hiprand_cpp_wrapper_prng, hiprand_cpp_test_types_prng);
TYPED_TEST_SUITE(hiprand_cpp_wrapper_qrng, hiprand_cpp_test_types_qrng);
TYPED_TEST_SUITE(hiprand_cpp_wrapper_offset, hiprand_cpp_test_types_offset);

TEST(hiprand_cpp_wrapper, hiprand_error)
{
    hiprand_cpp::error e(HIPRAND_STATUS_SUCCESS);
    EXPECT_EQ(e.error_code(), HIPRAND_STATUS_SUCCESS);
}

template<class T>
void hiprand_rng_ctor_template()
{
    hiprandGenerator_t generator = NULL;
    ASSERT_EQ(hiprandCreateGenerator(&generator, T::type()), HIPRAND_STATUS_SUCCESS);
    ASSERT_NE(generator, (hiprandGenerator_t)NULL);
    T x(generator);
    ASSERT_EQ(generator, (hiprandGenerator_t)NULL);

    try {
        T y(generator);
        FAIL() << "Expected hiprand_cpp::error";
    }
    catch(const hiprand_cpp::error& err) {
        EXPECT_EQ(err.error_code(), HIPRAND_STATUS_NOT_INITIALIZED);
    }
    catch(...) {
        FAIL() << "Expected hiprand_cpp::error";
    }
}

TYPED_TEST(hiprand_cpp_wrapper, hiprand_rng_ctor)
{
    typedef typename TestFixture::engine_type engine_type;
    ASSERT_NO_THROW(hiprand_rng_ctor_template<engine_type>());
}

template<class T,
         typename std::enable_if<std::is_same<T, typename hiprand_cpp::mtgp32>::value
                                     || std::is_same<T, typename hiprand_cpp::mt19937>::value,
                                 bool>::type
         = true>
void hiprand_prng_ctor_template()
{
    T();
    T(11ULL); // seed
}

template<class T,
         typename std::enable_if<!std::is_same<T, typename hiprand_cpp::mtgp32>::value
                                     && !std::is_same<T, typename hiprand_cpp::mt19937>::value,
                                 bool>::type
         = true>
void hiprand_prng_ctor_template()
{
    T();
    T(11ULL); // seed
    T(11ULL, 2ULL); // seed, offset

    hiprand_cpp::random_device rd;
    T(rd(), 2ULL); // seed, offset
}

TYPED_TEST(hiprand_cpp_wrapper_prng, hiprand_ctor)
{
    typedef typename TestFixture::engine_type engine_type;
    ASSERT_NO_THROW(hiprand_prng_ctor_template<engine_type>());
}

template<class T1, class T2>
void assert_same_types()
{
    ::testing::StaticAssertTypeEq<T1, T2>();
}

TYPED_TEST(hiprand_cpp_wrapper_32, hiprand_rng_result_type)
{
    typedef typename TestFixture::engine_type engine_type;
    assert_same_types<unsigned int, typename engine_type::result_type>();
}

TYPED_TEST(hiprand_cpp_wrapper_64, hiprand_rng_result_type)
{
    typedef typename TestFixture::engine_type engine_type;
    assert_same_types<unsigned long long int, typename engine_type::result_type>();
}

TYPED_TEST(hiprand_cpp_wrapper_offset, hiprand_rng_offset_type)
{
    typedef typename TestFixture::engine_type engine_type;
    assert_same_types<unsigned long long int, typename engine_type::offset_type>();
}

TEST(hiprand_cpp_wrapper, hiprand_prng_default_seed)
{
    EXPECT_EQ(hiprand_cpp::philox4x32_10::default_seed, HIPRAND_PHILOX4x32_DEFAULT_SEED);
    EXPECT_EQ(hiprand_cpp::xorwow::default_seed, HIPRAND_XORWOW_DEFAULT_SEED);
    EXPECT_EQ(hiprand_cpp::mrg32k3a::default_seed, HIPRAND_MRG32K3A_DEFAULT_SEED);
    EXPECT_EQ(hiprand_cpp::mtgp32::default_seed, HIPRAND_MTGP32_DEFAULT_SEED);
    EXPECT_EQ(hiprand_cpp::mt19937::default_seed, HIPRAND_MT19937_DEFAULT_SEED);
}

TYPED_TEST(hiprand_cpp_wrapper_qrng, hiprand_qrng_default_num_dimensions)
{
    typedef typename TestFixture::engine_type engine_type;
    EXPECT_EQ(engine_type::default_num_dimensions, 1U);
}

template<class T>
void hiprand_qrng_ctor_template()
{
    T();
    T(11U); // dimensions
    T(11U, 2ULL); // dimensions, offset
    T(20000, 2ULL); // dimensions, offset

    try {
        T(20001, 2ULL);
        FAIL() << "Expected hiprand_cpp::error";
    }
    catch(const hiprand_cpp::error& err) {
        EXPECT_EQ(err.error_code(), HIPRAND_STATUS_OUT_OF_RANGE);
    }
    catch(...) {
        FAIL() << "Expected hiprand_cpp::error";
    }
}

TYPED_TEST(hiprand_cpp_wrapper_qrng, hiprand_ctor)
{
    typedef typename TestFixture::engine_type engine_type;
    ASSERT_NO_THROW(hiprand_qrng_ctor_template<engine_type>());
}

template<class T>
void hiprand_prng_seed_template()
{
    T engine;
    engine.seed(11ULL);
    engine.seed(12ULL);
}

TYPED_TEST(hiprand_cpp_wrapper_prng, hiprand_seed)
{
    typedef typename TestFixture::engine_type engine_type;
    ASSERT_NO_THROW(hiprand_prng_seed_template<engine_type>());
}

template<class T>
void hiprand_qrng_dims_template()
{
    T engine;
    engine.dimensions(11U);
    engine.dimensions(20000U);

    try {
        engine.dimensions(20001U);
        FAIL() << "Expected hiprand_cpp::error";
    }
    catch(const hiprand_cpp::error& err) {
        EXPECT_EQ(err.error_code(), HIPRAND_STATUS_OUT_OF_RANGE);
    }
    catch(...) {
        FAIL() << "Expected hiprand_cpp::error";
    }
}

TYPED_TEST(hiprand_cpp_wrapper_qrng, hiprand_dims)
{
    typedef typename TestFixture::engine_type engine_type;
    ASSERT_NO_THROW(hiprand_qrng_dims_template<engine_type>());
}

template<class T>
void hiprand_rng_offset_template()
{
    T engine;
    engine.offset(11ULL);
    engine.offset(12ULL);
}

TYPED_TEST(hiprand_cpp_wrapper_offset, hiprand_rng_offset)
{
    typedef typename TestFixture::engine_type engine_type;
    ASSERT_NO_THROW(hiprand_rng_offset_template<engine_type>());
}

template<class T>
void hiprand_rng_stream_template()
{
    T engine;
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    engine.stream(stream);
    engine.stream(NULL);
    HIP_CHECK(hipStreamDestroy(stream));
}

TYPED_TEST(hiprand_cpp_wrapper, hiprand_rng_stream)
{
    typedef typename TestFixture::engine_type engine_type;
    ASSERT_NO_THROW(hiprand_rng_stream_template<engine_type>());
}

template<class T, class IntType>
void hiprand_uniform_int_dist_template()
{
    T engine;
    hiprand_cpp::uniform_int_distribution<IntType> d;

    const size_t output_size = 8192;
    IntType * output;
    HIP_CHECK(
        hipMallocHelper((void **)&output,
        output_size * sizeof(IntType))
    );
    HIP_CHECK(hipDeviceSynchronize());

    // generate
    EXPECT_NO_THROW(d(engine, output, output_size));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<IntType> output_host(output_size);
    HIP_CHECK(hipMemcpy(output_host.data(),
                        output,
                        output_size * sizeof(IntType),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v) / static_cast<double>(std::numeric_limits<IntType>::max());
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);
}

TYPED_TEST(hiprand_cpp_wrapper_32, hiprand_uniform_int_dist_32)
{
    typedef typename TestFixture::engine_type engine_type;
    ASSERT_NO_THROW((hiprand_uniform_int_dist_template<engine_type, unsigned int>()));
}

TYPED_TEST(hiprand_cpp_wrapper_64, hiprand_uniform_int_dist_64)
{
    typedef typename TestFixture::engine_type engine_type;
    ASSERT_NO_THROW((hiprand_uniform_int_dist_template<engine_type, unsigned long long int>()));
}

template<class T, class RealType>
void hiprand_uniform_real_dist_template()
{
    T engine;
    hiprand_cpp::uniform_real_distribution<RealType> d;

    const size_t output_size = 8192;
    RealType * output;
    HIP_CHECK(
        hipMallocHelper((void **)&output,
        output_size * sizeof(RealType))
    );
    HIP_CHECK(hipDeviceSynchronize());

    // generate
    EXPECT_NO_THROW(d(engine, output, output_size));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<RealType> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(RealType),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);
}

TYPED_TEST(hiprand_cpp_wrapper, hiprand_uniform_real_dist_float)
{
    typedef typename TestFixture::engine_type engine_type;
    ASSERT_NO_THROW((hiprand_uniform_real_dist_template<engine_type, float>()));
}

TYPED_TEST(hiprand_cpp_wrapper, hiprand_uniform_real_dist_double)
{
    typedef typename TestFixture::engine_type engine_type;
    ASSERT_NO_THROW((hiprand_uniform_real_dist_template<engine_type, double>()));
}

template<class T, class RealType>
void hiprand_normal_dist_template()
{
    T engine;
    hiprand_cpp::normal_distribution<RealType> d;

    const size_t output_size = 8192;
    RealType * output;
    HIP_CHECK(
        hipMallocHelper((void **)&output,
        output_size * sizeof(RealType))
    );
    HIP_CHECK(hipDeviceSynchronize());

    // generate
    EXPECT_NO_THROW(d(engine, output, output_size));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<RealType> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(RealType),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.0, 0.2);

    double stddev = 0;
    for(auto v : output_host)
    {
        stddev += std::pow(static_cast<double>(v) - mean, 2);
    }
    stddev = stddev / output_size;
    EXPECT_NEAR(stddev, 1.0, 0.2);
}

TYPED_TEST(hiprand_cpp_wrapper, hiprand_normal_dist_float)
{
    typedef typename TestFixture::engine_type engine_type;
    ASSERT_NO_THROW((hiprand_normal_dist_template<engine_type, float>()));
}

TYPED_TEST(hiprand_cpp_wrapper, hiprand_normal_dist_double)
{
    typedef typename TestFixture::engine_type engine_type;
    ASSERT_NO_THROW((hiprand_normal_dist_template<engine_type, double>()));
}

TEST(hiprand_cpp_wrapper, hiprand_normal_dist_param)
{
    hiprand_cpp::normal_distribution<> d1(1.0f, 3.0f);
    hiprand_cpp::normal_distribution<> d2(1.0f, 3.0f);
    hiprand_cpp::normal_distribution<> d3(2.0f, 4.0f);

    ASSERT_TRUE(d1.param() == d2.param());
    ASSERT_TRUE(d1.param() == d1.param());
    ASSERT_TRUE(d1.param() != d3.param());

    d3.param(d1.param());
    ASSERT_TRUE(d1.param() == d3.param());
}

template<class T, class RealType>
void hiprand_lognormal_dist_template()
{
    T engine;
    hiprand_cpp::lognormal_distribution<RealType> d(1.6, 0.25);

    const size_t output_size = 8192;
    RealType * output;
    HIP_CHECK(
        hipMallocHelper((void **)&output,
        output_size * sizeof(RealType))
    );
    HIP_CHECK(hipDeviceSynchronize());

    // generate
    EXPECT_NO_THROW(d(engine, output, output_size));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<RealType> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(RealType),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;

    double stddev = 0;
    for(auto v : output_host)
    {
        stddev += std::pow(v - mean, 2);
    }
    stddev = std::sqrt(stddev / output_size);

    double logmean = std::log(mean * mean / std::sqrt(stddev + mean * mean));
    double logstd = std::sqrt(std::log(1.0f + stddev/(mean * mean)));

    EXPECT_NEAR(1.6, logmean, 1.6 * 0.2);
    EXPECT_NEAR(0.25, logstd, 0.25 * 0.2);
}

TYPED_TEST(hiprand_cpp_wrapper, hiprand_lognormal_dist_float)
{
    typedef typename TestFixture::engine_type engine_type;
    ASSERT_NO_THROW((hiprand_lognormal_dist_template<engine_type, float>()));
}

TYPED_TEST(hiprand_cpp_wrapper, hiprand_lognormal_dist_double)
{
    typedef typename TestFixture::engine_type engine_type;
    ASSERT_NO_THROW((hiprand_lognormal_dist_template<engine_type, double>()));
}

TEST(hiprand_cpp_wrapper, hiprand_lognormal_dist_param)
{
    hiprand_cpp::lognormal_distribution<> d1(1.0f, 3.0f);
    hiprand_cpp::lognormal_distribution<> d2(1.0f, 3.0f);
    hiprand_cpp::lognormal_distribution<> d3(2.0f, 4.0f);

    ASSERT_TRUE(d1.m() == d1.param().m());
    ASSERT_TRUE(d1.m() == 1.0f);
    ASSERT_TRUE(d1.s() == d1.param().s());
    ASSERT_TRUE(d1.s() == 3.0f);

    ASSERT_TRUE(d1.param() == d2.param());
    ASSERT_TRUE(d1.param() == d1.param());
    ASSERT_TRUE(d1.param() != d3.param());

    d3.param(d1.param());
    ASSERT_TRUE(d1.param() == d3.param());
}

template<class T>
void hiprand_poisson_dist_template(const double lambda)
{
    T engine;
    hiprand_cpp::poisson_distribution<unsigned int> d(lambda);

    const size_t output_size = 8192;
    unsigned int* output;
    HIP_CHECK(hipMallocHelper((void**)&output, output_size * sizeof(unsigned int)));
    HIP_CHECK(hipDeviceSynchronize());

    // generate
    EXPECT_NO_THROW(d(engine, output, output_size));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<unsigned int> output_host(output_size);
    HIP_CHECK(hipMemcpy(output_host.data(),
                        output,
                        output_size * sizeof(unsigned int),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;

    double variance = 0;
    for(auto v : output_host)
    {
        variance += std::pow(v - mean, 2);
    }
    variance = variance / output_size;

    EXPECT_NEAR(mean, lambda, std::max(1.0, lambda * 1e-1));
    EXPECT_NEAR(variance, lambda, std::max(1.0, lambda * 1e-1));
}

constexpr double lambdas[] = {1.0, 5.5, 20.0, 100.0, 1234.5, 5000.0};

TYPED_TEST(hiprand_cpp_wrapper, hiprand_poisson_dist)
{
    typedef typename TestFixture::engine_type engine_type;
    for(size_t i = 0; i < sizeof(lambdas) / sizeof(lambdas[0]); i++)
    {
        ASSERT_NO_THROW((hiprand_poisson_dist_template<engine_type>(lambdas[i])));
    }
}

TEST(hiprand_cpp_wrapper, hiprand_poisson_dist_param)
{
    hiprand_cpp::poisson_distribution<> d1(1.0);
    hiprand_cpp::poisson_distribution<> d2(1.0);
    hiprand_cpp::poisson_distribution<> d3(2.0);

    ASSERT_TRUE(d1.mean() == d1.param().mean());
    ASSERT_TRUE(d1.mean() == 1.0);

    ASSERT_TRUE(d1.param() == d2.param());
    ASSERT_TRUE(d1.param() == d1.param());
    ASSERT_TRUE(d1.param() != d3.param());

    d3.param(d1.param());
    ASSERT_TRUE(d1.param() == d3.param());
}

template<typename Engine>
void hiprand_ordering_test_template(hiprandOrdering_t ordering)
{
    Engine engine;
    engine.order(ordering);
}

TEST_P(hiprand_ordering, hiprand_ordering_test)
{
    const hiprandOrdering_t ordering = GetParam();
    if(ordering == HIPRAND_ORDERING_QUASI_DEFAULT)
    {
        ASSERT_NO_THROW(hiprand_ordering_test_template<hiprand_cpp::sobol32>(ordering));
    }
    else
    {
        ASSERT_NO_THROW(hiprand_ordering_test_template<hiprand_cpp::xorwow>(ordering));
    }
}

TEST_P(hiprand_ordering, hiprand_invalid_ordering_test)
{
    const hiprandOrdering_t ordering = GetParam();
    if(ordering == HIPRAND_ORDERING_QUASI_DEFAULT)
    {
        ASSERT_THROW(hiprand_ordering_test_template<hiprand_cpp::xorwow>(ordering),
                     hiprand_cpp::error);
    }
    else
    {
        ASSERT_THROW(hiprand_ordering_test_template<hiprand_cpp::sobol32>(ordering),
                     hiprand_cpp::error);
    }
}

INSTANTIATE_TEST_SUITE_P(hiprand, hiprand_ordering, ::testing::ValuesIn(hiprand_ordering_types));
