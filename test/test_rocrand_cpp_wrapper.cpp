// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "test_common.hpp"
#include "test_utils/rocrand_cpp_wrapper_traits.hpp"
#include "test_utils/test_matrix.hpp"

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>
#include <rocrand/rocrand.hpp>

#include <cstddef>
#include <cstdio>
#include <numeric>
#include <type_traits>

TEST(rocrand_cpp_wrapper, rocrand_error)
{
    rocrand_cpp::error e(ROCRAND_STATUS_SUCCESS);
    EXPECT_EQ(e.error_code(), ROCRAND_STATUS_SUCCESS);
}

TEST(rocrand_cpp_wrapper, rocrand_prng_default_seed)
{
    EXPECT_EQ(rocrand_cpp::lfsr113::default_seed.x, ROCRAND_LFSR113_DEFAULT_SEED_X);
    EXPECT_EQ(rocrand_cpp::lfsr113::default_seed.y, ROCRAND_LFSR113_DEFAULT_SEED_Y);
    EXPECT_EQ(rocrand_cpp::lfsr113::default_seed.z, ROCRAND_LFSR113_DEFAULT_SEED_Z);
    EXPECT_EQ(rocrand_cpp::lfsr113::default_seed.w, ROCRAND_LFSR113_DEFAULT_SEED_W);
    EXPECT_EQ(rocrand_cpp::mrg31k3p::default_seed, ROCRAND_MRG31K3P_DEFAULT_SEED);
    EXPECT_EQ(rocrand_cpp::mrg32k3a::default_seed, ROCRAND_MRG32K3A_DEFAULT_SEED);
    EXPECT_EQ(rocrand_cpp::mt19937::default_seed, 0);
    EXPECT_EQ(rocrand_cpp::mtgp32::default_seed, 0);
    EXPECT_EQ(rocrand_cpp::philox4x32_10::default_seed, ROCRAND_PHILOX4x32_DEFAULT_SEED);
    // sobol generators don't support seeding
    EXPECT_EQ(rocrand_cpp::threefry2x32::default_seed, 0);
    EXPECT_EQ(rocrand_cpp::threefry2x64::default_seed, 0);
    EXPECT_EQ(rocrand_cpp::threefry4x32::default_seed, 0);
    EXPECT_EQ(rocrand_cpp::threefry4x64::default_seed, 0);
    EXPECT_EQ(rocrand_cpp::xorwow::default_seed, ROCRAND_XORWOW_DEFAULT_SEED);
}

using Generators = testing::Types<rocrand_cpp::lfsr113,
                                  rocrand_cpp::mrg31k3p,
                                  rocrand_cpp::mrg32k3a,
                                  rocrand_cpp::mt19937,
                                  rocrand_cpp::mtgp32,
                                  rocrand_cpp::philox4x32_10,
                                  rocrand_cpp::threefry2x32,
                                  rocrand_cpp::threefry2x64,
                                  rocrand_cpp::threefry4x32,
                                  rocrand_cpp::threefry4x64,
                                  rocrand_cpp::scrambled_sobol32,
                                  rocrand_cpp::scrambled_sobol64,
                                  rocrand_cpp::sobol32,
                                  rocrand_cpp::sobol64,
                                  rocrand_cpp::xorwow>;

using rocrand_cpp_distributions = test_utils::test_matrix<
    // Generators
    std::tuple<rocrand_cpp::lfsr113,
               rocrand_cpp::mrg31k3p,
               rocrand_cpp::mrg32k3a,
               rocrand_cpp::mt19937,
               rocrand_cpp::mtgp32,
               rocrand_cpp::philox4x32_10,
               rocrand_cpp::threefry2x32,
               rocrand_cpp::threefry2x64,
               rocrand_cpp::threefry4x32,
               rocrand_cpp::threefry4x64,
               rocrand_cpp::scrambled_sobol32,
               rocrand_cpp::scrambled_sobol64,
               rocrand_cpp::sobol32,
               rocrand_cpp::sobol64,
               rocrand_cpp::xorwow>,
    // Distributions
    std::tuple<rocrand_cpp::uniform_int_distribution<unsigned char>,
               rocrand_cpp::uniform_int_distribution<unsigned short>,
               rocrand_cpp::uniform_int_distribution<unsigned int>,
               rocrand_cpp::uniform_int_distribution<unsigned long long int>,
               rocrand_cpp::uniform_real_distribution<half>,
               rocrand_cpp::uniform_real_distribution<float>,
               rocrand_cpp::uniform_real_distribution<double>,
               rocrand_cpp::normal_distribution<half>,
               rocrand_cpp::normal_distribution<float>,
               rocrand_cpp::normal_distribution<double>,
               rocrand_cpp::lognormal_distribution<half>,
               rocrand_cpp::lognormal_distribution<float>,
               rocrand_cpp::lognormal_distribution<double>,
               rocrand_cpp::poisson_distribution<unsigned int>>>::test_types;

template<typename test_tuple>
struct rocrand_cpp_wrapper_distributions : public ::testing::Test
{
    using generator_t    = typename std::tuple_element<0, test_tuple>::type;
    using distribution_t = typename std::tuple_element<1, test_tuple>::type;
};

template<typename GeneratorType>
struct rocrand_cpp_wrapper : public ::testing::Test
{
    using generator_t = GeneratorType;
};

TYPED_TEST_SUITE(rocrand_cpp_wrapper, Generators);

TYPED_TEST(rocrand_cpp_wrapper, rocrand_rng_ctor)
{
    using generator_t = typename TestFixture::generator_t;

    rocrand_generator generator1 = nullptr;
    ASSERT_EQ(rocrand_create_generator(&generator1, generator_t::type()), ROCRAND_STATUS_SUCCESS);

    // generator1 should be set to null after having been moved to generator2
    generator_t generator2(generator1);
    ASSERT_EQ(generator1, nullptr);

    try {
        generator_t generator3(generator1);
        FAIL() << "Move-constructing generator from an already moved generator. Expected "
                  "constructor to throw rocrand_cpp::error";
    }
    catch(const rocrand_cpp::error& err) {
        EXPECT_EQ(err.error_code(), ROCRAND_STATUS_NOT_CREATED);
    }
    catch(...) {
        FAIL() << "Expected rocrand_cpp::error";
    }
}

// testing constructor of PRNGs that support offset
template<class generator_t>
auto test_ctor() -> typename std::enable_if<!is_qrng<generator_t>::value
                                            && supports_offset<generator_t>::value>::type
{
    generator_t g1;
    generator_t g2(11ULL); // seed
    generator_t g3(11ULL, ROCRAND_ORDERING_PSEUDO_DEFAULT); // seed, ordering
    generator_t g4(11ULL, 2ULL); // seed, offset
    generator_t g5(11ULL, 2ULL, ROCRAND_ORDERING_PSEUDO_DEFAULT); // seed, offset, ordering

    rocrand_cpp::random_device rd;
    generator_t                g6(rd()); // seed from random device
    generator_t                g7(rd(), 2ULL); // seed from random device, offset
    generator_t                g8(rd(),
                   2ULL,
                   ROCRAND_ORDERING_PSEUDO_DEFAULT); // seed from random device, offset, ordering
}

// testing constructor of PRNGs that don't support offset
template<class generator_t>
auto test_ctor() -> typename std::enable_if<!is_qrng<generator_t>::value
                                            && !supports_offset<generator_t>::value>::type
{
    using seed_type = typename generator_t::seed_type;
    generator_t g1;
    generator_t g2(11ULL);
    generator_t g3(11ULL, ROCRAND_ORDERING_PSEUDO_DEFAULT);

    // some generators have a different seed type than unsigned long long and
    // just support constructing with unsigned long long for compatibility
    generator_t g4(seed_type{});
    generator_t g5(seed_type{}, ROCRAND_ORDERING_PSEUDO_DEFAULT);

    // seed from random device
    rocrand_cpp::random_device rd;
    generator_t                g6(rd(), ROCRAND_ORDERING_PSEUDO_DEFAULT);
}

// testing constructor of QRNGs
template<class generator_t>
auto test_ctor() -> typename std::enable_if<is_qrng<generator_t>::value>::type
{
    generator_t g1;
    generator_t g2(11U); // dimensions
    generator_t g3(11U, ROCRAND_ORDERING_QUASI_DEFAULT); // dimensions, ordering
    generator_t g4(11U, 2ULL); // dimensions, offset
    generator_t g5(2000, 2ULL, ROCRAND_ORDERING_QUASI_DEFAULT); // dimensions, offset, ordering

    try
    {
        generator_t g6(20001, 2ULL);
        FAIL() << "Expected rocrand_cpp::error for QRNG dimension being out of range";
    }
    catch(const rocrand_cpp::error& err)
    {
        EXPECT_EQ(err.error_code(), ROCRAND_STATUS_OUT_OF_RANGE);
    }
    catch(...)
    {
        FAIL() << "Expected rocrand_cpp::error for QRNG dimension being out of range";
    }
}

TYPED_TEST(rocrand_cpp_wrapper, rocrand_prng_ctor)
{
    using generator_t = typename TestFixture::generator_t;
    ASSERT_NO_THROW(test_ctor<generator_t>());
}

template<class T1, class T2>
void assert_same_types()
{
    ::testing::StaticAssertTypeEq<T1, T2>();
}

template<class generator_t>
auto test_result_type() -> typename std::enable_if<is_64bit<generator_t>::value>::type
{
    assert_same_types<typename generator_t::result_type, unsigned long long int>();
}

template<class generator_t>
auto test_result_type() -> typename std::enable_if<!is_64bit<generator_t>::value>::type
{
    assert_same_types<typename generator_t::result_type, unsigned int>();
}

TYPED_TEST(rocrand_cpp_wrapper, rocrand_rng_result_type)
{
    using generator_t = typename TestFixture::generator_t;
    test_result_type<generator_t>();
}

template<class generator_t>
auto test_offset() -> typename std::enable_if<supports_offset<generator_t>::value>::type
{
    assert_same_types<unsigned long long, typename generator_t::offset_type>();
}

template<class generator_t>
auto test_offset() -> typename std::enable_if<!supports_offset<generator_t>::value>::type
{
    GTEST_SKIP();
}

TYPED_TEST(rocrand_cpp_wrapper, rocrand_rng_offset_type)
{
    using generator_t = typename TestFixture::generator_t;
    test_offset<generator_t>();
}

template<class generator_t>
auto qrng_default_dimensions() -> typename std::enable_if<!is_qrng<generator_t>::value>::type
{
    GTEST_SKIP();
}

template<class generator_t>
auto qrng_default_dimensions() -> typename std::enable_if<is_qrng<generator_t>::value>::type
{
    EXPECT_EQ(generator_t::default_num_dimensions, 1U);
}

TYPED_TEST(rocrand_cpp_wrapper, rocrand_qrng_default_num_dimensions)
{
    using generator_t = typename TestFixture::generator_t;
    qrng_default_dimensions<generator_t>();
}

template<class generator_t>
auto rocrand_prng_seed_template() -> typename std::enable_if<is_qrng<generator_t>::value>::type
{
    GTEST_SKIP();
}

template<class generator_t>
auto rocrand_prng_seed_template() -> typename std::enable_if<!is_qrng<generator_t>::value>::type
{
    generator_t engine;
    engine.seed(11ULL);
    engine.seed(12ULL);
}

TYPED_TEST(rocrand_cpp_wrapper, rocrand_prng_seed)
{
    using generator_t = typename TestFixture::generator_t;
    ASSERT_NO_THROW(rocrand_prng_seed_template<generator_t>());
}

template<class generator_t>
auto rocrand_qrng_dims_template() -> typename std::enable_if<!is_qrng<generator_t>::value>::type
{
    GTEST_SKIP();
}

template<class generator_t>
auto rocrand_qrng_dims_template() -> typename std::enable_if<is_qrng<generator_t>::value>::type
{
    generator_t engine;
    engine.dimensions(11U);
    engine.dimensions(20000U);

    try {
        engine.dimensions(20001U);
        FAIL() << "Expected rocrand_cpp::error for QRNG dimension being out of range";
    }
    catch(const rocrand_cpp::error& err) {
        EXPECT_EQ(err.error_code(), ROCRAND_STATUS_OUT_OF_RANGE);
    }
    catch(...) {
        FAIL() << "Expected rocrand_cpp::error for QRNG dimension being out of range";
    }
}

TYPED_TEST(rocrand_cpp_wrapper, rocrand_qrng_dims)
{
    using generator_t = typename TestFixture::generator_t;
    ASSERT_NO_THROW(rocrand_qrng_dims_template<generator_t>());
}

template<class generator_t>
auto rocrand_rng_offset_template() ->
    typename std::enable_if<supports_offset<generator_t>::value>::type
{
    generator_t engine;
    engine.offset(11ULL);
    engine.offset(12ULL);
}

template<class generator_t>
auto rocrand_rng_offset_template() ->
    typename std::enable_if<!supports_offset<generator_t>::value>::type
{
    GTEST_SKIP();
}

TYPED_TEST(rocrand_cpp_wrapper, rocrand_rng_offset)
{
    using generator_t = typename TestFixture::generator_t;
    ASSERT_NO_THROW(rocrand_rng_offset_template<generator_t>());
}

TYPED_TEST(rocrand_cpp_wrapper, rocrand_rng_stream)
{
    using generator_t = typename TestFixture::generator_t;

    generator_t engine;
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    ASSERT_NO_THROW(engine.stream(stream));
    ASSERT_NO_THROW(engine.stream(0));
    HIP_CHECK(hipStreamDestroy(stream));
}

namespace test_utils
{

template<typename T>
double mean(std::vector<T>& results)
{
    // half type needs to be explicitly cast to double in std::accumulate
    double mean = std::accumulate(results.begin(),
                                  results.end(),
                                  0.0,
                                  [](double partial_sum, T val)
                                  { return partial_sum + static_cast<double>(val); });
    return mean / results.size();
}

template<typename T>
double variance(double mean, std::vector<T>& results)
{
    // half type needs to be explicitly cast to double  in std::accumulate
    double variance = std::accumulate(results.begin(),
                                      results.end(),
                                      0.0,
                                      [mean](double partial_sum, T val)
                                      {
                                          double deviation = static_cast<double>(val) - mean;
                                          return (partial_sum + deviation * deviation);
                                      });
    return variance / results.size();
}

} // namespace test_utils

template<typename distribution_t>
struct distribution_test;

template<typename T>
struct distribution_test<rocrand_cpp::uniform_int_distribution<T>>
{
    static void verify(std::vector<T>& results, rocrand_cpp::uniform_int_distribution<T>& /*dist*/)
    {
        double mean = test_utils::mean(results);
        mean /= static_cast<double>(rocrand_cpp::uniform_int_distribution<T>::max());
        EXPECT_NEAR(mean, 0.5, 0.01);
    }
};

template<typename T>
struct distribution_test<rocrand_cpp::uniform_real_distribution<T>>
{
    static void verify(std::vector<T>& results, rocrand_cpp::uniform_real_distribution<T>& /*dist*/)
    {
        double mean = test_utils::mean(results);
        EXPECT_NEAR(mean, 0.5, 0.01);
    }
};

template<typename T>
struct distribution_test<rocrand_cpp::normal_distribution<T>>
{
    static void verify(std::vector<T>& results, rocrand_cpp::normal_distribution<T>& dist)
    {
        double expected_mean   = dist.mean();
        double expected_stddev = dist.stddev();
        double mean            = test_utils::mean(results);
        double variance        = test_utils::variance(mean, results);
        EXPECT_NEAR(mean, expected_mean, 0.01);
        EXPECT_NEAR(variance, expected_stddev, 0.05);
    }
};

template<typename T>
struct distribution_test<rocrand_cpp::lognormal_distribution<T>>
{
    static void verify(std::vector<T>& results, rocrand_cpp::lognormal_distribution<T>& dist)
    {
        double mu              = dist.m();
        double sigma           = dist.s();
        double expected_mean   = std::exp(mu + sigma * sigma / 2.0);
        double expected_stddev = std::sqrt(std::exp(sigma * sigma) - 1.0) * expected_mean;
        double mean            = test_utils::mean(results);
        double stddev          = std::sqrt(test_utils::variance(mean, results));
        EXPECT_NEAR(mean, expected_mean, 0.01);
        EXPECT_NEAR(stddev, expected_stddev, 0.1);
    }
};

template<>
struct distribution_test<rocrand_cpp::poisson_distribution<unsigned int>>
{
    static void verify(std::vector<unsigned int>&                       results,
                       rocrand_cpp::poisson_distribution<unsigned int>& dist)
    {
        double lambda   = dist.mean();
        double mean     = test_utils::mean(results);
        double variance = test_utils::variance(mean, results);
        EXPECT_NEAR(mean, lambda, lambda * 0.1);
        EXPECT_NEAR(variance, lambda, lambda * 0.1);
    }
};

template<class generator_t, class distribution_t>
auto rocrand_dist_test() -> typename std::enable_if<
    is_64bit<generator_t>::value
    || (!is_64bit<generator_t>::value
        && !std::is_same<distribution_t,
                         rocrand_cpp::uniform_int_distribution<unsigned long long int>>::value)>::
    type
{
    generator_t    engine;
    distribution_t distribution;
    using result_t = typename distribution_t::result_type;

    constexpr size_t output_size       = 1e6;
    constexpr size_t output_size_bytes = output_size * sizeof(result_t);
    result_t*        d_output;
    HIP_CHECK(hipMallocHelper(&d_output, output_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    ASSERT_NO_THROW(distribution(engine, d_output, output_size));

    std::vector<result_t> h_output(output_size);
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, output_size_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(d_output));

    distribution_test<distribution_t>::verify(h_output, distribution);
}

template<class generator_t, class distribution_t>
auto rocrand_dist_test() -> typename std::enable_if<
    !is_64bit<generator_t>::value
    && std::is_same<distribution_t,
                    rocrand_cpp::uniform_int_distribution<unsigned long long int>>::value>::type
{
    // 64 bit generation is not supported for these generators
    generator_t    engine;
    distribution_t distribution;
    using result_t = typename distribution_t::result_type;

    try
    {
        result_t* output_dummy = nullptr;
        distribution(engine, output_dummy, 0);
        GTEST_FAIL() << "Expected rocrand_cpp::error of type ROCRAND_STATUS_TYPE_ERROR";
    }
    catch(const rocrand_cpp::error& error)
    {
        EXPECT_EQ(error.error_code(), ROCRAND_STATUS_TYPE_ERROR)
            << "Expected rocrand_cpp::error of type ROCRAND_STATUS_TYPE_ERROR";
    }
    catch(...)
    {
        GTEST_FAIL() << "Expected rocrand_cpp::error of type ROCRAND_STATUS_TYPE_ERROR";
    }
}

TYPED_TEST_SUITE(rocrand_cpp_wrapper_distributions, rocrand_cpp_distributions);

TYPED_TEST(rocrand_cpp_wrapper_distributions, rocrand_dist)
{
    using generator_t    = typename TestFixture::generator_t;
    using distribution_t = typename TestFixture::distribution_t;

    rocrand_dist_test<generator_t, distribution_t>();
}

TEST(rocrand_cpp_wrapper, rocrand_normal_dist_param)
{
    rocrand_cpp::normal_distribution<> d1(1.0f, 3.0f);
    rocrand_cpp::normal_distribution<> d2(1.0f, 3.0f);
    rocrand_cpp::normal_distribution<> d3(2.0f, 4.0f);

    ASSERT_TRUE(d1.param() == d2.param());
    ASSERT_TRUE(d1.param() == d1.param());
    ASSERT_TRUE(d1.param() != d3.param());

    d3.param(d1.param());
    ASSERT_TRUE(d1.param() == d3.param());
}

TEST(rocrand_cpp_wrapper, rocrand_lognormal_dist_param)
{
    rocrand_cpp::lognormal_distribution<> d1(1.0f, 3.0f);
    rocrand_cpp::lognormal_distribution<> d2(1.0f, 3.0f);
    rocrand_cpp::lognormal_distribution<> d3(2.0f, 4.0f);

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

TEST(rocrand_cpp_wrapper, rocrand_poisson_dist_param)
{
    rocrand_cpp::poisson_distribution<> d1(1.0);
    rocrand_cpp::poisson_distribution<> d2(1.0);
    rocrand_cpp::poisson_distribution<> d3(2.0);

    ASSERT_TRUE(d1.mean() == d1.param().mean());
    ASSERT_TRUE(d1.mean() == 1.0);

    ASSERT_TRUE(d1.param() == d2.param());
    ASSERT_TRUE(d1.param() == d1.param());
    ASSERT_TRUE(d1.param() != d3.param());

    d3.param(d1.param());
    ASSERT_TRUE(d1.param() == d3.param());
}
