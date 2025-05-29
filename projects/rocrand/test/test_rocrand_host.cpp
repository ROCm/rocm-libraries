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

#include <rocrand/rocrand.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#define ROCRAND_CHECK(state) ASSERT_EQ(state, ROCRAND_STATUS_SUCCESS)

namespace
{

constexpr const unsigned long long seeds[]            = {0, 0xAAAAAAAAAAAULL};
constexpr const size_t             seeds_count        = sizeof(seeds) / sizeof(seeds[0]);
constexpr const size_t             random_seeds_count = 2;

std::vector<unsigned long long> get_seeds()
{
    std::vector<unsigned long long> ret(seeds_count + random_seeds_count);
    std::copy_n(seeds, seeds_count, ret.begin());
    std::default_random_engine rng(std::random_device{}());
    std::generate(ret.begin() + seeds_count, ret.end(), [&] { return rng(); });
    return ret;
}

struct host_test_params
{
    rocrand_rng_type rng_type;
    bool             blocking_host_generator;
    bool             use_default_stream;

    friend std::ostream& operator<<(std::ostream& os, const host_test_params& params)
    {
        os << "{ "
           << "rng_type: " << params.rng_type << ", blocking: " << params.blocking_host_generator
           << ", default_stream: " << params.use_default_stream << " }";
        return os;
    }
};

// Check all combinations of blocking_host_generator and use_default_stream for one PRNG (Philox)
// and one QRNG (Sobol32), others work the same way
constexpr host_test_params host_test_params_array[] = {
    {   ROCRAND_RNG_PSEUDO_PHILOX4_32_10, false,  true},
    {         ROCRAND_RNG_PSEUDO_LFSR113, false,  true},
    {        ROCRAND_RNG_PSEUDO_MRG31K3P, false,  true},
    {        ROCRAND_RNG_PSEUDO_MRG32K3A, false,  true},
    {         ROCRAND_RNG_PSEUDO_MT19937, false,  true},
    {          ROCRAND_RNG_PSEUDO_MTGP32, false,  true},
    { ROCRAND_RNG_PSEUDO_THREEFRY2_32_20, false,  true},
    { ROCRAND_RNG_PSEUDO_THREEFRY2_64_20, false,  true},
    { ROCRAND_RNG_PSEUDO_THREEFRY4_32_20, false,  true},
    { ROCRAND_RNG_PSEUDO_THREEFRY4_64_20, false,  true},
    {          ROCRAND_RNG_PSEUDO_XORWOW, false,  true},
    {ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32, false,  true},
    {ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64, false,  true},
    {          ROCRAND_RNG_QUASI_SOBOL32, false,  true},
    {          ROCRAND_RNG_QUASI_SOBOL64, false,  true},

    {   ROCRAND_RNG_PSEUDO_PHILOX4_32_10, false, false},
    {   ROCRAND_RNG_PSEUDO_PHILOX4_32_10,  true, false},
    {   ROCRAND_RNG_PSEUDO_PHILOX4_32_10,  true,  true},

    {          ROCRAND_RNG_QUASI_SOBOL32, false, false},
    {          ROCRAND_RNG_QUASI_SOBOL32,  true, false},
    {          ROCRAND_RNG_QUASI_SOBOL32,  true,  true},
};

} // namespace

class rocrand_generate_host_test : public ::testing::TestWithParam<host_test_params>
{
protected:
    void SetUp() override
    {
        if(GetParam().rng_type == ROCRAND_RNG_PSEUDO_MT19937)
        {
            ROCRAND_SKIP_SLOW_TEST_IF_NOT_ENABLED();
        }
        if(!GetParam().use_default_stream)
        {
            HIP_CHECK(hipStreamCreateWithFlags(&m_custom_stream, hipStreamNonBlocking));
        }
    }

    void TearDown() override
    {
        if(!GetParam().use_default_stream)
        {
            HIP_CHECK(hipStreamDestroy(m_custom_stream));
        }
    }

    rocrand_generator get_generator()
    {
        const auto        params = GetParam();
        rocrand_generator generator;
        if(params.blocking_host_generator)
        {
            EXPECT_EQ(ROCRAND_STATUS_SUCCESS,
                      rocrand_create_generator_host_blocking(&generator, params.rng_type));
        }
        else
        {
            EXPECT_EQ(ROCRAND_STATUS_SUCCESS,
                      rocrand_create_generator_host(&generator, params.rng_type));
        }
        if(!params.use_default_stream)
        {
            EXPECT_EQ(ROCRAND_STATUS_SUCCESS, rocrand_set_stream(generator, m_custom_stream));
        }
        return generator;
    }

private:
    hipStream_t m_custom_stream;
};

void test_int(rocrand_generator generator, const size_t test_size)
{
    std::vector<unsigned int> results(test_size);
    for(size_t i = 0; i < seeds_count + random_seeds_count; ++i)
    {
        const auto seed = i < seeds_count ? seeds[i] : rand();
        SCOPED_TRACE(testing::Message() << "with seed = " << seed);
        ROCRAND_CHECK(rocrand_set_seed(generator, seed));

        ROCRAND_CHECK(rocrand_generate(generator, results.data(), results.size()));
        // We need this because the generator is async and there is no memcpy
        // that implicitly synchronizes here
        HIP_CHECK(hipDeviceSynchronize());

        double mean = 0.0;
        for(unsigned int v : results)
        {
            mean += v;
        }

        mean /= static_cast<double>(std::numeric_limits<unsigned int>::max());
        mean /= results.size();

        ASSERT_NEAR(mean, 0.5, 0.05);
    }

    ROCRAND_CHECK(rocrand_destroy_generator(generator));
}

TEST_P(rocrand_generate_host_test, int_test)
{
    test_int(get_generator(), 11111);
}

TEST_P(rocrand_generate_host_test, int_test_large)
{
    ROCRAND_SKIP_SLOW_TEST_IF_NOT_ENABLED();
    constexpr size_t large_test_size = size_t(INT_MAX) + 1;
    test_int(get_generator(), large_test_size);
}

template<typename Type, typename F>
void test_int_parity(rocrand_generator                      host_generator,
                     rocrand_rng_type                       rng_type,
                     F                                      generate,
                     const std::vector<unsigned long long>& seeds = get_seeds())
{
    rocrand_generator device_generator;
    ROCRAND_CHECK(rocrand_create_generator(&device_generator, rng_type));

    std::vector<Type> host_results(218192);
    std::vector<Type> device_results(host_results.size());

    Type* output;
    HIP_CHECK(hipMallocHelper(&output, host_results.size() * sizeof(Type)));

    for(const unsigned long long seed : seeds)
    {
        SCOPED_TRACE(testing::Message() << "with seed = " << seed);
        ROCRAND_CHECK(rocrand_set_seed(host_generator, seed));
        ROCRAND_CHECK(rocrand_set_seed(device_generator, seed));

        ROCRAND_CHECK(generate(host_generator, host_results.data(), host_results.size()));
        ROCRAND_CHECK(generate(device_generator, output, host_results.size()));

        HIP_CHECK(hipMemcpy(device_results.data(),
                            output,
                            host_results.size() * sizeof(Type),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipDeviceSynchronize());

        assert_eq(host_results, device_results);
    }

    ROCRAND_CHECK(rocrand_destroy_generator(host_generator));
    ROCRAND_CHECK(rocrand_destroy_generator(device_generator));
    HIP_CHECK(hipFree(output));
}

TEST_P(rocrand_generate_host_test, char_parity_test)
{
    test_int_parity<unsigned char>(get_generator(), GetParam().rng_type, rocrand_generate_char);
}

TEST_P(rocrand_generate_host_test, short_parity_test)
{
    test_int_parity<unsigned short>(get_generator(), GetParam().rng_type, rocrand_generate_short);
}

TEST_P(rocrand_generate_host_test, int_parity_test)
{
    test_int_parity<unsigned int>(get_generator(), GetParam().rng_type, rocrand_generate);
}

template<typename Type, typename F>
void test_uniform_parity(rocrand_generator                      host_generator,
                         rocrand_rng_type                       rng_type,
                         F                                      generate,
                         const std::vector<unsigned long long>& seeds = get_seeds())
{
    rocrand_generator device_generator;
    ROCRAND_CHECK(rocrand_create_generator(&device_generator, rng_type));

    std::vector<Type> host_results(218192);
    std::vector<Type> device_results(host_results.size());

    Type* output;
    HIP_CHECK(hipMallocHelper(&output, host_results.size() * sizeof(Type)));

    for(const unsigned long long seed : seeds)
    {
        SCOPED_TRACE(testing::Message() << "with seed = " << seed);
        ROCRAND_CHECK(rocrand_set_seed(host_generator, seed));
        ROCRAND_CHECK(rocrand_set_seed(device_generator, seed));

        ROCRAND_CHECK(generate(host_generator, host_results.data(), host_results.size()));
        ROCRAND_CHECK(generate(device_generator, output, host_results.size()));

        HIP_CHECK(hipMemcpy(device_results.data(),
                            output,
                            host_results.size() * sizeof(Type),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipDeviceSynchronize());

        assert_eq(host_results, device_results);
    }

    ROCRAND_CHECK(rocrand_destroy_generator(host_generator));
    ROCRAND_CHECK(rocrand_destroy_generator(device_generator));
    HIP_CHECK(hipFree(output));
}

TEST_P(rocrand_generate_host_test, uniform_half_parity_test)
{
    test_uniform_parity<half>(get_generator(), GetParam().rng_type, rocrand_generate_uniform_half);
}

TEST_P(rocrand_generate_host_test, uniform_float_parity_test)
{
    test_uniform_parity<float>(get_generator(), GetParam().rng_type, rocrand_generate_uniform);
}

TEST_P(rocrand_generate_host_test, uniform_double_parity_test)
{
    test_uniform_parity<double>(get_generator(),
                                GetParam().rng_type,
                                rocrand_generate_uniform_double);
}

template<typename Type, typename F>
void test_normal_parity(rocrand_generator                      host_generator,
                        rocrand_rng_type                       rng_type,
                        F                                      generate,
                        double                                 eps,
                        const std::vector<unsigned long long>& seeds = get_seeds())
{
    if(rng_type == ROCRAND_RNG_PSEUDO_MT19937)
    {
        ROCRAND_SKIP_SLOW_TEST_IF_NOT_ENABLED();
    }

    Type mean   = static_cast<Type>(-12.0);
    Type stddev = static_cast<Type>(2.4);

    rocrand_generator device_generator;
    ROCRAND_CHECK(rocrand_create_generator(&device_generator, rng_type));

    std::vector<Type> host_results(218192);
    std::vector<Type> device_results(host_results.size());

    Type* output;
    HIP_CHECK(hipMallocHelper(&output, host_results.size() * sizeof(Type)));

    for(const unsigned long long seed : seeds)
    {
        SCOPED_TRACE(testing::Message() << "with seed = " << seed);
        ROCRAND_CHECK(rocrand_set_seed(host_generator, seed));
        ROCRAND_CHECK(rocrand_set_seed(device_generator, seed));

        ROCRAND_CHECK(
            generate(host_generator, host_results.data(), host_results.size(), mean, stddev));
        ROCRAND_CHECK(generate(device_generator, output, host_results.size(), mean, stddev));

        HIP_CHECK(hipMemcpy(device_results.data(),
                            output,
                            host_results.size() * sizeof(Type),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipDeviceSynchronize());

        // This rounding is required because the sine and cosine used in box-muller used in the normal
        // distribution is slightly different from the one used on the host.
        assert_near(host_results, device_results, eps);
    }

    ROCRAND_CHECK(rocrand_destroy_generator(host_generator));
    ROCRAND_CHECK(rocrand_destroy_generator(device_generator));
    HIP_CHECK(hipFree(output));
}

TEST_P(rocrand_generate_host_test, normal_half_parity_test)
{
    test_normal_parity<half>(get_generator(),
                             GetParam().rng_type,
                             rocrand_generate_normal_half,
                             0.1);
}

TEST_P(rocrand_generate_host_test, normal_float_parity_test)
{
    test_normal_parity<float>(get_generator(), GetParam().rng_type, rocrand_generate_normal, 0.005);
}

TEST_P(rocrand_generate_host_test, normal_double_parity_test)
{
    test_normal_parity<double>(get_generator(),
                               GetParam().rng_type,
                               rocrand_generate_normal_double,
                               0.000001);
}

TEST_P(rocrand_generate_host_test, log_normal_half_parity_test)
{
    test_normal_parity<half>(get_generator(),
                             GetParam().rng_type,
                             rocrand_generate_log_normal_half,
                             0.05);
}

TEST_P(rocrand_generate_host_test, log_normal_float_parity_test)
{
    test_normal_parity<float>(get_generator(),
                              GetParam().rng_type,
                              rocrand_generate_log_normal,
                              0.0001);
}

TEST_P(rocrand_generate_host_test, log_normal_double_parity_test)
{
    test_normal_parity<double>(get_generator(),
                               GetParam().rng_type,
                               rocrand_generate_log_normal_double,
                               0.0000001);
}

TEST_P(rocrand_generate_host_test, poisson_parity_test)
{
    const rocrand_rng_type rng_type = GetParam().rng_type;
    using Type                      = unsigned int;
    double lambda                   = 1.1;

    rocrand_generator host_generator = get_generator();
    rocrand_generator device_generator;
    ROCRAND_CHECK(rocrand_create_generator(&device_generator, rng_type));

    std::vector<Type> host_results(218192);
    std::vector<Type> device_results(host_results.size());

    Type* output;
    HIP_CHECK(hipMallocHelper(&output, host_results.size() * sizeof(Type)));

    for(const unsigned long long seed : get_seeds())
    {
        SCOPED_TRACE(testing::Message() << "with seed = " << seed);
        ROCRAND_CHECK(rocrand_set_seed(host_generator, seed));
        ROCRAND_CHECK(rocrand_set_seed(device_generator, seed));

        ROCRAND_CHECK(rocrand_generate_poisson(host_generator,
                                               host_results.data(),
                                               host_results.size(),
                                               lambda));
        ROCRAND_CHECK(
            rocrand_generate_poisson(device_generator, output, host_results.size(), lambda));

        HIP_CHECK(hipMemcpy(device_results.data(),
                            output,
                            host_results.size() * sizeof(Type),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipDeviceSynchronize());

        assert_eq(host_results, device_results);
    }

    ROCRAND_CHECK(rocrand_destroy_generator(host_generator));
    ROCRAND_CHECK(rocrand_destroy_generator(device_generator));
    HIP_CHECK(hipFree(output));
}

INSTANTIATE_TEST_SUITE_P(rocrand_generate_host_test,
                         rocrand_generate_host_test,
                         ::testing::ValuesIn(host_test_params_array));
