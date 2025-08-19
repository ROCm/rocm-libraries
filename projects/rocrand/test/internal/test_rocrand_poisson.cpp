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

#include <gtest/gtest.h>
#include <stdio.h>

#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#undef ROCRAND_DETAIL_BM_NOT_IN_STATE
#include <rocrand/rocrand_poisson.h>

#define HIP_CHECK(cmd)                                                                         \
    do                                                                                         \
    {                                                                                          \
        auto error = (cmd);                                                                    \
        if(error != hipSuccess)                                                                \
        {                                                                                      \
            std::cerr << "Encountered HIP error (" << hipGetErrorString(error) << ") at line " \
                      << __LINE__ << " in file " << __FILE__ << "\n";                          \
            exit(-1);                                                                          \
        }                                                                                      \
    }                                                                                          \
    while(0)

#define ROCRAND_CHECK(cmd)                                                                \
    do                                                                                    \
    {                                                                                     \
        auto status = cmd;                                                                \
        if(status != 0)                                                                   \
        {                                                                                 \
            std::cerr << "Encountered ROCRAND error: " << status << "at line" << __LINE__ \
                      << " in file " << __FILE__ << "\n";                                 \
            exit(status);                                                                 \
        }                                                                                 \
    }                                                                                     \
    while(0)

// If x is small then get withing 0.001 otherwise 5%
#define GET_EPS(x) x < 0.01 ? 0.01 : x * 0.05

// If x is small then get withing 0.001 otherwise 20%
#define GET_EPS_DEVICE(x) x < 0.01 ? 0.01 : x * 0.2

#define IS_SOBOL(x)                                                                       \
    (std::is_same_v<x, rocrand_state_sobol32> || std::is_same_v<x, rocrand_state_sobol64> \
     || std::is_same_v<x, rocrand_state_scrambled_sobol32>                                \
     || std::is_same_v<x, rocrand_state_scrambled_sobol64>)

/* #################################################

                TEST HOST SIDE

   ###############################################*/

template<class RocrandPRNGType>
inline void GetHostRocrandState(RocrandPRNGType* host_state)
{

    if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_sobol32>)
    {
        const unsigned int* directions;
        ROCRAND_CHECK(
            rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
        rocrand_init(directions, 123456, host_state);
    }
    // scrambled sobol32 case
    else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_scrambled_sobol32>)
    {
        const unsigned int* directions;
        ROCRAND_CHECK(
            rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
        rocrand_init(directions, 123456, 654321, host_state);
    }
    // sobol64 case
    else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_sobol64>)
    {
        const unsigned long long* directions;
        ROCRAND_CHECK(
            rocrand_get_direction_vectors64(&directions, ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));
        rocrand_init(directions, 123456, host_state);
    }
    // scrambled sobol64 case
    else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_scrambled_sobol64>)
    {
        const unsigned long long* directions;
        ROCRAND_CHECK(
            rocrand_get_direction_vectors64(&directions, ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));
        rocrand_init(directions, 123456, 654321, host_state);
    }
    // lfsr113 case
    else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_lfsr113>)
    {
        rocrand_init({0xabcd, 0xdabc, 0xcdab, 0xbcda}, 0, 0, host_state);
    }
    else
    {
        rocrand_init(123456, 654321, 0, host_state);
    }
}

using PoissonParams = ::testing::Types<rocrand_state_philox4x32_10,
                                       rocrand_state_mrg31k3p,
                                       rocrand_state_mrg32k3a,
                                       rocrand_state_xorwow,
                                       rocrand_state_sobol32,
                                       rocrand_state_scrambled_sobol32,
                                       rocrand_state_sobol64,
                                       rocrand_state_scrambled_sobol64,
                                       rocrand_state_lfsr113,
                                       rocrand_state_threefry2x32_20,
                                       rocrand_state_threefry2x64_20,
                                       rocrand_state_threefry4x32_20,
                                       rocrand_state_threefry4x64_20>;

template<typename T>
class PoissonTest : public ::testing::Test
{
public:
    using rocrand_prng_type                     = T;
    std::vector<double> small_poisson_lambdas   = {1, 2, 4, 8, 16, 32, 64};
    std::vector<double> large_poisson_lambdas   = {128, 256, 512, 1024, 2048};
    std::vector<double> massive_poisson_lambdas = {4096, 8192, 16384, 32768};
};

TYPED_TEST_SUITE(PoissonTest, PoissonParams);

template<typename OutputType, typename PrngState, class PoissonFunc>
void run_device_poisson_test(const PoissonFunc& pf, std::vector<double>& all_lambdas)
{
    constexpr size_t test_size = 100000;

    PrngState state;
    GetHostRocrandState(&state);

    std::vector<OutputType> output(test_size);

    for(const double& lambda : all_lambdas)
    {
        double expected_mean    = lambda;
        double expected_std_dev = std::sqrt(lambda);

        for(size_t i = 0; i < test_size; i++)
        {
            output[i] = pf(&state, lambda);
        }
        double actual_mean = std::accumulate(output.begin(),
                                             output.end(),
                                             (double)0,
                                             [=](double acc, OutputType x)
                                             { return acc + static_cast<double>(x); })
                             / static_cast<double>(test_size);
        double actual_std_dev
            = std::accumulate(output.begin(),
                              output.end(),
                              (double)0,
                              [=](double acc, OutputType x)
                              { return acc + std::pow(static_cast<double>(x) - actual_mean, 2); });
        actual_std_dev = std::sqrt(actual_std_dev / static_cast<double>(test_size - 1));

        double mean_eps    = expected_mean * 0.05;
        double std_dev_eps = expected_std_dev * 0.05;

        ASSERT_NEAR(expected_mean, actual_mean, mean_eps);
        ASSERT_NEAR(expected_std_dev, actual_std_dev, std_dev_eps);
    }
}

TYPED_TEST(PoissonTest, test_host_small_lambda)
{
    using PrngState = typename TestFixture::rocrand_prng_type;

    // Since Sobol uses the inv funciton
    if(!IS_SOBOL(PrngState))
    {
        std::vector<double> small_lambdas = {1, 2, 4, 8, 16, 32, 64};

        run_device_poisson_test<unsigned int, PrngState>(
            [=](PrngState* state, double lambda)
            { return rocrand_device::detail::poisson_distribution_small(state, lambda); },
            small_lambdas);
    }
}

TYPED_TEST(PoissonTest, test_host_large_lambda)
{
    using PrngState = typename TestFixture::rocrand_prng_type;

    // Since Sobol uses the inv funciton
    if(!IS_SOBOL(PrngState))
    {
        std::vector<double> large_lambdas = {128, 256, 512, 1024, 2048};

        run_device_poisson_test<unsigned int, PrngState>(
            [=](PrngState* state, double lambda)
            { return rocrand_device::detail::poisson_distribution_large(state, lambda); },
            large_lambdas);
    }
}

TYPED_TEST(PoissonTest, test_host_huge_lambda)
{
    using PrngState = typename TestFixture::rocrand_prng_type;

    // Since Sobol uses the inv funciton
    if(!IS_SOBOL(PrngState))
    {
        std::vector<double> huge_lambdas = {4096, 8192};

        run_device_poisson_test<unsigned int, PrngState>(
            [=](PrngState* state, double lambda)
            { return rocrand_device::detail::poisson_distribution_huge(state, lambda); },
            huge_lambdas);
    }
}

TYPED_TEST(PoissonTest, test_host_all_lambda)
{
    using PrngState = typename TestFixture::rocrand_prng_type;

    // Since Sobol uses the inv funciton
    if(!IS_SOBOL(PrngState))
    {
        std::vector<double> all_lambdas = {64, 2048, 4096};

        run_device_poisson_test<unsigned int, PrngState>(
            [=](PrngState* state, double lambda)
            { return rocrand_device::detail::poisson_distribution(state, lambda); },
            all_lambdas);
    }
}

TYPED_TEST(PoissonTest, test_host_inv)
{
    using PrngState = typename TestFixture::rocrand_prng_type;

    std::vector<double> all_lambdas = {1, 2, 4, 1024, 2048};

    run_device_poisson_test<unsigned int, PrngState>(
        [=](PrngState* state, double lambda)
        { return rocrand_device::detail::poisson_distribution_inv(state, lambda); },
        all_lambdas);
}

TYPED_TEST(PoissonTest, test_host_rocrand_poisson)
{
    using PrngState = typename TestFixture::rocrand_prng_type;

    std::vector<double> all_lambdas = {
        32,
        64,
        1024,
        2048,
        4096,
    };

    run_device_poisson_test<unsigned int, PrngState>([=](PrngState* state, double lambda)
                                                     { return rocrand_poisson(state, lambda); },
                                                     all_lambdas);
}

/* #################################################

                TEST DEVICE SIDE

   ###############################################*/

struct GlobalSizes
{
    static constexpr size_t items_per_thread = 10000;
    static constexpr size_t block_size       = 8;
    static constexpr size_t items_per_block  = items_per_thread * block_size;
    static constexpr size_t grid_size        = 8;
    static constexpr size_t size             = grid_size * items_per_block;
};

//get the rocrand state (device_state should be allocated)
template<class RocrandPRNGType>
inline void GetDeviceRocrandState(RocrandPRNGType* device_prngs)
{
    // Initialize for device code rocrand state. Each thread will get 1 "state"
    // Assumed that device_prngs is already initialized

    std::vector<RocrandPRNGType> host_states(GlobalSizes::block_size * GlobalSizes::grid_size);

    for(size_t bi = 0; bi < GlobalSizes::grid_size; bi++)
    {
        for(size_t ti = 0; ti < GlobalSizes::block_size; ti++)
        {
            const size_t offset = bi * GlobalSizes::block_size;
            const size_t prng_offset
                = (GlobalSizes::items_per_block * bi) + (GlobalSizes::items_per_thread * ti);

            if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_sobol32>)
            {
                const unsigned int* directions;
                ROCRAND_CHECK(
                    rocrand_get_direction_vectors32(&directions,
                                                    ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));

                rocrand_init(directions, prng_offset, &host_states[offset + ti]);
            }
            // scrambled sobol32 case
            else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_scrambled_sobol32>)
            {
                const unsigned int* directions;
                ROCRAND_CHECK(
                    rocrand_get_direction_vectors32(&directions,
                                                    ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
                rocrand_init(directions, 123456, prng_offset, &host_states[offset + ti]);
            }
            // sobol64 case
            else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_sobol64>)
            {
                const unsigned long long* directions;
                ROCRAND_CHECK(
                    rocrand_get_direction_vectors64(&directions,
                                                    ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));
                rocrand_init(directions, prng_offset, &host_states[offset + ti]);
            }
            // scrambled sobol64 case
            else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_scrambled_sobol64>)
            {
                const unsigned long long* directions;
                ROCRAND_CHECK(
                    rocrand_get_direction_vectors64(&directions,
                                                    ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));
                rocrand_init(directions, 123456, prng_offset, &host_states[offset + ti]);
            }
            // lfsr113 case
            else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_lfsr113>)
            {
                rocrand_init({0xabcd, 0xdabc, 0xcdab, 0xbcda},
                             0,
                             prng_offset,
                             &host_states[offset + ti]);
            }
            else
            {
                rocrand_init(123456, 654321, prng_offset, &host_states[offset + ti]);
            }
        }
    }

    HIP_CHECK(hipMemcpy(device_prngs,
                        host_states.data(),
                        sizeof(RocrandPRNGType) * GlobalSizes::grid_size * GlobalSizes::block_size,
                        hipMemcpyHostToDevice));
}

// Declaring typed test parameters

template<typename ReturnType, class RocRandPrngType, class PoissonFunc>
__global__
void poisson_kernel(RocRandPrngType* states,
                    ReturnType*      device_output,
                    const double     lambda,
                    PoissonFunc      f)
{
    const size_t offset = (GlobalSizes::items_per_block * blockIdx.x)
                          + (GlobalSizes::items_per_thread * threadIdx.x);
    const size_t state_offset = (GlobalSizes::block_size * blockIdx.x) + threadIdx.x;

    auto state = states + state_offset;
    for(size_t i = 0; i < GlobalSizes::items_per_thread; i++)
        device_output[offset + i] = f(state, lambda);

    states[state_offset] = *state;
}

// read_func is how to interpret the output (needed for special case like uint4)
// size_multiplier is needed for special cases like uint4 where each element is actually 4
template<typename RocRandPrngType, typename ReturnType, class PoissonFunc, class ReadFunc>
void run_poisson_test(std::vector<double>& all_lambdas,
                      const PoissonFunc    f,
                      const ReadFunc       read_func,
                      const size_t         size_multiplier = 1)
{
    ReturnType* host_output = new ReturnType[GlobalSizes::size];
    ReturnType* device_output;
    HIP_CHECK(hipMalloc(&device_output, sizeof(ReturnType) * GlobalSizes::size));

    RocRandPrngType* device_state;
    HIP_CHECK(
        hipMalloc(&device_state,
                  sizeof(RocRandPrngType) * GlobalSizes::block_size * GlobalSizes::grid_size));
    GetDeviceRocrandState(device_state);
    for(const double lambda : all_lambdas)
    {
        double expected_mean    = lambda;
        double expected_std_dev = std::sqrt(lambda);
        double mean_tol         = GET_EPS_DEVICE(expected_mean);
        double std_tol          = GET_EPS_DEVICE(expected_std_dev);

        poisson_kernel<ReturnType>
            <<<dim3(GlobalSizes::grid_size), dim3(GlobalSizes::block_size), 0, 0>>>(device_state,
                                                                                    device_output,
                                                                                    lambda,
                                                                                    f);
        HIP_CHECK(hipMemcpy(host_output,
                            device_output,
                            sizeof(ReturnType) * GlobalSizes::size,
                            hipMemcpyDeviceToHost));

        for(size_t block_idx = 0; block_idx < GlobalSizes::grid_size; block_idx++)
        {
            for(size_t thread_idx = 0; thread_idx < GlobalSizes::block_size; thread_idx++)
            {

                size_t offset = (block_idx * GlobalSizes::items_per_block)
                                + (thread_idx * GlobalSizes::items_per_thread);

                double actual_mean
                    = std::accumulate(host_output + offset,
                                      host_output + offset + GlobalSizes::items_per_thread,
                                      (double)0,
                                      [=](double acc, ReturnType x) { return acc + read_func(x); })
                      / static_cast<double>(GlobalSizes::items_per_thread * size_multiplier);
                double actual_std_dev
                    = std::accumulate(host_output + offset,
                                      host_output + offset + GlobalSizes::items_per_thread,
                                      (double)0,
                                      [=](double acc, ReturnType x) {
                                          return acc
                                                 + std::pow(static_cast<double>(read_func(x))
                                                                - (actual_mean * size_multiplier),
                                                            2);
                                      });
                actual_std_dev = std::sqrt(
                    actual_std_dev
                    / static_cast<double>((GlobalSizes::items_per_thread * size_multiplier) - 1));

                ASSERT_NEAR(expected_mean, actual_mean, mean_tol);
                ASSERT_NEAR(expected_std_dev, actual_std_dev, std_tol);
            }
        }
    }
    delete[] host_output;
    HIP_CHECK(hipFree(device_output));
    HIP_CHECK(hipFree(device_state));
}

enum class internal_poisson_type
{
    small,
    normal,
    large,
    huge,
    inv,
};

template<internal_poisson_type size, class T>
struct internal_poisson_dis
{
    __device__
    auto operator()(T* state, const double lambda)
    {
        if constexpr(size == internal_poisson_type::inv)
        {
            return rocrand_device::detail::poisson_distribution_inv(state, lambda);
        }
        else if constexpr(size == internal_poisson_type::small)
        {
            return rocrand_device::detail::poisson_distribution_small(state, lambda);
        }
        else if constexpr(size == internal_poisson_type::large)
        {
            return rocrand_device::detail::poisson_distribution_large(state, lambda);
        }
        else if constexpr(size == internal_poisson_type::huge)
        {
            return rocrand_device::detail::poisson_distribution_huge(state, lambda);
        }
        else
        {
            return rocrand_device::detail::poisson_distribution(state, lambda);
        }
    }
};

struct internal_poisson_reader
{
    __device__ __host__
    auto operator()(const unsigned int& x) const
    {
        return x;
    }
};

struct internal_poisson_reader4
{
    __device__ __host__
    auto operator()(const uint4& x) const
    {
        return (x.w + x.x + x.y + x.z);
    }
};

TYPED_TEST(PoissonTest, poisson_distribution_small_lambda_test)
{
    using type = typename TestFixture::rocrand_prng_type;

    // Since Sobol uses the inv funciton
    if(!IS_SOBOL(type))
    {
        run_poisson_test<type, unsigned int>(
            TestFixture::small_poisson_lambdas,
            internal_poisson_dis<internal_poisson_type::small, type>{},
            internal_poisson_reader{});
    }
}

TYPED_TEST(PoissonTest, poisson_distribution_large_lambda_test)
{
    using type = typename TestFixture::rocrand_prng_type;

    // Since Sobol uses the inv funciton
    if(!IS_SOBOL(type))
    {
        run_poisson_test<type, unsigned int>(
            TestFixture::large_poisson_lambdas,
            internal_poisson_dis<internal_poisson_type::large, type>{},
            internal_poisson_reader{});
    }
}

TYPED_TEST(PoissonTest, poisson_distribution_huge_lambda_test)
{
    using type = typename TestFixture::rocrand_prng_type;
    run_poisson_test<type, unsigned int>(TestFixture::massive_poisson_lambdas,
                                         internal_poisson_dis<internal_poisson_type::huge, type>{},
                                         internal_poisson_reader{});
}

TYPED_TEST(PoissonTest, poisson_distribution_test)
{
    using type = typename TestFixture::rocrand_prng_type;

    if(!IS_SOBOL(type))
    {
        run_poisson_test<type, unsigned int>(
            TestFixture::small_poisson_lambdas,
            internal_poisson_dis<internal_poisson_type::normal, type>{},
            internal_poisson_reader{});

        run_poisson_test<type, unsigned int>(
            TestFixture::large_poisson_lambdas,
            internal_poisson_dis<internal_poisson_type::normal, type>{},
            internal_poisson_reader{});

        run_poisson_test<type, unsigned int>(
            TestFixture::massive_poisson_lambdas,
            internal_poisson_dis<internal_poisson_type::normal, type>{},
            internal_poisson_reader{});
    }
}

TYPED_TEST(PoissonTest, poisson_distribution_inv_test)
{
    using type = typename TestFixture::rocrand_prng_type;

    run_poisson_test<type, unsigned int>(TestFixture::small_poisson_lambdas,
                                         internal_poisson_dis<internal_poisson_type::inv, type>{},
                                         internal_poisson_reader{});
}

// External Tests
template<typename type>
struct external_poisson_dis
{
    __device__ __host__
    auto operator()(type* state, const double lambda) const
    {
        return rocrand_poisson(state, lambda);
    }
};

TYPED_TEST(PoissonTest, external_rocrand_poisson)
{
    using type = typename TestFixture::rocrand_prng_type;

    run_poisson_test<type, unsigned int>(TestFixture::small_poisson_lambdas,
                                         external_poisson_dis<type>{},
                                         internal_poisson_reader{});
}

// Special Tests
struct special_poisson_dis
{
    __device__ __host__
    auto operator()(rocrand_state_philox4x32_10* state, const double lambda)
    {
        return rocrand_poisson4(state, lambda);
    }
};

TEST(PoissonTest, philox4x32_10_uint4_output)
{
    std::vector<double> small_poisson_lambdas = {1, 2, 4, 8, 16, 32, 64};

    run_poisson_test<rocrand_state_philox4x32_10, uint4>(small_poisson_lambdas,
                                                         special_poisson_dis{},
                                                         internal_poisson_reader4{},
                                                         4);
}
