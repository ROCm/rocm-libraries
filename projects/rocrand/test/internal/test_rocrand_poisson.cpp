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
#include <random>
#include <vector>

#include <rocrand/rocrand_mtgp32_11213.h>
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
            exit(-1);                                                                     \
        }                                                                                 \
    }                                                                                     \
    while(0)

struct GlobalSizes {
    static constexpr size_t items_per_thread = 10000;
    static constexpr size_t block_size = 8;
    static constexpr size_t items_per_block = items_per_thread * block_size;
    static constexpr size_t grid_size = 8;
    static constexpr size_t size = grid_size * items_per_block;
};

//get the rocrand state (device_state should be allocated)
template<class RocrandPRNGType>
inline void GetRocrandState(RocrandPRNGType * device_state){

    RocrandPRNGType * host_state = new RocrandPRNGType[GlobalSizes::block_size * GlobalSizes::grid_size];

    for(size_t i = 0; i < GlobalSizes::block_size * GlobalSizes::grid_size; i++){
        if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_sobol32>){
            const unsigned int* directions;
            ROCRAND_CHECK(rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
            rocrand_init(directions, 123456 ^ i, host_state + i);
        }
        // scrambled sobol32 case
        else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_scrambled_sobol32>){
            const unsigned int* directions;
            ROCRAND_CHECK(rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
            rocrand_init(directions, 123456 ^ i, 654321 ^ i, host_state + i);
        }
        // sobol64 case
        else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_sobol64>){
            const unsigned long long* directions;
            ROCRAND_CHECK(rocrand_get_direction_vectors64(&directions, ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));
            rocrand_init(directions, 123456 ^ i, host_state + i);
        }
        // scrambled sobol64 case
        else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_scrambled_sobol64>){
            const unsigned long long* directions;
            ROCRAND_CHECK(rocrand_get_direction_vectors64(&directions, ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));
            rocrand_init(directions, 123456 ^ i, 654321 ^ i, host_state + i);
        }
        // lfsr113 case
        else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_lfsr113>){
            rocrand_init({0xabcd, 0xdabc, 0xcdab, 0xbcda}, 0, 0, host_state + i);
        }
        else{
            rocrand_init(123456 ^ i, 654321 ^ i, 0, host_state + i);
        }
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(device_state + i, host_state + i, sizeof(RocrandPRNGType), hipMemcpyHostToDevice));
    }

    delete [] host_state;
}

// Declaring typed test parameters

template<class RocrandPRNGType>
struct PoissonParameterHolder{
    using prng_state = RocrandPRNGType;

};

using rocRANDStates = ::testing::Types<
    rocrand_state_philox4x32_10,
    rocrand_state_mrg31k3p,
    rocrand_state_mrg32k3a,
    rocrand_state_xorwow,
    rocrand_state_lfsr113,
    rocrand_state_sobol32,
    rocrand_state_scrambled_sobol32,
    rocrand_state_sobol64,
    rocrand_state_scrambled_sobol64,
    rocrand_state_threefry2x32_20,
    rocrand_state_threefry2x64_20,
    rocrand_state_threefry4x32_20,
    rocrand_state_threefry4x64_20
>;

template<class RocrandPRNGType>
class PoissonTest : public ::testing::Test{
    public:
        using prng_type = RocrandPRNGType;
        std::vector<double> small_poisson_lambdas = {1, 2, 4, 8, 16, 32, 64};
        std::vector<double> large_poisson_lambdas = {128, 256, 512, 1024, 2048};
        std::vector<double> massive_poisson_lambdas = {4096, 8192, 16384, 32768};
};

TYPED_TEST_SUITE(PoissonTest, rocRANDStates);

template<typename ReturnType, class RocRandPrngType, class PoissonFunc>
__global__ void poisson_kernel(RocRandPrngType * states, ReturnType * device_output, const double lambda, const PoissonFunc & f){
    const size_t offset = (GlobalSizes::items_per_block * blockIdx.x) + (GlobalSizes::items_per_thread * threadIdx.x);
    const size_t state_offset = (GlobalSizes::block_size * blockIdx.x) + threadIdx.x;

    auto state = states + state_offset;
    for(size_t i = 0; i < GlobalSizes::items_per_thread; i++)
        device_output[offset + i] = f(state, lambda);

    states[state_offset] = *state;
}

// read_func is how to interpret the output (needed for special case like uint4)
// size_multiplier is needed for special cases like uint4 where each element is actually 4
template<typename RocRandPrngType, typename ReturnType, class PoissonFunc, class ReadFunc>
void run_poisson_test(
    std::vector<double> & all_lambdas,
    const PoissonFunc & f,
    const ReadFunc & read_func,
    const size_t size_multiplier = 1)
    {

    constexpr bool is_sobol  =  std::is_same_v<RocRandPrngType, rocrand_state_sobol32> ||
                                std::is_same_v<RocRandPrngType, rocrand_state_sobol64> ||
                                std::is_same_v<RocRandPrngType, rocrand_state_scrambled_sobol32> ||
                                std::is_same_v<RocRandPrngType, rocrand_state_scrambled_sobol64>;

    ReturnType * host_output = new ReturnType[GlobalSizes::size];
    ReturnType * device_output;
    HIP_CHECK(hipMalloc(&device_output, sizeof(ReturnType) * GlobalSizes::size));

    RocRandPrngType * device_state;
    HIP_CHECK(hipMalloc(&device_state, sizeof(RocRandPrngType) * GlobalSizes::block_size * GlobalSizes::grid_size));
    GetRocrandState(device_state);
    for(const double lambda : all_lambdas){
        double expected_mean = lambda;
        double expected_std_dev  = std::sqrt(lambda);
        double mean_tol = expected_mean * 0.05;
        double std_tol  = expected_std_dev  * 0.05;

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(poisson_kernel<ReturnType>),
            dim3(GlobalSizes::grid_size), dim3(GlobalSizes::block_size), 0, 0,
            device_state, device_output, lambda, f
        );
        HIP_CHECK(hipMemcpy(host_output, device_output, sizeof(ReturnType) * GlobalSizes::size, hipMemcpyDeviceToHost));

        for(size_t block_idx = 0; block_idx < GlobalSizes::grid_size; block_idx++){
            for(size_t thread_idx = 0; thread_idx < GlobalSizes::block_size; thread_idx++){

                size_t offset = (block_idx * GlobalSizes::items_per_block) + (thread_idx * GlobalSizes::items_per_thread);

                double actual_mean = std::accumulate(
                    host_output + offset, host_output + offset + GlobalSizes::items_per_thread, (double) 0,
                    [=] (double acc, ReturnType x){
                        return acc + read_func(x);
                    }
                )
                    / static_cast<double>(GlobalSizes::items_per_thread * size_multiplier);
                double actual_std_dev = std::accumulate(
                    host_output + offset, host_output + offset + GlobalSizes::items_per_thread, (double) 0,
                    [=](double acc, ReturnType x) {
                        return acc + std::pow(static_cast<double>(read_func(x)) - (actual_mean * size_multiplier), 2);
                    }
                );
                actual_std_dev = std::sqrt(actual_std_dev / static_cast<double>((GlobalSizes::items_per_thread * size_multiplier) - 1));

                ASSERT_NEAR(expected_mean, actual_mean, mean_tol);
                if(!is_sobol)
                    ASSERT_NEAR(expected_std_dev, actual_std_dev, std_tol);
            }
        }
    }
    delete [] host_output;
    HIP_CHECK(hipFree(device_output));
    HIP_CHECK(hipFree(device_state));
}

TYPED_TEST(PoissonTest, poisson_distribution_small_lambda_test){
    using type = typename TestFixture::prng_type;
    run_poisson_test<type, unsigned int>(
        TestFixture::small_poisson_lambdas,
        [=] __host__ __device__ (type * state, const double lambda){
            return rocrand_device::detail::poisson_distribution_small(state, lambda);
        },
        [] (const unsigned int & x){
            return x;
        }
    );
}

TYPED_TEST(PoissonTest, poisson_distribution_large_lambda_test){
    using type = typename TestFixture::prng_type;
    run_poisson_test<type, unsigned int>(
        TestFixture::large_poisson_lambdas,
        [=] __host__ __device__ (type * state, const double lambda){
            return rocrand_device::detail::poisson_distribution_large(state, lambda);
        },
        [] (const unsigned int & x){
            return x;
        }
    );
}

TYPED_TEST(PoissonTest, poisson_distribution_huge_lambda_test){
    using type = typename TestFixture::prng_type;
    run_poisson_test<type, unsigned int>(
        TestFixture::massive_poisson_lambdas,
        [=] __host__ __device__ (type * state, const double lambda){
            return rocrand_device::detail::poisson_distribution_huge(state, lambda);
        },
        [] (const unsigned int & x){
            return x;
        }
    );
}

TYPED_TEST(PoissonTest, poisson_distribution_test){
    using type = typename TestFixture::prng_type;

    run_poisson_test<type, unsigned int>(
        TestFixture::small_poisson_lambdas,
        [=] __host__ __device__ (type * state, const double lambda){
            return rocrand_device::detail::poisson_distribution(state, lambda);
        },
        [] (const unsigned int & x){
            return x;
        }
    );

    run_poisson_test<type, unsigned int>(
        TestFixture::large_poisson_lambdas,
        [=] __host__ __device__ (type * state, const double lambda){
            return rocrand_device::detail::poisson_distribution(state, lambda);
        },
        [] (const unsigned int & x){
            return x;
        }
    );

    run_poisson_test<type, unsigned int>(
        TestFixture::massive_poisson_lambdas,
        [=] __host__ __device__ (type * state, const double lambda){
            return rocrand_device::detail::poisson_distribution(state, lambda);
        },
        [] (const unsigned int & x){
            return x;
        }
    );
}

TYPED_TEST(PoissonTest, poisson_distribution_inv_test){
    using type = typename TestFixture::prng_type;

    run_poisson_test<type, unsigned int>(
        TestFixture::small_poisson_lambdas,
        [=] __host__ __device__ (type * state, const double lambda){
            return rocrand_device::detail::poisson_distribution_inv(state, lambda);
        },
        [] (const unsigned int & x){
            return x;
        }
    );
}

// External Tests
TYPED_TEST(PoissonTest, external_rocrand_poisson){
    using type = typename TestFixture::prng_type;

    // TODO: Figure out why higher lambda is hanging
    run_poisson_test<type, unsigned int>(
        TestFixture::small_poisson_lambdas,
        [=] __host__ __device__ (type * state, const double lambda){
            return rocrand_poisson(state, lambda);
        },
        [] (const unsigned int & x){
            return x;
        }
    );
}

// Special Tests
TEST(PoissonTest, philox4x32_10_uint4_output){
    std::vector<double> small_poisson_lambdas = {1, 2, 4, 8, 16, 32, 64};

    run_poisson_test<rocrand_state_philox4x32_10, uint4>(
        small_poisson_lambdas,
        [=] __host__ __device__ (rocrand_state_philox4x32_10 * state, const double lambda){
            return rocrand_poisson4(state, lambda);
        },
        [] (const uint4 & x){
            return (x.w + x.x + x.y + x.z);
        },
        4
    );
}

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

using PoissonHostParams = ::testing::Types<rocrand_state_philox4x32_10,
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
class PoissonHostTest : public ::testing::Test
{
public:
    using rocrand_prng_type = T;
};

TYPED_TEST_SUITE(PoissonHostTest, PoissonHostParams);

TYPED_TEST(PoissonHostTest, poisson_host)
{
    using PrngState = typename TestFixture::rocrand_prng_type;

    constexpr size_t test_size = 50000;

    PrngState state;
    GetHostRocrandState(&state);

    std::vector<double> all_lambdas = {
        32,
        64,
        1024,
        2048,
        4096,
    };

    std::vector<unsigned int> output(test_size);

    for(const double& lambda : all_lambdas)
    {
        double expected_mean    = lambda;
        double expected_std_dev = std::sqrt(lambda);

        for(size_t i = 0; i < test_size; i++)
        {
            output[i] = rocrand_poisson(&state, lambda);
        }
        double actual_mean = std::accumulate(output.begin(),
                                             output.end(),
                                             (double)0,
                                             [=](double acc, unsigned int x)
                                             { return acc + static_cast<double>(x); })
                             / static_cast<double>(test_size);
        double actual_std_dev
            = std::accumulate(output.begin(),
                              output.end(),
                              (double)0,
                              [=](double acc, unsigned int x)
                              { return acc + std::pow(static_cast<double>(x) - actual_mean, 2); });
        actual_std_dev = std::sqrt(actual_std_dev / static_cast<double>(test_size - 1));

        double mean_eps    = expected_mean * 0.05;
        double std_dev_eps = expected_std_dev * 0.05;

        ASSERT_NEAR(expected_mean, actual_mean, mean_eps);
        ASSERT_NEAR(expected_std_dev, actual_std_dev, std_dev_eps);
    }
}
